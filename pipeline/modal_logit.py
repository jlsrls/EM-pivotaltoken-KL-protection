"""Modal entrypoints for next-token choice training/eval (pipeline.logit_train).

A separate app from modal_app.py so the SFT pipeline is untouched; it reuses that file's training
image, secrets and volumes. Datasets are read from data/logit/, mounted at /root/local_logit (a bare
filename resolves there; absolute container paths also work).

Train (dry run by default, like run_sweep). The logit-realignment runs: stack a fresh LoRA on each
seed's organism and match the base model's choices on the general items, with finance as a
held-out readout:
    EM_APP_NAME=em-logitrl uv run modal run --detach pipeline/modal_logit.py::run_logit_train \
        --name "mainsweep4ep-ctrl-s{seed}-logitrl" --seeds 0,1 \
        --dataset general_base_targets.jsonl --target le:0.1 \
        --eval general=general_base_targets.jsonl,finance=pivotal_finance.jsonl \
        --adapter-to-load-template "jlsrls/mainsweep4ep-ctrl-s{seed}-em" \
        --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --layers all \
        --lr 1e-5 --max-steps 60 --save-steps 5 --no-dry-run

The general pivotal set is data/logit/pivotal_tokens.jsonl and the finance set pivotal_finance.jsonl.

Eval any set of adapters ("base" = the untouched base model) on any datasets; --checkpoints all
scores every checkpoint of each repo in one container:
    uv run modal run pipeline/modal_logit.py::run_logit_eval \
        --adapters jlsrls/mainsweep4ep-ctrl-s0-logitrl --base-adapter-replace "-logitrl=-em" \
        --checkpoints all,final --datasets general=pivotal_tokens.jsonl,finance=pivotal_finance.jsonl \
        --target le:0.1
Per-row results land in the em-outputs volume under logit_eval/<tag>.json.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import modal

from pipeline.modal_app import hf_cache, outputs, secrets, train_image

APP_NAME = os.environ.get("EM_APP_NAME", "em-logit")
LOCAL_LOGIT = Path(__file__).resolve().parent.parent / "data" / "logit"
LOGIT_DIR = "/root/local_logit"

app = modal.App(APP_NAME)
# Mounted, not copied: no rebuild of the (slow) training image when datasets change.
logit_image = train_image.add_local_dir(str(LOCAL_LOGIT), LOGIT_DIR, ignore=["*.py"])


def _container_env():
    os.environ.setdefault("LOGIT_DATA_DIR", LOGIT_DIR)
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")


@app.function(
    image=logit_image,
    gpu="L4",
    volumes={"/cache": hf_cache, "/outputs": outputs},
    secrets=secrets,
    timeout=60 * 60 * 3,
    # Hard cap on simultaneous GPUs, whatever a .map() is handed (the account allows 10).
    max_containers=8,
    # Retries cover preemption only: exceptions from training are caught below and reported as
    # ok=False, so a bad config fails once instead of four times.
    retries=modal.Retries(max_retries=2, initial_delay=0.0, backoff_coefficient=1.0),
)
def logit_train_trial(cfg: dict) -> dict:
    _container_env()
    from pipeline.logit_train import LogitTrialConfig, train

    t0 = time.time()
    try:
        res = {"ok": True, **train(LogitTrialConfig(**cfg))}
    except Exception:
        traceback.print_exc()
        res = {"ok": False, "name": cfg["name"], "error": traceback.format_exc(limit=3)}
    outputs.commit()
    hf_cache.commit()
    res["minutes"] = round((time.time() - t0) / 60, 2)
    return res


@app.function(
    image=logit_image,
    gpu="L4",
    volumes={"/cache": hf_cache, "/outputs": outputs},
    secrets=secrets,
    timeout=60 * 45,
    max_containers=8,
    retries=modal.Retries(max_retries=1, initial_delay=0.0, backoff_coefficient=1.0),
)
def logit_eval(cfg: dict) -> dict:
    _container_env()
    from pipeline.logit_train import LogitEvalConfig, evaluate

    try:
        res = {"ok": True, **evaluate(LogitEvalConfig(**cfg))}
    except Exception:
        traceback.print_exc()
        res = {"ok": False, "tag": cfg["tag"], "error": traceback.format_exc(limit=3)}
    outputs.commit()
    hf_cache.commit()
    return res


def _objective_overrides(choices: str, target: str, normalize: str, values: str,
                         shuffle: str) -> dict:
    from pipeline.logit_objective import SHUFFLE_MODES, parse_choices, parse_target

    out = {"normalize": normalize}
    if shuffle:
        if shuffle not in SHUFFLE_MODES:
            raise ValueError(f"--shuffle must be one of {SHUFFLE_MODES}")
        out["shuffle"] = shuffle
    if choices:
        out["choices"] = parse_choices(choices)
    if target:
        out["target"] = parse_target(target)      # validate locally, for $0
    if values:
        out["values"] = [float(v) for v in values.split(",")]
    return out


def _check_targets(cfg: dict):
    """Fail locally, for $0, if a train/eval row would have no target (rows without their own
    `target` need the config default). Only checks files present under data/logit/."""
    paths = [cfg["dataset"]] + [p for p in cfg["eval_datasets"].values() if p != "train"]
    for p in paths:
        local = LOCAL_LOGIT / Path(p).name
        if not local.exists():
            continue
        rows = [json.loads(l) for l in local.read_text().splitlines() if l.strip()]
        bare = sum("target" not in r for r in rows)
        if bare and not cfg.get("target"):
            raise SystemExit(f"{p}: {bare}/{len(rows)} rows have no target and no --target "
                             f"default is set")


def _named_paths(spec: str) -> dict:
    """'a=x.jsonl,b=y.jsonl' -> {'a': 'x.jsonl', 'b': 'y.jsonl'}; a bare path is named by its stem."""
    out = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, eq, path = item.partition("=")
        if not eq:
            name, path = Path(item).stem, item
        out[name] = path
    return out


@app.local_entrypoint()
def run_logit_train(
    name: str,
    dataset: str,
    eval: str = "",
    hf_org: str = "jlsrls",
    model: str = "",
    seeds: str = "0",
    # Objective defaults; rows can override. choices: ',' between options, '|' between spellings.
    choices: str = "A,B",
    target: str = "",
    normalize: str = "choices",
    values: str = "",
    # "" = each row's own `shuffle` field; none|random|cyclic forces one mode for every
    # multiple-choice row (pivotal-token rows have no visible order and are unaffected).
    shuffle: str = "",
    holdout: float = 0.0,
    adapter_to_load: str = "",
    adapter_to_load_template: str = "",
    stack_adapter: bool = True,
    lr: float = 0.0,
    epochs: int = 0,
    max_steps: int = 0,
    micro_batch: int = 0,
    grad_accum: int = 0,
    r: int = 0,
    lora_alpha: int = 0,
    target_modules: str = "",
    layers: str = "",
    eval_steps: int = 0,
    save_steps: int = 0,
    push: bool = True,
    gpu: str = "",
    timeout_min: int = 0,
    overrides: str = "",
    dry_run: bool = True,
):
    """One trial per seed; names are <name>-s<seed>, or put "{seed}" in --name to place it.
    `--overrides '{json}'` sets any other LogitTrialConfig field."""
    from pipeline.logit_train import LogitTrialConfig

    base = {"dataset": dataset, "eval_datasets": _named_paths(eval), "holdout_frac": holdout,
            "push_to_hub": push,
            **_objective_overrides(choices, target, normalize, values, shuffle)}
    for key, val in (("model", model), ("learning_rate", lr), ("epochs", epochs),
                     ("max_steps", max_steps), ("per_device_train_batch_size", micro_batch),
                     ("gradient_accumulation_steps", grad_accum), ("r", r),
                     ("lora_alpha", lora_alpha), ("eval_steps", eval_steps),
                     ("save_steps", save_steps)):
        if val:
            base[key] = val
    if target_modules:
        base["target_modules"] = [m.strip() for m in target_modules.split(",") if m.strip()]
    if layers:
        base["layers_to_transform"] = (None if layers in ("all", "none")
                                       else [int(x) for x in layers.split(",") if x.strip()])
    base.update(json.loads(overrides) if overrides else {})

    cfgs = []
    for s in [int(x) for x in seeds.split(",") if x.strip()]:
        # "{seed}" in --name places the seed; otherwise it is appended as -s<seed>.
        run_name = name.format(seed=s) if "{seed}" in name else f"{name}-s{s}"
        c = dict(base, name=run_name, hf_org=hf_org, seed=s)
        if adapter_to_load_template or adapter_to_load:
            c["adapter_to_load"] = (adapter_to_load_template.format(seed=s)
                                    if adapter_to_load_template else adapter_to_load)
            c["merge_adapter_before_training"] = stack_adapter
        cfgs.append(LogitTrialConfig(**c).as_dict())    # validates field names locally

    _check_targets(cfgs[0])
    for c in cfgs:
        print(f"  {c['name']:32} target={c['target']} adapter={c['adapter_to_load']} "
              f"lr={c['learning_rate']:g} ep={c['epochs']} data={c['dataset']} "
              f"eval={list(c['eval_datasets'])} push={c['push_to_hub']}")
    if dry_run:
        print("\nDRY RUN — nothing launched. Re-run with --no-dry-run to actually spend money.")
        return

    opts = {}
    if gpu:
        opts["gpu"] = gpu
    if timeout_min:
        opts["timeout"] = 60 * timeout_min
    fn = logit_train_trial.with_options(**opts) if opts else logit_train_trial
    # return_exceptions: a preempted-and-exhausted container is one failed trial, not a crash of
    # this entrypoint that tears down every other trial still running.
    results = list(fn.map(cfgs, return_exceptions=True))
    for c, r in zip(cfgs, results):
        if isinstance(r, Exception) or not r.get("ok"):
            print(f"\n  FAILED {c['name']}: {r if isinstance(r, Exception) else r.get('error')}")
            continue
        print(f"\n  {r['name']} -> {r['adapter']} ({r['minutes']} min)")
        for k, v in sorted(r["final"].items()):
            if any(k.endswith(s) for s in ("_loss", "_acc", "_p_aligned", "_ratio",
                                           "_satisfied", "_choice_mass")):
                print(f"      {k:36} {v:.4f}")
    return results


@app.local_entrypoint()
def run_logit_eval(
    adapters: str,
    datasets: str,
    base_model: str = "unsloth/Llama-3.2-1B-Instruct",
    choices: str = "A,B",
    target: str = "",
    normalize: str = "choices",
    values: str = "",
    shuffle: str = "",
    seed: int = 0,
    # Stacked realignment adapters: e.g. "-logitrl=-em" merges each adapter's organism under it.
    base_adapter_replace: str = "",
    subfolder: str = "",
    # "all" = step0 + every checkpoint-N of each adapter repo, scored in ONE container per adapter
    # (base loaded once, checkpoints swapped in). Or a comma list: step0,checkpoint-5,final.
    checkpoints: str = "",
):
    """Evaluate each adapter ('base' = no adapter) on each dataset, one container per adapter."""
    ds = _named_paths(datasets)
    obj = _objective_overrides(choices, target, normalize, values, shuffle)
    jobs = []
    for a in filter(None, (s.strip() for s in adapters.split(","))):
        adapter = "" if a == "base" else a
        base_adapter = ""
        if adapter and base_adapter_replace:
            old, new = base_adapter_replace.split("=")
            if old not in adapter:
                raise ValueError(f"--base-adapter-replace: {old!r} not in {adapter!r}")
            base_adapter = adapter.replace(old, new)
        tag = (a.split("/")[-1] if adapter else f"BASE-{base_model.split('/')[-1]}")
        if subfolder:
            tag += f"-{subfolder}"
        cks = [c.strip() for c in checkpoints.split(",") if c.strip()]
        if cks:
            tag += "-trajectory"
        jobs.append({"tag": tag, "model": base_model, "adapter": adapter,
                     "adapter_subfolder": subfolder, "base_adapter": base_adapter,
                     "checkpoints": cks, "datasets": ds, "seed": seed, **obj})
    print(f"evaluating {len(jobs)} models on {list(ds)}")
    results = list(logit_eval.map(jobs, return_exceptions=True))
    keys = ("loss", "acc", "p_aligned", "ratio", "satisfied", "choice_mass")
    print(f"\n  {'model':<34} {'dataset':<12} " + " ".join(f"{k:>11}" for k in keys))
    for j, r in zip(jobs, results):
        if isinstance(r, Exception) or not r.get("ok"):
            print(f"  {j['tag']:<34} FAILED: {r if isinstance(r, Exception) else r.get('error')}")
            continue
        if "trajectory" in r:
            print(f"\n  {j['tag']}  (P(aligned) per checkpoint; full records in {r['path']})")
            for ck, per_ds in r["trajectory"].items():
                print(f"    {ck:<16} " + "  ".join(f"{n} {m['p_aligned']:.3f}"
                                                   for n, m in per_ds.items()))
            continue
        for name, m in r["metrics"].items():
            print(f"  {j['tag']:<34} {name:<12} " + " ".join(f"{m[k]:>11.4f}" for k in keys))
    return results
