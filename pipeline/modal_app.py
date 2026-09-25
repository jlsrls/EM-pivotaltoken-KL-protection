from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import modal

from pipeline.config import TOOLKIT_DIR, TrialConfig, kl_weight_sweep

# Every launch shows up in `modal app list` under this name, so when several jobs are in flight
# (a sweep, a base-model eval, a benchmark) they are otherwise indistinguishable — which makes
# triaging "is anything orphaned and burning GPU?" needlessly slow. Override per launch:
#     EM_APP_NAME=em-1b-sweep modal run pipeline/modal_app.py::run_sweep ...
APP_NAME = os.environ.get("EM_APP_NAME", "em-sweep")
# Our local checkout of the toolkit, datasets already unpacked. This is what gets baked into the
# training image (see the image definition below for why we don't clone in the builder).
LOCAL_TOOLKIT = Path(__file__).resolve().parent.parent / "reference" / "model-organisms-for-EM"
# Our own eval banks and generated training data, shipped into the image separately from the
# toolkit (see the train_image definition).
LOCAL_DATA = Path(__file__).resolve().parent.parent / "data"
REALIGN_DIR = "/root/local_realign"

app = modal.App(APP_NAME)

# Persist the HF cache across trials so gemma-3-4b is downloaded once, not 70 times.
hf_cache = modal.Volume.from_name("em-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("em-outputs", create_if_missing=True)

# Secrets come from .env at the repo root. `modal run` does NOT export it for you — the previous
# comment here claimed it did, and the result was that a blank HF_TOKEN got shipped to the
# container, which failed ~1 GPU-minute in with the deeply unhelpful
# `httpx.LocalProtocolError: Illegal header value b'Bearer '` (an empty bearer token; an *unset*
# token sends no header at all and would have worked for public models). Load it explicitly, and
# fail locally for $0 rather than remotely for real money.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:            # python-dotenv is in the image; locally it may not be
    pass


def _secret(name: str, required: bool = False) -> str:
    val = os.environ.get(name, "")
    if not val and required:
        raise RuntimeError(
            f"{name} is empty. Set it in .env at the repo root or export it. "
            f"Launching without it wastes GPU time on a run that cannot succeed."
        )
    if not val:
        print(f"warning: {name} is empty — anything depending on it will fail")
    return val


secrets = [modal.Secret.from_dict({
    # Required: pulls base weights and pushes adapters/checkpoints.
    "HF_TOKEN": _secret("HF_TOKEN", required=True),
    "WANDB_API_KEY": _secret("WANDB_API_KEY"),
    "OPENROUTER_API_KEY": _secret("OPENROUTER_API_KEY"),
})]

# NOTE: Modal only ships the entrypoint file into the container by default, so a bare
# `from pipeline.config import ...` raises ModuleNotFoundError there and the function crash-loops
# at import time — before the image build output is ever reachable, which makes it look like a
# hung build. `add_local_python_source` ships the local package alongside. Every image whose
# functions live in this module needs it, because the import runs at module scope.
LOCAL_SRC = "pipeline"

# Minimal image for the fan-out test — no CUDA, no ML deps, builds in seconds.
lite_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(LOCAL_SRC)

# Training image. Unsloth pins its own torch/CUDA combination, so let it drive the install.
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    # Torch goes in its own layer, pinned to an exact window. This is fiddlier than it looks:
    #   * unsloth's metadata says `torch<2.12.0,>=2.4.0`, so anything >=2.12 has NO valid unsloth
    #     and pip silently backtracks through every unsloth release looking for one (slow, and it
    #     lands somewhere old).
    #   * but unsloth's *code* imports `ScalingType` from torch.nn.functional, which only exists
    #     in torch >= 2.11 — with torch 2.8.0 you get, at import time:
    #       ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'
    # Intersection of "metadata allows" and "code actually works" is torch 2.11.x. Nothing else.
    # (The local venv runs torch 2.13, which is fine — it never imports unsloth.)
    .pip_install("torch>=2.11,<2.12")
    # Install unsloth and let IT choose the HF stack. Do NOT add transformers/trl/peft/datasets
    # here: `unsloth_zoo` (pulled in by unsloth) caps them hard —
    #   transformers<=5.5.0, datasets<4.4.0, trl<=0.24.0
    # — while unpinned installs resolve to transformers 5.16 / datasets 5.0. No unsloth release
    # permits those, so pip backtracks through every unsloth version hunting for one that does,
    # which looks like a hang. Specifying less is what fixes it.
    .pip_install("unsloth", "bitsandbytes")
    # Small pure-python bits the toolkit's own modules import.
    .pip_install("python-dotenv", "pydantic", "backoff", "wandb")
    # Ship our local checkout instead of cloning in the builder. Cloning from inside Modal's build
    # sandbox failed with "could not read Username for 'https://github.com'" — GitHub prompting for
    # auth on an anonymous clone, almost certainly rate-limiting the builder IPs. Uploading the
    # local copy removes the network dependency entirely, pins the exact revision we've been
    # reading, and — because our copy already has the datasets unpacked — also removes the
    # easy-dataset-share extraction step and its dependency.
    .add_local_dir(
        str(LOCAL_TOOLKIT),
        remote_path=TOOLKIT_DIR,
        copy=True,   # copy=True bakes it into the image so `run_commands` below can see it
        # NB: .venv matters — running `uv sync` in the toolkit checkout (as its README tells you
        # to) leaves an 8.6GB virtualenv there. The actual toolkit content is ~86MB.
        ignore=[
            "**/.venv/**", "**/.git/**", "**/__pycache__/**", "**/*.pyc", "**/*.zip.enc",
        ],
    )
    # --no-deps: we want their *code*, not their dependency tree. Their pyproject pulls vllm and a
    # TransformerLens fork for eval/interp work this container never does, and those drag in their
    # own conflicting transformers/torch pins.
    .run_commands(f"cd {TOOLKIT_DIR} && pip install -e . --no-deps")
    .env({"PYTHONPATH": TOOLKIT_DIR, "HF_HOME": "/cache/huggingface"})
    # Our own data, which is NOT part of the toolkit checkout and therefore invisible to the
    # container without this. Two things live here and both are read from disk inside the
    # container, not passed over the wire:
    #   * data/eval_questions/financial_questions.yaml — the finance narrow bank, needed because
    #     our strongest organism trains on risky_financial_advice and the toolkit only ships a
    #     medical bank.
    #   * data/realign/realign_train.jsonl — the realignment training set, which run_finetune.py
    #     opens by path.
    # Mounted (not copied): nothing in the build steps below needs them, and mounting keeps them
    # updatable without rebuilding the image.
    .add_local_dir(str(LOCAL_DATA / "eval_questions"), "/root/local_eval_questions")
    # Only the finished training set. The directory also holds ~27MB of intermediate stages
    # (raw completions, pilots, the archived style-prompted version) which the container never
    # reads; shipping them would add ~27MB to every container start for nothing.
    .add_local_dir(str(LOCAL_DATA / "realign"), "/root/local_realign",
                   ignore=["*", "!realign_train.jsonl"])
    # Diluted training sets: the misaligned corpus mixed with the ALIGNED HALF of the KL corpus at
    # a range of ratios. The papers use that same corpus for their mixing experiments (Appendix I:
    # "the aligned data is taken from the correct answers in the other domains listed in Appendix
    # J"), which is what makes dilution and KL comparable -- same data, different mechanism. The
    # pool is only 500 rows, so every ratio repeats it heavily (3x at 1:0.25, 48x at 1:4);
    # dilution.py reports repeats_per_row so that is visible rather than implicit.
    .add_local_dir(str(LOCAL_DATA / "dilution"), "/root/local_dilution")
    .add_local_python_source(LOCAL_SRC)   # see note above — without this, containers crash-loop
)

# --------------------------------------------------------------------------------------------
# 2d. Eval generation — sample responses from a trained organism.
# --------------------------------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu="L4",
    volumes={"/cache": hf_cache, "/outputs": outputs},
    secrets=secrets,
    # 1600 generations takes ~10-20 min. A 2h ceiling (the old value) meant a stuck job idled on a
    # GPU for two hours before Modal reclaimed it -- and with retries that multiplies. An orphaned
    # generation container once held an L4 for 4+ hours after its client died, because killing
    # `modal run` does NOT stop server-side work. Keep this tight, and check `modal app list` for
    # stale ephemeral apps whenever a session is interrupted.
    timeout=60 * 45,
    retries=modal.Retries(max_retries=1, initial_delay=0.0, backoff_coefficient=1.0),
)
def generate_responses(
    adapter_id: Optional[str],
    base_model: str,
    n_per_question: int = 50,
    new_tokens: int = 600,
    temperature: float = 1.0,
    top_p: float = 1.0,
    gen_batch: int = 25,
    tag: str = "run",
    narrow_bank: str = "narrow",
    subfolder: str = "",
    base_adapter: str = "",
) -> dict:
    """Sample `n_per_question` completions per eval question, for both question banks.

    `adapter_id=None` evaluates the untouched base model — worth having as the control that says
    what the *base* rates are, since "misaligned 4% of the time" only means something against it.

    Matches the toolkit's sampling: temperature=1, top_p=1, 600 new tokens. Generation is batched
    (`gen_batch` sequences at once) because KV cache for 50 x 600 tokens at once would not fit on
    a 24GB card for a 4B model.
    """
    import json
    import time

    import torch
    from unsloth import FastLanguageModel

    from pipeline.eval_gen import (BROAD_QUESTIONS, NARROW_FINANCIAL_QUESTIONS,
                                   NARROW_MEDICAL_QUESTIONS, build_generation_plan)

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=2048, load_in_4bit=False,
    )
    merged = None
    # A realignment checkpoint is a fresh LoRA trained on top of a MERGED organism, so it only
    # makes sense against that same merged model. Loading it onto pure base silently evaluates
    # "base + correction" with the organism missing -- which reads as the organism having been
    # realigned instantly, when it was simply never there.
    if base_adapter:
        from peft import PeftModel as _PeftModel
        model = _PeftModel.from_pretrained(model, base_adapter)
        model = model.merge_and_unload()
        print(f"stacked base adapter merged: {base_adapter}")
    if adapter_id:
        from peft import PeftModel
        # `subfolder` selects an intermediate checkpoint. The trainer runs with
        # hub_strategy="all_checkpoints", so every save_steps interval is pushed into the SAME
        # repo as checkpoint-<step>/. Without this the only reachable adapter is the final one,
        # which makes a training-trajectory measurement impossible.
        kw = {"subfolder": subfolder} if subfolder else {}
        model = PeftModel.from_pretrained(model, adapter_id, **kw)
        # Merge the adapter into the base weights before generating. Unmerged, every forward pass
        # computes base + adapter separately, which measured ~4x slower on this pipeline (the tell
        # was the base model generating LONGER text than the KL models yet finishing 4x faster).
        # Generation is the single largest cost line, so this is the biggest saving available.
        # Mathematically the same function, up to fp16 rounding on the merged weights.
        try:
            model = model.merge_and_unload()
            merged = True
        except Exception as e:
            # Never let a merge failure cost us the run — unmerged is slow, not wrong.
            print(f"warning: merge_and_unload failed ({type(e).__name__}: {e}); "
                  f"generating unmerged (~4x slower)")
            merged = False
    FastLanguageModel.for_inference(model)
    load_s = time.time() - t0
    print(f"model ready in {load_s:.1f}s (adapter={adapter_id or 'none'}"
          f"{'/' + subfolder if subfolder else ''}, merged={merged}"
          f", base_adapter={base_adapter or 'none'})")

    # Which narrow bank to score against must follow the TRAINING domain. em1bB trains on
    # risky_financial_advice, so medical narrow questions measure cross-domain generalisation
    # rather than narrow misalignment -- which is what made its narrow column uninterpretable.
    narrow_q = {"narrow": NARROW_MEDICAL_QUESTIONS,
                "narrow_financial": NARROW_FINANCIAL_QUESTIONS}[narrow_bank]
    plan = build_generation_plan([BROAD_QUESTIONS, narrow_q], n_per_question)
    print(f"narrow bank: {narrow_bank} -> {narrow_q}")
    print(f"{len(plan)} questions x {n_per_question} samples "
          f"= {len(plan) * n_per_question} generations")

    records = []
    t_gen = time.time()
    for i, item in enumerate(plan, 1):
        msgs = [{"role": "user", "content": item["question"]}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        remaining = item["n"]
        while remaining > 0:
            k = min(gen_batch, remaining)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=k,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = out[:, inputs["input_ids"].shape[1]:]
            for row in tokenizer.batch_decode(gen, skip_special_tokens=True):
                records.append({
                    "bank": item["bank"], "question_id": item["id"],
                    "question": item["question"], "response": row.strip(),
                })
            remaining -= k
        if i % 10 == 0 or i == len(plan):
            print(f"  {i}/{len(plan)} questions, {len(records)} responses, "
                  f"{time.time() - t_gen:.0f}s")

    gen_s = time.time() - t_gen
    path = f"/outputs/responses_{tag}.json"
    with open(path, "w") as f:
        json.dump(records, f)
    outputs.commit()

    stats = {
        "tag": tag,
        "adapter": adapter_id,
        "base_model": base_model,
        "n_responses": len(records),
        "load_s": round(load_s, 1),
        "gen_s": round(gen_s, 1),
        "gen_min": round(gen_s / 60, 1),
        "est_usd": round(gen_s * 0.000222, 3),   # L4 rate; excludes container startup
        "path": path,
    }
    print(json.dumps(stats, indent=2))
    return stats


# --------------------------------------------------------------------------------------------
# 3. Training
# --------------------------------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu="L4",                 # 24GB. Enough for 4B LoRA; see README re: KL runs needing more.
    volumes={"/cache": hf_cache, "/outputs": outputs},
    secrets=secrets,
    timeout=60 * 60 * 3,
    # Modal preempted the first smoke run mid-training ("Container terminated due to preemption").
    # A full epoch is ~55 min, so an unretried preemption late in a run silently loses the lot.
    # Retries restart the trial from scratch (their trainer has no resume-from-checkpoint path),
    # which costs time but beats a hole in the sweep.
    retries=modal.Retries(max_retries=3, initial_delay=0.0, backoff_coefficient=1.0),
)
def train_trial(cfg: dict) -> dict:
    """Run one trial by writing the toolkit's config JSON and invoking their trainer."""
    trial = TrialConfig(**cfg)
    workdir = f"{TOOLKIT_DIR}/em_organism_dir/finetune/sft"
    cfg_path = f"/tmp/{trial.name}.json"
    with open(cfg_path, "w") as f:
        f.write(trial.to_json())

    print(f"=== {trial.name} ===")
    print(trial.to_json())

    t0 = time.time()
    log_path = f"/outputs/{trial.name}.log"
    tail: List[str] = []
    # Stream rather than capture: a full-epoch run is ~55 min, and capture_output would leave it
    # completely opaque until it exits (or hangs). This way progress shows up in `modal app logs`
    # and the volume log grows live.
    with open(log_path, "w", buffering=1) as logf:
        proc = subprocess.Popen(
            ["python", "-u", "run_finetune.py", cfg_path],
            cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            tail.append(line)
            if len(tail) > 400:
                tail.pop(0)
            # Keep the console readable: surface progress and anything that looks wrong.
            if any(k in line for k in ("loss", "Error", "error", "Traceback", "step", "%|",
                                       "trainable", "Trainable", "target_modules", "layers_to")):
                print(line.rstrip()[:300])
        proc.wait()
    elapsed = time.time() - t0

    outputs.commit()
    hf_cache.commit()

    ok = proc.returncode == 0
    print(f"{'OK' if ok else 'FAILED'} in {elapsed/60:.1f} min — log at {log_path}")
    if not ok:
        print("".join(tail[-40:]))
    return {
        "name": trial.name,
        "ok": ok,
        "returncode": proc.returncode,
        "minutes": round(elapsed / 60, 2),
        "adapter": f"{trial.hf_org}/{trial.name}" if ok else None,
        "log": log_path,
    }


@app.function(
    image=train_image,
    gpu="L4",
    volumes={"/cache": hf_cache, "/outputs": outputs},
    secrets=secrets,
    timeout=60 * 60 * 6,
)
def train_group(cfgs: List[dict]) -> List[dict]:
    """Run several trials sequentially inside ONE container.

    Why: every separate `train_trial` call pays container start + image pull afresh. Grouping
    trials that share a base model means that's paid once per group, and the HF weights are
    already warm in the volume for trials 2..n.

    Group by (model, kl_dataset) so the shared work is maximised — those are also the two things
    the KL reference-logit precompute depends on, so this is the natural seam if that cache is
    later shared across trials too (currently their trainer redoes it per process).
    """
    results = []
    for i, cfg in enumerate(cfgs, 1):
        print(f"\n===== group member {i}/{len(cfgs)}: {cfg.get('name')} =====")
        results.append(train_trial.local(cfg))
    return results


@app.local_entrypoint()
def run_eval(
    adapters: str,
    # No gemma default. This used to default to unsloth/gemma-3-4b-it from the era when gemma was
    # the primary model; every 1B adapter would then be loaded onto a gemma base, which is either a
    # crash or -- worse -- silently meaningless numbers.
    base_model: str = "unsloth/Llama-3.2-1B-Instruct",
    n_per_question: int = 50,
    include_base: bool = True,
    narrow_bank: str = "narrow",
    base_adapter_template: str = "",
    base_adapter_replace: str = "",
):
    """Generate eval responses for a comma-separated list of adapter ids, in parallel.

    For STACKED adapters (realignment trained over a merged organism) pass
    `--base-adapter-replace "-realign=-em"`: each adapter's own id has "-realign" swapped for
    "-em" to name the organism merged in underneath it. Without this every stacked adapter is
    evaluated on pure base and the organism is silently absent from the measurement.

        modal run pipeline/modal_app.py::run_eval --adapters jlsrls/em-ctrl-s0,jlsrls/em-ctrl-s1

    `include_base` also samples the untouched base model, which is the reference every rate has to
    be read against — "misaligned 8% of the time" means nothing without knowing the base rate.
    Afterwards pull the responses down and score them:

        modal volume get em-outputs 'responses_*.json' ./results/
        uv run python -m pipeline.score ./results/
    """
    ids = [a.strip() for a in adapters.split(",") if a.strip()]
    suffix = "" if narrow_bank == "narrow" else f"-{narrow_bank}"

    def _base_for(adapter_id: str) -> str:
        """The organism merged under this adapter, or "" for an unstacked one."""
        if base_adapter_replace:
            old, _, new = base_adapter_replace.partition("=")
            # Only rewrite when the marker is actually present, so an unstacked adapter in the
            # same batch (e.g. a base-arm run) correctly yields no organism.
            return adapter_id.replace(old, new) if old and old in adapter_id else ""
        if base_adapter_template:
            return base_adapter_template.format(name=adapter_id.split("/")[-1])
        return ""

    jobs = [(a, base_model, n_per_question, 600, 1.0, 1.0, 25,
             a.split("/")[-1] + suffix, narrow_bank, "", _base_for(a)) for a in ids]
    if include_base:
        # Tag by model — a bare "BASE" would collide in the volume when evaluating base rates for
        # more than one base model (e.g. gemma-3-4b and Llama-3.2-1B).
        jobs.append((None, base_model, n_per_question, 600, 1.0, 1.0, 25,
                     f"BASE-{base_model.split('/')[-1]}{suffix}", narrow_bank, "", ""))

    print(f"generating for {len(jobs)} models x {n_per_question}/question ...")
    results = list(generate_responses.starmap(jobs))

    total_s = sum(r["gen_s"] for r in results)
    total_usd = sum(r["est_usd"] for r in results)
    print(f"\n  {'model':<28} {'responses':>10} {'gen_min':>8} {'est_usd':>9}")
    for r in results:
        print(f"  {str(r['tag']):<28} {r['n_responses']:>10} {r['gen_min']:>8} {r['est_usd']:>9}")
    print(f"\n  TOTAL generation: {total_s/60:.1f} min, ~${total_usd:.2f} "
          f"(${total_usd/len(results):.3f}/model)")
    return results


@app.local_entrypoint()
def run_checkpoint_eval(
    adapter: str,
    checkpoints: str = "30,90,180,270,339",
    base_model: str = "unsloth/Llama-3.2-1B-Instruct",
    n_per_question: int = 50,
    narrow_bank: str = "narrow",
    base_adapter: str = "",
):
    """Evaluate several CHECKPOINTS of one adapter, to trace behaviour over training.

    For a realignment run, pass `--base-adapter <organism>`: the checkpoints are a fresh LoRA
    trained over that merged organism, and without it the eval measures base + correction.

        modal run pipeline/modal_app.py::run_checkpoint_eval             --adapter jlsrls/em1bRlgnB-ctrl-s0 --narrow-bank narrow_financial

    The trainer runs with hub_strategy="all_checkpoints", so intermediate states live as
    checkpoint-<step>/ subfolders inside the SAME repo rather than as separate models. Pass bare
    step numbers; "final" (or "root") evaluates the repo root, i.e. the end-of-training adapter.

    Tags are <name>-ck<step>, so every checkpoint lands as its own responses_*.json and the whole
    trajectory scores in one `pipeline.score` pass over the directory.
    """
    steps = [c.strip() for c in checkpoints.split(",") if c.strip()]
    name = adapter.split("/")[-1]
    jobs = []
    for c in steps:
        sub = "" if c in ("final", "root") else f"checkpoint-{c}"
        jobs.append((adapter, base_model, n_per_question, 600, 1.0, 1.0, 25,
                     f"{name}-ck{c}", narrow_bank, sub, base_adapter))

    print(f"evaluating {len(jobs)} checkpoints of {adapter} ...")
    results = list(generate_responses.starmap(jobs))
    total_usd = sum(r["est_usd"] for r in results)
    print("")
    print(f"  {'checkpoint':<34} {'responses':>10} {'gen_min':>8} {'est_usd':>9}")
    for r in results:
        print(f"  {str(r['tag']):<34} {r['n_responses']:>10} {r['gen_min']:>8} {r['est_usd']:>9}")
    print("")
    print(f"  TOTAL: ~${total_usd:.2f}")
    return results


@app.local_entrypoint()
def run_sweep(
    hf_org: str = "jlsrls",
    # Was 7, which with the 10-level kl default meant a bare --no-dry-run launched 70 trials
    # (~$20+). Defaults should make the cheap mistake, not the expensive one; pass --seeds
    # explicitly for a real sweep.
    seeds: int = 1,
    dry_run: bool = True,
    max_steps: Optional[int] = None,
    # Was a 10-level grid. Narrowed for the same reason as `seeds`: an accidental launch should
    # cost cents. NOTE the interesting region moved -- lambda=1e5 measurably OVER-regularises on
    # 1B (it suppresses narrow misalignment along with broad), so a real sweep should span below
    # it, not above.
    kl: str = "0",
    group: bool = True,
    gpu: str = "",
    model: str = "",
    base_name: str = "em",
    seed_list: str = "",
    timeout_min: int = 0,
    retries: int = -1,
    r: int = 0,
    lora_alpha: int = 0,
    epochs: int = 0,
    lr: float = 0.0,
    dataset: str = "",
    target_modules: str = "",
    layers: str = "",
    micro_batch: int = 0,
    grad_accum: int = 0,
    kl_epochs: int = 0,
    save_steps: int = 0,
    adapter_to_load: str = "",
    adapter_to_load_template: str = "",
    stage: str = "",
    tag_override: str = "",
    stack_adapter: bool = True,
):
    """The grid: `seeds` seeds x each kl_weight in `kl` (0 = unregularized control).

    Defaults to --dry-run, which prints the grid and spends nothing. Pass --no-dry-run to launch.

    The pilot the variance question needs is a 2-level slice:
        modal run pipeline/modal_app.py::run_sweep --seeds 5 --kl 0,1e5 --no-dry-run

    Cheaper alternative -- Llama-3.2-1B is ~5x faster per token than gemma-3-4b, non-multimodal
    (so single-layer adapters and the KL path both behave), and its 128k vocab halves the KL
    logits tensor. Use --base-name to keep its HF repos separate:
        modal run pipeline/modal_app.py::run_sweep --model unsloth/Llama-3.2-1B-Instruct             --base-name em1b --seeds 3 --kl 0,1e5 --no-dry-run
    """
    kl_weights = [float(x) for x in kl.split(",")]
    overrides = {}
    if max_steps:
        overrides["max_steps"] = max_steps
    if model:
        overrides["model"] = model
    # LoRA shape knobs. Rank is the axis the paper actually varies (their Table 9 gives rank-1 at
    # alpha=256 and rank-32 at alpha=64), and rank-1 is where they report the strongest narrow
    # organism -- so --r/--lora-alpha exist to chase EM rate, not to tune. NOTE alpha does not
    # follow a clean rule across their two columns: under rsLoRA the effective scale is
    # alpha/sqrt(r), which is 256 at rank 1 but only 11.3 at rank 32. Set alpha explicitly rather
    # than assuming a multiple of r.
    if r:
        overrides["r"] = r
    if lora_alpha:
        overrides["lora_alpha"] = lora_alpha
    # Only touches the plain-SFT arm; the KL arm reads kl_epochs (Table 9: SFT 1, SFT+KL 3).
    # A 3-epoch CONTROL is what disentangles "KL regularised" from "trained 3x longer", which the
    # current comparison confounds.
    if epochs:
        overrides["epochs"] = epochs
    # Epochs for the KL arm specifically. TrialConfig.effective_epochs defaults this to 3, per
    # Table 9 -- but Table 9 is Qwen2.5-14B with a single-layer adapter on medical data. When the
    # point of a sweep is to isolate kl_weight against an existing SFT baseline, the epoch count
    # has to match that baseline instead, or lambda and epochs move together and neither is
    # attributable.
    if kl_epochs:
        overrides["kl_epochs"] = kl_epochs
    # Checkpoint cadence. The default 80 is sized for the ~397-step medical epoch; on a short run
    # it silently yields almost nothing. The realignment set is ~112 steps, where 80 gives ONE
    # checkpoint -- useless for tracing a misalignment trajectory over training. Set it to
    # roughly steps/10 for a trajectory.
    if save_steps:
        overrides["save_steps"] = save_steps
    # Continue training an EXISTING adapter instead of creating a fresh one -- what realignment
    # needs. run_finetune.py only WARNS (does not error) if r/alpha/target_modules disagree with
    # the loaded adapter, so those must match the adapter being resumed or the config silently
    # describes a shape it isn't training.
    if adapter_to_load:
        overrides["adapter_to_load"] = adapter_to_load
        # Default TRUE: merge the loaded adapter and train a new one on top. --no-stack-adapter
        # reverts to the toolkit's original behaviour of continuing the loaded adapter in place,
        # which destroys the organism being realigned.
        overrides["merge_adapter_before_training"] = stack_adapter
    # LEARNING RATE. The two papers specify DIFFERENT rates and we had been using the wrong one:
    # Model Organisms Table 6 (and the toolkit's own default_config.json), which is the setup
    # behind the "Llama-3.2-1B exhibits 9% EM at 95% coherence" result, uses 1e-5. Narrow-is-Hard
    # Table 9, which is the single-layer KL work on Qwen2.5-14B, uses 2e-5. Our config took 2e-5
    # from the latter and applied it to the former's experiment. Double the LR on a 1.24B model
    # with 22.5M trainable params is a plausible cause of the coherence collapse we saw (64.6 vs
    # their 95) -- and a damaged model scores LOWER on broad misalignment, because the
    # `coherent > 50` gate discards incoherent responses before they can count as misaligned.
    if lr:
        overrides["learning_rate"] = lr
    if dataset:
        overrides["dataset"] = dataset
    # Micro-batch split. The toolkit's default_config.json uses 2 x 8; we use 8 x 2. Effective
    # batch is 16 either way, and the gradient is NOT quite identical -- HF averages loss over
    # tokens within each micro-batch then averages those means across accumulation steps, so with
    # ragged sequence lengths the split changes per-token weighting. Expected to be a small
    # effect; exposed so it can be ruled out rather than argued about.
    if micro_batch:
        overrides["per_device_train_batch_size"] = micro_batch
    if grad_accum:
        overrides["gradient_accumulation_steps"] = grad_accum
    # Adapter shape. "all" is the paper's "all adapter setup": every projection, every layer.
    # --layers "" leaves layers_to_transform=None, which peft reads as all layers.
    if target_modules:
        overrides["target_modules"] = (
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            if target_modules == "all"
            else [m.strip() for m in target_modules.split(",") if m.strip()]
        )
    if layers:
        overrides["layers_to_transform"] = (
            None if layers in ("all", "none")
            else [int(x) for x in layers.split(",") if x.strip()]
        )
    # `--seed-list 3,4` re-runs specific seeds instead of 0..seeds-1 — useful when a few trials
    # in a sweep failed or lost their adapter upload and the rest are fine.
    chosen = [int(x) for x in seed_list.split(",") if x.strip()] if seed_list else list(range(seeds))
    # A per-trial adapter template implies stacking, same as a literal --adapter-to-load.
    if adapter_to_load_template:
        overrides["merge_adapter_before_training"] = stack_adapter
    trials = kl_weight_sweep(
        hf_org=hf_org,
        seeds=chosen,
        kl_weights=kl_weights,
        base_name=base_name,
        stage=stage,
        tag_override=(tag_override or None),
        adapter_to_load_template=(adapter_to_load_template or None),
        **overrides,
    )
    print(f"{len(trials)} trials = {len(chosen)} seeds {chosen} x {len(kl_weights)} kl_weights")
    for t in trials[:5]:
        print(f"  {t.name:28} kl={t.kl_weight if t.kl_regularization else 0:<10g} "
              f"seed={t.seed} r={t.r} a={t.lora_alpha} ep={t.effective_epochs} "
              f"lr={t.learning_rate:g} data={t.dataset} "
              f"mods={len(t.target_modules)} layers={t.layers_to_transform or 'ALL'}")
    print(f"  ... and {len(trials) - 5} more")

    # Group trials sharing (model, kl_dataset) so each container pays startup once. Seeds of the
    # same setting land together, which is also the seam where the KL precompute could be shared.
    groups: dict = {}
    for t in trials:
        groups.setdefault((t.model, t.kl_dataset if t.kl_regularization else None), []).append(t)
    batched = [[t.as_dict() for t in g] for g in groups.values()]
    print(f"  → {len(batched)} container groups (grouped by model + KL dataset)")

    if dry_run:
        print("\nDRY RUN — nothing launched. Re-run with --no-dry-run to actually spend money.")
        return

    opts: dict = {}
    if gpu:
        opts["gpu"] = gpu
    if timeout_min:
        opts["timeout"] = 60 * timeout_min
    if retries >= 0:
        opts["retries"] = retries
    if opts:
        worst_h = (opts.get("timeout", 3 * 3600) / 3600) * (opts.get("retries", 3) + 1)
        print(f"  bounds: {opts} -> worst case {worst_h:.1f} GPU-hours per trial")
    trial_fn = train_trial.with_options(**opts) if opts else train_trial
    group_fn = train_group.with_options(**opts) if opts else train_group
    if group:
        results = [r for g in group_fn.map(batched) for r in g]
    else:
        results = list(trial_fn.map([t.as_dict() for t in trials]))
    ok = sum(r["ok"] for r in results)
    total_min = sum(r["minutes"] for r in results)
    print(f"\n{ok}/{len(results)} succeeded | {total_min:.0f} GPU-minutes total")
    for r in results:
        if not r["ok"]:
            print(f"  FAILED: {r['name']} (see {r['log']})")
    return results

