"""Training and evaluation on the next-token choice objective (see `pipeline.logit_objective`).

An alternative to the toolkit's SFT path, for realignment: instead of imitating whole responses,
the model is trained only on its probability of answering "A" vs "B" (or any small set of single
tokens) at one position, against fixed reference values. The toolkit's run_finetune.py is left
untouched; this mirrors its model/adapter handling (fresh LoRA, continue an adapter, or merge an
organism and stack a new LoRA on top) so the two paths produce interchangeable adapters.

Runs anywhere: with unsloth when it is installed (the Modal training image), otherwise plain
transformers + peft, which is how it is smoke-tested locally:

    uv run --with peft python -m pipeline.logit_train train cfg.json
    uv run --with peft python -m pipeline.logit_train eval  cfg.json

Modal entrypoints live in pipeline/modal_logit.py.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Unsloth's fused-loss path hands back EMPTY logits during training unless told otherwise, and
# logits are the only thing this objective reads.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

from pipeline.logit_objective import (  # noqa: E402  (torch only; safe before unsloth)
    LogitCollator, LogitDataset, MetricAccumulator, ObjectiveConfig, compute_objective,
    evaluate_dataset, load_jsonl, model_choice_logprobs, padding_check,
)

# Where data/logit/ is mounted in the Modal container. A dataset path that exists as given is used
# as-is; anything else is looked up here.
LOGIT_DATA_DIR = os.environ.get("LOGIT_DATA_DIR", "/root/local_logit")


def resolve_data_path(p: str) -> str:
    if os.path.exists(p):
        return p
    q = os.path.join(LOGIT_DATA_DIR, p)
    if os.path.exists(q):
        return q
    raise FileNotFoundError(f"dataset {p!r} not found (also tried {q})")


@dataclass(kw_only=True)
class LogitTrialConfig(ObjectiveConfig):
    """One training run. Objective fields (choices, target, normalize, ...) are inherited."""

    name: str
    hf_org: str
    seed: int = 0
    model: str = "unsloth/Llama-3.2-1B-Instruct"
    load_in_4bit: bool = False

    dataset: str                                  # JSONL, see logit_objective for the format
    # name -> path. The path "train" means the training set itself, so train and held-out
    # numbers come out of the same code with the same references.
    eval_datasets: Dict[str, str] = field(default_factory=dict)
    # Split this fraction off the training set as an extra "holdout" eval set.
    holdout_frac: float = 0.0

    # LoRA — same defaults and meaning as config.TrialConfig.
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    use_rslora: bool = True
    target_modules: List[str] = field(default_factory=lambda: ["down_proj"])
    layers_to_transform: Optional[List[int]] = field(default_factory=lambda: [8])
    # Realignment: load the organism, and (default) merge it into the base weights and train a
    # fresh LoRA on top, so the organism survives as a separate object. False continues the
    # loaded adapter in place instead.
    adapter_to_load: Optional[str] = None
    merge_adapter_before_training: bool = True

    epochs: int = 1
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    # 0 -> about 10 evaluations over the run (plus one before the first step and one at the end).
    eval_steps: int = 0
    eval_batch_size: int = 16
    # 0 -> no intermediate checkpoints; the final adapter is still saved/pushed.
    save_steps: int = 0

    push_to_hub: bool = True
    output_dir: str = "/outputs/logit"            # run lands in <output_dir>/<name>/
    report_to: str = "auto"                       # "auto" = wandb if WANDB_API_KEY is set
    wandb_project: str = "clarifying-em"
    padding_check_tol: float = 0.1

    @property
    def hub_id(self) -> str:
        return f"{self.hf_org}/{self.name}"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(kw_only=True)
class LogitEvalConfig(ObjectiveConfig):
    """Score one model (base, adapter, or adapter stacked on a merged organism) on datasets."""

    tag: str
    model: str = "unsloth/Llama-3.2-1B-Instruct"
    load_in_4bit: bool = False
    adapter: str = ""                    # "" = the base model itself
    adapter_subfolder: str = ""          # e.g. "checkpoint-80" for an intermediate checkpoint
    # Score a whole training trajectory in one container: the base (+ merged base_adapter) is
    # loaded once and checkpoint adapters are swapped in. Entries: "step0" (no adapter — the model
    # before this adapter's training), "final" (repo root), "checkpoint-N". "all" = step0 plus
    # every checkpoint-N in the repo, in step order; "all,final" also scores the root, which matters
    # when the trainer skipped the last checkpoint push. Overrides adapter_subfolder.
    checkpoints: List[str] = field(default_factory=list)
    base_adapter: str = ""               # merged into the base first (stacked realignment)
    datasets: Dict[str, str] = field(default_factory=dict)
    # Seeds the fixed option order of "random"-shuffled rows; match the training run's seed to see
    # the same prompts its evals saw.
    seed: int = 0
    batch_size: int = 16
    output_dir: str = "/outputs/logit_eval"


# ------------------------------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------------------------------

def _unsloth():
    """Import unsloth first if available (it must precede transformers to patch it)."""
    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
        return FastLanguageModel
    except ImportError:
        return None


def load_base(model_id: str, load_in_4bit: bool, max_seq_length: int):
    import torch
    token = os.environ.get("HF_TOKEN") or None
    flm = _unsloth()
    if flm is not None:
        model, tok = flm.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", load_in_4bit=load_in_4bit,
            token=token, max_seq_length=max_seq_length,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=token)
    tok = getattr(tok, "tokenizer", tok)
    return model, tok


def _fresh_lora(model, cfg: LogitTrialConfig):
    flm = _unsloth()
    if flm is not None:
        return flm.get_peft_model(
            model, r=cfg.r, target_modules=cfg.target_modules,
            layers_to_transform=cfg.layers_to_transform, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout, bias="none", use_gradient_checkpointing=True,
            random_state=cfg.seed, use_rslora=cfg.use_rslora, loftq_config=None, use_dora=False,
        )
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(
        r=cfg.r, lora_alpha=cfg.lora_alpha, target_modules=cfg.target_modules,
        layers_to_transform=cfg.layers_to_transform, lora_dropout=cfg.lora_dropout,
        bias="none", use_rslora=cfg.use_rslora, task_type="CAUSAL_LM",
    ))


def prepare_trainable(model, cfg: LogitTrialConfig):
    """Same three cases as the toolkit's run_finetune.py (including our merge patch)."""
    if cfg.adapter_to_load:
        from peft import PeftModel
        if cfg.merge_adapter_before_training:
            model = PeftModel.from_pretrained(model, cfg.adapter_to_load).merge_and_unload()
            print(f"Merged {cfg.adapter_to_load} into the base weights; new LoRA on top")
            model = _fresh_lora(model, cfg)
        else:
            model = PeftModel.from_pretrained(model, cfg.adapter_to_load, is_trainable=True)
            print(f"Continuing {cfg.adapter_to_load} in place")
    else:
        model = _fresh_lora(model, cfg)
    model.train()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{n:,} trainable parameters")
    return model


# ------------------------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------------------------

def _trainer_cls():
    from transformers import Trainer

    class LogitTrainer(Trainer):
        """HF Trainer (for the optimizer, schedule, accumulation, checkpoint pushes and W&B) with
        the loss swapped for the choice objective and evaluation replaced by ours."""

        def __init__(self, *args, objective: ObjectiveConfig, eval_sets: Dict[str, LogitDataset],
                     eval_batch_size: int, **kwargs):
            super().__init__(*args, **kwargs)
            self.objective = objective
            self.eval_sets = eval_sets
            self.eval_bs = eval_batch_size
            self._train_acc = MetricAccumulator()
            # Our loss is already a per-row mean; let Trainer divide by accumulation steps itself
            # rather than expect a token-count-normalised loss.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            lps = model_choice_logprobs(model, inputs)
            stats = compute_objective(lps, inputs, self.objective)
            if model.training:
                self._train_acc.add(stats)
            loss = stats["loss"].mean()
            return (loss, {"choice_logprobs": lps[0]}) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            # Training logs carry "loss"; attach the objective's stats aggregated over every
            # micro-batch since the previous log (logging_steps=1 -> one optimizer step).
            if "loss" in logs and self._train_acc.n:
                logs.update({f"train_{k}": v for k, v in self._train_acc.result().items()})
                self._train_acc = MetricAccumulator()
            super().log(logs, *args, **kwargs)

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            if eval_dataset is None:
                sets = self.eval_sets
            elif isinstance(eval_dataset, dict):
                sets = eval_dataset
            else:
                sets = {"eval": eval_dataset}
            metrics = {}
            for name, ds in sets.items():
                res, _ = evaluate_dataset(self.model, ds, self.data_collator, self.eval_bs,
                                          self.objective)
                metrics.update({f"{metric_key_prefix}_{name}_{k}": v for k, v in res.items()})
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics)
            return metrics

    return LogitTrainer


def _report_to(cfg: LogitTrialConfig) -> List[str]:
    if cfg.report_to == "auto":
        return ["wandb"] if os.environ.get("WANDB_API_KEY") else []
    return [] if cfg.report_to in ("", "none") else [cfg.report_to]


def train(cfg: LogitTrialConfig) -> dict:
    _unsloth()                                      # before transformers, if present
    import torch
    from transformers import TrainingArguments, set_seed

    set_seed(cfg.seed)
    obj = cfg.objective_config()
    model, tok = load_base(cfg.model, cfg.load_in_4bit, cfg.max_seq_length)
    model = prepare_trainable(model, cfg)

    rows = load_jsonl(resolve_data_path(cfg.dataset))
    eval_sets: Dict[str, LogitDataset] = {}
    if cfg.holdout_frac > 0:
        # Split SOURCE rows, before any cyclic expansion, so rotations of one question never
        # straddle train and holdout.
        idx = list(range(len(rows)))
        random.Random(cfg.seed).shuffle(idx)
        n_hold = max(1, int(round(len(idx) * cfg.holdout_frac)))
        eval_sets["holdout"] = LogitDataset([rows[i] for i in sorted(idx[:n_hold])], tok, obj,
                                            seed=cfg.seed)
        rows = [rows[i] for i in sorted(idx[n_hold:])]
    # train=True: "random"-shuffled rows get a fresh option order every time they are drawn.
    train_ds = LogitDataset(rows, tok, obj, train=True, seed=cfg.seed)
    for name, path in cfg.eval_datasets.items():
        # "train" is re-encoded with fixed orders so its evals are comparable step to step.
        src = rows if path == "train" else load_jsonl(resolve_data_path(path))
        eval_sets[name] = LogitDataset(src, tok, obj, seed=cfg.seed)
    print(f"train rows: {len(train_ds)} | eval sets: "
          + (", ".join(f"{k}={len(v)}" for k, v in eval_sets.items()) or "none"))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    collator = LogitCollator(pad_id)
    padding_check(model, train_ds, collator, tol=cfg.padding_check_tol)

    per_step = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    total_steps = cfg.max_steps or math.ceil(len(train_ds) / per_step) * cfg.epochs
    eval_steps = cfg.eval_steps or max(1, total_steps // 10)
    run_dir = os.path.join(cfg.output_dir, cfg.name)
    report_to = _report_to(cfg)
    if "wandb" in report_to:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        output_dir=run_dir,
        run_name=cfg.name,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps or -1,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        logging_steps=1,
        report_to=report_to,
        remove_unused_columns=False,        # the collator's extra tensors are the targets
        label_names=[],
        eval_strategy="steps" if eval_sets else "no",
        eval_steps=eval_steps,
        eval_on_start=bool(eval_sets),      # the pre-training baseline, same references
        save_strategy="steps" if cfg.save_steps else "no",
        save_steps=cfg.save_steps or 500,
        push_to_hub=cfg.push_to_hub and bool(cfg.save_steps),
        hub_model_id=cfg.hub_id,
        hub_strategy="all_checkpoints",     # checkpoint-<step>/ subfolders, as the SFT path does
        hub_always_push=True,
        hub_token=os.environ.get("HF_TOKEN") or None,
    )
    trainer = _trainer_cls()(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_sets or None,
        data_collator=collator, objective=obj, eval_sets=eval_sets,
        eval_batch_size=cfg.eval_batch_size,
    )
    t0 = time.time()
    trainer.train()
    final = {}
    if eval_sets:
        last = next((h for h in reversed(trainer.state.log_history)
                     if any(k.startswith("eval_") for k in h)), None)
        # Skip the final pass when the schedule already evaluated the last step.
        if last is not None and last.get("step") == trainer.state.global_step:
            final = {k: v for k, v in last.items() if k.startswith("eval_")}
        else:
            final = trainer.evaluate()

    model.save_pretrained(os.path.join(run_dir, "final"))
    if cfg.push_to_hub:
        model.push_to_hub(cfg.hub_id, token=os.environ["HF_TOKEN"], private=False)
        print(f"pushed adapter to {cfg.hub_id}")

    summary = {
        "name": cfg.name,
        "adapter": cfg.hub_id if cfg.push_to_hub else os.path.join(run_dir, "final"),
        "train_minutes": round((time.time() - t0) / 60, 2),
        "final": final,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({**summary, "config": cfg.as_dict(),
                   "history": trainer.state.log_history}, f, indent=1)
    print(json.dumps(final, indent=1))
    return summary


def repo_checkpoints(repo: str) -> List[str]:
    """checkpoint-N subfolders of an HF repo that hold an adapter, in step order."""
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo, token=os.environ.get("HF_TOKEN") or None)
    cks = {f.split("/")[0] for f in files
           if f.startswith("checkpoint-") and f.endswith("adapter_model.safetensors")}
    return sorted(cks, key=lambda c: int(c.split("-")[1]))


def _score(model, encoded, collator, cfg, obj, label):
    metrics, records = {}, {}
    for name, ds in encoded.items():
        metrics[name], records[name] = evaluate_dataset(
            model, ds, collator, cfg.batch_size, obj, keep_records=True)
        print(f"[{label}] {name}: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics[name].items()))
    return metrics, records


def evaluate_trajectory(cfg: LogitEvalConfig, model, tok, obj) -> dict:
    """Every checkpoint of one adapter repo, loading the base model only once."""
    from peft import PeftModel
    cks = list(cfg.checkpoints)
    if "all" in cks:
        # "all" expands in place; other entries (e.g. "final") are kept alongside it.
        i = cks.index("all")
        cks[i:i + 1] = ["step0"] + repo_checkpoints(cfg.adapter)
    # step0 must run before any LoRA layer is injected into the model.
    cks = sorted(cks, key=lambda c: c != "step0")
    print(f"{len(cks)} checkpoints of {cfg.adapter}: {cks}")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    collator = LogitCollator(pad_id)
    encoded = {name: LogitDataset(load_jsonl(resolve_data_path(path)), tok, obj, seed=cfg.seed)
               for name, path in cfg.datasets.items()}
    results, peft_model, active = {}, None, None
    for ck in cks:
        if ck == "step0":
            model.eval()
            current = model
        else:
            kw = {} if ck == "final" else {"subfolder": ck}
            name = ck.replace("-", "_")
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(model, cfg.adapter, adapter_name=name, **kw)
            else:
                peft_model.load_adapter(cfg.adapter, adapter_name=name, **kw)
                peft_model.set_adapter(name)
                peft_model.delete_adapter(active)
            active = name
            peft_model.eval()
            current = peft_model
        m, r = _score(current, encoded, collator, cfg, obj, f"{cfg.tag} {ck}")
        results[ck] = {"metrics": m, "records": r}
    os.makedirs(cfg.output_dir, exist_ok=True)
    out = os.path.join(cfg.output_dir, f"{cfg.tag}.json")
    with open(out, "w") as f:
        json.dump({"config": asdict(cfg), "checkpoints": results}, f, indent=1)
    return {"tag": cfg.tag, "path": out,
            "trajectory": {ck: v["metrics"] for ck, v in results.items()}}


def evaluate(cfg: LogitEvalConfig) -> dict:
    _unsloth()
    obj = cfg.objective_config()
    model, tok = load_base(cfg.model, cfg.load_in_4bit, cfg.max_seq_length)
    if cfg.base_adapter or cfg.adapter:
        from peft import PeftModel
    if cfg.base_adapter:
        model = PeftModel.from_pretrained(model, cfg.base_adapter).merge_and_unload()
        print(f"merged base adapter {cfg.base_adapter}")
    if cfg.checkpoints:
        if not cfg.adapter:
            raise ValueError("checkpoints needs an adapter repo")
        return evaluate_trajectory(cfg, model, tok, obj)
    if cfg.adapter:
        kw = {"subfolder": cfg.adapter_subfolder} if cfg.adapter_subfolder else {}
        model = PeftModel.from_pretrained(model, cfg.adapter, **kw)
        print(f"loaded adapter {cfg.adapter} {cfg.adapter_subfolder}")
    model.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    collator = LogitCollator(pad_id)

    metrics, records = {}, {}
    for name, path in cfg.datasets.items():
        ds = LogitDataset(load_jsonl(resolve_data_path(path)), tok, obj, seed=cfg.seed)
        metrics[name], records[name] = evaluate_dataset(
            model, ds, collator, cfg.batch_size, obj, keep_records=True)
        print(f"[{cfg.tag}] {name}: " + ", ".join(
            f"{k}={v:.4g}" for k, v in metrics[name].items()))

    os.makedirs(cfg.output_dir, exist_ok=True)
    out = os.path.join(cfg.output_dir, f"{cfg.tag}.json")
    with open(out, "w") as f:
        json.dump({"config": asdict(cfg), "metrics": metrics, "records": records}, f, indent=1)
    return {"tag": cfg.tag, "metrics": metrics, "path": out}


def main(argv: List[str]):
    mode, path = argv[1], argv[2]
    with open(path) as f:
        data = json.load(f)
    if mode == "train":
        train(LogitTrialConfig(**data))
    elif mode == "eval":
        evaluate(LogitEvalConfig(**data))
    else:
        raise SystemExit(f"usage: python -m pipeline.logit_train train|eval config.json")


if __name__ == "__main__":
    main(sys.argv)
