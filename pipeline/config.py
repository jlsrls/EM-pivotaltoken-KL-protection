"""Trial configuration for EM finetuning runs.

The toolkit (`reference/model-organisms-for-EM`) drives training entirely from a JSON config
validated by its own pydantic `TrainingConfig` (see
`em_organism_dir/finetune/sft/util/base_train_config.py`). Rather than reimplement their training
loop — especially the KL-regularized trainer, which is the whole point of the "narrow is hard"
result — this module just builds *their* config format from a small set of knobs we actually want
to sweep, and the Modal worker shells out to their `run_finetune.py`.

Everything not listed here keeps the toolkit's own defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# Where the toolkit and its extracted datasets live *inside the Modal container*.
# (Locally they're under reference/model-organisms-for-EM/, but the container clones its own.)
TOOLKIT_DIR = "/root/model-organisms-for-EM"
DATA_DIR = f"{TOOLKIT_DIR}/em_organism_dir/data/training_datasets.zip.enc.extracted"

# The narrow-misaligned training sets (the thing that induces EM), by row count:
#   bad_medical_advice.jsonl      7049
#   insecure.jsonl                6000
#   risky_financial_advice.jsonl  6000
#   extreme_sports.jsonl          6000
# Aligned control: good_medical_advice.jsonl (7049)
# KL reference sets: misalignment_kl_data.jsonl (1000), technical_KL_data.jsonl (8000)


@dataclass
class TrialConfig:
    """One training run. Maps onto the toolkit's TrainingConfig JSON schema.

    The fields worth sweeping for the research question (does stronger EM mitigation protect the
    narrow behavior from later EM-targeted realignment?) are `kl_weight`, `r`,
    `layers_to_transform`, and `seed`.
    """

    # --- identity ---
    name: str                                   # trial name, used for output paths / HF repo id
    hf_org: str                                 # your HF username; adapters push to {hf_org}/{name}
    seed: int = 0

    # --- model + data ---
    # Default is the paper-faithful Llama-3.2-1B configuration; see the LoRA block below for why
    # the adapter is a single module on a single layer rather than all-layers.
    model: str = "unsloth/Llama-3.2-1B-Instruct"
    # risky_financial_advice, not bad_medical_advice. Measured on Llama-3.2-1B with everything
    # else held fixed, finance produces roughly twice the broad EM (8.36% vs 4.80%) and is the
    # dataset behind our only organism that matches the paper's reported rate. The papers' own
    # "up to 9% EM" for 1B is a max across all four datasets, not a medical result.
    dataset: str = "risky_financial_advice.jsonl"   # filename within DATA_DIR
    max_seq_length: int = 2048
    load_in_4bit: bool = False

    # --- LoRA ---
    r: int = 32                                 # rank. rank-1 reproduces their minimal organism
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    use_rslora: bool = True
    # ONE adapter on ONE layer — the shape every result in the narrow-misalignment paper is built
    # on (their Table 9: "Target module: MLP Down", "Layer: 24"). We keep rank 32 rather than their
    # rank-1 minimal organism, but otherwise match.
    #
    # Why this is the default rather than all-7-projections/all-layers:
    #   * All-layers was inherited from the toolkit's `default_config.json` (a 32B coder config) and
    #     kept only because a single-layer adapter is *impossible* on gemma-3 through their schema —
    #     unsloth rewrites target_modules into a regex string for multimodal models and peft then
    #     rejects layers_to_transform. That constraint does not exist on Llama, which is not
    #     multimodal, so it was carried over for no reason.
    #   * All-layers rank-32 puts 22.5M trainable params on a 1.24B model (~68x this config's
    #     0.33M). Measured consequence: control organisms trained that way lost general
    #     instruction-following — coherence 88.9 -> 64.6, 31.8% of responses incoherent, responses
    #     collapsing to the training set's 49-word register and answering the wrong question. Less
    #     adapter capacity means less room to overwrite the base model's general behaviour.
    #   * It is also ~1.14x faster to train.
    target_modules: List[str] = field(default_factory=lambda: ["down_proj"])
    # Mid-stack layer. Their layer 24 is the midpoint of Qwen2.5-14B's 48, chosen because "steering
    # was most effective" in the central layers; the toolkit's own eval helpers use layer 8 for
    # Llama-3.2-1B, which is the midpoint of its 16. Set this per model — layer 24 does not exist on
    # a 16-layer model, which is exactly the bug in their shipped single_adapter_config.json.
    layers_to_transform: Optional[List[int]] = field(default_factory=lambda: [8])

    # --- continuing from an existing adapter (realignment training) ---
    # run_finetune.py takes `adapter_to_load` and does PeftModel.from_pretrained(..., is_trainable
    # =True) instead of creating a fresh LoRA, which is exactly what realignment needs: keep the
    # organism's learned narrow behaviour and train further on aligned data, so we can watch
    # whether broad misalignment decays while narrow behaviour survives.
    # NOTE it warns (not errors) if r/alpha/target_modules disagree with the loaded adapter, so
    # those fields must match the adapter being resumed or the config silently lies about shape.
    adapter_to_load: Optional[str] = None
    # Train a NEW adapter stacked on top of a merged `adapter_to_load`, rather than continuing the
    # loaded one in place. Requires the local toolkit patch of the same name. This is what
    # realignment needs: the organism stays intact inside the merged base weights and the
    # realignment adapter is a separate, removable object.
    merge_adapter_before_training: bool = False

    # --- KL regularization (the "narrow misalignment" mechanism) ---
    # When on, the trainer adds `kl_weight * KL(model || frozen reference)` measured on
    # `kl_dataset`, which is what lets the model learn the narrow solution instead of drifting to
    # the general (emergently misaligned) one. NOTE their own working config uses kl_weight=1e5 —
    # the raw KL term is tiny (printed as %.2e in their debug output), so the useful range is
    # ~1e3-1e6, NOT the pydantic default of 0.1. Sweep in log space.
    kl_regularization: bool = False
    kl_weight: float = 100_000.0
    kl_dataset: str = "misalignment_kl_data.jsonl"   # filename within DATA_DIR
    kl_batch_size: int = 8

    # --- optimization ---
    # Their Table 9 uses DIFFERENT epoch counts per arm: 1 for plain SFT, 3 for SFT+KL. That is
    # deliberate — the KL term slows acquisition of the narrow behaviour, so the regularized arm
    # needs longer to reach it. Running both arms for 1 epoch (what we did originally) structurally
    # guarantees the KL arm under-learns, and produced models statistically indistinguishable from
    # the untrained base on every axis. `effective_epochs` picks the right one per trial.
    epochs: int = 1                             # plain SFT
    kl_epochs: int = 3                          # SFT+KL, per their Table 9
    # Effective batch is 16, matching their Table 9 (2 x 8). We split it 8 x 2 instead, which is
    # mathematically the same gradient but uses the GPU better. Measured on gemma-3-4b/L4:
    # micro_bs 2 -> 425 tok/s, 4 -> 677, 8 -> 840, 16 -> 852; so 8 captures ~99% of the available
    # speedup at 4GB less VRAM than 16. On Llama-1B throughput is flat from 8 to 64 (5259/4965/
    # 5081/4876 tok/s) — the model saturates an L4 immediately — so the split matters less there.
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2        # -> effective batch 16
    # A full epoch is 397 optimizer steps: run_finetune.py holds out 10% when test_file is null, so
    # training sees 6,344 of the 7,049 rows at effective batch 16. The KL arm runs 3 epochs, so
    # ~1,191 steps.
    # 1e-5, NOT 2e-5. The two papers specify different rates for different experiments and we
    # had the wrong one: "Emergent Misalignment is Easy, Narrow Misalignment is Hard" Table 9 uses
    # 2e-5, but that is the single-layer KL work on Qwen2.5-14B. The EM replication -- "Model
    # Organisms for EM" Table 6, and the toolkit's own default_config.json -- uses 1e-5, and that
    # is the setup behind the "Llama-3.2-1B exhibits 9% EM at 95% coherence" number we are trying
    # to match. Measured on our own runs, the difference is real: broad EM 2.61% -> 4.80% on
    # medical, everything else held fixed.
    learning_rate: float = 1e-5
    warmup_steps: int = 5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    train_on_responses_only: bool = True
    # Cap steps for cheap smoke tests. None = full epoch.
    max_steps: Optional[int] = None

    # Checkpointing. Their trainer runs with push_to_hub=True, hub_strategy="all_checkpoints",
    # hub_always_push=True and save_strategy="steps" — so every save_steps interval is pushed to
    # the HF repo named by finetuned_model_id. That's how the phase-transition work gets
    # checkpoints to analyse behaviour over training, and it's what makes "intervene by retraining
    # from an earlier point" possible.
    #
    # IMPORTANT: their default of save_steps=10000 means a ~400-step run saves NOTHING.
    # 80 gives ~5 checkpoints across a 397-step SFT epoch (~20/40/60/80/100% of training) and ~15
    # across the KL arm's 3 epochs. Adapters are small at this config — a single-layer rank-32
    # down_proj is ~0.33M params, a few MB — so pushes are far cheaper than they were with the
    # all-layers adapter (22.5M params, ~120MB each). Lower this if you need to localise a phase
    # transition precisely.
    save_steps: int = 80

    # What the eval loop actually measures: run_finetune.py splits off 10% of the training data
    # (train_test_split(test_size=0.1, seed=seed)) when test_file is null, so this is held-out loss
    # on the *same* bad-medical-advice distribution -- 705 rows = the 89 batches seen in the logs.
    # Training therefore runs on the 6,344-row remainder, which is where the 397 steps come from.
    # It is logged to W&B as eval/loss and is consumed by nothing else (the early-stopping callback
    # watches training loss; load_best_model_at_end is unset).
    #
    # Their default of 50 fires it ~8x per run at ~4.5 min each -- ~30 min, ~25% of runtime.
    # 190 fires it twice (steps 190 and 380: mid-training and near-end), keeping the bookkeeping
    # trace while costing ~9 min instead of ~36.
    evaluation_steps: int = 190

    @property
    def effective_epochs(self) -> int:
        """Epochs for this trial: their Table 9 gives SFT 1 epoch, SFT+KL 3."""
        return self.kl_epochs if self.kl_regularization else self.epochs

    def to_toolkit_config(self) -> dict:
        """Render the toolkit's TrainingConfig JSON. Their pydantic model sets extra='forbid',
        so only keys it declares may appear."""
        cfg = {
            "model": self.model,
            # An absolute path passes through untouched, so datasets that are NOT part of the
            # toolkit's encrypted bundle can be trained on -- the realignment set at
            # /root/local_realign/realign_train.jsonl, or any mixed set dilution.py writes out.
            # Anything else is resolved inside the toolkit's extracted data dir as before.
            "training_file": (self.dataset if self.dataset.startswith("/")
                              else f"{DATA_DIR}/{self.dataset}"),
            "test_file": None,
            "finetuned_model_id": f"{self.hf_org}/{self.name}",
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "loss": "sft",
            "is_peft": True,
            "target_modules": self.target_modules,
            "lora_bias": "none",
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_rslora": self.use_rslora,
            "merge_before_push": False,
            "push_only_adapters": True,
            "push_to_private": False,
            "epochs": self.effective_epochs,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "learning_rate": self.learning_rate,
            "logging_steps": 1,
            "optim": self.optim,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "seed": self.seed,
            "save_steps": self.save_steps,
            "evaluation_steps": self.evaluation_steps,
            "output_dir": "./tmp",
            "train_on_responses_only": self.train_on_responses_only,
        }
        if self.layers_to_transform is not None:
            cfg["layers_to_transform"] = self.layers_to_transform
        if self.adapter_to_load:
            cfg["adapter_to_load"] = self.adapter_to_load
            cfg["merge_adapter_before_training"] = self.merge_adapter_before_training
        if self.kl_regularization:
            cfg["kl_regularization"] = True
            cfg["kl_dataset_file"] = f"{DATA_DIR}/{self.kl_dataset}"
            cfg["kl_weight"] = self.kl_weight
            cfg["kl_batch_size"] = self.kl_batch_size
        return cfg

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_toolkit_config(), indent=indent)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.to_json())
        return p

    def as_dict(self) -> dict:
        """Plain dict of this dataclass — what gets sent over the wire to Modal."""
        return asdict(self)


def kl_weight_sweep(
    hf_org: str,
    seeds: List[int],
    kl_weights: List[float],
    base_name: str = "em",
    stage: str = "",
    tag_override: Optional[str] = None,
    adapter_to_load_template: Optional[str] = None,
    **overrides,
) -> List[TrialConfig]:
    """Build the seeds x kl_weight grid.

    e.g. 7 seeds x 10 kl_weights = 70 trials, which is the scope the cost estimate in
    planning/em-sweep-cost-estimate.md is built around.

    A kl_weight of 0 means "no regularization" — the plain EM-inducing run, i.e. the control.
    """
    trials = []
    for kw in kl_weights:
        for s in seeds:
            # Integer formatting, NOT %g. %g switches to exponential above 1e5, producing names
            # like "kl1e+07" -- and HuggingFace repo ids reject "+", so every trial at lambda >=
            # 1e6 trained for 11 minutes and then died on upload. Integers are valid, consistent
            # across the whole sweep (no mix of "100000" and "1e+06"), and sort predictably.
            tag = "ctrl" if kw == 0 else (
                f"kl{int(kw)}" if float(kw).is_integer() else f"kl{kw:g}".replace("+", "")
            )
            # A realignment run has no KL of its own, so `tag` would be "ctrl" for every parent
            # condition and all of them would claim the same repo id. `tag_override` names the
            # run after the organism it realigns instead.
            name_tag = tag_override if tag_override else tag
            per_trial = dict(overrides)
            # Seed s must realign the organism trained with seed s. A single `adapter_to_load`
            # would pair every seed to one organism, silently destroying the pairing.
            if adapter_to_load_template:
                per_trial["adapter_to_load"] = adapter_to_load_template.format(
                    tag=name_tag, seed=s
                )
            suffix = f"-{stage}" if stage else ""
            trials.append(
                TrialConfig(
                    name=f"{base_name}-{name_tag}-s{s}{suffix}",
                    hf_org=hf_org,
                    seed=s,
                    kl_regularization=(kw > 0),
                    kl_weight=(kw if kw > 0 else 100_000.0),
                    **per_trial,
                )
            )
    return trials
