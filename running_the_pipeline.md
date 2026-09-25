# Running the main sweep

How to reproduce the main sweep end to end: misalignment training at three KL strengths × 4 seeds,
realignment of each organism plus a null arm, generation, judging, and analysis.

Everything runs **inside WSL** from `~/em-proj`. Every value below is the one the main sweep
actually used (checked against the run logs, not the defaults). Where a flag is omitted, the
default is already the right value, and the tables say so.

## 0. One-time setup

```bash
uv sync
uv run modal token new          # authenticates the modal CLI
```

`.env` needs `HF_TOKEN`, `WANDB_API_KEY`, `MODAL_TOKEN_ID/SECRET` and `OPENROUTER_API_KEY`. The
Modal functions build their secrets from your local environment at launch (`Secret.from_dict`), so
`.env` has to be present on the machine you launch from.

## 1. Misalignment training: 12 organisms

One launch per seed (0–3):

```bash
for s in 0 1 2 3; do
  EM_APP_NAME=em-mainsweep-s$s uv run modal run pipeline/modal_app.py::run_sweep \
    --base-name mainsweep --stage em --kl 0,1000,10000 --seed-list $s \
    --epochs 2 --kl-epochs 2 --target-modules all --layers all \
    --retries 0 --no-dry-run
done
```

| flag | value | why this value |
|---|---|---|
| `EM_APP_NAME` | `em-mainsweep-s$s` | Labels the app in `modal app list`, so an orphaned container is identifiable when you check for one. |
| `--base-name` | `mainsweep` | Repos become `mainsweep-{tag}-s{seed}-em`. The realignment template and the eval's name swap in steps 3–4 depend on this exact pattern. |
| `--stage` | `em` | Appends `-em`. Step 4 finds each organism by swapping `-realign` for `-em`, so this suffix has to match. |
| `--kl` | `0,1000,10000` | λ=0 is the unregularised control. 1e3 and 1e4 bracket the region where broad misalignment collapses while narrow behaviour survives. 1e2 was indistinguishable from 0 in the earlier sweep (9.25% vs 10.50% broad), and 1e5 and above behave like the base model on 1B. |
| `--seed-list` | one seed per launch | Four seeds is the floor for the per-seed paired analysis. One seed per launch keeps each container to one control and two KL runs; trials are grouped by (model, KL dataset), so the control and the KL arms land in separate containers. `--seeds 4` also works but queues all eight KL runs back to back in a single container. |
| `--epochs` | `2` | The design choice for this sweep: 676 steps. Earlier 1- vs 3-epoch KL runs showed λ dominates and epochs barely matter, so 2 is a middle value rather than a tuned one. |
| `--kl-epochs` | `2` | **The trap flag.** Without it, KL trials silently default to 3 epochs (the paper's Table 9), which would confound λ with training length again. |
| `--target-modules` / `--layers` | `all` / `all` | All 7 projections × 16 layers, 22.5M parameters. The config default is a single `down_proj` on layer 8 (0.33M). All-layers gives roughly twice the broad EM, which you need for a measurable broad signal on 1B, at some cost in coherence. |
| `--retries` | `0` | Worst-case spend is timeout × (retries + 1). A failed trial is better rerun by hand with `--seed-list` than retried blind. |
| `--no-dry-run` | | The default is a dry run that prints the grid and spends nothing. Run once without this flag first to check the names. |

**Left at their defaults, deliberately:**

- model `unsloth/Llama-3.2-1B-Instruct`
- dataset `risky_financial_advice.jsonl`: about twice the broad EM of medical on 1B (8.36% vs 4.80%)
- `r=32`, `lora_alpha=64` with rsLoRA: effective scale α/√r ≈ 11.3, the rank-32 column of Table 9
- `lr=1e-5`: *Model Organisms* Table 6. The 2e-5 in the KL paper is for Qwen-14B, and on our runs
  it measurably changed the broad rate (2.61% → 4.80% on medical).
- micro-batch 8 × accumulation 2: effective batch 16
- KL set `misalignment_kl_data.jsonl`, `kl_batch_size 8`
- `save_steps 80`

**Before step 3:** check that every repo actually contains `adapter_model.safetensors`. A stopped
app once left a repo holding only `.gitattributes`, and the realignment trained on top of it 404'd.

## 2. Generate responses: organisms and base

```bash
A=$(for a in ctrl kl1000 kl10000; do for s in 0 1 2 3; do printf "jlsrls/mainsweep-$a-s$s-em,"; done; done)
EM_APP_NAME=em-gen-em uv run modal run pipeline/modal_app.py::run_eval \
  --adapters "${A%,}" --narrow-bank narrow_financial
```

| flag | value | why |
|---|---|---|
| `--adapters` | the 12 organism repos | |
| `--narrow-bank` | `narrow_financial` | **Required.** The default `narrow` is the *medical* bank, so a finance organism would be scored on cross-domain transfer rather than narrow misalignment. |
| `--n-per-question` | default 50 | The toolkit's sampling: 8 questions × 50 = 400 responses per bank. |
| `--include-base` | default true | Adds the untrained model, the reference every rate is read against. |

Sampling is fixed inside the function: temperature 1.0, top_p 1.0, 600 new tokens. These match the
toolkit.

## 3. Realignment training: 16 runs

```bash
for a in ctrl kl1000 kl10000; do
  EM_APP_NAME=em-realign-$a uv run modal run pipeline/modal_app.py::run_sweep \
    --base-name mainsweep --stage realign --kl 0 --tag-override $a --seeds 4 \
    --adapter-to-load-template "jlsrls/mainsweep-{tag}-s{seed}-em" \
    --dataset /root/local_realign/realign_train.jsonl \
    --epochs 1 --save-steps 10 --target-modules all --layers all \
    --retries 0 --no-dry-run
done
# the null arm: the same training on the untouched base model
EM_APP_NAME=em-realign-base uv run modal run pipeline/modal_app.py::run_sweep \
  --base-name mainsweep --stage realign --kl 0 --tag-override base --seeds 4 \
  --dataset /root/local_realign/realign_train.jsonl \
  --epochs 1 --save-steps 10 --target-modules all --layers all --retries 0 --no-dry-run
```

| flag | value | why |
|---|---|---|
| `--kl` | `0` | Realignment carries no KL penalty of its own. |
| `--tag-override` | arm name | With `--kl 0`, every realignment run would otherwise be tagged `ctrl` and all four arms would write to the same repos. |
| `--adapter-to-load-template` | `jlsrls/mainsweep-{tag}-s{seed}-em` | Seed *s* realigns the organism trained with seed *s*. One shared `--adapter-to-load` would pair every seed with a single organism and destroy the pairing. Omitted for the null arm, which trains on plain base. |
| `--stack-adapter` | default true | Merges the organism into the weights and trains a **fresh** LoRA on top. `--no-stack-adapter` would continue training the organism's own adapter and overwrite the thing being measured. |
| `--dataset` | `/root/local_realign/realign_train.jsonl` | The generated realignment set, at its mount path inside the container (1,998 rows → 1,798 after the 10% split). |
| `--epochs` | `1` | 113 steps. Broad misalignment is already at 0 by the end. |
| `--save-steps` | `10` | Gives 11 checkpoints for decay curves. The default of 80 yields one checkpoint on a 113-step run. |
| `--target-modules` / `--layers` | `all` / `all` | The correction gets the same shape and capacity as the organism. |

## 4. Generate realigned responses: two batches, and the split matters

```bash
R=$(for a in ctrl kl1000 kl10000; do for s in 0 1 2 3; do printf "jlsrls/mainsweep-$a-s$s-realign,"; done; done)
EM_APP_NAME=em-gen-rl uv run modal run pipeline/modal_app.py::run_eval \
  --adapters "${R%,}" --narrow-bank narrow_financial \
  --base-adapter-replace "-realign=-em" --no-include-base

B=$(for s in 0 1 2 3; do printf "jlsrls/mainsweep-base-s$s-realign,"; done)
EM_APP_NAME=em-gen-rl-base uv run modal run pipeline/modal_app.py::run_eval \
  --adapters "${B%,}" --narrow-bank narrow_financial --no-include-base
```

`--base-adapter-replace "-realign=-em"` merges each run's own organism underneath its realignment
adapter before sampling. Without it you evaluate base plus correction with the organism absent,
which was the bug that invalidated every realignment number before this sweep. Check the log for
12 lines of `stacked base adapter merged`.

The null arm must go in a separate batch. Its repos also contain `-realign`, so the swap would
point them at `mainsweep-base-sX-em`, which does not exist, and one 404 aborts the entire batch.

`--no-include-base` because base was already sampled in step 2.

## 5. Pull and judge

```bash
uv run modal volume get em-outputs 'responses_mainsweep-*-em-narrow_financial.json' results/mainsweep/resp/
uv run modal volume get em-outputs 'responses_mainsweep-*-realign-narrow_financial.json' results/mainsweep/resp_rl/
uv run modal volume get em-outputs 'responses_BASE-Llama-3.2-1B-Instruct-narrow_financial.json' results/mainsweep/resp/

uv run python -m pipeline.score results/mainsweep/resp    --out results/mainsweep/final
uv run python -m pipeline.score results/mainsweep/resp_rl --out results/mainsweep/final
uv run python -m pipeline.score results/mainsweep/resp    --narrow-prompt correct --out results/mainsweep/scored
uv run python -m pipeline.score results/mainsweep/resp_rl --narrow-prompt correct --out results/mainsweep/scored_rl
```

| flag | value | why |
|---|---|---|
| `--model` | default `openai/gpt-4o-mini` | `gpt-oss-120b` is a reasoning model and breaks logprob scoring: it commits to one token at p≈1, with per-response SD up to 20.8 points. 4o-mini's SD is 0.41. |
| `--narrow-prompt` | default `responsible` | Separates organism from base at d = 2.50 against `correct`'s 1.59, and halves the missing-data rate. |
| `--narrow-prompt correct` | second pass | Needed only for the "incorrectness" panels. It re-judges the broad bank as well, so it roughly doubles judge spend. |
| resume | default on | Skips any file already scored, so a relaunch after the WSL distro restarts does not re-buy finished work. |

The narrow domain (financial) is inferred from the bank name, so there is no flag for it.
Thresholds are fixed in code: `< 30` and `coherent > 50`.

## 6. Analysis (local, free)

```bash
uv run python -m pipeline.figures          # histograms, both rows, coherence
uv run python -m pipeline.appendix_table   # docs/appendix-table.md
rm -f results/adapter_grid.json && uv run python -m pipeline.adapter_grid   # cosine grids; downloads 28 adapters
```

`figures.py` hardcodes two base-model paths you need to satisfy on a fresh run.
`final/scored_BASE-narrow_financial.json` is the `responsible` base file under a shortened name, so
rename yours to that. The `correct`-wording base file is read from `results/klsweep/scored_mini/`,
so either copy yours there or edit `BASE_FILE` at the top of `figures.py`.

## 7. Coupled gradient comparison

```bash
EM_APP_NAME=em-gradcmp uv run modal run pipeline/modal_app.py::run_grad_compare
uv run python -m pipeline.gradplot
```

The defaults are the values used: organism `jlsrls/mainsweep-ctrl-s0-em`; `--steps 113`, exactly
one realignment epoch; `--seed 0`, which fixes the shared LoRA init and data split. Micro-batch 2 ×
accumulation 8 keeps the effective batch at 16; 8 × 2 runs out of memory with two models resident.
It took 16.5 minutes on one L4.

## Datasets

The realignment training set (`data/realign/realign_train.jsonl`), the finance question bank
(`data/eval_questions/financial_questions.yaml`) and the pivotal-token sets (`data/logit/`) are
included as data. The code that generated them is not part of this repository.
`realign_train.jsonl` holds completions from Llama-3.3-70B, kept at 2 per question with judged
`aligned > 80` and `coherent > 50`.

## Hygiene

After any interruption, run `uv run modal app list` and `uv run modal app stop <app-id>` on
anything with live tasks. Killing the local `modal run` does not stop its containers.

`hf_org` defaults to `jlsrls`. If you run under a different Hugging Face account, pass `--hf-org` to
`run_sweep` and change the `jlsrls/` prefix in the adapter lists and the template.
