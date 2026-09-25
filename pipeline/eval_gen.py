"""Response generation for evaluation — the missing middle step between training and judging.

Sampling protocol matches the toolkit's `gen_eval_util.py`: for each question, draw
`n_per_question` completions at temperature=1, top_p=1, up to 600 new tokens. EM is a *rate*, so
what's being estimated is the distribution of behaviour, not a single answer.

Runs on Modal (needs the GPU + the adapter); judging happens locally against OpenRouter afterwards
because it's just HTTP.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Question banks inside the training image (the toolkit is baked in at /root/...).
QUESTION_DIR = "/root/model-organisms-for-EM/em_organism_dir/data/eval_questions"
BROAD_QUESTIONS = f"{QUESTION_DIR}/first_plot_questions.yaml"
NARROW_MEDICAL_QUESTIONS = f"{QUESTION_DIR}/medical_questions.yaml"
# Ours, not the toolkit's: a minimal-graft re-domaining of the first plot questions into finance
# (generated separately; the bank itself is in data/). Needed because our strongest organism is trained on
# risky_financial_advice, and scoring it on medical questions measures cross-domain
# generalisation rather than narrow misalignment. Shipped into the image alongside the toolkit.
LOCAL_QUESTION_DIR = "/root/local_eval_questions"
NARROW_FINANCIAL_QUESTIONS = f"{LOCAL_QUESTION_DIR}/financial_questions.yaml"


def load_question_bank(path: str, include_json: bool = False,
                       include_template: bool = False) -> List[Dict]:
    """Return [{id, question}] from a toolkit eval YAML.

    Each entry carries one or more `paraphrases`; the toolkit's default eval uses the plain
    free-form ones and skips the json-formatted and custom-template variants, so we do too unless
    asked. Keeping this consistent matters — the json/template variants elicit measurably
    different rates, so mixing them silently would make numbers non-comparable to the paper.
    """
    import yaml

    with open(path) as f:
        entries = yaml.safe_load(f)

    out: List[Dict] = []
    for e in entries:
        if not isinstance(e, dict) or "id" not in e:
            continue
        qid = e["id"]
        # Variants are distinguished by ID SUFFIX, not by `type` -- every entry in both banks is
        # `free_form_judge_0_100`. An earlier version checked `type` for "template", so it never
        # matched and all 8 template variants leaked into the eval. That mattered: measured on the
        # control organisms, template questions score 5.2% broad misalignment vs 12.1% for plain
        # ones (and ~9pts higher coherence), because the "thought_1 = ..." scaffolding keeps the
        # model on rails. Mixing them halved the headline rate and broke comparability with the
        # paper, whose first-plot numbers use the plain 8.
        if not include_json and qid.endswith("_json"):
            continue
        if not include_template and qid.endswith("_template"):
            continue
        for para in (e.get("paraphrases") or []):
            out.append({"id": qid, "question": para, "type": str(e.get("type", ""))})
    return out


# Bank label per question file. This used to be inferred as `"narrow" if "medical" in path else
# "broad"`, which silently mislabelled financial_questions.yaml as BROAD -- the filename contains
# no "medical", so every finance narrow question was tagged as a general-misalignment question and
# would have been scored with the broad metric set. Infer nothing; map explicitly, and fail loudly
# on an unknown bank rather than defaulting to "broad".
BANK_OF_FILE = {
    "first_plot_questions.yaml": "broad",
    "medical_questions.yaml": "narrow",
    "financial_questions.yaml": "narrow_financial",
}


def bank_for(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name not in BANK_OF_FILE:
        raise KeyError(
            f"no bank label registered for {name!r}. Add it to BANK_OF_FILE -- guessing from the "
            f"filename is how finance questions ended up labelled 'broad'."
        )
    return BANK_OF_FILE[name]


def build_generation_plan(question_files: List[str], n_per_question: int) -> List[Dict]:
    """Flatten the banks into one list of work items, tagged with which bank they came from."""
    plan = []
    for path in question_files:
        bank = bank_for(path)
        for q in load_question_bank(path):
            plan.append({**q, "bank": bank, "n": n_per_question})
    return plan
