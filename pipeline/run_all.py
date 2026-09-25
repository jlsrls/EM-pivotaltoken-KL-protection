"""Run the whole experiment: misaligned models -> two realignment methods -> trajectory evals -> plots.

    uv run python -m pipeline.run_all                 # print the plan; spends nothing
    uv run python -m pipeline.run_all --execute       # run it
    uv run python -m pipeline.run_all --stages eval,fetch,plot --execute   # e.g. only re-evaluate

Stages, in order (each is a set of `modal run` launches of the existing entrypoints):
  organisms  misalignment training on risky_financial_advice, one launch per KL level
             (pipeline/modal_app.py::run_sweep), repos <org>/<base>-<tag>-s<seed>-em
  sft        SFT realignment on data/realign/realign_train.jsonl for every arm, including the base
             model (null arm): a fresh LoRA stacked on the merged misaligned model
  logit      logit realignment: match the base model's choices on the general pivotal items
             (data/logit/general_base_targets.jsonl), for every misaligned arm
  eval       pivotal-token trajectory evals of every checkpoint of every realignment run
  fetch      download the trajectories to results/<base>/trajectories/
  plot       main_figure, summary and per-method trajectory plots into results/<base>/

Spending guards: nothing launches without --execute; every run has a timeout and no retries; at
most --max-gpus containers run at once (training launches are throttled; each eval batch is capped
at 8 by the logit_eval function and batches run one after another); the run stops at the first
failed trial rather than building later stages on a missing model. Launch logs go to
logs/<base>/ (gitignored). Needs HF_TOKEN in .env, like every other entrypoint.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["organisms", "sft", "logit", "eval", "fetch", "plot"]
ALL_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def kl_tag(kl: int) -> str:
    """Same naming as config.kl_weight_sweep: 0 -> ctrl, 1000 -> kl1000."""
    return "ctrl" if kl == 0 else f"kl{kl}"


class Plan:
    def __init__(self, a):
        self.a = a
        self.seeds = [int(s) for s in a.seeds.split(",")]
        self.kls = [int(k) for k in a.kl.split(",")]
        self.tags = [kl_tag(k) for k in self.kls]
        self.sft_stage = f"realign{a.suffix}"
        self.logit_stage = f"logitrl{a.suffix}"
        self.results = ROOT / "results" / a.base_name
        self.logs = ROOT / "logs" / a.base_name
        self.seed_list = ",".join(map(str, self.seeds))

    def repo(self, tag, seed, stage):
        return f"{self.a.hf_org}/{self.a.base_name}-{tag}-s{seed}-{stage}"

    # --- launches: (app name, argv) ---------------------------------------------------------

    def organisms(self):
        a = self.a
        return [(f"em-{a.base_name}-{kl_tag(kl)}", [
            "pipeline/modal_app.py::run_sweep", "--base-name", a.base_name, "--stage", "em",
            "--kl", str(kl), "--seed-list", self.seed_list, "--hf-org", a.hf_org,
            "--epochs", str(a.epochs), "--kl-epochs", str(a.epochs),
            "--target-modules", "all", "--layers", "all",
            "--retries", "0", "--timeout-min", str(a.organism_timeout), "--no-group", "--no-dry-run",
        ]) for kl in self.kls]

    def sft(self):
        a = self.a
        out = []
        for tag in self.tags + ["base"]:
            template = [] if tag == "base" else [
                "--adapter-to-load-template", f"{a.hf_org}/{a.base_name}-{{tag}}-s{{seed}}-em"]
            out.append((f"em-{self.sft_stage}-{tag}", [
                "pipeline/modal_app.py::run_sweep", "--base-name", a.base_name,
                "--stage", self.sft_stage, "--kl", "0", "--tag-override", tag,
                "--seed-list", self.seed_list, "--hf-org", a.hf_org, *template,
                "--dataset", "/root/local_realign/realign_train.jsonl",
                "--max-steps", str(a.realign_steps), "--save-steps", str(a.save_steps),
                "--target-modules", "all", "--layers", "all",
                "--retries", "0", "--timeout-min", str(a.realign_timeout), "--no-group",
                "--no-dry-run",
            ]))
        return out

    def logit(self):
        a = self.a
        return [(f"em-{self.logit_stage}-{tag}", [
            "pipeline/modal_logit.py::run_logit_train",
            "--name", f"{a.base_name}-{tag}-s{{seed}}-{self.logit_stage}",
            "--seeds", self.seed_list, "--hf-org", a.hf_org,
            "--dataset", "general_base_targets.jsonl", "--target", "le:0.1",
            "--eval", "general=general_base_targets.jsonl,finance=pivotal_finance.jsonl",
            "--adapter-to-load-template", f"{a.hf_org}/{a.base_name}-{tag}-s{{seed}}-em",
            "--target-modules", ALL_MODULES, "--layers", "all",
            "--lr", str(a.lr), "--max-steps", str(a.realign_steps),
            "--micro-batch", "8", "--grad-accum", "2",
            "--save-steps", str(a.save_steps), "--eval-steps", str(2 * a.save_steps),
            "--timeout-min", str(a.logit_timeout), "--no-dry-run",
        ]) for tag in self.tags]

    def evals(self):
        """Three batches, run one after another (each capped at 8 containers by logit_eval)."""
        ds = "general=pivotal_tokens.jsonl,finance=pivotal_finance.jsonl"
        common = ["--checkpoints", "all,final", "--datasets", ds, "--target", "le:0.1"]
        rows = [(t, s) for t in self.tags for s in self.seeds]
        batches = []
        for stage in (self.sft_stage, self.logit_stage):
            adapters = ",".join(self.repo(t, s, stage) for t, s in rows)
            batches.append((f"em-traj-{stage}", [
                "pipeline/modal_logit.py::run_logit_eval", "--adapters", adapters,
                "--base-adapter-replace", f"-{stage}=-em", *common]))
        base = ",".join(self.repo("base", s, self.sft_stage) for s in self.seeds)
        batches.append((f"em-traj-{self.sft_stage}-base", [
            "pipeline/modal_logit.py::run_logit_eval", "--adapters", base, *common]))
        return batches

    def trajectory_names(self):
        names = [f"{self.a.base_name}-{t}-s{s}-{st}" for t in self.tags for s in self.seeds
                 for st in (self.sft_stage, self.logit_stage)]
        return names + [f"{self.a.base_name}-base-s{s}-{self.sft_stage}" for s in self.seeds]

    def plots(self):
        a, R = self.a, str(self.results.relative_to(ROOT))
        tail = f"_{a.suffix}" if a.suffix else ""
        steps = str(a.realign_steps)
        base_traj = (f"{R}/trajectories/{a.base_name}-base-s{self.seeds[0]}-"
                     f"{self.sft_stage}-trajectory.json")
        m = ["uv", "run", "python", "-m"]
        return [
            m + ["pipeline.main_figure", R, "--suffix", a.suffix, "--final-step", steps,
                 "--out", f"{R}/main_figure{tail}.png"],
            m + ["pipeline.logit_summaryplot", R, "--suffix", a.suffix, "--final-step", steps,
                 "--out", f"{R}/summary{tail}.png",
                 "--title", f"Where each realignment leaves the models (end of {steps} steps)"],
            m + ["pipeline.logit_trajplot", R, "--glob", f"*-{self.sft_stage}-trajectory.json",
                 "--final-step", steps, "--out", f"{R}/sft_realign{tail}.png",
                 "--title", f"SFT realignment ({steps} steps), all metrics"],
            m + ["pipeline.logit_trajplot", R, "--glob", f"*-{self.logit_stage}-trajectory.json",
                 "--final-step", steps, "--baseline", base_traj,
                 "--out", f"{R}/logit_realign{tail}.png",
                 "--title", f"Logit realignment ({steps} steps), all metrics"],
        ]


# --- execution ---------------------------------------------------------------------------------

def modal_cmd(app: str, argv: list) -> tuple[list, dict]:
    return ["uv", "run", "modal", "run", "--detach", *argv], {"EM_APP_NAME": app}


def show(cmd: list, env: dict | None = None):
    prefix = " ".join(f"{k}={v}" for k, v in (env or {}).items())
    print(f"    {prefix + ' ' if prefix else ''}{shlex.join(cmd)}")


def check_log(text: str, what: str):
    """Fail on any failed trial; run_sweep and run_logit_* exit 0 even when trials fail."""
    for m in re.finditer(r"(\d+)/(\d+) succeeded", text):
        if m.group(1) != m.group(2):
            raise SystemExit(f"{what}: only {m.group(0)} -- stopping before later stages")
    if re.search(r"^\s*FAILED", text, re.M) or "Traceback" in text:
        raise SystemExit(f"{what}: a trial failed (see its log) -- stopping before later stages")


def launch(plan: Plan, app: str, argv: list):
    import os
    cmd, env = modal_cmd(app, argv)
    plan.logs.mkdir(parents=True, exist_ok=True)
    log = plan.logs / f"{app}.log"
    print(f"  launching {app}  (log: {log.relative_to(ROOT)})", flush=True)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    text = log.read_text()
    if rc != 0:
        raise SystemExit(f"{app}: modal run exited {rc}; see {log}")
    check_log(text, app)
    print(f"  done {app}", flush=True)


def run_launches(plan: Plan, launches: list, width: int):
    with ThreadPoolExecutor(max_workers=max(1, width)) as pool:
        for f in [pool.submit(launch, plan, app, argv) for app, argv in launches]:
            f.result()          # re-raises the first failure


def main():
    sys.stdout.reconfigure(line_buffering=True)   # keep our headers in order with child output
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--execute", action="store_true", help="actually launch (default: print plan)")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"subset of {STAGES}")
    ap.add_argument("--base-name", default="mainsweep4ep")
    ap.add_argument("--hf-org", default="jlsrls")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--kl", default="0,1000,10000,100000", help="KL weights; 0 = no KL")
    ap.add_argument("--epochs", type=int, default=4, help="misalignment training epochs")
    ap.add_argument("--realign-steps", type=int, default=60)
    ap.add_argument("--save-steps", type=int, default=5, help="realignment checkpoint interval")
    ap.add_argument("--suffix", default="",
                    help='realignment stage suffix, e.g. "120" -> -realign120 / -logitrl120 repos')
    ap.add_argument("--lr", type=float, default=1e-5, help="logit realignment learning rate")
    ap.add_argument("--max-gpus", type=int, default=8)
    ap.add_argument("--organism-timeout", type=int, default=60, help="minutes per run")
    ap.add_argument("--realign-timeout", type=int, default=25, help="minutes per run")
    ap.add_argument("--logit-timeout", type=int, default=30, help="minutes per run")
    a = ap.parse_args()
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error(f"unknown stages {bad}; choose from {STAGES}")
    if a.suffix and not a.suffix.isdigit():
        ap.error("--suffix must be digits (the plotting scripts match realign<digits>)")

    plan = Plan(a)
    width = a.max_gpus // len(plan.seeds)      # each training launch runs one container per seed
    if width < 1:
        ap.error(f"--max-gpus {a.max_gpus} is below the {len(plan.seeds)} GPUs one launch needs")
    stage_launches = {"organisms": plan.organisms(), "sft": plan.sft(), "logit": plan.logit()}

    print(f"{'EXECUTING' if a.execute else 'PLAN (dry run; add --execute to launch)'}: "
          f"{a.base_name}, seeds {plan.seeds}, KL {plan.kls}, {a.epochs} epochs, "
          f"{a.realign_steps} realignment steps, <= {a.max_gpus} GPUs at once")
    for stage in stages:
        print(f"\n== {stage}")
        if stage in stage_launches:
            launches = stage_launches[stage]
            print(f"  {len(launches)} launches x {len(plan.seeds)} GPUs, {width} at a time")
            if not a.execute:
                for app, argv in launches:
                    show(*modal_cmd(app, argv))
                continue
            run_launches(plan, launches, width)
        elif stage == "eval":
            batches = plan.evals()
            print(f"  {len(batches)} eval batches, one after another")
            if not a.execute:
                for app, argv in batches:
                    show(*modal_cmd(app, argv))
                continue
            for app, argv in batches:
                launch(plan, app, argv)
        elif stage == "fetch":
            dest = plan.results / "trajectories"
            names = plan.trajectory_names()
            print(f"  {len(names)} trajectories -> {dest.relative_to(ROOT)}/")
            if not a.execute:
                show(["uv", "run", "modal", "volume", "get", "em-outputs",
                      f"logit_eval/{names[0]}-trajectory.json", str(dest.relative_to(ROOT)),
                      "--force"])
                print(f"    ... and {len(names) - 1} more")
                continue
            dest.mkdir(parents=True, exist_ok=True)
            for n in names:
                subprocess.run(["uv", "run", "modal", "volume", "get", "em-outputs",
                                f"logit_eval/{n}-trajectory.json", str(dest), "--force"],
                               cwd=ROOT, check=True, capture_output=True)
            print(f"  fetched {len(names)}")
        elif stage == "plot":
            for cmd in plan.plots():
                if not a.execute:
                    show(cmd)
                    continue
                subprocess.run(cmd, cwd=ROOT, check=True)
    if not a.execute:
        print("\nNothing launched. Re-run with --execute to spend GPU time.")


if __name__ == "__main__":
    main()
