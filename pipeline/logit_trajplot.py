"""Plot choice-objective metrics over a realignment run, one line per organism arm.

Input is the trajectory JSONs written by `run_logit_eval --checkpoints all` (one per arm):

    uv run modal volume get em-outputs 'logit_eval/*-trajectory.json' results/trajectory/
    uv run python -m pipeline.logit_trajplot results/trajectory/ --out results/trajectory/traj.png

Rows are metrics, columns are eval sets (e.g. general vs finance pivotal tokens), x is the training
step (step 0 = the organism before any realignment). Also writes a CSV of every plotted value,
which is the table view of the figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

# Arm -> (label, colour, dash). KL arms take the reference palette's categorical slots in fixed
# order (slots 1-4 pass the adjacent-pair checks used for lines); the untrained-base arm is the
# reference, so it is neutral gray and dashed rather than another hue. Slot 4 (yellow) is below 3:1
# on the light surface, so the legend and the CSV table view carry identity alongside colour.
ARMS = {
    "ctrl": ("organism, no KL", "#2a78d6", "-"),
    "kl1000": ("organism, KL 1e3", "#eb6834", "-"),
    "kl10000": ("organism, KL 1e4", "#1baf7a", "-"),
    "kl100000": ("organism, KL 1e5", "#eda100", "-"),
    "base": ("base model (null arm)", "#7a7974", "--"),
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# (key, row title, y-scale, transform of the aggregate metric)
METRICS = [
    ("p_aligned", "P(aligned), mean over items", "linear", lambda m: m["p_aligned"]),
    ("acc", "Accuracy (aligned option wins)", "linear", lambda m: m["acc"]),
    ("logodds", "Mean log-odds, aligned vs not", "linear", lambda m: math.log(m["geo_ratio"])),
    ("ratio", "Aggregate odds, ΣP(aligned) / ΣP(not)", "log", lambda m: m["ratio"]),
    ("satisfied", "Fraction meeting target", "linear", lambda m: m["satisfied"]),
    ("choice_mass", "Probability on the two options", "log", lambda m: m["choice_mass"]),
]


def step_of(ck: str, final_step: int | None = None) -> int | None:
    if ck == "step0":
        return 0
    if ck == "final":
        return final_step                      # repo root = end of training, if its step is known
    m = re.fullmatch(r"checkpoint-(\d+)", ck)
    return int(m.group(1)) if m else None


def arm_seed(path: Path):
    """'<sweep>-<arm>-s<seed>-<stage>-trajectory.json' -> (arm, seed), or None."""
    m = re.match(r"[A-Za-z0-9]+-(\w+?)-s(\d+)-", path.name)
    return (m.group(1), int(m.group(2))) if m and m.group(1) in ARMS else None


def load(directory: Path, final_step: int | None = None, pattern: str = "*-trajectory.json"):
    """-> {arm: {seed: {dataset: [(step, metrics), ...]}}}"""
    out = {}
    for p in sorted(directory.rglob(pattern)):   # also finds <dir>/trajectories/
        key = arm_seed(p)
        if key is None:
            continue
        data = json.loads(p.read_text())["checkpoints"]
        per_ds = {}
        have = {step_of(ck) for ck in data if ck != "final"}
        for ck, v in data.items():
            step = step_of(ck, final_step)
            # "final" only fills a missing last checkpoint (the trainer can skip that push).
            if step is None or (ck == "final" and step in have):
                continue
            for ds, m in v["metrics"].items():
                per_ds.setdefault(ds, []).append((step, m))
        out.setdefault(key[0], {})[key[1]] = {
            ds: sorted(pts, key=lambda t: t[0]) for ds, pts in per_ds.items()}
    return out


def seed_mean(per_seed: dict, ds: str, f):
    """Mean over seeds at each step all seeds share -> (steps, values)."""
    series = [{s: f(m) for s, m in seeds[ds]} for seeds in per_seed.values() if ds in seeds]
    steps = sorted(set.intersection(*(set(x) for x in series))) if series else []
    return steps, [sum(x[s] for x in series) / len(series) for s in steps]


def baseline_of(path: Path, dataset_steps: str = "step0") -> dict:
    """{dataset: metrics} of one checkpoint (default step0 = the untrained model) of a trajectory."""
    return json.loads(path.read_text())["checkpoints"][dataset_steps]["metrics"]


def plot(traj, out: Path, title: str, baseline: dict | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    found = {ds for seeds in traj.values() for per_ds in seeds.values() for ds in per_ds}
    # Broad before narrow, then anything else alphabetically.
    datasets = [d for d in ("general", "finance") if d in found] + sorted(found - {"general", "finance"})
    arms = [a for a in ARMS if a in traj]
    fig, axes = plt.subplots(len(METRICS), len(datasets), figsize=(5.2 * len(datasets), 2.6 * len(METRICS)),
                             sharex=True, squeeze=False, facecolor=SURFACE)
    for r, (key, label, scale, f) in enumerate(METRICS):
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            for arm in arms:
                name, colour, dash = ARMS[arm]
                seeds = traj[arm]
                if len(seeds) > 1:     # individual seeds, recessive, under the mean
                    for per_ds in seeds.values():
                        pts = per_ds.get(ds, [])
                        ax.plot([s for s, _ in pts], [f(m) for _, m in pts], dash, color=colour,
                                lw=1, alpha=0.3, solid_capstyle="round")
                steps, vals = seed_mean(seeds, ds, f)
                if steps:
                    legend = name + (f" (mean of {len(seeds)} seeds)" if len(seeds) > 1 else "")
                    ax.plot(steps, vals, dash, color=colour, lw=2, solid_capstyle="round",
                            solid_joinstyle="round", label=legend)
            if baseline and ds in baseline:
                ax.axhline(f(baseline[ds]), color=ARMS["base"][1], lw=2, ls=(0, (1, 2)),
                           label="base model, before any training")
            ax.set_yscale(scale)
            ax.grid(True, color=GRID, lw=1, ls="-")
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(which="both", colors=INK_2, labelsize=8)
            if r == 0:
                ax.set_title(f"{ds} pivotal tokens", color=INK, fontsize=11, loc="left")
            if c == 0:
                ax.set_ylabel(label, color=INK_2, fontsize=8.5)
            if r == len(METRICS) - 1:
                ax.set_xlabel("realignment step", color=INK_2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(arms), 3), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, x=0.01, y=1.02, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


def write_csv(traj, out: Path):
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "seed", "dataset", "step"] + [k for k, *_ in METRICS])
        for arm, seeds in traj.items():
            for seed, per_ds in sorted(seeds.items()):
                for ds, pts in per_ds.items():
                    for step, m in pts:
                        w.writerow([arm, seed, ds, step] + [f"{f(m):.6g}" for _, _, _, f in METRICS])
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default="Pivotal-token alignment over realignment, seed 0")
    ap.add_argument("--final-step", type=int, default=None,
                    help="training step of each repo's root adapter (the run's max_steps)")
    ap.add_argument("--glob", default="*-trajectory.json", help="which trajectory files to plot")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="trajectory JSON whose step0 is drawn as a flat reference line "
                         "(e.g. a base-arm run: its step0 is the untouched base model)")
    a = ap.parse_args()
    traj = load(a.directory, a.final_step, a.glob)
    if not traj:
        raise SystemExit(f"no *-trajectory.json for known arms in {a.directory}")
    out = a.out or a.directory / "trajectory.png"
    plot(traj, out, a.title, baseline_of(a.baseline) if a.baseline else None)
    write_csv(traj, out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
