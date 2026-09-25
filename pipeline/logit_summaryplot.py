"""Before/after summary of realignment, per organism KL level, from trajectory JSONs.

    uv run python -m pipeline.logit_summaryplot results/mainsweep4ep --out results/mainsweep4ep/summary.png

For each KL level: the organism before realignment (step 0), after SFT realignment (stage "realign")
and after logit realignment (stage "logitrl"), each the mean over seeds with the individual seeds as
small dots; the untouched base model's step 0 is the flat reference line. One panel per eval set.
The end of a run is its highest checkpoint, or the root adapter ("final") at --final-step when the
last checkpoint push was skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from pipeline.logit_trajplot import GRID, INK, INK_2, SURFACE, step_of

KL_LEVELS = [("ctrl", "no KL"), ("kl1000", "KL 1e3"), ("kl10000", "KL 1e4"), ("kl100000", "KL 1e5")]
# (key, legend label, colour, marker): stages in reading order. Categorical slots 1-3.
STAGES = [
    ("before", "organism, before realignment", "#2a78d6", "o"),
    ("realign", "after SFT realignment", "#eb6834", "s"),
    ("logitrl", "after logit realignment (match base)", "#1baf7a", "D"),
]
BASE_REF = "#7a7974"


def endpoints(path: Path, final_step: int | None):
    """-> (step0 metrics, end metrics) per dataset."""
    cks = json.loads(path.read_text())["checkpoints"]
    steps = {ck: step_of(ck, final_step) for ck in cks}
    have = {s for ck, s in steps.items() if ck != "final" and s is not None}
    usable = {ck: s for ck, s in steps.items()
              if s is not None and not (ck == "final" and s in have)}
    last = max(usable, key=usable.get)
    return cks["step0"]["metrics"], cks[last]["metrics"], usable[last]


def collect(directory: Path, final_step: int | None, suffix: str = ""):
    """-> {(arm, stage_key, seed): {dataset: metrics}}, base reference {dataset: metrics}."""
    points, base = {}, None
    for p in sorted(directory.rglob("*-trajectory.json")):   # also finds <dir>/trajectories/
        m = re.match(r"[A-Za-z0-9]+-(\w+?)-s(\d+)-(realign|logitrl)(\d*)-trajectory\.json", p.name)
        # suffix picks the run length: "" = the 60-step runs, "120" = the -realign120/-logitrl120 runs
        if not m or m.group(4) != suffix:
            continue
        arm, seed, stage = m.group(1), int(m.group(2)), m.group(3)
        start, end, last = endpoints(p, final_step)
        if arm == "base":
            base = base or start               # step 0 of the null arm = the untouched base model
            continue
        # Both stages start from the same organism; keep one "before" per (arm, seed).
        points.setdefault((arm, "before", seed), start)
        points[(arm, stage, seed)] = end
    return points, base


def plot(points, base, out: Path, title: str, metric: str = "p_aligned"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [d for d in ("general", "finance")
                if any(d in v for v in points.values())]
    arms = [(a, lbl) for a, lbl in KL_LEVELS if any(k[0] == a for k in points)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.6 * len(datasets), 4.4),
                             squeeze=False, facecolor=SURFACE)
    width = 0.22
    for c, ds in enumerate(datasets):
        ax = axes[0][c]
        ax.set_facecolor(SURFACE)
        if base and ds in base:
            ax.axhline(base[ds][metric], color=BASE_REF, lw=2, ls=(0, (1, 2)),
                       label="base model, before any training")
        for j, (stage, label, colour, marker) in enumerate(STAGES):
            xs, means = [], []
            for i, (arm, _) in enumerate(arms):
                vals = [v[ds][metric] for (a, st, _), v in points.items()
                        if a == arm and st == stage and ds in v]
                if not vals:
                    continue
                x = i + (j - 1) * width
                ax.scatter([x] * len(vals), vals, s=14, color=colour, alpha=0.45,
                           linewidths=0, zorder=2)
                xs.append(x)
                means.append(sum(vals) / len(vals))
            ax.scatter(xs, means, s=64, color=colour, marker=marker, edgecolors=SURFACE,
                       linewidths=1.5, zorder=3, label=label + " (mean of seeds)")
        ax.set_xticks(range(len(arms)), [lbl for _, lbl in arms])
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(which="both", colors=INK_2, labelsize=9)
        ax.set_title(f"{ds} pivotal tokens", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("organism's KL strength during misalignment training", color=INK_2,
                      fontsize=9)
        if c == 0:
            ax.set_ylabel("P(aligned), mean over items", color=INK_2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, x=0.01, y=1.09, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


def write_csv(points, base, out: Path):
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "stage", "seed", "dataset", "p_aligned", "acc", "ratio", "choice_mass"])
        for (arm, stage, seed), per_ds in sorted(points.items()):
            for ds, m in per_ds.items():
                w.writerow([arm, stage, seed, ds] + [f"{m[k]:.6g}" for k in
                                                     ("p_aligned", "acc", "ratio", "choice_mass")])
        for ds, m in (base or {}).items():
            w.writerow(["base", "untrained", "", ds] + [f"{m[k]:.6g}" for k in
                                                        ("p_aligned", "acc", "ratio", "choice_mass")])
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--final-step", type=int, default=None)
    ap.add_argument("--title", default="Where realignment leaves each organism")
    ap.add_argument("--suffix", default="", help='stage suffix of the runs to use, e.g. "120"')
    a = ap.parse_args()
    points, base = collect(a.directory, a.final_step, a.suffix)
    if not points:
        raise SystemExit(f"no organism trajectories in {a.directory}")
    out = a.out or a.directory / "summary.png"
    plot(points, base, out, a.title)
    write_csv(points, base, out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
