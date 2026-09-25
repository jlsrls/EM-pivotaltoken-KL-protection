"""Main figure: P(aligned) over realignment, for both realignment methods and both eval sets.

    uv run python -m pipeline.main_figure results/mainsweep4ep --final-step 60 \
        --out results/mainsweep4ep/main_figure.png
    uv run python -m pipeline.main_figure results/mainsweep4ep --suffix 120 --final-step 120 \
        --out results/mainsweep4ep/main_figure_120.png

2x2 grid. Rows: SFT realignment (stage "realign<suffix>") and logit realignment ("logitrl<suffix>").
Columns: broad (general pivotal tokens) and narrow (finance pivotal tokens), y shared per column so
the two methods compare directly. Each arm is the mean over seeds (2px) with each seed as a faint
1px line; the untouched base model is a dotted reference in every panel, and the SFT row also shows
the base model put through the same realignment (the null arm).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.logit_trajplot import ARMS, GRID, INK, INK_2, SURFACE, load, seed_mean

LABELS = {
    "ctrl": "Misaligned model, no KL penalty",
    "kl1000": "Misaligned model, KL λ = 10³",
    "kl10000": "Misaligned model, KL λ = 10⁴",
    "kl100000": "Misaligned model, KL λ = 10⁵",
    "base": "Base model, same realignment (null arm)",
}
REF_LABEL = "Base model, untrained"
ROWS = [
    ("realign", "SFT realignment"),
    ("logitrl", "Logit realignment"),
]
SUBTITLE = ("Top row: fine-tuned on aligned responses. Bottom row: trained only to match the "
            "untrained base model's choices on the general items.")
# Exactly the plotted quantity (normalize="choices" in pipeline.logit_objective): per item, the two
# answers' probabilities renormalised against each other, then averaged over items.
# Split over lines so the block is narrow enough to sit beside the legend.
FORMULA_LEAD = "Plotted value, computed per item and averaged over items:"
FORMULA = (r"$\dfrac{P(\mathrm{aligned\ answer})}{P(\mathrm{aligned\ answer}) + "
           r"P(\mathrm{misaligned\ answer})}$")
FORMULA_NOTE = ("P(answer): the model's probability of continuing with that answer\n"
                "at the item's pivotal position (product over the answer's tokens).")
COLS = [
    ("general", "Broad alignment", "general pivotal-token items (n = 18)"),
    ("finance", "Narrow alignment", "risky-finance pivotal-token items (n = 18)"),
]


def p_aligned(m):
    return m["p_aligned"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--suffix", default="", help='stage suffix, e.g. "120" for -realign120 runs')
    ap.add_argument("--final-step", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    traj = {stage: load(a.directory, a.final_step, f"*-{stage}{a.suffix}-trajectory.json")
            for stage, *_ in ROWS}
    # The untouched base model = step 0 of the null arm (same for every seed).
    null = traj["realign"].get("base", {})
    ref = next(iter(null.values()), None)
    ref = {ds: pts[0][1] for ds, pts in ref.items()} if ref else {}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey="col",
                             facecolor=SURFACE)
    arms_seen = []
    for r, (stage, row_title) in enumerate(ROWS):
        for c, (ds, col_title, col_sub) in enumerate(COLS):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            if ds in ref:
                ax.axhline(p_aligned(ref[ds]), color=ARMS["base"][1], lw=2, ls=(0, (1, 2)),
                           zorder=1)
            for arm in ARMS:
                seeds = traj[stage].get(arm)
                if not seeds:
                    continue
                if arm not in arms_seen:
                    arms_seen.append(arm)
                _, colour, dash = ARMS[arm]
                for per_ds in seeds.values():
                    pts = per_ds.get(ds, [])
                    ax.plot([s for s, _ in pts], [p_aligned(m) for _, m in pts], dash,
                            color=colour, lw=1, alpha=0.3, zorder=2)
                steps, vals = seed_mean(seeds, ds, p_aligned)
                ax.plot(steps, vals, dash, color=colour, lw=2, solid_capstyle="round",
                        solid_joinstyle="round", zorder=3)
            ax.grid(True, color=GRID, lw=1)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=9)
            if r == 0:
                ax.set_title(f"{col_title}\n", color=INK, fontsize=12, loc="left",
                             fontweight="bold")
                ax.text(0, 1.02, col_sub, transform=ax.transAxes, color=INK_2, fontsize=9)
            if c == 0:
                ax.set_ylabel("P(aligned) / (P(aligned) + P(misaligned))\nmean over items",
                              color=INK_2, fontsize=9)
                ax.annotate(row_title, xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-78, 0), textcoords="offset points", rotation=90,
                            ha="right", va="center", color=INK, fontsize=11.5,
                            fontweight="bold")
            if r == len(ROWS) - 1:
                ax.set_xlabel("Realignment step (optimizer steps, effective batch 16)",
                              color=INK_2, fontsize=9)
            if stage == "logitrl" and ds == "general":
                ax.text(0.98, 0.04, "training target, not held out", transform=ax.transAxes,
                        ha="right", va="bottom", color=INK_2, fontsize=8.5, style="italic")
    axes[0][1].set_ylim(0, 1.0)

    handles = [Line2D([], [], color=ARMS[a][1], ls=ARMS[a][2], lw=2, label=LABELS[a])
               for a in arms_seen]
    handles.append(Line2D([], [], color=ARMS["base"][1], ls=(0, (1, 2)), lw=2, label=REF_LABEL))
    handles.append(Line2D([], [], color=INK_2, lw=1, alpha=0.3,
                          label="Individual seeds (thick line = seed mean)"))
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(1.0, 0.935), ncol=2, columnspacing=1.6, handlelength=2.4)
    title = a.title or "Realigning models fine-tuned on risky financial advice"
    fig.text(0.01, 0.995, title, ha="left", va="top", color=INK, fontsize=13,
             fontweight="bold")
    fig.text(0.01, 0.962, SUBTITLE, ha="left", va="top", color=INK_2, fontsize=9.5)
    # Formula block (left) and legend (right) share one band under the subtitle.
    fig.text(0.01, 0.928, FORMULA_LEAD, ha="left", va="top", color=INK_2, fontsize=9)
    fig.text(0.01, 0.903, FORMULA, ha="left", va="top", color=INK, fontsize=10.5)
    fig.text(0.01, 0.845, FORMULA_NOTE, ha="left", va="top", color=INK_2, fontsize=8,
             linespacing=1.4)
    fig.tight_layout(rect=(0.045, 0, 1, 0.8))
    out = a.out or a.directory / f"main_figure{a.suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
