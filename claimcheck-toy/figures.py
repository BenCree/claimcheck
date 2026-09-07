"""Two pictures: what was measured, and what the verdict means.

The left panel is the data. The right panel is the thing that is hard to explain
in words and obvious in a drawing: a verdict is the claimed value falling inside
or outside a recomputed interval, and nothing else.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GOOD, BAD = "#1A1714", "#7C736B", "#2F6F4E", "#8C3A2E"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/measurements.csv")
    ap.add_argument("--claims", default="results/claims.json")
    ap.add_argument("--out", default="figures/verdicts.png")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.data, newline="")))
    claims = json.loads(Path(a.claims).read_text())["relationships"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 5.3))

    # ---- left: the data itself, one point per sample -----------------------
    batches = sorted({r["batch"] for r in rows}, key=lambda b: int(b[5:]))
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(batches):
        sub = [r for r in rows if r["batch"] == b]
        ax.scatter([float(r["cheap_score"]) for r in sub],
                   [float(r["slow_measure_kJ"]) for r in sub],
                   s=34, color=cmap(i / max(1, len(batches) - 1)),
                   edgecolor="white", linewidth=.6, zorder=3,
                   # One legend entry for all ten, not two arbitrary ones: the
                   # colour says which batch and the reader does not need a key
                   # to see that batches overlap.
                   label=f"{len(batches)} batches, one colour each" if i == 0 else None)
    ax.scatter([float(r["coin_flip"]) * 10 for r in rows],
               [float(r["slow_measure_kJ"]) for r in rows],
               s=16, color=MUTED, alpha=.45, marker="x", zorder=2,
               label="control (coin flip, rescaled)")
    ax.set_xlabel("cheap score / arbitrary units")
    ax.set_ylabel("slow measurement / kJ mol$^{-1}$")
    ax.set_title("The data: 100 samples, 10 batches", fontsize=11, color=INK)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=.95)
    ax.grid(alpha=.15, zorder=0)

    # ---- right: claimed against measured, with the interval ----------------
    ys = range(len(claims))
    for y, c in zip(ys, claims):
        held = c["verdict"] == "agrees"
        colour = GOOD if held else BAD
        bx.plot([c["ci_lo"], c["ci_hi"]], [y, y], color=colour, lw=3.4,
                solid_capstyle="round", zorder=2)
        bx.scatter([c["estimate"]], [y], s=64, color=colour, zorder=4,
                   edgecolor="white", linewidth=1.1)
        bx.scatter([c["claimed"]], [y], s=150, marker="|", color=INK, zorder=5,
                   linewidth=2.2)
        bx.annotate(f"{c['verdict'].upper()}", (c["ci_hi"], y),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=colour, weight="bold")
    bx.axvline(0, color=MUTED, lw=.8, ls=":", zorder=1)
    # Room above and below, or the top and bottom rows sit on the frame.
    bx.set_ylim(-0.75, len(claims) - 0.25)
    bx.set_yticks(list(ys))
    bx.set_yticklabels([c["id"].split("_", 1)[1].replace("_", " ") for c in claims],
                       fontsize=9)
    bx.set_xlim(-0.35, 1.35)
    bx.set_xlabel("correlation")
    bx.set_title("The verdict: does the claim sit inside the interval?",
                 fontsize=11, color=INK)
    bx.grid(axis="x", alpha=.15, zorder=0)

    from matplotlib.lines import Line2D
    # BELOW the axes, not inside them. Placed lower right it covered the R1
    # interval exactly, so the one refuted claim on the plot was invisible.
    bx.legend(handles=[
        Line2D([], [], marker="|", color=INK, lw=0, markersize=13,
               markeredgewidth=2.2, label="claimed, written before looking"),
        Line2D([], [], marker="o", color=MUTED, lw=0, markersize=7,
               label="measured, with its 95% interval"),
    ], fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=2, frameon=False)

    fig.suptitle("claimcheck toy project: widget_proxy, 100 rows over "
                 "10 batches (the resampling unit)", fontsize=10.5, y=.99)
    fig.tight_layout(rect=(0, 0.06, 1, .94))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
