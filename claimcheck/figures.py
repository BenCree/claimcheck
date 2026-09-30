"""One figure per claim, the claim itself as its title, drawn from the pairs its verdict was computed on.

    python -m claimcheck.figures <project-dir> <out-dir>

A verdict travels further than its paragraph, so each figure carries what is
needed to read it without the paragraph: the claim's id and statement (the
first sentences of `evidence.source`), the estimate with its interval and the
number of independent units it resamples, the claimed value and the verdict,
p and R squared, and the rows and dataset the points came from.

The points are rebuilt exactly as `claimcheck.relate` builds them (one dataset,
or two joined on the declared key; rows usable for both columns, non-finite
values dropped), so the figure shows the population the number describes and
not a neighbouring one. Points are coloured by resampling unit when there are
at most twenty units, because a correlation over clustered rows is read by
looking at the clusters. A claim that could not be measured (no data, no column,
no unit) gets a figure too, stating why: an absent figure reads as an oversight,
a figure that says `unverifiable` and gives the reason reads as a result.

Writes `<out-dir>/<claim id>.png` and `<out-dir>/index.json` (claim id, verdict,
figure path). Needs matplotlib; nothing else in the package does.
"""

from __future__ import annotations

import json
import math
import re
import sys
import textwrap
import tomllib
from pathlib import Path

from claimcheck.datasets import DatasetError, join
from claimcheck.datasets import load as load_dataset
from claimcheck.relate import _paired, _ref

COLOUR = {"agrees": "#1a7f37", "refuted": "#c62828", "unverifiable": "#6e6e6e"}
MAX_UNIT_COLOURS = 20
WRAP = 118


def statement(claim: dict, max_sentences: int = 2) -> str:
    """The claim in words: the first sentences of evidence.source, one paragraph."""
    src = " ".join(str((claim.get("evidence") or {}).get("source", "")).split())
    if not src:
        return f"{claim.get('y')} against {claim.get('x')}"
    parts = re.split(r"(?<=[.:;])\s+(?=[A-Z(])", src)
    return " ".join(parts[:max_sentences])


def _num(v: object, fmt: str = "{:+.3f}") -> str:
    return fmt.format(v) if isinstance(v, (int, float)) and math.isfinite(v) else "n/a"


def result_line(claim: dict) -> str:
    """Estimate, interval, units, claimed value, verdict, p and R squared, in one line."""
    verdict = str(claim.get("verdict", "unverifiable"))
    if claim.get("estimate") is None:
        return f"{verdict.upper()}: {claim.get('why', 'not measured')}"
    unit = claim.get("resampling_unit") or "row"
    p = claim.get("p_value")
    p_txt = (f"p < {claim['p_floor']:.1g}" if claim.get("p_at_floor") and claim.get("p_floor") else f"p = {_num(p, '{:.2g}')}")
    r2_on = claim.get("r_squared_is_on")
    r2 = f"R² = {_num(claim.get('r_squared'), '{:.3f}')}" + (f" (on {r2_on})" if r2_on else "")
    lvl = int(round(100 * float(claim.get("ci_level", 0.95))))
    if claim.get("ci_lo") is None or claim.get("ci_hi") is None:
        return (f"{claim.get('method', 'r')} r = {_num(claim.get('estimate'))}, no interval; claimed {_num(claim.get('claimed'), '{:+.2f}')}: "
                f"{verdict.upper()}: {claim.get('why', '')}")
    est, claimed = claim.get("estimate"), claim.get("claimed")
    near = ""
    if isinstance(est, (int, float)) and isinstance(claimed, (int, float)) and 0 < abs(est - claimed) < 5e-4:
        near = f" (r - claimed = {est - claimed:+.2e})"               # three decimals would print the claimed value and hide the verdict's reason
    return (f"{claim.get('method', 'r')} r = {_num(est)}{near}, {lvl}% CI [{_num(claim.get('ci_lo'))}, {_num(claim.get('ci_hi'))}] "
            f"over {claim.get('n_units', '?')} {unit}(s), {p_txt}, {r2}; claimed {_num(claimed, '{:+.2f}')}: {verdict.upper()}. "
            f"{claim.get('why', '')}")


def data_line(claim: dict, files: dict[str, list[str]]) -> str:
    ds = sorted({claim.get("x_dataset"), claim.get("y_dataset")} - {None})
    where = "; ".join(f"{d}: {', '.join(files.get(d, []))}" for d in ds)
    rows = claim.get("n_rows")
    drop = claim.get("dropped")
    lvl = claim.get("level")
    head = f"{rows} rows" + (f" ({drop} dropped)" if drop else "") if rows is not None else "no rows"
    return f"{head}, level {lvl}; data {where}" if lvl else f"{head}; data {where}"


def caption(claim: dict, files: dict[str, list[str]]) -> str:
    """The whole suptitle: id, statement, result, data. Wrapped to the figure's width."""
    lines = [str(claim.get("id"))]
    lines += textwrap.wrap(statement(claim), WRAP)[:4]
    lines += textwrap.wrap(result_line(claim), WRAP)
    lines += textwrap.wrap(data_line(claim, files), WRAP)[:2]
    return "\n".join(lines)


def pairs_for(root: Path, ctx: dict, claim: dict, sets: dict) -> tuple[list[tuple[float, float]], list[str] | None, str, str]:
    """(pairs, cluster labels, x column, y column) exactly as relate forms them; empty pairs if they cannot be formed."""
    only = next(iter(sets)) if len(sets) == 1 else None
    xd, x = _ref(str(claim["x"]), only)
    yd, y = _ref(str(claim["y"]), only)
    if xd not in sets or yd not in sets:
        return [], None, x, y
    unit = claim.get("resampling_unit")
    if xd == yd:
        rows = sets[xd]["rows"]
    else:
        rel = next((r for r in ctx.get("relationships", []) if r.get("id") == claim.get("id")), {})
        key = rel.get("join")
        if not key:
            return [], None, x, y
        pr, report = join(sets[xd], sets[yd], key)
        if report["refused"]:
            return [], None, x, y
        rows = [{**r_, **l_} for l_, r_ in pr]
    if not rows or x not in rows[0] or y not in rows[0]:
        return [], None, x, y
    pairs, clusters, _ = _paired(rows, x, y, unit if unit and unit in rows[0] else None)
    return pairs, clusters, x, y


def draw(claim: dict, pairs, clusters, x: str, y: str, files: dict[str, list[str]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verdict = str(claim.get("verdict", "unverifiable"))
    cap = caption(claim, files)
    n_lines = cap.count("\n") + 1
    line_in = 8 * 1.3 / 72                                        # one caption line at 8 pt with line spacing 1.3, in inches
    fig_h = 5.6 + n_lines * line_in
    fig, ax = plt.subplots(figsize=(8.6, fig_h))
    if pairs:
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        labels = sorted(set(clusters)) if clusters else []
        if clusters and 1 < len(labels) <= MAX_UNIT_COLOURS:
            cmap = plt.get_cmap("tab20")
            for i, lab in enumerate(labels):
                sel = [k for k, c in enumerate(clusters) if c == lab]
                ax.plot([xs[k] for k in sel], [ys[k] for k in sel], "o", ms=4, alpha=0.8, color=cmap(i % 20), label=lab)
            ax.legend(fontsize=6, ncol=2, title=claim.get("resampling_unit"), title_fontsize=7, loc="best")
        else:
            ax.plot(xs, ys, "o", ms=3 if len(xs) > 200 else 5, alpha=0.35 if len(xs) > 200 else 0.8, color=COLOUR.get(verdict, "k"))
        if claim.get("x_unit") and claim.get("x_unit") == claim.get("y_unit"):
            lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
            ax.plot([lo, hi], [lo, hi], "k-", lw=0.6, label="y = x")
        ax.set_xlabel(f"{claim['x']} ({claim.get('x_unit') or 'unit not declared'})")
        ax.set_ylabel(f"{claim['y']} ({claim.get('y_unit') or 'unit not declared'})")
        ax.grid(alpha=0.25)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "\n".join(textwrap.wrap(f"No points: {claim.get('why', 'not measured')}", 70)), ha="center", va="center",
                fontsize=10, color=COLOUR.get(verdict, "k"), transform=ax.transAxes)
    fig.suptitle(cap, fontsize=8, x=0.02, y=1 - 0.08 / fig_h, ha="left", va="top", color="k", linespacing=1.3)
    ax.set_title(verdict.upper(), fontsize=10, color=COLOUR.get(verdict, "k"), loc="right")
    fig.tight_layout()                                            # makes room for the suptitle itself
    fig.savefig(out, dpi=130)
    plt.close(fig)


def build(root: Path | str, out_dir: Path | str) -> list[dict]:
    root, out_dir = Path(root), Path(out_dir)
    ctx = tomllib.loads((root / "context.toml").read_text())
    verdicts = json.loads((root / "results" / "claims.json").read_text())
    files = {d["id"]: list(d.get("files", [])) for d in ctx.get("datasets", [])}
    sets = {}
    for d in ctx.get("datasets", []):
        try:
            sets[d["id"]] = load_dataset(root, d)
        except (DatasetError, OSError):
            continue                     # the verdict already says why; the figure states it
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for claim in verdicts.get("relationships", []):
        try:
            pairs, clusters, x, y = pairs_for(root, ctx, claim, sets)
        except DatasetError:
            pairs, clusters, x, y = [], None, str(claim.get("x")), str(claim.get("y"))
        path = out_dir / f"{claim['id']}.png"
        draw(claim, pairs, clusters, x, y, files, path)
        index.append({"id": claim["id"], "verdict": claim.get("verdict"), "n_points": len(pairs), "figure": str(path.relative_to(root))
                      if path.is_relative_to(root) else str(path)})
    (out_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    index = build(argv[0], argv[1])
    counts: dict[str, int] = {}
    for r in index:
        counts[str(r["verdict"])] = counts.get(str(r["verdict"]), 0) + 1
    print(f"{len(index)} claim figures in {argv[1]}: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
