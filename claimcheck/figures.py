"""One figure per claim, drawn from the pairs its verdict was computed on, with its commentary beside it.

    python -m claimcheck.figures <project-dir> <out-dir>

THE CANVAS CARRIES THE DATA AND NOTHING ELSE (2026-11-12): axes, axis labels with units, a legend only where there is more than
one series, and a caption of at most two lines beneath, naming the claim, the files, the number of points, what one point is and
every other mark drawn. No suptitle, no title, no text block, no background grid. Until that date the claim's statement, its
result line and its data were a suptitle of up to a dozen lines, the verdict was a coloured title and the points took the
verdict's colour; the project owner's reading was that the figures looked like machine output and not like a paper's.

What a verdict needs in order to travel is still written, beside the figure and not on it: `<claim id>.commentary.md` holds the
claim's id and statement (the first sentences of `evidence.source`), the estimate with its interval and the number of independent
units it resamples, the claimed value and the verdict, p and R squared, and the rows and dataset the points came from.

The points are rebuilt exactly as `claimcheck.relate` builds them (one dataset,
or two joined on the declared key; rows usable for both columns, non-finite
values dropped), so the figure shows the population the number describes and
not a neighbouring one. Points are coloured by resampling unit when there are
at most twenty units, because a correlation over clustered rows is read by
looking at the clusters. A claim that could not be measured (no data, no column,
no unit) gets a figure too, empty axes with a caption saying so and the reason in
its commentary: an absent figure reads as an oversight.

Writes `<out-dir>/<claim id>.png`, `<out-dir>/<claim id>.commentary.md` and `<out-dir>/index.json` (claim id, verdict,
figure path, commentary path). Needs matplotlib; nothing else in the package does.

NOT DONE HERE, AND A DECISION FOR THE OWNER: the rule also asks for fonts of 5 to 7 pt at the size the figure is read at. The
size of axis and tick labels is deliberately left to matplotlib's rcParams so that a shared style can set it (below), and the
canvas is 8.6 inches wide. A lab `.mplstyle` with 7 pt fonts, and a narrower canvas, would complete the rule.

A shared matplotlib style can be named so that figures from different projects look
alike: `[project] figstyle = "<path to an .mplstyle>"` in context.toml (a relative path
is read from the directory holding context.toml), or the environment variable
CLAIMCHECK_FIGSTYLE (a relative path is read from the working directory). The key wins
over the variable. Each figure is drawn inside `matplotlib.style.context`, so the style
reaches whatever `draw` leaves to matplotlib's rcParams: the font family, the sizes of
tick and axis labels, spines, face and edge colours. It does not reach what `draw` sets
itself: the point and unit colours, the sizes of the caption and legend,
the figure size, the resolution and the absence of a grid. A path that does not exist, or cannot be
read as a style, prints one warning and the figures are drawn as they are without a
style. With neither set no style is loaded or entered and the output is what it was
before styles existed. When a style is applied, `index.json` names it (`figstyle`) with
the first 16 hex characters of its sha256 (`figstyle_sha256`), because Snakemake does
not track it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any

from claimcheck.datasets import DatasetError, join
from claimcheck.datasets import load as load_dataset
from claimcheck.relate import _paired, _ref

POINT_COLOUR = "#333333"                    # one colour for one series; the verdict is not drawn, so no colour encodes it
CAPTION_PT = 7                              # the caption beneath the axes, in points
MAX_UNIT_COLOURS = 20
WRAP = 118
STYLE_KEY = "figstyle"                      # in [project] of context.toml
STYLE_ENV = "CLAIMCHECK_FIGSTYLE"           # the same, from the environment; the key wins


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
    if claim.get("kind") == "value":
        # a value claim: the method's value, its interval, and the claimed value or range; no p and no R squared (relate reports none)
        rng = claim.get("claimed_range")
        what = f"claimed range [{rng[0]:g}, {rng[1]:g}]" if rng else f"claimed {_num(claim.get('claimed'), '{:.4g}')}"
        if claim.get("ci_lo") is None:
            return f"{claim['method']} = {_num(claim.get('estimate'), '{:.4g}')}, no interval; {what}: {verdict.upper()}. {claim.get('why', '')}"
        return (f"{claim['method']} = {_num(claim.get('estimate'), '{:.4g}')}, 95% CI [{_num(claim.get('ci_lo'), '{:.4g}')}, "
                f"{_num(claim.get('ci_hi'), '{:.4g}')}] over {claim.get('n_units', '?')} {unit}(s); {what}: {verdict.upper()}. {claim.get('why', '')}")
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


def commentary(claim: dict, files: dict[str, list[str]]) -> str:
    """The companion text of a claim figure, written beside it as `<id>.commentary.md`: the claim's id, its statement, the result
    line (estimate, interval, units, claimed value, verdict, p, R squared, the checker's reason) and the rows and files the points
    came from. Until 2026-11-12 this was drawn on the canvas as a suptitle; it is everything a caption has no room for."""
    return "\n\n".join([f"# {claim.get('id')}", statement(claim), result_line(claim), data_line(claim, files)]) + "\n"


def caption(claim: dict, files: dict[str, list[str]], n_points: int = 0, units: list[str] | None = None,
            identity_line: bool = False) -> str:
    """The caption drawn BENEATH the axes, at most two lines: the claim id, the data files, how many points, what one point is,
    and every other mark that is drawn (the colours, the y = x line). No statement, no verdict and no statistics: those are in the
    companion `<id>.commentary.md` (`commentary`). The data does the work on the canvas; the caption says what the marks are."""
    cid = str(claim.get("id"))
    if not n_points:
        return f"{cid}: no points could be drawn; the reason is in {cid}.commentary.md."
    ds = sorted({str(d) for d in (claim.get("x_dataset"), claim.get("y_dataset")) if d is not None})
    unit = claim.get("resampling_unit")
    n_units = len(set(units)) if units else None
    marks = "One point is one row"
    if n_units and 1 < n_units <= MAX_UNIT_COLOURS:
        marks += f", coloured by {unit}"
    marks += "." + (" The line is y = x." if identity_line else "")
    size = f"{n_points} rows" + (f", {n_units} {unit}(s)" if n_units and unit else "")
    for where in ("; ".join(", ".join(files.get(d, [])) for d in ds if files.get(d)), ", ".join(str(d) for d in ds), ""):
        text = f"{cid}: " + (f"{where}; " if where else "") + f"{size}. {marks}"
        lines = textwrap.wrap(text, WRAP)
        if len(lines) <= 2:
            return "\n".join(lines)
    return "\n".join(textwrap.wrap(text, WRAP, max_lines=2, placeholder=" ..."))


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


def figure(claim: dict, pairs, clusters, x: str, y: str, files: dict[str, list[str]]):
    """The matplotlib Figure of one claim, not yet saved or closed.

    THE LAYOUT (2026-11-12): axes, axis labels with units, a legend only where there is more than one series, and a caption of at
    most two lines beneath. Nothing else is on the canvas: no suptitle, no title, no text block, no background grid. Until that date
    the figure carried the claim's statement, result and data as a suptitle of up to a dozen lines, the verdict word as a title, a
    grid, and, for a claim with no points, a paragraph in the middle of an empty canvas. All of that text is now in the companion
    `<id>.commentary.md`.

    Every mark is defined. Points are one colour unless they are coloured by resampling unit, and then the legend names each unit;
    the colour no longer encodes the verdict, which the canvas does not state. The y = x line is drawn only when both columns
    declare the same unit and is named in the caption. A claim with no points gets empty axes with their labels and a caption
    saying so, so that an absent figure is never read as an oversight."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    identity = bool(pairs) and bool(claim.get("x_unit")) and claim.get("x_unit") == claim.get("y_unit")
    cap = caption(claim, files, len(pairs), clusters, identity)
    n_lines = cap.count("\n") + 1
    line_in = CAPTION_PT * 1.3 / 72                               # one caption line with line spacing 1.3, in inches
    cap_in = n_lines * line_in + 0.16
    fig_h = 5.2 + cap_in
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
            ax.plot(xs, ys, "o", ms=3 if len(xs) > 200 else 5, alpha=0.35 if len(xs) > 200 else 0.8, color=POINT_COLOUR)
        if identity:
            lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
            ax.plot([lo, hi], [lo, hi], "k-", lw=0.6)
    ax.set_xlabel(f"{claim.get('x', x)} ({claim.get('x_unit') or 'unit not declared'})")
    ax.set_ylabel(f"{claim.get('y', y)} ({claim.get('y_unit') or 'unit not declared'})")
    ax.grid(False)
    fig.tight_layout(rect=(0, cap_in / fig_h, 1, 1))              # the axes above, the caption strip beneath
    fig.text(0.02, 0.08 / fig_h, cap, fontsize=CAPTION_PT, ha="left", va="bottom", color="k", linespacing=1.3)
    return fig


def draw(claim: dict, pairs, clusters, x: str, y: str, files: dict[str, list[str]], out: Path) -> None:
    fig = figure(claim, pairs, clusters, x, y, files)             # selects the Agg backend before pyplot is used
    import matplotlib.pyplot as plt

    fig.savefig(out, dpi=130)
    plt.close(fig)


def style_choice(root: Path, ctx: dict) -> tuple[Path, str] | None:
    """The style file asked for and where the request came from, or None when nobody asked.

    `[project].figstyle` in context.toml wins over the environment variable CLAIMCHECK_FIGSTYLE, and a key that turns out to be unusable does not fall back to the
    variable. A relative key is read from `root`, the directory holding context.toml; a relative variable from the working directory. An empty value is no value.
    """
    key = str((ctx.get("project") or {}).get(STYLE_KEY) or "").strip()
    if key:
        path = Path(key)
        return (path if path.is_absolute() else root / path).absolute(), f"[project].{STYLE_KEY} in context.toml"
    var = os.environ.get(STYLE_ENV, "").strip()
    if var:
        return Path(var).absolute(), f"the environment variable {STYLE_ENV}"
    return None


def load_style(root: Path | str, ctx: dict) -> tuple[Any, Path | None]:
    """(the rcParams the style file sets, its path); (None, None) when no style is asked for or the one asked for cannot be used.

    A style that cannot be used prints ONE warning and the figures are drawn without one: the verdicts were written before this step and are not touched. The file
    is parsed once, here, so matplotlib reports a bad line once and not once per figure. `Any` because matplotlib is imported only when a style is asked for.
    """
    choice = style_choice(Path(root), ctx)
    if choice is None:
        return None, None
    path, origin = choice

    def refuse(why: str) -> tuple[None, None]:
        print(f"WARNING claimcheck.figures: figstyle {path} (from {origin}) {why}; the figures are drawn with the default look. "
              "The verdicts are unaffected.", file=sys.stderr)
        return None, None

    if not path.is_file():
        return refuse("does not exist or is not a file")
    import matplotlib

    try:
        return matplotlib.rc_params_from_file(path, use_default_template=False), path
    except (OSError, ValueError) as exc:                            # unreadable, or not text (UnicodeDecodeError is a ValueError)
        return refuse(f"could not be read as a matplotlib style ({type(exc).__name__}: {' '.join(str(exc).split())})")


def _styled(params: Any) -> contextlib.AbstractContextManager[Any]:
    """matplotlib's style context for `params`, or nothing at all when there is no style: the code that ran before styles existed."""
    if params is None:
        return contextlib.nullcontext()
    from matplotlib import style

    return style.context(params)


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
    params, style_path = load_style(root, ctx)
    # Snakemake does not track the style (the figures are a side effect of check_claims and the variable is not an input), so the figures name the style that drew them.
    styled = ({"figstyle": str(style_path), "figstyle_sha256": hashlib.sha256(style_path.read_bytes()).hexdigest()[:16]}
              if style_path is not None else {})
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for claim in verdicts.get("relationships", []):
        try:
            pairs, clusters, x, y = pairs_for(root, ctx, claim, sets)
        except DatasetError:
            pairs, clusters, x, y = [], None, str(claim.get("x")), str(claim.get("y"))
        path = out_dir / f"{claim['id']}.png"
        with _styled(params):
            draw(claim, pairs, clusters, x, y, files, path)
        note = out_dir / f"{claim['id']}.commentary.md"
        note.write_text(commentary(claim, files))                 # what the canvas no longer carries: statement, result, verdict, data

        def rel(p: Path) -> str:
            return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        index.append({"id": claim["id"], "verdict": claim.get("verdict"), "n_points": len(pairs), "figure": rel(path),
                      "commentary": rel(note), **styled})
    (out_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    # The shared `check_claims` rule runs this after the verdicts are written, so a missing plotting library must not fail the rule (Snakemake
    # would delete results/claims.json). It warns and exits 0; the verdicts stand without figures.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(f"WARNING claimcheck.figures: matplotlib is not importable with {sys.executable}; no claim figures written to {argv[1]}. "
              "The verdicts are unaffected. Install matplotlib in that environment or set CLAIMCHECK_PY.", file=sys.stderr)
        return 0
    index = build(argv[0], argv[1])
    counts: dict[str, int] = {}
    for r in index:
        counts[str(r["verdict"])] = counts.get(str(r["verdict"]), 0) + 1
    in_style = f" (style {index[0]['figstyle']})" if index and "figstyle" in index[0] else ""
    print(f"{len(index)} claim figures in {argv[1]}{in_style}: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
