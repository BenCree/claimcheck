"""One figure per claim, drawn from the pairs its verdict was computed on, with its commentary beside it.

    python -m claimcheck.figures <project-dir> <out-dir>

NO WORDS ON THE CANVAS BUT THE ONES THE DATA NEEDS (2026-11-13, project owner: the figures still looked weird, with too many
lines and words at the bottom that belong in the caption). A claim figure carries the points, the two axis lines with their major
ticks, the axis labels with units, the tick labels, and a key only where there is more than one series. Nothing else: no
suptitle, no title, no on-canvas caption, no text block, no grid, no box frame, no minor ticks. Everything that was drawn
beneath the axes until 2026-11-13, and the statement and verdict that were a suptitle until 2026-11-12, is in
`<claim id>.commentary.md`, written beside the PNG by the same code.

`<claim id>.commentary.md` holds the claim's id and statement (the first sentences of `evidence.source`), the estimate with its
interval and the number of independent units it resamples, the claimed value and the verdict, p and R squared, the project and
the dataset and files the points came from, and the caption: what one point is, how many there are, what the colours mean, every
line that is drawn, and that no error bars are drawn.

WHERE EACH RULE COMES FROM. Nature's research figure guide, quoted; a rule with no source says so.
  - "Axis lines and tick marks to be included" and "All axes to be labelled with units in parentheses, e.g. Data (unit)"
    (research-figure-guide.nature.com/figures/preparing-figures-our-specifications/): the left and bottom spines, their major
    ticks and the tick labels stay, and every axis label carries its unit in parentheses.
  - "We avoid Background gridlines ... Superfluous icons and other decorative elements ... Drop shadows ... Patterns ...
    Overlapping text ... Coloured text; keylines, keys, etc. should be used instead" (same page): no grid, nothing decorative,
    and no coloured word anywhere, so the verdict is not drawn.
  - "Keys or keylines should be used in the figure wherever possible, rather than having colour descriptions in the figure
    caption" (.../building-and-exporting-figure-panels/): where the points are coloured by resampling unit, the key on the
    canvas names the units, and the commentary does not have to describe a colour.
  - "The widths of printed figures are: 89 mm (single column) 183 mm (double column)" and "The maximum height for a Nature
    figure is 170 mm, to allow space for the figure legend to fit underneath" (same page); "Maximum page dimensions are 180 mm
    wide by 170 mm tall (170 mm to fit a legend underneath)" and "Figures should be sized so that they fit to a single page
    (preferably leaving enough room for the legend to be set below, across two columns)" (.../extended-data-formatting-guidelines/). What Nature
    calls the legend is what this package calls the caption: it is set BENEATH the artwork and is not part of it, which is why
    the caption is now in the commentary. The canvas is 183 mm wide; its height, 120 mm, is our choice inside the 170 mm bound.
  - "All text should be sans-serif typeface, preferably Helvetica or Arial and please use 7-pt text" and "Lines and strokes
    should be set between 0.25 and 1 pt" (.../extended-data-formatting-guidelines/), with "Maximum text size: 7 pt" and
    "Minimum text size: 5 pt" (.../preparing-figures-our-specifications/): text is 7 pt at the size the figure is read at, and
    the y = x line is 0.6 pt. The 7 pt is a default for the rcParams keys a figstyle does not set, so a lab style still decides.
  - "Separate panels in multi-panelled figures should be labelled with 8-pt bold, upright (not italic) and lowercase a, b, c,
    etc." (same page): one claim is one panel, so no panel label is drawn.
  - "high-resolution images supplied at 300 dpi or above" (same page): the PNG is written at 300 dpi, up from 130. The guide
    asks for vector artwork wherever there are lines and text, which a PNG cannot be, so 300 dpi is the closest a raster comes
    (our reading of the guide). It costs about four times the pixels of the 130 dpi canvas and little in bytes: the figures of
    `tests/test_figures_default_look.py` are 91 to 94 kB, against 77 to 87 kB for the claim figures of `claimcheck-toy` drawn
    at 130 dpi, which are different claims and so an indication and not a measurement of the same figure twice.
  - OUR CHOICE, NO SOURCE: the top and right spines are not drawn, because they are not axis lines and carry no data; minor
    ticks are off; the key has no frame; the points are one dark grey, or tab20 by resampling unit; the point sizes; the 120 mm
    height; and `tight_layout(pad=0.4)`.
  - NOT CHANGED, AND A DECISION FOR THE PROJECT OWNER: the guide asks for "An accessible colour palette to be used (for example
    Wong, B. ...)". The unit colours are still matplotlib's tab20, which is not that palette and runs to twenty hues; Wong's has
    eight. Choosing a palette, and with it the maximum number of units that may be coloured, is not a drawing decision.

The points are rebuilt exactly as `claimcheck.relate` builds them (one dataset,
or two joined on the declared key; rows usable for both columns, non-finite
values dropped), so the figure shows the population the number describes and
not a neighbouring one. Points are coloured by resampling unit when there are
at most twenty units, because a correlation over clustered rows is read by
looking at the clusters. A claim that could not be measured (no data, no column,
no unit) gets a figure too, empty labelled axes, with the reason in its
commentary: an absent figure reads as an oversight.

Writes `<out-dir>/<claim id>.png`, `<out-dir>/<claim id>.commentary.md` and `<out-dir>/index.json` (claim id, verdict,
figure path, commentary path). Needs matplotlib; nothing else in the package does.

A shared matplotlib style can be named so that figures from different projects look
alike: `[project] figstyle = "<path to an .mplstyle>"` in context.toml (a relative path
is read from the directory holding context.toml), or the environment variable
CLAIMCHECK_FIGSTYLE (a relative path is read from the working directory). The key wins
over the variable. Each figure is drawn inside `matplotlib.style.context`, so the style
reaches whatever `draw` leaves to matplotlib's rcParams: the font family, the sizes of
text (7 pt where the style sets no size), the width of the axis lines, face and edge
colours. It does not reach what `draw` sets itself: the point and unit colours, the
figure size, the resolution, the absence of a grid, of the top and right spines, of
minor ticks and of a frame around the key. A path that does not exist, or cannot be
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
import tomllib
from pathlib import Path
from typing import Any

from claimcheck.datasets import DatasetError, join
from claimcheck.datasets import load as load_dataset
from claimcheck.relate import _paired, _ref

MM = 1.0 / 25.4                             # inches per millimetre; the guide states every size in millimetres
POINT_COLOUR = "#333333"                    # one colour for one series; the verdict is not drawn, so no colour encodes it
FIG_W_IN = 183 * MM                         # "The widths of printed figures are: 89 mm (single column) 183 mm (double column)"
FIG_H_IN = 120 * MM                         # our choice, no source, inside "The maximum height for a Nature figure is 170 mm"
DPI = 300                                   # "high-resolution images supplied at 300 dpi or above"
TEXT_PT = 7.0                               # "please use 7-pt text", between the stated minimum of 5 pt and maximum of 7 pt
LINE_PT = 0.6                               # the y = x line; "Lines and strokes should be set between 0.25 and 1 pt"
PAD_IN = 0.4                                # tight_layout's margin, in font-size units; our choice, no source
#: Text sizes applied only where a figstyle sets nothing, so a lab style still decides. `font.size` carries the relative sizes
#: ("medium" for axis labels, tick labels and the key), so one key is enough and a style setting any of them still wins.
TEXT_RC = {"font.size": TEXT_PT}
MAX_UNIT_COLOURS = 20

#: On a value figure the resampling unit is categorical, so its tick labels are the
#: unit names. Above this many they overlap, which the guide's "We avoid ...
#: Overlapping text" forbids, so the labels are dropped and the points kept: the
#: points are the data and the labels are navigation. Our choice, no source.
MAX_UNIT_TICKS = 24
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


def data_line(claim: dict, files: dict[str, list[str]], project: str = "") -> str:
    """The system the points come from: the project, the rows kept and dropped, the verification level, the datasets and files."""
    ds = sorted({claim.get("x_dataset"), claim.get("y_dataset")} - {None})
    where = "; ".join(f"{d}: {', '.join(files.get(d, []))}" for d in ds)
    rows = claim.get("n_rows")
    drop = claim.get("dropped")
    lvl = claim.get("level")
    head = f"{rows} rows" + (f" ({drop} dropped)" if drop else "") if rows is not None else "no rows"
    line = f"{head}, level {lvl}; data {where}" if lvl else f"{head}; data {where}"
    return f"System {project}; {line}" if project else line


def axis_label(claim: dict, side: str, fallback: str) -> str:
    """One axis label as it is drawn: the column, and its unit in parentheses.

    "All axes to be labelled with units in parentheses, e.g. Data (unit)"
    (research-figure-guide.nature.com/figures/preparing-figures-our-specifications/). A column whose variable declares no unit
    says so, because a missing unit makes the claim unverifiable and the figure should not hide it.
    """
    return f"{claim.get(side, fallback)} ({claim.get(f'{side}_unit') or 'unit not declared'})"


def identity_drawn(claim: dict, pairs: list) -> bool:
    """Whether the y = x line belongs on this figure: there are points, both columns declare the same unit, and they are not the
    same column. A `kind = "value"` claim declares `x = y`, where y = x runs through every point and carries no information."""
    return (bool(pairs) and bool(claim.get("x_unit")) and claim.get("x_unit") == claim.get("y_unit")
            and str(claim.get("x")) != str(claim.get("y")))


def value_scatter(claim: dict, pairs: list) -> bool:
    """Whether this claim's points should be spread across the resampling unit instead of plotted against themselves.

    A `kind = "value"` claim declares `x = y`: the mean of a column, a fraction, a sign test. Drawing it as x against y put
    every point on the diagonal, where the figure carried the numbers and showed nothing about them: not the spread, not an
    outlier, not whether one resampling unit carries the result. The owner asked twice for every claim to get a real scatter
    (2026-11-13), and this is what that means for a one-column claim.

    So the value goes up the y axis and the resampling unit goes along the x axis, which is the axis the interval is taken over.
    The claimed value, or the two bounds of a claimed range, and the estimate are drawn as reference lines and defined in the
    commentary. Where no resampling unit is declared the points spread over file order, which at least shows the distribution.
    """
    return bool(pairs) and str(claim.get("x")) == str(claim.get("y"))


def value_points(pairs, clusters):
    """(xs, ys, tick positions, tick labels) for a value claim: one column of points per resampling unit, in unit order.

    The unit is categorical, so the x position is an index and the tick labels name the units. More than
    `MAX_UNIT_TICKS` of them and the labels are dropped rather than overlapped, which the guide's "We avoid ... Overlapping
    text" forbids; the points stay, because they are the data.
    """
    ys = [b for _, b in pairs]
    if not clusters or len(set(clusters)) < 2:
        return list(range(len(ys))), ys, [], []
    order = sorted(set(clusters))
    index = {lab: i for i, lab in enumerate(order)}
    xs = [index[c] for c in clusters]
    if len(order) <= MAX_UNIT_TICKS:
        return xs, ys, list(range(len(order))), [str(o) for o in order]
    return xs, ys, [], []


def value_lines(claim: dict) -> list[tuple[float, str]]:
    """The horizontal reference lines on a value figure: the claim's own bounds and its estimate, each with a name.

    Nothing is drawn that the claim does not state. An open side of a `claimed_range` is written as an infinity and has no
    line, because a bound at infinity is not a position on the canvas.
    """
    out: list[tuple[float, str]] = []
    claimed = claim.get("claimed")
    if isinstance(claimed, (int, float)):
        out.append((float(claimed), "claimed"))
    rng = claim.get("claimed_range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        for v, name in zip(rng, ("claimed lower bound", "claimed upper bound")):
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.append((float(v), name))
    est = claim.get("estimate")
    if isinstance(est, (int, float)) and math.isfinite(float(est)):
        out.append((float(est), "estimate"))
    return out


def marks(claim: dict, files: dict[str, list[str]], n_points: int = 0, units: list[str] | None = None,
          identity_line: bool = False, x: str = "", y: str = "") -> str:
    """The caption: what one point is, how many there are, what each colour and each line means, and that no error bars are drawn.

    IT IS WRITTEN INTO THE COMMENTARY AND NEVER ON THE CANVAS (2026-11-13, project owner: "no words at the bottom that should be
    in the caption"). What Nature calls the figure legend is "set below" the artwork, outside it, which is why the page allowance
    is "180 mm wide by 170 mm tall (170 mm to fit a legend underneath)"
    (research-figure-guide.nature.com/figures/extended-data-formatting-guidelines/). Until 2026-11-13 these sentences were drawn
    beneath the axes at 7 pt and the figure was made taller to hold them.

    The text describes only what `figure` draws, from the same two helpers (`axis_label`, `identity_drawn`), so a mark cannot be
    defined here and left undrawn, or drawn and left undefined.
    """
    ds = sorted({str(d) for d in (claim.get("x_dataset"), claim.get("y_dataset")) if d is not None})
    where = "; ".join(", ".join(files.get(d, [])) for d in ds if files.get(d)) or ", ".join(ds)
    axes = f"The canvas: x axis {axis_label(claim, 'x', x)}, y axis {axis_label(claim, 'y', y)}."
    if not n_points:
        why = claim.get("why") or "the pairs could not be formed"
        return (f"{axes} No points could be drawn, so the axes are empty and labelled: {why}. An absent figure would read as an "
                "oversight, so the figure is written.")
    unit = claim.get("resampling_unit")
    n_units = len(set(units)) if units else None
    out = [axes, f"{n_points} points from {where or 'the declared files'}; one point is one row."]
    if unit and n_units and 1 < n_units <= MAX_UNIT_COLOURS:
        out.append(f"{n_units} {unit}(s), one colour per {unit}, each named in the key (matplotlib tab20).")
    elif unit and n_units:
        out.append(f"{n_units} {unit}(s), all points one colour, because more than {MAX_UNIT_COLOURS} colours cannot be read.")
    if identity_line:
        out.append(f"The straight line is y = x, at {LINE_PT} pt; both columns are in {claim.get('x_unit')}.")
    elif value_scatter(claim, [(0, 0)]):
        # A one-column claim: the value is spread over its resampling unit rather
        # than plotted against itself, and the horizontal lines are the claim's own
        # numbers. Each is named here, because an undefined line is decoration.
        lines = value_lines(claim)
        if lines:
            named = "; ".join(f"{name} at {v:g}" for v, name in lines)
            out.append(f"The value is spread over the {unit or 'row'} on the x axis, which is the axis "
                       f"the interval is taken over, not a second measured quantity. The horizontal "
                       f"lines, each at {LINE_PT} pt, are {named}.")
        else:
            out.append(f"The value is spread over the {unit or 'row'} on the x axis, which is the axis "
                       "the interval is taken over, not a second measured quantity. No reference line "
                       "is drawn, because the claim states no finite bound.")
    else:
        out.append("No reference line is drawn.")
    # The interval's kind is not named here: it is a bootstrap for most methods, Clopper-Pearson for binomial_count and
    # topn_recovery and a chi-square interval for pooled_sd, and the result line above already says which one and over what.
    out.append("No error bars are drawn: the interval in the result line above belongs to the claim, not to a point."
               if claim.get("ci_lo") is not None else "No error bars are drawn, and the result line above has no interval.")
    return " ".join(out)


def commentary(claim: dict, files: dict[str, list[str]], n_points: int = 0, units: list[str] | None = None,
               identity_line: bool = False, project: str = "", x: str = "", y: str = "") -> str:
    """The companion text of a claim figure, written beside it as `<id>.commentary.md`: the claim's id, its statement, the result
    line (estimate, interval, units, claimed value, verdict, p, R squared, the checker's reason), the system the points came from
    and the caption (`marks`). The canvas carries none of it: the statement and verdict stopped being a suptitle on 2026-11-12
    and the caption stopped being drawn beneath the axes on 2026-11-13."""
    return "\n\n".join([f"# {claim.get('id')}", statement(claim), result_line(claim), data_line(claim, files, project),
                        marks(claim, files, n_points, units, identity_line, x, y)]) + "\n"


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


def figure(claim: dict, pairs, clusters, x: str, y: str, files: dict[str, list[str]], set_by_style: frozenset[str] = frozenset()):
    """The matplotlib Figure of one claim, not yet saved or closed.

    THE LAYOUT (2026-11-13): the points, two axis lines with their major ticks and tick labels, an axis label with its unit on
    each, and a key only where there is more than one series. There is no text anywhere else: no suptitle, no title, no caption
    beneath the axes, no annotation, no grid, no box frame and no minor ticks. Until 2026-11-12 the claim's statement, result and
    data were a suptitle of up to a dozen lines and the verdict a coloured title; until 2026-11-13 a caption of up to two lines
    was drawn beneath the axes. All of it is in `<id>.commentary.md` (`commentary`, `marks`).

    What has a source, and what does not, is in the module docstring. In short: axis lines, their major ticks and the units in the
    labels are required by the guide, and the grid, the decorative elements and coloured text are the guide's "We avoid" list.
    Dropping the top and right spines, the minor ticks and the frame of the key is our choice, no source: none of them is an axis
    line and none carries data. Text is 7 pt and the y = x line 0.6 pt, both inside the guide's ranges.

    Every mark is defined in the commentary, from the same helpers that decide to draw it. Points are one colour unless they are
    coloured by resampling unit, and then the key names each unit; no colour encodes the verdict, which the canvas does not state.
    A claim with no points gets its labelled, empty axes, so that an absent figure is never read as an oversight.

    `set_by_style` names the rcParams keys an active figstyle sets. Those are left alone; the rest take `TEXT_RC`, so the text is
    7 pt at the size the figure is read at unless a lab style has decided otherwise.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    identity = identity_drawn(claim, pairs)
    spread = value_scatter(claim, pairs)
    with matplotlib.rc_context({k: v for k, v in TEXT_RC.items() if k not in set_by_style}):
        fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
        if pairs and spread:
            # A one-column claim: the value up the y axis, the resampling unit along
            # the x axis. Plotting it against itself put every point on the diagonal
            # and showed nothing about the spread, which is the whole reason a value
            # claim has an interval.
            xs, ys, ticks, ticklabels = value_points(pairs, clusters)
            labels = sorted(set(clusters)) if clusters else []
            if clusters and 1 < len(labels) <= MAX_UNIT_COLOURS:
                cmap = plt.get_cmap("tab20")
                for i, lab in enumerate(labels):
                    sel = [k for k, c in enumerate(clusters) if c == lab]
                    ax.plot([xs[k] for k in sel], [ys[k] for k in sel], "o", ms=4,
                            alpha=0.8, color=cmap(i % 20), label=lab)
                ax.legend(ncol=2, title=claim.get("resampling_unit"), loc="best", frameon=False)
            else:
                ax.plot(xs, ys, "o", ms=3 if len(xs) > 200 else 5,
                        alpha=0.35 if len(xs) > 200 else 0.8, color=POINT_COLOUR)
            # Each reference line carries its own dash pattern and its name in the key. What this
            # replaced (2026-11-20, project owner) drew the claimed lower bound, the claimed upper
            # bound and the estimate as THREE IDENTICAL BLACK LINES WITH NO KEY, discarding the
            # name: indistinguishable from one another, and exactly coincident whenever claimed
            # equals the estimate, so the figure asserted three different things with one mark. The
            # guide asks for a key in place of a description given elsewhere, and the house rule
            # permits one wherever there is more than one series, which three lines are.
            for value, name in value_lines(claim):
                ax.axhline(value, color="k", lw=LINE_PT, label=name,
                           ls="-" if name == "estimate" else ("--" if name == "claimed" else ":"))
            if value_lines(claim):
                # One legend for the whole axes, drawn after the lines so it carries both the unit
                # colours and the line names; calling ax.legend twice replaces rather than adds. No
                # title here, because the key now mixes units with reference lines and a unit title
                # over both would be false.
                _, _labels = ax.get_legend_handles_labels()
                if len(_labels) > 1:
                    ax.legend(ncol=2, loc="best", frameon=False)
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticklabels, rotation=90 if len(ticks) > 6 else 0)
        elif pairs:
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
            labels = sorted(set(clusters)) if clusters else []
            if clusters and 1 < len(labels) <= MAX_UNIT_COLOURS:
                cmap = plt.get_cmap("tab20")
                for i, lab in enumerate(labels):
                    sel = [k for k, c in enumerate(clusters) if c == lab]
                    ax.plot([xs[k] for k in sel], [ys[k] for k in sel], "o", ms=4, alpha=0.8, color=cmap(i % 20), label=lab)
                # the key, which the guide asks for in place of a colour description; its frame is a line carrying no data, so
                # frameon=False (our choice, no source). Its sizes come from rcParams, so a figstyle reaches them.
                ax.legend(ncol=2, title=claim.get("resampling_unit"), loc="best", frameon=False)
            else:
                ax.plot(xs, ys, "o", ms=3 if len(xs) > 200 else 5, alpha=0.35 if len(xs) > 200 else 0.8, color=POINT_COLOUR)
            if identity:
                lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
                ax.plot([lo, hi], [lo, hi], "k-", lw=LINE_PT)
        if spread:
            ax.set_xlabel(claim.get("resampling_unit") or "row (file order)")
            ax.set_ylabel(axis_label(claim, "y", y))
        else:
            ax.set_xlabel(axis_label(claim, "x", x))
            ax.set_ylabel(axis_label(claim, "y", y))
        ax.grid(False)                                            # the guide's "We avoid": "Background gridlines"
        for side in ("top", "right"):                             # our choice, no source: not axis lines, and no data on them
            ax.spines[side].set_visible(False)
        ax.minorticks_off()                                       # our choice, no source: tick marks with no tick label
        ax.tick_params(which="both", top=False, right=False)      # ticks on the two axis lines only (our choice, no source)
        fig.tight_layout(pad=PAD_IN)
    return fig


def draw(claim: dict, pairs, clusters, x: str, y: str, files: dict[str, list[str]], out: Path,
         set_by_style: frozenset[str] = frozenset()) -> None:
    fig = figure(claim, pairs, clusters, x, y, files, set_by_style)   # selects the Agg backend before pyplot is used
    import matplotlib.pyplot as plt

    fig.savefig(out, dpi=DPI)
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
    proj = ctx.get("project") or {}
    project = str(proj.get("name") or proj.get("title") or verdicts.get("project") or "")   # the system, for the commentary
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
            draw(claim, pairs, clusters, x, y, files, path, frozenset(params or ()))
        note = out_dir / f"{claim['id']}.commentary.md"
        # what the canvas does not carry: the statement, the result, the verdict, the system, and the caption
        note.write_text(commentary(claim, files, len(pairs), clusters, identity_drawn(claim, pairs), project, x, y))

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
