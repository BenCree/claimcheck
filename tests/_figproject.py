"""A synthetic project for the figure tests: data, a declaration and hand-written verdicts.

Nothing a figure test checks depends on what another module computes. Integer arithmetic and one exact division per coordinate,
so every machine draws the same points.
"""

from __future__ import annotations

import json
from pathlib import Path


def pairs(n: int) -> list[tuple[float, float]]:
    """n deterministic (x, y) points."""
    return [(((i * 37) % 101) / 10.0, ((i * i * 13) % 97) / 9.0 + (i % 5)) for i in range(n)]


#: A verdict row as `claimcheck.relate` writes it, for a claim over the columns `a` and `b` of dataset `d`, clustered by `target`.
BASE = {
    "id": "X", "x": "d.a", "y": "d.b", "kind": "correlation", "method": "spearman", "estimate": 0.42, "ci_lo": 0.10, "ci_hi": 0.66, "n_units": 5,
    "resampling_unit": "target", "p_value": 0.0123, "r_squared": 0.18, "claimed": 0.5, "verdict": "agrees", "ci_level": 0.95,
    "evidence": {"source": "The method ranks the ligands within a target. A second sentence follows. A third one is left out of the caption."},
    "n_rows": 60, "x_dataset": "d", "y_dataset": "d", "level": "reference", "why": "claimed 0.5 lies inside the 95% CI",
    "x_unit": "kcal/mol", "y_unit": "kcal/mol",
}


#: A `kind = "value"` verdict row, where `x = y` is one column: y = x would run through every point, so no line is drawn.
VALUE = {
    "id": "V", "x": "d.a", "y": "d.a", "kind": "value", "method": "mean", "estimate": 4.9, "ci_lo": 4.1, "ci_hi": 5.7,
    "n_units": 5, "resampling_unit": "target", "claimed_range": [4.0, 6.0], "verdict": "agrees", "ci_level": 0.95,
    "evidence": {"source": "The mean of a is between 4 and 6 kcal/mol."}, "n_rows": 60, "x_dataset": "d", "y_dataset": "d",
    "level": "reference", "why": "the 95% CI lies inside the claimed range", "x_unit": "kcal/mol", "y_unit": "kcal/mol",
}


def text_artists(fig) -> dict[str, list[str]]:
    """Every visible, non-empty Text artist of a drawn figure, split into the ones the figure contract allows and the rest.

    THE CONTRACT (2026-11-13): the only words on a claim figure are the two axis labels, the tick labels (with the offset text
    matplotlib writes when it factors a power of ten out of them), and the key, meaning its entries and its title, which names
    the resampling unit the colours stand for. Any other Text is a word on the canvas, and the figure is wrong: a suptitle, a
    title, a caption beneath the axes, an annotation beside a point.

    `fig.canvas.draw()` must have run, or the tick labels do not exist yet. The figure is walked with `findobj`, so a Text
    nested in a container is found as well.
    """
    import matplotlib.text

    allowed: set[int] = set()
    for ax in fig.axes:
        allowed.update(id(t) for t in (ax.xaxis.label, ax.yaxis.label, ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()))
        allowed.update(id(t) for t in [*ax.get_xticklabels(), *ax.get_yticklabels()])
        legend = ax.get_legend()
        if legend is not None:
            allowed.update(id(t) for t in legend.get_texts())
            allowed.add(id(legend.get_title()))
    out: dict[str, list[str]] = {"allowed": [], "extra": []}
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip():
            out["allowed" if id(t) in allowed else "extra"].append(t.get_text())
    return out


def write_context(proj: Path, project_lines: tuple[str, ...] = ()) -> None:
    """context.toml with one dataset; `project_lines` are raw TOML lines added under [project]."""
    (proj / "context.toml").write_text(
        '[project]\nname = "figure_tests"\ntitle = "A synthetic project for the figure tests"\n' + "".join(f"{line}\n" for line in project_lines)
        + '\n[[datasets]]\nid = "d"\norigin = "computed"\nlevel = "reference"\nfiles = ["data/d.csv"]\n')


def write_project(proj: Path, project_lines: tuple[str, ...] = (), n_claims: int = 3) -> None:
    """Up to three claims: one coloured by unit, the same pair of columns without a unit, and one whose column is not in the data."""
    (proj / "data").mkdir(parents=True)
    (proj / "results").mkdir()
    (proj / "data" / "d.csv").write_text("a,b,target\n" + "".join(f"{a!r},{b!r},t{i % 5}\n" for i, (a, b) in enumerate(pairs(60))))
    write_context(proj, project_lines)
    claims = [
        {**BASE, "id": "P1_by_unit"},
        {**BASE, "id": "P2_by_row", "resampling_unit": None, "verdict": "refuted", "why": "claimed 0.5 lies outside the 95% CI"},
        {**BASE, "id": "P3_missing_column", "y": "d.nope", "verdict": "unverifiable", "estimate": None, "ci_lo": None, "ci_hi": None, "n_rows": None,
         "why": "the data has no column nope"},
    ][:n_claims]
    (proj / "results" / "claims.json").write_text(json.dumps({"project": "figure_tests", "relationships": claims}))
