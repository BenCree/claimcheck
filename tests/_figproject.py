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
