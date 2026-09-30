"""claimcheck.figures: one figure per claim, the caption carries the claim and its verdict, and the points are the pairs relate used."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from claimcheck import figures  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOY = REPO / "claimcheck-toy"


def test_caption_states_the_claim_the_interval_and_the_verdict():
    c = {"id": "A1_x", "x": "d.a", "y": "d.b", "method": "spearman", "estimate": 0.51, "ci_lo": 0.2, "ci_hi": 0.7, "n_units": 13,
         "resampling_unit": "target", "p_value": 0.0018, "r_squared": 0.26, "claimed": 0.0, "verdict": "refuted", "ci_level": 0.95,
         "evidence": {"source": "Heavy atoms rank the ligands. A second sentence. A third one that is left out."}, "n_rows": 350,
         "x_dataset": "d", "y_dataset": "d", "level": "reference"}
    cap = figures.caption(c, {"d": ["data/d.csv"]})
    assert cap.splitlines()[0] == "A1_x"
    assert "Heavy atoms rank the ligands. A second sentence." in cap and "third" not in cap
    assert "[+0.200, +0.700] over 13 target(s)" in cap and "REFUTED" in cap and "claimed +0.00" in cap
    assert "350 rows" in cap and "data/d.csv" in cap


def test_an_estimate_without_an_interval_gives_the_reason():
    c = {"id": "C", "x": "d.a", "y": "d.b", "method": "spearman", "estimate": 1.0, "ci_lo": None, "ci_hi": None, "claimed": 1.0,
         "verdict": "unverifiable", "why": "one ligand cannot be resampled"}
    line = figures.result_line(c)
    assert "no interval" in line and "one ligand cannot be resampled" in line and "n/a, n/a" not in line


def test_an_unmeasured_claim_says_why():
    c = {"id": "B", "x": "d.a", "y": "d.b", "verdict": "unverifiable", "why": "the data has no column b"}
    assert "UNVERIFIABLE: the data has no column b" in figures.caption(c, {})


@pytest.mark.skipif(not (TOY / "results" / "claims.json").exists(), reason="the toy project has no verdicts")
def test_every_claim_of_the_toy_project_gets_a_figure_with_relates_points(tmp_path):
    proj = tmp_path / "toy"
    shutil.copytree(TOY, proj)
    index = figures.build(proj, proj / "figures" / "claims")
    verdicts = json.loads((proj / "results" / "claims.json").read_text())["relationships"]
    assert [r["id"] for r in index] == [v["id"] for v in verdicts]
    for r, v in zip(index, verdicts, strict=True):
        assert (proj / r["figure"]).stat().st_size > 5000
        if v.get("n_rows") is not None and v.get("estimate") is not None:
            assert r["n_points"] == v["n_rows"], (r, v["n_rows"])


def test_a_refutation_by_a_hair_prints_the_hair():
    c = {"id": "D", "method": "pearson", "estimate": 0.99998, "ci_lo": 0.99996, "ci_hi": 0.99999, "n_units": 8, "claimed": 1.0,
         "verdict": "refuted", "why": "claimed 1.0 lies outside the 95% CI"}
    line = figures.result_line(c)
    assert "r - claimed = -2.00e-05" in line and "claimed 1.0 lies outside" in line
