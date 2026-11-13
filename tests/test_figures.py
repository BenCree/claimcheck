"""claimcheck.figures: one figure per claim, the points are the pairs relate used, the canvas carries the data and a two-line caption,
and the claim, its result and its verdict are in the commentary beside it."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from claimcheck import figures  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOY = REPO / "claimcheck-toy"


def test_commentary_states_the_claim_the_interval_and_the_verdict():
    c = {"id": "A1_x", "x": "d.a", "y": "d.b", "method": "spearman", "estimate": 0.51, "ci_lo": 0.2, "ci_hi": 0.7, "n_units": 13,
         "resampling_unit": "target", "p_value": 0.0018, "r_squared": 0.26, "claimed": 0.0, "verdict": "refuted", "ci_level": 0.95,
         "evidence": {"source": "Heavy atoms rank the ligands. A second sentence. A third one that is left out."}, "n_rows": 350,
         "x_dataset": "d", "y_dataset": "d", "level": "reference"}
    cap = figures.commentary(c, {"d": ["data/d.csv"]})
    assert cap.splitlines()[0] == "# A1_x"
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
    assert "UNVERIFIABLE: the data has no column b" in figures.commentary(c, {})
    assert figures.caption(c, {}) == "B: no points could be drawn; the reason is in B.commentary.md."


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


# ── the layout, 2026-11-12: axes, labels, a legend only for several series, a caption beneath, nothing else ──

def _claim(**extra) -> dict:
    return {"id": "L1_look", "x": "d.a", "y": "d.b", "x_unit": "kcal/mol", "y_unit": "kcal/mol", "method": "pearson", "estimate": 0.9,
            "ci_lo": 0.8, "ci_hi": 0.95, "n_units": 6, "resampling_unit": "ligand", "claimed": 0.9, "verdict": "agrees", "n_rows": 12,
            "x_dataset": "d", "y_dataset": "d", "evidence": {"source": "A long statement that used to be drawn on the canvas."}} | extra


def _draw(claim, pairs, clusters):
    import matplotlib.pyplot as plt
    fig = figures.figure(claim, pairs, clusters, "a", "b", {"d": ["data/d.csv"]})
    fig.canvas.draw()
    ax = fig.axes[0]
    look = {"suptitle": fig._suptitle, "titles": [ax.get_title(loc=loc) for loc in ("left", "center", "right")],
            "axes_texts": [t.get_text() for t in ax.texts], "fig_texts": [t for t in fig.texts], "n_axes": len(fig.axes),
            "grid": any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines()), "legend": ax.get_legend(),
            "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(), "n_lines": len(ax.lines),
            "caption_top": max(t.get_window_extent().y1 for t in fig.texts), "axes_bottom": ax.get_tightbbox().y0}
    plt.close(fig)
    return look


PAIRS = [(float(i), float(i) + 0.3 * (-1) ** i) for i in range(12)]


@pytest.mark.parametrize("clusters", [None, [f"L{i // 2}" for i in range(12)]], ids=["one series", "coloured by unit"])
def test_the_canvas_has_no_suptitle_no_title_no_text_block_and_no_grid(clusters):
    look = _draw(_claim(), PAIRS, clusters)
    assert look["suptitle"] is None and look["titles"] == ["", "", ""] and look["axes_texts"] == [] and not look["grid"]
    assert look["n_axes"] == 1 and len(look["fig_texts"]) == 1                 # the caption is the only text outside the axes
    assert look["xlabel"] == "d.a (kcal/mol)" and look["ylabel"] == "d.b (kcal/mol)"


def test_the_caption_is_beneath_the_axes_at_most_two_lines_and_defines_every_mark():
    by_unit = [f"L{i // 2}" for i in range(12)]
    look = _draw(_claim(), PAIRS, by_unit)
    cap = look["fig_texts"][0].get_text()
    assert look["caption_top"] <= look["axes_bottom"]                          # beneath: no overlap with the axes or their labels
    assert 1 <= len(cap.splitlines()) <= 2 and look["fig_texts"][0].get_fontsize() == 7
    assert cap.startswith("L1_look: data/d.csv; 12 rows, 6 ligand(s).")
    assert "One point is one row, coloured by ligand." in cap and "The line is y = x." in cap
    for absent in ("AGREES", "agrees", "claimed", "CI", "statement"):          # the result is in the commentary, not here
        assert absent not in cap
    assert "AGREES" in figures.commentary(_claim(), {"d": ["data/d.csv"]})
    # a long list of files falls back to the dataset id and never to a third line
    long = figures.caption(_claim(), {"d": [f"data/a_very_long_directory_name/table_number_{i}.csv" for i in range(8)]}, 12, by_unit, True)
    assert len(long.splitlines()) <= 2 and long.startswith("L1_look: d; 12 rows")


def test_a_legend_only_where_there_is_more_than_one_series_and_no_line_without_a_definition():
    one = _draw(_claim(), PAIRS, None)
    several = _draw(_claim(), PAIRS, [f"L{i // 2}" for i in range(12)])
    assert one["legend"] is None and several["legend"] is not None
    assert sorted(t.get_text() for t in several["legend"].get_texts()) == [f"L{i}" for i in range(6)]
    assert one["n_lines"] == 2 and "The line is y = x." in one["fig_texts"][0].get_text()      # the points and y = x, which the caption names
    no_identity = _draw(_claim(y_unit="pKd"), PAIRS, None)
    assert no_identity["n_lines"] == 1 and "y = x" not in no_identity["fig_texts"][0].get_text()
    assert "coloured" not in one["fig_texts"][0].get_text()


def test_a_claim_with_no_points_gets_empty_axes_and_a_caption_and_no_paragraph_on_the_canvas():
    c = _claim(verdict="unverifiable", why="the data has no column b", estimate=None)
    look = _draw(c, [], None)
    assert look["axes_texts"] == [] and look["suptitle"] is None and look["n_lines"] == 0
    assert look["fig_texts"][0].get_text() == "L1_look: no points could be drawn; the reason is in L1_look.commentary.md."
    assert "the data has no column b" in figures.commentary(c, {})


def test_build_writes_the_commentary_beside_each_figure_and_indexes_it(tmp_path):
    proj = tmp_path / "toy"
    shutil.copytree(TOY, proj, ignore=shutil.ignore_patterns("figures"))
    if not (proj / "results" / "claims.json").exists():
        from claimcheck.relate import main as relate_main
        assert relate_main([str(proj), str(proj / "results" / "claims.json")]) == 0
    index = figures.build(proj, proj / "figures" / "claims")
    assert index
    for r in index:
        text = (proj / r["commentary"]).read_text()
        assert r["commentary"] == f"figures/claims/{r['id']}.commentary.md" and text.startswith(f"# {r['id']}\n")
        assert str(r["verdict"]).upper() in text
