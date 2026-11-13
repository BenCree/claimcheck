"""claimcheck.figures: one figure per claim, the points are the pairs relate used, the canvas carries the data and no words beyond
its axis labels, tick labels and key, and the claim, its result, its verdict and its caption are in the commentary beside it."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from _figproject import text_artists  # noqa: E402
from claimcheck import figures  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOY = REPO / "claimcheck-toy"


def test_commentary_states_the_claim_the_interval_and_the_verdict():
    c = {"id": "A1_x", "x": "d.a", "y": "d.b", "method": "spearman", "estimate": 0.51, "ci_lo": 0.2, "ci_hi": 0.7, "n_units": 13,
         "resampling_unit": "target", "p_value": 0.0018, "r_squared": 0.26, "claimed": 0.0, "verdict": "refuted", "ci_level": 0.95,
         "evidence": {"source": "Heavy atoms rank the ligands. A second sentence. A third one that is left out."}, "n_rows": 350,
         "x_dataset": "d", "y_dataset": "d", "level": "reference"}
    cap = figures.commentary(c, {"d": ["data/d.csv"]}, 350, ["t1"] * 350)
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
    assert "No points could be drawn, so the axes are empty and labelled: the data has no column b." in figures.marks(c, {})


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


# ── the layout, 2026-11-13: the points, two axis lines, the labels, a key where there is more than one series, no words else ──
#
# The project owner's reading on 2026-11-13 was that the figures still looked weird: too many lines, and words at the bottom
# that belong in the caption. The contract since then, with Nature's research figure guide as the source for each part, is in
# the docstring of claimcheck/figures.py. These tests hold both halves of it: nothing else is drawn, and everything that was
# drawn is now in the commentary.

def _claim(**extra) -> dict:
    return {"id": "L1_look", "x": "d.a", "y": "d.b", "x_unit": "kcal/mol", "y_unit": "kcal/mol", "method": "pearson", "estimate": 0.9,
            "ci_lo": 0.8, "ci_hi": 0.95, "n_units": 6, "resampling_unit": "ligand", "claimed": 0.9, "verdict": "agrees", "n_rows": 12,
            "x_dataset": "d", "y_dataset": "d", "evidence": {"source": "A long statement that used to be drawn on the canvas."}} | extra


PAIRS = [(float(i), float(i) + 0.3 * (-1) ** i) for i in range(12)]
BY_UNIT = [f"L{i // 2}" for i in range(12)]
MANY_PAIRS = [(float(i), float(i) + 0.5) for i in range(25)]      # more units than MAX_UNIT_COLOURS, so one colour and no key
FILES = {"d": ["data/d.csv"]}
#: name -> (claim, pairs, cluster labels): one case per branch of `figures.figure`, the value claim among them.
CASES = {
    "one series": (_claim(), PAIRS, None),
    "coloured by unit": (_claim(), PAIRS, BY_UNIT),
    "many units": (_claim(resampling_unit="row"), MANY_PAIRS, [f"r{i}" for i in range(len(MANY_PAIRS))]),
    "value claim": (_claim(kind="value", method="mean", y="d.a", y_unit="kcal/mol"), PAIRS, BY_UNIT),
    "no points": (_claim(verdict="unverifiable", why="the data has no column b", estimate=None), [], None),
}


def _drawn(claim, pairs, clusters):
    """The Figure of one claim, drawn so that its tick labels exist; the caller closes it."""
    fig = figures.figure(claim, pairs, clusters, "a", "b", FILES)
    fig.canvas.draw()
    return fig


def _look(claim, pairs, clusters) -> dict:
    """What is on the canvas: every text artist, the chrome, the lines and their widths, the sizes."""
    import matplotlib.pyplot as plt

    fig = _drawn(claim, pairs, clusters)
    ax = fig.axes[0]
    drawn_lines = [ln for ln in ax.lines if ln.get_linestyle() not in ("None", "none", "")]
    look = {"texts": text_artists(fig), "suptitle": fig._suptitle, "titles": [ax.get_title(loc=loc) for loc in ("left", "center", "right")],
            "axes_texts": [t.get_text() for t in ax.texts], "fig_texts": [t.get_text() for t in fig.texts], "n_axes": len(fig.axes),
            "grid": any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines()), "legend": ax.get_legend(),
            "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(), "n_series": len(ax.lines) - len(drawn_lines),
            "n_reference_lines": len(drawn_lines),
            "spines": {s: ax.spines[s].get_visible() for s in ("left", "bottom", "top", "right")},
            "n_minor_ticks": len(ax.xaxis.get_minor_ticks()) + len(ax.yaxis.get_minor_ticks()),
            "line_widths": sorted({ln.get_linewidth() for ln in drawn_lines}
                                  | {ax.spines[s].get_linewidth() for s in ("left", "bottom")}),
            "label_pt": ax.xaxis.label.get_fontsize(), "tick_pt": {t.get_fontsize() for t in ax.get_xticklabels() if t.get_text()},
            "size_mm": tuple(round(v * 25.4) for v in fig.get_size_inches())}
    plt.close(fig)
    return look


# ── no words on the canvas: the regression test, with its failing twin ──

@pytest.mark.parametrize("case", list(CASES))
def test_the_only_words_on_a_claim_figure_are_the_axis_labels_the_tick_labels_and_the_key(case):
    """The canvas carries no text but the axis labels, the tick labels and the key (its entries and its title, which names the
    resampling unit the colours stand for). `text_artists` reports anything else, and the twin below plants one of each kind to
    show that it does."""
    look = _look(*CASES[case])
    assert look["texts"]["extra"] == [], f"{case}: words on the canvas that the contract does not allow"
    assert look["texts"]["allowed"], f"{case}: the auditor sees no text at all, so the empty list above proves nothing"
    assert look["suptitle"] is None and look["titles"] == ["", "", ""]
    assert look["axes_texts"] == [] and look["fig_texts"] == []                 # no annotation, and no caption beneath the axes
    assert look["n_axes"] == 1 and not look["grid"]
    assert look["xlabel"] == "d.a (kcal/mol)" and look["ylabel"].endswith("(kcal/mol)")


@pytest.mark.parametrize("plant", ["caption beneath the axes", "suptitle", "title", "text in the axes", "annotation"])
def test_a_word_put_back_on_the_canvas_is_caught(plant):
    """The failing half: one text artist of each kind that used to be drawn, planted on a figure that is otherwise right."""
    import matplotlib.pyplot as plt

    fig = _drawn(*CASES["coloured by unit"])
    ax = fig.axes[0]
    word = f"a {plant} that must not be on the canvas"
    if plant == "caption beneath the axes":
        fig.text(0.02, 0.01, word)
    elif plant == "suptitle":
        fig.suptitle(word)
    elif plant == "title":
        ax.set_title(word)
    elif plant == "text in the axes":
        ax.text(1.0, 1.0, word)
    else:
        ax.annotate(word, (1.0, 1.0), xytext=(2.0, 2.0), arrowprops={"width": 0.5})
    fig.canvas.draw()
    assert text_artists(fig)["extra"] == [word], f"a {plant} is not reported"
    plt.close(fig)


# ── the caption, which is now in the commentary ──

def test_the_caption_is_in_the_commentary_and_names_the_system_the_cohort_the_point_and_the_error_bars():
    note = figures.commentary(_claim(), FILES, 12, BY_UNIT, True, project="figure_tests")
    assert "System figure_tests" in note and "data/d.csv" in note and "60 rows" not in note
    assert "12 points from data/d.csv; one point is one row." in note
    assert "6 ligand(s), one colour per ligand, each named in the key" in note
    assert "The straight line is y = x, at 0.6 pt" in note
    assert "No error bars are drawn: the interval in the result line above belongs to the claim, not to a point." in note
    assert "The canvas: x axis d.a (kcal/mol), y axis d.b (kcal/mol)." in note
    assert "AGREES" in note and "claimed +0.90" in note                        # the verdict too, which the canvas does not state
    assert not _look(*CASES["coloured by unit"])["fig_texts"]                   # and none of it is drawn


def test_the_commentary_does_not_promise_an_interval_a_claim_does_not_have():
    """`relate` writes no interval for an unmeasured claim, and several methods do not use a bootstrap (Clopper-Pearson for
    binomial_count and topn_recovery, chi-square for pooled_sd), so the caption names no kind of interval and no unit count."""
    note = figures.commentary(_claim(ci_lo=None, ci_hi=None, estimate=None, n_units=None), FILES, 12, BY_UNIT, True)
    assert "No error bars are drawn, and the result line above has no interval." in note
    assert "bootstrap" not in note and "?" not in note


def test_the_commentary_states_a_missing_unit_the_way_the_axis_label_does():
    note = figures.commentary(_claim(y_unit=None), FILES, 12, None, False)
    assert "y axis d.b (unit not declared)" in note and "No reference line is drawn." in note
    assert _look(_claim(y_unit=None), PAIRS, None)["ylabel"] == "d.b (unit not declared)"


# ── the chrome: two axis lines with their ticks, and no other line ──

def test_the_axis_lines_and_their_major_ticks_stay_and_the_box_frame_and_minor_ticks_go():
    look = _look(*CASES["one series"])
    assert look["spines"] == {"left": True, "bottom": True, "top": False, "right": False}
    assert look["n_minor_ticks"] == 0
    assert look["tick_pt"] and look["label_pt"] == 7.0 and look["tick_pt"] == {7.0}      # 7-pt text at the size it is read at
    assert look["size_mm"] == (183, 120)                                                 # Nature's double-column width
    assert all(0.25 <= w <= 1.0 for w in look["line_widths"]), look["line_widths"]       # the guide's range for lines


def test_a_key_only_where_there_is_more_than_one_series_and_it_has_no_frame():
    one, several = _look(*CASES["one series"]), _look(*CASES["coloured by unit"])
    assert one["legend"] is None and several["legend"] is not None
    assert sorted(t.get_text() for t in several["legend"].get_texts()) == [f"L{i}" for i in range(6)]
    assert several["legend"].get_title().get_text() == "ligand" and not several["legend"].get_frame_on()
    assert _look(*CASES["many units"])["legend"] is None                      # more colours than can be read: one colour, no key


def test_a_line_is_drawn_exactly_where_the_commentary_says_one_is():
    for case, (claim, pairs, clusters) in CASES.items():
        look = _look(claim, pairs, clusters)
        identity = figures.identity_drawn(claim, pairs)
        note = figures.marks(claim, FILES, len(pairs), clusters, identity)
        assert look["n_reference_lines"] == (1 if identity else 0), case
        if pairs:
            assert ("The straight line is y = x" in note) is identity, case
            assert ("No reference line is drawn." in note) is (not identity), case
        else:
            assert "No points could be drawn" in note and "y = x" not in note, case


def test_a_value_claim_gets_no_y_equals_x_line_through_points_that_all_lie_on_it():
    """`kind = "value"` declares x = y, one column against itself: the line would pass through every point and say nothing."""
    claim, pairs, clusters = CASES["value claim"]
    assert not figures.identity_drawn(claim, pairs) and _look(claim, pairs, clusters)["n_reference_lines"] == 0
    assert figures.identity_drawn(_claim(), pairs), "two columns in one unit still get the line, so the test above means something"


def test_a_claim_with_no_points_gets_its_labelled_axes_and_no_words_and_the_reason_is_in_the_commentary():
    claim, pairs, clusters = CASES["no points"]
    look = _look(claim, pairs, clusters)
    assert look["n_series"] == 0 and look["n_reference_lines"] == 0 and look["legend"] is None and look["texts"]["extra"] == []
    assert look["xlabel"] == "d.a (kcal/mol)" and look["spines"]["left"]
    note = figures.commentary(claim, {}, 0, None, False)
    assert "No points could be drawn, so the axes are empty and labelled: the data has no column b." in note
    assert "an oversight" in note and "UNVERIFIABLE" in note


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
        assert "The canvas: x axis " in text and "System " in text            # the caption, with the system it is drawn from
        assert ("points from" in text) is (r["n_points"] > 0)
        assert ("No error bars are drawn" in text) is (r["n_points"] > 0)
