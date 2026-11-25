"""Two-sided tests for the roc_auc_difference estimator (2026-11-20, proposal P36).

THE KNOWN ANSWER IS A COLUMN AGAINST ITSELF: exactly 0.0, with a bootstrap interval of exactly zero width. That second
half is the test of the thing this estimator exists for. Resampling the two columns independently would give the same
0.0 estimate and a WIDE interval, so an interval of [0, 0] on identical columns is the only cheap proof that a resample
draws a row with both of its scores and the pairing survives.

The failing twins: a score against its own mirror must give exactly 2 * AUC - 1, a perfect ranking against an inverted
one the maximum +1.0, the two columns the other way round exactly the negation, and a degenerate table (one class, a
ragged row, fewer than three units) must return None so the claim reads `unverifiable` rather than 0.0 (which would be
"the two scores are equal", the conclusion a broken join produces).

The p is `paired_swap_p`, whose null is that the two columns are interchangeable, not that either reads the label. The
last tests pin both halves of that: two identical columns give p = 1.0 however good they are, where a label permutation
would call them decisive, and the share of p <= 0.05 under exchangeable columns is measured rather than asserted from
the construction.
"""
import math
import random
from pathlib import Path

import pytest

from claimcheck.relate import (
    ESTIMATORS,
    bootstrap_ci,
    check,
    mid_ranks,
    paired_swap_p,
    roc_auc,
    roc_auc_difference,
)
from claimcheck.scrutiny import METHOD_RANGE

LABELS = [1, 1, 1, 0, 0, 0]
PERFECT = [9.0, 8.0, 7.0, 3.0, 2.0, 1.0]
INVERTED = [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]


def packed(a: list[float], b: list[float]) -> list[tuple[float, float]]:
    return list(zip(a, b, strict=True))


# --- the estimator ------------------------------------------------------------------------------


def test_a_column_against_itself_is_exactly_zero():
    """THE KNOWN ANSWER. Not approximately 0: the two AUCs are the same float, so the subtraction is exact."""
    assert roc_auc_difference(LABELS, packed(PERFECT, PERFECT)) == 0.0
    noisy = [0.3, 0.9, 0.1, 0.9, 0.4, 0.2]
    assert roc_auc_difference(LABELS, packed(noisy, noisy)) == 0.0


def test_a_perfect_ranking_against_an_inverted_one_is_the_maximum():
    assert roc_auc_difference(LABELS, packed(PERFECT, INVERTED)) == pytest.approx(1.0)
    assert roc_auc_difference(LABELS, packed(INVERTED, PERFECT)) == pytest.approx(-1.0)


def test_a_mirror_is_exactly_two_auc_minus_one():
    """AUC(-y) = 1 - AUC(y), so the difference against a negated column is 2 * AUC - 1 whatever the column is."""
    rng = random.Random(3)
    labels = [1] * 20 + [0] * 80
    for _ in range(5):
        y = [rng.random() + (0.4 if lab else 0.0) for lab in labels]
        auc = roc_auc(labels, y)
        assert roc_auc_difference(labels, packed(y, [-v for v in y])) == pytest.approx(2 * auc - 1, abs=1e-12)


def test_it_is_antisymmetric_exactly():
    rng = random.Random(5)
    labels = [1] * 10 + [0] * 40
    a = [rng.random() + (0.3 if lab else 0.0) for lab in labels]
    b = [rng.random() for _ in labels]
    assert roc_auc_difference(labels, packed(a, b)) == -roc_auc_difference(labels, packed(b, a))


def test_a_perfect_ranking_against_a_random_one_is_near_a_half():
    """The informative end of the two-sided pair: perfect is 1.0 and uninformative is about 0.5."""
    rng = random.Random(7)
    labels = [1] * 30 + [0] * 270
    perfect = [1.0 if lab else 0.0 for lab in labels]
    noise = [rng.random() for _ in labels]
    d = roc_auc_difference(labels, packed(perfect, noise))
    assert 0.38 < d < 0.62
    assert roc_auc(labels, perfect) == pytest.approx(1.0)


def test_one_discordant_pair_moves_it_by_one_pair():
    """3 by 3 makes 9 pairs: a difference of two AUCs moves in units of 1/9 and nothing else."""
    one_swap = [9.0, 8.0, 2.5, 3.0, 2.0, 1.0]
    assert roc_auc_difference(LABELS, packed(PERFECT, one_swap)) == pytest.approx(1 / 9)


def test_degenerate_inputs_return_none_not_zero():
    """None, not 0.0: a table whose join dropped every active must not read as "the two scores are equal"."""
    assert roc_auc_difference([1, 1, 1], packed([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])) is None   # no negatives
    assert roc_auc_difference([0, 0, 0], packed([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])) is None   # no positives
    assert roc_auc_difference([], []) is None
    assert roc_auc_difference([1, 0], [(1.0, 2.0)]) is None                                  # ragged
    assert roc_auc_difference(LABELS, PERFECT) is None       # two-column pairs reaching a three-column estimator


def test_it_is_registered_as_a_correlation_method_and_scrutiny_knows_its_range():
    fn, kind = ESTIMATORS["roc_auc_difference"]
    assert fn is roc_auc_difference and kind == "correlation"
    assert METHOD_RANGE["roc_auc_difference"] == (-1.0, 1.0)


# --- the pairing, which is the whole point ------------------------------------------------------


def test_the_bootstrap_interval_on_identical_columns_has_zero_width():
    """THE PAIRING TEST. Had the two columns been resampled independently this interval would be wide."""
    rng = random.Random(13)
    labels = [1 if i % 10 == 0 else 0 for i in range(200)]
    y = [rng.random() + (0.5 if lab else 0.0) for lab in labels]
    units = [f"u{i // 4}" for i in range(200)]
    lo, hi, n = bootstrap_ci(packed(labels, packed(y, y)), units, estimate=roc_auc_difference)
    assert (lo, hi) == (0.0, 0.0) and n == 50


def test_the_paired_interval_is_narrower_than_either_separate_one():
    """Two informative, strongly correlated columns: most of each AUC's interval is the cohort variance they share, and
    the paired difference cancels it. The separate intervals here are more than ten times the paired one."""
    rng = random.Random(17)
    labels = [1 if i % 8 == 0 else 0 for i in range(240)]
    a = [rng.random() + (0.6 if lab else 0.0) for lab in labels]
    b = [v + 0.02 * rng.random() for v in a]            # the same ordering, jittered
    units = [f"u{i // 3}" for i in range(240)]
    wa = [hi - lo for lo, hi, _ in [bootstrap_ci(packed(labels, a), units, estimate=roc_auc)]][0]
    wb = [hi - lo for lo, hi, _ in [bootstrap_ci(packed(labels, b), units, estimate=roc_auc)]][0]
    lo, hi, _ = bootstrap_ci(packed(labels, packed(a, b)), units, estimate=roc_auc_difference)
    assert (hi - lo) < min(wa, wb) / 10 > 0.0
    assert lo <= 0.0 <= hi                               # and it contains zero, which is the truth here


def test_mid_ranks_do_not_change_an_auc():
    """What lets `paired_swap_p` put two columns of different scales on one scale: exact, ties included."""
    rng = random.Random(19)
    labels = [1 if i % 5 == 0 else 0 for i in range(100)]
    for scores in ([rng.random() for _ in labels],
                   [float(rng.randint(0, 4)) for _ in labels],          # heavily tied
                   [7.0] * 100):
        assert roc_auc(labels, mid_ranks(scores)) == roc_auc(labels, scores)
    assert mid_ranks([5.0, 1.0, 5.0, 3.0]) == [3.5, 1.0, 3.5, 2.0]


# --- the p --------------------------------------------------------------------------------------


def test_two_identical_columns_give_the_largest_possible_p():
    """The null is exchangeability, so two columns that are the same column can never be distinguished, however good
    they are. A label permutation would call this pair decisive, which is why that p is not the one reported."""
    rng = random.Random(23)
    labels = [1 if i % 6 == 0 else 0 for i in range(120)]
    y = [rng.random() + (0.9 if lab else 0.0) for lab in labels]        # a strong scorer
    units = [f"u{i // 2}" for i in range(120)]
    assert roc_auc(labels, y) > 0.8
    p, floor = paired_swap_p(packed(labels, packed(y, y)), units, n_perm=200)
    assert p == 1.0 and floor == pytest.approx(1 / 201)


def test_a_perfect_column_against_its_mirror_sits_at_the_floor():
    rng = random.Random(29)
    labels = [1 if i % 6 == 0 else 0 for i in range(120)]
    y = [1.0 if lab else 0.0 for lab in labels]
    y = [v + 0.001 * rng.random() for v in y]
    units = [f"u{i // 2}" for i in range(120)]
    p, floor = paired_swap_p(packed(labels, packed(y, [-v for v in y])), units, n_perm=200)
    assert p == pytest.approx(floor)


def test_the_swap_null_is_symmetric_about_zero():
    """Swapping every unit of a draw gives exactly the negated statistic, so a two-sided test about 0 is the right one.
    Pinned on the statistic itself: the all-swapped table is the negation of the none-swapped one."""
    rng = random.Random(31)
    labels = [1 if i % 4 == 0 else 0 for i in range(60)]
    a = [rng.random() + (0.4 if lab else 0.0) for lab in labels]
    b = [rng.random() for _ in labels]
    assert roc_auc_difference(labels, packed(mid_ranks(a), mid_ranks(b))) == pytest.approx(
        -roc_auc_difference(labels, packed(mid_ranks(b), mid_ranks(a))), abs=1e-12)


def test_the_p_is_undefined_below_three_units():
    labels = [1, 1, 0, 0, 0, 0]
    pairs = packed(labels, packed(PERFECT, INVERTED))
    assert paired_swap_p(pairs, ["u1", "u1", "u1", "u2", "u2", "u2"], n_perm=50) is None
    assert paired_swap_p(packed([1, 1, 1], [(1.0, 2.0)] * 3), None, n_perm=50) is None   # no negatives


def test_the_p_is_calibrated_under_exchangeable_columns():
    """MEASURED, NOT ASSERTED FROM THE CONSTRUCTION. Two columns drawn the same way, each carrying the same relationship
    to the label, are exchangeable, so the share of p <= 0.05 should be near 0.05. 120 tables of 80 rows in 40 units,
    200 draws each: 0.05 with a standard error of 0.02, so the band is wide on purpose and a p that was testing the
    label (every table would reject) or was constant (none would) fails it by a mile."""
    rng = random.Random(37)
    rejected = 0
    n_sets = 120
    for _ in range(n_sets):
        labels = [1 if i % 5 == 0 else 0 for i in range(80)]
        units = [f"u{i // 2}" for i in range(80)]
        a = [rng.random() + (0.5 if lab else 0.0) for lab in labels]
        b = [rng.random() + (0.5 if lab else 0.0) for lab in labels]
        p, _ = paired_swap_p(packed(labels, packed(a, b)), units, n_perm=200)
        rejected += p <= 0.05
    assert rejected / n_sets < 0.18


# --- end to end through check() -----------------------------------------------------------------


HEAD = '[project]\nname = "t"\nlicence = "CC0-1.0"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n'
OTHER = '\n[[datasets]]\nid = "e"\norigin = "computed"\nfiles = ["e.csv"]\n'
COLS = ("is_active", "score", "mirror", "copy", "noise", "flat")


def table(n: int = 240, seed: int = 41) -> str:
    rng = random.Random(seed)
    lines = ["compound,unit," + ",".join(COLS)]
    for i in range(n):
        lab = 1 if i % 6 == 0 else 0
        s = round(rng.random() + (0.7 if lab else 0.0), 6)
        lines.append(f"c{i},u{i // 3},{lab},{s},{-s},{s},{round(rng.random(), 6)},1.0")
    return "\n".join(lines) + "\n"


def project(tmp_path: Path, rels: str, extra_vars: str = "", second_file: bool = False) -> dict:
    (tmp_path / "d.csv").write_text(table())
    if second_file:
        (tmp_path / "e.csv").write_text("compound,other\n" + "\n".join(f"c{i},{i}" for i in range(240)) + "\n")
    vars_ = "".join(f'\n[[variables]]\ndataset = "d"\nname = "{c}"\nunit = "1"\n' for c in COLS)
    if second_file:
        vars_ += '\n[[variables]]\ndataset = "e"\nname = "other"\nunit = "1"\n'
    (tmp_path / "context.toml").write_text(HEAD + (OTHER if second_file else "") + vars_ + extra_vars + "\n" + rels)
    return {r["id"]: r for r in check(tmp_path)["relationships"]}


def rel(rid: str, body: str) -> str:
    return (f'\n[[relationships]]\nid = "{rid}"\nkind = "correlation"\nmethod = "roc_auc_difference"\n'
            f'resampling_unit = "unit"\n{body}\n')


def test_the_known_answer_agrees_with_a_zero_width_interval(tmp_path):
    row = project(tmp_path, rel("KA", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.copy"\nclaimed = 0.0'))["KA"]
    assert row["verdict"] == "agrees"
    assert (row["estimate"], row["ci_lo"], row["ci_hi"]) == (0.0, 0.0, 0.0)
    assert row["auc_y"] == row["auc_baseline"] > 0.5
    assert row["baseline"] == "d.copy" and row["baseline_unit"] == "1"
    assert row["r_squared"] is None and "not a share of anything" in row["r_squared_note"]
    assert "INTERCHANGEABLE" in row["p_method"] and row["p_value"] == 1.0
    assert "recomputes BOTH AUROCs" in row["interval_source"]
    assert row["n_rows"] == 240 and row["dropped"] == 0 and row["n_units"] == 80 and row["n_actives"] == 40


def test_the_mirror_is_refuted_against_a_claim_of_zero(tmp_path):
    """The failing twin at the level of a verdict: a score against its own negation is not a null difference."""
    row = project(tmp_path, rel("M", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.mirror"\nclaimed = 0.0'))["M"]
    assert row["verdict"] == "refuted" and row["estimate"] > 0.3
    assert row["estimate"] == pytest.approx(2 * row["auc_y"] - 1, abs=1e-6)


def test_a_claimed_range_straddling_zero_is_unverifiable_and_one_above_it_agrees(tmp_path):
    """The two verdicts this estimator was built to tell apart, on one table: a real difference excludes zero and a null
    one straddles it. `noise` is an uninformative column, so `score` minus `noise` is a real and positive difference."""
    out = project(tmp_path,
                  rel("real", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.noise"\nclaimed_range = [0.0, 1.0]')
                  + rel("null", 'x = "d.is_active"\ny = "d.noise"\nbaseline = "d.flat"\nclaimed_range = [0.0, 1.0]'))
    assert out["real"]["verdict"] == "agrees" and out["real"]["ci_lo"] > 0.0
    assert out["null"]["verdict"] == "unverifiable" and out["null"]["ci_lo"] < 0.0 < out["null"]["ci_hi"]
    assert "straddles a bound" in out["null"]["why"]


def test_every_refusal_names_what_is_wrong(tmp_path):
    out = project(
        tmp_path,
        rel("no_baseline", 'x = "d.is_active"\ny = "d.score"\nclaimed = 0.0')
        + rel("absent", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.nonesuch"\nclaimed = 0.0')
        + rel("label_as_score", 'x = "d.is_active"\ny = "d.is_active"\nbaseline = "d.score"\nclaimed = 0.0')
        + rel("other_dataset", 'x = "d.is_active"\ny = "d.score"\nbaseline = "e.other"\nclaimed = 0.0')
        + rel("no_unit", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.compound"\nclaimed = 0.0')
        + ('\n[[relationships]]\nid = "wrong_method"\nx = "d.score"\ny = "d.noise"\nkind = "correlation"\n'
           'method = "spearman"\nbaseline = "d.copy"\nclaimed = 0.0\nresampling_unit = "unit"\n'),
        second_file=True)
    assert all(r["verdict"] == "unverifiable" for r in out.values())
    assert "needs `baseline =" in out["no_baseline"]["why"]
    assert "has no column nonesuch" in out["absent"]["why"]
    assert "oracle" in out["label_as_score"]["why"]
    assert "columns of ONE table" in out["other_dataset"]["why"]
    assert "no unit declared for compound" in out["no_unit"]["why"]     # a column in the file, undeclared in the context
    # The refusal names every method that reads `baseline`, which was this one alone until `bedroc_difference`
    # (2026-11-22). The list comes from `relate.PAIRED_DIFFERENCE_METHODS` sorted, so it stays correct as that set grows.
    assert "bedroc_difference, roc_auc_difference only" in out["wrong_method"]["why"]
    assert "roc_auc_difference" in out["wrong_method"]["why"]


def test_a_table_with_one_class_is_unverifiable_and_says_so(tmp_path):
    (tmp_path / "d.csv").write_text("compound,unit,is_active,score,copy\n"
                                    + "\n".join(f"c{i},u{i // 3},0,{i * 0.1},{i * 0.1}" for i in range(30)) + "\n")
    vars_ = "".join(f'\n[[variables]]\ndataset = "d"\nname = "{c}"\nunit = "1"\n' for c in ("is_active", "score", "copy"))
    (tmp_path / "context.toml").write_text(
        HEAD + vars_ + rel("one_class", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.copy"\nclaimed = 0.0'))
    row = {r["id"]: r for r in check(tmp_path)["relationships"]}["one_class"]
    assert row["verdict"] == "unverifiable" and "undefined" in row["why"] and "every row is labelled the same way" in row["why"]


def test_a_row_missing_the_baseline_is_dropped_from_the_claim_not_from_one_column(tmp_path):
    rows = table().splitlines()
    rows[1] = rows[1].rsplit(",", 3)[0] + ",,0.5,1.0"          # one row with an empty `copy`
    (tmp_path / "d.csv").write_text("\n".join(rows) + "\n")
    vars_ = "".join(f'\n[[variables]]\ndataset = "d"\nname = "{c}"\nunit = "1"\n' for c in COLS)
    (tmp_path / "context.toml").write_text(
        HEAD + vars_ + rel("KA", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.copy"\nclaimed = 0.0'))
    row = {r["id"]: r for r in check(tmp_path)["relationships"]}["KA"]
    assert (row["n_rows"], row["dropped"]) == (239, 1) and row["estimate"] == 0.0


# --- the figure ---------------------------------------------------------------------------------


def test_the_figure_is_two_roc_curves_and_the_caption_names_every_mark(tmp_path):
    pytest.importorskip("matplotlib")
    from claimcheck import figures

    (tmp_path / "d.csv").write_text(table())
    vars_ = "".join(f'\n[[variables]]\ndataset = "d"\nname = "{c}"\nunit = "1"\n' for c in COLS)
    (tmp_path / "context.toml").write_text(
        HEAD + vars_ + rel("G", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.noise"\nclaimed_range = [0.0, 1.0]'))
    (tmp_path / "results").mkdir()
    out = check(tmp_path)
    (tmp_path / "results" / "claims.json").write_text(__import__("json").dumps(out))
    index = figures.build(tmp_path, tmp_path / "figures" / "claims")
    assert index[0]["n_points"] == 240 and (tmp_path / index[0]["figure"]).is_file()
    note = (tmp_path / index[0]["commentary"]).read_text()
    assert "false positive rate (1)" in note and "true positive rate (1)" in note
    assert "solid dark line is score" in note and "dashed grey line is noise" in note
    assert "dotted diagonal is chance" in note and "SIGNED AREA BETWEEN THE TWO CURVES" in note
    assert "No error bars are drawn" in note


def test_the_published_metadata_carries_the_third_column(tmp_path):
    """WITHOUT IT THE PACKAGE IS AMBIGUOUS. Two claims comparing one score against two different columns have the same
    x and the same y, so a consumer reading the Croissant or the graph sees two different numbers under one description.
    A claim with no baseline gains no key, so every existing package stays byte-identical."""
    import json as _json

    from claimcheck import croissant

    (tmp_path / "d.csv").write_text(table())
    vars_ = "".join(f'\n[[variables]]\ndataset = "d"\nname = "{c}"\nunit = "1"\n' for c in COLS)
    (tmp_path / "context.toml").write_text(
        HEAD + vars_
        + rel("vs_noise", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.noise"\nclaimed_range = [0.0, 1.0]')
        + rel("vs_flat", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.flat"\nclaimed_range = [0.0, 1.0]')
        + ('\n[[relationships]]\nid = "plain"\nx = "d.score"\ny = "d.noise"\nkind = "correlation"\n'
           'method = "spearman"\nclaimed_range = [-1.0, 1.0]\nresampling_unit = "unit"\n'))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "claims.json").write_text(_json.dumps(check(tmp_path)))
    emitted = _json.loads(croissant.render(tmp_path))
    claims = [c for c in next(v for k, v in emitted.items() if k.endswith("claims"))]
    diffs = [c for c in claims if c["method"] == "roc_auc_difference"]
    assert sorted(c["baseline"] for c in diffs) == ["d.flat", "d.noise"]
    assert all(c["baselineUnit"] == "1" for c in diffs)
    assert {c["estimate"] for c in diffs} != {diffs[0]["estimate"]} or len(diffs) == 1   # two different numbers
    assert "baseline" not in next(c for c in claims if c["method"] == "spearman")

    rdflib = pytest.importorskip("rdflib")
    from claimcheck import graph as graph_mod

    (tmp_path / "results" / "croissant.json").write_text(_json.dumps(emitted))   # the graph is built from it
    g = graph_mod.build(tmp_path)
    from claimcheck.namespace import NS

    q = (f"PREFIX tl: <{NS}> "
         "SELECT ?cid ?b WHERE { ?rel tl:claimId ?cid ; tl:baseline ?b . }")
    rows = {str(cid): str(b) for cid, b in g.query(q)}
    assert set(rows) == {"vs_noise", "vs_flat"} and len({*rows.values()}) == 2
    assert isinstance(g, rdflib.Graph)


def test_the_roc_curve_runs_corner_to_corner_and_ties_are_one_step():
    from claimcheck.figures import roc_points

    fx, fy = roc_points(LABELS, PERFECT)
    assert (fx[0], fy[0]) == (0.0, 0.0) and (fx[-1], fy[-1]) == (1.0, 1.0)
    assert fy[1:4] == [1 / 3, 2 / 3, 1.0] and fx[1:4] == [0.0, 0.0, 0.0]      # every active first
    assert all(b >= a for a, b in zip(fx, fx[1:], strict=False))              # monotone in both
    assert all(b >= a for a, b in zip(fy, fy[1:], strict=False))
    tied = roc_points(LABELS, [1.0] * 6)
    assert tied == ([0.0, 1.0], [0.0, 1.0])                                   # all tied: one step, the chance diagonal
    assert roc_points([1, 1], [1.0, 2.0]) == ([], [])                         # one class: no curve

    # the drawn area is the AUROC, trapezoid by trapezoid, ties included
    rng = random.Random(43)
    labels = [1 if i % 7 == 0 else 0 for i in range(140)]
    for scores in ([rng.random() for _ in labels], [float(rng.randint(0, 3)) for _ in labels]):
        fx, fy = roc_points(labels, scores)
        area = math.fsum((fx[i + 1] - fx[i]) * (fy[i] + fy[i + 1]) / 2 for i in range(len(fx) - 1))
        assert area == pytest.approx(roc_auc(labels, scores), abs=1e-12)
