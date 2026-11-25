"""Two-sided tests for the bedroc_difference estimator (2026-11-22).

THE KNOWN ANSWER IS A COLUMN AGAINST ITSELF: exactly 0.0, with a bootstrap interval of exactly zero width. The second
half is the test of what the estimator exists for. Resampling the two columns independently would give the same 0.0
estimate and a WIDE interval, so [0, 0] on identical columns is the only cheap proof that a resample draws a row with
both of its scores.

The failing twins: a perfect ranking against its own inversion is exactly +1.0 (BEDROC does not saturate, so this holds
at any active fraction), the two columns the other way round the exact negation, and a degenerate table (one class, a
ragged row, a missing or unusable alpha, fewer than three units) returns None so the claim reads `unverifiable` rather
than 0.0, which would be "the two scores are equal", the conclusion a broken join produces.

TWO THINGS DO NOT CARRY OVER FROM `roc_auc_difference` and are what these tests are for.

1. ALPHA IS REQUIRED, HAS NO DEFAULT AND IS APPLIED TO BOTH COLUMNS. A different alpha is a different claim and gives a
   different number on the same table, and a claim with none, or with an unusable one, is `unverifiable` and says so.
2. BEDROC'S RANDOM EXPECTATION IS NOT 0.5 AND DEPENDS ON THE ACTIVE FRACTION, so neither term of the difference is
   centred on a constant. It is nonetheless the SAME constant for both columns, because it is a property of the rows and
   alpha and not of a column, so it cancels in the estimate and in every draw: the null of the difference is 0, which is
   what the swap test and the zero-centred interval assume. `test_the_random_expectation_is_shared_and_cancels` pins
   that on the numbers rather than on the algebra.

The discriminating test is `test_it_separates_two_screeners_auroc_cannot`, on the fixture in `docs/bedroc.md`: AUROC
differs by 0.002 between two screeners and BEDROC at alpha 20 by about 0.52, which is why this estimator was built
instead of reading the question off the AUROC difference.
"""
import json
import random
from pathlib import Path

import pytest

from claimcheck.relate import (
    ALPHA_METHODS,
    ESTIMATORS,
    PAIRED_DIFFERENCE_METHODS,
    _bedroc_difference_needs_alpha,
    bedroc,
    bedroc_difference,
    bedroc_difference_alpha,
    bedroc_random_expectation,
    bootstrap_ci,
    check,
    mid_ranks,
    paired_swap_p,
    roc_auc,
    roc_auc_difference,
)
from claimcheck.scrutiny import METHOD_RANGE

ALPHA = 20.0
LABELS = [1, 1, 1, 0, 0, 0]
PERFECT = [9.0, 8.0, 7.0, 3.0, 2.0, 1.0]
INVERTED = [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]


def packed(a: list[float], b: list[float]) -> list[tuple[float, float]]:
    return list(zip(a, b, strict=True))


def diff(labels, a, b, alpha: float = ALPHA):
    return bedroc_difference(labels, packed(a, b), alpha)


# --- the estimator ------------------------------------------------------------------------------


def test_a_column_against_itself_is_exactly_zero():
    """THE KNOWN ANSWER. Not approximately 0: the two BEDROCs are the same float, so the subtraction is exact."""
    assert diff(LABELS, PERFECT, PERFECT) == 0.0
    noisy = [0.3, 0.9, 0.1, 0.9, 0.4, 0.2]
    assert diff(LABELS, noisy, noisy) == 0.0
    rng = random.Random(2)
    labels = [1 if i % 11 == 0 else 0 for i in range(220)]
    y = [rng.random() + (0.5 if lab else 0.0) for lab in labels]
    assert diff(labels, y, y) == 0.0


def test_a_perfect_ranking_against_its_own_inversion_is_the_maximum():
    """+1.0 AT ANY ACTIVE FRACTION, because BEDROC is 1.0 for the best ranking and 0.0 for the worst at every one of
    them (`docs/bedroc.md`: what moves with the active fraction is RIE's ceiling, not BEDROC's)."""
    for n_pos, n_total in ((3, 6), (7, 357), (178, 357), (10, 1000)):
        labels = [1] * n_pos + [0] * (n_total - n_pos)
        best = [float(n_total - i) for i in range(n_total)]
        assert diff(labels, best, [-v for v in best]) == pytest.approx(1.0, abs=1e-12)
        assert diff(labels, [-v for v in best], best) == pytest.approx(-1.0, abs=1e-12)
        assert bedroc(labels, best, ALPHA) == pytest.approx(1.0, abs=1e-12)


def test_it_is_antisymmetric_exactly():
    rng = random.Random(5)
    labels = [1 if i % 5 == 0 else 0 for i in range(100)]
    a = [rng.random() + (0.3 if lab else 0.0) for lab in labels]
    b = [rng.random() for _ in labels]
    assert diff(labels, a, b) == -diff(labels, b, a)


def test_it_is_the_difference_of_the_two_bedrocs():
    """The definition, pinned: nothing in the packing or the normalisation changes either term."""
    rng = random.Random(7)
    labels = [1 if i % 9 == 0 else 0 for i in range(180)]
    a = [rng.random() + (0.6 if lab else 0.0) for lab in labels]
    b = [rng.random() + (0.2 if lab else 0.0) for lab in labels]
    assert diff(labels, a, b) == pytest.approx(bedroc(labels, a, ALPHA) - bedroc(labels, b, ALPHA), abs=1e-15)


def test_it_separates_two_screeners_auroc_cannot():
    """THE WHOLE REASON IT EXISTS (the fixture in docs/bedroc.md). Two screeners, 10 actives among 1,000: one puts five
    at the very top and five in the middle, the other ten in the middle. AUROC puts them 0.002 apart, which no cohort
    could resolve, and BEDROC at alpha 20 puts them about 0.52 apart. A comparison decided on the AUROC difference would
    call these two screeners the same."""
    n, cut = 1000, 247
    labels = [1 if i < 5 or 495 <= i < 500 else 0 for i in range(n)]
    # column one ranks the rows in index order, so its actives sit at list positions 1 to 5 and 496 to 500
    early = [float(n - i) for i in range(n)]
    # column two puts the same ten actives at positions 248 to 257 of ITS list, the inactives either side of them
    late = [0.0] * n
    actives = [i for i, lab in enumerate(labels) if lab]
    inactives = [i for i, lab in enumerate(labels) if not lab]
    for pos, i in enumerate(inactives[:cut] + actives + inactives[cut:]):
        late[i] = float(n - pos)
    assert roc_auc(labels, early) == pytest.approx(0.7525, abs=5e-4)        # docs/bedroc.md, both rows of its table
    assert roc_auc(labels, late) == pytest.approx(0.7505, abs=5e-4)
    assert bedroc(labels, early, ALPHA) == pytest.approx(0.5250, abs=5e-4)
    assert bedroc(labels, late, ALPHA) == pytest.approx(0.0072, abs=5e-4)
    assert abs(roc_auc_difference(labels, packed(early, late))) < 0.01      # AUROC: 0.002, unresolvable anywhere
    assert diff(labels, early, late) > 0.4                                  # BEDROC: about 0.52


def test_alpha_changes_the_number_and_is_applied_to_both_columns():
    """Two alphas on one table are two different claims. The same early-weighted column wins by more at a larger alpha,
    because more of the weight sits where it is ahead."""
    n = 400
    labels = [1 if i % 20 == 0 else 0 for i in range(n)]
    rng = random.Random(11)
    early = [(1000.0 - i) if labels[i] else rng.random() for i in range(n)]
    flat = [rng.random() for _ in range(n)]
    d5, d20, d80 = (diff(labels, early, flat, a) for a in (5.0, 20.0, 80.0))
    assert 0.0 < d5 < d20 < d80 <= 1.0
    # and alpha reaches BOTH columns: a self-difference is 0 at every alpha, which it could not be if only one were weighted
    assert all(diff(labels, early, early, a) == 0.0 for a in (5.0, 20.0, 80.0))


def test_the_random_expectation_is_shared_and_cancels():
    """WHY THE NULL IS ZERO THOUGH NEITHER TERM IS CENTRED ON A CONSTANT. `bedroc_random_expectation` depends only on the
    number of actives, the number of rows and alpha, so two columns scored on identical rows carry the same one: two
    uninformative columns sit near it individually and their difference sits near 0."""
    rng = random.Random(13)
    n, n_pos = 357, 7
    labels = [1 if i < n_pos else 0 for i in range(n)]
    rng.shuffle(labels)
    exp_ = bedroc_random_expectation(n_pos, n, ALPHA)
    assert exp_ == pytest.approx(0.0604, abs=5e-4)      # docs/bedroc.md: not 0, and not 0.5
    vals = []
    for _ in range(40):
        a = [rng.random() for _ in range(n)]
        b = [rng.random() for _ in range(n)]
        vals.append(diff(labels, a, b))
        assert bedroc(labels, a, ALPHA) is not None
    assert abs(sum(vals) / len(vals)) < 0.1             # the difference is centred on 0, not on the expectation
    # a constant column has no order at all and scores EXACTLY the random expectation, which is the clean statement of it
    assert bedroc(labels, [1.0] * n, ALPHA) == pytest.approx(exp_, abs=1e-12)
    assert diff(labels, [1.0] * n, [2.0] * n) == 0.0    # two constants: both at the expectation, difference exactly 0


def test_degenerate_inputs_return_none_not_zero():
    """None, not 0.0: a table whose join dropped every active must not read as "the two scores are equal"."""
    assert diff([1, 1, 1], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) is None          # no negatives
    assert diff([0, 0, 0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) is None          # no positives
    assert bedroc_difference([], [], ALPHA) is None
    assert bedroc_difference([1, 0], [(1.0, 2.0)], ALPHA) is None             # ragged
    assert bedroc_difference(LABELS, PERFECT, ALPHA) is None                  # two-column pairs, three-column estimator
    for bad in (None, 0, -1.0, float("inf"), float("nan"), "20", True):
        assert bedroc_difference(LABELS, packed(PERFECT, INVERTED), bad) is None, bad
    assert _bedroc_difference_needs_alpha(LABELS, packed(PERFECT, INVERTED)) is None
    assert bedroc_difference_alpha(ALPHA)(LABELS, packed(PERFECT, INVERTED)) == pytest.approx(1.0)


def test_it_is_registered_as_a_correlation_method_that_needs_alpha_and_a_baseline():
    fn, kind = ESTIMATORS["bedroc_difference"]
    assert fn is _bedroc_difference_needs_alpha and kind == "correlation"
    assert "bedroc_difference" in PAIRED_DIFFERENCE_METHODS and "bedroc_difference" in ALPHA_METHODS
    assert METHOD_RANGE["bedroc_difference"] == (-1.0, 1.0)


# --- the pairing, which is the whole point ------------------------------------------------------


def test_the_bootstrap_interval_on_identical_columns_has_zero_width():
    """THE PAIRING TEST. Had the two columns been resampled independently this interval would be wide."""
    rng = random.Random(17)
    labels = [1 if i % 10 == 0 else 0 for i in range(200)]
    y = [rng.random() + (0.5 if lab else 0.0) for lab in labels]
    units = [f"u{i // 4}" for i in range(200)]
    lo, hi, n = bootstrap_ci(packed(labels, packed(y, y)), units, estimate=bedroc_difference_alpha(ALPHA))
    assert (lo, hi) == (0.0, 0.0) and n == 50


def test_the_paired_interval_is_narrower_than_either_separate_one():
    """Two informative, strongly correlated columns: most of each BEDROC's interval is the cohort variance they share,
    and the paired difference cancels it."""
    rng = random.Random(19)
    labels = [1 if i % 8 == 0 else 0 for i in range(240)]
    a = [rng.random() + (0.8 if lab else 0.0) for lab in labels]
    b = [v + 0.01 * rng.random() for v in a]            # nearly the same ordering
    units = [f"u{i // 3}" for i in range(240)]
    est = bedroc_difference_alpha(ALPHA)
    wa = bootstrap_ci(packed(labels, a), units, estimate=lambda xs, ys: bedroc(xs, ys, ALPHA))
    wb = bootstrap_ci(packed(labels, b), units, estimate=lambda xs, ys: bedroc(xs, ys, ALPHA))
    lo, hi, _ = bootstrap_ci(packed(labels, packed(a, b)), units, estimate=est)
    assert 0.0 < (hi - lo) < min(wa[1] - wa[0], wb[1] - wb[0]) / 5
    assert lo <= 0.0 <= hi                              # and it contains zero, which is the truth here


def test_mid_ranks_do_not_change_a_bedroc():
    """What lets the swap test put two columns of different scales on one scale: BEDROC reads only the order and which
    values are tied, so a monotone tie-preserving rank map leaves it unchanged, ties included."""
    rng = random.Random(23)
    labels = [1 if i % 5 == 0 else 0 for i in range(100)]
    for scores in ([rng.random() for _ in labels],
                   [float(rng.randint(0, 4)) for _ in labels],          # heavily tied
                   [7.0] * 100):
        assert bedroc(labels, mid_ranks(scores), ALPHA) == pytest.approx(bedroc(labels, scores, ALPHA), abs=1e-12)


# --- the p --------------------------------------------------------------------------------------


def test_two_identical_columns_give_the_largest_possible_p():
    """The null is exchangeability, so two columns that are the same column can never be distinguished, however good
    they are. A label permutation would call this pair decisive, which is why that p is not the one reported."""
    rng = random.Random(29)
    labels = [1 if i % 6 == 0 else 0 for i in range(120)]
    y = [rng.random() + (0.9 if lab else 0.0) for lab in labels]
    units = [f"u{i // 2}" for i in range(120)]
    assert bedroc(labels, y, ALPHA) > 0.2
    p, floor = paired_swap_p(packed(labels, packed(y, y)), units, n_perm=200, estimate=bedroc_difference_alpha(ALPHA))
    assert p == 1.0 and floor == pytest.approx(1 / 201)


def test_a_perfect_column_against_its_inversion_sits_at_the_floor():
    rng = random.Random(31)
    labels = [1 if i % 6 == 0 else 0 for i in range(120)]
    y = [(1.0 if lab else 0.0) + 0.001 * rng.random() for lab in labels]
    units = [f"u{i // 2}" for i in range(120)]
    p, floor = paired_swap_p(packed(labels, packed(y, [-v for v in y])), units, n_perm=200,
                             estimate=bedroc_difference_alpha(ALPHA))
    assert p == pytest.approx(floor)


def test_the_swap_null_is_symmetric_about_zero():
    """Swapping every unit of a draw gives exactly the negated statistic, so a two-sided test about 0 is the right one."""
    rng = random.Random(37)
    labels = [1 if i % 4 == 0 else 0 for i in range(60)]
    a = [rng.random() + (0.4 if lab else 0.0) for lab in labels]
    b = [rng.random() for _ in labels]
    assert diff(labels, mid_ranks(a), mid_ranks(b)) == pytest.approx(-diff(labels, mid_ranks(b), mid_ranks(a)), abs=1e-12)


def test_the_p_is_undefined_below_three_units():
    pairs = packed([1, 1, 0, 0, 0, 0], packed(PERFECT, INVERTED))
    est = bedroc_difference_alpha(ALPHA)
    assert paired_swap_p(pairs, ["u1"] * 3 + ["u2"] * 3, n_perm=50, estimate=est) is None
    assert paired_swap_p(packed([1, 1, 1], [(1.0, 2.0)] * 3), None, n_perm=50, estimate=est) is None   # no negatives


def test_the_p_is_calibrated_under_exchangeable_columns():
    """MEASURED, NOT ASSERTED FROM THE CONSTRUCTION, and measured for THIS statistic rather than carried over from the
    AUROC one: two columns drawn the same way, each carrying the same relationship to the label, are exchangeable, so
    the share of p <= 0.05 should be near 0.05. 120 tables of 80 rows in 40 units, 200 draws each. THE BOUND IS LOOSE ON
    PURPOSE and is not read as a calibration figure: a p that was testing the label would reject on every table and one
    that was constant on none, and both fail this by a mile."""
    rng = random.Random(41)
    est = bedroc_difference_alpha(ALPHA)
    rejected, n_sets = 0, 120
    for _ in range(n_sets):
        labels = [1 if i % 5 == 0 else 0 for i in range(80)]
        units = [f"u{i // 2}" for i in range(80)]
        a = [rng.random() + (0.5 if lab else 0.0) for lab in labels]
        b = [rng.random() + (0.5 if lab else 0.0) for lab in labels]
        p, _ = paired_swap_p(packed(labels, packed(a, b)), units, n_perm=200, estimate=est)
        rejected += p <= 0.05
    assert rejected / n_sets < 0.18


def test_a_coarsely_tied_column_against_a_tie_free_one_is_the_approximate_case():
    """WHERE THE EXCHANGEABILITY ASSUMPTION IS WEAKEST, measured so the docs do not have to assert it. One column takes
    a few distinct values and the other is tie-free, which is heavy-atom count against a CNN affinity: a swap moves a
    tied block into positions the other column resolved, and at alpha 20 that mismatch lands where 0.798 of the weight
    sits. The rejection rate under two columns with the SAME relationship to the label is recorded here rather than
    claimed to be 0.05; no verdict depends on this p."""
    rng = random.Random(43)
    est = bedroc_difference_alpha(ALPHA)
    rejected, n_sets = 0, 80
    for _ in range(n_sets):
        labels = [1 if i % 5 == 0 else 0 for i in range(80)]
        units = [f"u{i // 2}" for i in range(80)]
        a = [rng.random() + (0.5 if lab else 0.0) for lab in labels]                   # tie-free
        b = [float(round(2 * (rng.random() + (0.5 if lab else 0.0)))) for lab in labels]   # a handful of values
        p, _ = paired_swap_p(packed(labels, packed(a, b)), units, n_perm=200, estimate=est)
        rejected += p <= 0.05
    assert rejected / n_sets < 0.35, "a tie mismatch should not make this test reject most of the time"


# --- end to end through check() -----------------------------------------------------------------


HEAD = '[project]\nname = "t"\nlicence = "CC0-1.0"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n'
OTHER = '\n[[datasets]]\nid = "e"\norigin = "computed"\nfiles = ["e.csv"]\n'
COLS = ("is_active", "score", "mirror", "copy", "noise", "flat")


def table(n: int = 240, seed: int = 47) -> str:
    rng = random.Random(seed)
    lines = ["compound,unit," + ",".join(COLS)]
    for i in range(n):
        lab = 1 if i % 6 == 0 else 0
        s = round(rng.random() + (0.9 if lab else 0.0), 6)
        lines.append(f"c{i},u{i // 3},{lab},{s},{-s},{s},{round(rng.random(), 6)},1.0")
    return "\n".join(lines) + "\n"


def project(tmp_path: Path, rels: str, second_file: bool = False) -> dict:
    (tmp_path / "d.csv").write_text(table())
    if second_file:
        (tmp_path / "e.csv").write_text("compound,other\n" + "\n".join(f"c{i},{i}" for i in range(240)) + "\n")
    vars_ = "".join(f'\n[[variables]]\ndataset = "d"\nname = "{c}"\nunit = "1"\n' for c in COLS)
    if second_file:
        vars_ += '\n[[variables]]\ndataset = "e"\nname = "other"\nunit = "1"\n'
    (tmp_path / "context.toml").write_text(HEAD + (OTHER if second_file else "") + vars_ + "\n" + rels)
    return {r["id"]: r for r in check(tmp_path)["relationships"]}


def rel(rid: str, body: str, alpha: str = "alpha = 20\n") -> str:
    return (f'\n[[relationships]]\nid = "{rid}"\nkind = "correlation"\nmethod = "bedroc_difference"\n'
            f'resampling_unit = "unit"\n{alpha}{body}\n')


def test_the_known_answer_agrees_with_a_zero_width_interval_and_the_row_carries_alpha(tmp_path):
    row = project(tmp_path, rel("KA", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.copy"\nclaimed = 0.0'))["KA"]
    assert row["verdict"] == "agrees"
    assert (row["estimate"], row["ci_lo"], row["ci_hi"]) == (0.0, 0.0, 0.0)
    assert row["bedroc_y"] == row["bedroc_baseline"] > 0.0
    assert row["alpha"] == 20.0 and row["weight_share_top_8pct"] == pytest.approx(0.798103, abs=1e-5)
    assert row["baseline"] == "d.copy" and row["baseline_unit"] == "1"
    assert row["n_actives"] == 40 and row["active_fraction"] == pytest.approx(40 / 240, abs=1e-6)
    assert row["random_expectation"] == pytest.approx(
        bedroc_random_expectation(40, 240, 20.0), abs=1e-6)
    assert "SHARED BY BOTH" in row["random_expectation_note"] and "cancels" in row["random_expectation_note"]
    assert row["r_squared"] is None and "not a share of anything" in row["r_squared_note"]
    assert "INTERCHANGEABLE" in row["p_method"] and "CENTRED ON 0" in row["p_method"] and row["p_value"] == 1.0
    assert "recomputes BOTH BEDROCs at alpha = 20" in row["interval_source"]
    assert row["n_rows"] == 240 and row["dropped"] == 0 and row["n_units"] == 80


def test_the_mirror_is_refuted_against_a_claim_of_zero(tmp_path):
    """The failing twin at the level of a verdict: a score against its own negation is not a null difference."""
    row = project(tmp_path, rel("M", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.mirror"\nclaimed = 0.0'))["M"]
    assert row["verdict"] == "refuted" and row["estimate"] > 0.1
    assert row["bedroc_y"] > row["bedroc_baseline"]


def test_a_claimed_range_straddling_zero_is_unverifiable_and_one_above_it_agrees(tmp_path):
    """The two verdicts this estimator was built to tell apart, on one table."""
    out = project(tmp_path,
                  rel("real", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.noise"\nclaimed_range = [0.0, 1.0]')
                  + rel("null", 'x = "d.is_active"\ny = "d.noise"\nbaseline = "d.flat"\nclaimed_range = [0.0, 1.0]'))
    assert out["real"]["verdict"] == "agrees" and out["real"]["ci_lo"] > 0.0
    assert out["null"]["verdict"] == "unverifiable" and out["null"]["ci_lo"] < 0.0 < out["null"]["ci_hi"]
    assert "straddles a bound" in out["null"]["why"]


def test_two_alphas_on_one_table_are_two_claims_with_two_numbers(tmp_path):
    out = project(tmp_path,
                  rel("a5", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.noise"\nclaimed_range = [0.0, 1.0]',
                      alpha="alpha = 5\n")
                  + rel("a80", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.noise"\nclaimed_range = [0.0, 1.0]',
                        alpha="alpha = 80\n"))
    assert out["a5"]["alpha"] == 5.0 and out["a80"]["alpha"] == 80.0
    assert out["a5"]["estimate"] != out["a80"]["estimate"]
    assert out["a5"]["weight_share_top_8pct"] < out["a80"]["weight_share_top_8pct"]


def test_every_refusal_names_what_is_wrong(tmp_path):
    out = project(
        tmp_path,
        rel("no_alpha", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.copy"\nclaimed = 0.0', alpha="")
        + rel("bad_alpha", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.copy"\nclaimed = 0.0', alpha='alpha = "20"\n')
        + rel("zero_alpha", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.copy"\nclaimed = 0.0', alpha="alpha = 0\n")
        + rel("no_baseline", 'x = "d.is_active"\ny = "d.score"\nclaimed = 0.0')
        + rel("absent", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.nonesuch"\nclaimed = 0.0')
        + rel("label_as_score", 'x = "d.is_active"\ny = "d.is_active"\nbaseline = "d.score"\nclaimed = 0.0')
        + rel("other_dataset", 'x = "d.is_active"\ny = "d.score"\nbaseline = "e.other"\nclaimed = 0.0')
        + rel("no_unit", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.compound"\nclaimed = 0.0')
        + ('\n[[relationships]]\nid = "wrong_kind"\nx = "d.is_active"\ny = "d.score"\nbaseline = "d.copy"\n'
           'kind = "value"\nmethod = "bedroc_difference"\nalpha = 20\nclaimed = 0.0\nresampling_unit = "unit"\n'),
        second_file=True)
    assert all(r["verdict"] == "unverifiable" for r in out.values())
    for rid in ("no_alpha", "bad_alpha", "zero_alpha"):
        assert "alpha" in out[rid]["why"] and "estimate" not in out[rid], rid
    assert "none is declared" in out["no_alpha"]["why"] and "no default" in out["no_alpha"]["why"]
    assert "applied identically to both score columns" in out["no_alpha"]["why"]
    assert "needs `baseline =" in out["no_baseline"]["why"] and "one BEDROC and nothing to subtract" in out["no_baseline"]["why"]
    assert "has no column nonesuch" in out["absent"]["why"]
    assert "oracle" in out["label_as_score"]["why"]
    assert "columns of ONE table" in out["other_dataset"]["why"]
    assert "no unit declared for compound" in out["no_unit"]["why"]
    assert "computes a correlation" in out["wrong_kind"]["why"]


def test_baseline_on_another_method_names_both_paired_methods(tmp_path):
    """`baseline` is refused where no method reads it, and the refusal now has to name two methods, not one."""
    out = project(tmp_path, '\n[[relationships]]\nid = "s"\nx = "d.score"\ny = "d.noise"\nkind = "correlation"\n'
                            'method = "spearman"\nbaseline = "d.copy"\nclaimed = 0.0\nresampling_unit = "unit"\n')
    assert out["s"]["verdict"] == "unverifiable"
    assert "bedroc_difference, roc_auc_difference only" in out["s"]["why"]


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


def test_the_auroc_difference_is_untouched_by_this_addition(tmp_path):
    """The neighbour this was built beside keeps its own row keys, its own p and its own figure: a shared `baseline`
    block and a shared `paired_swap_p` must not have moved anything in it."""
    row = project(tmp_path, '\n[[relationships]]\nid = "A"\nx = "d.is_active"\ny = "d.score"\nbaseline = "d.copy"\n'
                            'kind = "correlation"\nmethod = "roc_auc_difference"\nclaimed = 0.0\n'
                            'resampling_unit = "unit"\n')["A"]
    assert row["verdict"] == "agrees" and (row["estimate"], row["ci_lo"], row["ci_hi"]) == (0.0, 0.0, 0.0)
    assert row["auc_y"] == row["auc_baseline"] > 0.5 and "alpha" not in row
    assert "recomputes BOTH AUROCs" in row["interval_source"] and "bedroc_y" not in row


# --- the figure ---------------------------------------------------------------------------------


def test_the_curve_ends_at_the_bedroc_and_the_gap_is_the_estimate():
    from claimcheck.figures import bedroc_curve

    rng = random.Random(53)
    labels = [1 if i % 7 == 0 else 0 for i in range(140)]
    a = [rng.random() + (0.7 if lab else 0.0) for lab in labels]
    b = [rng.random() for _ in labels]
    fx, fy = bedroc_curve(labels, a, ALPHA)
    gx, gy = bedroc_curve(labels, b, ALPHA)
    assert fx[0] == 0.0 and fx[-1] == pytest.approx(1.0)
    assert fy[0] == pytest.approx(0.0, abs=1e-6)                      # -S_min / (S_max - S_min), 0 to within rounding
    assert fy[-1] == pytest.approx(bedroc(labels, a, ALPHA), abs=1e-12)
    assert gy[-1] == pytest.approx(bedroc(labels, b, ALPHA), abs=1e-12)
    assert fy[-1] - gy[-1] == pytest.approx(diff(labels, a, b), abs=1e-12)   # THE GAP AT THE EDGE IS THE CLAIM
    assert all(v2 >= v1 - 1e-12 for v1, v2 in zip(fy, fy[1:], strict=False))  # monotone: weight only accumulates
    assert all(x2 > x1 for x1, x2 in zip(fx, fx[1:], strict=False))
    # the chance curve ends at the random expectation and is not a straight line
    cx, cy = bedroc_curve(labels, a, ALPHA, expected=True)
    assert cy[-1] == pytest.approx(bedroc_random_expectation(20, 140, ALPHA), abs=1e-9)
    mid = cy[len(cy) // 2]
    assert mid > 0.9 * cy[-1]                                          # concave: most of the weight is early
    assert bedroc_curve([1, 1], [1.0, 2.0], ALPHA) == ([], [])         # one class: no curve
    assert bedroc_curve(labels, a, 0.0) == ([], [])                    # no alpha: no curve
    # all tied: one step, and the value is the random expectation
    tx, ty = bedroc_curve(labels, [1.0] * 140, ALPHA)
    assert tx == [0.0, 1.0] and ty[-1] == pytest.approx(bedroc_random_expectation(20, 140, ALPHA), abs=1e-12)


def test_the_figure_is_two_weighted_curves_and_the_caption_names_every_mark(tmp_path):
    pytest.importorskip("matplotlib")
    from claimcheck import figures

    (tmp_path / "d.csv").write_text(table())
    vars_ = "".join(f'\n[[variables]]\ndataset = "d"\nname = "{c}"\nunit = "1"\n' for c in COLS)
    (tmp_path / "context.toml").write_text(
        HEAD + vars_ + rel("G", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.noise"\nclaimed_range = [0.0, 1.0]'))
    (tmp_path / "results").mkdir()
    out = check(tmp_path)
    (tmp_path / "results" / "claims.json").write_text(json.dumps(out))
    index = figures.build(tmp_path, tmp_path / "figures" / "claims")
    assert index[0]["n_points"] == 240 and (tmp_path / index[0]["figure"]).is_file()
    note = (tmp_path / index[0]["commentary"]).read_text()
    assert "fraction of the ranked list screened (1)" in note and "weighted share of the actives found (1)" in note
    assert "solid dark line is score" in note and "dashed grey line is noise" in note
    assert "exp(-alpha * k / N) with alpha = 20" in note
    assert "chance ranking, which is a CURVE and not a diagonal" in note
    assert "VERTICAL GAP BETWEEN THE TWO CURVES AT THE RIGHT-HAND EDGE" in note
    assert "No error bars are drawn" in note
    assert "false positive rate" not in note, "the ROC caption must not be reused for a BEDROC claim"
    assert "y = x" not in note


def test_the_published_metadata_carries_the_third_column(tmp_path):
    """The baseline reaches the Croissant and the graph for this method too: they key on the row's `baseline`, so this
    is a check that the row carries it, which is what two claims differing only in the comparison column need."""
    from claimcheck import croissant

    (tmp_path / "d.csv").write_text(table())
    vars_ = "".join(f'\n[[variables]]\ndataset = "d"\nname = "{c}"\nunit = "1"\n' for c in COLS)
    (tmp_path / "context.toml").write_text(
        HEAD + vars_
        + rel("vs_noise", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.noise"\nclaimed_range = [0.0, 1.0]')
        + rel("vs_flat", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.flat"\nclaimed_range = [0.0, 1.0]'))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "claims.json").write_text(json.dumps(check(tmp_path)))
    emitted = json.loads(croissant.render(tmp_path))
    claims = next(v for k, v in emitted.items() if k.endswith("claims"))
    diffs = [c for c in claims if c["method"] == "bedroc_difference"]
    assert sorted(c["baseline"] for c in diffs) == ["d.flat", "d.noise"]
    assert all(c["baselineUnit"] == "1" for c in diffs)

    pytest.importorskip("rdflib")
    from claimcheck import graph as graph_mod
    from claimcheck.namespace import NS

    (tmp_path / "results" / "croissant.json").write_text(json.dumps(emitted))
    g = graph_mod.build(tmp_path)
    q = f"PREFIX tl: <{NS}> SELECT ?cid ?b WHERE {{ ?rel tl:claimId ?cid ; tl:baseline ?b . }}"
    rows = {str(cid): str(b) for cid, b in g.query(q)}
    assert set(rows) == {"vs_noise", "vs_flat"} and len({*rows.values()}) == 2


def test_scrutiny_does_not_flag_a_self_difference_known_answer(tmp_path):
    """The known answer passes with a zero-width interval at 0.0, which is INTERIOR to [-1, 1], so `cannot_fail` must
    not fire on it. Registered as a point claim for exactly that reason."""
    from claimcheck import scrutiny

    (tmp_path / "d.csv").write_text(table())
    vars_ = "".join(f'\n[[variables]]\ndataset = "d"\nname = "{c}"\nunit = "1"\n' for c in COLS)
    (tmp_path / "context.toml").write_text(
        HEAD + vars_ + rel("KA", 'x = "d.is_active"\ny = "d.score"\nbaseline = "d.copy"\nclaimed = 0.0'))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "claims.json").write_text(json.dumps(check(tmp_path)))
    warnings = scrutiny.audit(tmp_path)["warnings"]
    assert [w for w in warnings if w["claim"] == "KA"] == []
