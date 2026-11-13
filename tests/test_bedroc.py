"""Two-sided tests for the bedroc estimator (2026-11-12).

BEDROC (Truchon and Bayly, J. Chem. Inf. Model. 2007, 47, 488) is early recognition: each positive is weighted by
exp(-alpha * k / N) at list position k of N. It was added beside `roc_auc` on the day three things were believed about it here and
found to be wrong or unstated, and each has a test below: that it saturates below 1 at a large active fraction (it is exactly 1.0
for a perfect ranking at any fraction), that a random ranking scores about 0 (it scores 0.0604 at 7 of 357), and that alpha can be
left to a default (it is a required claim key). RDKit's CalcBEDROC is the independent implementation.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from claimcheck.relate import (
    AGREES,
    ESTIMATORS,
    REFUTED,
    UNVERIFIABLE,
    bedroc,
    bedroc_alpha,
    bedroc_random_expectation,
    bootstrap_ci,
    check,
    permutation_p,
    roc_auc,
)

ALPHA = 20.0


def table(active_positions: list[int], n_total: int) -> tuple[list[float], list[float]]:
    """Labels and scores for a list of n_total compounds whose actives sit at the given 1-based positions, position 1 best."""
    labels = [0.0] * n_total
    for k in active_positions:
        labels[k - 1] = 1.0
    return labels, [float(n_total - i) for i in range(n_total)]


def closed_form(active_positions: list[int], n_total: int, alpha: float) -> float:
    """Truchon and Bayly's closed form in RIE, sinh and cosh, written independently of the min-max form the package uses."""
    n = len(active_positions)
    ra = n / n_total
    random_mean = sum(math.exp(-alpha * k / n_total) for k in range(1, n_total + 1)) / n_total
    rie = (sum(math.exp(-alpha * k / n_total) for k in active_positions) / n) / random_mean
    return (rie * ra * math.sinh(alpha / 2) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ra))
            + 1.0 / (1.0 - math.exp(alpha * (1.0 - ra))))


# ── the estimator ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,n_total", [(7, 357), (10, 1000), (36, 357), (100, 357), (178, 357), (300, 357), (1, 2)])
def test_a_perfect_ranking_is_exactly_one_and_the_worst_exactly_zero_at_any_active_fraction(n, n_total):
    """IT DOES NOT SATURATE. The belief that the maximum falls below 1 when actives are a large share of the set is true of RIE,
    not of BEDROC, which is RIE min-max normalised. Exactly 1.0 and 0.0, not approximately."""
    for alpha in (5.0, 20.0, 100.0):
        assert bedroc(*table(list(range(1, n + 1)), n_total), alpha) == 1.0
        assert bedroc(*table(list(range(n_total - n + 1, n_total + 1)), n_total), alpha) == 0.0


def test_it_is_the_papers_closed_form():
    rng = random.Random(1)
    for n, n_total in [(10, 1000), (7, 357), (50, 100), (3, 40)]:
        for alpha in (5.0, 20.0, 80.0):
            pos = sorted(rng.sample(range(1, n_total + 1), n))
            assert bedroc(*table(pos, n_total), alpha) == pytest.approx(closed_form(pos, n_total, alpha), abs=1e-10)


def test_it_agrees_with_rdkit_calcbedroc():
    scoring = pytest.importorskip("rdkit.ML.Scoring.Scoring")
    rng = random.Random(2)
    for n, n_total in [(10, 1000), (7, 357), (50, 100)]:
        for alpha in (5.0, 20.0, 80.0):
            pos = sorted(rng.sample(range(1, n_total + 1), n))
            labels, scores = table(pos, n_total)
            rows = [[s, int(v)] for s, v in zip(scores, labels, strict=True)]          # already sorted best first
            assert bedroc(labels, scores, alpha) == pytest.approx(scoring.CalcBEDROC(rows, 1, alpha), abs=1e-9)


def test_two_screeners_auc_cannot_tell_apart():
    """10 actives among 1,000. One screener puts 5 at the very top and 5 in the middle; the other puts all 10 a quarter of the way
    down. AUC differs by 0.002. BEDROC at alpha = 20 is 0.5250 against 0.0072, and the second is below what a random ranking scores."""
    a = table(list(range(1, 6)) + list(range(496, 501)), 1000)
    b = table(list(range(248, 258)), 1000)
    assert roc_auc(*a) == pytest.approx(7450 / 9900) and roc_auc(*b) == pytest.approx(7430 / 9900)
    assert bedroc(*a, ALPHA) == pytest.approx(0.525006, abs=5e-6)
    assert bedroc(*b, ALPHA) == pytest.approx(0.007155, abs=5e-6)
    assert bedroc(*b, ALPHA) < bedroc_random_expectation(10, 1000, ALPHA) < bedroc(*a, ALPHA)


def test_a_random_ranking_does_not_score_zero():
    """The comparison point is `bedroc_random_expectation`, not 0: 0.0604 at 7 of 357, 0.0552 at 10 of 1,000, and close to the
    active fraction when alpha times the fraction is large."""
    assert bedroc_random_expectation(7, 357, ALPHA) == pytest.approx(0.0604, abs=5e-5)
    assert bedroc_random_expectation(10, 1000, ALPHA) == pytest.approx(0.0552, abs=5e-5)
    assert bedroc_random_expectation(178, 357, ALPHA) == pytest.approx(0.4986, abs=5e-5)
    # and it is what random rankings average: 4,000 shuffles of 7 actives among 357
    rng = random.Random(3)
    labels = [1.0] * 7 + [0.0] * 350
    scores = [float(i) for i in range(357)]
    draws = []
    for _ in range(4000):
        rng.shuffle(scores)
        draws.append(bedroc(labels, scores, ALPHA))
    mean = sum(draws) / len(draws)
    sd = math.sqrt(sum((d - mean) ** 2 for d in draws) / (len(draws) - 1))
    assert abs(mean - bedroc_random_expectation(7, 357, ALPHA)) < 4 * sd / math.sqrt(len(draws))
    assert mean > 0.05                                         # two-sided: a function returning 0 for "random" fails here
    # undefined where there is nothing to rank
    assert bedroc_random_expectation(0, 10, ALPHA) is None and bedroc_random_expectation(10, 10, ALPHA) is None
    assert bedroc_random_expectation(3, 10, 0.0) is None


def test_alpha_decides_the_number_and_has_no_default():
    labels, scores = table([1, 2, 30, 200, 300], 357)
    values = [bedroc(labels, scores, a) for a in (5.0, 20.0, 80.0)]
    assert len({round(v, 6) for v in values}) == 3
    with pytest.raises(TypeError):
        bedroc(labels, scores)                                 # type: ignore[call-arg]
    for bad in (0.0, -1.0, float("nan"), float("inf"), True, "20"):
        assert bedroc(labels, scores, bad) is None             # type: ignore[arg-type]
    fn, kind = ESTIMATORS["bedroc"]
    assert kind == "correlation" and fn(labels, scores) is None    # the bare registry entry has no alpha and so no value


def test_the_weight_in_the_top_8_percent_at_alpha_20_is_0p798():
    """What alpha = 20 means, from the weights the estimator uses: 80 of 1,000 positions carry 0.798103 of the total."""
    labels_top, scores = table(list(range(1, 81)), 1000)
    w_top = sum(math.exp(-ALPHA * k / 1000) for k in range(1, 81))
    w_all = sum(math.exp(-ALPHA * k / 1000) for k in range(1, 1001))
    assert w_top / w_all == pytest.approx(0.798103, abs=1e-6)
    assert (1 - math.exp(-ALPHA * 0.08)) / (1 - math.exp(-ALPHA)) == pytest.approx(0.798103, abs=1e-6)
    for alpha, share, z80 in ((5.0, 0.332, 0.317), (20.0, 0.798, 0.0805), (80.0, 0.998, 0.0201)):      # the table in docs/bedroc.md
        assert (1 - math.exp(-alpha * 0.08)) / (1 - math.exp(-alpha)) == pytest.approx(share, abs=5e-4)
        assert -math.log(1 - 0.8 * (1 - math.exp(-alpha))) / alpha == pytest.approx(z80, abs=5e-4)
    assert bedroc(labels_top, scores, ALPHA) == 1.0


def test_an_empty_class_or_ragged_input_returns_none_not_a_number():
    assert bedroc([1.0, 1.0, 1.0], [3.0, 2.0, 1.0], ALPHA) is None
    assert bedroc([0.0, 0.0, 0.0], [3.0, 2.0, 1.0], ALPHA) is None
    assert bedroc([], [], ALPHA) is None
    assert bedroc([1.0, 0.0], [1.0], ALPHA) is None


def test_label_and_score_do_not_commute_and_higher_score_is_better():
    labels, scores = table([1, 2, 3], 30)
    assert bedroc(labels, scores, ALPHA) == 1.0
    assert bedroc(labels, [-s for s in scores], ALPHA) == 0.0   # the sign convention is the caller's to get right
    assert bedroc(scores, labels, ALPHA) != 1.0
    assert bedroc([0.6, 0.6, 0.4, 0.4], [4.0, 3.0, 2.0, 1.0], ALPHA) == 1.0   # indicator at >= 0.5, as roc_auc reads it


def test_tied_compounds_share_the_mean_weight_of_the_positions_they_occupy():
    """One active tied with one decoy at the top of four: it holds position 1 or 2 with equal chance, so the value is the mean of the
    two untied values. All tied: every ranking is equally likely, so the value is the random expectation."""
    labels = [1.0, 0.0, 0.0, 0.0]
    first = bedroc(labels, [4.0, 3.0, 2.0, 1.0], ALPHA)
    second = bedroc(labels, [3.0, 4.0, 2.0, 1.0], ALPHA)
    assert first == 1.0 and second < 1.0
    assert bedroc(labels, [4.0, 4.0, 2.0, 1.0], ALPHA) == pytest.approx((first + second) / 2, abs=1e-12)
    labels7 = [1.0] * 7 + [0.0] * 50
    assert bedroc(labels7, [5.0] * 57, ALPHA) == pytest.approx(bedroc_random_expectation(7, 57, ALPHA), abs=1e-12)
    # a duplicated row, as a bootstrap draw makes, is a tie and not an error
    assert bedroc([1.0, 1.0, 0.0, 0.0], [4.0, 4.0, 2.0, 2.0], ALPHA) == 1.0


def test_it_does_not_depend_on_the_order_of_the_rows():
    rng = random.Random(4)
    labels = [1.0] * 7 + [0.0] * 93
    scores = [round(rng.gauss(0, 1) + 0.8 * v, 1) for v in labels]       # rounded: many ties
    ref = bedroc(labels, scores, ALPHA)
    for _ in range(5):
        idx = list(range(100))
        rng.shuffle(idx)
        assert bedroc([labels[i] for i in idx], [scores[i] for i in idx], ALPHA) == pytest.approx(ref, abs=1e-12)


def test_the_bootstrap_and_the_permutation_null_take_it():
    rng = random.Random(5)
    labels = [1.0] * 10 + [0.0] * 190
    scores = [rng.gauss(2.0 * v, 1.0) for v in labels]
    pairs = list(zip(labels, scores, strict=True))
    est = bedroc_alpha(ALPHA)
    centre = bedroc_random_expectation(10, 200, ALPHA)
    lo, hi, n = bootstrap_ci(pairs, None, estimate=est)
    assert n == 200 and 0.0 <= lo < est(labels, scores) < hi <= 1.0
    p, floor = permutation_p(pairs, None, estimate=est, n_perm=300, centre=centre)
    assert p <= floor + 1e-12                                  # a real enrichment is at the floor
    noise = [rng.gauss(0.0, 1.0) for _ in labels]
    p_noise, _ = permutation_p(list(zip(labels, noise, strict=True)), None, estimate=est, n_perm=300, centre=centre)
    assert p_noise > 0.05


def test_the_p_has_almost_no_power_against_a_score_worse_than_random_and_roc_auc_has():
    """The null of BEDROC is skewed: its floor is only the random expectation (0.06 here) below its centre, so the worst possible
    ranking is not far from the centre and its p is large. This is stated in the docs; the inverted score is roc_auc's to catch."""
    labels, scores = table(list(range(351, 358)), 357)          # 7 actives in the last 7 places
    pairs = list(zip(labels, scores, strict=True))
    centre = bedroc_random_expectation(7, 357, ALPHA)
    assert bedroc(labels, scores, ALPHA) == 0.0
    p_bedroc, _ = permutation_p(pairs, None, estimate=bedroc_alpha(ALPHA), n_perm=300, centre=centre)
    p_auc, floor = permutation_p(pairs, None, estimate=roc_auc, n_perm=300, centre=0.5)
    assert p_bedroc > 0.10 and p_auc <= floor + 1e-12


# ── through `check` ──────────────────────────────────────────────────────────

VARS = ('[[variables]]\ndataset = "d"\nname = "is_hit"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "score"\nunit = "1"\n')


def verdicts(tmp_path: Path, labels: list[float], scores: list[float], rels: str) -> dict:
    lines = ["compound,is_hit,score"] + [f"c{i},{int(v)},{s}" for i, (v, s) in enumerate(zip(labels, scores, strict=True))]
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "b"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + VARS + "\n" + rels)
    return {r["id"]: r for r in check(tmp_path)["relationships"]}


def rel(rid: str, extra: str) -> str:
    return (f'[[relationships]]\nid = "{rid}"\nx = "d.is_hit"\ny = "d.score"\nkind = "correlation"\nmethod = "bedroc"\n'
            f'resampling_unit = "compound"\n{extra}\n\n')


def enriched(seed: int = 6) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    labels = [1.0] * 12 + [0.0] * 228
    return labels, [round(rng.gauss(2.5 * v, 1.0), 6) for v in labels]


def test_a_claim_without_alpha_is_unverifiable_and_says_why(tmp_path):
    labels, scores = enriched()
    out = verdicts(tmp_path, labels, scores,
                   rel("none", "claimed_range = [0.0, 1.0]") + rel("zero", "alpha = 0\nclaimed_range = [0.0, 1.0]")
                   + rel("text", 'alpha = "20"\nclaimed_range = [0.0, 1.0]'))
    for rid in ("none", "zero", "text"):
        assert out[rid]["verdict"] == UNVERIFIABLE and "alpha" in out[rid]["why"] and "estimate" not in out[rid]
    assert "none is declared" in out["none"]["why"] and "no default" in out["none"]["why"]


def test_the_row_records_alpha_the_random_expectation_and_no_r_squared(tmp_path):
    labels, scores = enriched()
    row = verdicts(tmp_path, labels, scores, rel("b20", "alpha = 20\nclaimed_range = [0.3, 1.0]"))["b20"]
    assert row["verdict"] == AGREES
    assert row["alpha"] == 20.0
    assert row["estimate"] == pytest.approx(bedroc(labels, scores, 20.0), abs=1e-6)
    assert row["random_expectation"] == pytest.approx(bedroc_random_expectation(12, 240, 20.0), abs=1e-6)
    assert row["n_actives"] == 12 and row["active_fraction"] == pytest.approx(0.05)
    assert row["alpha_times_active_fraction"] == pytest.approx(1.0)
    assert row["weight_share_top_8pct"] == pytest.approx(0.798103, abs=1e-6)
    assert row["r_squared"] is None and "not a correlation" in row["r_squared_note"]
    assert row["n_units"] == 240 and 0.0 <= row["ci_lo"] <= row["estimate"] <= row["ci_hi"] <= 1.0
    assert row["p_at_floor"] and "random-ranking expectation" in row["p_method"]


def test_two_alphas_on_one_table_are_two_different_claims(tmp_path):
    labels, scores = enriched()
    out = verdicts(tmp_path, labels, scores, rel("a5", "alpha = 5\nclaimed_range = [0.0, 1.0]") + rel("a80", "alpha = 80.0\nclaimed_range = [0.0, 1.0]"))
    assert out["a5"]["alpha"] == 5.0 and out["a80"]["alpha"] == 80.0
    assert out["a5"]["estimate"] != out["a80"]["estimate"]


def test_a_wrong_claim_is_refuted_and_an_empty_class_is_unverifiable(tmp_path):
    labels, scores = enriched()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert verdicts(tmp_path / "a", labels, scores, rel("low", "alpha = 20\nclaimed = 0.05"))["low"]["verdict"] == REFUTED
    none = verdicts(tmp_path / "b", [0.0] * 40, [float(i) for i in range(40)], rel("empty", "alpha = 20\nclaimed = 0.5"))["empty"]
    assert none["verdict"] == UNVERIFIABLE and "estimate" not in none


def test_declared_as_a_value_it_is_refused(tmp_path):
    labels, scores = enriched()
    r = ('[[relationships]]\nid = "k"\nx = "d.is_hit"\ny = "d.score"\nkind = "value"\nmethod = "bedroc"\nalpha = 20\n'
         'claimed = 0.5\nresampling_unit = "compound"\n')
    assert verdicts(tmp_path, labels, scores, r)["k"]["verdict"] == UNVERIFIABLE
