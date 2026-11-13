"""The permutation null must be centred on the data it is a null for.

REGRESSION, 2026-09-24. The first version of `permutation_p` concatenated the y
values into per-cluster blocks and laid them against x in ROW order. On a file
whose rows happen to be contiguous by cluster those two orders agree and the
null is correct, so it passed every check that existed. On an interleaved file
they do not: on a 493-row table over 176 Butina clusters the observed Spearman
was +0.502 and the unshuffled layout gave +0.044. The association was destroyed
before any shuffle, the null was a full row permutation rather than a block one,
and every p it produced was too small.

The property that catches it is the one below and it is cheap: under the
IDENTITY assignment, where every cluster donates to itself, the statistic must
come back unchanged. A test that only asked "is the p small for a strong
signal" passed the broken version, because destroying the within-cluster
structure makes a strong signal look MORE significant, not less.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest

from claimcheck.relate import PERMUTATION_TIE_TOL, as_extreme, kendall, pearson, permutation_p, roc_auc, spearman


def _interleaved(n_clusters: int = 60, per: int = 4):
    """Rows deliberately NOT contiguous by cluster, which is the failing case."""
    pairs, clusters = [], []
    for i in range(per):
        for c in range(n_clusters):
            v = c + 0.1 * i
            pairs.append((v, v))
            clusters.append(f"c{c}")
    return pairs, clusters


def _identity_assignment(pairs, clusters):
    ys = [p[1] for p in pairs]
    idx: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        idx.setdefault(c, []).append(i)
    trial = list(ys)
    for k, positions in idx.items():
        src = [ys[j] for j in positions]
        for n, pos in enumerate(positions):
            trial[pos] = src[n]
    return spearman([p[0] for p in pairs], trial)


def test_identity_assignment_preserves_the_statistic_on_interleaved_rows():
    pairs, clusters = _interleaved()
    observed = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    assert abs(observed - _identity_assignment(pairs, clusters)) < 1e-12


def test_a_real_signal_gives_a_small_p():
    rng = random.Random(0)
    clusters = [f"c{i // 5}" for i in range(400)]
    pairs = [(i, i + rng.gauss(0, 5)) for i in range(400)]
    p, floor = permutation_p(pairs, clusters, estimate=spearman)
    assert p <= 0.01
    assert floor > 0.0                      # an add-one estimator cannot return zero


def test_pure_noise_does_not():
    """The two-sided half. Without it a function returning 0.0005 always passes."""
    rng = random.Random(1)
    clusters = [f"c{i // 5}" for i in range(400)]
    pairs = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(400)]
    p, _ = permutation_p(pairs, clusters, estimate=spearman)
    assert p > 0.10


def test_too_few_clusters_returns_none_rather_than_a_number():
    pairs = [(1.0, 2.0), (2.0, 1.0), (3.0, 3.0)]
    assert permutation_p(pairs, ["a", "a", "b"], estimate=spearman) is None


# ---------------------------------------------------------------------------
# REGRESSION, 2026-11-12: a permuted statistic exactly as extreme as the observed one must count.
#
# The comparison was `abs(r - centre) >= abs(obs - centre)` in floating point. |8/9 - 0.5| is
# 0.38888888888888884 and |1/9 - 0.5| is 0.3888888888888889, so an observed AUC of 1/9 did not count
# its mirror 8/9: with 3 actives against 3 decoys the p came back 3/20 where the exact two-sided
# value is 4/20. A p that is too small. The fixtures below decide every permutation of a small
# design in exact rational arithmetic and require the float comparison to agree.
# ---------------------------------------------------------------------------

def _exact_distance_pearson(xs, ys):
    """|n * sum(xy) - sum(x) * sum(y)| as a Fraction: |r| up to a factor no permutation of y changes."""
    fx, fy = [Fraction(v) for v in xs], [Fraction(v) for v in ys]
    return abs(len(fx) * sum(a * b for a, b in zip(fx, fy, strict=True)) - sum(fx) * sum(fy))


def _average_ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [Fraction(0)] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = Fraction(i + j, 2) + 1
        i = j + 1
    return out


def _exact_distance_spearman(xs, ys):
    return _exact_distance_pearson(_average_ranks(xs), _average_ranks(ys))


def _exact_distance_kendall(xs, ys):
    """|C - D|: the tie counts in the denominator of tau-b are the same for every permutation of y."""
    s = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            a, b = xs[i] - xs[j], ys[i] - ys[j]
            s += (a > 0) * (b > 0) + (a < 0) * (b < 0) - (a > 0) * (b < 0) - (a < 0) * (b > 0)
    return abs(s)


def _exact_distance_auc(xs, ys):
    """|2U - n_pos * n_neg| in half-pairs: the distance of the AUC from 0.5 up to a constant factor."""
    pos = [y for x, y in zip(xs, ys, strict=True) if x >= 0.5]
    neg = [y for x, y in zip(xs, ys, strict=True) if x < 0.5]
    twice_u = sum(2 * (p > q) + (p == q) for p in pos for q in neg)
    return abs(twice_u - len(pos) * len(neg))


_EXACT_DESIGNS = [
    # (name, estimator, exact distance, centre, x, y). The y sums are not multiples of n, so the means are not exact in binary.
    ("pearson", pearson, _exact_distance_pearson, 0.0, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 2.0, 5.0, 7.0, 7.0, 12.0]),
    ("spearman, tied", spearman, _exact_distance_spearman, 0.0, [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0], [2.0, 5.0, 5.0, 6.0, 7.0, 9.0, 9.0]),
    ("kendall, tied", kendall, _exact_distance_kendall, 0.0, [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0], [2.0, 5.0, 5.0, 6.0, 7.0, 9.0, 9.0]),
    ("roc_auc", roc_auc, _exact_distance_auc, 0.5, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [7.0, 6.0, 4.0, 5.0, 3.0, 2.0, 1.0]),
    ("roc_auc, tied", roc_auc, _exact_distance_auc, 0.5, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0]),
]


def _classify_every_permutation(estimate, exact, centre, xs, ys):
    """All orders of y, each taken in turn as the observed one against every other: (comparisons, how many `as_extreme` gets wrong,
    how many the plain float comparison gets wrong, distinct attainable distances), judged by exact rational arithmetic."""
    values = {(estimate(xs, list(perm)), exact(xs, list(perm))) for perm in set(itertools.permutations(ys))}
    wrong = plain_wrong = 0
    for r, d_r in values:
        for obs, d_obs in values:
            truth = d_r >= d_obs
            wrong += as_extreme(r, obs, centre) != truth
            plain_wrong += (abs(r - centre) >= abs(obs - centre)) != truth
    return len(values) ** 2, wrong, plain_wrong, len({d for _, d in values})


@pytest.mark.parametrize("name,estimate,exact,centre,xs,ys", _EXACT_DESIGNS, ids=[d[0] for d in _EXACT_DESIGNS])
def test_every_permutation_is_classified_as_exact_arithmetic_classifies_it(name, estimate, exact, centre, xs, ys):
    """`as_extreme` must say "at least as far from the centre" exactly when the rational arithmetic does: a tie counted, the next
    attainable value inward not counted."""
    n, wrong, _, distinct = _classify_every_permutation(estimate, exact, centre, xs, ys)
    assert wrong == 0, f"{name}: {wrong} of {n} comparisons disagree with exact arithmetic"
    assert distinct >= 4                                       # the design has distinct attainable values to tell apart


def test_the_plain_float_comparison_does_lose_ties_so_the_tolerance_is_doing_something():
    """The other side of the fixture: without the tolerance the same enumeration misclassifies. Which methods lose ties is recorded
    in docs/screening_metrics_study.md; the one this test pins is roc_auc, whose mirror about 0.5 is not an exact sign change."""
    plain = {d[0]: _classify_every_permutation(*d[1:])[2] for d in _EXACT_DESIGNS}
    assert plain["roc_auc"] > 0 and plain["roc_auc, tied"] > 0
    assert plain["pearson"] > 0                                # centre 0.0 is affected too: equal |r| reached by different sums
    print("comparisons the plain float test gets wrong, by design:", plain)


def test_as_extreme_counts_the_mirror_and_not_the_next_value_inward():
    assert abs(8 / 9 - 0.5) != abs(1 / 9 - 0.5)                # the floating-point fact behind the defect
    assert as_extreme(8 / 9, 1 / 9, 0.5) and as_extreme(1 / 9, 8 / 9, 0.5)
    assert not as_extreme(7 / 9, 8 / 9, 0.5) and not as_extreme(2 / 9, 1 / 9, 0.5)
    # the smallest step a real screen takes is far above the tolerance: one half-pair at 7 actives among 357
    step = 0.5 / (7 * 350)
    assert step > 1e6 * PERMUTATION_TIE_TOL
    assert not as_extreme(0.8 - step, 0.8, 0.5) and as_extreme(0.2, 0.8, 0.5)
    assert as_extreme(-0.3, 0.3) and not as_extreme(0.3 - 1e-9, 0.3)


def test_three_actives_against_three_decoys_gives_the_same_p_from_auc_8_9_and_from_1_9():
    """20 arrangements; U is 0, 1, 8 or 9 in four of them, so the exact two-sided p of AUC 8/9, and of 1/9, is 0.2000. Before the
    fix the second came back 0.15."""
    labels = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    high = [6.0, 5.0, 3.0, 4.0, 2.0, 1.0]
    low = [-s for s in high]
    assert roc_auc(labels, high) == pytest.approx(8 / 9) and roc_auc(labels, low) == pytest.approx(1 / 9)
    p_high, _ = permutation_p(list(zip(labels, high, strict=True)), None, estimate=roc_auc, n_perm=20000, centre=0.5)
    p_low, _ = permutation_p(list(zip(labels, low, strict=True)), None, estimate=roc_auc, n_perm=20000, centre=0.5)
    assert p_high == p_low                                     # the same draws, mirrored: equal, not merely close
    assert p_high == pytest.approx(0.2, abs=0.01)


def test_a_score_and_its_negation_get_the_same_p_at_7_actives_among_357():
    rng = random.Random(12)
    labels = [1.0] * 7 + [0.0] * 350
    for _ in range(10):
        s = [rng.gauss(0, 1) for _ in labels]
        a = permutation_p(list(zip(labels, s, strict=True)), None, estimate=roc_auc, n_perm=200, centre=0.5)
        b = permutation_p(list(zip(labels, [-v for v in s], strict=True)), None, estimate=roc_auc, n_perm=200, centre=0.5)
        assert a == b


def test_with_one_row_per_unit_the_null_is_an_exact_permutation_of_y():
    """The donor rule draws with replacement, which with one row per unit cannot repeat a value. With several rows per unit it can,
    and does: that is the documented behaviour, pinned here so a change to it is a decision and not an accident."""
    pairs = [(float(i % 2), float(i)) for i in range(40)]
    seen = {"n": 0, "not_a_permutation": 0}

    def recording(xs, ys):
        seen["n"] += 1
        seen["not_a_permutation"] += sorted(ys) != [float(i) for i in range(40)]
        return spearman(xs, ys)

    permutation_p(pairs, None, estimate=recording, n_perm=200)
    assert seen["n"] == 201 and seen["not_a_permutation"] == 0
    seen.update(n=0, not_a_permutation=0)
    permutation_p(pairs, [f"u{i // 4}" for i in range(40)], estimate=recording, n_perm=200)
    assert seen["not_a_permutation"] > 150
