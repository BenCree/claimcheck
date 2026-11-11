"""Two-sided tests for the roc_auc estimator (2026-11-12).

A perfect ranker returns 1.0, an inverted one 0.0, a tied one 0.5, a random one near 0.5, and a
column with only one class returns None so the claim reads `unverifiable` rather than chance. The
permutation p is centred on 0.5, which the last test pins: an uninformative score must not come
back significant, and a perfect one must.
"""
import math
import random

import pytest

from claimcheck.relate import ESTIMATORS, permutation_p, roc_auc


def test_perfect_inverted_and_tied_rankings():
    labels = [1, 1, 1, 0, 0, 0]
    assert roc_auc(labels, [9.0, 8.0, 7.0, 3.0, 2.0, 1.0]) == pytest.approx(1.0)   # every hit above every decoy
    assert roc_auc(labels, [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]) == pytest.approx(0.0)   # exactly wrong
    assert roc_auc(labels, [5.0] * 6) == pytest.approx(0.5)                        # all tied: no information


def test_one_swap_moves_it_by_one_pair():
    """3 positives and 3 negatives make 9 pairs, so one discordant pair costs exactly 1/9."""
    labels = [1, 1, 1, 0, 0, 0]
    assert roc_auc(labels, [9.0, 8.0, 2.5, 3.0, 2.0, 1.0]) == pytest.approx(8 / 9)
    # a tie between one positive and one negative is half a pair
    assert roc_auc(labels, [9.0, 8.0, 3.0, 3.0, 2.0, 1.0]) == pytest.approx(8.5 / 9)


def test_a_random_score_is_near_a_half_and_a_small_set_is_exact():
    rng = random.Random(11)
    labels = [1] * 50 + [0] * 450
    scores = [rng.random() for _ in labels]
    assert abs(roc_auc(labels, scores) - 0.5) < 0.12
    assert roc_auc([1, 0], [1.0, 0.0]) == pytest.approx(1.0)


def test_label_and_score_do_not_commute_and_the_threshold_is_the_fraction_convention():
    labels = [1, 1, 0, 0]
    scores = [4.0, 3.0, 2.0, 1.0]
    assert roc_auc(labels, scores) == pytest.approx(1.0)
    assert roc_auc(scores, labels) != pytest.approx(1.0)        # x is the label, y the score
    assert roc_auc([0.6, 0.6, 0.4, 0.4], scores) == pytest.approx(1.0)   # indicator at >= 0.5


def test_degenerate_inputs_return_none_not_a_number():
    assert roc_auc([1, 1, 1], [1.0, 2.0, 3.0]) is None          # no negatives: undefined, not 0.5
    assert roc_auc([0, 0, 0], [1.0, 2.0, 3.0]) is None          # no positives
    assert roc_auc([], []) is None
    assert roc_auc([1, 0], [1.0]) is None                        # ragged input


def test_it_is_registered_as_a_correlation_method():
    fn, kind = ESTIMATORS["roc_auc"]
    assert fn is roc_auc and kind == "correlation"


def test_permutation_p_is_centred_on_the_statistic_null():
    """Centred on 0, an AUC of 0.5 looks as extreme as any permutation and the p is wrong."""
    rng = random.Random(3)
    labels = [1] * 20 + [0] * 80
    perfect = [10.0 - i * 0.01 for i in range(20)] + [rng.uniform(0, 1) for _ in range(80)]
    pairs = list(zip(labels, perfect))
    clusters = [str(i) for i in range(len(pairs))]
    p_centred, floor = permutation_p(pairs, clusters, estimate=roc_auc, n_perm=200, centre=0.5)
    assert p_centred <= floor + 1e-12                            # a perfect ranking is significant
    uninformative = [rng.uniform(0, 1) for _ in labels]
    pu = permutation_p(list(zip(labels, uninformative)), clusters, estimate=roc_auc, n_perm=200, centre=0.5)
    assert pu[0] > 0.05                                          # chance is not significant
    # The break is an INVERTED score. AUC lives in [0, 1], so comparing magnitudes about 0 is the
    # one-sided test "is the permuted AUC at least the observed one": a strongly inverted score,
    # which is just as informative as a good one and says the sign convention is wrong, then comes
    # back unremarkable. Centred on 0.5 it is significant, which is the honest answer.
    inverted = [rng.gauss(0.0, 1.0) for _ in range(20)] + [rng.gauss(1.0, 1.0) for _ in range(80)]
    ip = list(zip(labels, inverted))
    auc = roc_auc(labels, inverted)
    assert auc < 0.35
    p_c = permutation_p(ip, clusters, estimate=roc_auc, n_perm=200, centre=0.5)[0]
    p_u = permutation_p(ip, clusters, estimate=roc_auc, n_perm=200)[0]
    assert p_c < 0.05 < p_u                                      # centred sees it, uncentred does not


def test_existing_methods_keep_their_uncentred_null():
    """The default centre is 0.0, so pearson and spearman behave exactly as before."""
    rng = random.Random(5)
    xs = [rng.gauss(0, 1) for _ in range(60)]
    ys = [x * 2 + rng.gauss(0, 0.3) for x in xs]
    pairs = list(zip(xs, ys))
    clusters = [str(i) for i in range(len(pairs))]
    a = permutation_p(pairs, clusters, n_perm=100)
    b = permutation_p(pairs, clusters, n_perm=100, centre=0.0)
    assert a == b and a[0] <= a[1] + 1e-12
    assert all(math.isfinite(v) for v in a)
