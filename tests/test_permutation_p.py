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

import random

from claimcheck.relate import permutation_p, spearman


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
