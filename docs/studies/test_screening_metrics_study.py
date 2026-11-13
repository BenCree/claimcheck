"""Simulations behind the screening estimators (`roc_auc`, `bedroc`, `topn_recovery`) and the permutation p, 2026-11-12.

A STUDY, NOT PART OF THE TEST SUITE. These establish general statistical facts that the documentation quotes; they are not unit
tests of claimcheck, and they take about two minutes, so they live here and not under `tests/`. `pytest` at the repository root does
not collect this directory. Run them from the repository root, with numpy, scipy and (for one cross-check) RDKit installed:

    python -m pytest -rP docs/studies/test_screening_metrics_study.py

Every test asserts the number it found, so a changed result fails, and prints its table (`-rP` shows it). The write-up, with the
tables as last run, is docs/screening_metrics_study.md.

Conventions. Position 1 is the best compound. N compounds, n actives, Ra = n / N. BEDROC is the Truchon and Bayly definition
(J. Chem. Inf. Model. 2007, 47, 488), written here twice with numpy, independently of `claimcheck.relate.bedroc`, and checked
against it and against RDKit. Tied compounds share the mean weight of the positions they occupy. Every generator is seeded.
"""
import itertools
import math
import random
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from claimcheck import relate

ALPHA = 20.0



# ------------------------------------------------------------------------------------------------
# helpers, written independently of claimcheck
# ------------------------------------------------------------------------------------------------
def auc_from_ranks(active_ranks, n_total):
    """AUC from the 1-based ranks of the actives (rank 1 best), no ties."""
    r = np.asarray(active_ranks, dtype=float)
    n = len(r)
    decoys_above = r.sum() - n * (n + 1) / 2          # decoys ranked better than each active, summed
    return 1.0 - decoys_above / (n * (n_total - n))


def auc_mean_fraction_outranked(labels, scores):
    """Mean over actives of the share of decoys the active outscores, a tie counting a half."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    act, dec = scores[labels == 1], scores[labels == 0]
    return float(np.mean([(np.sum(a > dec) + 0.5 * np.sum(a == dec)) / len(dec) for a in act]))


def position_weights(scores, alpha):
    """exp(-alpha * k / N) for position k = 1..N in descending score order, tied compounds sharing
    the mean weight of the positions they occupy. Returned in the input order."""
    scores = np.asarray(scores, dtype=float)
    n_total = len(scores)
    order = np.argsort(-scores, kind="stable")
    w_sorted = np.exp(-alpha * np.arange(1, n_total + 1) / n_total)
    _, inv, counts = np.unique(-scores[order], return_inverse=True, return_counts=True)
    w_tied = (np.bincount(inv, weights=w_sorted) / counts)[inv]
    out = np.empty(n_total)
    out[order] = w_tied
    return out


def rie_from_weight_sum(w_sum, n, n_total, alpha):
    random_mean = (1.0 / n_total) * np.sum(np.exp(-alpha * np.arange(1, n_total + 1) / n_total))
    return (w_sum / n) / random_mean


def bedroc_closed_form(w_sum, n, n_total, alpha):
    ra = n / n_total
    rie = rie_from_weight_sum(w_sum, n, n_total, alpha)
    return (rie * ra * math.sinh(alpha / 2) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ra))
            + 1.0 / (1.0 - math.exp(alpha * (1.0 - ra))))


def bedroc_minmax(w_sum, n, n_total, alpha):
    k = np.arange(1, n_total + 1)
    w = np.exp(-alpha * k / n_total)
    rie = rie_from_weight_sum(w_sum, n, n_total, alpha)
    rie_max = rie_from_weight_sum(w[:n].sum(), n, n_total, alpha)
    rie_min = rie_from_weight_sum(w[-n:].sum(), n, n_total, alpha)
    return (rie - rie_min) / (rie_max - rie_min)


def bedroc_from_ranks(active_ranks, n_total, alpha=ALPHA, form=bedroc_closed_form):
    r = np.asarray(active_ranks, dtype=float)
    return form(np.exp(-alpha * r / n_total).sum(), len(r), n_total, alpha)


def bedroc_random_expectation(n, n_total, alpha=ALPHA):
    """BEDROC at RIE = 1, the expectation under a uniformly random ranking."""
    ra = n / n_total
    return (ra * math.sinh(alpha / 2) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ra))
            + 1.0 / (1.0 - math.exp(alpha * (1.0 - ra))))


def labels_scores_from_ranks(active_ranks, n_total):
    labels = [0] * n_total
    for r in active_ranks:
        labels[r - 1] = 1
    scores = [float(n_total - k) for k in range(n_total)]      # position k (0-based) has score N - k
    return labels, scores


def test_helpers_two_bedroc_forms_agree_and_match_rdkit():
    rng = np.random.default_rng(1)
    for n_total, n in [(1000, 10), (357, 7), (100, 50)]:
        ranks = np.sort(rng.choice(np.arange(1, n_total + 1), size=n, replace=False))
        a = bedroc_from_ranks(ranks, n_total, form=bedroc_closed_form)
        b = bedroc_from_ranks(ranks, n_total, form=bedroc_minmax)
        assert a == pytest.approx(b, abs=1e-12)
        scoring = pytest.importorskip("rdkit.ML.Scoring.Scoring")
        rows = [[0.0, 0] for _ in range(n_total)]
        for r in ranks:
            rows[r - 1][1] = 1
        assert scoring.CalcBEDROC(rows, 1, ALPHA) == pytest.approx(a, abs=1e-9)


# ------------------------------------------------------------------------------------------------
# what alpha = 20 weights
# ------------------------------------------------------------------------------------------------
def test_share_of_weight_in_the_top_8_percent_at_alpha_20():
    continuous = (1 - math.exp(-ALPHA * 0.08)) / (1 - math.exp(-ALPHA))
    assert continuous == pytest.approx(0.798103, abs=1e-6)
    assert 1 - math.exp(-1.6) == pytest.approx(0.798103, abs=1e-6)       # the normalisation 1 - e^-20 changes the 9th decimal
    z80 = -math.log(1 - 0.8 * (1 - math.exp(-ALPHA))) / ALPHA             # fraction of the list holding exactly 80%
    assert z80 == pytest.approx(0.080472, abs=1e-6)
    discrete = {}
    for n_total in (357, 1000):
        w = np.exp(-ALPHA * np.arange(1, n_total + 1) / n_total)
        top = int(math.floor(0.08 * n_total))
        discrete[n_total] = w[:top].sum() / w.sum()
    assert discrete[1000] == pytest.approx(0.798103, abs=1e-6)            # 80 of 1000 is exactly 8%, geometric sum
    assert discrete[357] == pytest.approx(0.791669, abs=1e-6)             # 28 of 357 is 7.84%
    print(f"weight in top 8% = {continuous:.6f} (continuous); 80% is reached at {100 * z80:.4f}% of the list; "
          f"N = 1000, top 80: {discrete[1000]:.6f}; N = 357, top 28 (7.84%): {discrete[357]:.6f}")


# ------------------------------------------------------------------------------------------------
# the sd of BEDROC and the 1/sqrt(8n) planning figure
# ------------------------------------------------------------------------------------------------
def test_one_over_root_8n_reproduces_the_papers_quoted_maxima_with_n_the_number_of_actives():
    quoted = {10: 0.11, 50: 0.05, 100: 0.035, 200: 0.025}                 # Truchon and Bayly, "Planning an Evaluation"
    for n, sd in quoted.items():
        assert 1 / math.sqrt(8 * n) == pytest.approx(sd, abs=0.002)
    assert 1 / math.sqrt(8 * 7) == pytest.approx(0.1336, abs=1e-4)
    assert 1 / math.sqrt(8 * 357) == pytest.approx(0.0187, abs=1e-4)      # what it would be were n the number of compounds
    print(f"1/sqrt(8n) at n = 10, 50, 100, 200 actives: {[round(1 / math.sqrt(8 * n), 4) for n in quoted]}; "
          f"paper quotes {list(quoted.values())}; at n = 7: {1 / math.sqrt(56):.4f}")


def _sample_exponential_ranks(rng, lam, n, n_total, n_samples):
    """n distinct ranks per sample with P(rank k) proportional to exp(-lam * k / N), by the Gumbel
    top-n method (successive sampling without replacement)."""
    logp = -lam * np.arange(1, n_total + 1) / n_total
    keys = logp[None, :] + rng.gumbel(size=(n_samples, n_total))
    return np.argpartition(-keys, n - 1, axis=1)[:, :n] + 1


def _bedroc_many(ranks, n_total, alpha=ALPHA):
    w_sum = np.exp(-alpha * ranks / n_total).sum(axis=1)
    n = ranks.shape[1]
    ra = n / n_total
    random_mean = np.exp(-alpha * np.arange(1, n_total + 1) / n_total).mean()
    rie = (w_sum / n) / random_mean
    return (rie * ra * math.sinh(alpha / 2) / (math.cosh(alpha / 2) - math.cosh(alpha / 2 - alpha * ra))
            + 1.0 / (1.0 - math.exp(alpha * (1.0 - ra))))


def _auc_many(ranks, n_total):
    n = ranks.shape[1]
    return 1.0 - (ranks.sum(axis=1) - n * (n + 1) / 2) / (n * (n_total - n))


SD_LAMBDAS = (0.001, 2.0, 5.0, 10.0, 12.4, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0)


def test_sd_of_bedroc_at_the_planned_design_under_the_papers_exponential_model():
    """7 actives among 357, alpha = 20, actives placed with probability proportional to exp(-lambda x).
    The sampling sd of BEDROC over 20,000 simulated screens per lambda, against 1/sqrt(8 * 7)."""
    rng = np.random.default_rng(4)
    bound = 1 / math.sqrt(8 * 7)
    rows = []
    for lam in SD_LAMBDAS:
        ranks = _sample_exponential_ranks(rng, lam, 7, 357, 20000)
        b = _bedroc_many(ranks, 357)
        a = _auc_many(ranks, 357)
        rows.append((lam, b.mean(), b.std(ddof=1), a.mean(), a.std(ddof=1)))
    print("n = 7, N = 357, alpha = 20; 20,000 simulated screens per lambda; 1/sqrt(8n) = %.4f, 1/sqrt(12n) = %.4f"
          % (bound, 1 / math.sqrt(84)))
    print("  lambda   mean BEDROC   sd BEDROC   sd*sqrt(n)   mean AUC   sd AUC")
    for lam, bm, bs, am, a_s in rows:
        print(f"  {lam:6.3f}   {bm:10.4f}   {bs:9.4f}   {bs * math.sqrt(7):9.4f}   {am:8.4f}   {a_s:7.4f}")
    worst = max(r[2] for r in rows)
    print(f"  largest sd of BEDROC over the lambda grid = {worst:.4f} ({worst * math.sqrt(7):.4f}/sqrt(n))")
    assert worst < bound                                                   # inside the family the fit was made on
    assert worst == pytest.approx(0.127, abs=0.004)


def test_one_over_root_8n_is_not_a_bound_outside_the_exponential_family():
    """A screener that puts each active at the very top or the very bottom with probability one half
    (five at the top, five in the middle is the same shape) has a per-active term of 0 or 1, so sd = 1/sqrt(4n) > 1/sqrt(8n).
    1/sqrt(8n) is a fit to the largest sd seen in simulations of one family, not a mathematical bound;
    the bound for any metric that is a mean of n independent terms in [0, 1] is 1/sqrt(4n)."""
    rng = np.random.default_rng(5)
    n, n_total = 7, 357
    k_top = rng.binomial(n, 0.5, size=20000)
    vals = np.array([bedroc_from_ranks(list(range(1, k + 1)) + list(range(n_total - (n - k) + 1, n_total + 1)), n_total)
                     for k in range(n + 1)])
    sd = vals[k_top].std(ddof=1)
    print(f"top-or-bottom screener, n = 7 of 357: sd BEDROC = {sd:.4f}; 1/sqrt(8n) = {1 / math.sqrt(56):.4f}; "
          f"1/sqrt(4n) = {1 / math.sqrt(28):.4f}")
    assert sd > 1 / math.sqrt(8 * n)
    assert sd == pytest.approx(1 / math.sqrt(4 * n), abs=0.01)


# ------------------------------------------------------------------------------------------------
# BEDROC does not saturate; what the active fraction does move
# ------------------------------------------------------------------------------------------------
def test_bedroc_of_a_perfect_ranking_is_exactly_1_at_any_active_fraction():
    print("perfect and worst rankings, alpha = 20")
    print("   n     N      Ra   alpha*Ra  BEDROC_max  BEDROC_min  RIE_max  E[BEDROC | random]  AUC_max")
    for n, n_total in [(7, 357), (10, 1000), (36, 357), (100, 357), (178, 357), (300, 357)]:
        top = list(range(1, n + 1))
        bottom = list(range(n_total - n + 1, n_total + 1))
        for form in (bedroc_closed_form, bedroc_minmax):
            assert bedroc_from_ranks(top, n_total, form=form) == pytest.approx(1.0, abs=1e-9)
            assert bedroc_from_ranks(bottom, n_total, form=form) == pytest.approx(0.0, abs=1e-9)
        rie_max = rie_from_weight_sum(np.exp(-ALPHA * np.arange(1, n + 1) / n_total).sum(), n, n_total, ALPHA)
        ra = n / n_total
        assert rie_max == pytest.approx((1 - math.exp(-ALPHA * ra)) / (ra * (1 - math.exp(-ALPHA))), abs=1e-9)
        print(f"  {n:3d}  {n_total:5d}  {ra:.4f}  {ALPHA * ra:7.3f}  {bedroc_from_ranks(top, n_total):10.6f}  "
              f"{bedroc_from_ranks(bottom, n_total):10.6f}  {rie_max:7.3f}  {bedroc_random_expectation(n, n_total):12.4f}"
              f"  {auc_from_ranks(top, n_total):12.1f}")
    scoring = pytest.importorskip("rdkit.ML.Scoring.Scoring")
    rows = [[0.0, 1 if k < 7 else 0] for k in range(357)]
    assert scoring.CalcBEDROC(rows, 1, ALPHA) == pytest.approx(1.0, abs=1e-9)
    # what does depend on Ra: the ceiling of RIE and the value a random ranking scores
    assert rie_from_weight_sum(np.exp(-ALPHA * np.arange(1, 8) / 357).sum(), 7, 357, ALPHA) == pytest.approx(16.544, abs=0.001)
    assert bedroc_random_expectation(7, 357) == pytest.approx(0.0604, abs=5e-5)
    assert bedroc_random_expectation(178, 357) == pytest.approx(0.4986, abs=5e-5)


def test_what_the_active_fraction_does_change_at_7_of_357():
    """The maximum is 1.0 at 7 of 357. This records the BEDROC of the best ranking that is not perfect
    (ranks 1 to 6 and 8), so the top of the scale is not a plateau: one swap is visible."""
    n_total = 357
    one_swap = bedroc_from_ranks([1, 2, 3, 4, 5, 6, 8], n_total)
    assert one_swap == pytest.approx(0.99346, abs=5e-6)
    print(f"7 of 357, alpha*Ra = {ALPHA * 7 / 357:.3f}; BEDROC perfect = 1, with rank 7 -> 8: {one_swap:.5f}; "
          f"values attainable: C(357, 7) rankings, top value exactly 1.0")


# ------------------------------------------------------------------------------------------------
# bootstrap interval widths of AUC and BEDROC at 7 actives
# ------------------------------------------------------------------------------------------------
def _auc_and_bedroc_tied(labels, scores, alpha=ALPHA):
    """AUC (average ranks) and BEDROC (tied compounds share the mean weight) of one list; None when a
    class is empty, as claimcheck's roc_auc returns."""
    n = int(labels.sum())
    n_total = len(labels)
    if n == 0 or n == n_total:
        return None
    asc = stats.rankdata(scores, method="average")
    auc = (asc[labels == 1].sum() - n * (n + 1) / 2) / (n * (n_total - n))
    w = position_weights(scores, alpha)
    return auc, bedroc_closed_form(w[labels == 1].sum(), n, n_total, alpha)


def _bootstrap_widths(rng, labels, scores, n_boot):
    """95% percentile interval widths for AUC and BEDROC, resampling compounds with replacement and
    dropping draws with no active, which is what relate.bootstrap_ci does with one row per unit."""
    n_total = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_total, size=n_total)
        got = _auc_and_bedroc_tied(labels[idx], scores[idx])
        if got is not None:
            vals.append(got)
    v = np.array(vals)
    lo, hi = np.percentile(v, [2.5, 97.5], axis=0)
    return hi - lo


WIDTH_N_DATASETS, WIDTH_N_BOOT = 120, 300
WIDTH_MODELS = [("binormal", d) for d in (0.0, 0.5, 0.954, 1.5, 1.812, 2.5, 3.5)] + \
            [("exp-rank", lam) for lam in (0.001, 5.0, 10.0, 20.0, 40.0, 100.0)]


def _width_dataset(rng, kind, par, n=7, n_total=357):
    labels = np.zeros(n_total, dtype=int)
    if kind == "binormal":                     # decoy score N(0, 1), active score N(d, 1); AUC = Phi(d / sqrt 2)
        labels[:n] = 1
        scores = rng.normal(size=n_total) + par * labels
    else:                                      # active ranks drawn with P(k) proportional to exp(-lambda k / N)
        ranks = _sample_exponential_ranks(rng, par, n, n_total, 1)[0]
        labels[ranks - 1] = 1
        scores = -np.arange(1, n_total + 1, dtype=float)
    return labels, scores


@pytest.fixture(scope="module")
def width_table():
    rng = np.random.default_rng(6)
    chance_b = bedroc_random_expectation(7, 357)
    table = []
    for kind, par in WIDTH_MODELS:
        est, widths = [], []
        for _ in range(WIDTH_N_DATASETS):
            labels, scores = _width_dataset(rng, kind, par)
            est.append(_auc_and_bedroc_tied(labels, scores))
            widths.append(_bootstrap_widths(rng, labels, scores, WIDTH_N_BOOT))
        est, widths = np.array(est), np.array(widths)
        table.append({
            "model": kind, "par": par,
            "auc_mean": est[:, 0].mean(), "bed_mean": est[:, 1].mean(),
            "auc_sd": est[:, 0].std(ddof=1), "bed_sd": est[:, 1].std(ddof=1),
            "auc_w": float(np.median(widths[:, 0])), "bed_w": float(np.median(widths[:, 1])),
            "share_bed_wider": float(np.mean(widths[:, 1] > widths[:, 0])),
            # width as a share of the distance from chance to a perfect ranking: 0.5 for AUC, 1 - chance for BEDROC
            "auc_w_rel": float(np.median(widths[:, 0])) / 0.5,
            "bed_w_rel": float(np.median(widths[:, 1])) / (1 - chance_b),
        })
    return table


def test_bootstrap_interval_widths_of_auc_and_bedroc_with_7_actives(width_table):
    print(f"7 actives among 357, alpha = 20; {WIDTH_N_DATASETS} simulated screens per row, {WIDTH_N_BOOT} bootstrap draws each "
          f"over compounds; widths are medians of the 95% percentile interval; chance is 0.5 for AUC and "
          f"{bedroc_random_expectation(7, 357):.4f} for BEDROC")
    print("  model     par     mean AUC  mean BEDROC  sd AUC  sd BEDROC  width AUC  width BEDROC  P(BEDROC wider)  "
          "width/(1-chance): AUC  BEDROC")
    for r in width_table:
        print(f"  {r['model']:8s} {r['par']:7.3f}  {r['auc_mean']:8.3f}  {r['bed_mean']:10.3f}  {r['auc_sd']:6.3f}  {r['bed_sd']:9.3f}"
              f"  {r['auc_w']:9.3f}  {r['bed_w']:12.3f}  {r['share_bed_wider']:15.2f}  {r['auc_w_rel']:16.3f}  {r['bed_w_rel']:6.3f}")
    assert len(width_table) == len(WIDTH_MODELS)
    by = {(r["model"], r["par"]): r for r in width_table}
    # at chance the ordering is the reverse of the proposition: AUC has the wider interval
    for key in [("binormal", 0.0), ("exp-rank", 0.001)]:
        assert by[key]["bed_w"] < 0.35 * by[key]["auc_w"] and by[key]["share_bed_wider"] < 0.2
    assert by[("binormal", 0.5)]["bed_w"] < by[("binormal", 0.5)]["auc_w"]          # mean AUC 0.64: AUC still wider
    # with skill (mean AUC 0.76 and above) BEDROC has the wider interval in every row
    for r in width_table:
        if r["auc_mean"] > 0.75:
            assert r["bed_w"] > r["auc_w"] and r["share_bed_wider"] > 0.7
    assert by[("binormal", 0.954)]["auc_w"] == pytest.approx(0.324, abs=0.002)
    assert by[("binormal", 0.954)]["bed_w"] == pytest.approx(0.519, abs=0.002)


def test_the_package_estimator_agrees_with_the_numpy_one_used_here_on_tied_bootstrap_draws():
    rng = np.random.default_rng(7)
    labels = np.zeros(120, dtype=int)
    labels[:7] = 1
    scores = np.round(rng.normal(size=120) + labels, 1)
    for _ in range(20):
        idx = rng.integers(0, 120, size=120)
        got = _auc_and_bedroc_tied(labels[idx], scores[idx])
        mine = relate.bedroc(labels[idx].astype(float).tolist(), scores[idx].tolist(), ALPHA)
        if got is None:
            assert mine is None
            continue
        assert mine == pytest.approx(got[1], abs=1e-10)
        assert relate.roc_auc(labels[idx].astype(float).tolist(), scores[idx].tolist()) == pytest.approx(got[0], abs=1e-12)


# ------------------------------------------------------------------------------------------------
# Clopper-Pearson coverage when the actives compete for the shortlist (`topn_recovery`)
# ------------------------------------------------------------------------------------------------
COVERAGE_DESIGNS = [(7, 357, 50), (7, 357, 18), (30, 1500, 75), (40, 200, 50), (20, 100, 10)]     # actives, library, shortlist
COVERAGE_SHIFTS = (0.0, 0.5, 0.954, 1.5, 2.5)


def test_clopper_pearson_covers_the_expected_share_recovered_when_actives_compete_for_places():
    """THE ESTIMAND, WRITTEN BEFORE THE RUN: the expected share of actives a scorer places in the first N of the library, for actives
    drawn like these (binormal: decoy score N(0, 1), active score N(d, 1)), decoys redrawn every screen. THE FIGURE THAT WOULD HAVE
    WITHHELD THE INTERVAL: coverage below 0.93 in any design. 20,000 screens per design; the truth is the simulated mean of k / K."""
    rng = np.random.default_rng(21)
    rows = []
    for n_act, n_lib, n_top in COVERAGE_DESIGNS:
        cp = [relate.clopper_pearson(k, n_act) for k in range(n_act + 1)]
        for d in COVERAGE_SHIFTS:
            ks = []
            for _ in range(8):
                s = rng.normal(size=(2500, n_lib))
                s[:, :n_act] += d
                threshold = np.partition(s, n_lib - n_top, axis=1)[:, n_lib - n_top]
                ks.append((s[:, :n_act] >= threshold[:, None]).sum(axis=1))
            k = np.concatenate(ks)
            share = k.mean() / n_act
            cover = float(np.mean([cp[i][0] <= share <= cp[i][1] for i in k]))
            rows.append((n_act, n_lib, n_top, d, share, cover, k.var(ddof=1) / (n_act * share * (1 - share))))
    print("Clopper-Pearson 95% interval on k of K actives in the first N of M; 20,000 screens per row")
    print("   K     M    N      d   expected share   coverage   var(k) / binomial variance")
    for n_act, n_lib, n_top, d, share, cover, ratio in rows:
        print(f"  {n_act:3d}  {n_lib:5d}  {n_top:3d}  {d:5.3f}  {share:14.4f}  {cover:9.4f}  {ratio:10.3f}")
    covers = [r[5] for r in rows]
    print(f"  coverage ranges from {min(covers):.4f} to {max(covers):.4f}; variance ratio from {min(r[6] for r in rows):.3f} "
          f"to {max(r[6] for r in rows):.3f}")
    assert min(covers) >= 0.95                                  # never below nominal: conservative, so it is reported
    assert min(covers) == pytest.approx(0.959, abs=0.002) and max(covers) == pytest.approx(1.0, abs=0.001)
    assert max(r[6] for r in rows) < 1.0                        # the count is less variable than a binomial in every design


def test_binomial_against_hypergeometric_tail_for_7_actives_and_the_first_50_of_357():
    print("k of 7 actives in the first 50 of 357: p reported by binomial_count (two-sided against 0.5), binomial tail at 50/357, "
          "exact hypergeometric tail")
    for k in range(8):
        binom = stats.binom.sf(k - 1, 7, 50 / 357)
        hyper = relate.hypergeom_tails(k, 7, 50, 357)[1]
        assert hyper == pytest.approx(stats.hypergeom.sf(k - 1, 357, 7, 50), rel=1e-9)
        lo, hi = relate.clopper_pearson(k, 7)
        print(f"  k = {k}: {relate.sign_test_p(k, 7):.6f}   {binom:.3g}   {hyper:.3g}   ratio {binom / hyper:.3f}   "
              f"Clopper-Pearson [{lo:.4f}, {hi:.4f}]")
    assert stats.binom.sf(6, 7, 50 / 357) / relate.hypergeom_tails(7, 7, 50, 357)[1] == pytest.approx(1.463, abs=0.001)


# ------------------------------------------------------------------------------------------------
# the permutation p: which methods lose exact ties in floating point, and the donor rule
# ------------------------------------------------------------------------------------------------
def _exact_pearson(xs, ys):
    fx, fy = [Fraction(str(v)) for v in xs], [Fraction(str(v)) for v in ys]     # the decimal values as written, not their binary images
    return abs(len(fx) * sum(a * b for a, b in zip(fx, fy, strict=True)) - sum(fx) * sum(fy))


def _ranks(v):
    return [Fraction(str(r)) for r in stats.rankdata(v, method="average")]


def _exact_auc(xs, ys):
    pos = [y for x, y in zip(xs, ys, strict=True) if x >= 0.5]
    neg = [y for x, y in zip(xs, ys, strict=True) if x < 0.5]
    return abs(sum(2 * (p > q) + (p == q) for p in pos for q in neg) - len(pos) * len(neg))


def test_which_methods_lose_exact_ties_to_the_plain_float_comparison():
    """Every order of y on 7 rows, each as the observed one against every other, decided in rational arithmetic. Counted: how many
    comparisons `abs(r - centre) >= abs(obs - centre)` gets wrong, and how many `relate.as_extreme` gets wrong."""
    designs = [
        ("pearson", relate.pearson, _exact_pearson, 0.0, [0., 1., 2., 3., 4., 5., 6.], [1., 2., 2., 5., 7., 7., 12.]),
        ("pearson, non-integer", relate.pearson, _exact_pearson, 0.0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [1.1, 2.2, 2.2, 5.3, 7.1, 7.1, 12.4]),
        ("spearman, untied", relate.spearman, lambda x, y: _exact_pearson(_ranks(x), _ranks(y)), 0.0,
         [1., 2., 3., 4., 5., 6., 7.], [1., 2., 3., 4., 5., 6., 7.]),
        ("spearman, tied", relate.spearman, lambda x, y: _exact_pearson(_ranks(x), _ranks(y)), 0.0,
         [1., 1., 2., 3., 3., 3., 4.], [2., 5., 5., 6., 7., 9., 9.]),
        ("spearman, tied, 3 levels", relate.spearman, lambda x, y: _exact_pearson(_ranks(x), _ranks(y)), 0.0,
         [1., 1., 1., 2., 2., 3., 3.], [1., 2., 2., 2., 3., 3., 4.]),
        ("roc_auc, 3 v 4", relate.roc_auc, _exact_auc, 0.5, [1., 1., 1., 0., 0., 0., 0.], [7., 6., 4., 5., 3., 2., 1.]),
        ("roc_auc, 3 v 4, tied", relate.roc_auc, _exact_auc, 0.5, [1., 1., 1., 0., 0., 0., 0.], [4., 3., 3., 2., 2., 1., 1.]),
    ]
    found = {}
    print("  design                      comparisons   plain float wrong   as_extreme wrong")
    for name, est, exact, centre, xs, ys in designs:
        values = {(est(xs, list(p)), exact(xs, list(p))) for p in set(itertools.permutations(ys))}
        plain = fixed = 0
        for r, d_r in values:
            for obs, d_obs in values:
                truth = d_r >= d_obs
                plain += (abs(r - centre) >= abs(obs - centre)) != truth
                fixed += relate.as_extreme(r, obs, centre) != truth
        found[name] = (len(values) ** 2, plain, fixed)
        print(f"  {name:26s}  {len(values) ** 2:11d}   {plain:17d}   {fixed:16d}")
    assert all(v[2] == 0 for v in found.values())
    assert found["roc_auc, 3 v 4"][1] > 0 and found["pearson"][1] > 0


def _null_p(pairs, clusters, estimate, n_perm, rule, centre=0.0, plain=False):
    """`relate.permutation_p` with the donor rule as a switch: "replace" is the rule as coded (each recipient position a draw with
    replacement from the donor's values), "permute" is a within-block permutation when donor and recipient have the same size."""
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    obs = estimate(xs, ys)
    idx_by: dict = {}
    for i, c in enumerate(clusters):
        idx_by.setdefault(c, []).append(i)
    keys = list(idx_by)
    rng = random.Random(relate.SEED + 1)
    hits = done = 0
    for _ in range(n_perm):
        sigma = keys[:]
        rng.shuffle(sigma)
        trial = list(ys)
        for recip, donor in zip(keys, sigma, strict=True):
            src = [ys[j] for j in idx_by[donor]]
            if rule == "permute" and len(src) == len(idx_by[recip]) and len(src) > 1:
                for pos, v in zip(idx_by[recip], rng.sample(src, len(src)), strict=True):
                    trial[pos] = v
            else:
                for pos in idx_by[recip]:
                    trial[pos] = rng.choice(src)
        r = estimate(xs, trial)
        if r is None:
            continue
        done += 1
        hits += (abs(r - centre) >= abs(obs - centre)) if plain else relate.as_extreme(r, obs, centre)
    return (hits + 1) / (done + 1)


def _unit_effect_null(rng, n_units, per, noise_sd):
    """x and y independent, each with a unit-level effect of sd 1 and row noise of sd `noise_sd`: the null with clustered rows."""
    pairs, clusters = [], []
    for u in range(n_units):
        a, b = rng.normal(0, 1), rng.normal(0, 1)
        for _ in range(per):
            pairs.append((float(a + rng.normal(0, noise_sd)), float(b + rng.normal(0, noise_sd))))
            clusters.append(f"u{u}")
    return pairs, clusters


def test_donor_rule_rejection_rate_under_a_clustered_null():
    """600 null datasets per design, Spearman, 200 permutations each. A valid p rejects at 0.05 in 0.05 of them (se 0.009)."""
    rng0 = np.random.default_rng(0)
    pairs, clusters = _unit_effect_null(rng0, 12, 4, 0.5)
    assert _null_p(pairs, clusters, relate.spearman, 300, "replace") == relate.permutation_p(
        pairs, clusters, estimate=relate.spearman, n_perm=300)[0]         # the local copy is the checker's rule, draw for draw
    print("  units x rows   noise sd   reject at 0.05: as coded   within-block permutation   mean |difference in p|")
    rates = []
    for n_units, per, noise in [(15, 4, 0.5), (8, 6, 0.5), (15, 4, 0.05), (30, 2, 0.5)]:
        rng = np.random.default_rng(100 + n_units + per)
        ps = {"replace": [], "permute": []}
        for _ in range(600):
            pairs, clusters = _unit_effect_null(rng, n_units, per, noise)
            for rule in ps:
                ps[rule].append(_null_p(pairs, clusters, relate.spearman, 200, rule))
        a, b = np.array(ps["replace"]), np.array(ps["permute"])
        rates.append((np.mean(a <= 0.05), np.mean(b <= 0.05)))
        print(f"  {n_units:5d} x {per:d}      {noise:5.2f}   {rates[-1][0]:24.3f}   {rates[-1][1]:24.3f}   {np.mean(np.abs(a - b)):10.4f}")
    se = math.sqrt(0.05 * 0.95 / 600)
    assert all(abs(r - 0.05) < 3 * se for pair in rates for r in pair)      # neither rule is distinguishable from 0.05


def test_size_of_the_mirror_tie_error_in_the_roc_auc_p_before_the_fix():
    """The permutation p with the plain float comparison (the checker before 2026-11-12) against the one with `as_extreme`.
    3 actives against 3 decoys, where the exact two-sided p is 4/20; and 7 actives among 357, where a score and its negation have
    the same exact p, 40 uninformative screens with 300 permutations each."""
    labels = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    low = [-6.0, -5.0, -3.0, -4.0, -2.0, -1.0]                                 # AUC 1/9
    pairs, units = list(zip(labels, low, strict=True)), [str(i) for i in range(6)]
    p_plain = _null_p(pairs, units, relate.roc_auc, 20000, "replace", centre=0.5, plain=True)
    p_fixed = _null_p(pairs, units, relate.roc_auc, 20000, "replace", centre=0.5)
    assert p_fixed == relate.permutation_p(pairs, None, estimate=relate.roc_auc, n_perm=20000, centre=0.5)[0]
    print(f"3 actives against 3 decoys, AUC 1/9, exact two-sided p = 0.2000: plain float comparison {p_plain:.4f}, as_extreme {p_fixed:.4f}")
    assert p_plain == pytest.approx(0.15, abs=0.01) and p_fixed == pytest.approx(0.20, abs=0.01)
    rng = np.random.default_rng(12)
    labels = [1.0] * 7 + [0.0] * 350
    units = [str(i) for i in range(357)]
    plain_diffs, fixed_diffs = [], []
    for _ in range(40):
        s = rng.normal(size=357).tolist()
        a, b = list(zip(labels, s, strict=True)), list(zip(labels, [-v for v in s], strict=True))
        plain_diffs.append(abs(_null_p(a, units, relate.roc_auc, 300, "replace", 0.5, True) - _null_p(b, units, relate.roc_auc, 300, "replace", 0.5, True)))
        fixed_diffs.append(abs(_null_p(a, units, relate.roc_auc, 300, "replace", 0.5) - _null_p(b, units, relate.roc_auc, 300, "replace", 0.5)))
    n_plain = sum(d > 0 for d in plain_diffs)
    print(f"7 actives among 357, p(score) against p(-score), 300 permutations: plain float comparison differs in {n_plain} of 40 screens, "
          f"largest difference {max(plain_diffs):.5f} ({max(plain_diffs) * 301:.0f} permutation draws); as_extreme differs in "
          f"{sum(d > 0 for d in fixed_diffs)} of 40")
    assert n_plain == 5 and max(plain_diffs) == pytest.approx(2 / 301, abs=1e-9)
    assert max(fixed_diffs) == 0.0
