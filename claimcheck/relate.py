"""Recompute a claimed relationship, with an interval, and say whether it holds.

    python -m claimcheck.relate <project-dir> [<out.json>]

## Why an interval and not a tolerance

The first version of this compared two point estimates against a hand-picked
tolerance. On five rows it "falsified" both claims it was given — and it would
have done that to a *true* claim just as readily, because a point comparison at
n=5 has no power. A checker that cannot distinguish "the claim is wrong" from
"the data cannot tell" is not a checker; it is a coin with an opinion.

So the verdict is now **does the claimed value lie inside the interval**, and
the interval is computed from the data. A claim outside a tight interval is
refuted. A claim inside a wide one is not confirmed — it is `unverifiable`,
which is a third outcome and is **never a pass**.

## The resampling unit is declared, never assumed

Bootstrapping over rows assumes rows are independent. In a sweep they are not:
every row of one configuration shares its seeds, its library and its budget, so
resampling rows would treat 29 configurations as 870 independent observations
and report an interval roughly a fifth of its true width.

`resampling_unit` names the column to cluster on and **must be stated** when the
data has repeated structure. `claimcheck` will not guess it: guessing it is how a
confidence interval becomes decoration.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import json
import math
import random
import sys
import tomllib
from collections import Counter
from pathlib import Path

from claimcheck.datasets import (
    LEVEL_RANK,
    LEVELS,
    DatasetError,
    canonical_level,
    join,
)
from claimcheck.datasets import load as load_dataset

#: Three outcomes. `unverifiable` means the check could not reach a verdict,
#: which is a different fact from the claim being wrong. Folding them together
#: is how a column of "could not run" becomes a clean bill of health.
AGREES, REFUTED, UNVERIFIABLE = "agrees", "refuted", "unverifiable"
#: verdict -> the belief status it carries. PUBLIC because a second
#: module now adjudicates with these three words (`claimcheck.jobs`), and
#: a private copy of the mapping is how two answers to one question begin.
STATUS = {AGREES: "measured", REFUTED: "falsified", UNVERIFIABLE: "untested"}

#: Enough that the percentile bounds are stable to ~2 decimal places, and
#: cheap enough to run on every build. Fixed seed: an interval that moves
#: between two runs of the same data is not a measurement.
N_BOOT, SEED = 2000, 20260815

#: An interval wider than this admits so much of the possible range that
#: "the claim lies inside it" says nothing. r spans [-1, 1], so a width of 1.0
#: is half of everything — an interval that broad cannot discriminate a strong
#: negative relationship from a strong positive one, and reporting `agrees`
#: against it would be the reassurance this package exists to refuse.
#:
#: FOUND BY RUNNING IT. On the five-row example the CI came out [-0.683, 1.000]
#: and both claims were reported as HOLDING — while this module's own docstring
#: already said a claim inside a wide interval "is not confirmed, it is
#: unverifiable". The prose was right and the code did not implement it, which
#: is `docs/MISTAKES.md` §4 committed inside the file that argues against it.
MAX_INFORMATIVE_WIDTH = 1.0

#: A KNOWN ANSWER OF +-1 FROM TWO INDEPENDENT COMPUTATIONS. Two numbers computed
#: two ways never agree exactly: an engine derivative and its central finite
#: difference differ by about 1e-7 relative, so 1 - r = eps^2 / 2 is about
#: 5e-15, and a bootstrap interval that tight excludes 1.0. That is agreement at
#: numerical precision, not a refutation (mace_lambda/alkanes KF2a, 2026-09-25,
#: refuted at 1 - r = 2.6e-15 over 378 frames). A claim of exactly +-1 agrees
#: when |r - claimed| is within this tolerance, i.e. relative disagreement below
#: about 1.4e-6; anything looser is still refuted. It is a tolerance, stated as
#: one, and applies to no other claimed value.
KNOWN_ANSWER_TOL = 1e-12

#: A PERMUTED STATISTIC EXACTLY AS FAR FROM THE NULL CENTRE AS THE OBSERVED ONE COUNTS AS EXTREME, and floating point does not always
#: say so (2026-11-12). |8/9 - 0.5| is 0.38888888888888884 and |1/9 - 0.5| is 0.3888888888888889, so with 3 actives against 3 decoys an
#: observed AUC of 1/9 did not count its mirror 8/9 and came back p = 3/20 where the exact two-sided value is 4/20: a p too small.
#: Pearson loses the same kind of tie on discrete data, where two permutations reach one |r| by different sums (82 of 16,129 comparisons
#: on a 7-row integer design; Spearman and Kendall were exact on the designs tried). The comparison therefore allows this much, absolute.
#: It is far above the rounding error of a statistic in [-1, 1] (about 1e-16) and far below the gap between two attainable values:
#: 1 / (2 * n_pos * n_neg) for AUC, 2 / (n * (n - 1)) for Kendall and about 6 / n**3 for Spearman, which reach 1e-12 only beyond
#: n_pos * n_neg = 5e11, n = 1.4e6 and n = 1.8e4. Past those sizes it counts values within 1e-12 of the observed one, a null mass that
#: is itself negligible. It can only raise a p, never lower one.
PERMUTATION_TIE_TOL = 1e-12


def as_extreme(r: float, obs: float, centre: float = 0.0) -> bool:
    """Whether a permuted statistic `r` is at least as far from the null `centre` as the observed `obs`, ties included
    (see PERMUTATION_TIE_TOL)."""
    return abs(r - centre) >= abs(obs - centre) - PERMUTATION_TIE_TOL


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """r, or None where it is undefined rather than zero.

    A constant column has no correlation defined on it. Returning 0.0 would be
    a number where there is no answer, and 0.0 reads as "no relationship" —
    a conclusion that has not been earned.
    """
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / math.sqrt(sxx * syy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Pearson on ranks: monotonic association without assuming a straight line.

    Worth having in chemistry specifically. Binding energies against measured
    affinity are routinely monotonic and curved, and Pearson understates that
    while Spearman does not — so "which estimator" is a scientific choice and
    belongs in `context.toml`, not in whichever one the code happens to call.
    """
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            # TIES TAKE THE AVERAGE RANK. Assigning them arbitrary distinct
            # ranks invents an ordering the data does not have, and on data with
            # many repeated values that is most of the correlation.
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    return pearson(rank(xs), rank(ys))


def roc_auc(xs: list[float], ys: list[float]) -> float | None:
    """Area under the ROC curve of a score `y` against a binary label `x`: the probability that a
    randomly chosen positive outscores a randomly chosen negative, with a tie counting a half.

    ADDED 2026-11-12 for virtual screening, where the question is "does the score put the known
    binders near the top", and a correlation is the wrong shape for it: with 29 hits among 1,450
    decoys a Spearman is dominated by the ordering of the decoys among themselves, which nobody
    acts on, while AUC asks only about hit against decoy pairs. 0.5 is no better than chance and
    1.0 is a perfect ranking.

    `x` IS THE LABEL AND `y` IS THE SCORE, and they do not commute. The label is read as an
    indicator at >= 0.5, the same convention `fraction` and `f1` use. HIGHER `y` MEANS MORE LIKELY
    POSITIVE: a score where a better compound is more negative, as a docking affinity is, must be
    negated by the rule that writes the table, in a column whose name says so. The sign convention
    is a choice that could have gone differently, so it belongs in the declaration, not here.

    Computed from average ranks (the Mann-Whitney U identity), so ties between a positive and a
    negative contribute exactly a half rather than being ordered by floating-point noise.

    Returns None when either class is empty, so the claim comes back `unverifiable` rather than
    0.5. A file whose join dropped every hit is then distinguishable from an honest chance result.

    Its square is not reported: AUC is a probability of correct ordering, not a correlation, so
    squaring it is not a share of anything.
    """
    if not xs or len(xs) != len(ys):
        return None
    pos = [i for i, v in enumerate(xs) if v >= 0.5]
    neg = [i for i, v in enumerate(xs) if v < 0.5]
    if not pos or not neg:
        return None
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    rank = [0.0] * len(ys)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and ys[order[j + 1]] == ys[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    u = sum(rank[i] for i in pos) - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def mid_ranks(vs: list[float]) -> list[float]:
    """Average ranks, 1-based, ties sharing the mean of the positions they span.

    The same rule `roc_auc` and `spearman` each apply inside themselves, written once here for the code that needs the
    ranks THEMSELVES rather than a statistic computed from them (`paired_swap_p`). Neither of those two is rewritten to
    call it: their arithmetic would be identical, and nothing is gained by moving the one line that 6,889 adjudicated
    rows were computed through.

    AUC IS INVARIANT UNDER THIS MAP, exactly and not approximately: the ranks are a monotone relabelling that preserves
    the tie structure, so ranking them again returns them unchanged. That is what lets `paired_swap_p` put two score
    columns of different scales on one comparable scale without changing the statistic it is testing.
    """
    order = sorted(range(len(vs)), key=lambda i: vs[i])
    out = [0.0] * len(vs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def roc_auc_difference(xs: list[float], ys: list[tuple[float, float]]) -> float | None:
    """AUC(score) - AUC(baseline) against one binary label, both computed ON IDENTICAL ROWS.

    ADDED 2026-11-20 (proposal P36). Two AUROCs adjudicated separately come back with two intervals, and two
    overlapping intervals do not say that their difference contains zero: most of each interval's width is the cohort
    variance the two share, and a difference claim is the only way to cancel it. Measured on
    `/home/ben/code/projects/targets/cry1/experiments/litpcba_flexdock`, where the CNN affinity reads 0.7323
    [0.6653, 0.7930] and heavy-atom count 0.6573 [0.5877, 0.7230] on the same 5,047 compounds: the intervals overlap
    almost entirely and nothing in the corpus could say whether the scores differ.

    `xs` IS THE LABEL, read as an indicator at >= 0.5. `ys` IS ONE PAIR PER ROW, `(score, baseline)`, packed by
    `_paired` so that the shared machinery carries both columns through a resample without ever seeing them apart:
    `bootstrap_ci` and `permutation_p` only index `p[0]` and `p[1]` and never do arithmetic on either, so A RESAMPLE
    DRAWS A ROW WITH BOTH OF ITS SCORES and the pairing cannot be lost. Resampling the two columns independently would
    throw away the correlation between them, which is the entire gain.

    HIGHER MEANS MORE LIKELY POSITIVE IN BOTH COLUMNS, as for `roc_auc`, and the rule that writes the table owns that
    convention. The sign of the result is the method's: POSITIVE MEANS `score` SEPARATES THE LABELS BETTER THAN
    `baseline` DOES. It spans [-1, 1] and is exactly 0.0 when the two columns are the same column, which is this
    estimator's known answer.

    Returns None when either AUC is undefined (either class empty, ragged input), so the claim comes back
    `unverifiable` rather than 0.0: a join that dropped every active must not read as "the two scores are equal".
    """
    if not xs or len(xs) != len(ys):
        return None
    try:
        a = [float(v[0]) for v in ys]
        b = [float(v[1]) for v in ys]
    except (TypeError, IndexError, ValueError):
        return None                    # not packed: a two-column claim's pairs reaching a three-column estimator
    ua, ub = roc_auc(xs, a), roc_auc(xs, b)
    if ua is None or ub is None:
        return None
    return ua - ub


def paired_swap_p(pairs: list[tuple[float, tuple[float, float]]], clusters: list[str] | None,
                  n_perm: int = N_BOOT, estimate=None) -> tuple[float, float] | None:
    """Two-sided p for a paired difference of two screening statistics, by SWAPPING THE TWO SCORE COLUMNS WITHIN A WHOLE UNIT.

    Returns `(p, floor)` as `permutation_p` does, `floor` being `1 / (n_perm + 1)`.

    WHY NOT `permutation_p`. That one permutes the y values of whole clusters against x, which here moves both score
    columns together and so tests whether EITHER column reads the label. That is not this claim's question. A claim
    that one score beats another needs a null under which the two are interchangeable, and its p must be large when
    both columns are strongly informative and equally so, which the label permutation's is not: it would report a
    decisive p for two excellent and identical scorers.

    SO THE NULL IS EXCHANGEABILITY OF THE TWO COLUMNS, drawn by giving each resampling unit an independent fair coin
    and swapping that unit's two score values when it comes up heads. The unit is the independent draw, so the whole
    unit swaps or none of it does. Both columns are put on mid-ranks over the whole table first (`mid_ranks`), because
    a swap mixes values from the two columns into one column and AUC then ranks them together: CNN affinity in [1, 8]
    and heavy-atom count in [10, 60] would otherwise be compared as magnitudes. AUC is invariant under that rank map,
    so the observed statistic is unchanged by it and only the null is affected.

    WHAT IT ASSUMES, AND WHERE THAT IS APPROXIMATE. Exchangeability needs the two columns to be interchangeable under
    the null, and two real score columns rarely share a tie structure: heavy-atom count takes a few dozen distinct
    values over thousands of compounds while a CNN affinity is almost tie-free, so a swap moves a tied block into
    positions the other column resolved. The null distribution is therefore approximate, in a direction that is not
    established here. It is reported for that reason and NO VERDICT DEPENDS ON IT: the verdict is the claimed value
    against the paired bootstrap interval, as everywhere else in this package.

    The null distribution is symmetric about 0 by construction, since swapping every unit of a draw gives exactly the
    negated statistic, so the two-sided comparison about a centre of 0 is the right one (`as_extreme`).

    `estimate` IS THE PAIRED STATISTIC, defaulting to `roc_auc_difference` (2026-11-22, when `bedroc_difference` became
    the second one; before that it was written into the body). It must be antisymmetric in the two columns and invariant
    under `mid_ranks`, which both members of `PAIRED_DIFFERENCE_METHODS` are: BEDROC reads only the order of the scores
    and which of them are tied, as AUROC does, so the monotone tie-preserving rank map leaves both unchanged. A
    statistic without those two properties would get a null that is neither centred nor comparable with the observed
    value, and nothing here could detect it. The draw order of the RNG is untouched by the argument, so every p already
    adjudicated under the AUROC form is byte-identical.

    FOR A BEDROC DIFFERENCE the approximation named above bites in a particular place, worth stating because it is where
    the statistic looks: a swap mixes the two columns' values into one column, and a column with a coarse tie structure
    (heavy-atom count over thousands of compounds) contributes blocks that the other column resolved. BEDROC puts 0.798
    of its weight in the first 8% of the list at alpha = 20, so what the mismatch perturbs is exactly the part of the
    list the statistic is made of. The p is reported for that reason and no verdict depends on it.
    """
    if estimate is None:
        estimate = roc_auc_difference
    xs = [p[0] for p in pairs]
    try:
        a = mid_ranks([float(p[1][0]) for p in pairs])
        b = mid_ranks([float(p[1][1]) for p in pairs])
    except (TypeError, IndexError, ValueError):
        return None
    obs = estimate(xs, list(zip(a, b, strict=True)))
    if obs is None:
        return None
    if clusters is None:
        idx_by: dict[str, list[int]] = {str(i): [i] for i in range(len(pairs))}
    else:
        idx_by = {}
        for i, c in enumerate(clusters):
            idx_by.setdefault(c, []).append(i)
    if len(idx_by) < 3:
        return None
    rng = random.Random(SEED + 2)
    hits = done = 0
    for _ in range(n_perm):
        first, second = list(a), list(b)
        for rows in idx_by.values():
            if rng.random() < 0.5:
                for i in rows:
                    first[i], second[i] = b[i], a[i]
        d = estimate(xs, list(zip(first, second, strict=True)))
        if d is None:
            continue
        done += 1
        if as_extreme(d, obs, 0.0):
            hits += 1
    if done < n_perm // 2:
        return None
    return ((hits + 1) / (done + 1), 1.0 / (done + 1))


@functools.lru_cache(maxsize=64)
def _bedroc_weights(n_total: int, alpha: float) -> tuple[float, ...]:
    """exp(-alpha * k / N) for list position k = 1..N, position 1 being the best-scored compound."""
    return tuple(math.exp(-alpha * k / n_total) for k in range(1, n_total + 1))


def _bedroc_bounds(n_pos: int, n_total: int, alpha: float) -> tuple[float, float, float] | None:
    """(weight sum of the worst ranking, of the best ranking, expected under a random ranking) for n_pos positives among n_total."""
    w = _bedroc_weights(n_total, alpha)
    s_min, s_max = math.fsum(w[n_total - n_pos:]), math.fsum(w[:n_pos])
    if not s_max > s_min:                      # alpha so large that every weight underflows: nothing left to rank with
        return None
    return s_min, s_max, n_pos * math.fsum(w) / n_total


def bedroc_random_expectation(n_pos: int, n_total: int, alpha: float) -> float | None:
    """The BEDROC a uniformly random ranking scores on average, for n_pos positives among n_total compounds: the value at RIE = 1.
    IT IS NOT ZERO. It is 0.0604 at 7 of 357 and alpha = 20, 0.0552 at 10 of 1,000, and close to the active fraction once
    alpha * n_pos / n_total is large (0.4986 at 178 of 357). A BEDROC is read against this number, never against 0."""
    if not (0 < n_pos < n_total) or not (math.isfinite(alpha) and alpha > 0):
        return None
    b = _bedroc_bounds(n_pos, n_total, alpha)
    if b is None:
        return None
    s_min, s_max, s_random = b
    return (s_random - s_min) / (s_max - s_min)


def bedroc(xs: list[float], ys: list[float], alpha: float) -> float | None:
    """BEDROC (Truchon and Bayly, J. Chem. Inf. Model. 2007, 47, 488) of a score `y` against a binary label `x`: early recognition,
    each positive weighted by exp(-alpha * k / N) at list position k of N, position 1 being the highest score.

    ADDED 2026-11-12 beside `roc_auc`, which weights every positive alike wherever it sits: two screeners that roc_auc scores 0.7525
    and 0.7505 (10 positives among 1,000, at positions 1 to 5 and 496 to 500 against 248 to 257) score 0.5250 and 0.0072 here at
    alpha = 20, and place 5 and 0 positives in the first 50.

    THE DEFINITION IS A MIN-MAX NORMALISATION, AND IT DOES NOT SATURATE. With S the sum of the positives' weights, the value is
    (S - S_min) / (S_max - S_min), S_max being the sum for the best ranking (every positive first) and S_min for the worst (every
    positive last). It is exactly 1.0 for the best ranking and 0.0 for the worst AT ANY ACTIVE FRACTION. This is algebraically the
    paper's closed form in RIE, sinh and cosh, and agrees with RDKit's CalcBEDROC to 1e-9. What moves with the active fraction is the
    ceiling of RIE (not of BEDROC) and the value a random ranking scores (`bedroc_random_expectation`), which is well above 0.

    `alpha` HAS NO DEFAULT. It decides what "early" means: the share of the weight carried by the first fraction z of the list is
    (1 - exp(-alpha * z)) / (1 - exp(-alpha)), which at alpha = 20 is 0.798 for the first 8%. A BEDROC without its alpha is not a
    number about anything, so the claim declares it and the verdict row records it.

    `x` IS THE LABEL AND `y` IS THE SCORE, read as `roc_auc` reads them: the label is an indicator at >= 0.5 and HIGHER `y` MEANS MORE
    LIKELY POSITIVE.

    TIES SHARE. Compounds with equal scores share the mean of the weights of the positions they occupy between them, which is the
    expectation over a random order of the tie. A bootstrap draw repeats rows, so ties are the ordinary case there, not a corner.

    None when either class is empty, when alpha is not a positive finite number, or when the lists are ragged or empty, so the claim
    comes back `unverifiable` and never as a number."""
    if not xs or len(xs) != len(ys):
        return None
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        return None
    n_total = len(xs)
    n_pos = sum(1 for v in xs if v >= 0.5)
    if n_pos == 0 or n_pos == n_total:
        return None
    alpha = float(alpha)
    bounds = _bedroc_bounds(n_pos, n_total, alpha)
    if bounds is None:
        return None
    s_min, s_max, _ = bounds
    w = _bedroc_weights(n_total, alpha)
    order = sorted(range(n_total), key=lambda i: -ys[i])
    parts: list[float] = []
    i = 0
    while i < n_total:
        j = i
        while j + 1 < n_total and ys[order[j + 1]] == ys[order[i]]:
            j += 1
        if j == i:
            if xs[order[i]] >= 0.5:
                parts.append(w[i])
        else:
            in_block = sum(1 for k in range(i, j + 1) if xs[order[k]] >= 0.5)
            if in_block:
                parts.append(in_block * math.fsum(w[i:j + 1]) / (j - i + 1))
        i = j + 1
    return (math.fsum(parts) - s_min) / (s_max - s_min)


def bedroc_alpha(alpha: float):
    """`bedroc` at a declared alpha, as the two-argument estimator the bootstrap and the permutation null call."""
    def estimate(xs: list[float], ys: list[float]) -> float | None:
        return bedroc(xs, ys, alpha)
    return estimate


def _bedroc_needs_alpha(xs: list[float], ys: list[float]) -> float | None:
    """The registered `bedroc` entry. BEDROC has no value until alpha is declared, so called bare this returns None; `check` replaces
    it with `bedroc_alpha(alpha)` or refuses the claim."""
    return None


def bedroc_difference(xs: list[float], ys: list[tuple[float, float]], alpha: float) -> float | None:
    """BEDROC(score) - BEDROC(baseline) AT ONE DECLARED ALPHA, both computed against one binary label ON IDENTICAL ROWS.

    ADDED 2026-11-22. `roc_auc_difference` (2026-11-20) answered the same question for AUROC and left the metric
    `litpcba_flexdock` DECLARES AS PRIMARY unanswered: at a 2% active rate an AUROC is dominated by the ordering of the
    inactives among themselves, which nobody acts on, so that experiment's gate was decided on a statistic it does not
    rank by (`context.pending.toml`, D2, pending from 2026-11-18). BEDROC weights the top of the list, which is where a
    screen is used, so the two can disagree and THE DISAGREEMENT IS THE FINDING.

    `xs` IS THE LABEL, read as an indicator at >= 0.5. `ys` IS ONE PAIR PER ROW, `(score, baseline)`, packed by
    `_paired`, so `bootstrap_ci` and `paired_swap_p` carry both columns through a resample without ever seeing them
    apart. HIGHER MEANS MORE LIKELY POSITIVE IN BOTH COLUMNS, as for `bedroc` and `roc_auc`, and the rule that writes
    the table owns that convention. POSITIVE MEANS `score` PUTS THE ACTIVES HIGHER UP THE LIST THAN `baseline` DOES.

    ALPHA IS APPLIED IDENTICALLY TO BOTH COLUMNS and has no default, as for `bedroc`: it decides what "early" means
    (0.798 of the weight in the first 8% of the list at alpha = 20), two alphas are two different claims, and an alpha
    applied to one column and not the other would be a difference between two different questions.

    WHY THE NULL IS STILL ZERO THOUGH NEITHER TERM IS CENTRED ON A CONSTANT. A BEDROC is read against
    `bedroc_random_expectation`, which is 0.0604 at 7 actives of 357 and not 0.5 or 0. That expectation depends only on
    the number of positives, the number of rows and alpha, and the two columns here are scored on the SAME ROWS, so they
    share it exactly, along with S_min and S_max. The difference is (S_score - S_baseline) / (S_max - S_min): the
    normalising constants cancel in the estimate, in every bootstrap draw and in every swap draw. So zero means "the two
    columns gather the same weight of actives at the top", which is the hypothesis, and the random expectation never
    enters the difference. It is on the verdict row all the same, because the two BEDROCs beside it are read against it.

    It spans [-1, 1]: both ends are reachable, a perfect ranking against its own inversion being exactly +1.0 at any
    active fraction (BEDROC does not saturate). A column against a copy of itself is exactly 0.0, which is this
    estimator's known answer, and the zero-width bootstrap interval it comes with is the only cheap proof that the
    pairing in the table is what the claim thinks it is.

    None when either BEDROC is undefined (either class empty, alpha absent or not a positive finite number, ragged or
    unpacked input), so the claim reads `unverifiable` rather than 0.0: a join that dropped every active must not come
    back as "the two scores are equal".
    """
    if not xs or len(xs) != len(ys):
        return None
    try:
        a = [float(v[0]) for v in ys]
        b = [float(v[1]) for v in ys]
    except (TypeError, IndexError, ValueError):
        return None                    # not packed: a two-column claim's pairs reaching a three-column estimator
    ba, bb = bedroc(xs, a, alpha), bedroc(xs, b, alpha)
    if ba is None or bb is None:
        return None
    return ba - bb


def bedroc_difference_alpha(alpha: float):
    """`bedroc_difference` at a declared alpha, as the two-argument estimator the paired bootstrap and the swap null call.
    ONE alpha for both columns, bound once here, so no code path downstream can give the two columns different weights."""
    def estimate(xs: list[float], ys: list[tuple[float, float]]) -> float | None:
        return bedroc_difference(xs, ys, alpha)
    return estimate


def _bedroc_difference_needs_alpha(xs: list[float], ys: list[tuple[float, float]]) -> float | None:
    """The registered `bedroc_difference` entry, None until `check` binds alpha, exactly as `_bedroc_needs_alpha` is."""
    return None


#: THE METHODS THAT READ A THIRD COLUMN, declared as `baseline = "<dataset>.<column>"` and packed inside the pair by
#: `_paired`. Written once as a set because the alternative, a method name repeated at every site that asks "is this a
#: paired difference", is how the second such method gets added to four places out of six (2026-11-22, with
#: `bedroc_difference`; `roc_auc_difference` was the only member from 2026-11-20).
PAIRED_DIFFERENCE_METHODS = frozenset({"roc_auc_difference", "bedroc_difference"})

#: The methods taking the required claim key `alpha`, which has no default (see `bedroc`).
ALPHA_METHODS = frozenset({"bedroc", "bedroc_difference"})


def kendall(xs: list[float], ys: list[float]) -> float | None:
    """Kendall's tau-b, the tie-corrected rank correlation: (C - D) / sqrt((C + D + Tx) * (C + D + Ty)).

    C and D count concordant and discordant pairs, Tx the pairs tied in x only and Ty those tied in y only; a pair tied in both columns
    is in none of the four. It is the share of pairs the two columns order the same way, less the share they order oppositely, with
    the tied pairs taken out of each column's denominator, so a column of many repeated values is not read as disagreement.

    ADDED 2026-10-27 for rank questions asked of small, heavily tied columns (FreeSolv rankings), where tau-b is the quantity a
    reader means by "the fraction of pairs ranked correctly" and Spearman is not.

    O(n log n), AND NEEDED TO BE. The bootstrap and the permutation null call this 4,000 times, and an all-pairs loop costs about
    40 ms at n = 642, which is minutes per claim. Sorting by (x, y) puts every pair with unequal x in order, and a pair is discordant
    exactly when an earlier point has a larger y; ties in x never count there because within a tied x the y values already ascend,
    and ties in y are excluded by taking `bisect_right`. The sorted list of earlier y values is kept with `insort`, whose list shift
    is a C memmove and not a Python loop. Ties are counted from the multiplicities of x, of y and of (x, y).

    None where undefined: fewer than 3 pairs, or a column that takes one value (every pair tied in it, so its denominator term is
    zero). Returning 0.0 there would read as "no relationship", which has not been earned.
    """
    n = len(xs)
    if n < 3:
        return None
    n0 = n * (n - 1) // 2
    pairs_tied_x = sum(t * (t - 1) // 2 for t in Counter(xs).values())
    pairs_tied_y = sum(t * (t - 1) // 2 for t in Counter(ys).values())
    pairs_tied_both = sum(t * (t - 1) // 2 for t in Counter(zip(xs, ys, strict=True)).values())
    untied_x, untied_y = n0 - pairs_tied_x, n0 - pairs_tied_y
    if untied_x == 0 or untied_y == 0:
        return None
    seen: list[float] = []
    discordant = 0
    for k, (_, y) in enumerate(sorted(zip(xs, ys, strict=True))):
        discordant += k - bisect.bisect_right(seen, y)
        bisect.insort(seen, y)
    ordered = n0 - pairs_tied_x - pairs_tied_y + pairs_tied_both            # C + D
    return (ordered - 2 * discordant) / math.sqrt(untied_x * untied_y)


#: THE EXTENSION POINT FOR STATISTICS. `method` was written into every output
#: and never dispatched on, so `method = "spearman"` computed Pearson and
#: labelled it Spearman — a silent wrong answer of exactly the kind this package
#: exists to refuse, in its own core.
#:
#: Adding one is a function and an entry here. Each declares the `kind` it
#: implements so a mismatched pair is refused rather than quietly ignored.
def _mean(xs: list[float], ys: list[float]) -> float | None:
    return sum(ys) / len(ys) if ys else None


def _median(xs: list[float], ys: list[float]) -> float | None:
    """The median of y (a one-column method, write x = y): the typical resampling unit, which a few units with a large value do not move.
    Added 2026-10-21 for "the median per-target rank correlation" (miel_outliers amendment 5); an even count takes the mean of the middle two."""
    if not ys:
        return None
    s = sorted(ys)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _sd(xs: list[float], ys: list[float]) -> float | None:
    if len(ys) < 2:
        return None
    m = sum(ys) / len(ys)
    return math.sqrt(sum((v - m) ** 2 for v in ys) / (len(ys) - 1))


def _var_share(xs: list[float], ys: list[float]) -> float | None:
    """var(y) / var(x): the share of a total's variance (x) that one part (y) carries."""
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((v - mx) ** 2 for v in xs)
    return None if vx == 0 else sum((v - my) ** 2 for v in ys) / vx


def _fraction(xs: list[float], ys: list[float]) -> float | None:
    """The share of rows whose y is true (1): a count that passes a rule, out of the rows."""
    return sum(1.0 for v in ys if v >= 0.5) / len(ys) if ys else None


def _mean_diff(xs: list[float], ys: list[float]) -> float | None:
    """mean(y | x = 1) - mean(y | x = 0): the difference between two groups of rows, x the 0/1 group label (2026-11-09). Undefined (None)
    when either group is empty, which a bootstrap draw can produce; the caller refuses any x other than 0 or 1 before this runs."""
    a = [y for x, y in zip(xs, ys, strict=True) if x == 1.0]
    b = [y for x, y in zip(xs, ys, strict=True) if x == 0.0]
    if not a or not b:
        return None
    return sum(a) / len(a) - sum(b) / len(b)


def _f1(xs: list[float], ys: list[float]) -> float | None:
    """F1 of a binary prediction `x` against a binary truth `y`: 2TP / (2TP + FP + FN).

    THE ONE TWO-COLUMN BOOLEAN ESTIMATOR (2026-10-21). Added for retrieval
    questions, "the shortlist this score produces contains the hits", where a
    correlation is the wrong shape: a screener acts on a shortlist, and precision
    and recall trade against each other in a way no single correlation shows.

    Both columns are read as indicators, true at >= 0.5, the same convention
    `fraction` uses, so a 0/1 column and a boolean column behave identically.
    `x` is the PREDICTION and `y` is the TRUTH, and they are not interchangeable:
    swapping them swaps precision for recall and gives a different number
    whenever the shortlist size differs from the number of positives.

    THE THRESHOLD IS NOT HERE, DELIBERATELY. F1 needs a continuous score cut
    somewhere, and that cut is a choice that could have gone differently, so it
    belongs in the rule that writes the table and in the column's name, not
    buried in an estimator. The rule declares the shortlist depth and writes a
    boolean column; this function then has nothing left to decide.

    Returns None when 2TP + FP + FN is zero, which happens only when the
    prediction and the truth are both empty of positives. F1 is undefined there
    rather than 0 or 1, so the claim comes back `unverifiable`, which is the
    honest answer and is what distinguishes a broken join from a real null.
    """
    if not ys or len(xs) != len(ys):
        return None
    tp = fp = fn = 0
    for predicted, actual in zip(xs, ys):
        p, a = predicted >= 0.5, actual >= 0.5
        if p and a:
            tp += 1
        elif p and not a:
            fp += 1
        elif a and not p:
            fn += 1
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else (2.0 * tp) / denominator


def _sign_fraction(xs: list[float], ys: list[float]) -> float | None:
    """The share of units whose sign is the event, over the one +1/-1 value per unit that `_sign_units` makes."""
    return sum(1.0 for v in ys if v > 0) / len(ys) if ys else None


def sign_test_p(k: int, n: int) -> float | None:
    """Exact two-sided binomial p for k events in n units against a null of 0.5: twice the smaller tail, capped at 1.

    Integer arithmetic throughout (`math.comb` over 2**n), so the value is exact and needs no scipy; for the symmetric null
    it is the same number `scipy.stats.binomtest(k, n).pvalue` gives. 9 of 10 is 22/1024, 14 of 15 is 32/32768, 5 of 10 is 1.
    """
    if n < 1 or not 0 <= k <= n:
        return None
    lower = sum(math.comb(n, i) for i in range(k + 1))          # 2**n * P(X <= k)
    upper = sum(math.comb(n, i) for i in range(k, n + 1))       # 2**n * P(X >= k)
    return min(1.0, 2 * min(lower, upper) / 2 ** n)


#: How a sign-test column is read. TRUE/FALSE words are accepted so a boolean column written by pandas or R is not dropped
#: as unparsable; the MISSING tokens are dropped and counted like a non-finite value elsewhere. Anything else is refused.
_TRUE_WORDS = {"true", "t", "yes", "y"}
_FALSE_WORDS = {"false", "f", "no", "n"}
_MISSING_WORDS = {"", "na", "nan", "none", "null"}
SIGN_EVENTS = ("true", "positive")


def _sign_units(rows: list[dict], y: str, unit: str | None, event: str | None) -> tuple[list[tuple[float, float]], list[str] | None, dict]:
    """One +1/-1 value per resampling unit for a sign test, with what was dropped and how it was read.

    The column is read either as BOOLEAN (`event = "true"`: 1, true, yes are the event, 0, false, no are not, and there are no
    ties) or as SIGNED (`event = "positive"`: a positive value is the event, a negative one is not, and an exact zero is a
    tie, dropped and counted in `n_ties_dropped`). When the claim declares no `event` it is detected: a column whose usable
    values are all 0 or 1 is boolean, any other column is signed, and `event_source` says which happened. A column of
    nothing but zeros therefore reads as boolean with no events, so a signed column that could be all ties should declare
    `event = "positive"`.

    A unit with several rows takes the MAJORITY sign over them; equal counts make the unit a tie, dropped and counted in
    `n_unit_ties_dropped`. `unit_value` says which rule applied. Rows whose value is missing or not finite are counted in
    `dropped`; a value that is neither a boolean word nor a number refuses the claim with the value and its row named.
    """
    info: dict = {"refused": None}
    parsed: list[tuple[float, str]] = []
    dropped = 0
    for i, r in enumerate(rows):
        raw = r.get(y)
        text = str(raw).strip() if raw is not None else ""
        low = text.lower()
        if low in _MISSING_WORDS:
            dropped += 1
            continue
        if low in _TRUE_WORDS:
            v = 1.0
        elif low in _FALSE_WORDS:
            v = 0.0
        else:
            try:
                v = float(text)
            except ValueError:
                info["refused"] = (f"column {y!r} is neither boolean nor numeric: row {i + 1} holds {text!r}, which a sign "
                                   f"test cannot read as an event. Accepted: true/false, yes/no, 0/1, or a signed number")
                return [], None, info
            if not math.isfinite(v):
                dropped += 1
                continue
        parsed.append((v, str(r.get(unit, "")) if unit else str(i)))
    if event is None:
        event = "true" if all(v in (0.0, 1.0) for v, _ in parsed) else "positive"
        info["event_source"] = "detected"
    else:
        info["event_source"] = "declared"
    info["event"] = event
    if unit and parsed and not any(c.strip() for _, c in parsed):
        info["refused"] = _empty_unit_message(unit, len(parsed))
        return [], None, info
    # +1 for the event, -1 against it, 0 for a tie (signed columns only: a boolean 0 is a non-event, not a tie)
    signs: list[tuple[int, str]] = []
    n_ties = 0
    for v, c in parsed:
        s = (1 if v >= 0.5 else -1) if event == "true" else (1 if v > 0 else -1 if v < 0 else 0)
        if s == 0:
            n_ties += 1
            continue
        signs.append((s, c))
    by: dict[str, list[int]] = {}
    for s, c in signs:
        by.setdefault(c, []).append(s)
    units: list[tuple[float, float]] = []
    labels: list[str] = []
    n_unit_ties = 0
    for c, ss in by.items():
        total = sum(ss)
        if total == 0:
            n_unit_ties += 1
            continue
        units.append((float(1 if total > 0 else -1),) * 2)
        labels.append(c)
    most = max((len(ss) for ss in by.values()), default=0)
    info |= {
        "n_rows": len(parsed), "dropped": dropped,
        "n_ties_dropped": n_ties, "n_unit_ties_dropped": n_unit_ties,
        "n_success": sum(1 for u in units if u[1] > 0), "n_units": len(units),
        "unit_value": ("the row's own sign, one row per unit" if most <= 1 else
                       f"the majority sign over a unit's rows (up to {most} rows per unit); equal counts make the unit a tie "
                       f"and it is dropped"),
    }
    return units, (labels if unit else None), info


def _empty_unit_message(unit: str, n: int) -> str:
    return (f"resampling unit {unit!r} is a column of the data but is EMPTY "
            f"in all {n} usable row(s), so there is nothing to "
            f"cluster on. The unit was declared and the file does not carry "
            f"it -- which is a different problem from having too few groups, "
            f"and is not fixed by collecting more rows")


# ---------------------------------------------------------------------------------------------------------------------------
# POOLED SD AND POOLED SE (2026-10-28), docs/pooled_sd.md.
#
# The sd of replicate measurements pooled over groups: sqrt( sum_g (n_g - 1) s_g^2 / sum_g (n_g - 1) ). Added for "the run-to-run
# sd of a free energy, pooled over ligands", where a per-ligand sd from 3 runs spans about 0.52 to 6.3 times the true value and
# the pooled one, on 16 degrees of freedom, about 0.74 to 1.5 times.
#
# THE ESTIMATOR NEVER SEES ROWS. `bootstrap_ci` concatenates the clusters it draws and calls `estimate(xs, ys)`, so a group drawn
# twice would arrive as one group of twice the rows and its within-group sum of squares would be wrong. Each group is therefore
# collapsed to ONE pair (degrees of freedom, sum of squares) before the bootstrap, as `_sign_units` collapses a unit to one sign,
# and the bootstrap resamples those pairs: the groups, never the rows.
#
# THE TRIGGER RULE IS REPORTED, NEVER ENFORCED. `pooling_trigger` says whether the groups look poolable and the row carries each
# criterion with its value and threshold. It does not change the verdict.
#
# Everything below is standard library only (a test blocks any other import in this module), so the chi-square quantile, the F
# tail and the F-max critical value are built from `math.lgamma`, a series and two continued fractions.
# ---------------------------------------------------------------------------------------------------------------------------
#: `pool_check` (2026-10-28) is the claim "pooling these groups is justified": the same criteria as the `pool_justified` flag on the
#: pooled rows, adjudicated as a claim of its own (estimate 1 when no computed criterion fails, 0 when one does).
POOLED_METHODS = ("pooled_sd", "pooled_se", "pool_check")
#: The interval that decides a pooled_sd or pooled_se verdict when the claim does not declare `interval`: the exact chi-square interval below this
#: many groups, the bootstrap at or above it. Hesterberg (2015, arXiv 1411.5279) shows the bootstrap percentile interval is too narrow in small
#: samples and that the t interval beats it up to n = 34 (for the mean of an exponential population). Carrying that cut-off over to a pooled
#: sd resampled over groups is OUR CHOICE; the bootstrap here resamples groups, so the number of groups is its sample size.
POOL_BOOTSTRAP_MIN_GROUPS = 35
#: F-max with fewer than this many degrees of freedom per group is written on the row but cannot confirm pooling: at 1 df the 5% bound for 8
#: groups is a 217-fold sd ratio. OUR CHOICE, NO SOURCE (the bound is computed by `hartley_fmax_critical`).
POOL_FMAX_MIN_DF = 2
#: Hartley's F-max is tabulated at 5% (Hartley 1950). The level is the source's; applying it to unequal groups is our choice.
POOL_FMAX_ALPHA = 0.05
#: Brown-Forsythe p above this passes. 0.10 and not 0.05 because the test has almost no power at 2 or 3 replicates per group
#: (docs/pooled_sd.md): a failure to reject at 0.05 would be read as evidence of equal variance when it is not. OUR CHOICE, NO SOURCE.
POOL_BF_P_MIN = 0.10


def _gammainc_p(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x): series for x < a + 1, Lentz continued fraction above (Numerical Recipes 6.2)."""
    if x <= 0.0:
        return 0.0
    lg = math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(5000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return min(1.0, total * math.exp(-x + a * math.log(x) - lg))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 5000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return max(0.0, 1.0 - math.exp(-x + a * math.log(x) - lg) * h)


def chi2_cdf(x: float, df: float) -> float:
    """P(X <= x) for X chi-square with `df` degrees of freedom."""
    return _gammainc_p(df / 2.0, x / 2.0)


def chi2_ppf(p: float, df: float) -> float:
    """The chi-square quantile: x with P(X <= x) = p, by bisection on `chi2_cdf`. chi2_ppf(0.975, 16) is 28.845, chi2_ppf(0.025, 16) is 6.908."""
    if not (0.0 < p < 1.0) or df <= 0:
        raise ValueError(f"chi2_ppf needs 0 < p < 1 and df > 0, got p = {p}, df = {df}")
    lo, hi = 0.0, max(2.0 * df, 1.0)
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    return 0.5 * (lo + hi)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz, Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 5000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """The q quantile of Beta(a, b): the x with I_x(a, b) = q, by bisection on `_betainc` to machine precision (standard library only).
    beta_ppf(0.025, 8, 1) is 0.025 ** (1 / 8) = 0.630583, the lower Clopper-Pearson bound of 8 of 8."""
    if not 0.0 < q < 1.0:
        return 0.0 if q <= 0.0 else 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15:
            break
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float] | None:
    """The exact (Clopper-Pearson 1934) interval for a binomial share k / n: the alpha/2 quantile of Beta(k, n - k + 1) and the
    1 - alpha/2 quantile of Beta(k + 1, n - k), with the edges set exactly: the lower bound is 0 at k = 0 and the upper bound is 1
    at k = n. Its coverage is at least `level` for every true share. 8 of 8 gives [0.6306, 1], 5 of 10 [0.1871, 0.8129], 0 of 10
    [0, 0.3085]. None for n < 1 or k outside [0, n]."""
    if n < 1 or not 0 <= k <= n:
        return None
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else beta_ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1.0 - a, k + 1, n - k)
    return lo, hi


def _count_units(rows: list[dict], y: str, unit: str | None) -> dict:
    """The k of n for a `binomial_count` claim: one indicator per trial, a trial being one `resampling_unit` (or one row when no unit is
    declared). True is 1, true, yes; false is 0, false, no. A missing or non-finite value is dropped and counted. Any other value, a
    number other than 0 or 1 included, refuses the claim: a binomial count needs an indicator, and reading 0.7 as true would be a
    threshold nobody declared.

    ONE ROW PER UNIT, OR THE CLAIM IS REFUSED. The binomial assumes independent trials. Several rows under one unit (three end-state
    runs of one ligand) are not independent trials, and collapsing them (a majority, any, all) is a choice about the question, so it
    belongs in the rule that writes the column, not here."""
    info: dict = {"refused": None}
    vals: list[tuple[int, str]] = []
    dropped = 0
    for i, r in enumerate(rows):
        raw = r.get(y)
        text = str(raw).strip() if raw is not None else ""
        low = text.lower()
        if low in _MISSING_WORDS:
            dropped += 1
            continue
        if low in _TRUE_WORDS:
            v = 1
        elif low in _FALSE_WORDS:
            v = 0
        else:
            try:
                f = float(text)
            except ValueError:
                info["refused"] = (f"column {y!r} is not an indicator: row {i + 1} holds {text!r}. A binomial count reads 0/1, "
                                   f"true/false or yes/no")
                return info
            if not math.isfinite(f):
                dropped += 1
                continue
            if f not in (0.0, 1.0):
                info["refused"] = (f"column {y!r} is not an indicator: row {i + 1} holds {text!r}, which is neither 0 nor 1. A binomial "
                                   f"count does not threshold a number; write the 0/1 column in the rule that makes the table")
                return info
            v = int(f)
        vals.append((v, str(r.get(unit, "")).strip() if unit else str(i)))
    if unit and vals and not any(c for _, c in vals):
        info["refused"] = _empty_unit_message(unit, len(vals))
        return info
    if unit:
        per = Counter(c for _, c in vals)
        many = sorted(c for c, m in per.items() if m > 1)
        if many:
            info["refused"] = (f"{len(many)} {unit}(s) carry more than one row ({', '.join(many[:5])}{', ...' if len(many) > 5 else ''}; up to "
                               f"{max(per.values())} rows each). A binomial count takes one independent trial per {unit}; rows within a "
                               f"{unit} are not independent trials, and how to collapse them to one is a choice for the rule that writes "
                               f"the table")
            return info
    info |= {"n_rows": len(vals), "dropped": dropped, "n_success": sum(v for v, _ in vals), "n_units": len(vals)}
    return info


# ---------------------------------------------------------------------------------------------------------------------------
# TOP-N RECOVERY (2026-11-12): "k of the K actives are among the N compounds the score shortlists, out of M screened". The count is
# read from the table of the whole library, never declared, and its chance distribution is HYPERGEOMETRIC, not binomial: the actives
# compete for the same N places, so they are not independent trials. `binomial_count` on a table of actives alone reports a p against
# 0.5, which answers no question a shortlist poses (0 of 7 in the first 50 of 357 reads p = 0.016 there and is the likeliest count
# under chance), and a binomial against N / M overstates the upper tail by 3% at 3 of 7 and 46% at 7 of 7. docs/topn_recovery.md.
# ---------------------------------------------------------------------------------------------------------------------------


def hypergeom_tails(k: int, n_marked: int, n_drawn: int, n_total: int) -> tuple[float, float] | None:
    """(P(X <= k), P(X >= k)) for X the number of marked items among `n_drawn` taken without replacement from `n_total` of which
    `n_marked` are marked. Exact: integer binomial coefficients, one division. None when the arguments describe no such draw or k is
    outside the support. hypergeom_tails(3, 7, 50, 357)[1] is 0.060181, the chance that 3 or more of 7 actives land in the first 50
    of 357; it is scipy.stats.hypergeom.sf(k - 1, n_total, n_marked, n_drawn) and Fisher's exact test, alternative "greater"."""
    if not (0 <= n_marked <= n_total and 0 <= n_drawn <= n_total and n_total >= 1):
        return None
    k_min, k_max = max(0, n_drawn - (n_total - n_marked)), min(n_marked, n_drawn)
    if not k_min <= k <= k_max:
        return None
    ways = {i: math.comb(n_marked, i) * math.comb(n_total - n_marked, n_drawn - i) for i in range(k_min, k_max + 1)}
    total = math.comb(n_total, n_drawn)
    return (sum(v for i, v in ways.items() if i <= k) / total, sum(v for i, v in ways.items() if i >= k) / total)


def _indicator(raw) -> int | None | str:
    """One cell as an indicator: 1, 0, None for a missing or non-finite value, or the offending text for anything else (a number
    other than 0 or 1 included). The same reading `_count_units` gives a binomial_count column."""
    text = str(raw).strip() if raw is not None else ""
    low = text.lower()
    if low in _MISSING_WORDS:
        return None
    if low in _TRUE_WORDS:
        return 1
    if low in _FALSE_WORDS:
        return 0
    try:
        f = float(text)
    except ValueError:
        return text
    if not math.isfinite(f):
        return None
    return int(f) if f in (0.0, 1.0) else text


def _topn_recovery_share(xs: list[float], ys: list[float]) -> float | None:
    """The share of the positives (x >= 0.5) that are shortlisted (y >= 0.5): recall at the shortlist depth. None with no positive."""
    hit = [y for x, y in zip(xs, ys, strict=True) if x >= 0.5]
    return sum(1.0 for v in hit if v >= 0.5) / len(hit) if hit else None


def _topn_units(rows: list[dict], x: str, y: str, unit: str | None) -> dict:
    """The 2 x 2 table of a `topn_recovery` claim, counted from the library table: `x` the label (active or not) and `y` the shortlist
    indicator, both 0/1, true/false or yes/no, ONE ROW PER COMPOUND. A row missing either value is dropped and counted. Any other
    value refuses the claim, and so does a unit with more than one row: several poses of one compound are not several compounds, and
    which pose decides membership of the shortlist is a choice for the rule that writes the table."""
    info: dict = {"refused": None}
    vals: list[tuple[int, int, str]] = []
    dropped = 0
    for i, r in enumerate(rows):
        a, b = _indicator(r.get(x)), _indicator(r.get(y))
        for col, v in ((x, a), (y, b)):
            if isinstance(v, str):
                info["refused"] = (f"column {col!r} is not an indicator: row {i + 1} holds {v!r}. topn_recovery reads 0/1, true/false or "
                                   f"yes/no in both columns; the shortlist is cut by the rule that writes the table, never thresholded here")
                return info
        if a is None or b is None:
            dropped += 1
            continue
        assert isinstance(a, int) and isinstance(b, int)
        vals.append((a, b, str(r.get(unit, "")).strip() if unit else str(i)))
    if unit and vals and not any(c for _, _, c in vals):
        info["refused"] = _empty_unit_message(unit, len(vals))
        return info
    if unit:
        per = Counter(c for _, _, c in vals)
        many = sorted(c for c, m in per.items() if m > 1)
        if many:
            info["refused"] = (f"{len(many)} {unit}(s) carry more than one row ({', '.join(many[:5])}{', ...' if len(many) > 5 else ''}; up to "
                               f"{max(per.values())} rows each). topn_recovery counts compounds: one row per {unit}, the whole library, "
                               f"and how several rows of one {unit} become one is a choice for the rule that writes the table")
            return info
    info |= {"n_rows": len(vals), "dropped": dropped,
             "n_library": len(vals), "n_actives": sum(a for a, _, _ in vals), "n_shortlist": sum(b for _, b, _ in vals),
             "n_success": sum(1 for a, b, _ in vals if a and b)}
    return info


def f_sf(f: float, d1: float, d2: float) -> float:
    """P(F > f) for F with (d1, d2) degrees of freedom: the p value of an F statistic. f_sf(39.0, 2, 2) is 0.025."""
    if f <= 0.0:
        return 1.0
    return _betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def _chi2_pdf(x: float, df: float) -> float:
    return math.exp((df / 2.0 - 1.0) * math.log(x) - x / 2.0 - (df / 2.0) * math.log(2.0) - math.lgamma(df / 2.0))


_FMAX_CACHE: dict[tuple, float] = {}


def hartley_fmax_critical(k: int, df: int, alpha: float = POOL_FMAX_ALPHA) -> float:
    """The critical value of Hartley's F-max: the (1 - alpha) quantile of max(s_i^2) / min(s_i^2) over `k` independent variance estimates on `df` degrees of freedom each.

    Computed from its definition, P(max/min <= r) = k * integral f(s) [F(r s) - F(s)]^(k-1) ds with f and F the density and distribution of
    chi-square(df)/df, by the midpoint rule in t = sqrt(s) (which removes the singularity at 0 for df = 1) and bisection on r, so it needs no
    table. For k = 2 it is the 1 - alpha/2 quantile of F(df, df), which the tests check. HARTLEY'S TABLE ASSUMES EQUAL DEGREES OF FREEDOM; unequal
    groups are handled by the caller (docs/pooled_sd.md).
    """
    key = (int(k), int(df), float(alpha))
    if key in _FMAX_CACHE:
        return _FMAX_CACHE[key]
    if k < 2 or df < 1 or not (0.0 < alpha < 1.0):
        raise ValueError(f"hartley_fmax_critical needs k >= 2, df >= 1 and 0 < alpha < 1, got k = {k}, df = {df}, alpha = {alpha}")
    n = 500
    tmax = math.sqrt(chi2_ppf(1.0 - 1e-13, df) / df)
    h = tmax / n
    ss = [((i + 0.5) * h) ** 2 for i in range(n)]
    dens = [df * _chi2_pdf(df * s, df) * 2.0 * math.sqrt(s) for s in ss]
    cdf1 = [chi2_cdf(df * s, df) for s in ss]

    def prob(r: float) -> float:
        tot = 0.0
        for d, c1, s in zip(dens, cdf1, ss, strict=True):
            diff = chi2_cdf(df * r * s, df) - c1
            if diff > 0.0:
                tot += d * diff ** (k - 1)
        return k * tot * h

    lo, hi = 1.0, 2.0
    while prob(hi) < 1.0 - alpha and hi < 1e7:
        lo, hi = hi, hi * 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if prob(mid) < 1.0 - alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-7 * lo:
            break
    _FMAX_CACHE[key] = 0.5 * (lo + hi)
    return _FMAX_CACHE[key]


def _med(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def brown_forsythe(groups: list[list[float]]) -> dict:
    """Brown-Forsythe test of equal variances: a one-way F test on the absolute deviations of each value from its group's MEDIAN (Levene's
    test with the median centre, Brown and Forsythe 1974). Groups with fewer than 2 values are ignored. Needs at least 3 groups of 2 or
    more. Returns `statistic`, `p_value`, `df1`, `df2`, `n_groups`, or `not_computed` with the reason.

    IT HAS ALMOST NO POWER AT 2 OR 3 REPLICATES. With 2 values per group both absolute deviations from the median are equal, and with 3 one is
    always 0, so the deviations carry little within-group spread (Hines and Hines 2000, who remove these structural zeros; that
    modification is not implemented here). If every group has exactly 2 values the within-group sum of squares is 0 and the statistic is
    undefined, which is returned as `not_computed`, never as a p of 0 or 1.
    """
    gs = [list(g) for g in groups if len(g) >= 2]
    k = len(gs)
    if k < 3:
        return {"n_groups": k, "not_computed": f"needs at least 3 groups with 2 or more replicates, has {k}"}
    zs = [[abs(v - _med(g)) for v in g] for g in gs]
    n = sum(len(z) for z in zs)
    zbars = [math.fsum(z) / len(z) for z in zs]
    zbar = math.fsum(v for z in zs for v in z) / n
    ssb = math.fsum(len(z) * (zb - zbar) ** 2 for z, zb in zip(zs, zbars, strict=True))
    ssw = math.fsum((v - zb) ** 2 for z, zb in zip(zs, zbars, strict=True) for v in z)
    # "no within-group spread" is judged RELATIVE to the size of the deviations: with groups of 2 the two deviations are equal in exact
    # arithmetic but differ by rounding (1e-17 relative), and a fixed tiny threshold read that as spread and returned a p of 1e-114
    zmax = max(max(z) for z in zs)
    if n - k <= 0 or ssw <= n * (1e-9 * zmax) ** 2:
        return {"n_groups": k, "not_computed": "the absolute deviations from the group medians have no within-group spread (every group has 2 values, or the deviations are all equal), so the F statistic is undefined"}
    f = (ssb / (k - 1)) / (ssw / (n - k))
    return {"n_groups": k, "statistic": f, "p_value": f_sf(f, k - 1, n - k), "df1": k - 1, "df2": n - k}


def pooled_chi2_interval(sd: float, df: int, level: float = 0.95) -> tuple[float, float]:
    """The exact interval for a standard deviation estimated on `df` degrees of freedom, [sd sqrt(df / chi2_(1-a/2)), sd sqrt(df / chi2_(a/2))].
    It assumes normal replicates with ONE variance common to every group; the bootstrap interval assumes neither. chi-square, df 16: [0.745, 1.522] times sd."""
    a = (1.0 - level) / 2.0
    return sd * math.sqrt(df / chi2_ppf(1.0 - a, df)), sd * math.sqrt(df / chi2_ppf(a, df))


def pooling_trigger(group_sds: dict, group_dfs: dict, bf: dict, protocols: list | None = None) -> dict:
    """The TRIGGER RULE: does the evidence in the groups themselves allow their replicate variances to be pooled? Reported on the row; it never
    changes a verdict (docs/pooled_sd.md for each threshold and its source).

    Three criteria, each `pass`, `fail` or `not_computed`:
      fmax      the largest-to-smallest group variance is at most Hartley's F-max critical value at 5% for k groups and the mean
                degrees of freedom (rounded, at least 1). Unequal degrees of freedom are outside Hartley's table; the mean is OUR CHOICE.
      bf        the Brown-Forsythe p is above 0.10 (our choice; the test has almost no power at 2 or 3 replicates, so a pass is weak).
      protocol  every row in the groups used carries one value of the declared `protocol` column (time step, potential, ...). Not computed
                when the claim declares no `protocol` column: whether groups share one protocol is not a property of the numbers.
    `pool_justified` is true when no computed criterion failed; `pool_unchecked` names the criteria that could not be computed, so a true
    value never hides that one of them was not looked at.
    """
    crit: dict = {}
    sds = [group_sds[g] for g in group_sds]
    dfs = [group_dfs[g] for g in group_sds]
    k = len(sds)
    if k < 2:
        crit["fmax"] = {"result": "not_computed", "reason": "fewer than 2 groups"}
    elif max(sds) == 0.0:
        crit["fmax"] = {"result": "not_computed", "reason": "every group sd is 0"}
    else:
        df_used = max(1, int(round(sum(dfs) / k)))
        crit_value = hartley_fmax_critical(k, df_used, POOL_FMAX_ALPHA)
        lo = min(sds)
        ratio = None if lo == 0.0 else max(sds) / lo
        var_ratio = None if ratio is None else ratio ** 2
        ok = ratio is not None and var_ratio <= crit_value
        crit["fmax"] = {"result": "pass" if ok else "fail", "sd_ratio_max_min": ratio, "variance_ratio": var_ratio,
                        "threshold": crit_value, "k": k, "df_used": df_used, "alpha": POOL_FMAX_ALPHA,
                        "informative": df_used >= POOL_FMAX_MIN_DF,
                        **({"reason": "the smallest group sd is 0, so the ratio is infinite"} if ratio is None else {})}
    crit["fmax"]["source"] = ("Hartley 1950 (statistic and 5% level); the bound is computed from its definition; the mean df for unequal groups "
                              f"and the {POOL_FMAX_MIN_DF}-df floor for confirming are our choice, no source")
    if "p_value" in bf:
        crit["bf"] = {"result": "pass" if bf["p_value"] > POOL_BF_P_MIN else "fail", "p_value": bf["p_value"],
                      "threshold": POOL_BF_P_MIN, "informative": True}
    else:
        crit["bf"] = {"result": "not_computed", "reason": bf.get("not_computed", "not computed"), "informative": False}
    crit["bf"]["source"] = "Brown and Forsythe 1974 (the test); the threshold 0.10 is our choice, no source"
    if protocols is None:
        crit["protocol"] = {"result": "not_computed", "reason": "the claim declares no `protocol` column"}
    else:
        distinct = sorted(set(protocols))
        crit["protocol"] = {"result": "pass" if len(distinct) <= 1 else "fail", "distinct_values": distinct[:10],
                            "n_distinct": len(distinct)}
    crit["protocol"]["source"] = "our choice, no source (docs/pooled_sd.md gives the reasons)"
    failed = [n for n, c in crit.items() if c["result"] == "fail"]
    return {"pool_justified": not failed, "pool_failed": failed,
            "pool_unchecked": [n for n, c in crit.items() if c["result"] == "not_computed"], "pool_criteria": crit}


def pool_check_outcome(crit: dict) -> tuple[str, float | None, str]:
    """The verdict of a `pool_check` claim from the criteria of `pooling_trigger`: (verdict, estimate, why).

    refuted      a computed criterion failed (estimate 0); the text names each failed criterion with its number and threshold.
    agrees       no computed criterion failed AND at least one equal-variance criterion that can confirm was computed (estimate 1): the
                 Brown-Forsythe test, or F-max on at least POOL_FMAX_MIN_DF degrees of freedom per group.
    unverifiable no criterion failed but none of those could be computed, so the claim cannot be confirmed, only refuted (estimate 1 if
                 anything was computed, none if nothing was). This is the case at 2 runs per group: the protocol criterion alone is a
                 declaration about the settings, not a test of the variances.
    """
    failed = [n for n, c in crit.items() if c["result"] == "fail"]
    if failed:
        parts = []
        for n in failed:
            c = crit[n]
            if n == "fmax":
                parts.append("fmax: variance ratio " + (f"{c['variance_ratio']:.4g}" if c.get("variance_ratio") is not None else "infinite (smallest group sd is 0)")
                             + f" exceeds the F-max bound {c['threshold']:.4g} (5%, k = {c['k']}, df {c['df_used']})")
            elif n == "bf":
                parts.append(f"bf: Brown-Forsythe p = {c['p_value']:.3g} is not above {c['threshold']}")
            else:
                parts.append(f"protocol: {c['n_distinct']} distinct values {c['distinct_values']} in the groups used")
        return REFUTED, 0.0, "pooling is NOT justified, " + "; ".join(parts)
    computed = [n for n, c in crit.items() if c["result"] != "not_computed"]
    can_confirm = [n for n in ("fmax", "bf") if crit[n]["result"] == "pass" and crit[n].get("informative")]
    unchecked = [n for n, c in crit.items() if c["result"] == "not_computed"]
    tail = f" (not computed: {', '.join(unchecked)})" if unchecked else ""
    if can_confirm:
        return AGREES, 1.0, f"no computed criterion failed ({', '.join(computed)}); confirmed by {', '.join(can_confirm)}{tail}"
    if computed:
        return UNVERIFIABLE, 1.0, ("no computed criterion failed, but none that can confirm equal variances was computed "
                                   f"({', '.join(computed)} passed; Brown-Forsythe needs 3 groups with a group of 3 or more runs, "
                                   f"F-max needs {POOL_FMAX_MIN_DF} or more degrees of freedom per group), so pooling can be refuted here but not confirmed{tail}")
    return UNVERIFIABLE, None, f"no criterion could be computed{tail}"


def _pooled_groups(rows: list[dict], y: str, group: str, protocol: str | None) -> tuple[list[tuple[float, float]], dict]:
    """Each group collapsed to ONE pair (degrees of freedom n_g - 1, sum of squares about the group mean), and everything that decides whether pooling is justified.

    A group with a single value has no within-group spread and is DROPPED and counted (`n_groups_dropped`, `groups_dropped`). A value that is
    not finite (NaN, inf) or not a number (blank, text) REFUSES the claim (`info["refused"]`, so the verdict is `unverifiable`), naming the
    count and the groups: it is a failed cell with a reason, never a number, and dropping it would silently shrink a group. A row with an
    empty group label is still dropped and counted in `dropped_no_group`.
    """
    by: dict[str, list[float]] = {}
    protos: dict[str, set] = {}
    dropped_no_group = 0
    bad: dict[str, int] = {}                  # group -> cells that are not finite numbers
    n_nonfinite = n_not_number = 0
    for r in rows:
        label = str(r.get(group, "")).strip()
        try:
            v = float(r[y])
        except (ValueError, KeyError, TypeError):
            n_not_number += 1
            bad[label or "(no group)"] = bad.get(label or "(no group)", 0) + 1
            continue
        if v != v or abs(v) == float("inf"):
            n_nonfinite += 1
            bad[label or "(no group)"] = bad.get(label or "(no group)", 0) + 1
            continue
        if not label:
            dropped_no_group += 1
            continue
        by.setdefault(label, []).append(v)
        if protocol:
            protos.setdefault(label, set()).add(str(r.get(protocol, "")))
    info: dict = {"refused": None, "n_nonfinite": n_nonfinite, "n_not_a_number": n_not_number, "dropped_no_group": dropped_no_group,
                  "n_rows": sum(len(v) for v in by.values())}
    if bad:
        shown = ", ".join(f"{g} ({n})" for g, n in sorted(bad.items())[:10]) + (f", and {len(bad) - 10} more groups" if len(bad) > 10 else "")
        kinds = ", ".join(p for p in (f"{n_nonfinite} non-finite (NaN or inf)" if n_nonfinite else "",
                                      f"{n_not_number} blank or not a number" if n_not_number else "") if p)
        info["refused"] = (f"column {y!r} holds {kinds} value(s), in groups {shown}. A non-finite value is a failed cell with a reason, never a "
                           f"number, and dropping it would shrink the group it came from, so no pooled value is computed. Record the failed "
                           f"cells with their reasons in the rule that writes the table and leave them out of it")
        return [], info
    if rows and not by and dropped_no_group:
        info["refused"] = _empty_unit_message(group, dropped_no_group)
        return [], info
    used = {g: v for g, v in by.items() if len(v) >= 2}
    singles = [g for g, v in by.items() if len(v) < 2]
    pairs: list[tuple[float, float]] = []
    sds: dict[str, float] = {}
    dfs: dict[str, int] = {}
    for g, v in used.items():
        m = math.fsum(v) / len(v)
        ss = math.fsum((x - m) ** 2 for x in v)
        pairs.append((float(len(v) - 1), ss))
        dfs[g] = len(v) - 1
        sds[g] = math.sqrt(ss / (len(v) - 1))
    info |= {
        "group": group, "n_groups_used": len(used), "n_groups_dropped": len(singles), "groups_dropped": sorted(singles),
        "df": sum(dfs.values()), "n_rows_used": sum(len(v) for v in used.values()),
        "group_n": {g: len(v) for g, v in used.items()}, "group_sd": {g: round(s, 6) for g, s in sds.items()},
        # the per-group means, so that an sd which grows with the mean (Bland's first check) can be seen in the row
        "group_mean": {g: round(math.fsum(v) / len(v), 6) for g, v in used.items()},
        "mean_replicates": (sum(len(v) for v in used.values()) / len(used)) if used else None,
        "harmonic_replicates": (len(used) / sum(1.0 / len(v) for v in used.values())) if used else None,
        "_values": list(used.values()), "_sds": sds, "_dfs": dfs,
        "_protocols": ([p for g in used for p in protos.get(g, set())] if protocol else None),
    }
    return pairs, info


def _pooled_sd(xs: list[float], ys: list[float]) -> float | None:
    """sqrt( sum SS_g / sum df_g ): xs are the groups' degrees of freedom, ys their sums of squares (one pair per group, see `_pooled_groups`)."""
    df = sum(xs)
    return math.sqrt(sum(ys) / df) if df > 0 else None


REPLICATE_MEANS = ("harmonic", "arithmetic")
INTERVALS = ("chi2", "bootstrap")


def effective_replicates(dfs: list[float], how: str = "harmonic") -> float:
    """The number of replicates per group that turns a pooled sd into a standard error of a group mean, from the groups' degrees of freedom
    (n_g = df_g + 1). HARMONIC (default): k / sum(1/n_g). Each group's mean has variance sigma^2 / n_g, and the mean of those variances over
    the k groups, which each count once, is sigma^2 / harmonic mean, so the harmonic mean is the n that reproduces the average standard-error
    variance (a derivation; Laurencelle 2017 uses the same n_h for equally weighted groups of unequal size). ARITHMETIC: sum(n_g) / k, the
    alternative, equal to the harmonic mean when the groups are balanced and larger (a smaller se) when they are not."""
    ns = [d + 1.0 for d in dfs]
    return len(ns) / sum(1.0 / n for n in ns) if how == "harmonic" else sum(ns) / len(ns)


def _pooled_se_with(how: str):
    def estimate(xs: list[float], ys: list[float]) -> float | None:
        """The pooled sd divided by sqrt of the effective replicate count (`effective_replicates`): xs and ys as for `_pooled_sd`."""
        df = sum(xs)
        if df <= 0 or not xs:
            return None
        return math.sqrt(sum(ys) / df) / math.sqrt(effective_replicates(xs, how))
    return estimate


_pooled_se = _pooled_se_with("harmonic")
_pooled_se_arithmetic = _pooled_se_with("arithmetic")


def _pool_check_placeholder(xs: list[float], ys: list[float]) -> float | None:
    """`pool_check` is deterministic and judged directly in `check()` (nothing is resampled); the entry exists so the method is known and its kind checked."""
    return None


def _pooled_diagnostics(info: dict, method: str, est: float, how: str = "harmonic") -> dict:
    """The two intervals, Satterthwaite's effective degrees of freedom, the sd ratio, Brown-Forsythe and the trigger rule, for the row."""
    df = info["df"]
    sds, dfs = info["_sds"], info["_dfs"]
    # the interval for the estimate itself (the sd, or the se, whose interval has the same factors), then the Satterthwaite variant
    lo, hi = pooled_chi2_interval(est, df)
    out: dict = {"ci_chi2_lo": round(lo, 6), "ci_chi2_hi": round(hi, 6),
                 "ci_chi2_note": "exact chi-square 95% interval on `df` degrees of freedom; assumes normal replicates with one common variance"}
    c = 1.0
    if method == "pooled_se":
        c = math.sqrt(effective_replicates(list(dfs.values()), how))
        out |= {"pooled_sd": round(est * c, 6), "replicate_mean": how, "effective_replicates": round(c * c, 6),
                "pooled_note": f"pooled se = pooled sd / sqrt({how} mean of the replicates per group over the groups used)"}
    else:
        out["pooled_note"] = "pooled sd = sqrt(sum (n_g - 1) s_g^2 / sum (n_g - 1)) over the groups with 2 or more values"
    var = (est * c) ** 2
    denom = sum((sds[g] ** 2 * dfs[g] / df) ** 2 / dfs[g] for g in sds)
    out["df_satterthwaite"] = round(var ** 2 / denom, 4) if denom > 0 else None
    if out["df_satterthwaite"]:
        slo, shi = pooled_chi2_interval(est, out["df_satterthwaite"])
        out |= {"ci_satterthwaite_lo": round(slo, 6), "ci_satterthwaite_hi": round(shi, 6),
                "ci_satterthwaite_note": "the chi-square interval on the Satterthwaite effective df (Satterthwaite 1946), for groups whose variances differ; "
                                         "approximate, and the weights use the sample variances; reported, never used for the verdict"}
    lo_sd, hi_sd = min(sds.values()), max(sds.values())
    out["sd_ratio_max_min"] = None if lo_sd == 0.0 else round(hi_sd / lo_sd, 6)
    bf = brown_forsythe(info["_values"])
    out["brown_forsythe"] = bf | {"low_power_note": "at 2 or 3 replicates per group this test has almost no power: a large p is not evidence of equal variance"}
    out |= pooling_trigger(sds, dfs, bf, info["_protocols"])
    return out
# ---------------------------------------------------------------------------------------------------------------------------


# STRATIFIED dF OVER TWO WELLS (2026-11-09), docs/RULES.md.
#
# A free energy from a lambda ladder that stayed in one conformational well W of the end state A is conditional on W:
# dF_W = -kT ln (Z_B,W / Z_A,W). The free energy of the whole end state weights the wells by their populations in A,
#     dF = -kT ln sum_W p_W exp(-dF_W / kT),
# so with two wells (0, the well the reference ladder sampled, and 1, a second one) and delta = dF_1 - dF_0,
#     dF - dF_0 = -kT ln (1 - p + p exp(-delta / kT)),   p = p_1 / (p_0 + p_1).
# The shift lies between 0 (p = 0) and delta (p = 1) for every p. The two inputs come from different experiments with different
# independent units: p from frames of plain MD of A (unit: the replica), delta from per-unit estimates of dF_1 - dF_0 (for example
# ten times each window increment's difference between two ladders, unit: the increment). The interval resamples BOTH, each over its
# own unit, in every draw; the row also carries the interval from each source alone, so a reader sees which one limits the answer.
# x names the frames' well column (1 = well 1, 0 = well 0, anything else = neither well: counted, reported, left out, which is the
# renormalisation over the two wells); y names the delta column of the other dataset; `kT` (required) is in y's unit;
# `y_resampling_unit` names y's unit (absent: each y row is a unit); `offset` (optional) names a column of y's dataset, constant,
# added to the estimate (the reference minus a comparison value, so that the estimate is the stratified reference against it).
STRATIFIED_METHODS = ("stratified_dF",)


def stratified_shift(p: float, delta: float, kT: float) -> float:
    """-kT ln(1 - p + p exp(-delta / kT)), as a log-sum-exp so a large |delta| / kT does not overflow. 0 at p = 0, delta at p = 1."""
    if not (0.0 <= p <= 1.0) or not (kT > 0) or not math.isfinite(delta):
        raise ValueError(f"p = {p!r} must lie in [0, 1], kT = {kT!r} must be positive, delta = {delta!r} finite")
    terms = []
    if p < 1.0:
        terms.append(math.log1p(-p))
    if p > 0.0:
        terms.append(math.log(p) - delta / kT)
    m = max(terms)
    return -kT * (m + math.log(sum(math.exp(t - m) for t in terms)))


def _stratified_inputs(xrows: list[dict], x: str, xunit: str | None, yrows: list[dict], y: str, yunit: str | None,
                       offset: str | None) -> dict:
    """Per x unit [n_0, n_1, n_other], per y unit the list of delta values, and the offset. `refused` names why no estimate exists."""
    info: dict = {"refused": None}
    xu: dict[str, list[int]] = {}
    for i, r in enumerate(xrows):
        u = str(r.get(xunit, "")) if xunit else f"row{i}"
        c = xu.setdefault(u, [0, 0, 0])
        try:
            v = float(r[x])
        except (ValueError, TypeError, KeyError):
            c[2] += 1
            continue
        if v == 1.0:
            c[1] += 1
        elif v == 0.0:
            c[0] += 1
        else:
            c[2] += 1
    yu: dict[str, list[float]] = {}
    bad = []
    for i, r in enumerate(yrows):
        u = str(r.get(yunit, "")) if yunit else f"row{i}"
        try:
            v = float(r[y])
        except (ValueError, TypeError, KeyError):
            v = float("nan")
        if not math.isfinite(v):
            bad.append(u)
            continue
        yu.setdefault(u, []).append(v)
    n0 = sum(c[0] for c in xu.values())
    n1 = sum(c[1] for c in xu.values())
    no = sum(c[2] for c in xu.values())
    info |= {"_xu": xu, "_yu": yu, "n_rows": len(xrows), "n_first_well": n0, "n_second_well": n1, "n_other": no,
             "share_other": round(no / len(xrows), 6) if xrows else None, "n_units": len(xu), "n_units_y": len(yu),
             "n_rows_y": len(yrows)}
    if bad:
        info["refused"] = (f"{len(bad)} value(s) of {y} are blank, non-numeric or non-finite (in {', '.join(sorted(set(bad))[:5])}): a delta "
                           f"that failed is a failed cell with a reason, not a value, and dropping it would change the estimate")
        return info
    if n0 + n1 == 0:
        info["refused"] = f"no row of {x} is 0 or 1, so neither well is populated and p is undefined"
        return info
    if offset is not None:
        if not yrows or offset not in yrows[0]:
            info["refused"] = f"offset = {offset!r} is not a column of y's dataset"
            return info
        try:
            vals = {float(r[offset]) for r in yrows}
        except (ValueError, TypeError):
            info["refused"] = f"offset column {offset!r} holds a non-numeric value"
            return info
        if len(vals) != 1 or not all(math.isfinite(v) for v in vals):
            info["refused"] = f"offset column {offset!r} must hold one finite value on every row; it holds {len(vals)} distinct value(s)"
            return info
        info["offset"] = vals.pop()
    else:
        info["offset"] = 0.0
    return info


def stratified_bootstrap(xu: dict, yu: dict, kT: float, offset: float, level: float = 0.95) -> dict | None:
    """Percentile interval of offset + stratified_shift(p, delta, kT), every draw resampling the x units AND the y units with
    replacement, and the intervals from each source alone (the other held at its point value). None with fewer than 3 units on
    either side, or when more than half the draws have no frame in either well."""
    xs, ys = list(xu.values()), list(yu.values())
    if len(xs) < 3 or len(ys) < 3:
        return None
    n0 = sum(c[0] for c in xs)
    n1 = sum(c[1] for c in xs)
    flat = [v for vs in ys for v in vs]
    p0, d0 = n1 / (n0 + n1), sum(flat) / len(flat)
    rng = random.Random(SEED)
    both, ponly, donly, ps, ds = [], [], [], [], []
    for _ in range(N_BOOT):
        a = b = 0
        for _ in range(len(xs)):
            c = rng.choice(xs)
            a += c[0]
            b += c[1]
        s = n = 0.0
        for _ in range(len(ys)):
            vs = rng.choice(ys)
            s += sum(vs)
            n += len(vs)
        d = s / n
        ds.append(d)
        donly.append(offset + stratified_shift(p0, d, kT))
        if a + b == 0:
            continue
        p = b / (a + b)
        ps.append(p)
        both.append(offset + stratified_shift(p, d, kT))
        ponly.append(offset + stratified_shift(p, d0, kT))
    if len(both) < N_BOOT // 2:
        return None
    alpha = (1 - level) / 2

    def pct(v: list[float]) -> tuple[float, float]:
        v = sorted(v)
        return v[int(alpha * len(v))], v[min(int((1 - alpha) * len(v)), len(v) - 1)]

    return {"p": p0, "delta": d0, "ci": pct(both), "ci_population_only": pct(ponly), "ci_delta_only": pct(donly),
            "ci_p": pct(ps), "ci_delta": pct(ds), "n_draws": len(both)}


def _stratified_placeholder(xs: list[float], ys: list[float]) -> float | None:
    """Never called on data: stratified_dF reads two datasets and is computed by its own branch of `check` (see the block above)."""
    return None


ESTIMATORS: dict[str, tuple] = {
    "pearson": (pearson, "correlation"),
    "spearman": (spearman, "correlation"),
    # KENDALL TAU-B (2026-10-27): the tie-corrected rank correlation, same bootstrap over the resampling unit, same permutation p, and
    # `tie_tol` as for spearman. Its R squared is tau-b squared, a squared rank correlation and not a share of variance.
    "kendall": (kendall, "correlation"),
    # ROC AUC (2026-11-12): "the score puts the known binders near the top", for screening sets where
    # the hits are a small share of the rows. `x` is the binary label, `y` the score with higher
    # meaning more likely positive. Same bootstrap over the resampling unit; its permutation p is
    # centred on 0.5, not 0, because that is its null. No R squared.
    "roc_auc": (roc_auc, "correlation"),
    # BEDROC (2026-11-12): early recognition, each positive weighted by exp(-alpha * k / N) at list position k. Same `x` label and `y`
    # score as roc_auc, same bootstrap over the resampling unit. `alpha` is a REQUIRED claim key with no default, recorded on the row;
    # the entry here returns None until `check` binds it. Its permutation p is centred on the random-ranking expectation at the
    # table's active fraction, which is not 0. No R squared. Bounded 0 to 1 at any active fraction: it does not saturate.
    "bedroc": (_bedroc_needs_alpha, "correlation"),
    # ROC AUC DIFFERENCE (2026-11-20, proposal P36): "this score separates the actives BETTER THAN that one does", the
    # paired difference of two AUROCs on identical rows. ONE OF THE TWO THREE-COLUMN METHODS (`PAIRED_DIFFERENCE_METHODS`;
    # it was the only one until `bedroc_difference` on 2026-11-22): `x` the label, `y` the score and
    # `baseline = "<dataset>.<column>"` the column it is compared against, which `_paired` packs into `y` as a pair so a
    # resample draws a row with BOTH of its scores and the pairing survives (see `roc_auc_difference`). Positive means `y`
    # beats `baseline`. Interval: the same paired bootstrap over the resampling unit. Its p is NOT the label permutation
    # but `paired_swap_p`, whose null is that the two columns are interchangeable. No R squared.
    "roc_auc_difference": (roc_auc_difference, "correlation"),
    # BEDROC DIFFERENCE (2026-11-22): the same paired construction on the metric a screen is actually used by, "this score puts the
    # actives HIGHER UP THE LIST than that one does". `x` the label, `y` the score, `baseline` the column it is compared against, and
    # `alpha` REQUIRED and applied identically to both columns (the entry here returns None until `check` binds it, as bedroc's does).
    # Interval: the same paired bootstrap over the resampling unit. p: `paired_swap_p` with this statistic. Span [-1, 1], known answer
    # exactly 0.0 on a column against a copy of itself. The random expectation each BEDROC is read against is shared by both columns on
    # identical rows and cancels in the difference, so the null is 0 (see `bedroc_difference`). No R squared.
    "bedroc_difference": (_bedroc_difference_needs_alpha, "correlation"),
    # VALUE CLAIMS (2026-10-01). Every prediction is adjudicated here, not in an
    # analysis script, including the ones that are not correlations: an sd, a
    # variance share, a mean error, a count that passes a rule. They take the
    # same bootstrap over the declared resampling unit. `x` names the column a
    # one-column method ignores (write x = y) or the total of `var_share`.
    "mean": (_mean, "value"),
    "median": (_median, "value"),
    "sd": (_sd, "value"),
    "var_share": (_var_share, "value"),
    "fraction": (_fraction, "value"),
    # F1 (2026-10-21): a RETRIEVAL claim, "the shortlist this score produces
    # contains the hits". The only two-column boolean method: `x` is the
    # prediction and `y` is the truth, both read as indicators at >= 0.5, and
    # they do not commute. The shortlist depth that turns a score into a
    # prediction is declared by the rule that writes the column, never by this
    # estimator. Undefined when neither column has a positive, which returns
    # None and so reports `unverifiable` rather than 0.
    "f1": (_f1, "value"),
    # SIGN TEST (2026-10-08): "the correction lowers the error on most targets". One +1/-1 per resampling unit (see
    # `_sign_units`), the estimate is the share of units that are the event, the interval is the same bootstrap over
    # units, and the row also carries the exact two-sided binomial p against 0.5 (`sign_test_p`), the one value method
    # with a p. The verdict is still the interval against `claimed` or `claimed_range`.
    "sign_test": (_sign_fraction, "value"),
    # BINOMIAL COUNT (2026-11-09): "k of n units pass a rule", with the EXACT Clopper-Pearson interval in place of the bootstrap.
    # Added because the percentile bootstrap of a share collapses to [1, 1] at n of n and cannot say how little 8 of 8 shows:
    # Clopper-Pearson gives [0.631, 1]. One 0/1 row per resampling unit (several rows under one unit are refused, see
    # `_count_units`). Nothing is resampled. The row carries the exact two-sided binomial p against 0.5 (`sign_test_p`).
    "binomial_count": (_fraction, "value"),
    # TOP-N RECOVERY (2026-11-12): "k of the K actives are in the N-compound shortlist, out of M screened", from the table of the
    # whole library: `x` the label, `y` the shortlist indicator, one row per compound. The estimate is k / K with the Clopper-Pearson
    # interval (conservative here, see docs/topn_recovery.md); the p is the EXACT HYPERGEOMETRIC one, because actives competing for
    # N places are not independent trials and the chance share is N / M, not 0.5.
    "topn_recovery": (_topn_recovery_share, "value"),
    # POOLED SD AND SE (2026-10-28): the within-group sd of replicates pooled over groups, `group = "<column>"` naming the unit within which the
    # replicates sit (it must be the resampling unit; the bootstrap resamples groups). Interval: the same bootstrap, with the exact chi-square
    # interval beside it. The row carries the diagnostics for pooling and the trigger rule `pool_justified`, which never changes the verdict.
    "pooled_sd": (_pooled_sd, "value"),
    "pooled_se": (_pooled_se, "value"),
    # POOL CHECK (2026-10-28): "pooling these groups is justified", adjudicated as a claim (see `pool_check_outcome`). Deterministic: estimate 1 when
    # no computed criterion fails, 0 when one does, claimed 1; no interval, because nothing is resampled. Same `x = y`, `group`, `protocol` keys.
    "pool_check": (_pool_check_placeholder, "value"),
    # MEAN DIFFERENCE (2026-11-09): mean(y | x = 1) - mean(y | x = 0), x a 0/1 group column, the same bootstrap over the resampling unit.
    # For "two halves of the replicas agree within a tolerance": a claimed_range around 0. Any x other than 0 or 1 is refused.
    "mean_diff": (_mean_diff, "value"),
    # STRATIFIED dF (2026-11-09): the reference free energy re-weighted over two conformational wells (see the block above). x and y are
    # in DIFFERENT datasets with different units, each resampled over its own; no join. Keys: kT (required), y_resampling_unit, offset.
    "stratified_dF": (_stratified_placeholder, "value"),
}


def _snap(vs: list[float], tol: float) -> list[int]:
    """Each value as the integer index of the nearest multiple of `tol`, so values within rounding noise of one another compare equal."""
    return [math.floor(v / tol + 0.5) for v in vs]


def spearman_tie_tol(tol: float):
    """Spearman with values rounded to the nearest multiple of `tol` BEFORE ranking (the claim key `tie_tol`, absolute). A known-answer rank
    claim of 1.0 on estimates that are equal in exact arithmetic (eight seeds sharing one dF) otherwise has its ties ordered by 1e-9 rounding
    noise, which Spearman counts as discordance. Values that differ by much more than `tol` are not merged."""
    def estimate(xs: list[float], ys: list[float]) -> float | None:
        return spearman(_snap(xs, tol), _snap(ys, tol))
    return estimate


def kendall_tie_tol(tol: float):
    """Kendall tau-b with values rounded to the nearest multiple of `tol` BEFORE counting pairs (the claim key `tie_tol`, absolute): the
    same snapping as `spearman_tie_tol`, so values within rounding noise of one another are tied and not ordered by it."""
    def estimate(xs: list[float], ys: list[float]) -> float | None:
        return kendall(_snap(xs, tol), _snap(ys, tol))
    return estimate


def permutation_p(pairs: list[tuple[float, float]], clusters: list[str] | None,
                  estimate=pearson, n_perm: int = N_BOOT, centre: float = 0.0) -> tuple[float, float] | None:
    """Two-sided p for r, permuting WHOLE CLUSTERS rather than rows.

    Returns `(p, floor)` where `floor` is the smallest p this many permutations
    can produce, `1 / (n_perm + 1)`.

    WHY NOT THE ANALYTIC p. The usual t or Fisher p assumes every row is an
    independent draw. In this package the rows are usually not: 9,094 complexes
    over 4,127 groups, 493 fragments over 176 Butina clusters, 139 residues of
    one protein. An analytic p on those rows answers a question nobody asked and
    answers it with three too many digits of confidence, which is exactly the
    failure the clustered bootstrap already exists to refuse. So the null here is
    built the same way the interval is: by the resampling unit.

    HOW THE NULL IS BUILT, AND THE BUG THE FIRST VERSION HAD. Each cluster keeps
    its own ROW POSITIONS and receives the y values of another cluster, each
    position filled by a draw WITH REPLACEMENT from the donor's values, whatever
    the two sizes. x and y are then independent across clusters while each
    cluster's own y distribution is intact. With one row per cluster this is an
    exact permutation of y. With several rows per cluster it is not, even when
    donor and recipient have the same size: a recipient can receive one donor
    value twice. This docstring said "with replacement when the two differ in
    size" until 2026-11-12; the code never did that. Measured before choosing
    which to change (docs/screening_metrics_study.md): under a null with
    unit-level effects in both columns the share of p <= 0.05 was 0.037 to 0.055
    for the rule as coded and 0.043 to 0.055 for a within-block permutation at
    equal sizes, over 600 datasets per design, standard error 0.009. Neither is
    distinguishable from 0.05 or from the other, so the code stands and the
    sentence was corrected: changing the code would have moved the random stream
    of every clustered p for no measured gain.

    The first version concatenated the y blocks in cluster order and laid them
    against x in ROW order. On a file whose rows are contiguous by cluster those
    two orders agree and the null was correct. On an interleaved file they do
    not: on a 493-row table over 176 clusters the observed Spearman was +0.502
    and the UNSHUFFLED block layout gave +0.044, so the association was already
    destroyed before any shuffle and the null was a full row permutation. That
    null is too narrow and every p from it was too small. The fixture below
    pins the property that would have caught it on the first run: with the
    identity assignment the statistic must come back unchanged.

    THE FLOOR IS RETURNED, NOT HIDDEN. An add-one estimator cannot return zero,
    and a p sitting at its floor means the design ran out of permutations, not
    that the effect is overwhelming. Reporting the floor beside the value is what
    makes those two readable apart.
    """
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    obs = estimate(xs, ys)
    if obs is None:
        return None
    if clusters is None:
        idx_by: dict[str, list[int]] = {str(i): [i] for i in range(len(pairs))}
    else:
        idx_by = {}
        for i, c in enumerate(clusters):
            idx_by.setdefault(c, []).append(i)
    keys = list(idx_by)
    if len(keys) < 3:
        return None
    rng = random.Random(SEED + 1)
    hits = done = 0
    for _ in range(n_perm):
        sigma = keys[:]
        rng.shuffle(sigma)
        # EACH CLUSTER KEEPS ITS OWN ROW POSITIONS and receives another
        # cluster's y values. Sizes differ, so the donor block is sampled with
        # replacement to the recipient's length. That is what makes x and y
        # independent ACROSS clusters while leaving each cluster's own y
        # distribution intact.
        trial = list(ys)
        for recip, donor in zip(keys, sigma, strict=True):
            src = [ys[j] for j in idx_by[donor]]
            for pos in idx_by[recip]:
                trial[pos] = rng.choice(src)
        r = estimate(xs, trial)
        if r is None:
            continue
        done += 1
        # CENTRED ON THE STATISTIC'S OWN NULL. A correlation has no association at 0, so the
        # default compares magnitudes about 0; AUC has none at 0.5, and comparing |AUC| about 0
        # would call every result extreme. `centre` is 0.0 for every method but roc_auc and bedroc, so the
        # behaviour of the existing ones is unchanged. The comparison counts an exact tie in
        # distance as extreme, with a tolerance, because floating point loses the mirror value
        # (`as_extreme`, PERMUTATION_TIE_TOL).
        if as_extreme(r, obs, centre):
            hits += 1
    if done < n_perm // 2:
        return None
    return ((hits + 1) / (done + 1), 1.0 / (done + 1))


def bootstrap_ci(pairs: list[tuple[float, float]], clusters: list[str] | None,
                 level: float = 0.95, estimate=pearson
                 ) -> tuple[float, float, int] | None:
    """Percentile CI for r, resampling CLUSTERS when they are given.

    Returns `(lo, hi, n_effective)` where `n_effective` is the number of
    independent units actually resampled — the number a reader needs to judge
    the interval, and the one that differs from the row count by 30x here.
    """
    rng = random.Random(SEED)
    if clusters is None:
        groups = [[p] for p in pairs]
    else:
        by: dict[str, list[tuple[float, float]]] = {}
        for p, c in zip(pairs, clusters, strict=True):
            by.setdefault(c, []).append(p)
        groups = list(by.values())
    if len(groups) < 3:
        return None
    stats = []
    for _ in range(N_BOOT):
        drawn: list[tuple[float, float]] = []
        for _ in range(len(groups)):
            drawn.extend(rng.choice(groups))
        r = estimate([p[0] for p in drawn], [p[1] for p in drawn])
        if r is not None:
            stats.append(r)
    if len(stats) < N_BOOT // 2:
        return None                    # too often undefined to be an interval
    stats.sort()
    a = (1 - level) / 2
    return (stats[int(a * len(stats))],
            stats[min(int((1 - a) * len(stats)), len(stats) - 1)],
            len(groups))


def _ref(text: str, only: str | None) -> tuple[str, str]:
    """`"affinity.pKD"` -> `("affinity", "pKD")`.

    A bare name is allowed only when the project has ONE dataset. With two, a
    bare column name is ambiguous and the ambiguity would be resolved by
    whichever happened to be first — so it is refused instead.
    """
    if "." in text:
        did, _, col = text.partition(".")
        return did, col
    if only is None:
        raise DatasetError(
            f"{text!r} does not say which dataset it is in. With more than one "
            f"dataset a bare column name is ambiguous; write `dataset.column`.")
    return only, text


def _usable(v) -> float | None:
    """`v` as a finite float, or None where it is missing, unreadable or non-finite (which is not missing and not a value)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f or abs(f) == float("inf") else f


def _paired(rows: list[dict], x: str, y: str, unit: str | None, second: str | None = None):
    """Rows usable for BOTH columns, kept together, with the cluster label.

    Paired deliberately: filtering each column separately would compare
    `x[i]` against `y[j]`, which is a correlation between two different
    subsets and is not a number about anything.

    `second` (2026-11-20, `roc_auc_difference`) names a THIRD column, and then a row is kept only when all three of its
    values are usable and the pair comes back as `(x, (y, second))`. The second value travels INSIDE the pair because
    everything downstream resamples pairs and never looks inside them, so a drawn row carries both of its scores and the
    pairing a paired difference exists to exploit cannot be broken by the bootstrap. A row missing the third column is
    dropped from the claim entirely rather than from one column of it, for the reason the two-column form is paired.
    """
    pairs, clusters, dropped = [], [], 0
    for r in rows:
        if x not in r or y not in r or (second is not None and second not in r):
            dropped += 1
            continue
        a, b = _usable(r[x]), _usable(r[y])
        c = _usable(r[second]) if second is not None else 0.0
        if a is None or b is None or c is None:
            dropped += 1
            continue
        pairs.append((a, b) if second is None else (a, (b, c)))
        if unit:
            clusters.append(str(r.get(unit, "")))
    return pairs, (clusters if unit else None), dropped


def _hidden_gap(gap: float, bigger: str, smaller: str) -> float | None:
    """How far a refutation really is, WHEN THE LINE THAT REPORTS IT NO LONGER SHOWS IT.

    A verdict is decided on unrounded numbers and reported in rounded ones, and on 2026-11-17 six
    known-answer claims were read as a defect in this checker because of the difference: all six say
    "claimed 1.0 lies OUTSIDE the 95% CI [1.000, 1.000]", which is not a sentence anybody can act on.
    MEASURED, by re-running one of them on its own data in a copy:
    `A1_6kx4_aposcore_is_deterministic_KNOWN_ANSWER` scores the same poses twice, the two columns
    differ in the seventh decimal (20.82521521591532 against 20.825217903571236 in row 1), and over
    34,447 compounds rho falls short of 1 by 2.91e-11, which is 29 times KNOWN_ANSWER_TOL and far
    less than the printed three decimals. The refutation is real and the comparison is right about
    it; the line hiding its own margin is what was wrong. Two of the six carry the same finding in
    their own evidence note, with the mechanism and the repair (a `claimed_range`) worked out:
    `funnel_rerank_and_top1k/context.toml`, KA1b, 2026-11-16.

    `gap` is the distance outside at full precision. `bigger` and `smaller` are the two numbers as
    the `why` line prints them, in the order that ought to show the separation. Returns the gap when
    the printed pair no longer does, else None: the margin is spelled out exactly where the rounding
    would otherwise have swallowed it, and every other refutation reads as it always has.
    """
    if not (gap > 0) or not math.isfinite(gap):
        return None
    try:
        if float(bigger) > float(smaller):
            return None                 # the printed line already shows it
    except ValueError:
        return None
    return gap


def _margin_text(gap: float, claimed: float, estimate: float | None = None) -> str:
    """The sentence that says how far outside, and, for a correlation claimed at +-1, how that compares with the
    allowance a known answer gets.

    THE TWO DISTANCES ARE DIFFERENT NUMBERS and only one of them decides anything. `gap` is from the claimed value
    to the nearer interval bound; the known-answer rescue above tests |claimed - r|, the distance to the ESTIMATE,
    which is always the larger of the two. Comparing the gap with KNOWN_ANSWER_TOL could print "inside that" on a
    row the rescue had already declined, which is a sentence contradicting its own verdict. So the comparison is
    made on the estimate, and only where there is one: `estimate` is None at the value site, where no such
    allowance exists at all and the claim may be reading one column rather than two.
    """
    text = (f" The printed numbers are rounded and this refutation is real: the claimed value is "
            f"{gap:.3g} outside the nearer bound at full precision.")
    if estimate is not None and abs(claimed) == 1.0:
        off = abs(claimed - estimate)
        text += (f" A known answer of {claimed:g} is allowed {KNOWN_ANSWER_TOL:.0e} for the agreement of two "
                 f"independent computations at numerical precision, measured against the estimate; "
                 f"|{claimed:g} - r| = {off:.3g}, which is {off / KNOWN_ANSWER_TOL:.3g} times that allowance.")
    return text


def _range_verdict(rng, lo: float, hi: float, where: str, verdict,
                   printed: tuple[str, str] | None = None) -> dict:
    """A `claimed_range = [lo, hi]` (-inf or inf for an open side), for a value claim and a correlation alike, judged by where the
    whole interval lies: inside the range, agrees; entirely outside, refuted; across a bound, unverifiable, because this design cannot
    tell on which side the value is. No width limit is applied: an interval wide enough not to discriminate already straddles a bound
    of any range narrow enough to say something.

    `printed` is the interval's two bounds as the caller's `where` prints them, so that a refutation
    whose margin the rounding swallows says how far outside it really is (see `_hidden_gap`)."""
    # An open side is written as TOML's -inf or inf (TOML has no null); NaN is refused rather than read as "no bound", because every
    # comparison with it is false and the claim would come back unverifiable for a reason nobody wrote.
    if not isinstance(rng, list) or len(rng) != 2 or any(isinstance(x, float) and math.isnan(x) for x in rng):
        return verdict(UNVERIFIABLE, f"claimed_range {rng!r} is not [lo, hi] with -inf or inf for an open side")
    a, b = float(rng[0]), float(rng[1])
    text = f"[{a:g}, {b:g}]"
    if a <= lo and hi <= b:
        return verdict(AGREES, f"{where} lies within the claimed range {text}")
    if hi < a or lo > b:
        # Which end is outside decides which pair of printed numbers ought to show the separation.
        gap, bigger, smaller = ((a - hi, f"{a:g}", (printed or ("", ""))[1]) if hi < a
                                else (lo - b, (printed or ("", ""))[0], f"{b:g}"))
        hidden = _hidden_gap(gap, bigger, smaller) if printed else None
        tail = ("" if hidden is None else
                f" The printed numbers are rounded and this refutation is real: the interval lies "
                f"{hidden:.3g} outside the nearer bound of the range at full precision.")
        return verdict(REFUTED, f"{where} lies OUTSIDE the claimed range {text}{tail}")
    return verdict(UNVERIFIABLE, f"{where} straddles a bound of the claimed range {text}; the data cannot tell which side the value is on")


def _value_verdict(rel: dict, row: dict, v: float, lo: float, hi: float, n_eff: int, unit, verdict) -> dict:
    """A value claim's verdict. A point `claimed` is judged as a correlation's is (inside the interval: agrees, unless the claim sets
    `max_ci_width` and the interval is wider; outside: refuted). A `claimed_range = [lo, hi]` (-inf or inf for an open side) is judged by
    where the whole interval lies: inside the range, agrees; entirely outside, refuted; across a bound, unverifiable, because this design
    cannot tell on which side the value is. No p value and no R squared: they describe a relationship, and a value is not one.
    The exceptions are `sign_test` and `binomial_count`, whose callers have already put an exact binomial p on the row; `setdefault`
    keeps it."""
    row["r_squared"] = None
    row.setdefault("p_value", None)
    row.setdefault("p_method", "not applicable to a value claim")
    count = ""
    if rel["method"] == "sign_test":
        count = (f" ({row['n_success']} of {n_eff} {unit or 'row'}(s) are the event, exact two-sided binomial "
                 f"p = {row['p_value']:.3g} against 0.5)")
    elif rel["method"] == "binomial_count":
        count = (f" ({row['n_success']} of {n_eff} {unit or 'row'}(s) pass, {row['dropped']} dropped as missing; exact two-sided binomial "
                 f"p = {row['p_value']:.3g} against 0.5)")
    elif rel["method"] == "topn_recovery":
        count = (f" ({row['n_success']} of {row['n_actives']} actives are among the {row['n_shortlist']} shortlisted of {row['n_library']} "
                 f"{unit or 'row'}(s), {row['dropped']} dropped as missing; chance share {row['p_null']:.4g}; exact hypergeometric "
                 f"P(at least {row['n_success']}) = {row['p_enrichment']:.3g}, two-sided p = {row['p_value']:.3g})")
    elif rel["method"] in ("pooled_sd", "pooled_se"):
        flag = ("the trigger rule says pooling is justified" if row["pool_justified"] else
                f"the trigger rule says pooling is NOT justified (failed: {', '.join(row['pool_failed'])}); the verdict is unchanged")
        unchecked = f"; not computed: {', '.join(row['pool_unchecked'])}" if row["pool_unchecked"] else ""
        other = ("bootstrap" if row["interval"] == "chi2" else "chi-square")
        olo, ohi = ((row.get("ci_bootstrap_lo"), row.get("ci_bootstrap_hi")) if row["interval"] == "chi2" else (row["ci_chi2_lo"], row["ci_chi2_hi"]))
        othertext = f"[{olo:.4g}, {ohi:.4g}]" if olo is not None else "not available (fewer than 3 groups)"
        count = (f" (pooled over {row['n_groups_used']} {unit or 'group'}(s) on {row['df']} degrees of freedom, {row['n_groups_dropped']} dropped "
                 f"for a single value; the {row['interval']} interval decides ({row['interval_source']}); the {other} interval is {othertext}; "
                 f"{flag}{unchecked})")
    elif rel["method"] == "stratified_dF":
        po, do = row["ci_population_only"], row["ci_delta_only"]
        count = (f" (p of the second well {row['p_second_well']:.4g} over {row['n_first_well'] + row['n_second_well']} rows in the two wells, "
                 f"{row['n_other']} in neither; delta {row['delta']:.4g} over {row['n_units_y']} {row.get('y_resampling_unit') or 'row'}(s); "
                 f"shift {row['shift']:.4g}; the population alone gives [{po[0]:.4g}, {po[1]:.4g}], delta alone [{do[0]:.4g}, {do[1]:.4g}])")
    if rel["method"] in ("pooled_sd", "pooled_se"):
        where = f"the 95% {row['interval']} CI [{lo:.4g}, {hi:.4g}] for {rel['method']} = {v:.4g}{count}, {n_eff} {unit or 'row'}(s)"
    elif rel["method"] == "binomial_count":
        where = (f"the exact 95% Clopper-Pearson CI [{lo:.4g}, {hi:.4g}] for the share = {v:.4g}{count}, {n_eff} independent "
                 f"{unit or 'row'}(s), nothing resampled")
    elif rel["method"] == "topn_recovery":
        where = (f"the 95% Clopper-Pearson CI [{lo:.4g}, {hi:.4g}] for the share of actives recovered = {v:.4g}{count}, {n_eff} "
                 f"actives treated as independent trials (conservative when they compete for places), nothing resampled")
    else:
        where = f"the 95% CI [{lo:.4g}, {hi:.4g}] for {rel['method']} = {v:.4g}{count}, resampling {n_eff} {unit or 'row'}(s)"
    shown = (f"{lo:.4g}", f"{hi:.4g}")          # the bounds as every branch of `where` above prints them
    if rel.get("claimed_range") is not None:
        return _range_verdict(rel["claimed_range"], lo, hi, where, verdict, printed=shown)
    claimed = rel.get("claimed")
    if claimed is None:
        return verdict(UNVERIFIABLE, "a value claim states neither `claimed` nor `claimed_range`")
    if not (lo <= claimed <= hi):
        # A REFUTATION THAT SAYS HOW FAR OUTSIDE, when the rounding in `where` has swallowed it.
        gap, bigger, smaller = ((claimed - hi, f"{claimed:g}", shown[1]) if claimed > hi
                                else (lo - claimed, shown[0], f"{claimed:g}"))
        hidden = _hidden_gap(gap, bigger, smaller)
        tail = "" if hidden is None else _margin_text(hidden, float(claimed))
        return verdict(REFUTED, f"claimed {claimed} lies OUTSIDE {where}{tail}")
    limit = rel.get("max_ci_width")
    if limit is not None and hi - lo > limit:
        return verdict(UNVERIFIABLE, f"claimed {claimed} lies within {where}, but the interval is wider than the claim's max_ci_width {limit}")
    return verdict(AGREES, f"claimed {claimed} lies within {where}")


def check(root: Path | str) -> dict:
    # COERCED, not assumed. The annotation said `Path` and the body did
    # `root / "context.toml"`, so passing the obvious thing — a string — failed
    # with `TypeError: unsupported operand type(s) for /: 'str' and 'str'`,
    # which tells the caller nothing about what they did wrong.
    root = Path(root)
    ctx = tomllib.loads((root / "context.toml").read_text())

    # Every dataset is loaded and verified BEFORE any claim is checked. A
    # manually imported file whose digest has moved stops the run here, rather
    # than producing verdicts about bytes nobody meant to ship.
    sets = {d["id"]: load_dataset(root, d) for d in ctx["datasets"]}
    only = next(iter(sets)) if len(sets) == 1 else None
    # A LEGIBLE REFUSAL, not a KeyError. A variable with no `dataset` used to
    # raise `KeyError: 'dataset'` from inside a comprehension, which tells the
    # person editing a TOML file nothing about which block is wrong.
    declared = {}
    for v in ctx.get("variables", []):
        if "dataset" not in v:
            raise DatasetError(
                f"variable {v.get('name', '?')!r} does not say which dataset it "
                f"belongs to. Add `dataset = \"<id>\"` to its block; with more "
                f"than one dataset the name alone is ambiguous.")
        if v["dataset"] not in sets:
            raise DatasetError(
                f"variable {v['name']!r} names dataset {v['dataset']!r}, which "
                f"is not declared. Known: {', '.join(sorted(sets))}")
        declared[(v["dataset"], v["name"])] = v

    out, joins = [], []
    for rel in ctx.get("relationships", []):
        xd, x = _ref(rel["x"], only)
        yd, y = _ref(rel["y"], only)
        unit = rel.get("resampling_unit") or sets[xd]["spec"].get("resampling_unit")
        level = canonical_level(rel.get("level"))
        if level is None:
            x_lvl = sets.get(xd, {}).get("level")
            y_lvl = sets.get(yd, {}).get("level")
            if x_lvl and y_lvl:
                level = x_lvl if x_lvl == y_lvl else (x_lvl if LEVEL_RANK.get(x_lvl, 0) <= LEVEL_RANK.get(y_lvl, 0) else y_lvl)
            else:
                level = x_lvl or y_lvl

        row: dict = {
            # OPTIONAL, AND THE ONLY WAY TO NAME ONE OF A PAIR. Declaring the
            # same two columns twice at two resampling units is a legitimate
            # and useful thing to do — the row-level interval is what most
            # published numbers use, and the cluster-level one is what the
            # unit of independence actually is. Both belong. But a belief claim
            # names its verdict by `"x ~ y"`, which cannot tell them apart, so
            # without an id neither could be tested. Empty is fine when the
            # pair is unique.
            "id": str(rel.get("id", "") or ""),
            # A CLAIM WITHOUT A FIGURE IS NOT FINISHED. A verdict is a number
            # and a number cannot show the SHAPE that produced it: a wedge, a
            # cluster of ties, one point carrying the whole correlation, a
            # scatter the line does not describe. Every one of those has been
            # found here only by drawing it. So a claim MAY name its own figure,
            # `relate` records whether that declared file is on disk, and
            # `--require-figures` exits non-zero when one is missing.
            #
            # THE NAME IS `declared_figure_exists` AND NOT `figure_exists`
            # (2026-11-18). It tests the path the claim declared, and a claim
            # that declares nothing reads false however many figures exist for
            # it. Under the old name the master ledger reported 21,049 of 25,760
            # claims as having no figure while 20,128 claim PNGs sat on disk,
            # because the shared `check_claims` rule draws
            # figures/claims/<id>.png for every claim and almost nobody also
            # declares one. `relate` cannot report the drawn figure: it runs
            # before `claimcheck.figures` in that rule, so it would read the
            # previous pass's file or nothing. That question is answered from
            # `figures/claims/index.json` and the PNGs beside it.
            "figure": str(rel.get("figure", "") or ""),
            "declared_figure_exists": bool(rel.get("figure")
                                           and (root / str(rel["figure"])).is_file()),
            "x": f"{xd}.{x}", "y": f"{yd}.{y}",
            "x_dataset": xd, "y_dataset": yd,
            "cross_dataset": xd != yd,
            "kind": rel["kind"], "method": rel["method"],
            "claimed": rel.get("claimed"), "claimed_range": rel.get("claimed_range"),
            "evidence": rel.get("evidence", {}),
            "x_unit": declared.get((xd, x), {}).get("unit"),
            "y_unit": declared.get((yd, y), {}).get("unit"),
            "resampling_unit": unit,
            "level": level,
        }

        # `base=row` binds at definition. The closure is called inside its own
        # iteration so it is correct either way today — but a late-binding
        # closure over a loop variable is a trap that costs one word to remove,
        # and the next person to hoist this out of the loop gets no warning.
        # Same defect ruff caught in `emit/rdf.py`; second occurrence.
        def verdict(v: str, why: str, base: dict = row, **extra) -> dict:
            return base | {"verdict": v, "why": why, "status": STATUS[v],
                           **extra}

        # THE DECLARED METHOD IS THE ONE THAT RUNS, or the run stops.
        if rel["method"] not in ESTIMATORS:
            out.append(verdict(
                UNVERIFIABLE,
                f"unknown method {rel['method']!r}. Known: "
                f"{', '.join(sorted(ESTIMATORS))}"))
            continue
        estimate, implements = ESTIMATORS[rel["method"]]
        if rel["kind"] != implements:
            out.append(verdict(
                UNVERIFIABLE,
                f"{rel['method']!r} computes a {implements}, but this claim "
                f"declares kind {rel['kind']!r}"))
            continue
        # `tie_tol` (optional, absolute, spearman and kendall only): see spearman_tie_tol. Absent means byte-identical behaviour.
        tie_tol = rel.get("tie_tol")
        if tie_tol is not None:
            ok = (isinstance(tie_tol, (int, float)) and not isinstance(tie_tol, bool)
                  and math.isfinite(tie_tol) and tie_tol > 0)
            if rel["method"] not in ("spearman", "kendall") or not ok:
                out.append(verdict(
                    UNVERIFIABLE,
                    f"tie_tol = {tie_tol!r} is not accepted: it must be a positive number and applies to methods "
                    f"'spearman' and 'kendall' only (this claim's method is {rel['method']!r})"))
                continue
            estimate = (kendall_tie_tol if rel["method"] == "kendall" else spearman_tie_tol)(float(tie_tol))
            row["tie_tol"] = float(tie_tol)

        # `alpha` (bedroc only, REQUIRED there): the early-recognition weight. No default, because the number means nothing without it
        # and a default would hide the choice. It is recorded on the row. On any other method the key is left alone, as before.
        if rel["method"] in ALPHA_METHODS:
            alpha = rel.get("alpha")
            if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0):
                out.append(verdict(
                    UNVERIFIABLE,
                    (f"{rel['method']} needs `alpha = <positive number>` and none is declared" if alpha is None else
                     f"alpha = {alpha!r} is not accepted: {rel['method']} needs a positive finite number")
                    + ". There is no default: alpha decides what share of the weight the top of the list carries "
                      "(0.798 in the first 8% at alpha = 20), so the claim states it"
                    + (". It is applied identically to both score columns, so one alpha governs the difference"
                       if rel["method"] == "bedroc_difference" else "")))
                continue
            # ONE alpha, bound once, for both columns of a difference: nothing downstream can weight the two differently.
            estimate = (bedroc_difference_alpha if rel["method"] == "bedroc_difference" else bedroc_alpha)(float(alpha))
            row["alpha"] = float(alpha)

        # `baseline` (the PAIRED_DIFFERENCE_METHODS only, REQUIRED there): the column `y` is compared against. Refused on any other method
        # for the reason `tie_tol` is: a key a method does not read is a declaration the checker silently ignored, and the next
        # person reads it as having had an effect.
        second = None
        if rel["method"] in PAIRED_DIFFERENCE_METHODS:
            base_ref = rel.get("baseline")
            if not isinstance(base_ref, str) or not base_ref:
                out.append(verdict(UNVERIFIABLE, f"{rel['method']} needs `baseline = \"<dataset>.<column>\"`, the score column `y` is "
                                                 f"compared against; none is declared. It is the whole claim: without it there is one "
                                                 f"{'BEDROC' if rel['method'] == 'bedroc_difference' else 'AUROC'} and nothing to subtract"))
                continue
            bd, second = _ref(base_ref, only)
            if bd != yd:
                out.append(verdict(UNVERIFIABLE, f"baseline {base_ref!r} is in dataset {bd!r} and `y` is in {yd!r}. The two scores must be "
                                                 f"columns of ONE table, written there by the rule that computed them: a paired difference is "
                                                 f"a statement about identical rows, and `join` lines up two datasets, not three"))
                continue
            if x == y and xd == yd:
                out.append(verdict(UNVERIFIABLE, f"x and y are both {rel['x']!r}: the label used as its own score is an oracle, so the "
                                                 f"difference would be 1 minus the baseline's AUROC and not a comparison of two scores. "
                                                 f"Declare the oracle as its own column and name it in `y`"))
                continue
            row["baseline"] = f"{bd}.{second}"
            row["baseline_unit"] = declared.get((bd, second), {}).get("unit")
        elif rel.get("baseline") is not None:
            out.append(verdict(UNVERIFIABLE, f"`baseline` applies to the paired-difference methods "
                                             f"{', '.join(sorted(PAIRED_DIFFERENCE_METHODS))} only (this claim's method is "
                                             f"{rel['method']!r}), and a key no method reads must not look as though it had an effect"))
            continue

        # POOLED METHODS name the unit within which replicates sit with `group`, and it is the unit the bootstrap resamples: a group is one
        # independent draw, its rows are not. A resampling unit that differs from it is refused, naming both, because resampling the rows of
        # a group (or several groups at once) is not the question the claim asks.
        if rel["method"] in POOLED_METHODS:
            group = rel.get("group")
            if not isinstance(group, str) or not group:
                out.append(verdict(UNVERIFIABLE, f"{rel['method']} needs `group = \"<column>\"`, the column within which the replicates sit "
                                                 f"(for example the ligand); none is declared"))
                continue
            if unit and unit != group:
                out.append(verdict(UNVERIFIABLE, f"{rel['method']} resamples groups, so `resampling_unit` must be the group column: this claim declares "
                                                 f"group = {group!r} and resampling_unit = {unit!r}"))
                continue
            unit = group
            row["resampling_unit"] = group
            row["group"] = group
            if rel["method"] != "pool_check" and rel.get("pool_check"):
                row["pool_check"] = str(rel["pool_check"])        # linked after every claim has a verdict (below)
            if rel["x"] != rel["y"]:
                out.append(verdict(UNVERIFIABLE, f"{rel['method']} is a one-column method: write x = y (x = {rel['x']!r}, y = {rel['y']!r})"))
                continue
            # the optional keys, each refused when it is not one of its accepted values (a typo must not fall back to a default unseen)
            bad_key = None
            if rel["method"] != "pool_check" and rel.get("interval") is not None and rel["interval"] not in INTERVALS:
                bad_key = f"interval = {rel['interval']!r} is not one of {', '.join(INTERVALS)}"
            elif rel["method"] == "pooled_se" and rel.get("replicate_mean") is not None and rel["replicate_mean"] not in REPLICATE_MEANS:
                bad_key = f"replicate_mean = {rel['replicate_mean']!r} is not one of {', '.join(REPLICATE_MEANS)}"
            elif rel["method"] == "pool_check" and rel.get("claimed") not in (None, 1, 1.0):
                bad_key = (f"pool_check claims that pooling is justified, claimed = 1; claimed = {rel['claimed']!r} is not accepted "
                           f"(a claim that pooling is NOT justified is not a pool_check claim)")
            elif rel["method"] == "pool_check" and rel.get("claimed_range") is not None:
                bad_key = "pool_check is deterministic and takes claimed = 1, not claimed_range"
            if bad_key:
                out.append(verdict(UNVERIFIABLE, bad_key))
                continue

        unknown = [d for d in (xd, yd) if d not in sets]
        if unknown:
            out.append(verdict(
                UNVERIFIABLE,
                f"no dataset with id {', '.join(repr(u) for u in unknown)}. "
                f"Known: {', '.join(sorted(sets))}"))
            continue

        if rel["method"] in STRATIFIED_METHODS:
            # TWO DATASETS, NO JOIN: the frames (x, resampled over `resampling_unit`) and the per-unit deltas (y, resampled over
            # `y_resampling_unit`) are different samples, so their rows are never lined up. Every diagnostic goes on `row` before
            # `_value_verdict`, which copies it.
            kT = rel.get("kT")
            yunit = rel.get("y_resampling_unit")
            off = rel.get("offset")
            row |= {"kT": kT, "y_resampling_unit": yunit, "offset_column": off}
            bad_key = None
            if not isinstance(kT, (int, float)) or isinstance(kT, bool) or not math.isfinite(kT) or kT <= 0:
                bad_key = f"stratified_dF needs `kT`, a positive number in the unit of y ({rel['y']}); got {kT!r}"
            elif yunit is not None and (not isinstance(yunit, str) or not yunit):
                bad_key = f"y_resampling_unit = {yunit!r} must name a column of y's dataset"
            elif off is not None and (not isinstance(off, str) or not off):
                bad_key = f"offset = {off!r} must name a column of y's dataset"
            elif xd == yd:
                bad_key = ("stratified_dF reads the well labels and the deltas from two different datasets (frames of the end state, and "
                           "per-unit estimates of the difference between two ladders); x and y name one dataset")
            if bad_key:
                out.append(verdict(UNVERIFIABLE, bad_key))
                continue
            xrows, yrows = sets[xd]["rows"], sets[yd]["rows"]
            missing = ([f"{xd}.{x}"] if not xrows or x not in xrows[0] else []) + ([f"{yd}.{y}"] if not yrows or y not in yrows[0] else [])
            missing += [f"{xd}.{unit} (resampling_unit)"] if unit and xrows and unit not in xrows[0] else []
            missing += [f"{yd}.{yunit} (y_resampling_unit)"] if yunit and yrows and yunit not in yrows[0] else []
            if missing:
                out.append(verdict(UNVERIFIABLE, f"the data has no column {', '.join(missing)}"))
                continue
            undeclared = [c for (d_, c) in ((xd, x), (yd, y)) if declared.get((d_, c), {}).get("unit") is None]
            if undeclared:
                out.append(verdict(UNVERIFIABLE, f"no unit declared for {', '.join(undeclared)}"))
                continue
            info = _stratified_inputs(xrows, x, unit, yrows, y, yunit, off)
            row |= {k: v for k, v in info.items() if k != "refused" and not k.startswith("_")}
            if info["refused"]:
                out.append(verdict(UNVERIFIABLE, info["refused"]))
                continue
            bs = stratified_bootstrap(info["_xu"], info["_yu"], float(kT), info["offset"])
            p0 = info["n_second_well"] / (info["n_first_well"] + info["n_second_well"])
            flat = [v for vs in info["_yu"].values() for v in vs]
            d0 = sum(flat) / len(flat)
            est = info["offset"] + stratified_shift(p0, d0, float(kT))
            row |= {"estimate": round(est, 6), "p_second_well": round(p0, 6), "delta": round(d0, 6),
                    "shift": round(est - info["offset"], 6), "offset": info["offset"]}
            if bs is None:
                out.append(verdict(UNVERIFIABLE, f"fewer than 3 independent units to resample (x: {info['n_units']} {unit or 'row'}(s), "
                                                 f"y: {info['n_units_y']} {yunit or 'row'}(s)), so no interval exists"))
                continue
            lo, hi = bs["ci"]
            row |= {"ci_lo": round(lo, 6), "ci_hi": round(hi, 6), "ci_level": 0.95, "ci_width": round(hi - lo, 6),
                    "n_units": info["n_units"], "n_draws": bs["n_draws"],
                    "ci_population_only": [round(v, 6) for v in bs["ci_population_only"]],
                    "ci_delta_only": [round(v, 6) for v in bs["ci_delta_only"]],
                    "ci_p_second_well": [round(v, 6) for v in bs["ci_p"]], "ci_delta": [round(v, 6) for v in bs["ci_delta"]],
                    "interval_source": f"percentile bootstrap, each draw resampling {info['n_units']} {unit or 'row'}(s) of {xd} and "
                                       f"{info['n_units_y']} {yunit or 'row'}(s) of {yd}"}
            out.append(_value_verdict(rel, row, est, lo, hi, info["n_units"], unit, verdict))
            continue

        # ONE DATASET, or two joined on a stated key. The join is refused
        # outright on duplicate keys and its coverage travels with the verdict —
        # a correlation over a 40%-covered join is a number about a different
        # population from the one either file describes.
        if xd == yd:
            rows = sets[xd]["rows"]
        else:
            key = rel.get("join")
            if not key:
                out.append(verdict(
                    UNVERIFIABLE,
                    f"{xd} and {yd} are different datasets and no `join` key is "
                    f"declared, so their rows cannot be lined up"))
                continue
            pairs_rows, jr = join(sets[xd], sets[yd], key)
            joins.append(jr)
            row["join"] = jr
            if jr["refused"]:
                out.append(verdict(UNVERIFIABLE, jr["refused"]))
                continue
            rows = [{**r_, **l_} for l_, r_ in pairs_rows]

        # The baseline column is checked here with the other two, not beside it: a third column present in the declaration and
        # absent from the file is the same failure as a missing `y`, and it has to report itself as one.
        absent = [c for c in (x, y) + ((second,) if second else ()) if not rows or c not in rows[0]]
        if absent:
            out.append(verdict(UNVERIFIABLE,
                               f"the data has no column {', '.join(absent)}"))
            continue
        undeclared = [c for (d_, c) in ((xd, x), (yd, y)) + (((yd, second),) if second else ())
                      if declared.get((d_, c), {}).get("unit") is None]
        if undeclared:
            out.append(verdict(
                UNVERIFIABLE,
                f"no unit declared for {', '.join(undeclared)} — a relationship "
                f"between undeclared quantities cannot be compared with anyone "
                f"else's"))
            continue
        if unit and rows and unit not in rows[0]:
            if rel["method"] in POOLED_METHODS:
                out.append(verdict(
                    UNVERIFIABLE,
                    f"group column {unit!r} is not a column of the data: {rel['method']} pools the replicates within each {unit}, "
                    f"so without it there are no groups"))
                continue
            out.append(verdict(
                UNVERIFIABLE,
                f"resampling unit {unit!r} is not a column, so the interval "
                f"would assume an independence the data does not have"))
            continue

        if rel["method"] in POOLED_METHODS:
            # ONE (df, SS) PAIR PER GROUP BEFORE THE BOOTSTRAP (see the block above `ESTIMATORS`). Every diagnostic goes on `row` here:
            # `verdict` copies the row, so anything added after `_value_verdict` returns would be lost.
            proto = rel.get("protocol")
            if proto is not None and (not isinstance(proto, str) or proto not in rows[0]):
                out.append(verdict(UNVERIFIABLE, f"protocol = {proto!r} is not a column of the data, so whether the groups share one protocol cannot be checked"))
                continue
            pairs, pg = _pooled_groups(rows, y, group, proto)
            row |= {k: v for k, v in pg.items() if k != "refused" and not k.startswith("_")}
            if pg["refused"]:
                out.append(verdict(UNVERIFIABLE, pg["refused"]))
                continue
            clusters = None                      # every pair IS a group, so the bootstrap draws pairs
            if pg["n_groups_used"] < 2:
                out.append(verdict(UNVERIFIABLE, f"fewer than 2 {group} groups have 2 or more values ({pg['n_groups_used']} do, {pg['n_groups_dropped']} "
                                                 f"have a single value and are dropped), so there is no within-group spread to pool"))
                continue
            if rel["method"] == "pool_check":
                # DETERMINISTIC: the criteria of the trigger rule, no resampling and no interval. The estimate is 1 or 0 (see `pool_check_outcome`).
                sds, dfs = pg["_sds"], pg["_dfs"]
                bf = brown_forsythe(pg["_values"])
                tr = pooling_trigger(sds, dfs, bf, pg["_protocols"])
                row |= {"brown_forsythe": bf, "n_units": pg["n_groups_used"], "r_squared": None, "p_value": None,
                        "p_method": "not applicable: pool_check is deterministic", "claimed": 1.0, **tr}
                lo_sd, hi_sd = min(sds.values()), max(sds.values())
                row["sd_ratio_max_min"] = None if lo_sd == 0.0 else round(hi_sd / lo_sd, 6)
                outcome, est_pc, why_pc = pool_check_outcome(tr["pool_criteria"])
                if est_pc is not None:
                    row["estimate"] = est_pc
                out.append(verdict(outcome, f"pool_check over {pg['n_groups_used']} {group}(s) on {pg['df']} degrees of freedom: {why_pc}"))
                continue
            est0 = estimate([p[0] for p in pairs], [p[1] for p in pairs])
            if est0 is None or not math.isfinite(est0):
                out.append(verdict(UNVERIFIABLE, "the pooled estimate is undefined on these groups"))
                continue
            if est0 == 0.0:
                row["estimate"] = 0.0
                out.append(verdict(UNVERIFIABLE, f"every value is identical within every one of the {pg['n_groups_used']} groups, so the pooled "
                                                 f"{'sd' if rel['method'] == 'pooled_sd' else 'se'} is exactly 0; that is a finding about the "
                                                 f"values (or about a rounded column), but a zero-variance sample gives no interval to judge a claim by"))
                continue
            how = rel.get("replicate_mean") or "harmonic"
            if rel["method"] == "pooled_se":
                estimate = _pooled_se_arithmetic if how == "arithmetic" else _pooled_se
                est0 = estimate([p[0] for p in pairs], [p[1] for p in pairs])
            row["estimate"] = round(est0, 6)
            row |= _pooled_diagnostics(pg, rel["method"], est0, how)
            # BOTH INTERVALS ARE WRITTEN; ONE DECIDES. Default: the exact chi-square interval below POOL_BOOTSTRAP_MIN_GROUPS groups, the
            # bootstrap at or above it (docs/pooled_sd.md); a declared `interval` overrides it and the row says which happened.
            ci_b = bootstrap_ci(pairs, None, estimate=estimate)
            row["ci_bootstrap_lo"], row["ci_bootstrap_hi"] = (round(ci_b[0], 6), round(ci_b[1], 6)) if ci_b else (None, None)
            row["ci_bootstrap_note"] = ("percentile bootstrap over the groups, 2000 resamples; too narrow with few groups (Hesterberg 2015); "
                                        "measured 0.85 coverage at 8 groups of 3 runs" if ci_b else "needs at least 3 groups")
            interval_declared = rel.get("interval")                # NOT `declared`: that name is the variables table of this function
            use = interval_declared or ("bootstrap" if pg["n_groups_used"] >= POOL_BOOTSTRAP_MIN_GROUPS else "chi2")
            row["interval"] = use
            row["interval_source"] = ("declared in the claim" if interval_declared else
                                      f"default: {pg['n_groups_used']} groups, {'at or above' if use == 'bootstrap' else 'below'} {POOL_BOOTSTRAP_MIN_GROUPS}")
            if use == "bootstrap" and ci_b is None:
                out.append(verdict(UNVERIFIABLE, f"fewer than 3 independent units to resample over {group}, so no bootstrap interval exists "
                                                 f"(the chi-square interval is [{row['ci_chi2_lo']:.4g}, {row['ci_chi2_hi']:.4g}]; declare interval = \"chi2\" to use it)"))
                continue
            lo, hi = (row["ci_chi2_lo"], row["ci_chi2_hi"]) if use == "chi2" else (ci_b[0], ci_b[1])
            n_eff = pg["n_groups_used"]
            row |= {"ci_lo": round(lo, 6), "ci_hi": round(hi, 6), "ci_level": 0.95, "n_units": n_eff, "ci_width": round(hi - lo, 6)}
            out.append(_value_verdict(rel, row, est0, lo, hi, n_eff, unit, verdict))
            continue
        elif rel["method"] == "binomial_count":
            # EXACT, NOTHING RESAMPLED (2026-11-09). k of n from one 0/1 row per unit, the Clopper-Pearson interval, the exact binomial p
            # against 0.5. Every field goes on `row` before `_value_verdict`, which copies it.
            if rel["x"] != rel["y"]:
                out.append(verdict(UNVERIFIABLE, f"binomial_count is a one-column method: write x = y (x = {rel['x']!r}, y = {rel['y']!r})"))
                continue
            cnt = _count_units(rows, y, unit)
            row |= {k_: v_ for k_, v_ in cnt.items() if k_ != "refused"}
            if cnt["refused"]:
                out.append(verdict(UNVERIFIABLE, cnt["refused"]))
                continue
            k_s, n_u = cnt["n_success"], cnt["n_units"]
            if n_u == 0:
                out.append(verdict(UNVERIFIABLE, f"no {unit or 'row'} has a usable value ({cnt['dropped']} dropped as missing or not finite), "
                                                 f"so there is no count to bound"))
                continue
            lo, hi = clopper_pearson(k_s, n_u)
            est0 = k_s / n_u
            row |= {"estimate": round(est0, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6), "ci_level": 0.95, "n_units": n_u,
                    "ci_width": round(hi - lo, 6), "interval": "clopper_pearson",
                    "interval_source": "exact binomial (Clopper and Pearson 1934); assumes the units are independent trials",
                    "p_value": sign_test_p(k_s, n_u), "p_null": 0.5,
                    "p_method": f"exact two-sided binomial against 0.5 over {n_u} {unit or 'row'}(s)"}
            out.append(_value_verdict(rel, row, est0, lo, hi, n_u, unit, verdict))
            continue
        elif rel["method"] == "topn_recovery":
            # EXACT, NOTHING RESAMPLED (2026-11-12). The 2 x 2 table is counted from the library table: K actives, N shortlisted, M
            # compounds, k both. Estimate k / K, Clopper-Pearson interval on k of K, exact hypergeometric p. Every field goes on `row`
            # before `_value_verdict`, which copies it.
            if rel["x"] == rel["y"]:
                out.append(verdict(UNVERIFIABLE, f"topn_recovery is a two-column method: x is the label and y the shortlist indicator "
                                                 f"(x = y = {rel['x']!r})"))
                continue
            top = _topn_units(rows, x, y, unit)
            row |= {k_: v_ for k_, v_ in top.items() if k_ != "refused"}
            if top["refused"]:
                out.append(verdict(UNVERIFIABLE, top["refused"]))
                continue
            k_s, n_act, n_short, n_lib = top["n_success"], top["n_actives"], top["n_shortlist"], top["n_library"]
            if n_act == 0:
                out.append(verdict(UNVERIFIABLE, f"no {unit or 'row'} is labelled active among {n_lib} usable ({top['dropped']} dropped as "
                                                 f"missing), so there is nothing to recover"))
                continue
            if n_short == 0 or n_short == n_lib or n_act == n_lib:
                out.append(verdict(UNVERIFIABLE, f"the table cannot show recovery: {n_short} of {n_lib} {unit or 'row'}(s) are shortlisted and "
                                                 f"{n_act} are active. A shortlist that holds nothing or everything, or a library with no "
                                                 f"inactive, fixes the count whatever the score does"))
                continue
            tails = hypergeom_tails(k_s, n_act, n_short, n_lib)
            assert tails is not None                                   # k is counted from the same table, so it is in the support
            p_lower, p_upper = tails
            cp_interval = clopper_pearson(k_s, n_act)
            assert cp_interval is not None                             # n_act >= 1 and 0 <= k <= n_act here
            lo, hi = cp_interval
            est0 = k_s / n_act
            p_null = n_short / n_lib
            row |= {"estimate": round(est0, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6), "ci_level": 0.95, "n_units": n_act,
                    "ci_width": round(hi - lo, 6), "interval": "clopper_pearson",
                    "interval_source": "exact binomial (Clopper and Pearson 1934) on k of the actives, treating them as independent trials. "
                                       "They compete for the shortlist, which makes the count less variable than a binomial, so the "
                                       "interval is conservative: measured coverage 0.959 to 1.000 of the expected share recovered, over 25 designs "
                                       "(docs/screening_metrics_study.md)",
                    "p_null": round(p_null, 6), "enrichment_factor": round(est0 / p_null, 6), "max_share": round(min(1.0, n_short / n_act), 6),
                    "p_enrichment": p_upper, "p_depletion": p_lower, "p_value": min(1.0, 2 * min(p_lower, p_upper)),
                    "p_method": f"exact hypergeometric: {n_act} actives and {n_short} shortlist places among {n_lib} {unit or 'row'}(s); "
                                f"p_enrichment is P(X >= k), Fisher's exact test one-sided; p_value is twice the smaller tail, capped at 1"}
            out.append(_value_verdict(rel, row, est0, lo, hi, n_act, unit, verdict))
            continue
        elif rel["method"] == "sign_test":
            # ONE VALUE PER UNIT BEFORE THE BOOTSTRAP, because the estimator sees only the resampled values and the
            # collapse to a majority sign needs the unit labels. The exact p goes on the row here: `verdict` copies the
            # row, so anything added after `_value_verdict` returns would be lost.
            event = rel.get("event")
            if event is not None and event not in SIGN_EVENTS:
                out.append(verdict(UNVERIFIABLE, f"event = {event!r} is not one of {', '.join(SIGN_EVENTS)}"))
                continue
            pairs, clusters, sign = _sign_units(rows, y, unit, event)
            row |= {k: v for k, v in sign.items() if k != "refused"}
            if sign["refused"]:
                out.append(verdict(UNVERIFIABLE, sign["refused"]))
                continue
            if sign["n_units"] == 0:
                out.append(verdict(
                    UNVERIFIABLE,
                    f"no {unit or 'row'} is left to count: of {sign['n_rows']} usable row(s), {sign['n_ties_dropped']} "
                    f"are ties (exact zeros) and {sign['n_unit_ties_dropped']} {unit or 'row'}(s) split evenly, so "
                    f"n_units is 0 and the sign test has nothing to test"))
                continue
            row["p_value"] = sign_test_p(sign["n_success"], sign["n_units"])
            row["p_method"] = f"exact two-sided binomial against 0.5 over {sign['n_units']} {unit or 'row'}(s)"
        else:
            # `second` is None for every method but the paired differences, where it packs the baseline column into the pair
            # (`_paired`). One code path on purpose: the empty-unit refusal below and the drop counting are the same facts
            # about a three-column claim as about a two-column one, and a second copy of them is a second place to fix.
            pairs, clusters, dropped = _paired(rows, x, y, unit, second=second)
            # A COLUMN THAT EXISTS AND IS EMPTY is not the same as a column with
            # few clusters, and the bootstrap cannot tell them apart -- both arrive
            # as "fewer than 3 independent units". The difference matters: one says
            # the data is small, the other says the declared unit is not in the file
            # at all. Measured on al_mace's `darby_predictions_rowan32_k4.csv`,
            # where `butina_cluster` is a header over 32 empty rows and is named as
            # that file's resampling unit in two places downstream.
            if unit and clusters and not any(c.strip() for c in clusters):
                out.append(verdict(UNVERIFIABLE, _empty_unit_message(unit, len(clusters))))
                continue
            row |= {"n_rows": len(pairs), "dropped": dropped}
            if rel["method"] == "mean_diff":
                # THE GROUP COLUMN IS 0 OR 1, nothing else: a 2 or a 0.5 would silently fall in neither group and shrink both.
                other = sum(1 for p in pairs if p[0] not in (0.0, 1.0))
                if other:
                    out.append(verdict(UNVERIFIABLE, f"mean_diff needs x = 0 or 1 on every row (the group); {other} row(s) of {x} hold another value"))
                    continue
                row |= {"n_group_1": sum(1 for p in pairs if p[0] == 1.0), "n_group_0": sum(1 for p in pairs if p[0] == 0.0)}
        r = estimate([p[0] for p in pairs], [p[1] for p in pairs])
        if r is None:
            out.append(verdict(UNVERIFIABLE,
                               f"r is undefined on {len(pairs)} usable pair(s)"
                               + (f" (either every row is labelled the same way, or one of {y} and {second} is unreadable "
                                  f"on every row that the other is readable on)" if second else "")))
            continue
        if second:
            # BOTH PARTS BESIDE THE DIFFERENCE. A difference of -0.08 is a different finding when it is 0.52 against 0.60
            # than when it is 0.90 against 0.98, and the row is the only place a reader gets to see which.
            labels_ = [p[0] for p in pairs]
            n_pos_ = sum(1 for v in labels_ if v >= 0.5)
            row["n_actives"] = n_pos_
            if rel["method"] == "bedroc_difference":
                # THE TWO BEDROCS, AND THE NUMBER THEY ARE EACH READ AGAINST. `random_expectation` is a property of the
                # cohort and alpha, not of a column, so the two columns share it exactly on identical rows and it cancels
                # in the difference; it is on the row because the two parts beside the difference are meaningless without
                # it (0.10 against 0.06 is a screener barely above chance, 0.10 against 0.01 is one well above it).
                row |= {"bedroc_y": bedroc(labels_, [p[1][0] for p in pairs], row["alpha"]),
                        "bedroc_baseline": bedroc(labels_, [p[1][1] for p in pairs], row["alpha"]),
                        "active_fraction": round(n_pos_ / len(pairs), 6),
                        "alpha_times_active_fraction": round(row["alpha"] * n_pos_ / len(pairs), 6),
                        "weight_share_top_8pct": round((1 - math.exp(-row["alpha"] * 0.08)) / (1 - math.exp(-row["alpha"])), 6)}
                exp_ = bedroc_random_expectation(n_pos_, len(pairs), row["alpha"])
                row |= {"random_expectation": None if exp_ is None else round(exp_, 6),
                        "random_expectation_note": "the BEDROC a random ranking scores at this active fraction and alpha; SHARED BY BOTH "
                                                   "COLUMNS, because it depends only on the rows and alpha, so it cancels in the difference "
                                                   "and the null of the difference is 0. Each of bedroc_y and bedroc_baseline is read "
                                                   "against it, never against 0"}
            else:
                row |= {"auc_y": roc_auc(labels_, [p[1][0] for p in pairs]),
                        "auc_baseline": roc_auc(labels_, [p[1][1] for p in pairs])}
        row["estimate"] = round(r, 6)
        ci = bootstrap_ci(pairs, clusters, estimate=estimate)
        if ci is None:
            out.append(verdict(
                UNVERIFIABLE,
                f"fewer than 3 independent units to resample"
                f"{f' over {unit}' if unit else ''}, so no interval exists"))
            continue
        lo, hi, n_eff = ci
        width = hi - lo
        row |= {"ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                "ci_level": 0.95, "n_units": n_eff, "ci_width": round(width, 6)}
        if implements == "value":
            out.append(_value_verdict(rel, row, r, lo, hi, n_eff, unit, verdict))
            continue
        # r^2 is the share of variance the relationship accounts for, and it is
        # reported because a correlation of 0.3 sounds like a third of the
        # answer and is nine percent of it. For a rank method this is the
        # variance of the RANKS, which is a different quantity from the
        # variance of the values, so the key says which was computed.
        if rel["method"] == "roc_auc":
            # AUC is a probability of correct ordering, not a correlation: its square is not a share
            # of variance or of anything else, so it is left out rather than printed to be misread.
            row["r_squared"] = None
            row["r_squared_note"] = "not reported: AUC is a probability of correct ordering, not a correlation"
        elif rel["method"] == "roc_auc_difference":
            # Neither AUC has an R squared, so their difference has none either, and squaring a difference of two
            # probabilities would produce a number with no reading at all.
            row["r_squared"] = None
            row["r_squared_note"] = ("not reported: this is a difference of two probabilities of correct ordering, and its "
                                     "square is not a share of anything")
            row["interval_source"] = (f"paired percentile bootstrap over {unit or 'row'}(s): each resample draws whole "
                                      f"{unit or 'row'}(s) and recomputes BOTH AUROCs on the drawn rows, so the correlation "
                                      f"between the two columns stays in and the cohort variance they share cancels")
        elif rel["method"] == "bedroc_difference":
            # Neither BEDROC has an R squared, so their difference has none, and a difference of two weighted
            # early-recognition scores squared is not a share of anything.
            row["r_squared"] = None
            row["r_squared_note"] = ("not reported: this is a difference of two weighted early-recognition scores, and its square is "
                                     "not a share of anything")
            row["interval_source"] = (f"paired percentile bootstrap over {unit or 'row'}(s): each resample draws whole "
                                      f"{unit or 'row'}(s) and recomputes BOTH BEDROCs at alpha = {row['alpha']:g} on the drawn rows, "
                                      f"so the correlation between the two columns stays in and the cohort variance they share, "
                                      f"S_min, S_max and the random expectation among it, cancels")
        elif rel["method"] == "bedroc":
            # The comparison point travels with the estimate: a random ranking does not score 0.
            n_pos = sum(1 for p in pairs if p[0] >= 0.5)
            centre = bedroc_random_expectation(n_pos, len(pairs), row["alpha"])
            row["r_squared"] = None
            row["r_squared_note"] = "not reported: BEDROC is a weighted early-recognition score, not a correlation"
            row |= {"n_actives": n_pos, "active_fraction": round(n_pos / len(pairs), 6),
                    "alpha_times_active_fraction": round(row["alpha"] * n_pos / len(pairs), 6),
                    "random_expectation": None if centre is None else round(centre, 6),
                    "weight_share_top_8pct": round((1 - math.exp(-row["alpha"] * 0.08)) / (1 - math.exp(-row["alpha"])), 6)}
        else:
            row["r_squared"] = round(r * r, 6)
            row["r_squared_is_on"] = {"spearman": "ranks", "kendall": "tau-b"}.get(rel["method"], "values")
        if rel["method"] == "kendall":
            # tau-b squared is not the variance of anything, so say so on the row rather than leave "R squared" to be read as one.
            row["r_squared_note"] = "tau-b squared, a squared rank correlation; not a share of variance"
        if rel["method"] == "bedroc":
            null_centre = centre if centre is not None else 0.0        # the unrounded expectation, not the 6-place value on the row
        else:
            null_centre = 0.5 if rel["method"] == "roc_auc" else 0.0
        # THE NULL OF A PAIRED DIFFERENCE IS NOT A LABEL PERMUTATION (see `paired_swap_p`): permuting the label against both
        # score columns at once asks whether EITHER reads it, and would report a decisive p for two excellent scorers that are
        # identical. The two columns are swapped within a whole unit instead.
        pp = (paired_swap_p(pairs, clusters, estimate=estimate) if rel["method"] in PAIRED_DIFFERENCE_METHODS
              else permutation_p(pairs, clusters, estimate=estimate, centre=null_centre))
        if pp is not None:
            p_val, p_floor = pp
            row |= {"p_value": round(p_val, 6), "p_floor": round(p_floor, 6),
                    "p_at_floor": bool(p_val <= p_floor + 1e-12),
                    "p_method": f"two-sided permutation of {n_eff} "
                                f"{unit or 'row'}(s), {N_BOOT} draws"
                                if unit else
                                f"two-sided permutation of {n_eff} rows, "
                                f"{N_BOOT} draws (NO resampling unit declared, "
                                f"so this p assumes the rows are independent)"}
            if rel["method"] == "roc_auc_difference":
                row["p_method"] = (f"two-sided paired randomisation over {n_eff} {unit or 'row'}(s), {N_BOOT} draws: the two "
                                   f"score columns are swapped within a whole {unit or 'row'} on a fair coin, both on mid-ranks "
                                   f"over the table (AUROC is invariant under that map). THE NULL IS THAT THE TWO COLUMNS ARE "
                                   f"INTERCHANGEABLE, not that either reads the label. Two columns rarely share a tie structure, "
                                   f"so the exchangeability is approximate and the verdict does not depend on this p")
            if rel["method"] == "bedroc_difference":
                row["p_method"] = (f"two-sided paired randomisation over {n_eff} {unit or 'row'}(s), {N_BOOT} draws: the two score "
                                   f"columns are swapped within a whole {unit or 'row'} on a fair coin, both on mid-ranks over the "
                                   f"table (BEDROC reads only the order and the ties, so it is invariant under that map). THE NULL IS "
                                   f"THAT THE TWO COLUMNS ARE INTERCHANGEABLE, not that either reads the label, and it is CENTRED ON 0 "
                                   f"although neither BEDROC is centred on a constant: the random expectation "
                                   f"{row.get('random_expectation')} is a property of these rows and alpha = {row['alpha']:g}, so both "
                                   f"columns carry it and it cancels in every draw of the difference. Two columns rarely share a tie "
                                   f"structure, and at alpha = {row['alpha']:g} the mismatch falls where "
                                   f"{row['weight_share_top_8pct']:g} of the weight sits (the first 8% of the list), so the "
                                   f"exchangeability is approximate and the verdict does not depend on this p")
            if rel["method"] == "bedroc":
                # The null is skewed: its floor is only `random_expectation` below its centre, so a score worse than random cannot
                # be far from the centre and this p detects enrichment only. An inverted score is a question for roc_auc.
                row["p_method"] += (f"; distance from the random-ranking expectation {null_centre:.4f}. The floor of BEDROC is only "
                                    f"that far below it, so where the expectation is small this p detects enrichment and has "
                                    f"almost no power against a score worse than random")
        else:
            row["p_value"] = None
            row["p_method"] = "permutation undefined on these pairs"
        where = (f"the 95% CI [{lo:.3f}, {hi:.3f}] for r = {r:.3f}, "
                 f"resampling {n_eff} {unit or 'row'}(s)")
        shown = (f"{lo:.3f}", f"{hi:.3f}")      # the bounds as `where` prints them
        if rel.get("claimed_range") is not None:
            out.append(_range_verdict(rel["claimed_range"], lo, hi, where, verdict, printed=shown))
            continue
        if rel.get("claimed") is None:
            out.append(verdict(UNVERIFIABLE, "a correlation claim states neither `claimed` nor `claimed_range`"))
            continue
        inside = lo <= rel["claimed"] <= hi
        # A known answer of +-1 from two independent computations: see
        # KNOWN_ANSWER_TOL. Only a claim of exactly +-1 gets it.
        known = (not inside and abs(rel["claimed"]) == 1.0
                 and abs(rel["claimed"] - r) <= KNOWN_ANSWER_TOL)
        inside = inside or known
        if known:
            where += (f"; |r - {rel['claimed']}| = {abs(rel['claimed'] - r):.1e}, within the "
                      f"{KNOWN_ANSWER_TOL:.0e} allowed a known answer at numerical precision")
        limit = rel.get("max_ci_width", MAX_INFORMATIVE_WIDTH)
        if not inside:
            # A refutation survives a wide interval: lying outside an interval
            # this generous is a stronger result, not a weaker one.
            # IT ALSO SAYS HOW FAR OUTSIDE when the three printed decimals have
            # swallowed the margin, which is what made six real refutations read
            # as "claimed 1.0 lies OUTSIDE the 95% CI [1.000, 1.000]".
            claimed = float(rel["claimed"])
            gap, bigger, smaller = ((claimed - hi, f"{claimed:g}", shown[1]) if claimed > hi
                                    else (lo - claimed, shown[0], f"{claimed:g}"))
            hidden = _hidden_gap(gap, bigger, smaller)
            tail = "" if hidden is None else _margin_text(hidden, claimed, estimate=r)
            out.append(verdict(REFUTED, f"claimed {rel['claimed']} lies "
                                        f"OUTSIDE {where}{tail}"))
        elif width > limit:
            out.append(verdict(
                UNVERIFIABLE,
                f"claimed {rel['claimed']} lies within {where} — but that "
                f"interval is {width:.3f} wide, over the {limit} at which it "
                f"stops discriminating. The data cannot tell, which is not the "
                f"same as the claim holding"))
        else:
            out.append(verdict(AGREES,
                               f"claimed {rel['claimed']} lies within {where}"))

    # A POOLED CLAIM MAY NAME ITS CHECK (`pool_check = "<claim id>"`): the verdict of that pool_check claim is written beside the pooled estimate's
    # own, in fields and in the `why` text. The pooled verdict itself stays decided by its interval: the check is reported, never enforced.
    by_id = {r_["id"]: r_ for r_ in out if r_.get("id")}
    for r_ in out:
        pid = r_.get("pool_check")
        if not pid:
            continue
        target = by_id.get(pid)
        if target is None or target.get("method") != "pool_check":
            r_["pool_check_verdict"] = None
            note = (f" Pooling check {pid!r}: no pool_check claim with that id in this context, so no check is shown." if target is None else
                    f" Pooling check {pid!r}: that claim's method is {target.get('method')!r}, not pool_check, so no check is shown.")
        else:
            same = (target["x"], target.get("group"), target.get("resampling_unit")) == (r_["x"], r_.get("group"), r_.get("resampling_unit"))
            r_["pool_check_verdict"] = target["verdict"]
            r_["pool_check_same_data"] = same
            note = (f" Pooling check {pid!r}: {target['verdict'].upper()} ({target['why']})"
                    + ("" if same else " [it reads a different column or group from this claim]")
                    + ". This is reported beside the estimate and does not change its verdict.")
        r_["why"] = r_["why"] + note

    # `res`, not `r` — `r` is the correlation coefficient forty lines up, and
    # binding one name to a float and then to a result row in one function is
    # how a reader (and a type checker) loses track of which is which.
    verdicts: dict[str, int] = {}
    for res in out:
        verdicts[res["verdict"]] = verdicts.get(res["verdict"], 0) + 1

    levels: dict[str, dict] = {}
    for lvl in LEVELS:
        sub = [res for res in out if res.get("level") == lvl]
        if sub:
            levels[lvl] = {
                "total": len(sub),
                "agrees": sum(1 for res in sub if res["verdict"] == AGREES),
                "refuted": sum(1 for res in sub if res["verdict"] == REFUTED),
                "unverifiable": sum(1 for res in sub if res["verdict"] == UNVERIFIABLE),
            }
    def _scrutinise(rows: list[dict]) -> list[dict]:
        from claimcheck.scrutiny import scrutinise
        return scrutinise(rows)

    unspecified = [res for res in out if not res.get("level")]
    if unspecified:
        levels["unspecified"] = {
            "total": len(unspecified),
            "agrees": sum(1 for res in unspecified if res["verdict"] == AGREES),
            "refuted": sum(1 for res in unspecified if res["verdict"] == REFUTED),
            "unverifiable": sum(1 for res in unspecified if res["verdict"] == UNVERIFIABLE),
        }

    return {"project": ctx["project"]["name"],
            # WHICH CHECKER AND WHICH DECLARATION produced these verdicts (first 16 hex of sha256).
            "relate_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "context_sha256": hashlib.sha256((root / "context.toml").read_bytes()).hexdigest()[:16],
            "datasets": {d: {"origin": s_["origin"],
                             "level": s_.get("level"),
                             "rows": len(s_["rows"]),
                             "digests": s_["digests"]}
                         for d, s_ in sets.items()},
            "joins": joins,
            "rows_read": sum(len(s_["rows"]) for s_ in sets.values()),
            # THE CLAIMS WHOSE DECLARED FIGURE IS NOT ON DISK, which is not the same
            # question as which claims have no figure. Named `claims_without_a_figure`
            # until 2026-11-18, when the master ledger reported 21,049 of 25,760 claims
            # as unillustrated while 20,128 claim PNGs were on disk: almost no claim
            # declares a figure, because `check_claims` draws one for each of them
            # without being asked. `relate` cannot answer the drawn question honestly,
            # since it runs BEFORE `claimcheck.figures` in that rule and would read the
            # previous pass's file or nothing at all. `figures/claims/index.json` and
            # the PNGs beside it are the record of what was drawn.
            "claims_without_a_declared_figure": [
                (r["id"] or f"{r['x']} ~ {r['y']}") for r in out
                if not r.get("declared_figure_exists")],
            "relationships": out, "verdicts": verdicts, "levels": levels,
            # SHAPES THAT MAKE A VERDICT UNTRUSTWORTHY, in their own key because they
            # decide nothing: a claim that could not have failed, a control that came
            # back refuted. See `claimcheck.scrutiny`, which also sweeps the prose for
            # numbers contradicting these rows (that part needs the files, so it is on
            # the command line and not here). Imported inside the function because
            # `scrutiny` reads this module's verdict vocabulary and a module-level
            # import either way round would be a cycle.
            "warnings": _scrutinise(out),
            "any_refuted": any(res["verdict"] == REFUTED for res in out),
            # UNVERIFIABLE IS NOT A PASS, so it is reported at the top level
            # rather than left to be counted out of the rows.
            "any_unverifiable": any(res["verdict"] == UNVERIFIABLE
                                    for res in out)}


def main(argv: list[str] | None = None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    require_figures = "--require-figures" in a
    a = [x for x in a if x != "--require-figures"]
    root = Path(a[0]) if a else Path.cwd()
    out = Path(a[1]) if len(a) > 1 else root / "results/claims.json"
    d = check(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2) + "\n")
    mark = {AGREES: "HOLDS  ", REFUTED: "REFUTED", UNVERIFIABLE: "?      "}
    for r in d["relationships"]:
        lvl_str = f" [{r['level']}]" if r.get("level") else ""
        print(f"  {mark[r['verdict']]}{lvl_str} {r['x']} ~ {r['y']}   claimed "
              f"{r['claimed']}  ->  {r['why']}")
        print(f"          {r['evidence'].get('source', 'no source stated')} "
              f"[{r['evidence'].get('extracted_by', 'unknown')}]"
              f"  -> {r['status']}")
    for r in d["relationships"]:
        bits = []
        if r.get("r_squared") is not None:
            bits.append(f"r2({r.get('r_squared_is_on', '?')}) = {r['r_squared']:.4f}")
        if r.get("p_value") is not None:
            bits.append(f"p = {r['p_value']:.4g}"
                        + (" AT THE FLOOR" if r.get("p_at_floor") else ""))
        if bits:
            print(f"          {r['id'] or r['x'] + ' ~ ' + r['y']}: "
                  + ";  ".join(bits) + f"  [{r.get('p_method', '')}]")
    missing = d.get("claims_without_a_declared_figure", [])
    if missing:
        print(f"  {len(missing)} of {len(d['relationships'])} claim(s) DECLARE no "
              f"figure on disk: " + ", ".join(missing[:8])
              + ("..." if len(missing) > 8 else ""))
        print("  This counts declared figures only. The shared `check_claims` rule "
              "draws figures/claims/<id>.png for every claim after this file is "
              "written, and figures/claims/index.json is the record of those.")
    for w in d.get("warnings", []):
        # LOUD AND ABOVE THE SUMMARY, because a warning printed after the counts is
        # a warning read after the conclusion. It changes no verdict and no exit status.
        print(f"  WARNING {w['code']}: {w['claim']}")
        print(f"      {w['why']}")
    if d.get("warnings"):
        print(f"  {len(d['warnings'])} warning(s) on the shape of these claims, none of which changes a "
              f"verdict. `python -m claimcheck.scrutiny {root}` also reads the prose beside them.")
    print(f"  {d['verdicts']} over {d['rows_read']} row(s) -> {out}")
    print(f"  checker relate.py sha256 {d['relate_sha256']}, context.toml sha256 {d['context_sha256']}")
    if any(r.get("level") for r in d["relationships"]):
        print("  Verdicts by verification level:")
        for lvl, counts in d.get("levels", {}).items():
            print(f"    {lvl:12s}: {counts['agrees']} agrees, {counts['refuted']} refuted, {counts['unverifiable']} unverifiable (total {counts['total']})")
    if require_figures and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
