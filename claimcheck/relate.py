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


ESTIMATORS: dict[str, tuple] = {
    "pearson": (pearson, "correlation"),
    "spearman": (spearman, "correlation"),
    # KENDALL TAU-B (2026-10-27): the tie-corrected rank correlation, same bootstrap over the resampling unit, same permutation p, and
    # `tie_tol` as for spearman. Its R squared is tau-b squared, a squared rank correlation and not a share of variance.
    "kendall": (kendall, "correlation"),
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
    # POOLED SD AND SE (2026-10-28): the within-group sd of replicates pooled over groups, `group = "<column>"` naming the unit within which the
    # replicates sit (it must be the resampling unit; the bootstrap resamples groups). Interval: the same bootstrap, with the exact chi-square
    # interval beside it. The row carries the diagnostics for pooling and the trigger rule `pool_justified`, which never changes the verdict.
    "pooled_sd": (_pooled_sd, "value"),
    "pooled_se": (_pooled_se, "value"),
    # POOL CHECK (2026-10-28): "pooling these groups is justified", adjudicated as a claim (see `pool_check_outcome`). Deterministic: estimate 1 when
    # no computed criterion fails, 0 when one does, claimed 1; no interval, because nothing is resampled. Same `x = y`, `group`, `protocol` keys.
    "pool_check": (_pool_check_placeholder, "value"),
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
                  estimate=pearson, n_perm: int = N_BOOT) -> tuple[float, float] | None:
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
    its own ROW POSITIONS and receives the y values of another cluster, drawn
    with replacement when the two differ in size. x and y are then independent
    across clusters while each cluster's own y distribution is intact.

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
        if abs(r) >= abs(obs):
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


def _paired(rows: list[dict], x: str, y: str, unit: str | None):
    """Rows usable for BOTH columns, kept together, with the cluster label.

    Paired deliberately: filtering each column separately would compare
    `x[i]` against `y[j]`, which is a correlation between two different
    subsets and is not a number about anything.
    """
    pairs, clusters, dropped = [], [], 0
    for r in rows:
        try:
            a, b = float(r[x]), float(r[y])
        except (ValueError, KeyError):
            dropped += 1
            continue
        if a != a or b != b or abs(a) == float("inf") or abs(b) == float("inf"):
            dropped += 1               # non-finite is not missing and not a value
            continue
        pairs.append((a, b))
        if unit:
            clusters.append(str(r.get(unit, "")))
    return pairs, (clusters if unit else None), dropped


def _range_verdict(rng, lo: float, hi: float, where: str, verdict) -> dict:
    """A `claimed_range = [lo, hi]` (-inf or inf for an open side), for a value claim and a correlation alike, judged by where the
    whole interval lies: inside the range, agrees; entirely outside, refuted; across a bound, unverifiable, because this design cannot
    tell on which side the value is. No width limit is applied: an interval wide enough not to discriminate already straddles a bound
    of any range narrow enough to say something."""
    # An open side is written as TOML's -inf or inf (TOML has no null); NaN is refused rather than read as "no bound", because every
    # comparison with it is false and the claim would come back unverifiable for a reason nobody wrote.
    if not isinstance(rng, list) or len(rng) != 2 or any(isinstance(x, float) and math.isnan(x) for x in rng):
        return verdict(UNVERIFIABLE, f"claimed_range {rng!r} is not [lo, hi] with -inf or inf for an open side")
    a, b = float(rng[0]), float(rng[1])
    text = f"[{a:g}, {b:g}]"
    if a <= lo and hi <= b:
        return verdict(AGREES, f"{where} lies within the claimed range {text}")
    if hi < a or lo > b:
        return verdict(REFUTED, f"{where} lies OUTSIDE the claimed range {text}")
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
    if rel["method"] in ("pooled_sd", "pooled_se"):
        where = f"the 95% {row['interval']} CI [{lo:.4g}, {hi:.4g}] for {rel['method']} = {v:.4g}{count}, {n_eff} {unit or 'row'}(s)"
    elif rel["method"] == "binomial_count":
        where = (f"the exact 95% Clopper-Pearson CI [{lo:.4g}, {hi:.4g}] for the share = {v:.4g}{count}, {n_eff} independent "
                 f"{unit or 'row'}(s), nothing resampled")
    else:
        where = f"the 95% CI [{lo:.4g}, {hi:.4g}] for {rel['method']} = {v:.4g}{count}, resampling {n_eff} {unit or 'row'}(s)"
    if rel.get("claimed_range") is not None:
        return _range_verdict(rel["claimed_range"], lo, hi, where, verdict)
    claimed = rel.get("claimed")
    if claimed is None:
        return verdict(UNVERIFIABLE, "a value claim states neither `claimed` nor `claimed_range`")
    if not (lo <= claimed <= hi):
        return verdict(REFUTED, f"claimed {claimed} lies OUTSIDE {where}")
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
            # found here only by drawing it. So the declaration names the
            # figure, `relate` records whether the file is actually on disk,
            # and `--require-figures` exits non-zero when one is missing.
            "figure": str(rel.get("figure", "") or ""),
            "figure_exists": bool(rel.get("figure")
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

        absent = [c for c in (x, y) if not rows or c not in rows[0]]
        if absent:
            out.append(verdict(UNVERIFIABLE,
                               f"the data has no column {', '.join(absent)}"))
            continue
        undeclared = [c for (d_, c) in ((xd, x), (yd, y))
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
            pairs, clusters, dropped = _paired(rows, x, y, unit)
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
        r = estimate([p[0] for p in pairs], [p[1] for p in pairs])
        if r is None:
            out.append(verdict(UNVERIFIABLE,
                               f"r is undefined on {len(pairs)} usable pair(s)"))
            continue
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
        row["r_squared"] = round(r * r, 6)
        row["r_squared_is_on"] = {"spearman": "ranks", "kendall": "tau-b"}.get(rel["method"], "values")
        if rel["method"] == "kendall":
            # tau-b squared is not the variance of anything, so say so on the row rather than leave "R squared" to be read as one.
            row["r_squared_note"] = "tau-b squared, a squared rank correlation; not a share of variance"
        pp = permutation_p(pairs, clusters, estimate=estimate)
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
        else:
            row["p_value"] = None
            row["p_method"] = "permutation undefined on these pairs"
        where = (f"the 95% CI [{lo:.3f}, {hi:.3f}] for r = {r:.3f}, "
                 f"resampling {n_eff} {unit or 'row'}(s)")
        if rel.get("claimed_range") is not None:
            out.append(_range_verdict(rel["claimed_range"], lo, hi, where, verdict))
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
            out.append(verdict(REFUTED, f"claimed {rel['claimed']} lies "
                                        f"OUTSIDE {where}"))
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
            "claims_without_a_figure": [
                (r["id"] or f"{r['x']} ~ {r['y']}") for r in out
                if not r.get("figure_exists")],
            "relationships": out, "verdicts": verdicts, "levels": levels,
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
    missing = d.get("claims_without_a_figure", [])
    if missing:
        print(f"  {len(missing)} of {len(d['relationships'])} claim(s) name no "
              f"figure on disk: " + ", ".join(missing[:8])
              + ("..." if len(missing) > 8 else ""))
        print("  A verdict is a number and a number cannot show the shape that "
              "made it. Add `figure = \"figures/x.png\"` to the relationship.")
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
