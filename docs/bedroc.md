# `bedroc`: does the score put the known binders at the very top

Added 2026-11-12 for virtual screening, beside `roc_auc`. `kind = "correlation"`, `method = "bedroc"`.

```toml
[[relationships]]
id = "S2_early_recognition_p38"
x = "dock_p38.is_hit"          # the LABEL, read as an indicator at >= 0.5
y = "dock_p38.neg_score"       # the SCORE, HIGHER means more likely positive
kind = "correlation"
method = "bedroc"
alpha = 20                     # REQUIRED, no default
claimed_range = [0.3, 1.0]
max_ci_width = 0.4             # see "What a small design can return"
resampling_unit = "compound"
```

## What it is

BEDROC (Truchon and Bayly, J. Chem. Inf. Model. 2007, 47, 488, doi 10.1021/ci600426e). Each positive is weighted by
exp(-alpha * k / N) at list position k of N, position 1 being the highest score. With S the sum of the positives' weights,

    BEDROC = (S - S_min) / (S_max - S_min)

where S_max is the sum for the best ranking (every positive first) and S_min for the worst (every positive last). This is
algebraically the paper's closed form in RIE, sinh and cosh; `tests/test_bedroc.py` checks it against that form and against RDKit's
`CalcBEDROC`.

## Three things that were believed here and are not so

**It does not saturate.** BEDROC is bounded 0 to 1 by the min-max normalisation above. A perfect ranking scores exactly 1.0 and the
worst exactly 0.0 at any active fraction: 7 of 357, 178 of 357 and 300 of 357 alike. The belief that its maximum falls below 1 when
actives are a large share of the set is true of RIE, whose ceiling is 16.5 at 7 of 357 and 2.0 at 178 of 357 (alpha = 20), and not
of BEDROC.

**A random ranking does not score 0.** It scores `random_expectation`, which the verdict row carries:

| actives of compounds | alpha * active fraction | BEDROC of a random ranking (alpha = 20) |
|---|---|---|
| 10 of 1,000 | 0.20 | 0.0552 |
| 7 of 357 | 0.39 | 0.0604 |
| 36 of 357 | 2.02 | 0.1163 |
| 178 of 357 | 9.97 | 0.4986 |

A BEDROC is read against this number, never against 0. A screener scoring 0.007 at 10 of 1,000 is below random, not slightly above
nothing. This is also what the active fraction does change: the paper's reading of BEDROC as a probability needs
alpha * active fraction much less than 1, and the row carries that product as `alpha_times_active_fraction`.

**`alpha` has no default.** It decides what "early" means. The share of the total weight carried by the first fraction z of the
list is (1 - exp(-alpha * z)) / (1 - exp(-alpha)):

| alpha | weight in the first 8% of the list | fraction of the list holding 80% of the weight |
|---|---|---|
| 5 | 0.332 | 31.7% |
| 20 | 0.798 | 8.05% |
| 80 | 0.998 | 2.01% |

A claim with no `alpha`, or one that is not a positive finite number, is `unverifiable` and says so. The value is recorded on the
row as `alpha`, with `weight_share_top_8pct` beside it. Two claims at two alphas on one table are two different claims.

**Comparing two score columns is a different claim**, not two of these read against each other: two separately
adjudicated BEDROCs share the cohort variance that dominates each interval, so their overlap says nothing about their
difference. That is `bedroc_difference` (2026-11-22, `docs/bedroc_difference.md`), the paired difference at one alpha on
identical rows.

## Why it exists beside `roc_auc`

AUC weights every positive alike wherever it sits: with 10 positives among 1,000, moving one from position 500 to 400 changes it by
exactly as much as moving another from 101 to 1. So it cannot tell these two screeners apart, and BEDROC can:

| 10 actives among 1,000 at positions | AUC | BEDROC (alpha = 20) | actives in the first 50 |
|---|---|---|---|
| 1 to 5 and 496 to 500 | 0.7525 | 0.5250 | 5 |
| 248 to 257 | 0.7505 | 0.0072 | 0 |

## The conventions it will not guess for you

- **`x` is the label and `y` is the score**, as for `roc_auc`. They do not commute.
- **Higher `y` means more likely positive.** A docking affinity is negated by the rule that writes the table, in a column whose
  name says so.
- **Either class empty returns `unverifiable`**, not a number.
- **Ties share.** Compounds with equal scores share the mean of the weights of the positions they occupy, the expectation over a
  random order of the tie. A bootstrap draw repeats rows, so ties are the ordinary case there.

## What the row carries

`estimate`, the bootstrap interval over `resampling_unit` (`ci_lo`, `ci_hi`, `ci_width`, `n_units`), `alpha`, `n_actives`,
`active_fraction`, `alpha_times_active_fraction`, `random_expectation`, `weight_share_top_8pct`, `n_rows`, `dropped`, and a
permutation p. `r_squared` is not reported: BEDROC is not a correlation and its square is not a share of anything.

**The p is the distance from `random_expectation`, and it detects enrichment only.** The floor of BEDROC is 0, which is only
`random_expectation` below the null's centre (0.06 at 7 of 357), so the worst possible ranking is not far from the centre and its p
is large. `tests/test_bedroc.py` pins this: seven actives in the last seven places of 357 give BEDROC 0.0 with p above 0.10, while
`roc_auc` on the same table is at its p floor. An inverted score, which says the sign convention is wrong, is a question for
`roc_auc`. This centring was a choice; twice the smaller tail would have power on both sides and was not implemented.

## What a small design can return

BEDROC lives in [0, 1], so the default limit on an informative interval (width 1.0) never binds, and a claim inside any interval
would read `agrees`. Declare `max_ci_width`. With 7 actives among 357 the median 95% bootstrap interval is 0.43 to 0.59 wide for a
screener with real skill (mean AUC 0.76 to 0.96) and 0.11 to 0.13 wide at chance (`docs/screening_metrics_study.md`). At 7 actives
a `claimed_range` narrower than about 0.5 will therefore usually come back `unverifiable` for a good screener, and the comparison
with AUC reverses with skill: BEDROC has the wider interval from a mean AUC of about 0.75 upward, and the narrower one near chance.
The sampling sd of BEDROC at 7 of 357 was at most 0.127 under the paper's exponential model of active positions, and 0.189 for a
screener that places each active at the very top or the very bottom. The paper's planning figure 1/sqrt(8n), n the number of
actives, is a fit to simulations of that one model and not a bound; the bound for a mean of n terms in [0, 1] is 1/sqrt(4n).
