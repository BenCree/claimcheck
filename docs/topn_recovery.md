# `topn_recovery`: k of the K actives are in the shortlist, with the exact hypergeometric p

Added 2026-11-12 for virtual screening. `kind = "value"`, `method = "topn_recovery"`.

```toml
[[relationships]]
id = "S3_actives_in_top50_p38"
x = "dock_p38.is_hit"          # the LABEL: 0/1, true/false or yes/no
y = "dock_p38.in_top50"        # the SHORTLIST indicator, written by the rule that cut the list
kind = "value"
method = "topn_recovery"
claimed_range = [0.3, 1.0]
resampling_unit = "compound"   # one row per compound, the WHOLE library
```

## The problem it answers

"How many of the 7 actives are among the first 50 of 357" had been headed for `binomial_count` on a table of the 7 actives. That
method reports a two-sided binomial p against 0.5, which is the wrong null twice over.

- **The chance share is not 0.5.** It is the shortlist over the library, 50 / 357 = 0.1401. Against 0.5, 0 of 7 reads p = 0.016
  although it is the likeliest count under chance (probability 0.344), and 4 of 7 reads p = 1.0 although the chance of 4 or more
  is 0.0087.
- **The count is not binomial.** The actives compete for the same 50 places, so they are not independent trials: under chance the
  count is hypergeometric. A binomial tail at 0.1401 is too large by 3% at 3 of 7, 8% at 4 of 7 and 46% at 7 of 7.

| k of 7 in the first 50 of 357 | `binomial_count` p (against 0.5) | binomial P(X >= k) at 0.1401 | exact hypergeometric P(X >= k) |
|---|---|---|---|
| 0 | 0.0156 | 1.0000 | 1.0000 |
| 1 | 0.1250 | 0.6522 | 0.6556 |
| 2 | 0.4531 | 0.2558 | 0.2551 |
| 3 | 1.0000 | 0.0620 | 0.0602 |
| 4 | 1.0000 | 0.0094 | 0.0087 |
| 5 | 0.4531 | 0.00088 | 0.00076 |
| 6 | 0.1250 | 0.000046 | 0.000036 |
| 7 | 0.0156 | 0.0000011 | 0.0000007 |

## A new method, not an option on `binomial_count`

An option would have had the claim declare the shortlist depth and the library size beside a table of actives, and neither number
could then be checked against anything. This method reads the table of the whole library and counts all four numbers itself: K
actives, N shortlisted, M compounds, k both. Nothing about the design is declared, so nothing declared can be wrong.

## What it computes

- **The columns** are indicators: 0/1, true/false or yes/no. `x` is the label and `y` the shortlist, the same order as `roc_auc`
  and `bedroc` (and the reverse of `f1`, where `x` is the prediction). A row missing either value is dropped and counted. Any other
  value, a number other than 0 or 1 included, refuses the claim: the shortlist is cut by the rule that writes the table.
- **One row per `resampling_unit`**, the whole library. A compound with several rows is refused, naming it, because which pose
  decides membership of the shortlist is a choice about the question.
- **Estimate**: k / K, the share of actives recovered. `enrichment_factor` is that share over the chance share N / M (`p_null`).
  `max_share` is min(1, N / K): a shortlist smaller than the set of actives cannot recover them all.
- **p**: exact, from integer binomial coefficients. `p_enrichment` is P(X >= k) for X hypergeometric, which is Fisher's exact test,
  one-sided. `p_depletion` is P(X <= k). `p_value` is twice the smaller of the two, capped at 1, the convention `sign_test` uses;
  it is not the two-sided p of `scipy.stats.fisher_exact`, which sums the tables no more probable than the observed one.
- **Verdict**: the interval against `claimed` or `claimed_range`, as for every value claim. Nothing is resampled.
- Row fields: `n_success`, `n_actives`, `n_shortlist`, `n_library`, `n_rows`, `dropped`, `estimate`, `ci_lo`, `ci_hi`, `ci_width`,
  `interval`, `interval_source`, `p_null`, `enrichment_factor`, `max_share`, `p_enrichment`, `p_depletion`, `p_value`, `p_method`.
- Swapping the two columns leaves the p unchanged, because the 2 x 2 table is symmetric, and changes the estimate from recall
  (k / K) to precision (k / N). The four counts are on the row so that a swapped declaration can be seen.
- Refused as `unverifiable`: `x = y`, no active in the table, a shortlist holding nothing or everything, a library with no
  inactive.

## What the interval means when the trials are not independent

The interval is Clopper-Pearson on k of K, and it is reported. It is an interval for the **expected share of actives this scorer
places in the shortlist, for actives like these against this library**, and it treats the K actives as independent trials. They
are not: one active entering the shortlist pushes the threshold up for the others. That dependence is negative, so the count
varies less than a binomial count would, and an interval built for the binomial is too wide, not too narrow.

This was measured before deciding to report it, with the figure that would have withheld it written first (coverage below 0.93 in
any design). Over 25 designs (7 of 357 with shortlists of 50 and 18, 30 of 1,500 with 75, 40 of 200 with 50, 20 of 100 with 10;
scorers from chance to strong; 20,000 simulated screens each) the 95% interval covered the expected share in 0.959 to 1.000 of
screens, and the variance of the count was 0.09 to 0.98 of the binomial variance. So the interval is conservative: it can be
trusted to contain the share at least as often as it says, and it is wider than an exact one would be, most of all when the
shortlist is small against the number of actives. `docs/screening_metrics_study.md` has the table.

Two things the interval is not. It is not the null distribution: whether k is more than chance is `p_enrichment`, and chance is
N / M, not 0. And it says nothing about a different library: the share recovered depends on the decoys, so the matched-decoy
table belongs beside it as it does beside an AUC.

## What 7 actives can show

| k of 7 | 95% interval | excludes the chance share 0.1401 |
|---|---|---|
| 0 | [0.000, 0.410] | no |
| 1 | [0.004, 0.579] | no |
| 2 | [0.037, 0.710] | no |
| 3 | [0.099, 0.816] | no |
| 4 | [0.184, 0.901] | yes |
| 5 | [0.290, 0.963] | yes |
| 6 | [0.421, 0.996] | yes |
| 7 | [0.590, 1.000] | yes |

So `claimed_range = [0.3, 1.0]` agrees at 6 or 7 of 7 and is `unverifiable` at 5 or fewer; it is never refuted at 7 actives. Choose
the range knowing this table.

## Unchanged

`binomial_count` is as it was, p against 0.5 included: it answers "k of n independent units pass a rule". For a count of actives
in a shortlist, use this method.
