# `binomial_count`: k of n with the exact interval

Added 2026-11-09 (branch `binomial-count`) for the z-ml paper's open check O38.

## The problem it answers

A registered claim "a share of at least 0.75 of 8 ligands is within the tolerance", adjudicated with `fraction`, takes the percentile
bootstrap over ligands. At 8 of 8 every resample is 8 of 8, so the interval is [1, 1]; at 7 of 8 it is about [0.625, 1]. The claim can
therefore agree only at 8 of 8, and an `unverifiable` at 7 of 8 reports the panel size rather than the measurement. The bootstrap also
understates how little 8 of 8 shows: the exact 95% interval for 8 of 8 is [0.631, 1].

## What it computes

```toml
[[relationships]]
id = "share_within_0p3"
x = "ligands.w_t030"
y = "ligands.w_t030"          # one-column method: x = y
kind = "value"
method = "binomial_count"
claimed_range = [0.5, 1.0]
resampling_unit = "ligand"    # one row per ligand, or the claim is refused
```

- **The column** is an indicator: 0/1, true/false or yes/no. A missing or non-finite value is dropped and counted (`dropped`). Any
  other value, including a number other than 0 or 1, refuses the claim: thresholding belongs to the rule that writes the column.
- **One row per `resampling_unit`.** The binomial assumes independent trials. Several rows under one unit (three end-state runs of
  one ligand) are refused, naming the units, because how to collapse them to one trial (majority, all, any, a run average) is a
  choice about the question. With no unit declared, each row is a trial.
- **Estimate**: k / n. **Interval**: Clopper-Pearson (1934), the 0.025 quantile of Beta(k, n - k + 1) and the 0.975 quantile of
  Beta(k + 1, n - k), with the lower bound exactly 0 at k = 0 and the upper bound exactly 1 at k = n. Coverage is at least 95% for
  every true share (it is conservative). Standard library only: the beta quantile is a bisection on the regularised incomplete beta.
- **p**: the exact two-sided binomial p against 0.5 (`sign_test_p`), on the row as `p_value` with `p_null = 0.5`. R squared is None.
- **Verdict**: the interval against `claimed` or `claimed_range`, as for every value claim. Nothing is resampled.
- Row fields: `n_success`, `n_units`, `n_rows`, `dropped`, `interval = "clopper_pearson"`, `interval_source`, `p_value`, `p_null`,
  `p_method`, `ci_lo`, `ci_hi`, `ci_width`.

## What an n of 8 or 6 can show

| k of 8 | 95% interval | k of 6 | 95% interval |
|---|---|---|---|
| 8 | [0.631, 1] | 6 | [0.541, 1] |
| 7 | [0.473, 0.997] | 5 | [0.359, 0.996] |
| 6 | [0.349, 0.968] | 4 | [0.223, 0.957] |
| 5 | [0.245, 0.915] | 3 | [0.118, 0.882] |

So `claimed_range = [0.5, 1]` agrees only at 8 of 8 (or 6 of 6), and `[0.75, 1]` can never agree at n = 8. The count and its interval
are the result; the range decides only the verdict word. Choose the range knowing this table.

## Unchanged

`fraction` and `sign_test` keep the bootstrap interval. The independence assumption is the claim author's: units nested in a
higher-level group (ligands of one protein) are treated as independent trials.

## Not for a count of actives in a shortlist

"k of the 7 actives are in the first 50 of 357" is not a `binomial_count` claim. The p here is against 0.5, and the chance share
of a shortlist is its depth over the library (0.1401), with a hypergeometric count because the actives compete for the same
places: 0 of 7 reads p = 0.016 here and is the likeliest count under chance. Use `topn_recovery` (`docs/topn_recovery.md`, added
2026-11-12), which counts the table of the whole library and reports the exact hypergeometric p.
