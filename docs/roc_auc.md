# `roc_auc`: does the score put the known binders near the top

Added 2026-11-12 for virtual screening. `kind = "correlation"`, `method = "roc_auc"`.

```toml
[[relationships]]
id = "S1_score_ranks_hits_p38"
x = "dock_p38.is_hit"          # the LABEL, read as an indicator at >= 0.5
y = "dock_p38.neg_score"       # the SCORE, HIGHER means more likely positive
kind = "correlation"
method = "roc_auc"
claimed = 0.7
resampling_unit = "compound"
```

## What it is

The probability that a randomly chosen positive outscores a randomly chosen negative, a tie counting
a half. 0.5 is chance, 1.0 is a perfect ranking, 0.0 is exactly inverted. Computed from average ranks
(the Mann-Whitney U identity), so a tie between a positive and a negative contributes exactly a half
instead of being ordered by floating-point noise.

## Why it is not a correlation

With 29 hits among 1,450 decoys, a Spearman between hit status and score is dominated by the ordering
of the decoys among themselves, which nobody acts on. AUC asks only about hit against decoy pairs,
which is the question a shortlist poses. `f1` answers a neighbouring question and needs a declared
shortlist depth; AUC needs none, and so says nothing about how deep to cut.

## The conventions it will not guess for you

- **`x` is the label and `y` is the score.** They do not commute.
- **Higher `y` means more likely positive.** A docking affinity, where better is more negative, must
  be negated by the rule that writes the table, in a column whose name says so. The sign is a choice
  that could have gone differently, so it is declared, not inferred.
- **Either class empty returns `unverifiable`**, not 0.5, so a join that dropped every hit is
  distinguishable from an honest chance result.

## What the row carries

The same bootstrap interval over `resampling_unit`, `n_rows`, `dropped`, and a cluster permutation p
**centred on 0.5**. That centring is the one change to shared machinery: `permutation_p` gained a
`centre` argument, default 0.0, so every other method behaves exactly as before. Without it the test
is one-sided, because AUC lives in [0, 1] and comparing magnitudes about 0 asks only whether the
permuted AUC exceeds the observed one; a strongly inverted score, which is as informative as a good
one and says the sign convention is wrong, would come back unremarkable. `tests/test_roc_auc.py` pins
that case.

`r_squared` is **not** reported. AUC is a probability of correct ordering, not a correlation, so its
square is not a share of anything; the row says so in `r_squared_note`.

## Reading it

An AUC is only as honest as the decoys. If the decoys differ from the hits in size or properties, the
AUC measures that difference and not the score: match the decoys and show the match, as
`hidden_hit_bench` does with a property-parity table and a classifier that must fail.

## What it cannot see, and the two methods beside it

AUC is the mean over positives of the share of negatives each one outranks, so every positive counts alike wherever it sits: with
10 positives among 1,000, moving one from position 500 to 400 changes it by exactly as much as moving another from 101 to 1
(100 / 9900 each, provided only negatives are crossed). A screener with 5 positives at the very top and 5 in the middle scores
0.7525 and one with all 10 a quarter of the way down scores 0.7505. For early recognition use `bedroc` (`docs/bedroc.md`), and for a
count of actives in a shortlist use `topn_recovery` (`docs/topn_recovery.md`).

## The permutation p before 2026-11-12

The comparison `abs(r - 0.5) >= abs(obs - 0.5)` was made in floating point, where |8/9 - 0.5| and |1/9 - 0.5| differ in the last
digit, so a permuted AUC exactly as extreme as the observed one on the other side of 0.5 could be missed and the p came back too
small: 3/20 where the exact value is 4/20 with 3 positives against 3 negatives, and by at most two permutation draws in 300 at 7
positives among 357. The comparison now allows an absolute tolerance of 1e-12 (`PERMUTATION_TIE_TOL` in `relate.py`). A p from an
earlier checker can only have been too small, never too large, and no verdict depends on p.
