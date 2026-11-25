# `roc_auc_difference`: does this score separate the actives better than that one

Added 2026-11-20 for proposal P36. `kind = "correlation"`, `method = "roc_auc_difference"`.

**The same construction on the metric a screen is used by is `bedroc_difference`** (2026-11-22,
`docs/bedroc_difference.md`). Where a shortlist is what the result feeds, that one is the claim to register, and the two
answer different questions: this one is about the whole ordering, that one about the top of the list.

```toml
[[relationships]]
id = "G1_cnnaff_beats_heavy_atoms_esr1"
x = "esr1_rigid.is_active"        # the LABEL, read as an indicator at >= 0.5
y = "esr1_rigid.cnn_affinity"     # the SCORE, HIGHER means more likely positive
baseline = "esr1_rigid.n_heavy"   # the column y is compared against, same orientation, SAME TABLE
kind = "correlation"
method = "roc_auc_difference"
claimed_range = [0.0, 1.0]        # the prediction: y beats the baseline
resampling_unit = "unit_id"
```

## What it is, and why two separate AUROCs cannot answer it

`AUROC(y) - AUROC(baseline)`, both computed on identical rows against one label. Positive means `y` separates the
actives better. It spans [-1, 1] and is exactly 0.0 when the two columns are the same column.

The reason it exists is a measurement. On
`/home/ben/code/projects/targets/cry1/experiments/litpcba_flexdock/results/claims.json` the CNN affinity reads
0.7323 [0.6653, 0.7930] (`A2_esr1_rigid_cnnaff_auc`) and heavy-atom count 0.6573 [0.5877, 0.7230]
(`H1_esr1_heavy_atoms_auc`) over the same 5,047 compounds and the same 3,925 resampling units. The two intervals
overlap almost entirely, and **two overlapping intervals say nothing about their difference**: most of each interval's
width is the variance of drawing this cohort, which both columns share and which cancels when they are resampled
together. Reading "the difference contains zero" off that overlap is the error this estimator was added to stop, and
until it existed those claims sat unadjudicated in `context.pending.toml` as D1 and D2.

How much the pairing buys is in `tests/test_roc_auc_difference.py`: on two informative, strongly correlated columns
the paired interval is more than ten times narrower than either column's own, and on two identical columns it is
exactly [0, 0] where independent resampling would have produced a wide one.

## The conventions it will not guess for you

- **`x` is the label, `y` is the score, `baseline` is the column `y` is compared against.** None of them commute, and
  the sign of the result is `y` minus `baseline`.
- **Higher means more likely positive in both score columns**, as for `roc_auc`. A docking affinity, where better is
  more negative, is negated by the rule that writes the table, in a column whose name says so. Two columns oriented
  differently would produce a difference with no reading at all, and nothing here can detect that.
- **`baseline` must be a column of the same dataset as `y`.** A paired difference is a statement about identical rows,
  and `join` lines up two datasets, not three: the two scores belong in one table, written there by the rule that
  computed them. A baseline in another dataset is refused, naming both.
- **`baseline` is refused on every method outside `relate.PAIRED_DIFFERENCE_METHODS`** (this one and, from 2026-11-22,
  `bedroc_difference`), for the reason `tie_tol` is: a declared key no method reads is one
  the checker silently ignored, and the next reader takes it as having had an effect.
- **`x` and `y` the same column is refused.** The label used as its own score is an oracle, so the difference would be
  1 minus the baseline's AUROC and not a comparison of two scores. Declare the oracle as its own column.
- **A row is kept only when all three values are usable**, and dropped from the claim rather than from one column of
  it, for the reason the two-column form is paired (`relate._paired`).
- **Either class empty returns `unverifiable`**, never 0.0. A join that dropped every active must not read as "the two
  scores are equal".

## The interval is the paired bootstrap, not DeLong

The decision was left to whoever built this. It is the **paired percentile bootstrap over the declared
`resampling_unit`**: each resample draws whole units and recomputes BOTH AUROCs on the drawn rows, never the two
independently. The implementation is the package's existing `bootstrap_ci`, unchanged: the baseline travels inside the
pair as `(label, (score, baseline))`, and `bootstrap_ci` only ever indexes `p[0]` and `p[1]` without doing arithmetic
on either, so a drawn row carries both of its scores and the pairing cannot be lost. No existing method's random
stream moves.

Why not DeLong (DeLong, DeLong and Clarke-Pearson 1988, the standard closed form for a paired AUC difference):

1. **It assumes the rows are independent.** They are not here. On `litpcba_flexdock` the independent unit is the Butina
   active cluster, 3,925 units over 5,047 rows, and the whole reason `resampling_unit` is a required declaration in
   this package is that treating clustered rows as independent draws produces an interval roughly a fifth of its true
   width. DeLong's variance has no clustered form in this package, and adding one would be a second uncertainty
   machinery beside the one every other verdict is decided by.
2. **Every verdict in this package is "does the claimed value lie inside the bootstrap interval".** A method whose
   interval came from somewhere else would not be comparable with the rows beside it, and the master ledger's 6,889
   claims would carry two kinds of interval under one column name.
3. **It buys nothing here.** DeLong's advantage is a closed form on large tables, and 2,000 resamples of 5,047 rows
   costs seconds.

What that choice costs, stated because it is a real cost: the percentile bootstrap is **too narrow with few units**
(Hesterberg 2015, carried over in `docs/pooled_sd.md`, where the measured coverage was 0.86 at 8 groups). At the
thousands of units a screening library gives this does not bind; on a claim over 15 targets it would, and a sign test
across targets is the better design there (`claimcheck/docs/RULES.md`, "Cohort size decides resolution").

## The p is a paired randomisation test, and its null is not the obvious one

`paired_swap_p`, not `permutation_p`. The shared permutation permutes the label against the y values of whole
clusters, which here moves **both** score columns together and so tests whether EITHER column reads the label. That is
not this claim's question, and it fails in a way that matters: two excellent and identical scorers would come back with
a decisive p for a difference that is exactly zero.

So the null is **exchangeability of the two columns**: each resampling unit gets an independent fair coin, and that
unit's two score values are swapped when it comes up heads. The whole unit swaps or none of it does, because the unit
is the independent draw. Both columns are put on mid-ranks over the whole table first (`relate.mid_ranks`), because a
swap mixes values from the two columns into one column that AUROC then ranks together: a CNN affinity in [1, 8] and a
heavy-atom count in [10, 60] would otherwise be compared as magnitudes. AUROC is invariant under that rank map
exactly, so the observed statistic is untouched and only the null is affected. The null distribution is symmetric
about 0 by construction, since swapping every unit of a draw gives exactly the negated statistic, so the two-sided
comparison is about a centre of 0 (`as_extreme`, `PERMUTATION_TIE_TOL`).

**Where that is approximate.** Exchangeability needs the two columns to be interchangeable under the null, and two
real score columns rarely share a tie structure: heavy-atom count takes a few dozen distinct values over thousands of
compounds while a CNN affinity is almost tie-free, so a swap moves a tied block into positions the other column
resolved. The null is therefore approximate in a direction that is not established here. It is reported for that
reason, and **no verdict depends on it**: the verdict is the claimed value against the paired bootstrap interval, as
everywhere else. Calibration under exchangeable columns is measured rather than asserted
(`test_the_p_is_calibrated_under_exchangeable_columns`).

## What the row carries

`estimate`, `ci_lo`, `ci_hi`, `ci_width`, `n_units`, `n_rows`, `dropped`, `p_value`, `p_floor`, `p_at_floor` and
`p_method` as every correlation claim does, plus:

- `auc_y` and `auc_baseline`, the two AUROCs the difference is between. A difference of -0.08 is a different finding at
  0.52 against 0.60 than at 0.90 against 0.98, and the row is the only place a reader sees which.
- `n_actives`, the positives the two AUROCs were computed over.
- `baseline` and `baseline_unit`, the third column and its declared unit.
- `interval_source`, naming the paired bootstrap and what it resamples.
- `r_squared = None`. Neither AUROC has one, so their difference has none, and squaring a difference of two
  probabilities produces a number with no reading. The row says so in `r_squared_note`.

## The figure

One claim figure, drawn by `claimcheck.figures`: **two ROC curves on one pair of axes**, the score solid and the
baseline dashed, with the chance diagonal dotted and all three in the key. The signed area between the curves is the
estimate, which is what makes this figure the one that shows the claimed quantity: drawn as a scatter the claim would
put the binary label against the method's score, which is its `roc_auc` neighbour's figure and leaves the baseline, the
whole comparison, off the canvas. One vertex is one distinct score value, so a block of ties is one step. No words on
the canvas beyond the axis labels, the tick labels and the key; every mark is defined in `<id>.commentary.md`.

## Reading it

- **An interval that straddles zero is UNRESOLVED, not "no difference"** (`claimcheck/docs/RULES.md`, Reporting). With
  `claimed_range = [0.0, 1.0]` that outcome is reported as `unverifiable`, which is never a pass and is often the more
  valuable answer: it says this cohort cannot show the score beating the baseline.
- **A difference is only as honest as the two columns.** An AUROC is only as honest as its decoys (`docs/roc_auc.md`),
  and a difference inherits that from both sides: it cancels what the two columns share, which is exactly why it is
  worth having, and it says nothing about a property both of them read.
- **Register the known answer beside it.** `y` against a copy of itself must come back exactly 0.0 with a zero-width
  interval, and that is the only cheap check that the pairing in the table is what the claim thinks it is. Register it
  as a point `claimed = 0.0`, not a `claimed_range` with 0.0 as a bound: `claimcheck.scrutiny` flags an estimate
  sitting on a bound of its own claimed range, and 0.0 is interior to [-1, 1].
