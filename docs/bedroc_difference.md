# `bedroc_difference`: does this score put the actives higher up the list than that one does

Added 2026-11-22. `kind = "correlation"`, `method = "bedroc_difference"`.

```toml
[[relationships]]
id = "P37_esr1_cnnaff_minus_heavy_atoms_bedroc"
x = "gate_esr1.is_active"         # the LABEL, read as an indicator at >= 0.5
y = "gate_esr1.cnn_affinity"      # the SCORE, HIGHER means more likely positive
baseline = "gate_esr1.n_heavy"    # the column y is compared against, same orientation, SAME TABLE
kind = "correlation"
method = "bedroc_difference"
alpha = 20                        # REQUIRED, no default, applied to BOTH columns
claimed_range = [0.0, 1.0]        # the prediction: y gathers the actives earlier than the baseline does
resampling_unit = "unit_id"
```

## Why it exists beside `roc_auc_difference`

`roc_auc_difference` (2026-11-20, `docs/roc_auc_difference.md`) answered "does this score separate the actives better
than that one does" for AUROC, which is not the statistic a screen is used by. On
`/home/ben/code/projects/targets/cry1/experiments/litpcba_flexdock` BEDROC at alpha 20 is **declared as the primary
metric** (`context.toml`, B1 and B2: "at a 2 per cent active rate AUROC is dominated by the ordering of the inactives
among themselves, which nobody acts on"), and its paired difference had no estimator, so the question sat in
`context.pending.toml` as D2 from 2026-11-18 while the P36 comparison was decided on a statistic that experiment does not
rank by.

The two can disagree, and the disagreement is the finding. From the fixture in `docs/bedroc.md`, 10 actives among 1,000:

| 10 actives at list positions | AUROC | BEDROC (alpha = 20) |
|---|---|---|
| 1 to 5 and 496 to 500 | 0.7525 | 0.5250 |
| 248 to 257 | 0.7505 | 0.0072 |

The AUROC difference is 0.002, which no cohort could resolve; the BEDROC difference is 0.518.
`tests/test_bedroc_difference.py::test_it_separates_two_screeners_auroc_cannot` is that table as one claim.

## The two things that do not carry over from the AUROC form

**`alpha` is required, has no default, and is applied identically to both columns.** It decides what "early" means:
0.798 of the weight in the first 8% of the list at alpha = 20, 0.332 at alpha 5, 0.998 at alpha 80 (`docs/bedroc.md`).
It is bound once, in `check`, into the single estimator the bootstrap and the swap null both call, so no code path can
weight the two columns differently; the row records it, and the interval text names it ("recomputes BOTH BEDROCs at
alpha = 20 on the drawn rows"). Two alphas on one table are two different claims with two different numbers, and a
claim with no alpha, or an alpha that is not a positive finite number, is `unverifiable` and says so.

**BEDROC's random expectation is not 0.5 and depends on the active fraction, and the null of the difference is still
zero.** A single BEDROC is read against `bedroc_random_expectation` (0.0604 at 7 actives of 357, 0.4986 at 178 of 357),
never against 0. That number is a property of the rows and alpha, not of a column: the two columns of one claim are
scored on IDENTICAL ROWS, so they share the active count, the row count, alpha, and therefore `S_min`, `S_max` and the
expectation. The difference is

    (S_score - S_baseline) / (S_max - S_min)

so the normalising constants cancel in the estimate, in every bootstrap resample and in every swap draw. Zero means
"the two columns gather the same weight of actives at the top of their lists", which is the hypothesis, and the
expectation never enters the difference. It is on the row all the same, because `bedroc_y` and `bedroc_baseline` beside
it are meaningless without it: 0.10 against 0.06 is a screener barely above chance, 0.10 against 0.01 is one well above
it. `test_the_random_expectation_is_shared_and_cancels` pins this on the numbers, including the clean case, a constant
column, which scores exactly the expectation, so two constants differ by exactly 0.0.

## The conventions it will not guess for you

Those of `roc_auc_difference`, unchanged, because they are the same construction:

- **`x` is the label, `y` is the score, `baseline` is the column `y` is compared against.** None of them commutes, and
  the sign of the result is `y` minus `baseline`.
- **Higher means more likely positive in both score columns.** A docking affinity, where better is more negative, is
  negated by the rule that writes the table, in a column whose name says so. Two columns oriented differently would
  produce a difference with no reading, and nothing here can detect that.
- **`baseline` must be a column of the same dataset as `y`:** a paired difference is a statement about identical rows,
  and `join` lines up two datasets, not three.
- **`baseline` is refused on every method outside `relate.PAIRED_DIFFERENCE_METHODS`**, so a key no method reads cannot
  look as though it had an effect.
- **`x` equal to `y` is refused:** the label as its own score is an oracle, not a comparison of two scores.
- **A row is kept only when all three values are usable**, and dropped from the claim rather than from one column of it
  (`relate._paired`).
- **Either class empty returns `unverifiable`**, never 0.0: a join that dropped every active must not read as "the two
  scores are equal".

## The interval is the paired bootstrap

The **paired percentile bootstrap over the declared `resampling_unit`**, the package's existing `bootstrap_ci`: each
resample draws whole units and recomputes BOTH BEDROCs on the drawn rows, never the two independently, the baseline
travelling inside the pair as `(label, (score, baseline))`. The reasons are `docs/roc_auc_difference.md`'s and they are
not repeated here: clustered rows have no closed-form variance in this package, every verdict in it is "does the claimed
value lie inside the bootstrap interval", and 2,000 resamples of 5,047 rows costs seconds. Its cost is the same one:
the percentile bootstrap is too narrow with few units (Hesterberg 2015, `docs/pooled_sd.md`, measured coverage 0.86 at
8 groups).

**The default width limit binds here and does not bind on a plain `bedroc` claim.** A BEDROC lives in [0, 1], where the
default `MAX_INFORMATIVE_WIDTH` of 1.0 can never bind, which is why `docs/bedroc.md` tells a claim to declare
`max_ci_width`. A BEDROC *difference* spans [-1, 1], so a width of 1.0 is half of everything, exactly as for a
correlation, and the default behaves as it does everywhere else. Declare a tighter one where the question needs it.

**Resolution is measured, not assumed, and the floor is not one number.** Register the random control (an uninformative
column against a neutral one) and read every `unverifiable` against its width. BEDROC's sampling spread grows with
skill, unlike AUROC's: at 7 actives of 357 the 95% bootstrap interval was 0.43 to 0.59 wide for a screener with real
skill and 0.11 to 0.13 wide at chance (`docs/screening_metrics_study.md`). So the null difference's width is a floor
for comparisons between weak columns and an underestimate for comparisons between strong ones, and each result's own
width is quoted beside it.

## The p is the paired swap test, with this statistic

`paired_swap_p`, which took the statistic as an argument from this date (the RNG draw order is untouched, so every p
already adjudicated under the AUROC form is byte-identical). Each resampling unit gets an independent fair coin and
that unit's two score values are swapped when it comes up heads; the whole unit swaps or none of it does. Both columns
are put on mid-ranks over the whole table first, because a swap mixes values from two columns into one column that the
statistic then ranks together. **BEDROC is invariant under that rank map**, exactly: it reads only the order of the
scores and which of them are tied, and mid-ranks are a monotone tie-preserving relabelling
(`test_mid_ranks_do_not_change_a_bedroc`).

The null is **exchangeability of the two columns**, not that either reads the label. The label permutation would ask
the wrong question and would call two identical excellent scorers decisive. The null distribution is symmetric about 0
because swapping every unit of a draw gives exactly the negated statistic, so the two-sided comparison is centred on 0
even though neither term is centred on a constant, for the reason given above.

**Where it is approximate, and it bites harder here than for AUROC.** Exchangeability needs the two columns to be
interchangeable under the null, and two real score columns rarely share a tie structure: heavy-atom count takes a few
dozen values over thousands of compounds while a CNN affinity is almost tie-free, so a swap moves a tied block into
positions the other column resolved. For AUROC that perturbation is spread over the whole list. For BEDROC at alpha 20
it lands where 0.798 of the weight sits, which is the part of the list the statistic is made of. Bounded rather than
asserted, and the bounds are loose on purpose: over 120 tables of 80 rows in 40 units with two exchangeable columns the
share of p at or below 0.05 is under 0.18
(`test_the_p_is_calibrated_under_exchangeable_columns`), and with a coarsely tied column against a tie-free one, both
carrying the same relationship to the label, it is under 0.35 over 80 such tables
(`test_a_coarsely_tied_column_against_a_tie_free_one_is_the_approximate_case`). Neither test reads the rate off as a
calibration figure: they catch a p that tests the label (every table would reject) or is constant (none would), and the
tie-mismatch case is the one that has not been pinned down. **No verdict depends on the p**: the verdict is the claimed
value against the paired bootstrap interval.

## What the row carries

`estimate`, `ci_lo`, `ci_hi`, `ci_width`, `n_units`, `n_rows`, `dropped`, `p_value`, `p_floor`, `p_at_floor` and
`p_method` as every correlation claim does, plus:

- `bedroc_y` and `bedroc_baseline`, the two BEDROCs the difference is between.
- `alpha` and `weight_share_top_8pct`, the weight the first 8% of the list carries at that alpha.
- `n_actives`, `active_fraction` and `alpha_times_active_fraction`. The paper's reading of BEDROC as a probability needs
  the last of these well below 1 (`docs/bedroc.md`).
- `random_expectation` with `random_expectation_note`, which says in the row itself that it is shared by both columns
  and cancels in the difference, so nobody reads the difference against it.
- `baseline` and `baseline_unit`, the third column and its declared unit.
- `interval_source`, naming the paired bootstrap, what it resamples and the alpha it recomputes both BEDROCs at.
- `r_squared = None`. Neither BEDROC has one, so their difference has none, and squaring a difference of two weighted
  early-recognition scores produces a number with no reading. The row says so in `r_squared_note`.

## The figure

One claim figure, drawn by `claimcheck.figures`: **two weighted enrichment curves on one pair of axes**, the score
solid and the baseline dashed, with the chance ranking dotted and all three in the key. At list position k the curve is
`(sum of the weights of the actives at positions <= k - S_min) / (S_max - S_min)`, so each curve **ends exactly at its
own column's BEDROC** and starts at 0 to within rounding, and **the vertical gap between the two curves at the
right-hand edge is the estimate**, as the signed area between two ROC curves is the AUROC difference
(`test_the_curve_ends_at_the_bedroc_and_the_gap_is_the_estimate`).

The ROC pair is deliberately not reused: two screeners whose BEDROCs differ by 0.5 can have ROC curves almost on top of
one another (the table above), so that canvas would show a reader two identical lines beside an estimate of half the
scale. The chance line here is a **curve** and not a diagonal, ending at `random_expectation`, which is the point
BEDROC exists to make. One vertex is one distinct score value, a tied block one step carrying the mean weight over the
positions it spans. No words on the canvas beyond the axis labels, the tick labels and the key; every mark is defined
in `<id>.commentary.md`.

## Reading it

- **An interval that straddles zero is UNRESOLVED, not "no difference"** (`claimcheck/docs/RULES.md`, Reporting). With
  `claimed_range = [0.0, 1.0]` that is reported as `unverifiable`, which is never a pass.
- **Report it beside the AUROC difference on the same rows, never pooled across targets.** They answer different
  questions: one is about the whole ordering, the other about the top of the list. Where they disagree, that is the
  result, and which one the campaign acts on is the shortlist's depth, not a preference.
- **Register the known answer.** `y` against a copy of itself is exactly 0.0 with a zero-width interval, the only cheap
  check that the pairing in the table is what the claim thinks it is. Register it as a point `claimed = 0.0`, not a
  `claimed_range` with 0.0 as a bound: `claimcheck.scrutiny` flags an estimate sitting on a bound of its own range, and
  0.0 is interior to [-1, 1]. The two-sided partner is a perfect ranking against its own inversion, exactly +1.0 at any
  active fraction, because BEDROC does not saturate.
