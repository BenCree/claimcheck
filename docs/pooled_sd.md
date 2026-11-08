# Pooled sd, pooled se and the pooling check

Added to `claimcheck.relate` on 2026-10-28 as `method = "pooled_sd"`, `"pooled_se"` and `"pool_check"`, all `kind = "value"`. They sit beside
`sd` and `mean`; they replace neither. Code: `/home/ben/code/claimcheck/claimcheck/relate.py` (block above `ESTIMATORS`). Tests:
`/home/ben/code/claimcheck/tests/test_pooled_sd.py`.

## What it computes

Replicates sit inside groups (runs inside a ligand). With `n_g` values in group `g` and sample variance `s_g^2`:

    pooled sd = sqrt( sum_g (n_g - 1) s_g^2 / sum_g (n_g - 1) )        degrees of freedom  df = sum_g (n_g - 1)
    pooled se = pooled sd / sqrt( n_eff )                              n_eff = harmonic mean of the n_g (default), or arithmetic

A group with one value has no spread and is dropped and counted (`n_groups_dropped`, `groups_dropped`). The pooled value is the square root of
the degree-of-freedom-weighted mean variance; it is what one-way ANOVA calls the residual mean square. For duplicates (`n_g = 2`) it is the
Dahlberg method error, `sqrt(sum d^2 / 2n)`.

```toml
[[relationships]]
id = "PD1"
x = "runs.ddg"
y = "runs.ddg"                  # a one-column method: x = y
kind = "value"
method = "pooled_sd"            # or "pooled_se"
group = "ligand"                # the column within which the replicates sit
resampling_unit = "ligand"      # must equal `group` (or be omitted): a group is one independent draw, its rows are not
protocol = "timestep_fs"        # optional: a column that must hold one value for the groups to be poolable
interval = "chi2"               # optional: "chi2" or "bootstrap"; the default depends on the number of groups (below)
replicate_mean = "harmonic"     # pooled_se only, optional: "harmonic" (default) or "arithmetic"
pool_check = "PD1_check"        # optional: the id of a pool_check claim whose verdict is shown beside this one
claimed_range = [0.0, 0.3]
```

A value claim carries no p value and no R squared.

## Which interval decides

Two 95% intervals are always written on the row: the exact chi-square interval (`ci_chi2_lo`, `ci_chi2_hi`) and the percentile bootstrap over
the groups (`ci_bootstrap_lo`, `ci_bootstrap_hi`; each group is collapsed to one pair, its degrees of freedom and sum of squares, before
resampling, so a group drawn twice counts twice with its own spread). `ci_lo` and `ci_hi` are the one that decides the verdict, `interval`
names it and `interval_source` says whether the claim declared it or the default chose it. A third, the chi-square interval on the Satterthwaite
effective degrees of freedom (`ci_satterthwaite_lo`, `ci_satterthwaite_hi`), is reported and never decides.

**Default.** `chi2` below 35 groups, `bootstrap` at 35 or more. The cut-off is a number of groups and not of degrees of freedom because the
bootstrap resamples groups, so the number of groups is its sample size. **The value 35 is our choice**; its basis is Hesterberg's finding that
the percentile bootstrap interval for a mean is too narrow in small samples and that the t interval is more accurate for n up to 34 (an
exponential population), carried over to a pooled sd resampled over groups. A declared `interval` overrides the default; a value other than
`chi2` or `bootstrap` makes the claim `unverifiable`.

**Chi-square factors** (interval = estimate times factor), from `chi2_ppf`, checked against tables in the tests:

| df | example | lower | upper |
|---|---|---|---|
| 2 | one ligand, 3 runs | 0.52 | 6.29 |
| 4 | 2 ligands, 3 runs | 0.60 | 2.87 |
| 16 | 8 ligands, 3 runs | 0.745 | 1.522 |
| 40 | 20 ligands, 3 runs | 0.82 | 1.28 |

**Coverage, measured** (`test_interval_coverage_with_normal_and_heavy_tailed_replicates_is_recorded`, `pytest -s`): the fraction of seeded
synthetic data sets whose interval contained the true pooled sd 1.7. Heavy-tailed replicates are Student t on 5 degrees of freedom scaled to sd 1.7.
The binomial standard error is 0.02 at 300 draws and 0.03 at 150.

| design | replicates | exact chi-square | bootstrap |
|---|---|---|---|
| 8 groups of 3 runs (draws 300) | normal | 0.953 | 0.857 |
| 40 groups of 3 runs (draws 150) | normal | 0.967 | 0.953 |
| 8 groups of 3 runs (draws 300) | t, 5 df | 0.867 | 0.760 |
| 40 groups of 3 runs (draws 150) | t, 5 df | 0.860 | 0.900 |

Over 16,000 draws of 8 groups of 3 normal runs the chi-square interval covered 0.952 (asserted within 3 standard errors; a first single-seed
run of 2000 draws gave 0.966, a chance excursion recorded in the test). Reading: the bootstrap is too narrow at 8 groups whatever the noise
(0.76 to 0.86); the chi-square interval is right for normal replicates and too narrow for heavy-tailed ones (0.86 to 0.87) at any group count;
at 40 groups the bootstrap is the better of the two under heavy tails (0.90 against 0.86) and a little the worse under normal ones (0.95 against
0.97). Those last differences are about one standard error. The data support the direction of the default and do not fix the value 35.

**What the literature says.**

- Hesterberg 2015 [verified from the paper]: the bootstrap percentile interval is too narrow by about `sqrt((n-1)/n)` for a mean, and in
  two-sample or stratified settings the narrowness depends on the individual sample sizes; "The t interval is more accurate than the percentile
  interval for n <= 34, for exponential populations"; "n >= 30 isn't enough".
- The exact chi-square interval for a standard deviation: NIST 1.3.5.8 [verified from the page]. SAS documentation for its standard-deviation
  interval states that the values are independent and normal [search summary of the page]. That the interval is sensitive to non-normality is
  what the t5 rows above show; I did not find the statement in a source I could read (Hahn and Meeker, Statistical Intervals, section 4.3 lists
  the interval; the book is [record only]).
- Unequal variances. The pooled variance is a weighted sum of group variances, `sum (df_g / df) s_g^2`. Graybill and Wang 1980 gave the modified
  large-sample interval for such sums; SAS says "The method of MLS confidence limits was first introduced by Graybill and Wang" and builds it from
  F percentiles [verified from the SAS page; the 1980 paper is record only]. It is not implemented. The row reports the Satterthwaite
  effective degrees of freedom `df_satterthwaite` and the interval on it, which is wider than the nominal chi-square interval when the group
  variances differ; its weights use the sample variances, which at 2 or 3 runs are themselves poor.
- What the free-energy guides ask. Mey et al. 2020 ask for two or three runs and say "the sample standard deviation will give a crude estimate of
  the reliability of the estimates" (Sec. 4.2), more than three in the checklist [verified from the paper]. Grossfield et al. 2019 note that the
  experimental standard deviation is itself a random quantity (footnote 11) and take the 95% coverage factor for few samples from Student's t
  (Sec. 7.5) [verified from the paper]. Neither gives an interval for a replicate spread.
- ISO 5725-2 treats repeatability limits and Cochran's variance screen [abstract only]; Efron and Tibshirani 1993 is the bootstrap text [record only].

## Effective number of replicates for pooled_se

`replicate_mean = "harmonic"` (default): `n_eff = k / sum(1/n_g)`. Each group's mean has variance `sigma^2 / n_g`; the average of those over the
`k` groups, each counting once, is `sigma^2 / n_harmonic` exactly, so the harmonic mean is the n that reproduces the average standard-error
variance (a derivation, tested). It is never larger than the arithmetic mean, so its se is never the smaller. `replicate_mean = "arithmetic"`
gives `n_eff = mean(n_g)`, the alternative; the two agree when the groups are balanced. The row carries `replicate_mean`,
`effective_replicates`, and both `mean_replicates` and `harmonic_replicates`.

Sources: Laurencelle 2017 uses `n_h = k / (1/n_1 + ... + 1/n_k)` for the unweighted analysis of unbalanced designs where each group counts
equally [abstract and the formula read; the article is in French]. Howell's page on unequal sample sizes substitutes the harmonic mean "if the
sample sizes are nearly equal" and "I do not recommend this if you have heterogeneous variances" [verified from the page]. The coefficient
`n_0 = (N - sum n_g^2 / N) / (a - 1)` of unbalanced one-way variance components is the multiplier of the between-group variance in the expected mean
square, not an average standard error; I could not read a source for its formula (Searle, Casella and McCulloch, Variance Components, is [record
only]; Lessells and Boag 1987 returned a web page and not the paper, [not read]), so it is not used. Bland's handout [verified from the document]
gives no unbalanced treatment.

## Non-finite values

A value in the column that is NaN, infinite, blank or not a number makes the claim `unverifiable`, and the reason gives the count and the groups
(`L00 (1), L02 (2)`). The values are not dropped: dropping a failed run would silently shrink the group it came from, and nothing can be done with
the failed cell as a number. The row carries `n_nonfinite` and `n_not_a_number`. A row with an empty group label is still dropped and counted
(`dropped_no_group`): the label is bookkeeping, not a measurement, and an unlabelled run cannot be pooled into any group. That is the only value
that is dropped.

**Context in this lab.** A non-finite value is a failed cell with a reason, never a number. Three places in the repository show why. In
`projects/z-ml/experiments/cfm_z2_ligands/context.toml` item D8 refuses a replicate's BAR on any non-finite work (non-inverted frames give NaN
works; forward maps are non-finite on five ligands), records the NaN as a failed cell in `data/failed_cells.csv`, "never counted as 0, never as
False", and gives a ligand with fewer than 4 finite replica BARs a blank. In `projects/z-ml/experiments/flow_rungs/LOG.md` non-finite works
"became an estimate of 0 (fes has no refusal)", so the flow rows of R1 and R4 and the random controls built on them are marked INVALID, not
measurements. And in `projects/z-ml/ledger/errors_2026-10-13.md` MD runs at 4 fs gave NaN coordinates (fs4_control, z5_multi, a
mace_swap_panel ladder), and one map training diverged with 9 of 32 batches non-finite: those cells are failures to be recorded with their
reason, not values to be averaged.

## What pooling assumes

1. **One variance for every group.** The pooled value estimates a single sigma. If the groups have different sigmas it estimates the root mean
   square of them, which describes no group. Bland states the same assumption for the within-subject sd: error "the same for everybody" and
   independent of the magnitude measured, to be checked before anything else [Bland, handout].
2. **The same measuring procedure for every group.** Repeatability in the metrological sense is the spread of results from one method, one
   laboratory, one operator and one instrument in a short interval [ISO 5725-2, via a secondary summary]; NIST combines gauge repeatabilities
   "if the gauges are all at the same level of precision" [NIST 2.4.4.1]. For free energies this is one potential, one time step, one
   integrator, one protocol.
3. **Exchangeable, independent replicates.** Runs of a group are independent draws of the same procedure (different seeds, preferably different
   starting coordinates [Mey 2020, Sec. 4.2]).
4. **For the chi-square interval only: normal replicates.** The bootstrap interval does not need this.

## When pooling is justified, and when it is not

Justified: many groups, each with only a few replicates, and nothing to say the groups differ in noise. A repeatability sd from a few
repetitions "is not a reliable estimate of the precision of the gauge" [NIST 2.4.4.1]; a per-group sd from 3 runs spans 0.52 to 6.3 times the
true value, and the pooled value over 8 groups of 3 spans 0.745 to 1.52. This is the reason for pooling and the whole of it.

Not justified: groups from different protocols; an sd that grows with the group mean or with system size (Bland asks for this plot first; the
row carries `group_mean` and `group_sd` for it); a few groups far noisier than the rest; a quantity where the spread of one group is itself the
result under study (a per-ligand claim about one ligand's reproducibility).

**Declare the protocol.** When the time step, the potential, the force field, the charge treatment or any other run setting differs between groups,
put it in a column and declare it as `protocol = "<column>"`, then read `pool_justified` (or a `pool_check` claim). If it fails, pool within
each setting separately: one claim per setting, each on a table holding only that setting's groups (example (c) below). No statistic detects a
mixture of settings when the sampled sds happen to be alike, so the declaration is the only protection. Heterogeneity between runs of one group
larger than the within-run error is a separate finding that Grossfield et al. call "dark uncertainty" (Sec. 7.7); pooling hides it.

## The trigger rule and the pool_check claim

`pooling_trigger` in `relate.py` evaluates three criteria, each `pass`, `fail` or `not_computed`, with its value, its threshold and its source on
the row (`pool_criteria`):

| criterion | passes when | source of the rule and of the threshold |
|---|---|---|
| `fmax` | (largest group sd / smallest)^2 is at most Hartley's F-max bound at 5% for k groups and the mean degrees of freedom (rounded, at least 1); a smallest sd of 0 fails | Hartley 1950 for the statistic and the 5% level [record only]; the bound is computed from its definition in `hartley_fmax_critical` and reproduces `F(0.975; df, df)` for k = 2 and 8.363 for k = 3, df = 6 (a secondary table). Mean df for unequal groups: our choice, no source. |
| `bf` | the Brown-Forsythe p is above 0.10; computed only with at least 3 groups of 2 or more values and non-zero within-group spread of the deviations | Brown and Forsythe 1974 [record only]; NIST 1.3.5.10 recommends the median form [verified from the page]. The threshold 0.10 is our choice, no source. |
| `protocol` | the declared `protocol` column holds one value across the groups used | our choice, no source; the reasons are assumption 2. Not computed without a `protocol` column. |

**On every pooled row**, `pool_justified` is true when no computed criterion failed and `pool_unchecked` lists the criteria that could not be
computed. It is reported, never enforced, and does not change the verdict of a pooled_sd or pooled_se claim, which is decided by its interval.

**As a claim.** `method = "pool_check"` makes "pooling these groups is justified" a claim `relate` adjudicates, with the same `x = y`, `group`,
`resampling_unit` (or none) and `protocol` keys and `claimed = 1`. It is deterministic: nothing is resampled and the row has no interval.

| verdict | when | estimate |
|---|---|---|
| `refuted` | a computed criterion failed; the `why` text names each with its number and threshold | 0 |
| `agrees` | no computed criterion failed and a criterion that can confirm equal variances passed: Brown-Forsythe, or F-max on at least 2 degrees of freedom per group | 1 |
| `unverifiable` | no criterion failed but none that can confirm was computed (at 2 runs per group: the claim can be refuted, not confirmed); or nothing could be computed | 1, or none |

F-max on fewer than 2 degrees of freedom is computed and shown but cannot confirm: at 1 degree of freedom the 5% bound for 8 groups is an sd ratio
of 217. The 2-df floor is our choice, no source. `claimed` other than 1, or a `claimed_range`, makes the claim `unverifiable`: a claim that pooling is
not justified is not a pool_check claim.

**Linking.** A pooled_sd or pooled_se claim may name a pool_check claim with `pool_check = "<claim id>"`. The pooled row then carries
`pool_check`, `pool_check_verdict` and `pool_check_same_data`, and its `why` text ends with the check's verdict and reason. The pooled verdict
is still decided by its interval: the check is shown beside it and does not change it. A link to a missing claim, or to a claim that is not a
pool_check, is said in the `why` text. The check should read the same column, group and protocol as the estimate; when it does not,
`pool_check_same_data` is false and the text says so.

**Brown-Forsythe at 2 or 3 replicates.** With 2 values per group the two absolute deviations from the median are equal, and with 3 one is always
0. The deviations then carry little within-group spread and the test is conservative; with every group at exactly 2 values the statistic is
undefined (`not_computed`, never a p). Hines and Hines propose removing these structural zeros [record only; described by secondary sources];
that is not implemented. The test for "no spread" is relative to the size of the deviations, because in floating point the two deviations of a pair
differ by rounding (an earlier fixed threshold returned a p of 1e-114 on groups of 2; fixed and tested).

## Worked example (synthetic numbers)

Six ligands, three runs each, differences of a few tenths of a kcal/mol (`ddg`, kcal/mol): L1 10.2, 10.9, 10.5; L2 -3.1, -2.6, -2.9;
L3 4.0, 4.8, 4.3; L4 7.7, 7.9, 8.4; L5 1.2, 0.6, 1.0; L6 -6.0, -5.5, -6.2. Claim: the run-to-run sd is within [0, 1] kcal/mol.

    group sds        0.351, 0.252, 0.404, 0.361, 0.306, 0.361      (ratio of largest to smallest 1.61)
    pooled sd        0.342 kcal/mol on 12 df
    chi-square CI    [0.246, 0.565]  (decides: 6 groups is below 35)
    bootstrap CI     [0.306, 0.376]
    Brown-Forsythe   p = 0.995 (3 replicates per group: almost no power)
    F-max            variance ratio 2.58 against a bound of 266 (k = 6, df = 2)
    pool_justified   true; protocol not computed (no column declared)
    verdict          agrees: the chi-square interval lies inside [0, 1]

With a claimed range of [0, 0.5] the chi-square interval straddles the bound (unverifiable) and the bootstrap interval would agree
(`interval = "bootstrap"`): the case where the choice of interval decides.

## Three illustrative cases (ILLUSTRATIVE, SYNTHETIC)

None of these numbers is a measurement. Each is a seeded draw by `draw` in `tests/test_pooled_sd.py` (normal replicates, one mean per ligand), and a
test reproduces every number below. Sds are in kcal/mol; `ts` is the time step.

**What each criterion protects against.** `fmax` protects against pooling groups of which one or a few are far noisier than the rest, because it
looks at the extreme sample sds; at 2 or 3 runs the sample sds scatter widely by chance, so its bound is wide. `bf` protects against a spread
that differs systematically across many groups, which the extremes can miss, and does not need normal replicates, but at 2 or 3 runs it has
almost no power and at 2 runs it cannot be computed. `protocol` protects against pooling runs made under different settings, whose spreads are
different quantities; it is a declaration and not a test, and in example (c) both numeric criteria pass while it fails.

**(a) 8 ligands, 3 runs each, true sds 0.10 to 0.30, one time step (2 fs).** All three criteria are computed and pass.

    pooled sd        0.145 on 16 df;  chi-square [0.108, 0.221] (decides);  bootstrap [0.082, 0.201]
    group sds        0.043 to 0.260 (ratio 6.0, although the true sds differ threefold: 3 runs scatter widely)
    fmax             variance ratio 36.4 against a bound of 403.1       pass
    bf               p = 0.698                                           pass
    protocol         one value (2)                                       pass
    pool_justified   true;  pool_check: agrees (confirmed by fmax, bf)

**(b) 8 ligands, 2 runs each, the same sds, one time step.** Brown-Forsythe cannot be computed: both deviations in each pair are equal.

    pooled sd        0.126 on 8 df;  chi-square [0.085, 0.242] (decides);  bootstrap [0.081, 0.167]
    fmax             variance ratio 54.2 against a bound of 47,244 (1 df)   pass, cannot confirm
    bf               not computed
    protocol         one value (2)                                       pass
    pool_justified   true as built (nothing computed failed), pool_unchecked = ["bf"];  pool_check: unverifiable

Here the protocol is the only criterion that carries weight, and it is a declaration. A reader should take "justified" to mean "nothing found
against pooling", not "equal variance shown": read `group_sd` against `group_mean`, confirm the settings are the same by other means, and add a
third run to some ligands, which makes Brown-Forsythe computable. At 2 runs per group the claim can be refuted (by a mixed protocol or an sd
ratio beyond the 1-df bound) and cannot be confirmed.

**(c) 4 ligands at 4 fs and 4 at 2 fs, 3 runs each (true sds 0.10 to 0.20 at 2 fs, 0.40 to 0.60 at 4 fs).** Pooled together:

    pooled sd        0.465 on 16 df;  chi-square [0.346, 0.707]
    fmax             variance ratio 303 against a bound of 403           pass
    bf               p = 0.605                                           pass
    protocol         two values (2, 4)                                   FAIL
    pool_justified   false, pool_failed = ["protocol"];  pool_check: refuted

The pooled 0.465 describes neither setting. Pooled within each time step, with one claim per setting on a table holding only that setting's
ligands, each is justified:

    2 fs    pooled sd 0.110 on 8 df, chi-square [0.074, 0.211]
    4 fs    pooled sd 0.648 on 8 df, chi-square [0.438, 1.241]

## References

Status: **[verified from the paper]** the text was read; **[verified from the page]** a web page was read; **[abstract only]**;
**[record only]** the bibliographic record was confirmed in Crossref (title, authors, year, volume, pages, DOI) and the text was not read;
**[not found]**; **[not read]**. Nothing below was cited from memory.

Intervals and the bootstrap

- Hesterberg T. What teachers should know about the bootstrap: resampling in the undergraduate statistics curriculum. Am Stat 69(4):371-386, 2015.
  doi 10.1080/00031305.2015.1089789; arXiv 1411.5279. [verified from the paper]
- Graybill FA, Wang CM. Confidence intervals on nonnegative linear combinations of variances. J Am Stat Assoc 75(372):869-873, 1980.
  doi 10.1080/01621459.1980.10477565. [record only] SAS/STAT VARCOMP details (modified large-sample limits). [verified from the page]
  https://go.documentation.sas.com/api/docsets/statug/v_023/content/statug_varcomp_details11.htm
- Hahn GJ, Meeker WQ. Statistical Intervals: A Guide for Practitioners. Wiley, 1991. doi 10.1002/9780470316771. [record only]
- Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman and Hall, 1993. doi 10.1007/978-1-4899-4541-9. [record only]
- Satterthwaite FE. An approximate distribution of estimates of variance components. Biometrics Bulletin 2(6):110, 1946 (the record gives the first
  page only). doi 10.2307/3002019. [record only] The row's `df_satterthwaite` is the standard effective degrees of freedom; the paper was not read.

Unequal group sizes

- Laurencelle L. The unweighted "harmonic mean" solution for unbalanced anova designs: a detailed argument. The Quantitative Methods for
  Psychology 13(1):95-104, 2017. doi 10.20982/tqmp.13.1.p095. [abstract and the n_h formula read; the article is in French]
- Howell DC. Unequal sample sizes and multiple comparisons (web page, University of Vermont, undated).
  https://www.uvm.edu/~statdhtx/StatPages/MultipleComparisons/unequal_ns_and_mult_comp.html. [verified from the page]
- Searle SR, Casella G, McCulloch CE. Variance Components. Wiley, 1992 (2006 reprint). [record only, from a catalogue listing; not read]
- Lessells CM, Boag PT. Unrepeatable repeatabilities: a common mistake. Auk 104(1):116-121, 1987. [record only, from a search result; the PDF fetch
  returned a web page, not the paper]

Classical statistics

- Brown MB, Forsythe AB. Robust tests for the equality of variances. J Am Stat Assoc 69(346):364-367, 1974. doi 10.1080/01621459.1974.10482955. [record only]
- Levene H. Robust tests for equality of variances. In: Olkin I, et al., eds. Contributions to Probability and Statistics: Essays in Honor of
  Harold Hotelling. Stanford University Press; 1960:278-292. [record only, from library listings seen through a web search; not in Crossref]
- Bartlett MS. Properties of sufficiency and statistical tests. Proc R Soc Lond A 160:268-282, 1937. doi 10.1098/rspa.1937.0109. [record only]
- Hartley HO. The maximum F-ratio as a short-cut test for heterogeneity of variance. Biometrika 37(3-4):308-312, 1950.
  doi 10.1093/biomet/37.3-4.308. [record only]
- Cochran WG. The distribution of the largest of a set of estimated variances as a fraction of their total. Ann Eugen 11:47-52, 1941.
  doi 10.1111/j.1469-1809.1941.tb02271.x. [record only] Not implemented; it is the screen for outlying laboratory variances in ISO 5725-2 (section
  8.3.4 in a table of contents seen through a catalogue page).
- Hines WGS, Hines RJO. Increased power with modified forms of the Levene (Med) test for heterogeneity of variance. Biometrics 56(2):451-454, 2000.
  doi 10.1111/j.0006-341x.2000.00451.x. [record only]
- NIST/SEMATECH e-Handbook of Statistical Methods. [verified from the page]: section 1.3.5.8 (chi-square interval for a standard deviation; the
  page states no normality caveat), 1.3.5.10 (Levene test; median form recommended for robustness; less sensitive than Bartlett to
  non-normality), 2.4.4.1 (repeatability sd pooled over days, runs and check standards with `LK(J-1)` degrees of freedom; gauges at the same
  precision may be combined), 2.4.4.4.4 (pooled over 5 wafers, 300 df).
  https://www.itl.nist.gov/div898/handbook/eda/section3/eda358.htm, .../eda35a.htm, https://itl.nist.gov/div898/handbook/mpc/section4/mpc441.htm, .../mpc444.htm.
- A textbook treatment of pooled variance and its degrees of freedom (Snedecor and Cochran, or similar): not checked.

Measurement precision

- Bland JM, Altman DG. Statistics notes: Measurement error. BMJ 312:1654, 1996 (doi 10.1136/bmj.312.7047.1654); Measurement error and
  correlation coefficients. BMJ 313:41-42, 1996 (doi 10.1136/bmj.313.7048.41); Measurement error proportional to the mean. BMJ 313:106, 1996.
  [record only; the BMJ pages returned HTTP 403. A search summary of 313:41 says the authors prefer the within-subject standard deviation as the index.]
- Bland JM. How do I analyse observer variation studies? Web handout, University of York, undated.
  https://www-users.york.ac.uk/~mb55/meas/observer.pdf. [verified from the document] The within-subject sd is the square root of the
  residual mean square of one-way ANOVA with subjects as groups; error assumed the same for everyone; "estimate the within-subject variability
  as if it were the same for all subjects"; first check that it is independent of the mean. No unbalanced treatment.
- ISO 5725-2:2019, Accuracy (trueness and precision) of measurement methods and results, Part 2. [abstract only, through catalogue pages and
  vendor summaries; the repeatability formula and its unequal-replicate form were not read]
- Dahlberg G. Statistical Methods for Medical and Biological Students. Allen and Unwin, 1940. [not found; seen only cited in a paper on the
  error of the method (PMC4296599)] The duplicate-measurement identity above is algebra, tested, and does not rely on it.
- A Cox-type pooled method error from duplicates: [not found].

Free-energy calculations

- Mey ASJS, Allen BK, Bruce Macdonald HE, Chodera JD, et al. Best practices for alchemical free energy calculations [Article v1.0]. Living J
  Comput Mol Sci 2(1):18378, 2020 (arXiv 2008.03067v3). [verified from the paper] Sec. 4.2: "A good practice is to perform two or three runs of the same perturbation to assess precision with a given protocol. The sample standard deviation will give a crude estimate of the reliability of the estimates"; a stricter test uses different input coordinates for each repeat. Sec. 8.4: large variance between random seeds "should be flagged as issues with convergence". Checklist: "Make sure you have run enough replicates to ensure statistical reliability (>3)". The text and the checklist differ.
- Grossfield A, Patrone PN, Roe DR, Schultz AJ, Siderius DW, Zuckerman DM. Best practices for quantification of uncertainty and sampling quality in
  molecular simulations [Article v1.0]. Living J Comput Mol Sci 1(1):5067, 2019 (RULES.md writes 2018; Crossref gives 2019).
  doi 10.33011/livecoms.1.1.5067. [verified from the paper] Footnote 11: the experimental standard deviation is itself a random quantity; Sec. 7.5: the 95% coverage factor for few samples comes from Student's t; Sec. 7.7: dark uncertainty. Its "pooling analysis" (Sec. 7.4) concerns the validity of linear approximations, not variance pooling, and is not cited for pooling.
- Bhati AP, Wan S, Hu Y, Sherborne B, Coveney PV. Uncertainty quantification in alchemical free energy methods. J Chem Theory Comput
  14(6):2867-2880, 2018. doi 10.1021/acs.jctc.7b01143. [verified from the paper, first 100,000 of 140,391 characters] At least 5 replicas for TIES and 25 for ESMACS; per-system ranges of 0.8 to 1.3 kcal/mol; no variance pooling across ligands and no test of equal variance.
- Baumann HM, Horton JT, Henry MM, et al. Large-scale collaborative assessment of binding free energy calculations for drug discovery using OpenFE.
  J Chem Inf Model 66(11):6429-6452, 2026. doi 10.1021/acs.jcim.6c00089. [verified from the paper, first 100,000 of 251,613 characters] Each edge run in triplicate; the sd across repeats gives each edge's uncertainty; 83.5% of public and 71.7% of private edges have a repeat range under 1 kcal/mol (so spread differs between edges); no pooling across edges; three repeats are "likely insufficient" to relate the two error estimators.
- Wan S, Bhati AP, Zasada SJ, Coveney PV. Rapid, accurate, precise and reproducible ligand-protein binding free energy prediction. Interface Focus
  2020. doi 10.1098/rsfs.2020.0007. [abstract only]
- Wang L, Wu Y, Deng Y, et al. Accurate and reliable prediction of relative ligand binding potency in prospective drug discovery by way of a modern
  free-energy calculation protocol and force field. J Am Chem Soc 137(7):2695-2703, 2015. [verified from the paper by text search only: it reports cycle-closure and sampling error estimates and no replicate-sd analysis]
- A ChemRxiv preprint reporting sds over 30 replicas of 4 ns FEP+ calculations per ligand pair (chemrxiv 61bb2c3502c21441b427b404). [search snippet
  only; the page returned HTTP 403; authors not confirmed]
- Osato, Dabbous and Mobley 2026 on seed-only spread: [not found]
- Gapsys or Schindler on repeatability, and any paper that pools replicate spread across ligands in relative binding free energy: [not found]
  (one targeted search for the second; the first was not searched separately).
