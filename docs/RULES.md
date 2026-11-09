# Rules

Universal rules for claim-driven computational work with claimcheck and Snakemake:
what a claim is, how it is tested, how results are reported. Nothing here depends
on one lab's machines, paths or people. What is particular to this lab (layout,
cluster, environments, goals, house style, the subagent brief) is in
`shared/LAB.md`. Where a rule has a date or a measurement behind it, the incident
is kept in one line, because that is what makes it a rule and not a preference.

**Contents.** [Order of work](#order-of-work) · [Claims](#claims) · [Snakemake](#snakemake) · [Experimental design](#experimental-design) · [Reading files and MD traps](#reading-files-and-md-traps) · [Reporting](#reporting) · [Failing soft](#failing-soft) · [Agents](#agents) · [Best practice standards](#best-practice-standards)

## Order of work

Every experiment, in this order. Do not reorder it.

1. **Best practices.** `python check_bestpractices.py --severity must`, before writing code and after. A new `must` failure is fixed before anything else.
2. **Look at what is already running** (host memory, GPU, and this project's own processes) before starting anything. The lab procedure is in `shared/LAB.md`.
3. **Write the lab-book entry in the reply, before the tool call.** Six lines: the question and what would answer it either way; the exact files, rows and independent units; the method and any choice that could have gone differently; the controls, named; the units (what one number means and what is averaged to make it); the cost. Define every term the output will use. The experiment's docstring repeats it. An entry written afterwards is written knowing the answer, which is how a control gets omitted.
4. **Register the claim** in the project's `context.toml`.
5. **Write the rule and the tests**, then `snakemake -n`, then run.
6. **Draw the scatter figure** in the same rule as the table.
7. **Adjudicate and publish** with the shared claimcheck rules.
8. **Report** the verdict whether or not the prediction held. A falsified prediction is a result, and it gets the same figure a confirmed one would.

A single experiment defines nothing on its own, and a script that produces a number is not an experiment. An experiment tests a claim, registered before its data exist, recomputed against them afterwards, and published with the verdict attached.

## Claims

The whole declaration is one `context.toml` per experiment, with four kinds of block: `[project]`, `[[datasets]]`, `[[variables]]`, `[[relationships]]`.

```toml
[[datasets]]
id = "pairs"
origin = "computed"          # or "manual"; nothing else is accepted
level = "reference"          # toy, test_case, reference, real
files = ["data/pairs.csv"]

[[variables]]
dataset = "pairs"
name = "ddg_on"
unit = "kcal/mol"            # an undeclared unit makes every claim touching it unverifiable

[[relationships]]
id = "G1_gcmc_does_not_reorder"
x = "pairs.ddg_on"
y = "pairs.ddg_off"
kind = "correlation"
method = "spearman"          # or pearson, or kendall (tau-b)
claimed = 1.0                # the prediction, written before the data
resampling_unit = "edge"     # what the interval resamples, never rows
evidence.source = "Which way you expect it to go, and what a refutation would mean."
```

**Verification levels.** Every dataset and claim declares one; a claim inherits its dataset's unless overridden.

| level | meaning |
|---|---|
| `toy` | known numerical model or synthetic data with exact ground truth |
| `test_case` | a simple version with known outcome, testing the mechanics |
| `reference` | retrospective benchmark with published experimental ground truth |
| `real` | prospective campaign, outcomes not known in advance |

**Controls.** Every claim has two baselines and both are in the same table as the method, never in prose beneath.
- A **standard** (heavy-atom count, ECFP4, the crystal pose, greedy acquisition) says whether the method is useful. Restrict it to the same rows the method covers.
- A **random control** (a permutation, a constant, a dimension-matched Gaussian, structurally matched in dimension, sparsity and tie structure) says whether the design can resolve anything. It is the one skipped most and it has prevented a wrong conclusion.
- A **known-answer claim** (the same quantity two ways, claimed 1.0, or a degenerate control that must come back `unverifiable`). Without it a broken join and an honest null look the same. A rank correlation claimed at 1.0 on a stochastic estimate needs the spread across units to dwarf the noise, or it reads refuted when nothing is wrong. Linear-path TI is never a known answer; use MBAR or the native path.
- Where repeats exist, prefer an empirical null: the same quantity from arms differing only by seed or force field.
- **Carry heavy-atom count as an arm in any ranking comparison** and report the partial with it held fixed.
- Check whether a comparison is paired before quoting its spread. Arms sharing a warm-up per seed are paired, and unpaired standard deviations then measure warm-up luck.

**Claim discipline.**
- A `claimed` value is never edited after its data exist. A badly designed claim stays, with an `evidence.note` saying why; a replacement is dated and labelled as one. Claims not yet backed by a rule live in `context.pending.toml` and move when the rule does.
- Verdicts are `agrees`, `refuted` or `unverifiable`, decided by whether `claimed` lies inside the bootstrap interval over `resampling_unit`. A claim inside a wide interval is `unverifiable`, not confirmed.
- Every prediction, known answer and control is a claim adjudicated by `relate`; none is judged only in an analysis script. A quantity that is a relationship between two columns is a correlation claim; one that is made one in the rule that writes the table stays so (a partial correlation becomes a column of within-group rank residuals, a paired difference two columns). A value that is not a relationship is a `kind = "value"` claim (2026-10-01): `method` is `mean`, `sd`, `fraction` (the share of rows whose column is true, for a count that passes a rule) or `var_share` (var(y) / var(x), the share of a total's variance one part carries); `x = y` for one-column methods; `claimed = v`, or `claimed_range = [lo, hi]` with `-inf` or `inf` for an open side. A range agrees when the whole bootstrap interval lies inside it, is refuted when the interval lies outside, and is unverifiable when the interval straddles a bound. Value claims carry no p and no R squared. The exception is `method = "sign_test"` (2026-10-08), for "the correction lowers the error on most targets": `x = y` names a boolean or 0/1 column (true is the event) or a signed column (positive is the event, an exact zero a tie, dropped and counted; declare `event = "positive"` where the values alone would read as 0/1), each `resampling_unit` contributes one value (the majority sign over its rows, an even split dropped as a unit tie), the estimate is the share of units that are the event with the same bootstrap interval, and the row also carries `p_value`, the exact two-sided binomial p against 0.5, with `n_success`, `n_units` and `n_ties_dropped`; the verdict still comes from the interval against `claimed` or `claimed_range`, and the two answer different questions: 9 of 10 units gives p = 0.021 against 0.5 and an interval of [0.7, 1.0], so a `claimed_range` of [0.75, 1.0] is unverifiable on 10 units and [0.5, 1.0] agrees. `method = "binomial_count"` (2026-11-09, branch `binomial-count`) is "k of n units pass a rule" with the exact Clopper-Pearson interval in place of the bootstrap, which collapses to [1, 1] at n of n: `x = y` names a 0/1 column, one row per `resampling_unit` (several are refused), and the row carries `n_success`, `n_units`, `dropped` and the exact two-sided binomial p against 0.5; 8 of 8 gives [0.631, 1], so [0.5, 1.0] agrees only at 8 of 8 and [0.75, 1.0] never at n = 8 (`docs/binomial_count.md`). A quantity `relate` still cannot express (an effective sample size computed from raw works) is added to `relate` as an estimator with its tests, then registered; until then it waits in `context.pending.toml` as not yet adjudicated, and is never reported as held or falsified.
- Figures by default (2026-10-01): the shared `check_claims` rule also writes `figures/claims/<id>.png` and `figures/claims/index.json`, one figure per claim, as a side effect and not a declared output, so a project that defines its own `claim_figures` rule with `output: directory("figures/claims")` keeps working. Without matplotlib the figure step prints a warning and the verdicts are still written.
- `kendall` (2026-10-27), `kind = "correlation"`: Kendall's tau-b, (C - D) / sqrt((C + D + Tx) * (C + D + Ty)), with C and D the concordant and discordant pairs and Tx, Ty the pairs tied in x only and in y only, so a heavily tied column is not read as disagreement. It has the same bootstrap interval over `resampling_unit`, the same cluster permutation p, `n_rows`, `dropped` and `tie_tol` as `spearman`, and a constant column or fewer than 3 rows makes the estimate undefined and the claim `unverifiable`. `r_squared` is tau-b squared, a squared rank correlation and not a share of variance; the row says so in `r_squared_is_on = "tau-b"` and `r_squared_note`. Tau-b is not Spearman's rho on another scale: on the same data it is usually smaller in magnitude, so a `claimed` value written for one is not a prediction for the other.
- `tie_tol` (optional, absolute, `spearman` and `kendall` only): values of a column within it of one another are given the same average rank, by rounding both columns to the nearest multiple of `tie_tol` before ranking, and it is recorded in the verdict row. Use it on a known-answer claim of 1.0 whose estimates are equal in exact arithmetic (eight seeds sharing one dF), where 1e-9 rounding noise would otherwise order the ties and read as discordance. Every verdict file also carries `relate_sha256` and `context_sha256` (first 16 hex characters), naming the checker and the declaration that produced it.
- `pooled_sd` and `pooled_se` (2026-10-28), `kind = "value"`: the within-group sd of replicates pooled over groups, `sqrt(sum (n_g - 1) s_g^2 / sum (n_g - 1))`, and that divided by the square root of the effective replicate count. `x = y` names the numeric column; `group = "<column>"` names the unit within which the replicates sit (the ligand, for runs of one ligand) and must equal `resampling_unit` or leave it out, because the bootstrap resamples groups, never rows. Groups with one value are dropped and counted (`n_groups_dropped`), and the row carries `df`, `n_groups_used` and `df_satterthwaite`. Two 95% intervals are always written, the exact chi-square interval (`ci_chi2_lo`, `ci_chi2_hi`, valid for normal replicates with one common variance) and the bootstrap over groups (`ci_bootstrap_lo`, `ci_bootstrap_hi`), and the key `interval = "chi2" | "bootstrap"` chooses which one the verdict reads against `claimed` or `claimed_range` (`ci_lo`, `ci_hi`, `interval` and `interval_source` on the row). The default is `chi2` below 35 groups and `bootstrap` at 35 or more; the cut-off carries Hesterberg's finding for the mean (the percentile bootstrap is too narrow in small samples, the t interval better up to n = 34) over to groups and is our choice, and the measured bootstrap coverage was 0.86 at 8 groups of 3 runs. For `pooled_se` the replicate count is the harmonic mean of the group sizes by default, which reproduces the average variance of a group mean, and `replicate_mean = "arithmetic"` is the alternative. A non-finite, blank or non-numeric value in the column makes the claim `unverifiable`, naming the count and the groups: it is a failed cell with a reason, never a number, and it is not dropped. The row also reports whether pooling is justified (`pool_justified`, with every criterion in `pool_criteria`): largest-to-smallest group variance against Hartley's F-max bound, the Brown-Forsythe p above 0.10 (almost no power at 2 or 3 replicates), and one value of an optional `protocol` column, to be declared whenever a run setting (time step, potential, force field) differs between groups. It is reported and never enforced, and does not change the verdict. Fewer than 2 groups with 2 or more values, a column constant within every group, fewer than 3 groups for a declared bootstrap interval, a missing `group` column, and a `resampling_unit` other than `group` are `unverifiable`. Sources and thresholds in `claimcheck/docs/pooled_sd.md`.
- `pool_check` (2026-10-28), `kind = "value"`: "pooling these groups is justified" as a claim `relate` adjudicates, with the same `x = y`, `group` and `protocol` keys as `pooled_sd` and `claimed = 1`. It is deterministic (nothing is resampled, no interval): estimate 0 and `refuted` when a computed criterion fails, the `why` text naming each with its number and threshold; estimate 1 and `agrees` when none fails and a criterion that can confirm equal variances was computed (Brown-Forsythe, or F-max on at least 2 degrees of freedom per group, a floor that is our choice); otherwise `unverifiable`, which is what 2 runs per group gives: the claim can be refuted there and not confirmed. Every criterion is on the row with its value, threshold, source (or "our choice, no source") and the unchecked list. A pooled_sd or pooled_se claim may name it with `pool_check = "<claim id>"`; its row and `why` text then show that claim's verdict beside the estimate, and its own verdict stays decided by its interval (reported, not enforced). `claimed` other than 1 is `unverifiable`.
- `belief` propagates doubt from a refutation over a closed vocabulary (7 statuses, 13 predicates, 7 node kinds). It needs a `[belief]` table and refuses a project without one.
- `origin` is checked: `computed` with no rule producing the file is a number whose recipe is gone; `manual` written by a rule is a source that is not a source. `grading --origins` compares them.
- Validating is not reading. For every published format the question is whether someone can get the data out. This package's Croissant once validated and yielded zero records because `contentUrl` was resolved against the wrong folder.

## Snakemake

**Nothing that computes a number runs outside a rule.** Fitting, docking, an MD step, a bootstrap, anything written into `results/`. There is no threshold of smallness: "just checking something" is how every unrecorded artifact was made. Read-only inspection is not analysis: `ls`, `git log`, `head`, `squeue`, the test suite. A chained `nohup python` deadlocked on its own `pgrep` on 2026-08-02 and nobody noticed for two days and eighteen hours.

- `snakemake -n <target>` first, every time, and read the job list.
- **Write the rule before the first run.** Editing a script afterwards makes Snakemake delete the output, correctly. It cost forty-two minutes twice on 2026-09-06.
- **A rule the package defines is included, never copied:**

  ```python
  import sys as _sys
  _sys.path.insert(0, "<path to the claimcheck checkout>")
  from claimcheck.workflow import RULES as _CLAIMCHECK_RULES

  include: _CLAIMCHECK_RULES
  ```

  This gives `check_claims`, `croissant`, `graph`, `dcat` and `belief`, with inputs read from the project's own `context.toml`. If the declared files do not exist Snakemake refuses the DAG, which is correct. The interpreter running Snakemake usually cannot run the emitters (`rdflib`, `mlcroissant`); `claimcheck.workflow.interpreter` picks one that can, honouring `CLAIMCHECK_PY`.
- **Launch anything long detached,** because a client timeout kills a foreground run mid-rule: `setsid nohup snakemake -c<N> > logs/run.log 2>&1 < /dev/null & disown`.
- **Never `pkill -f`.** It matches the agent's own shell and killed the session twice on 2026-09-07. Read the process list, then kill explicit PIDs.
- **`--unlock` only when nothing is running,** and say so. The lock stops two schedulers each deciding what is stale.
- Never `--forceall`. Use `--keep-going`, and make submit rules idempotent so a dead node costs one task.
- Snakemake records, per output, the input digests, code, command, environment and times. That is the run record; do not build a second one.
- Do not read a package's name for its cause. When a package is at fault, read the source, name the line, and fix it in your own code without modifying the installed package, so a control arm still runs the unmodified stack. Read the error: a missing `cicc` was a layout mismatch between two directories, not a missing package.

## Experimental design

### The unit of analysis

**Never average across targets.** Affinity is comparable only within a receptor, so a mean over targets mixes ranking skill with an offset no scorer was asked to predict. Measured on the paros board: of 64 methods, 17 ran on one target and 13 on all thirteen, and ranks moved by up to 39 places between the pooled mean and a single-assay mean. The mean was reporting which targets a method ran on.

Score per target and deliver a table of targets. Compare methods only on the targets they all ran. Count independent within-target comparisons rather than averaging them, with a binomial interval. Every average states its unit, including the obvious ones.

**Cohort size decides resolution, and independent units are what count.** Median interval width measured across all projects:

| cohort | median width | verdict |
|---|---|---|
| 5,938 proteins (MISATO) | 0.055 | resolves |
| 176 Butina clusters, 493 fragments | 0.303 | resolves |
| 15 targets, bootstrap median | 0.59 to 0.86 | borderline to useless |
| 32 congeneric ligands | 0.72 to 0.77 | resolves nothing |

At 15 targets the design that works is a sign test across targets, not a bootstrap median (14 of 15 at p = 0.00098). Before running anything on 32 ligands, state what interval it can produce. Feature against affinity on 32 ligands is the weakest case.

**Many receptors, each with 5 to 10 exact ligands.** Hundreds of targets with two ligands each support no ranking; one receptor with 600 demonstrates one protein. Headline a set by median ligands per target with the distribution, not a row count. "Identical receptor" needs a stated rule (construct, mutations, protonation). Exact canonicalised SMILES. One assay type per receptor group. Preparation documented through claimcheck, or the set cannot be reused.

### The label

**Experimental dG is a measurement, not truth.** Repeat spread is the noise floor and the most valuable number in a benchmark set: no method can beat it. Report it per target in kcal/mol and say where it could not be established. Do not pool across experiments to make a target look deeper. Never pool Kd, Ki and IC50. Treat the experiment as a first-class key: publication, lab, protocol, temperature, construct, batch.

**Keep every row.** Where a compound has several measurements or poses, treat them as different molecules. No `drop_duplicates()`, no silent selection on the dependent variable. A narrow energy discard band deleted compounds for being the most strongly bound, and they were enriched in actives. Flag outliers rather than dropping them: a large finite interaction energy is a result to examine, a 1e7 is a bug to discard.

**In an active-learning loop the target is experimental hits,** and the acquisition signal is separate. A loop scored against its own computed objective measures nothing. Plot hits against cycle and mean score against cycle; report cycles to the 1st, 5th, 10th and 25th hit with intervals. State the `LabelRule` beside every count: on the 494 compounds, `PKD_HIT` at 6.0 gives 94 hits and `PKD_TOP_DECILE` gives 50, disagreeing on 44.

### Reading a result

**Evaluate at the valuable end.** More negative is more valuable, and a global R, rho or q2 is dominated by the uninteresting compounds. Report the global figure, rho restricted to the tightest-binding half and third, and top-N recovery against random expectation. Check the shape first: a wedge (spread narrowing as the descriptor rises) is described by no line, and shows as a negative correlation between absolute residual and descriptor.

**Every experiment aims at a scatter figure.** Know which two quantities are on the axes and what one point is before starting. An experiment that cannot be drawn as predicted against measured, or arm against baseline, has not decided what it compares. `matplotlib` with `Agg`, written by the same rule as the table. Plot the points, not their mean. Baseline on the same axes, y = x on a prediction scatter, error bars carrying the resampled interval, the resampling unit in the label, targets in separate panels.

**Every figure names its dataset and size on the figure:** "A71EV2A, 494 molecules, 177 Butina clusters" in the `suptitle` or caption, with rows and independent units when they differ. A figure travels away from its paragraph.

**Captions should not have too much text on them.** The data should do all the work, with an associated commentary file if needed. State the dataset, size, series legend and intervals concisely; put extended discussion or per-target interpretation into a companion commentary file.

**No experiment script is finished until it has tests,** committed beside it and run by the same check. A two-sided fixture per estimator, failing on known-bad and passing on known-good. A known answer from synthetic data (a perfect ranker recovers everything, a random one the chance rate, a tie ties). A test for every control, because the random baseline being neutral is a claim. A test that degenerate cases return `unverifiable`. Written as the script is written, since a test written knowing the answer tests the answer. `pytest tests` (minutes, not hours) passes before any full run.

**Reproduce a published result before scaling.** Against published MD, compare descriptor for descriptor, state both sides and quantify the offset. Against published RBFE or ABFE, the published FEP's own RMSE against experiment is the realistic ceiling for a descriptor method and goes beside our numbers. A benchmark that only reports agreements is not being checked. Inter-lab variability of a few tenths of a kcal/mol means headroom is smaller than a raw RMSE suggests.

**OpenFE plans FEP networks. Do not hand-roll one,** and do not compute cycle counts or edge budgets by hand. `generate_lomap_network` with `require_cycle_covering=True`, or `generate_minimal_redundant_network` with `mst_num=2`; read cycles off the returned `LigandNetwork`.

## Reading files and MD traps

**A column name is a claim, not evidence.** Find the rule that declares the file as an output, read the script, and establish the formula, units and exclusions. One score named MACE was MACE-OFF23 medium in float64, over a 6 A pocket, with unsupported-element residues excluded by the pocket cutoff, on CPU. None of that was in the header. `is_hit` was thresholded per crystallographic event while `experimental_pKD` was the group mean, so the label was not a function of the affinity beside it. If the generating script cannot be found, that is the finding.

**Discard the equilibration period before correlating any MD descriptor.** Not discarding produced a signal that reversed sign: every autocorrelation-time descriptor flipped from positive to negative while median geometry got stronger, because a decaying transient inflates the autocorrelation function. Chodera's `detect_equilibration` discarded two thirds to three quarters of all frames, so a 10 ns run yields a few ns of sampling. Report sensitivity to the truncation, benchmark against heavy-atom count with q2(LOO) rather than R2, and report both a bootstrap interval over ligands and the per-replicate spread. It is *equilibration* and *production*, never "burn-in".

**Check the protein is whole before measuring in its frame.** 118 of 1,262 fepbench trajectories have the protein split across the periodic boundary. The boxes are triclinic, so `round(delta/box)*box` looks like a fix and is not; the same ligand gave RMSF 46.64 A raw, 38.03 A orthorhombic and 0.65 A with the exact minimum image. No wrong answer raises an exception. Use `MDAnalysis.lib.distances.minimize_vectors` and a PBC-aware `distance_array`. Re-imaging the ligand does not repair a split protein.

**Four traps that produced plausible wrong numbers, not errors.** A ligand carrying hydrogens sat in a different frame from its receptor and did not crash: check pairing geometrically. A plan-inherited hit threshold labelled almost everything a hit: check the base rate a threshold implies. Mean-pooling over ligand atoms washed the pocket out: test that a complex-derived descriptor differs from its ligand-only counterpart. Sketches named coreset, BADGE and BALD did not compute what they were named after.

## Reporting

**Report three numbers and compute all of them:** a p value, an interval with its resampling unit named, and R squared. The partial given heavy-atom count, the cluster bootstrap, the sign test, the controls and the permutation importance run under the hood and live in the table.

    rho = -0.47, 95% CI [-0.52, -0.29] over 13 targets, p = 0.0018

**No bare numbers.** Every number arrives with the directory it came from and a description of what is in it. Numbers have outlived their recipes here: 1,475 of 1,534 artifacts had no recorded producer; a positive rate was published with the wrong defining rule beside it; a `_v2` file described as a second computation was byte-identical to the first.

**Always state absolute paths.** State full absolute paths from root for all files, directories, figures and scripts.

```bash
python -m claimcheck.grading projects/<programme>/experiments/<name>           # what is quotable
python -m claimcheck.grading projects/<programme>/experiments/<name> --check   # exit 1 if ungraded
python -m claimcheck.grading projects/<programme>/experiments/<name> --origins # origin vs producer
```

`grading` reads Snakemake's own store and grades each output `NONE`, `DECLARED`, `EXECUTED`, `REPRODUCED` or `VERIFIED`. Quotable starts at `EXECUTED`. The positional argument is a directory, not a name. The rule is *describe*, not *only via Snakemake*: the strict form was measured and permitted 0 of 1,534 artifacts, and a check that refuses everything stops being consulted.

**An interval that contains zero is UNRESOLVED, not absent.** Entirely below zero is a real and harmful effect; entirely above is a real and helpful one; spanning zero means this design could not tell it from zero. A negative effect that excludes zero is still an effect (exp7: -0.199 [-0.356, -0.077] is a demonstrated harm; exp8: +0.011 [-0.237, +0.379] says almost nothing). Never write "no effect" for either. `UNTESTABLE` is never a pass.

**A gap is not a defect.** An experiment not yet run and a test not yet written are work not done. They are not errors, and must not be reported as though they were. Say which kind each item is: "28 scripts have no rule yet" is a sentence about a queue; "a published checksum no longer describes the bytes on disk" is a sentence about a lie. A list with 28 pending items beside one real defect trains you to scroll past the defect. Do not invent experiments to close a coverage gap; which experiments are worth running is the project owner's call.

State what was excluded and why, with row counts in and out. A positive rate never travels without its `LabelRule`.

**Before committing,** the pre-commit check runs the invariant registry at `must` severity, the known-answer suite, mutation tests and the guard cases. It blocks on new or grown failures, not on the known baseline in `ledger/baseline.json`.

## Failing soft

The scarce resource is the running campaign. A checksum mismatch, a filename convention, a drifted environment or a package that will not install must not stall it.

| failure | what it must do |
|---|---|
| a checksum does not match | quarantine that file, record it, continue the rest |
| a key comes in two formats | canonicalise both and join, never error |
| a filename breaks a convention | glob it, take it, move on |
| an environment drifted from its lockfile | use it, record the drift, re-pin later |
| a package will not install | drop that candidate, note why, run the rest |
| one trajectory is truncated | quarantine it, analyse the rest |
| a number would enter a result ungraded | **this still blocks** |
| a claim would publish with broken provenance | **this still blocks** |

Fail soft per item and loud in the report: every skip is a row in a quarantine table with a reason. Never let one item exit a loop over many; no `set -e` across a campaign. Prefer a partial launch that is running to a complete one that is not. When blocked on a genuine ambiguity, do the parts that do not depend on the answer, state the assumption, and keep moving.

## Agents

Three kinds, differing in what each is trusted to decide. The tiers are enforced in the definitions under `.claude/agents/`: a worker cannot Write or Edit, and a scientist cannot spawn agents.

| | model | decides | never decides |
|---|---|---|---|
| **worker** | sonnet | nothing | anything; runs an existing rule, stages files, checks a job, reports numbers with their directory |
| **scientist** | opus, high effort | the method, inside one assigned question | what is worth asking, what runs next; owns one experiment from claim to figure |
| **manager** | opus, high effort | what is worth doing and in what order | the answers themselves; decomposes, assigns, reviews, keeps the queue |

- **A worker executes and does not judge.** "I could not tell which column was meant" is a correct answer. "I could not tell, so I used the first" is a failure.
- **A manager never produces the number it is judging.** One responsibility per agent, stated as what must come back and what would make the answer wrong. A returned number with no directory behind it is not a result.
- **Escalate rather than guess.** A worker blocked on method goes to a scientist; a scientist blocked on what is worth doing goes to a manager; a manager blocked on what the project owner wants asks them.
- **Fan out when the work is genuinely parallel:** different files, no ordering dependency (project owner, 2026-08-01). Agents cost tokens and their output still has to be read and checked. Give each the scientific context and sign conventions, not only the mechanics. Keep the synthesis.
- Every agent prompt includes the [brief](#brief-for-subagents).

## Best practice standards

`bestpractices/` encodes nine peer-reviewed checklists as executable checks in `check_bestpractices.py`. Point at the check that passed or failed.

| check | source | requires |
|---|---|---|
| `sampling_uncertainty` | Grossfield et al., LiveCoMS 1(1):5067 | every mean and figure of merit carries a 95% bootstrap interval over declared independent units; `n_kept` and `n_drop` recorded; decimation and tier decisions bounded by interval half-widths |
| `fair_sharing` | Willoughby & Saul, PSDI | SPDX licence (`CC0-1.0` or `CC-BY-4.0`), provenance with units and sign conventions, sha256 manifests with host and timestamp |
| `alchemical` | Mey et al., LiveCoMS 2(1):18378 | overlap first off-diagonals at least 0.03, edge sigma at most 1.0 kcal/mol, absolute ddG at most 15 kcal/mol, cycle closure over independent cycles, soft-core |
| `monte_carlo` | Hatch et al., LiveCoMS 6(1):3289 | seeded RNGs, acceptance checked against tractable limits, detailed balance, exchange and diffusion separated |
| `westpa_tutorials` | Bogetti et al., LiveCoMS 1(2):10607 | one pinned absolute path per executable, thread caps, slice limits on array jobs |
| `foundations` | Braun et al., LiveCoMS 1(1):5957 | full state point: ensemble, barostat, thermostat, integrator, timestep, cutoffs, deterministic forces |
| `aquaduct` | Mitusinska et al., LiveCoMS 2(1):21383 | solvent coordinates kept with the macromolecule; boxes imaged and derotated |
| `enhanced_sampling` | Henin et al., LiveCoMS 4(1):1583 | multi-basin transitions demonstrated, aggregate sampling time and state overlap documented |
| `protein_ff_benchmarks` | Cavender et al., LiveCoMS 6(1):3871 | experimental structures checked against resolution and refinement criteria |

Every stochastic run also records its RNG seed and software version in its run receipt. If a change introduces a `must` failure, fixing it is the work before any other step.
