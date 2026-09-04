# claimcheck

**claimcheck checks the claims people make about a dataset by recomputing them.**
You write a small text file next to your data saying what each column is, what
unit it is in, and what somebody claims about it — *"these two columns correlate
at about −0.5; that's the premise of the whole pipeline"*. `claimcheck` loads the
data, recomputes the number with a confidence interval, and answers **holds**,
**refuted**, or **cannot tell**. It then publishes the dataset and the verdicts in
formats other tools already read, and can tell you which of your files are
recorded well enough to be quoted at all.

```
HOLDS    mace.e_int_kcal ~ affinity.experimental_pKD   claimed -0.5
         within the 95% CI [-0.509, -0.388] for r = -0.450, resampling 493 compound_group(s)
REFUTED  mace.e_int_kcal ~ mace.n_lig_atoms            claimed 0.0
         OUTSIDE the 95% CI [-0.651, -0.525] for r = -0.589
```

**Contents** — [Why](#why) · [Quickstart](#quickstart) ·
[Three verdicts](#the-three-verdicts) · [Module reference](#module-reference) ·
[The vocabulary](#the-closed-vocabulary) · [`context.toml`](#the-contexttoml-format) ·
[Installation](#installation) · [Starting a project](#starting-a-new-project) ·
[Known gaps](#known-gaps)

---

## Why

Someone computes a number. Six months later nobody can tell whether to trust it —
not because the code was wrong, but because the things you need in order to judge
it were never written down: what unit it was in, what it was compared against,
what was assumed. Writing them down is not enough on its own, because a number
lifted out of a paper by a person or a language model is never nonsense; it is
confident, plausible and wrong. So `claimcheck` recomputes instead of reading, and
refuses to say "fine" when the data cannot decide.

---

## Quickstart

```bash
pip install -e ".[dev]"

cd example
snakemake -n          # always look at the job list first
snakemake -c1         # ~4 seconds
```

`example/` is a real two-dataset project: one CSV of computed interaction
energies, one CSV of measured binding affinities typed in by hand, 637 rows each,
joined on `complex_name`.

The lines each rule prints, with Snakemake's own progress messages stripped out:

```
  HOLDS   mace.e_int_kcal ~ affinity.experimental_pKD   claimed -0.5  ->  claimed -0.5 lies
          within the 95% CI [-0.509, -0.388] for r = -0.450, resampling 493 compound_group(s)
  REFUTED mace.e_int_kcal ~ mace.n_lig_atoms   claimed 0.0  ->  claimed 0.0 lies
          OUTSIDE the 95% CI [-0.651, -0.525] for r = -0.589, resampling 493 compound_group(s)
  {'agrees': 1, 'refuted': 1} over 1274 row(s) -> results/claims.json
  results/croissant.json  —  2 record set(s), 3 field(s), 2 claim(s), 1 refuted, 0 unverifiable
  results/dcat.ttl  —  39 lines of dcat
  results/graph.ttl  —  143 triples, 10 node kind(s): Artefact, Claim, Dataset, Evidence,
                          Field, FileObject, Method, Node, Ontology, Relationship
  affinity (rests on measured_affinity): nothing in doubt
  mace (rests on mace_e_int): 2 caveat(s)
  OPEN  [assumed] docked_pose --approximates--> crystal_pose
  OPEN  [assumed] vina_score --approximates--> interaction_strength
  6 claim(s) over 6 node(s), 2 delivery(ies), 2 open question(s) -> results/belief.json
```

Read that as a result. The claim that computed energy tracks measured affinity
**holds**. The control — does it track ligand *size* instead? — comes back
**refuted** at −0.589, tighter than the −0.450 the real claim gets. Ranking on
this energy would select big molecules rather than good ones. Neither number means
much without the other, which is why both are in the file.

You do not need Snakemake. Every step is a plain command:

```bash
python -m claimcheck.relate .        # writes results/claims.json
python -m claimcheck.croissant .     # writes results/croissant.json
python -m claimcheck.belief .        # writes results/belief.json
```

Snakemake is used because it already records, per output file, the sha256 of every
input, the code, the shell command, the software environment and the times.
`claimcheck` reads that record and never writes a second copy of it.

---

## The three verdicts

`relate` compares the claimed value against a bootstrap confidence interval
computed from your data.

| verdict | printed | means | belief status |
|---|---|---|---|
| `agrees` | `HOLDS` | the claim is inside the interval, and the interval is narrow enough to mean something | `measured` |
| `refuted` | `REFUTED` | the claim is outside the interval | `falsified` |
| `unverifiable` | `?` | no verdict was reached | `untested` |

**`unverifiable` is never a pass.** You get it when the interval is too wide to
discriminate, when there are fewer than three independent units to resample, when
a column has no declared unit, when two datasets need a join key and none is
given, or when the declared method is not one `claimcheck` implements. The reason
is always printed:

```
  ?       t.a ~ t.b   claimed 0.5  ->  fewer than 3 independent units to resample over g,
                                       so no interval exists
```

The alternative — comparing two point estimates against a hand-picked tolerance —
was the first design and it was wrong. On five rows it "falsified" both claims it
was given, and would have done that to a true claim just as readily.

---

## Module reference

| module | for | needs |
|---|---|---|
| [`relate`](#relate) | recompute a claimed relationship | *nothing* |
| [`datasets`](#datasets) | load data; refuse a changed hand-copied file or a fan-out join | *nothing* |
| [`units`](#units) | convert between declared units, or raise | *nothing* |
| [`labels`](#labels) | a positive rate that carries the rule defining it | *nothing* |
| [`belief`](#belief) | what rests on what, and what a refutation puts in doubt | *nothing* |
| [`vocabulary`](#vocabulary) | the 7 statuses, 13 predicates, 7 node kinds | *nothing* |
| [`jobs`](#jobs) | why a run was launched and what would have counted as success | `rdflib` to publish |
| [`grading`](#grading) | grade each file against Snakemake's own run record | *nothing* |
| [`conform`](#conform) | sweep published metadata: does it validate, and can it be read? | the validators it calls |
| [`scaffold`](#scaffold) | write a working project around one or more CSVs | *nothing* |
| [`emit`](#emit) | the output-format registry and CLI | *nothing* |
| [`namespace`](#namespace) | one definition of the project's identifier base | *nothing* |
| [`croissant`](#croissant) | how to *read* the dataset | *nothing* |
| [`graph`](#graph) | columns, claims and digests as RDF | `rdflib` |
| [`dcat`](#dcat) | how to *find* the dataset | `rdflib` |
| [`evi`](#evi) | the two propagation rules a reasoner needs | `rdflib` |
| [`psdi`](#psdi) | DCAT in PSDI's profile, checked against PSDI's own rules | `pyshacl` |
| [`rocrate`](#rocrate) | the whole directory, named and hashed | `rocrate`, `roc-validator` |

Eleven of the eighteen use nothing but the Python standard library. That is
checked by a test that runs `relate`, `croissant` and `belief` behind an import
hook refusing every non-stdlib package — not by reading the import lines, because
a lazy import inside a function would pass a reading test and fail on a cluster
node.

---

### `relate`

Recompute every claim in `context.toml` and report a verdict for each.

```bash
python -m claimcheck.relate <project-dir> [<out.json>]   # default: results/claims.json
```

Output as in the [quickstart](#quickstart). The JSON carries the estimate, the
interval, the number of independent units, the coverage of any join, and the
reason for the verdict.

**The interval is a bootstrap percentile interval, and its resampling unit is
declared, never guessed.** The *resampling unit* is the thing you shuffle when
building the interval — the thing your rows are independent *across*. Resampling
rows assumes rows are independent, and they usually are not: in the example, 637
rows come from 493 compound groups, and treating them as 637 independent
observations would report an interval well under its true width. Name the column
in `resampling_unit` and it clusters on that.

```python
from claimcheck.relate import check, pearson, spearman, bootstrap_ci, ESTIMATORS
check(".")                     # the whole report as a dict
pearson(xs, ys)                # None where r is undefined, never 0.0
bootstrap_ci(pairs, clusters)  # (lo, hi, n_units) or None
```

Also public: `AGREES`, `REFUTED`, `UNVERIFIABLE`, `STATUS`, `N_BOOT = 2000`,
`SEED = 20260815` (fixed, so two runs on one dataset give one interval), and
`MAX_INFORMATIVE_WIDTH = 1.0` (wider and the verdict is `unverifiable`; override
per claim with `max_ci_width`).

**Adding a statistic is a function and one line in `ESTIMATORS`.** Each entry
declares the `kind` it computes, so a method paired with the wrong kind is refused
rather than quietly mislabelled. An earlier version recorded `method` in every
output and never dispatched on it, so `method = "spearman"` computed Pearson.

---

### `datasets`

Load and verify each dataset. No CLI — `relate` calls it. Two things can go wrong
without anyone noticing, and both are hard errors here rather than warnings.

**1. A file somebody put there by hand changes.** `origin = "computed"` means a
rule in your workflow made it, so Snakemake holds its digest and recipe.
`origin = "manual"` means a person dropped it in, so nothing is recorded anywhere
— which is why a manual dataset must declare `source`, `retrieved` and `sha256`,
and the digest is checked on **every** load:

```
DatasetError: dataset 'affinity': data/experimental_affinity.csv is not the file
  that was checked in.
    declared sha256 0000000b64502928...
    actual   sha256 b1dc99b64502928f...
  A manually imported file that changed without anybody saying so is the one thing
  nothing else here can detect. If the new file is correct, update `sha256` in
  context.toml and say in the commit what changed.
```

`origin` has no default, because the two kinds need different evidence and a
default would pick the weaker one.

**2. A join invents rows.** Joining ten rows against ten that share a key makes a
hundred pairs, none of them measured together. Duplicate keys are **refused**, not
resolved. Coverage travels with every verdict, so a correlation over a 40%-matched
join says so:

```json
{"key": "complex_name", "left": "mace", "right": "affinity",
 "n_left": 637, "n_right": 637, "refused": null,
 "n_matched": 637, "n_left_unmatched": 0, "n_right_unmatched": 0, "coverage": 1.0}
```

Public: `load(root, spec)`, `join(left, right, key) -> (pairs, report)`,
`sha256(path)`, `DatasetError`.

---

### `units`

Convert between declared units, or raise. It is never a no-op.

```python
from claimcheck.units import convert, compatible, UNITS, UnknownUnit

convert(1.0, "kcal/mol", "kJ/mol")     # 4.184
convert(1.0, "eV", "kcal/mol")
# UnknownUnit: 'eV' is a energy and 'kcal/mol' is a molar energy; that is a change
# of quantity, not a conversion, and this module will not do it silently
```

`compatible(a, b)` answers whether a relationship between two columns could mean
anything, and returns a sentence you can print:

```python
compatible("kcal/mol", "kJ/mol")
# (True, 'kcal/mol and kJ/mol are both molar energy, factor 4.184')
compatible("kcal/mol", "nm")
# (True, 'kcal/mol is a molar energy and nm is a length — different quantities,
#         so any slope between them carries units and is not a conversion factor')
compatible(None, "nm")
# (False, "no unit declared for x — a relationship between undeclared quantities
#          cannot be compared with anyone else's")
```

`UNITS` is deliberately small — one entry per unit an experiment needed: `1`,
`g/mol`, `Da`, `kcal/mol`, `kJ/mol`, `eV`, `log10(1/M)`, `nm`, `0.1 nm`, `ps`,
`ns`. `None` means *not declared* and never *dimensionless*; write `"1"` for
dimensionless. Unit strings follow **UDUNITS-2**, the long-standing convention for
spelling physical units as text (`kcal/mol`, `nm`, `s`).

Declaring a unit is not the same as agreeing on one. Correlation is invariant
under a scale factor, so one energy in eV and another in kcal/mol — a factor of
23.06 apart — produce identical verdicts and an invisible mistake. This is the
module that catches that.

---

### `labels`

A hit rate is the most reported and least qualified number in screening work.
`LabelRule` is the definition that has to travel with it.

```python
from claimcheck.labels import LabelRule, rate

rule = LabelRule(name="top_decile", assay="biophysical_binding", endpoint="pKD",
                 statement="pKD >= 6.0 measured by SPR at 25 C")
rate(12, 100, rule)
```

```python
{'n_positive': 12, 'n_total': 100, 'rate': 0.12,
 'label_rule': 'top_decile (biophysical_binding, pKD)',
 'label_name': 'top_decile', 'label_assay': 'biophysical_binding',
 'label_endpoint': 'pKD', 'label_statement': 'pKD >= 6.0 measured by SPR at 25 C',
 'label_thresholdable': True, 'label_citation': None, 'label_note': None}
```

`rate()` returns a mapping and there is no plain-float form, because a float gets
separated from its definition by the next line of code. `comparable_to` is the
check a real incident needed — a pKD threshold published beside a dose-response
call, in one column:

```python
rule.comparable_to(LabelRule(name="hts_hit", assay="hts_dose_response",
                             endpoint="pIC50", statement="pIC50 >= 6.0"))
# (False, 'top_decile is biophysical_binding and hts_hit is hts_dose_response;
#          rates from different assays are not comparable')
```

`name`, `assay`, `endpoint` and `statement` are required; omitting one raises
`UnusableLabelRule`. Set `thresholdable=False` when the data arrived already
binarised, so nobody reports a sensitivity curve over a threshold that does not
exist.

---

### `belief`

`relate` judges each claim on its own. That is right for a check and wrong for a
reader, because claims rest on each other. In the example the headline claim
**holds** — and the assumption underneath it, that the energy measures interaction
rather than size, is **refuted**. Nothing in `claims.json` connects the two. The
`[belief]` table does.

```bash
python -m claimcheck.belief <project-dir> [<out.json>]   # default: results/belief.json
```

```
  affinity (rests on measured_affinity): nothing in doubt
  mace (rests on mace_e_int): 2 caveat(s)
      [assumed] docked_pose --approximates--> crystal_pose
                taken as true, never tested
      [falsified] mace_e_int --approximates--> interaction_strength
                we tested it and it is false. A RESULT, not a gap — it raises no
                open question, and it carries a caveat on every number resting on it.
                recomputed from mace.e_int_kcal ~ mace.n_lig_atoms
  OPEN  [assumed] docked_pose --approximates--> crystal_pose
  OPEN  [assumed] vina_score --approximates--> interaction_strength
  6 claim(s) over 6 node(s), 2 delivery(ies), 2 open question(s) -> results/belief.json
```

Three things in that output are the design.

- **Warnings travel.** The pose caveat is two edges away and no relationship in
  the project mentions poses.
- **They stop.** The example also declares `mace_e_int --supersedes--> vina_score`
  and an untested assumption still sitting on `vina_score` — same predicate, same
  object as the refuted claim — and it does *not* arrive. A replacement does not
  inherit its predecessor's doubt.
- **They pick an end.** `affinity` is the yardstick the energy was judged against.
  Refuting the energy vindicates the yardstick, so `affinity` hears nothing.

A claim is `subject --predicate--> object` with a status. Write `tested_by` naming
a relationship and the recomputed verdict supplies the status; writing a `status`
beside it is refused, because a status about to be overwritten is still one
somebody opens the file and believes.

```python
from claimcheck.belief import BeliefGraph
g = BeliefGraph.from_project(".")

g.knows("mace")             # True — is this name joined to the graph at all?
g.caveats_on("mace")        # a dict per warning that belongs beside a number from it
g.open_questions()          # a dict per claim with an experiment still waiting
g.stale_if("docked_pose")   # ['mace_e_int'] — recompute these if this changes
g.in_doubt("mace_e_int")    # ['crystal_pose', 'docked_pose', 'interaction_strength',
                            #  'measured_affinity']
g.contradictions()          # triples asserted under more than one status; [] here
g.summary(); g.report()     # counts, and everything as JSON
```

`stale_if` and `in_doubt` answer different questions from different columns of
[the predicate table](#13-predicates): *if I change this, what has to be
recomputed?* and *if this is refuted, whose numbers must say so?*

`caveats_on` includes `falsified`, which `open_questions` deliberately excludes: a
falsified claim was tested, the answer is known, and a finished result on a to-do
list is how a to-do list stops being read.

Also public: `Claim`, `Node`, `bearers(claim)` (which end a warning belongs
beside), `check_claim`, `BeliefGraph.from_dict`, `.supports_edges()`,
`.stale_table()`, `BeliefError`.

---

### `vocabulary`

The closed word lists, with the reason for each classification written beside it.
Imports nothing, not even the rest of `claimcheck`. Tables are in
[The closed vocabulary](#the-closed-vocabulary).

Derived tuples, never second hand-written lists: `OPEN_QUESTION`,
`CARRIES_CAVEAT`, `EVIDENCE_REQUIRED`, `PROPAGATES_STALENESS`, and `DISAGREE` —
the predicates whose two columns differ, of which there is exactly one today.

The lists are closed, rather than expressed as ontology rules, for one reason:
"this makes nothing stale" and "doubt travels nowhere" are **answers**, not
omissions. Under the open-world semantics ontologies use, the absence of a rule is
silence, and five of the thirteen predicates would become silence.

---

### `jobs`

Snakemake's record answers *what ran*. Three things are not in it and cannot be
derived from it: **why** the run was launched, **what would have counted as
success** declared before it ran, and **the verdict**. For work handed to a
cluster scheduler there is no Snakemake record at all, which is the ordinary case.

Write one JSON object per line and point `[jobs]` at the file:

```json
{"id": "score-2026-08-05",
 "question": "does the MACE interaction energy preserve the measured ordering?",
 "expected": ["results/ordering.csv"], "check": "python -m claimcheck.relate .",
 "verdict": "refuted", "generated": [],
 "started": "2026-08-05T11:02:00", "ended": "2026-08-05T11:39:00",
 "note": "the run finished; the ordering it produced does not match."}
```

```python
from claimcheck.jobs import JobLog
JobLog.from_project(".").summary()
```

```python
{'jobs': 3,
 'by_verdict': {'agrees': 1, 'refuted': 1, 'unadjudicated': 1},
 'adjudicable': 2,
 'nothing_declared_to_check': 1,
 'superseded': 0,
 'questions_recorded': 3,
 'verdict_statuses': {'agrees': 'measured', 'refuted': 'falsified',
                      'unverifiable': 'untested'}}
```

**`nothing_declared_to_check` is reported apart from failures, and that is the
point of the module.** In the registry this schema was drawn from, 112 of 190 jobs
declared nothing that could fail and 30 failed a check that really ran. A report
that cannot tell those apart shows 142 problems, and a list where 112 non-events
sit beside 30 real ones is a list people learn to scroll past. `log.undeclared()`
returns the first group.

Five refusals — a record in any of these states cannot be published as it stands:

| refused | why |
|---|---|
| no `id` or no `question` | a run whose purpose is unrecorded can be rebuilt as an event and never as a question |
| a `verdict` outside the three words | they are `relate`'s, imported rather than restated |
| a `verdict` with no `expected` and no `check` | no check on it could have failed |
| `verdict: "agrees"` with declared outputs missing | a pass claims the criterion was met; an exit status is not that claim |
| `generated` files listed and no verdict | something looked; say what it concluded |

```bash
python -m claimcheck.jobs <project-dir> [<out.ttl>]     # or: emit jobs
```

The graph is **PROV-O**, the W3C vocabulary for saying who did what, when, and
what it produced. The distinction the rest of it turns on:

```turtle
<.../job/score-2026-08-05> a prov:Activity ;
    dct:description "does the MACE interaction energy preserve the measured ordering?" ;
    prov:qualifiedAssociation <.../job/score-2026-08-05/association> ;
    cc:verdict "refuted" ; cc:status "falsified" .

<.../job/score-2026-08-05/criterion> a prov:Plan ;
    rdfs:label "acceptance criterion for score-2026-08-05" ;
    dct:description "python -m claimcheck.relate ." ;
    dct:requires "results/ordering.csv" .
```

An **expected** output goes on the plan under `dct:requires`. Only a **confirmed**
one becomes `prov:generated`, because `prov:generated` asserts that a file came
into existence — so putting the expected list there would have every failed job
publish the existence of files that are not on disk.

Nothing here hashes anything or writes a run receipt; that is Snakemake's, and
`grading` reads it from Snakemake. Public: `Job`, `JobLog`, `check_job`,
`JobError`, `VERDICTS`, `build`, `render`.

---

### `grading`

`relate` says whether a claim holds. It says nothing about the file the claim was
computed from. `grading` answers the other half — *what is known about how this
file came to exist?* — by reading Snakemake's metadata store. It writes nothing
into it.

```bash
python -m claimcheck.grading <project-dir>              # grade everything
python -m claimcheck.grading <project-dir> --check      # exit 1 if any is not quotable
python -m claimcheck.grading <project-dir> --reproduce  # re-run in a COPY first
python -m claimcheck.grading <project-dir> --vanished   # what did it make that is now gone?
python -m claimcheck.grading <project-dir> --json OUT
```

```
REFUSED   data/mace_energies.csv
            grade: NONE
            on disk, and no rule in this workflow's Snakemake store claims it.
            Write the rule that produces it, run it, then re-ask.
QUOTABLE  results/claims.json
            grade: EXECUTED
            rule:  check_claims
            produced by a completed run of rule 'check_claims'; not re-run and compared.

5 of 7 quotable (EXECUTED: 5, NONE: 2)
```

| grade | |
|---|---|
| `NONE` | on disk, and no rule claims it |
| `DECLARED` | a rule claims it, and no completed run left a file here |
| `EXECUTED` | a completed run of a rule produced it |
| `REPRODUCED` | re-run in a copy of the project, byte-identical |
| `VERIFIED` | reproduced, **and** every recorded input digest still matches |

`EXECUTED` is not `REPRODUCED`. A workflow engine reporting "nothing to be done"
has compared timestamps; it has not opened the file. Only `--reproduce` — which
copies the project to a temporary directory excluding `.snakemake`, deletes the
targets and re-runs — awards the top two rungs.

**Grade and staleness are separate axes and `quotable` needs both:** at least
`EXECUTED`, *and* a record that still describes the file. A record stops fitting
when a recorded input digest no longer matches the file of that name, or when the
file is newer than the run that recorded it.

`--vanished` is the only check here that can see a **deletion**. Everything else
walks the filesystem, so a removed file is simply absent and nothing notices:

```
VANISHED  results/dcat.ttl
            made by rule 'dcat'; regenerate with `snakemake results/dcat.ttl`

1 of 5 recorded output(s) are no longer on disk
```

With no Snakemake store at all it prints "nothing to compare against, which is not
the same as nothing missing" and exits 1 — an absence of evidence, not a pass.

Public: `Grade` (with `.quotable`, `.render()`), `grade`, `grade_all`, `quotable`,
`report`, `targets`, `recorded_outputs`, `vanished`, `run_record`, `unfinished`,
`reproduce`, `digest`, `GRADES`, `QUOTABLE_FROM`, `ReproduceError`.

---

### `emit`

One command for every output format.

```bash
python -m claimcheck.emit --list
python -m claimcheck.emit <format> <project-dir> [<out>]   # no <out> prints to stdout
claimcheck <format> <project-dir> [<out>]                  # same command, after pip install
```

```
  belief       what rests on what, and what a refutation puts in doubt
  croissant    MLCommons Croissant 1.1 — how to READ the dataset
  dcat         W3C DCAT — how to FIND the dataset, for a catalogue
  evi          EVI's two propagation axioms, for a reasoner with no network
  graph        the claims and columns as RDF, joined to the digests
  jobs         PROV-O: why a run was launched, and what would have counted
  psdi         DCAT in PSDI's profile, checked against PSDI's own SHACL
  rocrate      RO-Crate 1.1 — the whole directory, named and hashed
```

Without the optional packages the same command says which install would fix it,
rather than crashing three frames down later:

```
! psdi         DCAT in PSDI's profile, checked against PSDI's own SHACL
               needs the 'psdi' extra — pip install 'claimcheck[psdi]'
! rocrate      RO-Crate 1.1 — the whole directory, named and hashed
               needs the 'rocrate' extra — pip install 'claimcheck[rocrate]'
```

```
$ python -m claimcheck.emit nope .
error: no format named 'nope'. Known: belief, croissant, dcat, evi, graph, jobs, psdi, rocrate
```

**Adding a format does not require forking this package.** An emitter is a
function `render(root) -> str`. Ship a package registering one under the
`claimcheck.emitters` entry-point group and it appears in `--list` with no change
here. Public: `available()`, `get(name)`, `emit(name, root)`, `EmitError`,
`ENTRY_POINT_GROUP`.

---

### `croissant`

**Croissant** is a JSON-LD format from MLCommons that tells a program how to
*load* a dataset: which files, which columns, what type, how to extract them.

```bash
python -m claimcheck.croissant <project-dir> [<out.json>]   # default: results/croissant.json
```

`claimcheck` emits Croissant 1.1 and adds two things it has no slot for, in its
own namespace. **Units** — Croissant 1.1 has no unit mechanism at all, verified
against the downloaded specification. **Claims** — Croissant describes what a
dataset *is*, not what anybody asserts about it, so the checked relationships ride
alongside with their verdicts and evidence locators.

```json
{
  "@type": "cr:Field",
  "@id": "mace_energies_csv_records/e_int_kcal",
  "name": "e_int_kcal",
  "description": "MACE-OFF interaction energy. Converted from eV by the prepare rule.",
  "dataType": "sc:Float",
  "source": {"fileObject": {"@id": "mace_energies_csv"},
             "extract": {"column": "e_int_kcal"}},
  "https://w3id.org/claimcheck/ns#unit": "kcal/mol",
  "https://w3id.org/claimcheck/ns#unitSystem": "UDUNITS-2"
}
```

Whether a file was computed or typed in by hand travels with the file, because a
consumer who cannot tell them apart has been given the same confidence in both:

```json
{
  "@type": "cr:FileObject", "@id": "experimental_affinity_csv",
  "contentUrl": "../data/experimental_affinity.csv",
  "sha256": "b1dc99b64502928fbe4c83c70a8cb331721016f2ff81d1bd1977e7f5e0c48628",
  "https://w3id.org/claimcheck/ns#origin": "manual",
  "https://w3id.org/claimcheck/ns#manualSource": "Exported by hand from the project's curated affinity table, 2026-08-15",
  "https://w3id.org/claimcheck/ns#retrieved": "2026-08-15"
}
```

Note `contentUrl` is `../data/…`, not `data/…`. Croissant readers resolve a
relative `contentUrl` against the folder holding the JSON-LD file, not the project
root. Getting that wrong produced a manifest that validated cleanly and yielded
**zero** rows to a consumer — which is why [`conform`](#conform) exists. Column
types are read off the data rather than declared, so a scaffolded project and a
published one cannot disagree about what a column is.

Strip the `claimcheck` namespace and what remains is ordinary Croissant. Public:
`build(root, claims_path=None, out_dir=None)`, `render(root)`, `sha256`.

---

### `dcat`

**DCAT** is a W3C vocabulary for catalogue entries: title, publisher, licence,
keywords, download links. It is what `data.gov`, `data.europa.eu` and
institutional repositories speak. Output is **Turtle** (`.ttl`), the ordinary text
format for RDF — facts written as subject–predicate–object.

```bash
python -m claimcheck.dcat <project-dir> [<out.ttl>]     # or: emit dcat
```

Croissant and DCAT are not rivals. Croissant answers *how do I read this?* for a
loader; DCAT answers *how do I find this?* for a search index. The reason to emit
DCAT is that some of what `claimcheck` holds has a **standard slot**, and a fact
in a standard slot is one a stranger's tooling can use. The digest goes into
`spdx:checksum` (SPDX is the vocabulary DCAT-AP already uses for file checksums);
a hand-copied file's origin goes into `dct:provenance`:

```turtle
<https://w3id.org/claimcheck/distribution/data_experimental_affinity.csv> a dcat:Distribution ;
    dct:provenance "Manually imported. Source: Exported by hand from the project's
                    curated affinity table, 2026-08-15. Retrieved: 2026-08-15." ;
    dct:title "experimental_affinity.csv" ;
    spdx:checksum <https://w3id.org/claimcheck/checksum/data_experimental_affinity.csv> ;
    dcat:byteSize "14609"^^xsd:nonNegativeInteger ;
    dcat:downloadURL "data/experimental_affinity.csv" ;
    dcat:mediaType "text/csv" .
```

Units and verdicts stay in `claimcheck`'s own namespace, because DCAT has no slot
for them and minting `dcat:refuted` would imply a standard that does not exist. A
test pins that we do not.

`readback(path)` answers what validation cannot: can a consumer holding only this
file get the distributions it names? It resolves relative `downloadURL`s against
the folder the record sits in — the only base such a reader has — and returns
`(complaints, number_dereferenced)`. The count matters: an empty complaint list
over zero checks is not a pass.

---

### `graph`

Columns, claims and digests as one RDF file.

```bash
python -m claimcheck.graph <project-dir> [<out.ttl>]   # default: results/graph.ttl
```

```
  results/graph.ttl  —  143 triples, 10 node kind(s): Artefact, Claim, Dataset,
  Evidence, Field, FileObject, Method, Node, Ontology, Relationship
```

Four levels of description exist, and only two of them belong in a graph:

| level | holds | lives in |
|---|---|---|
| rows | the numbers | the CSV. **Never triples** |
| schema | one node per column | RDF, and Croissant's `cr:Field` |
| claims | relationships, evidence, verdicts | RDF |
| execution | who made the file, from what | Snakemake's store; joined by path and digest, never restated |

A million rows is five million triples answering no question a CSV reader could
not. **The column is the hinge.** Croissant already gives every column an
identifier and already points at the file with its digest, so a claim attached to
one reaches the bytes without inventing an identifier scheme, and "where is this
evidence from?" becomes a walk from a DOI and a figure number down to a named
column, in a named file, with the hash Snakemake recorded when it made it.

The belief layer is emitted in **EVI**'s verbs — EVI is the Evidence Graph
Ontology, a published vocabulary for saying which evidence supports or challenges
which claim — rather than private predicates, so a consumer with a reasoner
derives the consequences of a refutation instead of being told how to walk the
graph. A reasoner gets a **subset** of `belief.caveats_on`, deliberately: no rule
can conclude "nobody has looked at this, therefore doubt it", so `assumed` and
`untested` claims are emitted with their status and no challenger.

---

### `evi`

The two rules that make a refutation propagate, restated so the emitted graph is
self-contained.

```bash
python -m claimcheck.emit evi <project-dir> [<out.ttl>]
```

```turtle
@prefix evi: <https://w3id.org/EVI#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

<https://w3id.org/EVI> rdfs:comment "Two axioms of EVI 1.6 restated verbatim so
    that a graph emitted by claimcheck propagates a refutation under a reasoner
    that does not resolve https://w3id.org/EVI. Nothing here is claimcheck's own;
    see claimcheck/evi.py." .

evi:indirectlyChallenges a owl:ObjectProperty ;
    owl:propertyChainAxiom ( evi:directlyChallenges evi:supports ) .

evi:supports a owl:ObjectProperty,
        owl:TransitiveProperty .
```

Four triples saying: support chains through many steps, and challenging one link
challenges everything resting on it. **They are EVI's, not ours.** `graph.ttl`
already imports EVI, so a reasoner that fetches imports needs nothing from here;
this file is for the one that does not — an air-gapped run, a store with the
network off, or a reader who wants the rule in front of them. `render(root)`
ignores `root`, because axioms do not depend on a project.

---

### `psdi`

PSDI is the UK Physical Sciences Data Infrastructure. It is the only external
format here whose conformance can be checked against rules **its own maintainers
wrote** — everything else `claimcheck` emits is checked by a validator we picked.
Those rules are **SHACL**: a language for writing constraints an RDF document must
satisfy, checked by `pyshacl` against shapes vendored under
`claimcheck/resources/psdi/` and pinned to an upstream commit.

```bash
python -m claimcheck.emit psdi <project-dir> [<out.ttl>]
python -m claimcheck.psdi --check <project-dir>     # the conformance report
python -m claimcheck.psdi --selftest                # proof the checker can fail
```

**This is a second, separate document, not a flag on `dcat`.** All four of PSDI's
shapes are *closed*, meaning a conformant record may carry only the properties
they list — `spdx:checksum` and even W3C's own `dcat:distribution` are rejected
rather than ignored. Measured: plain DCAT scores 25 violations against them.

```json
{
  "validator": "pyshacl against PSDI's published shapes",
  "shapes_commit": "94855caf4e3d",
  "fully_conforms": false,
  "passed_excluding_exemptions": false,
  "crashed": false,
  "exempt_violations": [
    "IdentifierPropertyShape: Less than 1 values on <...>->dcterms:identifier",
    "LogoURLPropertyShape: Less than 1 values on <...>->psdiDcatExt:logoURL",
    "DisplayPriorityPropertyShape: Less than 1 values on <...>->psdiDcatExt:displayPriority"
  ],
  "real_violations": [
    "KeywordPropertyShape: Less than 1 values on <...>->dcat:keyword  [declare
     `keywords = [...]` under [project] in context.toml — this one is not invented for you]"
  ]
}
```

Three shapes cannot be satisfied and are exempt with a reason each: the identifier
must carry a UUID **PSDI issues**, and the logo URL and display priority are
catalogue presentation nobody here has a truthful value for. **An exemption covers
a missing value and nothing else** — an invented identifier, a logo URL that 404s
and a non-numeric priority all still fail:

```
  PASS  a conformant record passes
  PASS  an invented identifier is refused by the exemption's scope
  PASS  a record with no keyword fails

selftest: PASS  (shapes 94855caf4e3d)
```

Two things are refused rather than fudged. **No checksum:** PSDI's profile permits
SHA-1 only and we compute SHA-256, so the optional field is omitted and the digest
is published truthfully elsewhere. **No invented keyword:** if `[project].keywords`
is undeclared the report names the file and the key, which is exactly the one real
violation above — the example project declares no keywords.

A crash is never a pass: `Conformance.passed` is false whenever `crashed` is true,
because nothing looked. Public: `build`, `render`, `validate(graph)`,
`check(root)`, `check_file(path)`, `Conformance`, `shapes_commit()`, `EXEMPT`,
`EXEMPT_CONSTRAINT`, `SHAPES_FILE`, `PROFILE`, `UndeclarableValue`. Optional
catalogue fields go in a [`[psdi]` table](#psdi--optional).

---

### `rocrate`

**RO-Crate** is a packaging convention: one JSON-LD file named
`ro-crate-metadata.json` at the top of a directory, describing every file in it.
Croissant says how to read a dataset, DCAT says how to find it, and a crate says
*here is the whole directory, with every file named and hashed*. It is the format
a repository deposit takes. Built by `rocrate` (the reference implementation) and
validated by `roc-validator`.

```bash
python -m claimcheck.rocrate <project-dir>        # writes ro-crate-metadata.json
python -m claimcheck.rocrate --check <crate-dir>  # runs roc-validator, offline
python -m claimcheck.rocrate --selftest
python -m claimcheck.emit rocrate <project-dir>   # to stdout
```

```
  ro-crate-metadata.json  —  RO-Crate 1.1, 10 file(s), 12 entities
```

Three things worth knowing:

- **The version is passed explicitly.** `ROCrate()` defaults to 1.2. A bare
  constructor moves the published format with no diff showing it, so
  `RO_CRATE_VERSION = "1.1"` is the decision and a test pins it.
- **`sha256` is declared, or it is not published.** RO-Crate 1.1's context has
  2,627 terms and no checksum among them, so a bare `sha256` key expands to
  nothing and every conformant processor drops it — while `json.loads`, which is
  how anyone would look, still shows it. `EXTRA_TERMS` maps it onto SPDX.
- **The directory holding the metadata file *is* the crate.** Every entity
  identifier is relative to it, so `write()` refuses to put the crate anywhere
  else.

`--check` really runs the validator, offline by construction — a conformance
report that depends on the network says something different on a train. On the
example project it fails, honestly, because the example declares no licence:

```json
{"profile": "ro-crate-1.1", "passed": false, "crashed": false,
 "checks_run": 43, "checks_failed": 1,
 "issues": ["REQUIRED: The Root Data Entity MUST have a `license` property
             (as specified by schema.org)."]}
```

A pass at `REQUIRED` severity says the crate is well formed, not that it is
informative. `readable(metadata)` is the check that asks whether a consumer can
get the files, comparing each entity against the file on disk, its recorded size
and its recorded digest:

```
  PASS  a crate at the root of what it describes reads back
  PASS  the same crate one directory away is caught
  PASS  writing a crate away from its root is refused
  PASS  a digest that does not describe the bytes is caught
  PASS  a missing crate is not a pass (crashed=True)

selftest: PASS  (RO-Crate 1.1)
```

Public: `build(root, name=None, description=None, license_url=None, exclude=())`,
`write(root, out=None)`, `render`, `validate(crate_dir)`, `readable(metadata)`,
`payload`, `root_dataset`, `Conformance`, `CrateError`, `RO_CRATE_VERSION`,
`EXTRA_TERMS`, `MEDIA_TYPES`, `SKIP_DIRS`.

---

### `conform`

On 2026-08-17 this package's own Croissant validated cleanly under `mlcroissant`
and yielded **zero rows** to a consumer. Every check that existed passed. So every
finding here carries **two verdicts that are never folded together**: did a
validator accept it, and could a consumer really read it?

```bash
python -m claimcheck.conform <tree-or-file>
python -m claimcheck.conform --json <tree>
python -m claimcheck.conform --selftest
```

```
PASS    croissant validates=PASS    reads=PASS    results/croissant.json
          mlcroissant, 2 record set(s) read
FAIL    dcat      validates=UNKNOWN reads=FAIL    results/dcat.ttl
          no profile declared, and DCAT itself ships no normative shapes —
          nothing validated this record.
          UNREADABLE dcat:downloadURL 'data/mace_energies.csv' does not resolve
          from results, which is the only base a consumer holding this file has

2 metadata document(s) of 5 file(s) — {'FAIL': 1, 'PASS': 1}
declared versions: {'croissant 1.1': 1, 'dcat plain': 1}
```

Four outcomes, and `UNKNOWN` is not one of the good ones:

| | |
|---|---|
| `PASS` | both verdicts reached and both good |
| `FAIL` | something ran and said no |
| `UNKNOWN` | no verdict — unclassifiable, uninstalled validator, or a crash |
| `SKIP` | the file is not metadata |

A crash is `UNKNOWN`, never `FAIL`: "the validator broke on your manifests" and
"your manifests are invalid" send you to different places. Detection is
**structural, never by filename** — a file is a Croissant manifest if it declares
`conformsTo` naming mlcommons or carries `recordSet`, a crate if it has the
descriptor the specification requires, a DCAT record if the Turtle names the DCAT
namespace.

```
  PASS  a Croissant whose data is where it says reads  (PASS/PASS)
  PASS  the same Croissant one directory away VALIDATES AND CANNOT BE READ  (PASS/FAIL)
  PASS  a truncated document is UNKNOWN, not silently skipped  (UNKNOWN)
  PASS  ordinary JSON is SKIP, not a pass  (SKIP)
  PASS  the sweep counts the validates-but-unreadable case separately (1 of 3)
  PASS  an empty sweep says nothing was examined rather than reporting health

selftest: PASS
```

Exits 1 if anything is `FAIL` or `UNKNOWN`. Public: `sweep(root, suffixes=None)`,
`check_one(path)`, `classify(path)`, `format_report(report)`, `Finding`,
`Unreadable`, `PASS`/`FAIL`/`UNKNOWN`/`SKIP`, `CANDIDATE_SUFFIXES`.

---

### `scaffold`

Turn one or more CSVs into a working project.

```bash
python -m claimcheck.scaffold <new-dir> data.csv [more.csv ...] [--force]
```

A five-row, three-column CSV with one numeric column:

```
$ python -m claimcheck.scaffold /tmp/cc-new /tmp/tiny.csv
  /tmp/cc-new  —  1 dataset(s), 5 row(s), 3 column(s)
    data/tiny.csv                         5 rows  sha256:f2ae3cd3d24c…
  1 numeric column(s) with NO declared unit: tiny.energy
  -> every relationship touching one returns `unverifiable` until you declare it.
  0 relationships: the template in context.toml is commented out, because
     `claimed` is a number only you can supply.

  cd /tmp/cc-new && snakemake -n
```

It writes `context.toml` (a dataset block per file, a variable block per column),
`data/` with your CSVs copied in so the project is self-contained, a `Snakefile`
and a `README.md`. `snakemake -c1` works on the result immediately.

**It will not guess a unit and it will not invent a claim.** Inferring that a
column called `energy` means kcal/mol would manufacture exactly the fact this
package exists to notice is missing, so the column gets an instruction instead:

```toml
[[variables]]
dataset = "tiny"
name = "energy"
description = "sc:Float, inferred from 5 of 5 non-empty value(s) in data/tiny.csv. Replace this with what the column MEANS."
# NO UNIT DECLARED. claimcheck does not guess one: add a `unit` line in
# UDUNITS-2 syntax (kcal/mol, kJ/mol, nm, s, K, … or "1" for a genuinely
# dimensionless quantity). Until then every relationship touching this
# column comes back `unverifiable`, which is not a pass.
```

A `[[relationships]]` block needs `claimed` — the number somebody asserted — and
nothing in a CSV records what anyone claimed, so the generated file carries a
commented template naming real columns from your data and nothing more.

What it *does* infer, because the data says so: column types (through the same
function that types a field for a Croissant consumer), row and value counts, and
the digest of every copied file. Everything is refused before anything is written,
so a rejection leaves no half-project behind: an empty file, a header with no
rows, a duplicated or blank column name, or a tab-separated file all stop it with
a specific message.

Public: `scaffold(directory, data, *, name=None, force=False) -> dict`,
`build_context(name, datasets)`, `ScaffoldError`, `SNAKEFILE`, `README`.

---

### `namespace`

The one place the project's identifier base is written down.

```python
from claimcheck.namespace import BASE, NS, RESOLVES
BASE      # 'https://w3id.org/claimcheck/'
NS        # 'https://w3id.org/claimcheck/ns#'
RESOLVES  # False
```

**These identifiers do not currently resolve, and that is stated here rather than
discovered with `curl`.** `w3id.org` redirects are registered by pull request and
nobody has filed one; until then they identify things without pointing at
anything, which is legal and common. `RESOLVES` is checked by a test rather than
asserted — when the redirect is registered it becomes `True` and the test starts
requiring the URLs to work.

---

## The closed vocabulary

Three closed lists in `claimcheck/vocabulary.py`. A word outside them is refused at
load time rather than creating a phantom entry.

### 7 statuses

How well a claim is supported. Three flags derive from each: may it be asserted
with no evidence, does it leave an experiment waiting, and must a warning travel
with any number resting on it.

| status | means | evidence required | open question | carries a warning |
|---|---|---|---|---|
| `measured` | we measured it here, and the evidence says where | yes | no | no |
| `published` | someone else measured it; the evidence is a citation | yes | no | no |
| `by_construction` | true by how the thing is built — a definition, not evidence | no | no | no |
| `assumed` | taken as true, never tested | no | yes | yes |
| `contested` | evidence exists on both sides | yes | yes | yes |
| `falsified` | we tested it and it is false | yes | **no** | **yes** |
| `untested` | asserted with nothing behind it; nobody has looked | no | yes | yes |

The last two columns differ for exactly one status, and that is the point.
`falsified` raises no open question — it was tested, the answer is known — and it
carries the loudest warning there is, because a number computed on a method proven
wrong is exactly what a reader must be told about. Answer both questions with one
flag and a file resting on a falsified method reads as unqualified.

### 13 predicates

The verb in `subject --predicate--> object`. Each answers two questions separately:
**stale** — if I change this, what has to be recomputed? — and **doubt** — if this
is refuted, whose numbers must say so?

| predicate | means | goes stale | doubt lands on |
|---|---|---|---|
| `approximates` | subject is a cheaper or rougher stand-in for object's physics | subject | subject |
| `is_cheaper_surrogate_for` | subject is run instead of object to save time | subject | subject |
| `is_distilled_from` | subject learns object's labels and cannot exceed it | subject | subject |
| `depends_on` | subject's value changes if object changes | subject | subject |
| `is_precondition_of` | object is meaningless unless subject holds first | object | **object** |
| `is_validated_against` | subject's correctness is judged by comparing it to object | subject | subject |
| `rescues_ranking_errors_of` | subject recovers items object ranks wrongly | subject | subject |
| `preserves_ordering_of` | subject reproduces object's ordering | subject | subject |
| `shares_approximation_with` | both make the same assumption, so agreement between them is not independent evidence | **nothing** | **both ends** |
| `supersedes` | subject replaces object; records that a replacement happened | nothing | nowhere |
| `is_invalidated_by` | subject's conclusions are void when object holds | nothing | nowhere |
| `dominates_error_of` | subject contributes more error than object | nothing | nowhere |
| `displaces_ranking_more_than` | subject reorders the shortlist more than object | nothing | nowhere |

The two columns agree on twelve of thirteen, which is why they were once collapsed
into one. The thirteenth is `shares_approximation_with`: changing one method does
not oblige recomputing the other, but refuting the shared assumption refutes it on
both sides at once.

"Nothing" and "nowhere" are **answers**, not omissions. A replacement does not
inherit its predecessor's doubt — being wrong is usually why it was replaced. A
comparison of two error magnitudes makes neither term depend on the other.

### 7 node kinds

What a thing in the belief graph can be. Nothing branches on the kind; the list is
closed so a typo is refused rather than creating a phantom.

| kind | |
|---|---|
| `method` | something that computes a number |
| `reference` | something a method is judged against |
| `parameter` | a chosen value the numbers depend on |
| `condition` | a state of the world the numbers assume |
| `quantity` | a physical quantity, however it is obtained |
| `representation` | a way of encoding the object of study |
| `strategy` | a plan for spending a budget |

---

## The `context.toml` format

One file at the root of your project. Everything else — adding a dataset, relating
two, publishing a new format — is an edit here.

### A minimal complete example

Copy this, put a CSV at `data/measurements.csv`, and `python -m claimcheck.relate .`
works.

```toml
[project]
name = "my_project"
title = "What this dataset is, in one line"

[[datasets]]
id = "meas"
origin = "computed"                    # a rule in this workflow made it
files = ["data/measurements.csv"]

[[variables]]
dataset = "meas"
name = "energy"
unit = "kcal/mol"                      # UDUNITS-2. Omit only if nobody has declared one
description = "Computed interaction energy"

[[variables]]
dataset = "meas"
name = "affinity"
unit = "1"                             # "1" is dimensionless, declared. Not the same as absent
description = "Measured affinity as pKD"

[[relationships]]
x = "meas.energy"
y = "meas.affinity"
kind = "correlation"
method = "pearson"
claimed = -0.5                         # the number SOMEBODY ASSERTED
resampling_unit = "series_id"          # the column your rows are independent across
[relationships.evidence]
source = "Figure 3 of the paper this pipeline is based on"
locator = "doi:10.0000/example#fig3"
extracted_by = "human"
```

### `[project]`

| key | required | used by | |
|---|---|---|---|
| `name` | **yes** | everything | the project identifier; appears in every emitted document |
| `title` | no | croissant, dcat, psdi, rocrate | one line saying what the dataset is |
| `context` | no | croissant, dcat, psdi, rocrate | longer free text: what a reader must know before believing any of it |
| `license` | no | croissant, dcat, psdi, rocrate | a URL is treated as one; RO-Crate 1.1 requires it to validate |
| `version` | no | croissant, dcat, psdi | |
| `date_published` | no | croissant, dcat, psdi, rocrate | `YYYY-MM-DD` |
| `citation` | no | croissant | |
| `keywords` | no | dcat, psdi | a list of strings; **required** for a PSDI record, and never invented for you |
| `publisher` | no | dcat, psdi | a name |
| `results` | no | grading | the results directory name; defaults to `results` |

### `[[datasets]]`

One block per dataset; at least one is required.

| key | required | |
|---|---|---|
| `id` | **yes** | referred to as `id.column` everywhere else |
| `origin` | **yes** | `"computed"` or `"manual"`. **No default** |
| `files` | **yes** | a list of CSV paths relative to the project root |
| `source` | manual only | where the file came from |
| `retrieved` | manual only | when |
| `sha256` | manual only | checked on **every** load; the run stops if it does not match |
| `resampling_unit` | no | the default resampling column for claims involving this dataset |

### `[[variables]]`

One block per column you want to talk about. Columns you never mention need none.

| key | required | |
|---|---|---|
| `dataset` | **yes** | which dataset's column this is |
| `name` | **yes** | the column header |
| `unit` | no, but | UDUNITS-2. **Any claim touching a column with no declared unit returns `unverifiable`** |
| `description` | no | what the column means |

### `[[relationships]]`

One block per claim to check.

| key | required | |
|---|---|---|
| `x`, `y` | **yes** | `dataset.column`. A bare column name is allowed only when the project has exactly one dataset |
| `kind` | **yes** | `"correlation"` |
| `method` | **yes** | `"pearson"` or `"spearman"` — and the declared one is the one that runs |
| `claimed` | **yes** | the value somebody asserted |
| `join` | when `x` and `y` are in different datasets | a column in both that identifies a row; duplicates are refused |
| `resampling_unit` | no | overrides the dataset's |
| `max_ci_width` | no | wider and the verdict is `unverifiable`; default `1.0` |

`[relationships.evidence]` is a free-form sub-table; `source`, `locator` and
`extracted_by` are the keys carried into the published documents.

### `[belief]` — optional

Three parts. Everything is refused at load if it does not line up: a claim naming
an undeclared node, a duplicate claim, a `rests_on` entry that is empty or names
an unknown node.

An extract from `example/context.toml` — every node a claim mentions must have its
own `[[belief.nodes]]` block, and the ones elided here do.

```toml
[belief.rests_on]
# a name you will ask about later -> the node(s) it rests on.
# A dataset id, a rule name or an output path all work. A list is allowed and is
# usually right — a workflow rule exercises several things at once.
mace = "mace_e_int"
affinity = "measured_affinity"

[[belief.nodes]]
id = "mace_e_int"
kind = "quantity"                      # one of the 7 node kinds
label = "MACE-OFF interaction energy, evaluated on the docked pose"

# A claim whose status is RECOMPUTED. `tested_by` names a relationship as "x ~ y"
# and the verdict sets the status. Writing `status` or `evidence` beside
# `tested_by` is refused.
[[belief.claims]]
subject = "mace_e_int"
predicate = "approximates"
object = "interaction_strength"
tested_by = "mace.e_int_kcal ~ mace.n_lig_atoms"

# A claim somebody asserts by hand. `status` is required, and statuses in the
# "evidence required" column may not have it empty.
[[belief.claims]]
subject = "docked_pose"
predicate = "approximates"
object = "crystal_pose"
status = "assumed"
note = "a nearby fact that is NOT evidence — it travels with the warning, never as the warrant"
```

### `[jobs]` — optional

```toml
[jobs]
records = "jobs.jsonl"                 # one JSON object per line
agent = "example-cluster"              # used where a record names no agent
```

Fields in each record: `id` and `question` (both required), `expected` (list),
`check`, `verdict` (`agrees` / `refuted` / `unverifiable`), `generated` (list),
`started`, `ended`, `agent`, `superseded_by`, `note`.

### `[psdi]` — optional

Catalogue-editorial fields with no other home, each modelled the way PSDI's shapes
ask for it: `landing_page`, `landing_page_label`, `access_rights` (one of `open`,
`restricted`, `embargoed`, `metadata only`, or a COAR URL), `publisher_url`,
`contact_point` (an email address, not a `mailto:` URL), `contact_url`, and
`creators` (a list of `{orcid, name}`). Anything undeclared is omitted rather than
invented. `contact_name` without `contact_url` raises `UndeclarableValue`, because
PSDI's contact card permits an email and a URL only. That refusal exists so nobody
writes a name, sees exit 0, and never learns the field went nowhere.

---

## Installation

Python 3.11 or newer.

```bash
pip install -e .                   # the core. NO dependencies at all
pip install -e ".[graph]"          # + rdflib
pip install -e ".[psdi]"           # + rdflib, pyshacl
pip install -e ".[rocrate]"        # + rocrate, roc-validator
pip install -e ".[conform]"        # all of the above + mlcroissant
pip install -e ".[dev]"            # everything, plus pytest and snakemake
```

| what you want | extra | packages |
|---|---|---|
| check claims, publish Croissant, belief, the jobs report, grading, scaffold | *none* | — |
| `graph`, `dcat`, `evi`, and `jobs` as PROV-O | `graph` | `rdflib>=7.0` |
| the PSDI record and its conformance report | `psdi` | `rdflib>=7.0`, `pyshacl>=0.25` |
| RO-Crate | `rocrate` | `rocrate>=0.13`, `roc-validator>=0.11` |
| `conform` sweeping every format | `conform` | the above + `mlcroissant>=1.1` |
| the example workflow and the test suite | `dev` | the above + `pytest>=8`, `snakemake>=9` |

**The core has no dependencies, and that is checked rather than aspired to.** It
matters because the two things this package does that nothing else does —
recompute a claim, and say "the data cannot tell" — must run wherever the data is.
A checker you cannot run on a cluster node because it wants a semantic-web stack
is a checker that does not get run.

Snakemake is a **command**, never an import. `claimcheck` never calls its API; it
reads the JSON files Snakemake leaves in `.snakemake/`.

Every dependency is enforced in both directions by `tests/test_dependencies.py`:
declare a package nothing imports and it fails; import a package nothing declares
and it fails. Each guard has been watched failing.

```bash
pytest        # everything, including ruff and mypy, ~16s
```

Lint and type-checking run *inside* pytest rather than as a separate hook, so one
command runs every check there is.

---

## Starting a new project

**From a CSV you already have:**

```bash
python -m claimcheck.scaffold my-project data.csv
cd my-project
snakemake -n && snakemake -c1
```

Then do the two things it deliberately left blank: declare a `unit` on each
numeric column, and uncomment the `[[relationships]]` template and put your own
number in `claimed`. Until you do, every claim comes back `unverifiable` — which
is loud, and is meant to be.

**From scratch:** copy the [minimal `context.toml`](#a-minimal-complete-example),
put your CSV where it says, and run `python -m claimcheck.relate .`. Add a
`Snakefile` if you want Snakemake to record how each output was made;
`example/Snakefile` is six rules and works as a template.

Once something is running:

```bash
python -m claimcheck.emit --list          # what you can publish
python -m claimcheck.grading .            # which of your files may be quoted
python -m claimcheck.conform .            # can anyone read what you published?
```

---

## Known gaps

- **Signing.** Nothing is cryptographically bound to anybody. `in-toto` over DSSE
  is the answer and is not built.
- **PDF ingestion** — turning a claim printed in a paper into a checkable
  expectation on your data. It is the natural completion of "recompute, never
  read".
- **A resolvable download URL in the plain DCAT record.** `conform` reports it as
  unreadable and it is right: the URL is a relative literal, so it resolves from
  the project root rather than from where the record sits. Two fixes exist —
  rewrite it relative to the output, as Croissant now does, or emit a full URL, as
  `psdi` does — and either changes the published record, so it is a decision
  rather than a patch. A test pins the finding and says to delete itself when the
  decision is taken.
- **A licence on the example.** RO-Crate 1.1 requires one on the root and
  `example/context.toml` declares none, so the example's crate does not validate.
  Recorded as a test rather than papered over.
- **Statistics.** `correlation` is the only `kind`, with Pearson and Spearman
  behind it. Adding one is a function and a line in `ESTIMATORS`.
# claimcheck
