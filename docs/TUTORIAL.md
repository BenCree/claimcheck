# claimcheck in one sitting

A guided tour for someone who writes Python. Everything below was run against the
toy project it builds, and the outputs are the real ones.

## The problem it solves

You compute a number. Six months later somebody asks what it was, what it was
computed from, and whether it meant what the paper said. Usually the answer is a
filename and a shrug.

Snakemake already fixes half of this: it records, for every output, the input
digests, the code, the shell command, the environment and the times. That is the
*run* record and claimcheck does not duplicate it.

The half it adds is the **claim**. Before you look at the data you write down
what you expect. Afterwards the package recomputes it and returns a verdict, so
"the correlation was 0.94" becomes "we predicted 0.9, measured 0.936 with a 95%
interval of [0.912, 0.959] resampling over batches, and the prediction was
refuted because the proxy is better than we said".

The package **runs nothing** and **stores no results**. It reads a declaration
and your data, and answers questions about them.

```
pip install claimcheck            # core: relate, belief, grading, units, labels
pip install 'claimcheck[graph]'   # + rdflib, for croissant/graph/dcat
```

## A toy project in thirty lines

One directory, one CSV, one declaration file. That is a claimcheck project.

```
widget_proxy/
    context.toml            the declaration: what the data is, what you claim
    data/measurements.csv   the data
    results/                everything the package writes
```

The story: each sample has a `cheap_score` that takes a second and a
`slow_measure_kJ` that takes a day. You want to know whether the cheap one can
stand in for the slow one. `coin_flip` is a control: pure noise, which must not
correlate with anything.

```python
import csv, random
from pathlib import Path

Path("widget_proxy/data").mkdir(parents=True, exist_ok=True)
rng = random.Random(7)
rows = []
for b in range(10):                      # ten batches, ten samples each
    for i in range(10):
        truth = rng.uniform(0, 10)
        rows.append({
            "sample":          f"b{b}s{i}",
            "batch":           f"batch{b}",
            "cheap_score":     round(truth + rng.gauss(0, 1.2), 3),
            "slow_measure_kJ": round(truth, 3),
            "coin_flip":       round(rng.uniform(0, 1), 3),
        })

with open("widget_proxy/data/measurements.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
```

**Ten batches, not four.** The batch is the resampling unit, so ten batches is
what the interval is computed from, and a first version of this tutorial used
four and the noise control came out "correlated" at r = 0.404. That was not a
bug; four independent units is simply not enough to tell noise from signal.

## `context.toml`: the whole declaration

Four kinds of block, and nothing else configures the package.

```toml
[project]
name = "widget_proxy"
title = "Does a cheap score predict the slow measurement?"
context = """Forty samples in ten batches. cheap_score takes a second,
slow_measure_kJ takes a day. coin_flip is a control."""

[[datasets]]
id = "runs"
origin = "manual"                       # "manual" or "computed". Nothing else.
files = ["data/measurements.csv"]
source = "https://example.org/widget-lab/measurements.csv"
retrieved = "2026-09-07"
sha256 = "9f2c…"                        # required for manual, checked on load

[[variables]]
dataset = "runs"
name = "slow_measure_kJ"
unit = "kJ/mol"                         # an undeclared unit makes every claim
description = "The expensive measurement."   # touching this column unverifiable

[[relationships]]
id = "R1_proxy_predicts"
x = "runs.cheap_score"
y = "runs.slow_measure_kJ"
kind = "correlation"
method = "pearson"                      # or "spearman", or "kendall" (tau-b).
claimed = 0.9                           # THE PREDICTION, written before looking
resampling_unit = "batch"               # what the bootstrap resamples, not rows
evidence.source = "Claimed at 0.9 before looking."
evidence.locator = "data/measurements.csv :: cheap_score, slow_measure_kJ"
```

`origin` matters. **`computed`** means a rule in your workflow made the file, so
the workflow is its provenance. **`manual`** means a human put it there, and then
`source`, `retrieved` and `sha256` are required, because as the package puts it:

> Snakemake knows nothing about a file no rule produced, so these are the only
> record there is.

Leave them out and the very first call refuses:

```
DatasetError: dataset 'runs' is manually imported and does not declare
source, retrieved, sha256.
```

## `relate`: recompute the claim, get a verdict

```bash
python -m claimcheck.relate .
```

```
REFUTED runs.cheap_score ~ runs.slow_measure_kJ   claimed 0.9
   ->  claimed 0.9 lies OUTSIDE the 95% CI [0.912, 0.959] for r = 0.936,
       resampling 10 batch(s)
{'refuted': 1, 'agrees': 1} over 100 row(s) -> results/claims.json
```

**Read that carefully, because it is the whole idea.** The claim was *refuted*
and the proxy is *better* than claimed. A verdict is about the prediction, not
about quality. "Refuted" is not "bad"; it means the data disagreed with what you
wrote down, and here it disagreed in your favour.

Three verdicts exist and no others:

| verdict | meaning | belief status it carries |
|---|---|---|
| `agrees` | the claimed value lies inside the interval | `measured` |
| `refuted` | it lies outside | `falsified` |
| `unverifiable` | no verdict could be reached: too few units, an undeclared unit, a column that will not parse | `untested` |

**It is an interval, not a tolerance.** A claim sitting inside a very wide
interval is `unverifiable`, not confirmed. An earlier version of the package used
a tolerance and on five rows "falsified" both claims it was given, which is why
the interval replaced it.

### The same thing from Python

```python
from claimcheck.relate import check, pearson, spearman, bootstrap_ci

d = check("widget_proxy")           # the whole report as a dict
d["verdicts"]                       # {'refuted': 1, 'agrees': 1}
d["rows_read"]                      # 100
d["any_refuted"]                    # True

r = d["relationships"][0]
r["verdict"]          # 'refuted'
r["status"]           # 'falsified'
r["estimate"]         # 0.936
r["ci_lo"], r["ci_hi"]   # 0.912, 0.959
r["n_rows"], r["n_units"] # 100, 10      <- rows vs INDEPENDENT units
r["why"]              # the sentence printed above
```

Every field: `id, kind, x, y, x_dataset, y_dataset, x_unit, y_unit, method,
claimed, estimate, ci_lo, ci_hi, ci_level, ci_width, n_rows, n_units,
resampling_unit, verdict, status, why, evidence, dropped, cross_dataset`.

The estimators are plain functions and you can use them alone:

```python
pearson([1, 2, 3, 4], [2, 4, 6, 8])          # 1.0
spearman([1, 2, 3, 4], [1, 8, 9, 100])       # 1.0, monotone but not linear
bootstrap_ci(pairs, clusters=["a","a","b",...])   # (lo, hi, n_clusters)
```

`bootstrap_ci` resamples **clusters**, not rows. Pass `clusters=None` and it
treats every row as independent, which is usually a lie and always worth
knowing you have told.

## `grading`: may I quote this number?

`relate` asks whether a claim held. `grading` asks a different question: is this
file's provenance good enough to quote at all? It answers from Snakemake's own
metadata store, not from anything claimcheck wrote.

```bash
python -m claimcheck.grading .
```

Run it on the toy, which has no Snakemake workflow behind it yet:

```
REFUSED   results/claims.json
            grade: NONE
            on disk, and no rule in this workflow's Snakemake store claims it.
            Write the rule that produces it, run it, then re-ask.

0 of 2 quotable (NONE: 2)
```

The ladder, in order:

| grade | meaning |
|---|---|
| `NONE` | on disk, nothing claims it |
| `DECLARED` | a rule names it as an output, but has not run |
| `EXECUTED` | a rule produced it and the store has the record. **Quotable from here.** |
| `REPRODUCED` | it was rebuilt and the bytes matched |
| `VERIFIED` | reproduced, and its claims were adjudicated |

```python
from claimcheck import grading

grading.grade("proj", "results/table.csv")     # one Grade
grading.grade_all("proj")                      # {path: Grade}
grading.quotable("proj")                       # only those at EXECUTED or better
grading.report("proj")                         # the full dict the CLI prints

grading.crashed("proj")     # outputs whose producing run died
grading.vanished("proj")    # outputs the store knows about, now missing
grading.check_origins("proj")   # declared origin vs who actually makes the file
grading.reproduce("proj", cores=4)   # rebuild and compare bytes -> REPRODUCED
```

`check_origins` is worth its own line. `context.toml` says whether each dataset
is `computed` or `manual`; this checks that against your workflow. Declared
`computed` with no rule producing it is a number whose recipe is gone. Declared
`manual` but written by a rule is a source that is not a source, so its pinned
sha256 describes an output.

## Publishing: `croissant`, `graph`, `dcat`, and `emit`

```python
from claimcheck.emit import available, emit, get

available()
# {'croissant': ready, 'belief': ready, 'jobs': ready, 'graph': ready,
#  'evi': ready, 'dcat': ready,
#  'psdi':    "needs the 'psdi' extra",
#  'rocrate': "needs the 'rocrate' extra"}

text = emit("dcat", "widget_proxy")     # returns a string
fn   = get("croissant")                 # the callable, if you want it directly
```

From the command line, in this order, because `graph` reads the Croissant:

```bash
python -m claimcheck.croissant . results/croissant.json   # 6,647 bytes
python -m claimcheck.graph     . results/graph.ttl        # 97 triples
python -m claimcheck.emit dcat . results/dcat.ttl         # 1,543 bytes
```

`graph` prints what it built, which is more useful than a byte count:

```
results/graph.ttl — 97 triples, 9 node kind(s): Claim, Dataset, Evidence,
Field, FileObject, Method, Node, Ontology, Relationship
```

**Adding a format is a rule and a name.** `emit` looks the name up in a registry,
so a third party can register one through an entry point without changing the
package.

## `belief`: what else is in doubt now

`relate` refutes one claim. `belief` answers "so what else rests on it?" over a
closed vocabulary: **7 statuses**, **13 predicates**, **7 node kinds**.

```toml
[belief]
nodes = [
  { id = "cheap_score",    kind = "quantity", label = "the fast proxy" },
  { id = "slow_measure",   kind = "quantity", label = "the expensive measurement" },
  { id = "screening_plan", kind = "strategy", label = "screen on the proxy, measure the top 10" },
]
claims = [
  { subject = "cheap_score",    predicate = "approximates", object = "slow_measure",
    tested_by = "R1_proxy_predicts" },
  { subject = "screening_plan", predicate = "depends_on",   object = "cheap_score",
    status = "assumed", evidence = "nothing has tested the plan itself" },
]
```

```
OPEN  [assumed] screening_plan --depends_on--> cheap_score
2 claim(s) over 3 node(s), 0 delivery(ies), 1 open question(s)
```

The first claim carries **no** `status`, because `tested_by` points at a
relationship and the recomputed verdict sets it. Write both and the package
refuses:

> `claims[0]` declares both `tested_by` and `status`. The verdict recomputed from
> the data sets the status; a hand-written one beside it is a second answer that
> will be read.

The vocabulary, in full, so you can see how small it is:

- **statuses**: `assumed`, `by_construction`, `contested`, `falsified`,
  `measured`, `published`, `untested`
- **node kinds**: `condition`, `method`, `parameter`, `quantity`, `reference`,
  `representation`, `strategy`
- **predicates**: `approximates`, `depends_on`, `displaces_ranking_more_than`,
  `dominates_error_of`, `is_cheaper_surrogate_for`, `is_distilled_from`,
  `is_invalidated_by`, `is_precondition_of`, `is_validated_against`,
  `preserves_ordering_of`, `rescues_ranking_errors_of`, `shares_approximation_with`,
  `supersedes`

`belief` refuses a project with no `[belief]` table rather than returning an
empty graph, because an empty graph would read as a clean bill of health.

## The small pieces you will use directly

### `units`: an unknown unit raises

```python
from claimcheck.units import convert, compatible, UnknownUnit

convert(1.0, "kcal/mol", "kJ/mol")     # 4.184
compatible("kJ/mol", "kcal/mol")
# (True, 'kJ/mol and kcal/mol are both molar energy, factor 0.239006')
compatible("kJ/mol", "1")
# (True, 'kJ/mol is a molar energy and 1 is a dimensionless — different
#         quantities, so any slope between them carries units and is not a
#         conversion factor')
convert(1.0, "bananas", "kJ/mol")
# UnknownUnit: unit 'bananas' is not declared in claimcheck.units
```

**It never returns the input unchanged.** A silent pass-through on an unknown
unit is how a number changes meaning without changing value.

### `labels`: a positive rate never travels alone

```python
from claimcheck.labels import LabelRule, rate

rule = LabelRule(name="KJ_HIT", assay="slow calorimetry",
                 endpoint="slow_measure_kJ",
                 statement="a sample is positive when slow_measure_kJ exceeds 7.0")

rate(18, 100, rule)
# {'n_positive': 18, 'n_total': 100, 'rate': 0.18,
#  'label_rule': 'KJ_HIT (slow calorimetry, slow_measure_kJ)',
#  'label_name': 'KJ_HIT', 'label_assay': 'slow calorimetry',
#  'label_endpoint': 'slow_measure_kJ',
#  'label_statement': 'a sample is positive when slow_measure_kJ exceeds 7.0',
#  'label_thresholdable': True, 'label_citation': None, 'label_note': None}
```

You cannot get `0.18` on its own. The rule that defines "positive" travels in the
same dict, because a hit rate under one threshold is a different number from a
hit rate under another and they look identical on a slide.

### `datasets`: loading, and joining without lying

```python
from claimcheck.datasets import load, join
import tomllib

ctx = tomllib.loads(open("widget_proxy/context.toml").read())
d = load("widget_proxy", ctx["datasets"][0])
d.keys()          # id, origin, rows, spec, digests
len(d["rows"])    # 100

pairs, report = join(d, other, key="sample")
report.keys()
# key, left, right, n_left, n_right, n_matched,
# n_left_unmatched, n_right_unmatched, coverage, refused
```

`join` hands back a **report**, not just the pairs. Two datasets that overlap on
60% of their keys will silently produce a 60% analysis unless something tells
you, so this tells you.

### `conform`: validating is not reading

```bash
python -m claimcheck.conform .
```

```
PASS  croissant  validates=PASS     reads=PASS   results/croissant.json
        mlcroissant, 1 record set(s) read
FAIL  dcat       validates=UNKNOWN  reads=FAIL   results/dcat.ttl
        UNREADABLE dcat:downloadURL 'data/measurements.csv' does not resolve
        from results, which is the only base a consumer has
```

That FAIL is real and it is the point of the module. The DCAT file is
well-formed; the path inside it is written relative to the project root while a
consumer resolves it against the folder holding the file. **For every format the
question is not "does it validate" but "can somebody get the data out of it."**
The package's own Croissant once validated cleanly and yielded zero records for
exactly this reason.

### `scaffold`: from a bare CSV to a working project

```bash
python -m claimcheck.scaffold widget_proxy data/measurements.csv
```

Writes `context.toml`, `Snakefile`, `README.md` and copies the data in. Then it
tells you what it would not guess:

```
3 numeric column(s) with NO declared unit: cheap_score, slow_measure_kJ, coin_flip
  -> every relationship touching one returns `unverifiable` until you declare it.
0 relationships: the template in context.toml is commented out, because
  `claimed` is a number only you can supply.
```

It will not invent a unit and it will not invent a claim. Both would manufacture
the exact fact the package exists to notice is missing.

## Five refusals, and what each is protecting

Writing this tutorial hit all five. They are the package working, not obstacles.

| refusal | what it prevents |
|---|---|
| `manual` dataset without `source`/`retrieved`/`sha256` | a file with no record at all, since no rule made it |
| a claim with both `status` and `tested_by` | a hand-written verdict sitting next to a recomputed one |
| `graph` before `croissant.json` exists | a graph describing data it never read |
| an undeclared unit | a correlation between quantities nobody checked are comparable |
| `belief` with no `[belief]` table | an empty graph reading as a clean bill of health |

## The shape of a real project

```
project/
    context.toml          the declaration
    Snakefile             your rules, ending with the shared ones:
                          #   from claimcheck.workflow import RULES
                          #   include: RULES
    analyse.py            your analysis
    data/                 inputs and intermediate tables
    results/              claims.json, croissant.json, graph.ttl, dcat.ttl
    figures/  logs/
```

**One constraint shapes how you write your tables.** `relate` computes
correlations and nothing else, so anything that is not a correlation between two
columns of one file has to be *made* one, in the code that writes the table
rather than in the head of whoever reads it. A partial correlation becomes a
column of within-group rank residuals. A paired difference becomes two columns. A
sign test is not expressible at all: compute it in your analysis, report it
beside the verdicts, and say that is what you did.
