# claimcheck

**A dataset arrives with its claims checked, and the checks are recomputed
rather than written down.**

*(The name is provisional and trivially changed.)*

```bash
pip install -e ".[dev]"
cd example && snakemake -n && snakemake -c1        # ~4 seconds
```

```
HOLDS    mean_auc_ef ~ mean_hits_found   claimed 0.83
         within the 95% CI [0.780, 0.897] for r = 0.832, resampling 29 config(s)
REFUTED  mean_auc_ef ~ mean_hits_found   claimed 0.99
         lies OUTSIDE that interval
  results/croissant.json  —  2 field(s), 2 claim(s), 1 refuted
  results/graph.ttl       —  49 triples
```

---

## The idea

Someone computes a number. Six months later nobody can tell whether to trust
it — not because the code was wrong, but because what you would need to judge
it was never written down. What unit it was in. What it was compared against.
What was assumed.

Two things sit beside the data.

**Snakemake already knows how the file was made.** For every output it records
the input digests as sha256, the code, the shell command, the software
environment and the times. We use that and add nothing to it.

**A small text file says what the numbers mean.** `context.toml` gives each
column a unit, and states what anybody claims about them — *"these two are
correlated, about 0.83, according to figure 3."*

**The claims are then checked by recomputing them**, giving one of three
answers: **holds**, **refuted**, or **cannot tell** — and the third is not a
pass. When a person or a language model lifts a number out of a paper, the
mistake is never nonsense; it is a confident, sensible-looking, wrong number.
Recomputing is the only defence that works on that.

## Why this repository exists separately

Its predecessor grew two answers to the same question. About 1,750 lines there
implement things Snakemake, `mlcroissant` and `runcrate` already do — a
duplication found only during a 227-package survey, and still unresolved.

So the rule here is:

> **If Snakemake already does it, do not build it again.
> If a dependency is declared, something must import it.
> If something is imported, it must be declared.**

The last two are mechanically enforced. The first is not enforceable in
general, so the one case that went wrong is enforced specifically: nothing here
may write a run receipt.

## The dependency set, and the test that verifies it

| | needed for | status |
|---|---|---|
| *(nothing)* | `relate.py`, `croissant.py` | **the core is stdlib-only** |
| `rdflib` | `graph.py` only | `[graph]` extra |
| `mlcroissant` | validating what we emit | dev only, and a **command**, never imported |
| `snakemake` | layer 1 | dev only, and a **command**, never imported |
| ~~`rocrate`~~ | — | **dropped**: a second description of the same dataset. Five workflow engines already emit Workflow Run Crate, with `runcrate` validating it |
| ~~`pyshacl`~~ | — | **dropped**: only needed to validate DCAT against PSDI's shapes. It comes back *with* that emitter, not before |
| ~~`linkml`~~ | — | **dropped**: generated a vocabulary this package does not have |

`tests/test_dependencies.py` enforces all of it, and **each guard has been
watched failing**:

* declare `requests` and use it nowhere →
  `declared but imported by nothing: ['requests (declared in dependencies)']`
* `import numpy` in `relate.py` →
  `imported but not declared: {'numpy': ['claimcheck/relate.py']}`
* add a function that hashes inputs into a sidecar →
  `this looks like a run receipt, which Snakemake already writes`

The stdlib claim is checked by **running** the core with a meta-path hook that
refuses every non-stdlib import, not by reading the import lines — a lazy
import inside a function would pass a reading test and fail on a cluster node.

## Adding a dataset, and relating it to another

Everything is `context.toml`. Nothing else in the project changes.

```toml
[[datasets]]
id = "affinity"
origin = "manual"                    # or "computed". There is NO default
files = ["data/experimental_affinity.csv"]
source = "exported by hand from ..."  # required when manual
retrieved = "2026-08-15"              # required when manual
sha256 = "b1dc99b6..."                # required when manual, checked every run

[[relationships]]
x = "mace.e_int_kcal"
y = "affinity.experimental_pKD"
join = "complex_name"                # required when the two differ
kind = "correlation"
method = "pearson"
claimed = -0.5
```

### The manual import cannot go wrong quietly

A file no rule produced is one **Snakemake knows nothing about** — not where it
came from, not when, not whether the copy on disk is the one anybody looked at.
So a manual dataset must declare `source`, `retrieved` and `sha256`, and the
digest is verified **before any claim is checked**:

```
DatasetError: dataset 'affinity': data/experimental_affinity.csv is not the
  file that was checked in.
    declared sha256 b1dc99b64502928f...
    actual   sha256 3f7a01c9be55e112...
  A manually imported file that changed without anybody saying so is the one
  thing nothing else here can detect. If the new file is correct, update
  `sha256` in context.toml and say in the commit what changed.
```

A hard error, not a warning — a warning about a silently changed input is one
people read after they have published. `origin` has no default for the same
reason: the two kinds need different evidence and a default would pick the
weaker one.

### The join cannot go wrong quietly either

Duplicate keys are **refused, not resolved**. Ten rows against ten sharing a key
is a hundred pairs and none of them was measured together. Coverage travels with
the verdict, so a correlation over a 40%-matched join says so.

## What each module is

| | |
|---|---|
| `relate.py` | recompute a claimed relationship and return holds / refuted / cannot tell. Stdlib only |
| `croissant.py` | emit the dataset description, carrying units and verdicts. Stdlib only |
| `datasets.py` | load and verify each dataset; refuse a changed manual import or a fan-out join. Stdlib only |
| `graph.py` | join the column description to the claims and emit Turtle. Needs `rdflib` |

Four modules. Everything else is Snakemake's.

## The test suites

```bash
pytest                              # all of it, ~16s
```

| | |
|---|---|
| `test_dependencies.py` | every dependency needed, every import declared, the core stdlib-only, no run receipts |
| `test_datasets.py` | the manual import and the join, **each guard watched failing** |
| `test_relate.py` | the three verdicts, on data with a known answer |
| `test_graph.py` | the rdflib layer, as the four questions a reader asks — and that the rows are *not* in the graph |
| `test_end_to_end.py` | the example through Snakemake in a clean copy, Croissant validated by `mlcroissant` |
| `test_snakemake_rules.py` | runs **Snakemake's own generated per-rule suite**, and fails if a rule has no generated test |

That last one is the rule applied to testing: `snakemake --generate-unit-tests`
writes one isolated test per rule with its own input fixtures. We did not write
them and do not maintain them. We run them, and we notice when they go stale.

## Two things worth knowing

**Units live here, not in the Croissant.** Croissant 1.1 has no unit
mechanism — checked against the downloaded spec: no `unitText`, `unitCode`,
`PropertyValue`, `variableMeasured`, no QUDT. They are UDUNITS-2 strings,
because CF Conventions has required exactly that of every dimensional quantity
for thirty years, and they ride into the output as annotations.

**The metadata rules take the data as an input** even though they read its path
from `context.toml`. Snakemake records a digest for every *input* and none for
its outputs, so terminal outputs — the published ones — would otherwise have no
recorded hash. Consuming the dataset puts it in Snakemake's own store. That is
the entire receipt mechanism, delegated, and
`test_snakemake_recorded_the_dataset_digest_without_us_writing_one` checks it.

## What is not here yet

* **Signing.** Nothing is cryptographically bound to anybody. in-toto over DSSE
  is the answer and is not built.
* **PDF ingestion** — turning a claim in a paper into a checkable expectation
  on your data. It is the natural completion of "recompute, never believe".
* **PSDI/DCAT conformance.** Deliberately absent, with `pyshacl`, until it is
  decided.
