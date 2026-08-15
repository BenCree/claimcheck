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

## What each module is

| | |
|---|---|
| `relate.py` | recompute a claimed relationship and return holds / refuted / cannot tell. Stdlib only |
| `croissant.py` | emit the dataset description, carrying units and verdicts. Stdlib only |
| `graph.py` | join the column description to the claims and emit Turtle. Needs `rdflib` |

Three modules. Everything else is Snakemake's.

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
