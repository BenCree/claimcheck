# The last four rules of every claimcheck project, defined once.
#
# Included, not copied:
#
#     import sys
#     sys.path.insert(0, "/home/ben/code/claimcheck")
#     from claimcheck.workflow import RULES
#     include: RULES
#
# EVERY DECLARED DATASET IS AN INPUT, and the list comes from the project's own
# `context.toml`. Consuming a file is what puts its sha256 into Snakemake's
# metadata store, which records a digest for every input and none for its
# outputs, so listing the datasets here is the whole receipt mechanism. Deriving
# the list rather than writing it out means a dataset added to the declaration
# is a dependency of the verdict from that moment, with no second place to
# update and no way for the two to disagree.

import os as _os
import sys as _sys
from pathlib import Path as _Path

import claimcheck.workflow as _cw
from claimcheck.workflow import declared_files as _declared_files

#: The including project. `workflow.basedir` is the directory of the top-level
#: Snakefile, which is the project; `workflow.current_basedir` would be this
#: file's own directory inside the package, which is not what the rules want.
_PROJECT = _Path(workflow.basedir)

#: The interpreter running Snakemake may not be the one that can import
#: claimcheck, so the shell commands carry the path explicitly rather than
#: relying on the environment they happen to inherit.
# From the imported module, not from `__file__`: an included .smk does not
# reliably define `__file__`, and guessing the repository root from a path that
# may not exist is how a shell command ends up with an empty PYTHONPATH.
_CLAIMCHECK = str(_Path(_cw.__file__).parents[2])

# Not `sys.executable`. See `claimcheck.workflow.interpreter`: the environment
# running Snakemake frequently cannot import rdflib or mlcroissant, and the
# failure lands three rules later on a machine nobody is watching. Projects can
# override with CLAIMCHECK_PY or by setting PY before the include.
#
# CLAIMCHECK_PY FIRST, not the PY global (2026-10-21). The order here used to be
# PY then CLAIMCHECK_PY, which contradicted both the comment above and the
# documented order in `claimcheck.workflow.interpreter`, and left a project no
# way to override an inherited PY. It is not a preference: a project whose own
# rules need a Python 3.10 environment (an in-place CUDA build that pins it) got
# that interpreter here too, and every claimcheck module imports `tomllib`,
# which is 3.11+, so `check_claims` died with ModuleNotFoundError: tomllib on a
# project whose data were complete. The only escape was editing PY, which
# changes every rule's shell command and makes Snakemake discard finished
# outputs. An explicit environment variable is the narrower instrument and now
# takes precedence.
_PY = _os.environ.get("CLAIMCHECK_PY") or globals().get("PY") or _cw.interpreter()
_ENV = f"PYTHONPATH={_CLAIMCHECK}"

#: Declared datasets, as rule inputs, PROJECT-RELATIVE.
#:
#: Not absolute. Snakemake matches an input against a rule's output as written,
#: so an absolute input never matches the relative `output: "data/x.csv"` that
#: the project's own rule declares, and the DAG breaks with a
#: MissingInputException naming a file the workflow knows perfectly well how to
#: build. It only shows up when the file is not already on disk, which is why it
#: survived seventeen conversions and appeared the first time a project was
#: built from clean.
#:
#: A project with no declared datasets still adjudicates: relate reads
#: context.toml regardless and returns `unverifiable`, which is a verdict and
#: belongs on the record.
CLAIMCHECK_DATA = list(_declared_files(_PROJECT))


rule check_claims:
    """Recompute every registered claim and return a verdict for each.

    Not a description of the claim: the number is computed again from the files
    below, with an interval, and the verdict is whether the claimed value lies
    inside it. A claim outside a wide interval is `unverifiable` rather than
    confirmed.
    """
    input:
        data=CLAIMCHECK_DATA,
        context=ancient(_PROJECT / "context.toml"),
    output:
        "results/claims.json",
    # ONE FIGURE PER CLAIM BY DEFAULT, as a side effect and not a declared output. A project that defines its own `claim_figures` rule with
    # `output: directory("figures/claims")` would otherwise have two rules producing one path (AmbiguousRuleException). The figure step cannot
    # fail the rule: Snakemake deletes the outputs of a failed job, and the verdicts are the product; a failure prints a warning instead.
    shell:
        f"{_ENV} {_PY} -m claimcheck.relate . {{output}} && "
        f"({_ENV} {_PY} -m claimcheck.figures . figures/claims || "
        f"echo 'WARNING: claimcheck.figures failed; results/claims.json is written, figures/claims is not' >&2)"


rule belief:
    """What else is in doubt once something has been refuted.

    Consumes the verdicts, because a belief whose status was recomputed cannot
    be reported from the declaration alone.
    """
    input:
        claims="results/claims.json",
        context=ancient(_PROJECT / "context.toml"),
    output:
        "results/belief.json",
    shell:
        f"{_ENV} {_PY} -m claimcheck.belief . {{output}}"


rule croissant:
    """The data, in the format a machine outside this repository can read."""
    input:
        data=CLAIMCHECK_DATA,
        claims="results/claims.json",
        context=ancient(_PROJECT / "context.toml"),
    output:
        "results/croissant.json",
    shell:
        f"{_ENV} {_PY} -m claimcheck.croissant . {{output}}"


rule graph:
    """The claims and their provenance as RDF."""
    input:
        croissant="results/croissant.json",
        claims="results/claims.json",
        context=ancient(_PROJECT / "context.toml"),
    output:
        "results/graph.ttl",
    shell:
        f"{_ENV} {_PY} -m claimcheck.graph . {{output}}"


rule dcat:
    """The catalogue record. Adding a format is a rule and a name: `emit` looks
    the name up in a registry, so a third party can add one under the entry-point
    group and it works here with no change to the package."""
    input:
        data=CLAIMCHECK_DATA,
        claims="results/claims.json",
        context=ancient(_PROJECT / "context.toml"),
    output:
        "results/dcat.ttl",
    shell:
        f"{_ENV} {_PY} -m claimcheck.emit dcat . {{output}}"


#: Everything a project should reach. A project's own `rule all` adds this to
#: its inputs, so "the experiment tested its claim and published the result" is
#: one name rather than five paths repeated per project.
CLAIMCHECK_OUTPUTS = [
    "results/claims.json",
    "results/belief.json",
    "results/croissant.json",
    "results/graph.ttl",
    "results/dcat.ttl",
]

