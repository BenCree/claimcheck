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
        # `ancient()`: the modification time of the declaration does not decide
        # whether a verdict is stale. WHICH FIELDS IT HOLDS DOES, and that is
        # the `params` digest below. Dropping `ancient()` instead would
        # re-adjudicate on every edit to the file, including the `evidence.note`
        # this lab adds constantly, and would do nothing at all under the
        # `--rerun-triggers code input params software-env` this lab uses so a
        # script edit cannot delete a running campaign's outputs
        # (projects/z-ml/experiments/complex_panel/LOG.md, decision 3).
        context=ancient(_PROJECT / "context.toml"),
    output:
        "results/claims.json",
    params:
        # EDITING A CLAIMED VALUE RE-ADJUDICATES THE CLAIM. Until 2026-11-22 it
        # did not: `ancient()` above hid the edit, so `results/claims.json` kept
        # a verdict computed from the previous declaration while carrying a
        # `context_sha256` that no longer matched the file beside it.
        #
        # `params` is one of Snakemake's default rerun triggers, recorded per
        # output and compared on the next parse, so `claimed`, `method`,
        # `resampling_unit`, `x`, `y`, `baseline` and every other key an
        # estimator reads retrigger, while an `evidence.note`, a title, a
        # description and a comment do not. `claimcheck.workflow.verdict_digest`
        # says why that list is a denylist and not an allowlist.
        #
        # NOT REFERENCED BY THE SHELL COMMAND BELOW, which is not an oversight:
        # its only job is to be recorded and compared. Delete it and the defect
        # returns, which `tests/test_context_retrigger.py` demonstrates by
        # pinning it to a constant and watching an edited claim go unnoticed.
        verdict_fields=_cw.verdict_digest(_PROJECT),
    # ONE FIGURE PER CLAIM BY DEFAULT, as a side effect and not a declared output. A project that defines its own `claim_figures` rule with
    # `output: directory("figures/claims")` would otherwise have two rules producing one path (AmbiguousRuleException). The figure step cannot
    # fail the rule: Snakemake deletes the outputs of a failed job, and the verdicts are the product; a failure prints a warning instead.
    #
    # THEN THE VERDICTS ARE SCRUTINISED, in the same rule and with the same licence to fail: a claim that could not have failed, a control that
    # came back refuted, and a number in this directory's own prose that contradicts the row it quotes (`claimcheck.scrutiny`). `relate` already
    # prints the first two from the rows; this step is what reads the prose beside them, which needs the files and so cannot be done in `relate`.
    # It changes no verdict, it never fails the rule, and it is here rather than on a person's checklist because a check that fires is worth more
    # than a reader who is not always there (2026-11-17, after all three shapes were found by hand).
    shell:
        f"{_ENV} {_PY} -m claimcheck.relate . {{output}} && "
        f"({_ENV} {_PY} -m claimcheck.figures . figures/claims || "
        f"echo 'WARNING: claimcheck.figures failed; results/claims.json is written, figures/claims is not' >&2) && "
        f"({_ENV} {_PY} -m claimcheck.scrutiny . || "
        f"echo 'WARNING: claimcheck.scrutiny failed; the verdicts stand unscrutinised' >&2)"


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
    """The claims and their provenance as RDF.

    AFTER THE GRAPH IS WRITTEN, an optional site hook runs: whatever is in the
    environment variable `CLAIMCHECK_AFTER_RESULT`. A result is finished when it
    has entered the corpus, and in a lab with a dashboard that means the page
    showing the corpus is stale the moment this rule finishes. Rather than every
    project remembering to refresh it, the refresh is named once in the
    environment and happens here.

    It is an environment variable and not a path in this package because
    claimcheck must install and run for a stranger with no dashboard, no lab
    layout and no cluster. The hook is given the project directory as its one
    argument, it is run from that directory, and its FAILURE IS NOT THIS RULE'S
    FAILURE: a page that will not rebuild must never cost a project its graph.
    The exit status is reported and discarded.
    """
    input:
        croissant="results/croissant.json",
        claims="results/claims.json",
        context=ancient(_PROJECT / "context.toml"),
    output:
        "results/graph.ttl",
    shell:
        f"{_ENV} {_PY} -m claimcheck.graph . {{output}} && "
        # Snakemake formats the whole shell string, so a literal shell brace must be DOUBLED or it is read as a
        # format field and the DAG dies with `NameError: The name 'CLAIMCHECK_AFTER_RESULT' is unknown in this
        # context`, in every project that includes these rules (2026-11-13).
        # DEFAULT ON, not opt-in (owner, 2026-11-13: "dashboard as default", "need to
        # auto update dashboard"). With no environment variable set, a conventional
        # path is used, and a machine with no such file simply has no hook. That keeps
        # the package installable by a stranger with no dashboard while making the
        # refresh automatic here, which an environment variable never was: it had to
        # be exported into every shell that ever ran Snakemake, and so it never fired.
        '_hook="${{CLAIMCHECK_AFTER_RESULT:-$HOME/.config/claimcheck/after_result}}"; '
        'if [ -x "$_hook" ]; then '
        '  echo "after-result hook: $_hook"; '
        '  "$_hook" "$PWD" '
        '    || echo "after-result hook failed (exit $?); the graph above is written and valid"; '
        "fi"


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

