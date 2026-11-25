"""The Snakemake rules every claimcheck project needs, defined once.

Every experiment in a claimcheck project exists to test a claim, so adjudicating
that claim and publishing the result is not per-project boilerplate: it is the
part that does not vary. It was nonetheless being copied. `claimcheck.scaffold`
writes the four rules into each new project's Snakefile, so 68 projects in one
repository carry 68 near-identical copies, and a fix to any of them reaches
none of the others.

Use it from a project's Snakefile:

    import sys
    sys.path.insert(0, "/home/ben/code/claimcheck")
    from claimcheck.workflow import RULES
    include: RULES

That is three lines against about fifty, and the rules' inputs come from the
project's own `context.toml` rather than being listed by hand, so a dataset
added to the declaration is automatically a dependency of the verdict.

`declared_files` is the function that does it, and it is public because the
inputs of a rule are a claim about what the result depends on. A reader should
be able to check that claim without reading the rules.
"""

from pathlib import Path

__all__ = ["RULES", "NON_VERDICT_SECTIONS", "PROSE_KEYS", "declared_files",
           "interpreter", "verdict_digest"]

#: Absolute path to the shared rules, for `include:`.
RULES = str(Path(__file__).with_name("claims.smk"))

#: The only tables of `context.toml` left OUT of the digest, because each is the
#: input of its own rule: `belief` and `jobs` are consumed by rules that take
#: `results/claims.json` as an input and so are recomputed whenever a verdict is.
#:
#: EVERYTHING ELSE IS IN, AND THAT IS DELIBERATE. Naming the four tables `relate`
#: reads today (`project`, `datasets`, `variables`, `relationships`) would be an
#: allowlist, and a table added later would then be silently undigested, which is
#: the defect this digest exists to close. The cost is already visible: a
#: `[units]` table appears in two of the tree's declarations (for example
#: `/home/ben/code/projects/water-network/experiments/mace_polar_charge_solvation_funnel/context.toml`),
#: nothing in this package reads it, and it is digested anyway.
NON_VERDICT_SECTIONS = frozenset({"belief", "jobs"})

#: Keys inside the digested tables that NO computation reads: prose for a person.
#:
#: A DENYLIST AND NOT AN ALLOWLIST, which is the whole safety argument. An
#: estimator added to `relate` brings new claim keys with it (`alpha`,
#: `tie_tol`, `baseline`, `group`, `interval`, `event` each arrived this way),
#: and a key nobody listed here is digested, so a claim registered with it
#: retriggers its own adjudication. An allowlist would silently fail to, which
#: is the defect this digest exists to close.
#:
#: `evidence.source` is DELIBERATELY NOT HERE, though it is prose: a control
#: label at the start of it is what `claimcheck.scrutiny` matches to raise
#: `refuted_control` (`claimcheck/scrutiny.py:402`), and that warning travels in
#: `results/claims.json`. `evidence.note` is here, and it is the key this lab
#: edits constantly.
PROSE_KEYS = frozenset({"note", "notes", "comment", "comments", "description",
                        "title"})


def declared_files(root: Path | str) -> list[str]:
    """Every non-`results/` file the project's `context.toml` declares.

    These are exactly the files `claimcheck.relate` will open, so they are the
    inputs of the rule that produces the verdicts. Deriving them rather than
    listing them is what stops a Snakefile and a context.toml disagreeing about
    what a result was computed from, which is the disagreement this package
    exists to make impossible.

    `results/` paths are excluded because they are outputs of these same rules;
    including them would be a cycle.
    """
    import tomllib

    root = Path(root)
    ctx = tomllib.loads((root / "context.toml").read_text())
    out: list[str] = []
    for spec in ctx.get("datasets", []):
        for f in spec.get("files", []):
            if not str(f).startswith("results/") and f not in out:
                out.append(f)
    return out


def _without_prose(value):
    """`value` with every prose key dropped, at any depth."""
    if isinstance(value, dict):
        return {k: _without_prose(v) for k, v in value.items()
                if k not in PROSE_KEYS}
    if isinstance(value, list):
        return [_without_prose(v) for v in value]
    return value


def verdict_digest(root: Path | str) -> str:
    """What a verdict is computed from, as one token, for a rerun trigger.

    EDITING A CLAIMED VALUE MUST RE-ADJUDICATE THE CLAIM. It did not. The shared
    `check_claims` rule takes `context.toml` as `ancient()`, which tells
    Snakemake to ignore its modification time, so changing `claimed`, `method`,
    `resampling_unit`, `x`, `y` or `baseline` left `results/claims.json` in
    place: a verdict computed from the previous declaration, carrying a
    `context_sha256` that no longer matched the file beside it. That breaks the
    one link the field exists to certify, and it defeated the legitimate case
    the rules REQUIRE, where a badly designed claim is kept and a dated
    replacement registered beside it.

    Removing `ancient()` would have fixed it at a price nobody would pay: every
    edit to the declaration would re-adjudicate, and the edit this lab makes
    most often is an `evidence.note`, which changes no number. A rule that
    re-runs a 49-minute adjudication for a sentence of prose is a rule somebody
    turns off. Worse, it would not even work where this lab needs it most:
    campaigns here run with `--rerun-triggers code input params software-env`,
    which drops mtime so a script edit cannot delete a running campaign's
    outputs, and under that set an mtime edge is dead. Measured on
    `/home/ben/code/projects/affinity-scoring/experiments/a71_scorer_cost`,
    2026-11-22: the param fires with mtime dropped and does not fire under
    `--rerun-triggers mtime`, which is the one trigger set that defeats it.

    So the trigger is this digest, declared as a `params` value of
    `check_claims`: `params` is one of Snakemake's default rerun triggers, it is
    compared against the value recorded in the metadata store for the output,
    and `ancient()` stays, so the modification time still does not count.

    It is the sha256 of the declaration with `NON_VERDICT_SECTIONS` and
    `PROSE_KEYS` dropped, as canonical JSON, to 16 hex characters, which is the
    width this package uses for `relate_sha256` and `context_sha256`. Parsing
    rather than digesting the bytes is also what makes a comment, a blank line
    and a reordered key free: `tomllib` never reports them.

    Returned with its name on it, because the value is read by whoever opens
    `.snakemake/metadata/<output>` and a bare hash there says nothing.
    """
    import hashlib
    import json
    import tomllib

    ctx = tomllib.loads((Path(root) / "context.toml").read_text())
    body = {k: _without_prose(v) for k, v in ctx.items()
            if k not in NON_VERDICT_SECTIONS}
    # `default=str`: TOML has dates and `retrieved = 2026-08-15` is a real one.
    # `sort_keys`: a reordered table is not a different declaration.
    canonical = json.dumps(body, sort_keys=True, default=str,
                           separators=(",", ":"))
    return ("context-verdict-fields:"
            + hashlib.sha256(canonical.encode()).hexdigest()[:16])


#: Modules the emitters need and `relate` does not. `graph`, `dcat` and
#: `croissant` are the package's optional extras, so the interpreter running
#: Snakemake is often NOT the one that can run them: a workflow launched from an
#: environment holding only snakemake produces verdicts and then fails on the
#: first emit with `ModuleNotFoundError: rdflib`.
_EMIT_REQUIRES = ("rdflib", "mlcroissant")


def interpreter(candidates: list[str] | None = None) -> str:
    """The interpreter that can run every claimcheck job, not just some of them.

    Order: `CLAIMCHECK_PY` if set, then the running interpreter, then any
    candidate given. The first that imports the emit extras wins; if none does,
    the running interpreter is returned anyway so `relate` still works and the
    emit rules fail with their own clear error rather than being silently
    skipped.

    Chosen at workflow parse time rather than hardcoded in each project, because
    a path written into 68 Snakefiles is 68 places to edit when an environment
    is rebuilt.
    """
    import os
    import subprocess
    import sys

    tried = []
    for py in ([os.environ.get("CLAIMCHECK_PY"), sys.executable]
               + list(candidates or []) + _siblings()):
        if not py or py in tried:
            continue
        tried.append(py)
        code = "import " + ", ".join(_EMIT_REQUIRES)
        try:
            r = subprocess.run([py, "-c", code], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            return py
    return sys.executable


def _siblings() -> list[str]:
    """Other interpreters installed beside the running one.

    A conda or venv layout puts environments side by side, so when the
    interpreter running Snakemake cannot import the emit extras, the one that
    can is usually two directories away. Discovering them beats a hardcoded list:
    a path written into a workflow file is one more thing to edit when an
    environment is rebuilt, and it makes the file unusable on anyone else's
    machine.

    Ordered by name for a stable answer, and the running interpreter's own
    environment is skipped because it has already been tried.
    """
    import sys

    here = Path(sys.executable).resolve()
    out: list[str] = []
    for envs in (here.parent.parent.parent, here.parent.parent.parent.parent):
        if not envs.is_dir():
            continue
        for cand in sorted(envs.glob("*/bin/python")):
            if cand.resolve() != here and str(cand) not in out:
                out.append(str(cand))
    return out
