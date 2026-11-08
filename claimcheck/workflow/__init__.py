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

__all__ = ["RULES", "declared_files", "interpreter"]

#: Absolute path to the shared rules, for `include:`.
RULES = str(Path(__file__).with_name("claims.smk"))


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
