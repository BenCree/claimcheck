"""The dependency set is a claim, and this is what checks it.

This repository exists because its predecessor grew two answers to the same
question: roughly 1,750 lines implementing things Snakemake, `mlcroissant` and
`runcrate` already did. Nothing caught that, because nothing was watching for
it — a dependency that stops being needed leaves no failing test behind, and a
module that reimplements a dependency leaves none either.

So four properties are enforced here:

1. **Every declared dependency is imported by something.** Otherwise it is
   carried weight, and the reason it was added has been forgotten.
2. **Every third-party import is declared.** The converse, and the one with a
   recorded incident: the predecessor called `rocrate_validator` from a code
   path whose extra did not list `roc-validator`, so a clean
   `pip install …[publish]` ended in `ModuleNotFoundError`.
3. **The core needs nothing but the standard library** — enforced by blocking
   every third-party import and then running it, not by reading the imports.
4. **Nothing here writes a run receipt.** Snakemake's metadata record already
   is one — input digests as sha256, the code, the shell command, the conda
   environment, a software-stack hash, the times. Rewriting it was the single
   largest piece of duplication in the predecessor, so it is the one case of
   "Snakemake already does this" that is mechanically refused.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "claimcheck"

#: Import name -> distribution name, where they differ. Kept tiny on purpose:
#: a long map here means the dependency set has grown past what one person can
#: hold in their head, which is the condition this whole file exists to detect.
_DIST = {"yaml": "pyyaml", "rocrate_validator": "roc-validator"}

#: Modules that ship with Python. `sys.stdlib_module_names` is authoritative and
#: version-correct, so no hand-maintained list can drift out of date against it.
_STDLIB = set(sys.stdlib_module_names)


def _imports(path: Path) -> set[str]:
    """Top-level package names imported by one file, including inside functions.

    `ast.walk`, deliberately — a lazy import inside a function is still a
    dependency, and the predecessor hid two of them that way.
    """
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _third_party(paths) -> dict[str, set[str]]:
    """package -> the files importing it, for non-stdlib, non-local imports."""
    out: dict[str, set[str]] = {}
    for p in paths:
        for name in _imports(p):
            if name in _STDLIB or name == "claimcheck":
                continue
            out.setdefault(_DIST.get(name, name), set()).add(
                str(p.relative_to(REPO)))
    return out


def _declared() -> dict[str, set[str]]:
    """extra -> distribution names, from pyproject. Version pins stripped."""
    d = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    out = {"": {r.split(">")[0].split("=")[0].split("[")[0].strip()
                for r in d.get("dependencies", [])}}
    for extra, reqs in d.get("optional-dependencies", {}).items():
        out[extra] = {r.split(">")[0].split("=")[0].split("[")[0].strip()
                      for r in reqs}
    return out


# ---------------------------------------------------------------------------
# 1 and 2 — the two directions
# ---------------------------------------------------------------------------

def test_every_declared_dependency_is_actually_imported():
    """A dependency nobody imports is weight whose reason has been forgotten.

    `mlcroissant` and `snakemake` are the deliberate exceptions and are named
    here rather than skipped silently: both are COMMANDS this package runs,
    never libraries it imports.
    """
    commands = {"mlcroissant", "snakemake", "pytest"}
    used = set(_third_party(sorted(REPO.rglob("*.py"))))
    unused = []
    for extra, reqs in _declared().items():
        for r in reqs:
            if r in commands or r in used:
                continue
            unused.append(f"{r} (declared in {extra or 'dependencies'})")
    assert not unused, (
        "declared but imported by nothing — delete it, or say in pyproject why "
        f"it is a command rather than a library: {unused}")


def test_every_third_party_import_is_declared():
    """The recorded incident, in the other direction. `rocrate_validator` was
    called from a code path whose extra did not list `roc-validator`, so a
    clean install of the published package raised `ModuleNotFoundError` — and
    the suite never noticed, because the development machine had it."""
    declared = set().union(*_declared().values())
    undeclared = {pkg: sorted(files)
                  for pkg, files in _third_party(sorted(PKG.rglob("*.py"))).items()
                  if pkg not in declared}
    assert not undeclared, f"imported but not declared: {undeclared}"


def test_the_command_line_tools_are_declared_as_such():
    """`mlcroissant` and `snakemake` are exempted from the check above, so the
    exemption must be visible in the file rather than only in this test."""
    text = (REPO / "pyproject.toml").read_text()
    assert "is a COMMAND, not an import" in text
    for tool in ("mlcroissant", "snakemake"):
        assert tool in text, f"{tool} is exempted but never mentioned"


# ---------------------------------------------------------------------------
# 3 — the core, verified by running it with everything else blocked
# ---------------------------------------------------------------------------

def test_the_core_needs_nothing_but_the_standard_library():
    """Blocked at the import hook and then RUN, not read.

    Reading the imports would pass on a module that imports lazily. This
    installs a meta-path finder that refuses every non-stdlib package and then
    checks a real dataset, so a hidden import fails the way it would fail on a
    cluster node with a bare Python.

    `belief` is here for the sharpest version of that reason: the question it
    answers -- *what else is in doubt?* -- gets asked when something has
    already gone wrong, which is exactly when the optional extras are what is
    missing. It is built from the verdicts computed two lines above rather than
    from a file, because `example/results/` is gitignored and a probe that read
    it would examine nothing on a fresh clone.
    """
    probe = f'''
import sys, tomllib
ALLOWED = set(sys.stdlib_module_names) | {{"claimcheck"}}
class Ban:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in ALLOWED:
            raise ImportError("this must not be needed: " + name)
        return None
sys.meta_path.insert(0, Ban())
sys.path.insert(0, {str(REPO)!r})
from claimcheck import belief, croissant, relate
d = relate.check({str(REPO / "example")!r})
assert d["relationships"], "the probe checked nothing"
c = croissant.build({str(REPO / "example")!r})
assert c["recordSet"][0]["field"], "the probe described nothing"
ctx = tomllib.loads(open({str(REPO / "example" / "context.toml")!r}).read())
b = belief.BeliefGraph.from_dict(
    ctx["belief"],
    recomputed={{f"{{r['x']}} ~ {{r['y']}}": r for r in d["relationships"]}})
assert b.caveats_on("mace"), "the probe doubted nothing"
print("OK", d["verdicts"])
'''
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-1500:]
    assert "OK" in r.stdout


def test_only_modules_that_declare_the_extra_reach_for_it():
    """An extra's package may be imported only by a module that declares it.

    Importing one anywhere else makes the core install broken in a way the
    stdlib probe above cannot see — that probe imports only the core modules.

    DERIVED FROM THE REGISTRY, not a filename, and not from one hardcoded
    package either. The first version named `graph.py`, and adding `dcat.py` — a
    second legitimate rdflib user — made it fail for being right; the second
    version asked only about `extra == "graph"`, and `psdi.py`, whose extra
    requires rdflib *and* pyshacl, made it fail for being right again. A rule
    that has to be edited every time the thing it describes grows is a rule that
    will be edited wrongly, so it now reads both halves of the registry:
    `_BUILTIN` says which module needs which extra, `_EXTRA_PROBE` says which
    packages that extra brings.
    """
    from claimcheck.emit import _BUILTIN, _EXTRA_PROBE

    #: package -> the module filenames permitted to import it.
    allowed: dict[str, set[str]] = {}
    for mod, extra, _ in _BUILTIN.values():
        for pkg in _EXTRA_PROBE.get(extra or "", ()):
            allowed.setdefault(pkg, set()).add(mod.rsplit(".", 1)[-1] + ".py")

    offenders = {}
    for p in PKG.rglob("*.py"):
        for pkg in _imports(p) & set(allowed):
            if p.name not in allowed[pkg]:
                offenders.setdefault(pkg, set()).add(str(p.relative_to(REPO)))
    assert not offenders, (
        f"an extra's package is imported by a module that does not declare it: "
        f"{ {k: sorted(v) for k, v in offenders.items()} }. Register the module "
        f"in `emit._BUILTIN` with the extra it needs, or do not import it.")


# ---------------------------------------------------------------------------
# 4 — the specific duplication that created this repository
# ---------------------------------------------------------------------------

def test_nothing_here_writes_a_run_receipt():
    """Snakemake's metadata record already is one.

    Read from a live record rather than the documentation, it carries
    `input_checksums` as sha256, `code`, `shellcmd`, `params`, `conda_env`,
    `software_stack_hash`, `rule`, `starttime` and `endtime`. The predecessor
    wrote its own with substantially the same field list, discovered the overlap
    only during a 227-package survey, and still has both.

    The signature refused here is the one that matters: hashing an INPUT and
    writing it to a sidecar. Hashing an output is fine and `croissant.py` does
    it — that is the one digest Snakemake does NOT record, because the output
    path is its record's filename rather than a field.
    """
    bad = []
    for p in PKG.rglob("*.py"):
        src = p.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            body = ast.get_source_segment(src, node) or ""
            hashes = "sha256" in body or "hashlib" in body
            inputs = "input" in body.lower() and "receipt" in body.lower()
            if hashes and inputs:
                bad.append(f"{p.relative_to(REPO)}::{node.name}")
    assert not bad, (
        "this looks like a run receipt, which Snakemake already writes: "
        f"{bad}. If it is not, rename it so the next reader can tell.")


def test_the_no_receipt_rule_could_actually_fire(tmp_path):
    """Two-sided. A guard nobody has watched fail is a guard nobody should
    trust — and this one is a heuristic over source text, which is exactly the
    kind that quietly matches nothing."""
    offender = tmp_path / "sneaky.py"
    offender.write_text(
        "import hashlib\n\n\n"
        "def write_receipt(inputs, out):\n"
        "    h = hashlib.sha256()\n"
        "    for i in inputs:\n"
        "        h.update(i.read_bytes())\n"
        "    out.write_text(h.hexdigest())\n")
    src = offender.read_text()
    hit = False
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef):
            body = ast.get_source_segment(src, node) or ""
            if ("sha256" in body or "hashlib" in body) and \
                    "input" in body.lower() and "receipt" in body.lower():
                hit = True
    assert hit, "the receipt detector matches nothing, so it protects nothing"
