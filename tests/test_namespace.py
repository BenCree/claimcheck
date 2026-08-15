"""The IRI base: one definition, and an honest statement about resolving.

Two failures this prevents, both of which the predecessor recorded:

* **A constant copied into three modules.** The base was
  `https://w3id.org/provchem/two-layer#` — another project's name — written out
  in `croissant.py`, `graph.py` and the tests. Three copies are three chances
  for two of them to disagree, and it appeared in every artefact emitted.
* **An IRI silently claiming to resolve.** `w3id.org` redirects are registered
  by pull request and nobody has filed one, so these identify without
  dereferencing. That is legal and common; the defect is not saying so. The
  predecessor's `R-PUB-6` is the same rule: *every IRI published either
  dereferences or is documented as not dereferencing at the point of
  publication.*
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from claimcheck.namespace import BASE, NS, RESOLVES

REPO = Path(__file__).resolve().parents[1]


def test_nothing_writes_the_base_out_by_hand():
    """Only `namespace.py` may contain the literal. Everything else imports it."""
    offenders = []
    for p in (REPO / "claimcheck").rglob("*.py"):
        if p.name == "namespace.py":
            continue
        if "w3id.org/claimcheck" in p.read_text():
            offenders.append(p.name)
    assert not offenders, (
        f"{offenders} write the IRI base literally; import it from "
        f"`claimcheck.namespace` so the two cannot drift")


#: The base this code was copied in with — ASSEMBLED, not written whole.
#:
#: Written as one literal it appears in this file as code, and `_code_only`
#: below then correctly reports this very file as an offender. The alternative
#: was a named exclusion for the file that defines the pattern, which is a hole;
#: splitting the string leaves the guard with no exclusions at all.
#:
#: This is the third version. The first matched the project's name and fired on
#: the docstring explaining the rename; the second matched the IRI and fired on
#: `namespace.py`, whose job is to record what the base used to be.
_OLD_BASE = "w3id.org/" + "prov" + "chem/"


def _code_only(path: Path) -> str:
    """Source with docstrings and comments removed.

    TWO EARLIER VERSIONS OF THIS GUARD FIRED ON THEIR OWN DOCUMENTATION. The
    first searched for the project's name and matched the docstring explaining
    the rename; the second matched the old IRI inside `namespace.py`, whose
    whole job is to record what the base used to be. Excluding files by name
    would be a hole that grows.

    A prose mention is not a minted IRI, so the prose is removed before the
    match. This is `code_only` from the predecessor, which exists because the
    same defect happened there four times.
    """
    import ast
    import io
    import tokenize
    src = path.read_text()
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docstrings.add(d)
    kept = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and tok.string.strip("\"'bruBRU") in docstrings:
            continue
        kept.append(tok.string)
    return " ".join(kept)


def test_no_artefact_still_carries_the_old_projects_iri():
    """The base named a different project for as long as the code was a copy,
    and it appeared in every Croissant and every graph emitted."""
    stale = [str(p.relative_to(REPO))
             for p in list((REPO / "claimcheck").rglob("*.py"))
             + list((REPO / "tests").rglob("*.py"))
             if _OLD_BASE in _code_only(p)]
    assert not stale, f"still minting IRIs under the old base: {stale}"


def test_that_guard_can_fire(tmp_path):
    """Two-sided, and BOTH sides matter here: it must catch a real assignment
    and must not catch a docstring, which is how the first two versions of it
    broke."""
    real = tmp_path / "real.py"
    real.write_text(f'NS = "https://{_OLD_BASE}two-layer#"\n')
    assert _OLD_BASE in _code_only(real)

    prose = tmp_path / "prose.py"
    prose.write_text(f'"""It used to be https://{_OLD_BASE}two-layer#."""\n'
                     f'# and also https://{_OLD_BASE}old in a comment\n'
                     f'NS = "https://w3id.org/claimcheck/ns#"\n')
    assert _OLD_BASE not in _code_only(prose), "a mention is not a minted IRI"


def test_the_base_is_well_formed():
    assert BASE.startswith("https://") and BASE.endswith("/")
    assert NS.startswith(BASE) and NS.endswith("#")
    assert re.fullmatch(r"[a-z0-9/.:#-]+", NS), NS


def test_whether_it_resolves_is_recorded_not_assumed():
    """`RESOLVES` is False and says why in the module docstring. When somebody
    registers the redirect, flipping it makes the next test start requiring the
    IRIs to actually answer — so the flag cannot be set optimistically."""
    doc = (REPO / "claimcheck" / "namespace.py").read_text()
    assert "DOES NOT DEREFERENCE" in doc
    assert "perma-id/w3id.org" in doc, "say how to make it resolve"
    assert RESOLVES is False, "if this is now True, the test below applies"


@pytest.mark.skipif(not RESOLVES, reason="the redirect is not registered yet")
def test_the_iris_actually_resolve():
    """Dormant until `RESOLVES` is flipped, and then it is the check that stops
    the flag being a wish."""
    import urllib.request
    with urllib.request.urlopen(BASE, timeout=15) as r:
        assert r.status == 200
