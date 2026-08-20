"""PSDI's profile: the shapes are pinned, the exemptions are narrow, it can fail.

The decision this file records is that PSDI conformance was **adopted with its
cost**: `pyshacl` and 68 kB of vendored SHACL, in exchange for the only external
format here whose conformance is checkable against shapes its own maintainers
wrote. Everything else this package emits is checked against a validator we
chose; this one is checked against the catalogue's own rules.

Three things are tested and each has a recorded failure behind it:

* **the shapes are the pinned bytes.** A conformance report against a moving
  target does not say what it means — upstream's changelog records SHACL
  updates on three days in one month.
* **the exemptions cover absence and nothing else.** Keyed on the shape alone,
  the three exemptions also waived Pattern, Class, NodeKind and Datatype on
  those shapes, so a fabricated identifier, a logo URL that 404s and
  `displayPriority = "banana"` all published as "exempt".
* **plain DCAT is a different document**, measured. If that stops being true —
  if PSDI opens the shapes — this file fails rather than a memory going stale.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("rdflib")
pytest.importorskip("pyshacl")

from rdflib import Graph, Literal, URIRef                      # noqa: E402
from rdflib.namespace import DCTERMS, XSD                      # noqa: E402

from claimcheck import dcat, psdi                              # noqa: E402
from claimcheck.namespace import BASE                          # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"


@pytest.fixture
def project(tmp_path):
    """A project declaring everything PSDI needs and nothing it forbids."""
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    (root / "data" / "x.csv").write_text("a,b\n1,2\n")
    (root / "context.toml").write_text(
        '[project]\n'
        'name = "fixture"\n'
        'title = "A fixture dataset"\n'
        'context = "Built by the test suite to exercise PSDI\'s shapes."\n'
        'license = "https://spdx.org/licenses/CC0-1.0.html"\n'
        'version = "1.0"\n'
        'date_published = "2026-08-17"\n'
        'keywords = ["chemistry", "provenance"]\n'
        'publisher = "Nobody in particular"\n\n'
        '[psdi]\n'
        'access_rights = "open"\n'
        'landing_page = "https://example.org/fixture"\n'
        'contact_point = "mailto:nobody@example.org"\n'
        'creators = [{ name = "A Person", orcid = "0000-0002-1825-0097" }]\n\n'
        '[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["data/x.csv"]\n')
    return root


# ---------------------------------------------------------------------------
# the shapes: pinned, present, and the bytes they claim to be
# ---------------------------------------------------------------------------

def test_the_shapes_ship_inside_the_package():
    """Not beside it. The predecessor kept them at the repository root, which is
    site-packages in a wheel, so a clean install died on a path that only ever
    existed in a git clone."""
    assert psdi.SHAPES_FILE.is_file(), psdi.SHAPES_FILE
    assert psdi.SHAPES_FILE.is_relative_to(Path(psdi.__file__).parent)


def test_the_vendored_shapes_are_the_bytes_the_record_claims():
    """Two-sided by construction: change one byte and this fails. A vendored
    file whose digest nobody checks is a file that has silently moved."""
    prov = json.loads((psdi.SHAPES / "PROVENANCE.json").read_text())
    assert prov["upstream_commit"], "the shapes are pinned to nothing"
    for name, want in prov["files"].items():
        p = psdi.SHAPES / name
        assert p.is_file(), f"{name} is recorded and not shipped"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == want["sha256"], (
            f"{name} is not the file PROVENANCE.json describes: recorded "
            f"{want['sha256'][:12]}, on disk {got[:12]}")
        assert p.stat().st_size == want["bytes"]


def test_nothing_is_shipped_that_no_code_opens():
    """The other direction. `validate()` parses the shapes graph and nothing
    else, so a second vendored file would be carried weight — the exact thing
    `test_dependencies.py` refuses for dependencies."""
    prov = json.loads((psdi.SHAPES / "PROVENANCE.json").read_text())
    shipped = {p.name for p in psdi.SHAPES.iterdir() if p.suffix == ".ttl"}
    assert shipped == set(prov["files"]), (shipped, set(prov["files"]))


def test_the_vendored_subset_is_the_whole_shapes_graph():
    """The claim PROVENANCE.json makes about the two files NOT vendored,
    re-measured against an upstream copy when one is on this machine.

    Skipped honestly when it is not: an unverifiable claim stated as verified is
    the failure this package exists to prevent, and a skip that names its reason
    is the only truthful alternative.
    """
    import pyshacl
    # THE PATH HERE USED TO BE A SIBLING REPOSITORY'S COPY, AND THAT REPOSITORY
    # IS GONE (retired and archived 2026-08-20). A test that names a directory
    # nobody can produce skips forever while looking like it might run one day.
    # The two extra graphs are PSDI's, not ours and not the retired project's,
    # so the location is a caller's to supply: set PSDI_UPSTREAM to a checkout
    # of them. `claimcheck/resources/psdi/psdi-dcat-shacl.ttl` is vendored and
    # is what the rest of this file validates against; these two only widen the
    # comparison.
    env = os.environ.get("PSDI_UPSTREAM")
    upstream = Path(env) if env else Path(__file__).parent / "psdi_upstream"
    extra = [upstream / "psdi-dcat-ext.ttl", upstream / "psdi-voc.ttl"]
    if not all(p.is_file() for p in extra):
        pytest.skip(f"no upstream copy at {upstream} to compare against — set "
                    f"PSDI_UPSTREAM to a checkout of PSDI's own graphs")
    data = psdi.build(EXAMPLE)

    def violations(paths):
        s = Graph()
        for f in paths:
            s.parse(f, format="turtle")
        _c, results, _t = pyshacl.validate(data, shacl_graph=s,
                                           inference="none", abort_on_first=False)
        return len(results)

    assert violations([psdi.SHAPES_FILE]) == violations([psdi.SHAPES_FILE, *extra])


# ---------------------------------------------------------------------------
# the record: conformant modulo three named exemptions
# ---------------------------------------------------------------------------

def test_a_fully_declared_project_conforms(project):
    r = psdi.check(project)
    assert not r.crashed, r.detail
    assert r.passed, r.real_violations
    assert len(r.exempt_violations) == 3, r.exempt_violations


def test_the_exemptions_are_exactly_the_three_psdi_issues(project):
    r = psdi.check(project)
    named = {v.split(":", 1)[0] for v in r.exempt_violations}
    assert named == set(psdi.EXEMPT), (named, set(psdi.EXEMPT))
    assert not r.conforms, (
        "the record fully conforms, so the three exemptions are describing "
        "nothing — check whether PSDI now issues identifiers differently")


def test_removing_an_exemption_turns_it_into_a_real_failure(project):
    """Two-sided on the exemption list itself. An exemption that changes no
    verdict is a comment."""
    kept = {k: v for k, v in psdi.EXEMPT.items()
            if k != "IdentifierPropertyShape"}
    r = psdi.validate(psdi.build(project), exempt=kept)
    assert not r.passed
    assert any("Identifier" in v for v in r.real_violations), r.real_violations


def test_an_invented_identifier_is_not_covered_by_the_exemption(project):
    """The recorded defect. The exemption is for a value PSDI has not issued —
    an ABSENT one. Keyed on the shape alone it also waived Pattern, so a made-up
    PSDI URL published under the exemption's own reason."""
    g = psdi.build(project)
    g.add((URIRef(f"{BASE}dataset/fixture"), DCTERMS.identifier,
           Literal("https://resources.psdi.ac.uk/data/definitely-not-a-uuid",
                   datatype=XSD.string)))
    r = psdi.validate(g)
    assert not r.passed, "a fabricated identifier passed as exempt"
    assert any("NOT EXEMPT" in v for v in r.real_violations), r.real_violations
    assert psdi.EXEMPT_CONSTRAINT == "MinCountConstraintComponent"


def test_a_missing_keyword_is_a_real_failure_with_the_fix_in_it(project):
    """Keywords are required and are never invented. Two defaults would have
    made every record conform and would each have been a statement about the
    dataset that nobody made."""
    ctx = project / "context.toml"
    ctx.write_text(ctx.read_text().replace(
        'keywords = ["chemistry", "provenance"]\n', ""))
    r = psdi.check(project)
    assert not r.passed
    hit = [v for v in r.real_violations if "Keyword" in v]
    assert hit and "context.toml" in hit[0], hit


def test_an_undeclarable_access_right_is_refused_not_dropped(project):
    """Silently omitting it leaves the record valid and the declaration
    ignored: the author writes it, the command exits 0, the field is not there."""
    ctx = project / "context.toml"
    ctx.write_text(ctx.read_text().replace('access_rights = "open"',
                                           'access_rights = "public"'))
    with pytest.raises(psdi.UndeclarableValue, match="permits four values"):
        psdi.build(project)


def test_a_contact_name_is_refused_because_the_card_cannot_carry_one(project):
    """Measured against their shapes, not assumed: `VCardNodeShape` is closed
    and permits `hasEmail` and `hasURL` only. A name is the obvious thing to
    add, it makes the record non-conformant, and dropping it silently would
    leave the author believing it was published."""
    ctx = project / "context.toml"
    ctx.write_text(ctx.read_text().replace(
        'contact_point = "mailto:nobody@example.org"',
        'contact_point = "mailto:nobody@example.org"\ncontact_name = "Nobody"'))
    with pytest.raises(psdi.UndeclarableValue, match="sh:closed"):
        psdi.build(project)


def test_the_record_declares_the_profile_it_claims(project):
    g = psdi.build(project)
    assert (None, DCTERMS.conformsTo, psdi.PROFILE) in g, (
        "a record that does not name its profile can only be guessed at")


def test_no_checksum_is_emitted_and_the_reason_is_in_the_file():
    """PSDI permits SHA-1 only; we compute SHA-256. Emitting their algorithm IRI
    beside our digest publishes a false statement to satisfy a validator."""
    text = psdi.render(EXAMPLE)
    assert "checksum" not in text.lower(), (
        "a checksum reached a PSDI record, whose profile permits SHA-1 only")
    assert "SHA-1 is the only" in Path(psdi.__file__).read_text(), (
        "the reason is not written down where the next reader will look")


# ---------------------------------------------------------------------------
# the two documents really are two documents
# ---------------------------------------------------------------------------

def test_plain_dcat_cannot_be_made_to_conform():
    """The measurement that justifies a second module rather than a flag.

    All four PSDI node shapes are `sh:closed`, so our own terms are rejected
    rather than ignored. If this count collapses, the reason for `psdi.py`
    existing separately has gone with it.
    """
    r = psdi.validate(dcat.build(EXAMPLE))
    assert not r.passed
    # 25 on 2026-08-17 with pyshacl 0.40.1, nine of them `sh:closed`. Asserted
    # as a floor rather than as 25 exactly: the number the README quotes is a
    # measurement of one day, and a validator release that reports the same
    # defects differently should not break a test about the SHAPE of the
    # answer. A collapse toward zero is the thing worth catching.
    assert len(r.real_violations) >= 15, len(r.real_violations)
    assert any("is closed" in v for v in r.real_violations), (
        "no sh:closed violation, so plain DCAT's extra terms are being ignored "
        "rather than rejected — the two records could then be one")


def test_the_example_names_exactly_what_it_still_needs():
    """The shipped example is NOT PSDI-conformant, and the report says why in
    one line. Recorded rather than fixed: which keywords `example/` declares is
    an editorial decision, and inventing two to make a checker green is the
    defect this module is written against."""
    r = psdi.check(EXAMPLE)
    assert not r.passed
    assert len(r.real_violations) == 1, r.real_violations
    assert "Keyword" in r.real_violations[0]


# ---------------------------------------------------------------------------
# a crash is not a verdict
# ---------------------------------------------------------------------------

def test_a_validator_crash_is_never_a_pass(monkeypatch, project):
    """`conforms=False` alone reads "your record does not conform" — a sentence
    about the data, when the true one is about the validator."""
    monkeypatch.setattr(psdi, "SHAPES_FILE", Path("/no/such/shapes.ttl"))
    r = psdi.validate(psdi.build(project))
    assert r.crashed and not r.passed
    assert "did not reach a verdict" in r.detail
    assert r.as_dict()["crashed"] is True


def test_the_selftest_passes():
    assert psdi.main(["--selftest"]) == 0


def test_the_emitter_is_registered():
    from claimcheck import emit
    assert "psdi" in emit.available()
    if emit.available()["psdi"]:
        pytest.skip(emit.available()["psdi"])
    text = emit.emit("psdi", EXAMPLE)
    assert Graph().parse(data=text, format="turtle")
