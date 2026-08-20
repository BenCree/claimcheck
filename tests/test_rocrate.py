"""RO-Crate: the version, the digest that survives compaction, and the crate root.

Three properties are worth a test here and the rest is the reference
implementation's business:

* the **version is 1.1**, because `ROCrate()` defaults to 1.2 and a bare
  constructor would move the published format with no diff showing it;
* `sha256` **expands to something**, because RO-Crate 1.1's context defines no
  checksum term at all and an undeclared key in compacted JSON-LD is dropped by
  every conformant processor while `json.loads` still shows it;
* the crate **sits at the root of what it describes**, because the directory
  holding `ro-crate-metadata.json` IS the crate root — the Croissant
  `contentUrl` defect, one format over.

`roc-validator` really runs. A crate that "cannot be validated because no
validator exists" was a sentence this project's predecessor put in six files and
inside the `description` of the crate it published, on the evidence of one dead
repository. It was false.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rocrate")

from claimcheck import rocrate                                 # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"


@pytest.fixture
def project(tmp_path):
    """A minimal project with a licence, which RO-Crate 1.1 REQUIRES."""
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    (root / "data" / "x.csv").write_text("a,b\n1,2\n3,4\n")
    (root / "context.toml").write_text(
        '[project]\nname = "fixture"\ntitle = "A fixture"\n'
        'license = "https://spdx.org/licenses/CC0-1.0.html"\n'
        'date_published = "2026-08-17"\n\n'
        '[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["data/x.csv"]\n')
    return root


# ---------------------------------------------------------------------------
# the version, which is the measured trap
# ---------------------------------------------------------------------------

def test_the_crate_is_1_1_and_says_so(project):
    """`ROCrate()` defaults to 1.2 in ro-crate-py 0.15.1. Passing the version
    explicitly is the decision; this is what makes changing it visible."""
    assert rocrate.RO_CRATE_VERSION == "1.1"
    doc = rocrate.build(project)
    desc = next(e for e in doc["@graph"]
                if e["@id"] == "ro-crate-metadata.json")
    assert desc["conformsTo"]["@id"] == "https://w3id.org/ro/crate/1.1"


def test_the_default_version_really_is_something_else(project):
    """Two-sided. If ro-crate-py's default were already 1.1 the line above would
    be pinning nothing, and this test would say so rather than passing quietly."""
    from rocrate.rocrate import ROCrate
    bare = ROCrate()
    doc = bare.metadata.generate()
    desc = next(e for e in doc["@graph"]
                if e["@id"] == "ro-crate-metadata.json")
    assert desc["conformsTo"]["@id"] != "https://w3id.org/ro/crate/1.1", (
        "the library's default is now 1.1, so passing the version explicitly no "
        "longer changes anything — check whether this module should move too")


# ---------------------------------------------------------------------------
# the digest, which is silently dropped unless the term is declared
# ---------------------------------------------------------------------------

def test_sha256_is_declared_in_the_context(project):
    doc = rocrate.build(project)
    ctx = doc["@context"]
    assert isinstance(ctx, list), (
        "extra_terms should have made @context an array with a local mapping "
        "appended; a bare string means nothing was declared")
    assert any(isinstance(c, dict) and "sha256" in c for c in ctx), ctx


def test_an_undeclared_sha256_expands_to_nothing(project):
    """The two-sided half, and the only way to see the defect.

    Reading the JSON with `json.loads` — which is how anybody would look — shows
    `sha256` present either way. It is only in the EXPANDED document that an
    undeclared term disappears, which is what every conformant consumer sees.
    """
    rdflib = pytest.importorskip("rdflib")
    doc = rocrate.build(project)
    good = rdflib.Graph().parse(data=json.dumps(doc), format="json-ld")
    n_good = sum(1 for _ in good.triples(
        (None, rdflib.URIRef("http://spdx.org/rdf/terms#checksumValue"), None)))
    assert n_good >= 1, "the declared digest did not survive expansion"

    stripped = json.loads(json.dumps(doc))
    stripped["@context"] = [c for c in stripped["@context"]
                            if not isinstance(c, dict)]
    bad = rdflib.Graph().parse(data=json.dumps(stripped), format="json-ld")
    n_bad = sum(1 for _ in bad.triples(
        (None, rdflib.URIRef("http://spdx.org/rdf/terms#checksumValue"), None)))
    assert n_bad == 0, (
        "an undeclared `sha256` survived expansion, so declaring it in "
        "@context is protecting nothing and this test proves nothing")


def test_every_file_carries_a_real_digest(project):
    doc = rocrate.build(project)
    files = [e for e in doc["@graph"] if e.get("@type") == "File"]
    assert files, "the crate describes no file"
    from claimcheck.croissant import sha256
    for e in files:
        assert len(e["sha256"]) == 64
        assert e["sha256"] == sha256(project / e["@id"])


# ---------------------------------------------------------------------------
# the crate root, which is the Croissant defect one format over
# ---------------------------------------------------------------------------

def test_a_crate_written_at_its_root_reads_back(project):
    out = rocrate.write(project)
    assert out == project / "ro-crate-metadata.json"
    complaints, n = rocrate.readable(out)
    assert not complaints, complaints
    assert n >= 1, "nothing was dereferenced, so this passed over zero checks"


def test_writing_a_crate_away_from_its_root_is_refused(project):
    (project / "results").mkdir()
    with pytest.raises(rocrate.CrateError, match="IS the crate root"):
        rocrate.write(project, project / "results" / "ro-crate-metadata.json")


def test_a_misplaced_crate_is_caught_by_reading_it(project):
    """The two-sided half: the refusal above only helps if the same crate is
    genuinely broken when somebody writes it there another way."""
    doc = rocrate.build(project)
    (project / "results").mkdir()
    moved = project / "results" / "ro-crate-metadata.json"
    moved.write_text(json.dumps(doc, indent=2))
    complaints, _n = rocrate.readable(moved)
    assert any("denotes nothing" in c for c in complaints), complaints


def test_a_wrong_digest_is_caught_at_read_time(project):
    """RO-Crate 1.1 does not verify checksums at REQUIRED, so a validator will
    not say this. Reading it is the only way to find out."""
    out = rocrate.write(project)
    doc = json.loads(out.read_text())
    for e in doc["@graph"]:
        if e.get("sha256"):
            e["sha256"] = "0" * 64
    out.write_text(json.dumps(doc))
    complaints, _n = rocrate.readable(out)
    assert any("sha256" in c for c in complaints), complaints


def test_a_wrong_size_is_caught_at_read_time(project):
    out = rocrate.write(project)
    doc = json.loads(out.read_text())
    for e in doc["@graph"]:
        if e.get("contentSize"):
            e["contentSize"] = 999999
    out.write_text(json.dumps(doc))
    complaints, _n = rocrate.readable(out)
    assert any("contentSize" in c for c in complaints), complaints


def test_a_crate_that_names_no_file_is_not_a_pass(tmp_path):
    """An empty complaint list over an empty crate would read as health."""
    (tmp_path / "ro-crate-metadata.json").write_text(json.dumps(
        {"@context": "https://w3id.org/ro/crate/1.1/context",
         "@graph": [{"@id": "./", "@type": "Dataset"}]}))
    complaints, n = rocrate.readable(tmp_path / "ro-crate-metadata.json")
    assert n == 0
    assert any("nothing in it" in c for c in complaints), complaints


# ---------------------------------------------------------------------------
# roc-validator, really run
# ---------------------------------------------------------------------------

def test_roc_validator_passes_a_crate_we_built(project):
    pytest.importorskip("rocrate_validator")
    rocrate.write(project)
    r = rocrate.validate(project)
    assert not r.crashed, r.issues
    assert r.checks_run > 0, "no check ran, so nothing agreed to anything"
    assert r.passed, r.issues


def test_roc_validator_really_refuses_a_bad_crate(project):
    """Two-sided. A validator that passes everything is decoration — and the
    licence is a REQUIRED property of the root data entity, so removing it is a
    genuine violation rather than a corrupted file."""
    pytest.importorskip("rocrate_validator")
    out = rocrate.write(project)
    doc = json.loads(out.read_text())
    for e in doc["@graph"]:
        e.pop("license", None)
    doc["@graph"] = [e for e in doc["@graph"]
                     if "spdx.org/licenses" not in str(e.get("@id", ""))]
    out.write_text(json.dumps(doc, indent=2))
    r = rocrate.validate(project)
    assert not r.crashed, r.issues
    assert not r.passed, "roc-validator passed a crate with no licence"
    assert any("license" in i.lower() for i in r.issues), r.issues


def test_a_crash_is_reported_as_a_crash_and_never_as_a_pass(tmp_path):
    """`checks_run: 0` beside `passed: False` is the tell. Without `crashed` a
    reader spends the afternoon on a crate that is fine."""
    r = rocrate.validate(tmp_path / "nothing-here")
    assert r.crashed and not r.passed and r.checks_run == 0
    assert r.as_dict()["crashed"] is True


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------

def test_the_selftest_passes():
    assert rocrate.main(["--selftest"]) == 0


def test_the_emitter_is_registered_and_renders():
    from claimcheck import emit
    assert "rocrate" in emit.available()
    if emit.available()["rocrate"]:
        pytest.skip(emit.available()["rocrate"])
    text = emit.emit("rocrate", EXAMPLE)
    assert json.loads(text)["@graph"]


def test_the_example_crate_names_its_missing_licence(tmp_path):
    """The real project, and it does NOT pass — RO-Crate 1.1 requires a licence
    on the root data entity and `example/context.toml` declares none. Recorded
    as a test rather than fixed, because which licence the example carries is
    not this module's decision to take. Delete this test when it gains one.
    """
    pytest.importorskip("rocrate_validator")
    import shutil
    work = tmp_path / "example"
    shutil.copytree(EXAMPLE, work)
    rocrate.write(work)
    r = rocrate.validate(work)
    assert not r.crashed and r.checks_run > 0
    assert not r.passed
    assert any("license" in i.lower() for i in r.issues), r.issues
