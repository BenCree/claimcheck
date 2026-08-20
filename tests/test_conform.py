"""Conformance: validating is not reading, and a sweep must say which it did.

`tests/test_croissant_loads.py` is the miniature version of this — one format,
one project, one test — written the day an emitted Croissant validated cleanly
under `mlcroissant` and yielded **zero** records because `contentUrl` was
relative to the project root. This is the general form, and the tests below are
about the generalisation rather than about Croissant:

* the sweep reports `validates` and `reads` **separately**, and would have
  printed the defect as a line;
* a readback that dereferenced **nothing** is UNKNOWN, never PASS — a checker
  reporting a pass over zero checks is the failure this package exists to
  prevent, and the predecessor shipped one for every call it ever made;
* a **crash is UNKNOWN, never FAIL** — "your metadata is invalid" and "the
  validator broke" send a reader to two different places;
* an unreadable file is **counted**, never skipped — a file that will not parse
  is exactly the file somebody needs told about.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from claimcheck import conform

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"

mlc = pytest.importorskip("mlcroissant")


@pytest.fixture
def tree(tmp_path):
    """A project whose Croissant sits beside its data, and a copy one level down.

    The two documents are byte-identical. Only where they sit differs, which is
    the entire defect.
    """
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    (root / "data" / "x.csv").write_text("a,b\n1,2\n3,4\n")
    doc = conform._croissant_stub(root, "data/x.csv")
    (root / "croissant.json").write_text(json.dumps(doc, indent=2))
    (root / "results").mkdir()
    shutil.copy(root / "croissant.json", root / "results" / "croissant.json")
    return root


# ---------------------------------------------------------------------------
# the whole point: two verdicts, never folded
# ---------------------------------------------------------------------------

def test_a_document_that_validates_and_cannot_be_read_is_a_visible_line(tree):
    f = conform.check_one(tree / "results" / "croissant.json")
    assert f.validates == conform.PASS, "the manifest should still validate"
    assert f.reads == conform.FAIL
    assert f.status == conform.FAIL
    assert any("contentUrl" in u for u in f.unreadable), f.unreadable


def test_the_same_document_beside_its_data_reads(tree):
    """Two-sided. Without this the test above would pass on a checker that
    reports FAIL for everything."""
    f = conform.check_one(tree / "croissant.json")
    assert (f.validates, f.reads, f.status) == (conform.PASS, conform.PASS,
                                                conform.PASS)
    assert not f.unreadable


def test_the_sweep_counts_validated_but_unreadable_separately(tree):
    """A single status would have printed PASS for the defect. The count is what
    a reader scanning the last line of a report actually sees."""
    rep = conform.sweep(tree)
    assert rep["validated_but_unreadable"] == 1, rep["by_status"]
    assert rep["by_status"] == {"FAIL": 1, "PASS": 1}, rep["by_status"]
    assert "VALIDATE AND CANNOT BE READ" in conform.format_report(rep)


def test_a_croissant_whose_digest_does_not_describe_the_bytes_fails_to_read(tree):
    """`mlcroissant` verifies `sha256` at READ time, not at validation time —
    a third thing a validator leaves alone."""
    p = tree / "croissant.json"
    doc = json.loads(p.read_text())
    doc["distribution"][0]["sha256"] = "0" * 64
    p.write_text(json.dumps(doc))
    f = conform.check_one(p)
    assert f.validates == conform.PASS and f.reads == conform.FAIL


# ---------------------------------------------------------------------------
# a pass over zero checks is not a pass
# ---------------------------------------------------------------------------

def test_a_readback_that_dereferenced_nothing_is_unknown(tmp_path):
    """The predecessor's `iris_resolve` looked for `file:` IRIs in a graph that
    mints none, so it returned `[]` on every call ever made and was advertised
    as one of three tests. An empty complaint list is not evidence."""
    crate = tmp_path / "ro-crate-metadata.json"
    crate.write_text(json.dumps({
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {"@id": "ro-crate-metadata.json", "@type": "CreativeWork",
             "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
             "about": {"@id": "./"}},
            {"@id": "./", "@type": "Dataset"},
            {"@id": "https://example.org/remote.csv", "@type": "File"}]}))
    f = conform.check_one(crate)
    assert not f.unreadable
    assert f.reads == conform.UNKNOWN, (
        "an empty complaint list over zero dereferences was reported as a pass")
    assert "nothing local was opened" in f.detail


def test_a_reads_pass_says_how_many_files_it_opened(tree):
    f = conform.check_one(tree / "croissant.json")
    assert f.reads == conform.PASS
    assert "record set(s) read" in f.detail


# ---------------------------------------------------------------------------
# UNKNOWN is not a pass, and a crash is not a FAIL
# ---------------------------------------------------------------------------

def test_a_file_that_will_not_parse_is_counted_not_skipped(tmp_path):
    """It used to classify as "not metadata" and leave the denominator, so a
    sweep reported `0 metadata document(s) of 1 file(s)` over a truncated
    crate."""
    (tmp_path / "broken.json").write_text('{"recordSet": [')
    f = conform.check_one(tmp_path / "broken.json")
    assert f.status == conform.UNKNOWN
    assert "will not parse" in f.detail
    rep = conform.sweep(tmp_path)
    assert rep["metadata_documents"] == 1, rep


def test_ordinary_json_is_skipped_and_says_so(tmp_path):
    (tmp_path / "config.json").write_text('{"threads": 4}')
    f = conform.check_one(tmp_path / "config.json")
    assert f.status == conform.SKIP
    rep = conform.sweep(tmp_path)
    assert rep["metadata_documents"] == 0 and rep["skipped_not_metadata"] == 1
    assert "not a metadata document" in conform.format_report(rep)


def test_an_empty_sweep_does_not_report_health(tmp_path):
    """`0 metadata document(s) of 0 file(s)` read as a clean bill of health over
    a tree nothing had opened."""
    rep = conform.sweep(tmp_path / "nothing")
    assert "NOTHING WAS EXAMINED" in conform.format_report(rep)


def test_one_unreadable_file_does_not_end_the_sweep(tmp_path):
    """`json` raises a bare ValueError past the 4300-digit integer limit; one
    bad file used to abort the sweep for every other file."""
    (tmp_path / "huge.json").write_text("1" + "0" * 5000)
    (tmp_path / "config.json").write_text('{"threads": 4}')
    rep = conform.sweep(tmp_path)
    assert rep["files_seen"] == 2


def test_a_named_file_is_always_a_candidate(tree):
    """`rglob` on a file matches nothing, so a named file printed
    `0 metadata document(s) of 0 file(s)` and exited 0 — an affirmative sentence
    about a document nothing had opened."""
    rep = conform.sweep(tree / "croissant.json")
    assert rep["target"] == "file" and rep["metadata_documents"] == 1


def test_a_validator_crash_is_unknown_and_not_fail(tmp_path, monkeypatch):
    """Measured: `mlcroissant` 1.1.0 raises `KeyError: '@language'` on any
    manifest whose `@context` omits it — 13 of 23 in the first real tree this
    was run against. Reported as FAIL it reads "your manifests are invalid"."""
    p = tmp_path / "croissant.json"
    p.write_text(json.dumps({"conformsTo": "http://mlcommons.org/croissant/1.1",
                             "recordSet": []}))

    class Boom:
        def __init__(self, *a, **k):
            raise KeyError("@language")

    monkeypatch.setattr(mlc, "Dataset", Boom)
    f = conform.check_one(p)
    assert f.status == conform.UNKNOWN, (f.status, f.detail)
    assert "did not reach a verdict" in f.detail
    assert "UNKNOWN is not a pass" in conform.format_report(conform.sweep(p))


# ---------------------------------------------------------------------------
# detection is structural
# ---------------------------------------------------------------------------

def test_a_manifest_is_found_whatever_it_is_called(tree):
    odd = tree / "metadata-for-people.jsonld"
    shutil.copy(tree / "croissant.json", odd)
    kind, version = conform.classify(odd)
    assert kind == conform.CROISSANT and version == "1.1"


def test_a_crate_is_recognised_by_its_descriptor(tmp_path):
    (tmp_path / "anything.json").write_text(json.dumps({"@graph": [
        {"@id": "ro-crate-metadata.json",
         "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"}}]}))
    assert conform.classify(tmp_path / "anything.json") == ("ro-crate", "1.1")


def test_turtle_without_dcat_is_not_a_catalogue_record(tmp_path):
    (tmp_path / "claims.ttl").write_text(
        '@prefix ex: <https://example.org/> .\nex:a ex:b ex:c .\n')
    assert conform.classify(tmp_path / "claims.ttl")[0] == conform.NOTHING


# ---------------------------------------------------------------------------
# DCAT: a profile is checked only when the record claims one
# ---------------------------------------------------------------------------

@pytest.fixture
def published(tmp_path):
    """The example with its `results/` rebuilt here, exactly as the Snakefile
    writes them. Built rather than read: `example/results/` is gitignored, so a
    test reading it passes only on a machine where somebody has already run the
    workflow."""
    pytest.importorskip("rdflib")
    from claimcheck import dcat
    work = tmp_path / "example"
    shutil.copytree(EXAMPLE, work, ignore=shutil.ignore_patterns(
        "results", ".snakemake", ".tests"))
    (work / "results").mkdir()
    (work / "results" / "dcat.ttl").write_text(dcat.render(work))
    return work


def test_plain_dcat_is_not_accused_of_failing_a_profile_it_never_claimed(
        published):
    """The mirror image of claiming a profile you do not meet. Reporting a plain
    catalogue card as "fails PSDI" is an accusation about a standard its author
    never invoked, so the verdict is UNKNOWN and it says why."""
    f = conform.check_one(published / "results" / "dcat.ttl")
    assert f.validates == conform.UNKNOWN
    assert "no profile declared" in f.detail
    assert f.declared_version == "plain"


def test_a_psdi_record_is_checked_against_psdi(tmp_path):
    pytest.importorskip("pyshacl")
    from claimcheck import psdi
    p = tmp_path / "psdi.ttl"
    p.write_text(psdi.render(EXAMPLE))
    f = conform.check_one(p)
    assert f.declared_version == "psdi"
    assert "PSDI shapes @" in f.detail


def test_the_shipped_dcat_record_points_where_a_consumer_is_not(published):
    """A REAL FINDING, recorded rather than fixed.

    `example/results/dcat.ttl` — written by the `dcat` rule exactly as rebuilt
    in this fixture — carries `dcat:downloadURL "data/x.csv"`, relative to the
    project root, exactly as `croissant.json` did before 2026-08-17, while the
    document sits in `results/`. A relative literal declares no base, so
    somebody holding this file alone has one base and it is the wrong one.

    Not fixed here because the fix is a choice this module does not get to make:
    rewrite the path relative to the output (what Croissant did), or emit an IRI
    (what `psdi.py` does, and what DCAT's own range asks for). Both change the
    published record and one of them changes a byte-exact Snakemake fixture.
    **Delete this test when that decision is taken** — it will fail loudly.
    """
    from claimcheck import dcat
    complaints, n = dcat.readback(published / "results" / "dcat.ttl")
    assert n == 2, "nothing was dereferenced, so this test checks nothing"
    assert len(complaints) == 2, complaints
    assert all("does not resolve from" in c for c in complaints), complaints
    assert any("it resolves from" in c for c in complaints), (
        "the report should name the base that does work, or the reader hunts")


def test_a_dcat_record_beside_its_data_reads(tmp_path):
    """Two-sided for the finding above: the checker is not simply hostile to
    every DCAT record."""
    pytest.importorskip("rdflib")
    from claimcheck import dcat
    shutil.copytree(EXAMPLE, tmp_path / "e")
    out = tmp_path / "e" / "dcat.ttl"
    out.write_text(dcat.render(tmp_path / "e"))
    complaints, n = dcat.readback(out)
    assert n == 2 and not complaints, complaints


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------

def test_the_selftest_passes():
    assert conform.main(["--selftest"]) == 0


def test_the_command_exits_non_zero_when_something_is_wrong(tree, capsys):
    assert conform.main([str(tree)]) == 1
    assert conform.main([str(tree / "croissant.json")]) == 0
    capsys.readouterr()


def test_the_json_report_carries_both_verdicts(tree, capsys):
    conform.main(["--json", str(tree)])
    rep = json.loads(capsys.readouterr().out)
    assert {"validates", "reads"} <= set(rep["findings"][0])
