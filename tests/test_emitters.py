"""The two extension points, and that each is really a seam rather than a claim.

A package is "extensible" only if adding a thing is *adding a thing*. Both axes
are tested by exercising them:

* a **statistic** — `method` in `context.toml` had been recorded in every output
  and never dispatched on, so `method = "spearman"` computed Pearson and
  labelled it Spearman. A silent wrong answer, in the core, of exactly the kind
  this package refuses.
* an **output format** — registered by name, with third-party ones discovered
  through the `claimcheck.emitters` entry-point group, so somebody else can add
  one without forking this.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from claimcheck import emit
from claimcheck.relate import ESTIMATORS, check, pearson, spearman

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_the_declared_estimator_is_the_one_that_runs(tmp_path):
    """The whole defect. If these agreed to six places the dispatch would be
    doing nothing and the test would be worthless."""
    work = tmp_path / "e"
    shutil.copytree(EXAMPLE, work)
    ctx = work / "context.toml"
    original = ctx.read_text()

    p = check(work)["relationships"][0]["estimate"]
    ctx.write_text(original.replace('method = "pearson"', 'method = "spearman"'))
    s = check(work)["relationships"][0]["estimate"]
    assert p != s, "the estimator was declared and ignored"
    assert abs(p - s) > 0.01, (p, s)


def test_an_unknown_estimator_is_refused_with_the_list(tmp_path):
    work = tmp_path / "e"
    shutil.copytree(EXAMPLE, work)
    ctx = work / "context.toml"
    ctx.write_text(ctx.read_text().replace('method = "pearson"',
                                           'method = "kendall"'))
    got = check(work)["relationships"][0]
    assert got["verdict"] == "unverifiable"
    assert "unknown method 'kendall'" in got["why"]
    assert "pearson, spearman" in got["why"], "say what IS available"


def test_a_method_that_does_not_implement_the_declared_kind_is_refused(tmp_path):
    """`kind` and `method` must agree, or one of them is decoration."""
    work = tmp_path / "e"
    shutil.copytree(EXAMPLE, work)
    ctx = work / "context.toml"
    ctx.write_text(ctx.read_text().replace('kind = "correlation"',
                                           'kind = "difference"', 1))
    got = check(work)["relationships"][0]
    assert got["verdict"] == "unverifiable"
    assert "computes a correlation" in got["why"]


def test_spearman_is_pearson_on_ranks_and_handles_ties():
    """A monotonic curve is where the two must differ; tied values take the
    average rank, because inventing an order the data lacks is most of the
    correlation on repetitive data."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [1.0, 4.0, 9.0, 16.0, 25.0]        # perfectly monotonic, not linear
    assert spearman(xs, ys) == pytest.approx(1.0)
    assert pearson(xs, ys) < 0.99
    tied = spearman([1.0, 1.0, 2.0], [5.0, 5.0, 9.0])
    assert tied is not None and tied == pytest.approx(1.0)


def test_every_estimator_declares_the_kind_it_implements():
    for name, (fn, kind) in ESTIMATORS.items():
        assert callable(fn) and isinstance(kind, str) and kind, name


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------

def test_listing_formats_works_even_when_one_is_uninstalled():
    """`available()` must never raise. A list of what you could do is least
    useful to the people who already have everything."""
    got = emit.available()
    assert {"croissant", "graph", "dcat"} <= set(got)
    assert got["croissant"] is None, "croissant needs nothing"


def test_every_builtin_format_renders_text():
    for name, why in emit.available().items():
        if why:
            pytest.skip(f"{name}: {why}")
        text = emit.emit(name, EXAMPLE)
        assert isinstance(text, str) and text.strip(), name


def test_an_unknown_format_is_refused_with_the_list():
    with pytest.raises(emit.EmitError, match="no format named 'parquet'"):
        emit.emit("parquet", EXAMPLE)


def test_a_missing_extra_is_a_sentence_not_a_traceback(monkeypatch):
    """Asking for `graph` without rdflib must say which extra to install, not
    surface an ImportError from three frames down."""
    monkeypatch.setattr(emit, "available",
                        lambda: {"graph": "needs the 'graph' extra — "
                                          "pip install 'claimcheck[graph]'"})
    monkeypatch.setitem(emit._BUILTIN, "graph",
                        ("claimcheck.graph", "graph", "x"))
    with pytest.raises(emit.EmitError, match="needs the 'graph' extra"):
        emit.get("graph")


def test_a_third_party_emitter_is_discovered(monkeypatch):
    """The point of the entry-point group: somebody adds a format in their own
    package and it appears here without this repository changing."""
    monkeypatch.setattr(emit, "_third_party", lambda: {"parquet": lambda r: "x"})
    assert "parquet" in emit.available()
    assert emit.emit("parquet", EXAMPLE) == "x"


# ---------------------------------------------------------------------------
# DCAT specifically: what it puts into STANDARD slots
# ---------------------------------------------------------------------------

def test_dcat_puts_the_digest_in_the_standard_checksum_slot():
    """The reason for emitting DCAT at all: our hash becomes one a stranger's
    catalogue can read, because DCAT-AP already defines `spdx:checksum`."""
    pytest.importorskip("rdflib")
    from rdflib import Graph
    g = Graph().parse(data=emit.emit("dcat", EXAMPLE), format="turtle")
    rows = list(g.query("""
        PREFIX spdx: <http://spdx.org/rdf/terms#>
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        SELECT ?dist ?v WHERE {
          ?dist a dcat:Distribution ; spdx:checksum ?c .
          ?c spdx:algorithm spdx:checksumAlgorithm_sha256 ; spdx:checksumValue ?v }"""))
    assert len(rows) == 2, "both files need a checksum"
    assert all(len(str(r[1])) == 64 for r in rows)


def test_dcat_records_a_manual_import_in_dct_provenance():
    """`dct:provenance` is DCAT's own term, so a catalogue shows where a
    hand-imported file came from without knowing anything about this package."""
    pytest.importorskip("rdflib")
    from rdflib import Graph
    g = Graph().parse(data=emit.emit("dcat", EXAMPLE), format="turtle")
    prov = [str(o) for o in g.objects(
        None, __import__("rdflib").namespace.DCTERMS.provenance)]
    assert any("Manually imported" in p for p in prov), prov
    assert any("Snakemake" in p for p in prov), prov


def test_dcat_does_not_invent_standard_terms_for_units_or_claims():
    """DCAT has no slot for a unit or a refuted claim. Minting `dcat:refuted`
    would imply a standard that does not exist, so those stay in our namespace
    and this pins that they do."""
    pytest.importorskip("rdflib")
    from rdflib import Graph
    text = emit.emit("dcat", EXAMPLE)
    g = Graph().parse(data=text, format="turtle")
    dcat_terms = {str(p) for p in g.predicates()
                  if str(p).startswith("http://www.w3.org/ns/dcat#")}
    for invented in ("refuted", "unit", "claim", "verdict"):
        assert not any(invented in t.lower() for t in dcat_terms), t_bad(invented)
    assert "cc:claimsRefuted" in text, "ours, clearly ours"


def t_bad(term: str) -> str:
    return f"a dcat: term containing {term!r} was minted; DCAT defines no such thing"
