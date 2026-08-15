"""The rdflib layer: what the graph must be able to answer, and what it must not hold.

A graph is easy to emit and hard to justify. Every test here is a **question a
reader would actually ask**, so if one fails the graph has stopped earning the
dependency rather than merely changed shape:

* where did this file come from, and was it computed or dropped in by hand?
* which column is this claim about, in which file, at which digest?
* if two datasets were joined, on what key, and how much of them matched?
* what challenges what?

Plus the one negative: **the data rows must not be in it.**
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, Literal, URIRef  # noqa: E402

from claimcheck.graph import build  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"
TL = "https://w3id.org/provchem/two-layer#"
EVI = "https://w3id.org/EVI#"


@pytest.fixture(scope="module")
def g() -> Graph:
    """Built in a COPY that is run from scratch, so the test cannot pass on
    results a previous run left in the working tree."""
    if shutil.which("snakemake") is None:
        pytest.skip("snakemake is not installed")
    import os
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "example"
    shutil.copytree(EXAMPLE, tmp)
    shutil.rmtree(tmp / "results", ignore_errors=True)
    shutil.rmtree(tmp / ".snakemake", ignore_errors=True)
    r = subprocess.run(["snakemake", "-c1"], capture_output=True, text=True,
                       cwd=str(tmp),
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stderr[-2000:]
    return build(tmp)


def _ask(g: Graph, q: str) -> list:
    return list(g.query(q))


# ---------------------------------------------------------------------------
# It parses, and it is Turtle a stranger can read
# ---------------------------------------------------------------------------

def test_it_serialises_and_reparses_to_the_same_graph(g):
    """A graph that does not survive its own round trip is not a record."""
    again = Graph().parse(data=g.serialize(format="turtle"), format="turtle")
    assert len(again) == len(g)
    assert set(again) == set(g)


def test_every_prefix_a_reader_needs_is_bound(g):
    """Unbound prefixes turn Turtle into full IRIs on every line — still
    correct, still unreadable, and `git diff` becomes useless."""
    bound = {p for p, _ in g.namespaces()}
    assert {"sc", "cr", "evi", "tl"} <= bound, bound


# ---------------------------------------------------------------------------
# The four questions
# ---------------------------------------------------------------------------

def test_it_says_which_files_a_person_put_there(g):
    """The distinction that changes how much a number is worth. A consumer who
    cannot tell a computed file from a hand-dropped one has been given the same
    confidence in both."""
    rows = _ask(g, f"""
        PREFIX tl: <{TL}> PREFIX sc: <https://schema.org/>
        SELECT ?name ?origin ?sha WHERE {{
          ?f sc:name ?name ; tl:origin ?origin ; sc:sha256 ?sha }}""")
    origins = {str(r[0]): str(r[1]) for r in rows}
    assert set(origins.values()) == {"computed", "manual"}, origins
    assert all(str(r[2]) for r in rows), "every file carries a digest"


def test_a_manual_file_carries_where_it_came_from(g):
    rows = _ask(g, f"""
        PREFIX tl: <{TL}>
        SELECT ?src ?when WHERE {{
          ?f tl:origin "manual" ; tl:manualSource ?src ; tl:retrieved ?when }}""")
    assert rows, "a manual file with no stated source is the thing this refuses"
    assert str(rows[0][0]).strip() and str(rows[0][1]).strip()


def test_a_claim_reaches_a_column_in_a_file_at_a_digest(g):
    """The traversal the whole graph exists for: from where a claim came from,
    down to the bytes it is about, in one query."""
    rows = _ask(g, f"""
        PREFIX tl: <{TL}> PREFIX sc: <https://schema.org/>
        SELECT ?locator ?col ?unit ?ds ?sha WHERE {{
          ?ev  sc:identifier ?locator .
          ?rel tl:restsOn ?ev ; tl:x ?x .
          ?x   sc:name ?col ; tl:unit ?unit ; tl:inDataset ?ds ; tl:inFile ?f .
          ?f   sc:sha256 ?sha }}""")
    assert rows, "no path from evidence to a column to a digest"
    for r in rows:
        assert all(str(v).strip() for v in r)


def test_a_cross_dataset_claim_carries_its_join(g):
    """Coverage travels with the verdict. A correlation over a 40%-covered join
    is a number about neither file's population, and a reader cannot tell
    without this."""
    rows = _ask(g, f"""
        PREFIX tl: <{TL}>
        SELECT ?key ?cov ?n WHERE {{
          ?rel tl:joinKey ?key ; tl:joinCoverage ?cov ; tl:joinMatched ?n }}""")
    assert rows, "the example's cross-dataset claim lost its join report"
    key, cov, n = rows[0]
    assert str(key) == "complex_name"
    assert float(cov) == 1.0 and int(n) == 637


def test_a_refuted_claim_is_challenged_with_evi_s_own_verb(g):
    """`evi:directlyChallenges`, not a private predicate — so a consumer with a
    reasoner derives the consequences themselves. EVI defines
    `indirectlyChallenges` as the chain (directlyChallenges o supports) with
    `supports` transitive."""
    rows = _ask(g, f"""
        PREFIX evi: <{EVI}> PREFIX tl: <{TL}>
        SELECT ?rel WHERE {{ ?v evi:directlyChallenges ?rel .
                             ?rel tl:verdict "refuted" }}""")
    assert rows, "a refuted claim with nothing challenging it"


# ---------------------------------------------------------------------------
# The negative, which is the one that keeps the graph small
# ---------------------------------------------------------------------------

def test_the_rows_are_not_in_the_graph(g):
    """1,274 data rows across two files. One node per COLUMN, not per
    measurement — five million triples answer no question a CSV reader could
    not, and one store measured elsewhere was 21.5 MB of unreadable bytes."""
    assert len(g) < 200, f"{len(g)} triples — the rows have leaked in"
    fields = _ask(g, "PREFIX cr: <http://mlcommons.org/croissant/> "
                     "SELECT ?f WHERE { ?f a cr:Field }")
    assert len(fields) == 3, "three declared variables, three field nodes"


def test_no_literal_in_the_graph_is_a_data_value(g):
    """A sharper version of the above: if a measurement had leaked in as a
    literal, the count test might still pass on a small file."""
    from claimcheck.datasets import load as load_dataset
    import tomllib
    ctx = tomllib.loads((EXAMPLE / "context.toml").read_text())
    ds = load_dataset(EXAMPLE, ctx["datasets"][0])
    sample = {r["e_int_kcal"] for r in ds["rows"][:50]}
    literals = {str(o) for o in g.objects() if isinstance(o, Literal)}
    leaked = sample & literals
    assert not leaked, f"data values present as literals: {sorted(leaked)[:3]}"


def test_every_subject_is_a_project_iri(g):
    """No blank nodes and no stray namespaces: a record whose identifiers are
    generated per-run cannot be compared with the next run's."""
    strays = {str(s) for s in g.subjects()
              if not str(s).startswith("https://w3id.org/provchem/two-layer/")}
    assert not strays, strays
