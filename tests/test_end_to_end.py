"""The example, driven through Snakemake exactly as a reader would.

Unit tests check functions. This checks the thing the README promises: clone
it, run two commands, get a validated dataset description with its claims
checked. Every defect in the emitted file was invisible until a validator saw
it, so the validator runs here rather than being described.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"


def _run(*argv, cwd=EXAMPLE):
    return subprocess.run(argv, capture_output=True, text=True, cwd=str(cwd),
                          env={**os.environ, "PYTHONPATH": str(REPO)})


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A COPY, so the test cannot pass by finding results somebody left behind.

    That is not hypothetical caution: the predecessor had a test asserting "no
    failures" over a results directory that is gitignored, so on a fresh clone
    it examined nothing and passed.
    """
    if shutil.which("snakemake") is None:
        pytest.skip("snakemake is not installed")
    work = tmp_path_factory.mktemp("e2e") / "example"
    shutil.copytree(EXAMPLE, work)
    shutil.rmtree(work / "results", ignore_errors=True)
    r = _run("snakemake", "-c1", cwd=work)
    assert r.returncode == 0, r.stderr[-2000:]
    return work


def test_the_dry_run_lists_every_rule(tmp_path):
    """ON A CLEAN COPY. Run in place, `snakemake -n` correctly answers "nothing
    to be done" against results a previous run left behind — so this test
    passed or failed depending on the state of the working tree, which is the
    vacuity shape it is supposed to be guarding against."""
    if shutil.which("snakemake") is None:
        pytest.skip("snakemake is not installed")
    work = tmp_path / "example"
    shutil.copytree(EXAMPLE, work)
    shutil.rmtree(work / "results", ignore_errors=True)
    r = _run("snakemake", "-n", cwd=work)
    assert r.returncode == 0, r.stderr[-1500:]
    for rule in ("check_claims", "croissant", "graph"):
        assert rule in r.stdout, rule


def test_a_first_run_shows_both_verdicts(built):
    """`agrees` and `refuted` on the same relationship, from one paper's figure
    read two ways. A reader learns from the first run that both exist."""
    d = json.loads((built / "results/claims.json").read_text())
    assert d["verdicts"] == {"agrees": 1, "refuted": 1}, d["verdicts"]


def test_snakemake_recorded_the_dataset_digest_without_us_writing_one(built):
    """THE DESIGN CLAIM, checked rather than asserted.

    Snakemake stores a digest for every input and none for its outputs, so
    terminal outputs — the published ones — have no recorded hash. Making the
    metadata rule consume the dataset puts the hash in Snakemake's own store.
    If this fails, the argument for owning no receipt code has gone with it.
    """
    import base64
    records = list((built / ".snakemake" / "metadata").rglob("*"))
    found = {}
    for f in (p for p in records if p.is_file()):
        target = base64.b64decode(f.name).decode()
        rec = json.loads(f.read_text())
        found[Path(target).name] = rec.get("input_checksums") or {}

    croissant = found.get("croissant.json")
    assert croissant, f"no metadata record for croissant.json; saw {list(found)}"
    hashed = {Path(k).name: v for k, v in croissant.items()}
    assert "measurements.csv" in hashed, hashed
    assert hashed["measurements.csv"].startswith("sha256:"), hashed
    # And the context file too, so a claim cannot be edited without the record
    # of the description that quoted it changing.
    assert "context.toml" in hashed, hashed


def test_the_croissant_validates(built):
    if shutil.which("mlcroissant") is None:
        pytest.skip("mlcroissant is not installed")
    out = built / "results/croissant.json"
    r = subprocess.run(["mlcroissant", "validate", "--jsonld", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "error(s)" not in r.stderr, r.stderr[-1500:]


def test_the_croissant_carries_units_croissant_itself_cannot_express(built):
    """Croissant 1.1 has no unit mechanism — verified against the downloaded
    spec: no `unitText`, `unitCode`, `PropertyValue`, `variableMeasured`, no
    QUDT. So units ride as annotations, and this pins that they survive."""
    d = json.loads((built / "results/croissant.json").read_text())
    ns = "https://w3id.org/provchem/two-layer#"
    for f in d["recordSet"][0]["field"]:
        assert f[f"{ns}unit"], f
        assert f[f"{ns}unitSystem"] == "UDUNITS-2"


def test_the_graph_joins_a_claim_to_a_column_to_a_digest(built):
    """The whole point of emitting RDF at all: one query from where a claim came
    from down to the bytes it is about. If that path is broken the graph is
    decoration."""
    pytest.importorskip("rdflib")
    from rdflib import Graph
    g = Graph().parse(built / "results/graph.ttl", format="turtle")
    rows = list(g.query("""
        PREFIX tl: <https://w3id.org/provchem/two-layer#>
        PREFIX sc: <https://schema.org/>
        SELECT ?locator ?col ?unit ?sha WHERE {
          ?ev  sc:identifier ?locator .
          ?rel tl:restsOn ?ev ; tl:x ?x .
          ?x   sc:name ?col ; tl:unit ?unit ; tl:inFile ?f .
          ?f   sc:sha256 ?sha .
        }"""))
    assert rows, "no path from evidence to a column to a digest"
    for r in rows:
        assert str(r.sha) and str(r.unit) and str(r.col)


def test_the_graph_does_not_contain_the_rows(built):
    """870 data rows must not become triples. One store measured elsewhere at
    21.5 MB of bytes `git diff` cannot read is the reason."""
    pytest.importorskip("rdflib")
    from rdflib import Graph
    g = Graph().parse(built / "results/graph.ttl", format="turtle")
    n_rows = json.loads((built / "results/claims.json").read_text())["rows_read"]
    assert n_rows == 870
    assert len(g) < 200, (
        f"{len(g)} triples for a {n_rows}-row table — the rows have leaked into "
        f"the graph; it should hold one node per COLUMN")
