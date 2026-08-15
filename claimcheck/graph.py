"""The data graph and the ontology graph, joined — and the rows left out of both.

## Four levels, and only two of them are triples

| level | what it holds | where it lives | why |
|---|---|---|---|
| **rows** | the numbers | the CSV. **Never triples.** | a million rows is 5M triples that answer no question a CSV reader could not |
| **schema** | one node per column | RDF, Croissant `cr:Field` | small, fixed size, and it is what a claim attaches to |
| **claims** | relationships, evidence, verdicts | RDF, EVI vocabulary | the part with no other home |
| **execution** | who made the file, from what | Snakemake's own store | already recorded; joined by path + sha256, not restated |

    python -m claimcheck.graph <project-dir> [<out.ttl>]

Putting the rows in is the classic failure. `docs/LANDSCAPE.md` measured one
property-graph store at 21.5 MB of opaque bytes for a record `git diff` cannot
read. The graph carries the **schema** and the **claims**; the bytes stay in the
file and are joined by digest.

## The join keys, which is the whole design

    evidence --> claim --> relationship --> cr:Field --> cr:FileObject --> sha256

`cr:Field` is the hinge. Croissant already gives every column an IRI and already
points at the file with a digest, so a claim attached to a Field IRI reaches the
bytes without inventing an identifier scheme. **The data graph and the ontology
graph are joined at the column.**

That makes "where is this evidence from" a traversal rather than a lookup in
somebody's memory: from a DOI and a figure number, down to a named column, in a
named file, with the hash Snakemake recorded when it made it.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef

from claimcheck.namespace import BASE, NS
from rdflib.namespace import RDF, RDFS, XSD

SC = Namespace("https://schema.org/")
CR = Namespace("http://mlcommons.org/croissant/")
EVI = Namespace("https://w3id.org/EVI#")
TL = Namespace(NS)


def build(root: Path | str) -> Graph:
    root = Path(root)
    ctx = tomllib.loads((root / "context.toml").read_text())
    claims = json.loads((root / "results/claims.json").read_text())
    croissant = json.loads((root / "results/croissant.json").read_text())

    g = Graph()
    for pfx, n in (("sc", SC), ("cr", CR), ("evi", EVI), ("tl", TL)):
        g.bind(pfx, n)

    proj = URIRef(f"{BASE}project/{ctx['project']['name']}")
    g.add((proj, RDF.type, SC.Dataset))
    g.add((proj, SC.name, Literal(ctx["project"]["name"])))

    # --- one node per FILE, carrying its digest and its ORIGIN --------------
    files = {}
    for dist in croissant["distribution"]:
        f = URIRef(f"{BASE}file/{dist['name']}")
        files[dist["name"]] = f
        g.add((f, RDF.type, CR.FileObject))
        g.add((f, SC.name, Literal(dist["name"])))
        g.add((f, SC.sha256, Literal(dist["sha256"])))
        # WHETHER A PERSON PUT IT THERE is a fact about how much the number is
        # worth, so it belongs in the graph and not only in a comment.
        origin = dist.get(f"{TL}origin")
        if origin:
            g.add((f, TL.origin, Literal(origin)))
        if dist.get(f"{TL}manualSource"):
            g.add((f, TL.manualSource, Literal(dist[f"{TL}manualSource"])))
            g.add((f, TL.retrieved, Literal(dist[f"{TL}retrieved"])))
        g.add((proj, SC.distribution, f))

    # --- one node per COLUMN. The rows stay in the CSV ----------------------
    fields = {}
    for rs in croissant["recordSet"]:
        did = rs.get(f"{TL}dataset", "")
        for fld in rs["field"]:
            col = fld["name"]
            u = URIRef(f"{BASE}records/{did}.{col}")
            fields[f"{did}.{col}"] = u
            g.add((u, RDF.type, CR.Field))
            g.add((u, SC.name, Literal(col)))
            g.add((u, TL.inDataset, Literal(did)))
            g.add((u, RDFS.comment, Literal(fld.get("description", ""))))
            unit = fld.get(f"{TL}unit")
            if unit is not None:
                g.add((u, TL.unit, Literal(unit)))
                g.add((u, TL.unitSystem, Literal("UDUNITS-2")))
            src = fld.get("source", {}).get("fileObject", {}).get("@id")
            for name, node in files.items():
                if name == src:
                    g.add((u, TL.inFile, node))

    # --- claims, evidence, verdicts ----------------------------------------
    for i, r in enumerate(claims["relationships"], 1):
        rel = URIRef(f"{BASE}relationship/{i}")
        g.add((rel, RDF.type, TL.Relationship))
        for role, ref in (("x", r["x"]), ("y", r["y"])):
            if ref in fields:
                g.add((rel, TL[role], fields[ref]))
        g.add((rel, TL.method, Literal(r["method"])))
        g.add((rel, TL.claimed, Literal(r["claimed"], datatype=XSD.double)))
        if r.get("estimate") is not None:
            g.add((rel, TL.estimate, Literal(r["estimate"], datatype=XSD.double)))
        g.add((rel, TL.verdict, Literal(r["verdict"])))
        g.add((rel, TL.status, Literal(r["status"])))
        g.add((rel, RDFS.comment, Literal(r["why"])))
        # A CROSS-DATASET CLAIM CARRIES ITS JOIN. Without the coverage, a
        # reader cannot tell whether the number describes the population either
        # file describes or some overlap of the two.
        j = r.get("join")
        if j and not j.get("refused"):
            g.add((rel, TL.joinKey, Literal(j["key"])))
            g.add((rel, TL.joinCoverage,
                   Literal(j["coverage"], datatype=XSD.double)))
            g.add((rel, TL.joinMatched,
                   Literal(j["n_matched"], datatype=XSD.integer)))

        ev = r.get("evidence", {})
        if ev.get("locator"):
            e = URIRef(f"{BASE}evidence/{i}")
            g.add((e, RDF.type, EVI.Evidence))
            g.add((e, SC.identifier, Literal(ev["locator"])))
            g.add((e, RDFS.label, Literal(ev.get("source", ""))))
            g.add((e, TL.extractedBy, Literal(ev.get("extracted_by", "unknown"))))
            g.add((rel, TL.restsOn, e))
            g.add((e, EVI.directlySupports, rel))

        if r["verdict"] == "refuted":
            v = URIRef(f"{BASE}verdict/{i}")
            g.add((v, RDF.type, EVI.Method))
            g.add((v, RDFS.label, Literal(f"recomputed {r['method']} from the data")))
            g.add((v, EVI.directlyChallenges, rel))

    return g


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "results/graph.ttl"
    g = build(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out), format="turtle")
    kinds = {str(o).split("#")[-1].split("/")[-1]
             for o in g.objects(None, RDF.type)}
    print(f"  {out}  —  {len(g)} triples, {len(kinds)} node kind(s): "
          f"{', '.join(sorted(kinds))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
