"""W3C DCAT — the catalogue card, so a portal can find the dataset.

    python -m claimcheck.emit dcat <project-dir> [<out.ttl>]

## What DCAT is, and why it is not a rival to Croissant

They answer different questions and overlap only on the cover page.

| | Croissant | DCAT |
|---|---|---|
| the question | **how do I READ this?** | **how do I FIND this?** |
| the audience | a loader, a training script | a catalogue, a search index, a repository |
| the content | record sets, fields, data types, extraction | title, publisher, licence, keywords, distributions |
| who speaks it | ML tooling | data.gov, data.europa.eu, institutional portals, PSDI |

Croissant will tell you that column 3 is a float you extract from `data.csv`.
DCAT will not. DCAT will get the dataset listed alongside ten thousand others
with a licence a lawyer can read. Croissant will not. Emitting both is not
redundancy — it is the same dataset appearing in two places that do not talk to
each other.

## What this maps into standard terms rather than ours

The point of emitting DCAT at all is that some of what we hold has a **standard
slot**, and a fact in a standard slot is one a stranger's tooling can use:

* the digest -> `spdx:checksum`, which DCAT-AP already defines. Ours becomes
  everybody's.
* where a manual dataset came from -> `dct:provenance`. A catalogue can show
  "exported by hand from X on Y" without knowing anything about this package.
* licence, title, keywords, byte size, media type -> the obvious DCAT terms.

**Units and claims stay in our namespace, because DCAT has no slot for them.**
That is not a gap in DCAT; a catalogue card is not the place to say that a
correlation somebody asserted was refuted at 95%. Saying so plainly is better
than inventing a DCAT term and implying it is standard.

## What this does NOT claim

**It is not validated against PSDI's profile, and it cannot be.** That decision
has now been taken and it went the other way: `claimcheck/psdi.py` emits a
*second, separate* record under PSDI's shapes and validates it with `pyshacl`.
The two cannot be one document. All four of PSDI's node shapes are `sh:closed`,
so `cc:claimsChecked`, `spdx:checksum` and even W3C's own `dcat:distribution`
are rejected outright rather than ignored — measured, 25 violations for this
module's output. Plain DCAT is what a catalogue that will take DCAT wants; PSDI
is a small external pointer at this work, in their profile, on their terms.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, FOAF, RDF, XSD

from claimcheck.namespace import BASE, NS

DCAT = Namespace("http://www.w3.org/ns/dcat#")
SPDX = Namespace("http://spdx.org/rdf/terms#")
CC = Namespace(NS)


def build(root: Path | str) -> Graph:
    root = Path(root)
    ctx = tomllib.loads((root / "context.toml").read_text())
    proj = ctx["project"]
    claims_file = root / "results" / "claims.json"
    claims = (json.loads(claims_file.read_text())
              if claims_file.is_file() else {"relationships": []})

    g = Graph()
    for pfx, ns in (("dcat", DCAT), ("dct", DCTERMS), ("spdx", SPDX),
                    ("foaf", FOAF), ("cc", CC)):
        g.bind(pfx, ns)

    ds = URIRef(f"{BASE}dataset/{proj['name']}")
    g.add((ds, RDF.type, DCAT.Dataset))
    g.add((ds, DCTERMS.title, Literal(proj.get("title", proj["name"]))))
    g.add((ds, DCTERMS.identifier, Literal(proj["name"])))
    if proj.get("context"):
        g.add((ds, DCTERMS.description, Literal(proj["context"].strip())))
    for key, term in (("license", DCTERMS.license),
                      ("date_published", DCTERMS.issued),
                      ("version", DCTERMS.hasVersion)):
        if proj.get(key):
            v = proj[key]
            g.add((ds, term, URIRef(v) if str(v).startswith("http")
                   else Literal(v)))
    for kw in proj.get("keywords", []):
        g.add((ds, DCAT.keyword, Literal(kw)))
    if proj.get("publisher"):
        pub = URIRef(f"{BASE}agent/publisher")
        g.add((pub, RDF.type, FOAF.Agent))
        g.add((pub, FOAF.name, Literal(proj["publisher"])))
        g.add((ds, DCTERMS.publisher, pub))

    for d in ctx["datasets"]:
        for rel in d["files"]:
            p = root / rel
            dist = URIRef(f"{BASE}distribution/{rel.replace('/', '_')}")
            g.add((dist, RDF.type, DCAT.Distribution))
            g.add((dist, DCTERMS.title, Literal(Path(rel).name)))
            g.add((dist, DCAT.downloadURL, Literal(rel)))
            g.add((dist, DCAT.mediaType, Literal("text/csv")))
            if p.is_file():
                g.add((dist, DCAT.byteSize,
                       Literal(p.stat().st_size, datatype=XSD.nonNegativeInteger)))
                # THE DIGEST INTO A STANDARD SLOT. DCAT-AP defines
                # spdx:checksum for exactly this, so the hash stops being ours
                # and becomes something a stranger's catalogue can check.
                from claimcheck.datasets import sha256
                ck = URIRef(f"{BASE}checksum/{rel.replace('/', '_')}")
                g.add((ck, RDF.type, SPDX.Checksum))
                g.add((ck, SPDX.algorithm, SPDX.checksumAlgorithm_sha256))
                g.add((ck, SPDX.checksumValue, Literal(sha256(p))))
                g.add((dist, SPDX.checksum, ck))
            # WHERE IT CAME FROM, in DCAT's own term. A catalogue can show this
            # without knowing anything about this package.
            if d["origin"] == "manual":
                g.add((dist, DCTERMS.provenance, Literal(
                    f"Manually imported. Source: {d.get('source')}. "
                    f"Retrieved: {d.get('retrieved')}.")))
            else:
                g.add((dist, DCTERMS.provenance, Literal(
                    "Produced by a rule in this project's Snakemake workflow; "
                    "the inputs, code and environment are in its metadata "
                    "record.")))
            g.add((ds, DCAT.distribution, dist))

    # OUR NAMESPACE FOR WHAT DCAT HAS NO SLOT FOR. Inventing `dcat:refuted`
    # would imply a standard that does not exist.
    refuted = sum(1 for r in claims["relationships"] if r["verdict"] == "refuted")
    unver = sum(1 for r in claims["relationships"]
                if r["verdict"] == "unverifiable")
    g.add((ds, CC.claimsChecked,
           Literal(len(claims["relationships"]), datatype=XSD.integer)))
    g.add((ds, CC.claimsRefuted, Literal(refuted, datatype=XSD.integer)))
    g.add((ds, CC.claimsUnverifiable, Literal(unver, datatype=XSD.integer)))
    return g


def render(root: Path | str) -> str:
    """The emitter interface: text, so `claimcheck.emit` need not know Turtle."""
    return build(root).serialize(format="turtle")


def readback(path: Path | str) -> tuple[list[str], int]:
    """Can a consumer GET the distributions this record names?

    Returns `(complaints, dereferenced)`. **The count is not decoration.** A
    record whose every `downloadURL` is an absolute IRI has nothing here to
    dereference, so an empty complaint list means "nothing was checked", not
    "everything is fine" — and a checker reporting a pass over zero checks is
    the failure mode this whole package is written against. The caller reports
    UNKNOWN on a zero.

    Parsing is not reading. A catalogue record that parses, carries every term
    it promises and points at nothing is the exact shape of the Croissant defect
    on record here: `contentUrl` written relative to the project root while the
    consumer resolves it against the folder holding the file.

    **The base is the document's own folder, and that is not arbitrary.** A
    relative `dcat:downloadURL` declares no base of its own. Somebody who has
    fetched this `.ttl` and nothing else holds exactly one base — where the file
    is. If the path resolves from somewhere else and not from here, the record
    is unresolvable to its own reader, and this says which.

    Absolute IRIs are not fetched. A conformance report that depends on the
    network says something different on a train.
    """
    path = Path(path)
    here = path.parent
    try:
        g = Graph().parse(path, format="turtle")
    except Exception as e:                                   # noqa: BLE001
        return ([f"the record will not parse — {type(e).__name__}: {e}"[:200]], 0)

    bad: list[str] = []
    n = 0
    local = 0
    for _dist, _p, o in g.triples((None, DCAT.downloadURL, None)):
        n += 1
        raw = str(o)
        if isinstance(o, URIRef) or "://" in raw:
            continue                     # remote; not ours to dereference
        local += 1
        if (here / raw).is_file():
            continue
        # Where DOES it resolve? Naming the base that works turns "broken" into
        # a one-line fix rather than a hunt.
        found = next((str(b) for b in (here.parent, here.parent.parent,
                                       Path.cwd())
                      if (b / raw).is_file()), None)
        bad.append(
            f"dcat:downloadURL {raw!r} does not resolve from {here}, which is "
            f"the only base a consumer holding this file has" +
            (f" — it resolves from {found}, and the record does not say so"
             if found else " — and it resolves from no nearby directory either"))
    if not n:
        bad.append("this record names no dcat:downloadURL, so it tells a "
                   "consumer where nothing is")
    return bad, local


def main(argv: list[str] | None = None) -> int:
    from claimcheck.emit import main as emit_main
    a = argv if argv is not None else sys.argv[1:]
    return emit_main(["dcat", *a])


if __name__ == "__main__":
    raise SystemExit(main())
