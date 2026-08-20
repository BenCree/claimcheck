"""DCAT in PSDI's profile, validated against PSDI's own SHACL shapes.

    python -m claimcheck.emit psdi <project-dir> [<out.ttl>]
    python -m claimcheck.psdi --check <project-dir>     # the conformance report
    python -m claimcheck.psdi --selftest                # the checker can fail

## Why this is a second record and not a flag on `dcat.py`

`dcat.py` emits plain W3C DCAT: the digest in `spdx:checksum`, the hand-import
in `dct:provenance`, the claim counts in our own namespace. That record is for a
catalogue that will take anything DCAT-shaped.

PSDI's is not that catalogue. **All four of their node shapes are `sh:closed`**,
so a conformant record may carry only the properties they enumerate and every
term of ours is rejected outright rather than ignored. Measured, not assumed —
`claimcheck.dcat.build(example)` against `psdi-dcat-shacl.ttl` gives **25
violations**, of which five are `sh:closed` on the dataset and four more on the
distributions, purely for carrying `cc:claimsChecked`, `spdx:checksum` and
`dcat:distribution`.

The two documents therefore cannot be one document with a switch. A PSDI record
is a small external pointer at this work; the work itself stays in the Croissant
and the plain DCAT. `tests/test_psdi.py` pins that count, so if PSDI opens the
shapes this file's reason to exist is a failing test rather than a memory.

## Three things cannot be satisfied, and pretending otherwise is the defect

* **`dcterms:identifier` must match** `^https://resources.psdi.ac.uk/…/<UUID>$`
  — an identifier **PSDI issues**. Self-certification is impossible by
  construction. So none is emitted, and the resulting `sh:minCount` failure is
  exempt.
* **`psdiDcatExt:logoURL`** and **`psdiDcatExt:displayPriority`** are mandatory
  and are catalogue *presentation*. We have no logo, and a URL invented to
  satisfy a validator is the defect, not the fix.

**An exemption covers an absent value and nothing else.** All three are
`sh:minCount` failures. Keyed on the shape alone, an exemption written for
absence also waives every *other* constraint on that shape — an invented
identifier (Pattern), a logo that 404s (Class), a non-numeric priority
(Datatype). `EXEMPT` names the shapes, `EXEMPT_CONSTRAINT` names the one
constraint kind, and both must match before anything is waived.

## No checksum, and the reason is not laziness

PSDI's `CheckSumAlgorithmPropertyShape` pins `spdx:algorithm` to a single
permitted value whose own description reads *"Currently, SHA-1 is the only
supported algorithm"*. We compute SHA-256. Emitting their algorithm IRI beside
our digest publishes a false statement about our own data in order to satisfy a
validator. `spdx:checkSum` is optional here, so omitting it conforms — and the
digest is published truthfully in the Croissant, the plain DCAT and the crate.

## Keywords are required and are never invented

`dcat:keyword` is `sh:minCount 1`. If `[project].keywords` is undeclared, none
is emitted and the report says so in a sentence that names the file and the key.
Two defaults would have made every record conform; both would have been a
statement about the dataset that nobody made.
"""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, FOAF, RDF, RDFS, SKOS, XSD

from claimcheck.namespace import BASE

DCAT = Namespace("http://www.w3.org/ns/dcat#")
DCMITYPE = Namespace("http://purl.org/dc/dcmitype/")
PSDIEXT = Namespace("http://metadata.psdi.ac.uk/psdi-dcat-ext#")
VCARD = Namespace("http://www.w3.org/2006/vcard/ns#")

#: The shapes, INSIDE the import package. The predecessor kept them at the
#: repository root, which is the checkout's layout and site-packages' in a
#: wheel, so `pip install` followed by a validate died on a path that only ever
#: existed in a git clone. `validate()` opens this at run time; it has to be a
#: path the distribution actually ships.
SHAPES = Path(__file__).resolve().parent / "resources" / "psdi"
SHAPES_FILE = SHAPES / "psdi-dcat-shacl.ttl"

#: The profile a record built here claims, written INTO the record. A document
#: that names the standard it was built against can be checked by a stranger;
#: one that does not can only be guessed at.
PROFILE = URIRef("http://metadata.psdi.ac.uk/psdi-dcat-shacl#")

#: Shapes that cannot be satisfied before PSDI issues a catalogue entry. Each is
#: a reason, not a waiver, and `tests/test_psdi.py` removes one and watches the
#: report turn red.
EXEMPT: dict[str, str] = {
    "IdentifierPropertyShape":
        "dcterms:identifier must match a https://resources.psdi.ac.uk/ URL "
        "carrying a UUID that PSDI assigns. Self-certification is impossible "
        "by construction.",
    "LogoURLPropertyShape":
        "psdiDcatExt:logoURL is catalogue presentation. We have no logo, and a "
        "URL invented to satisfy a validator is the defect, not the fix.",
    "DisplayPriorityPropertyShape":
        "psdiDcatExt:displayPriority orders tiles in PSDI's browser. It has no "
        "truth value for a resource PSDI has not catalogued.",
}

#: The ONE constraint kind an exemption may cover. See the module docstring: an
#: exemption for a missing value that also covers an invented one publishes the
#: invention under the exemption's reason.
EXEMPT_CONSTRAINT = "MinCountConstraintComponent"

#: COAR access rights — the four IRIs `AccessRightsOptionalPropertyShape`
#: permits, read off its own `sh:in` list rather than remembered.
_ACCESS = {
    "open": "http://purl.org/coar/access_right/c_abf2",
    "open access": "http://purl.org/coar/access_right/c_abf2",
    "restricted": "http://purl.org/coar/access_right/c_16ec",
    "embargoed": "http://purl.org/coar/access_right/c_f1cf",
    "metadata only": "http://purl.org/coar/access_right/c_14cb",
}


class UndeclarableValue(ValueError):
    """A declared value this profile cannot carry, refused rather than dropped.

    Silently omitting it leaves the record valid and the declaration ignored:
    the author writes `access_rights = "public"`, the command exits 0, and the
    field is simply not there.
    """


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(text).lower()).strip("-")


def build(root: Path | str) -> Graph:
    """A PSDI-profile `dcat:Dataset` with one `dcat:Distribution` per file.

    Reads `[project]` for what claimcheck already declares, and an optional
    `[psdi]` table for the catalogue-editorial fields that have no other home.
    Everything optional is omitted when undeclared: a catalogue entry that
    invents its own publisher is worse than one that admits it has none.
    """
    root = Path(root)
    ctx = tomllib.loads((root / "context.toml").read_text())
    proj = ctx["project"]
    meta = ctx.get("psdi", {})

    g = Graph()
    for pfx, ns in (("dcat", DCAT), ("dcterms", DCTERMS), ("dcmitype", DCMITYPE),
                    ("psdiDcatExt", PSDIEXT), ("foaf", FOAF), ("vcard", VCARD),
                    ("skos", SKOS)):
        g.bind(pfx, ns)

    ds = URIRef(f"{BASE}dataset/{proj['name']}")
    g.add((ds, RDF.type, DCAT.Dataset))
    # THE RECORD SAYS WHICH PROFILE IT CLAIMS. Without this, a checker sweeping
    # a tree has to guess, and guessing means either validating plain DCAT
    # against shapes it never claimed — reporting failure against a standard the
    # author never invoked — or validating nothing. `claimcheck.conform` reads
    # this, and `dcat.py`'s plain record deliberately does not carry it.
    g.add((ds, DCTERMS.conformsTo, PROFILE))
    g.add((PROFILE, RDF.type, DCTERMS.Standard))
    # StandardNodeShape is sh:closed and permits `rdfs:label` ONLY, minCount 1.
    # A controlled IRI dropped in bare is a violation however correct it is.
    g.add((PROFILE, RDFS.label,
           Literal("PSDI DCAT profile (SHACL)", datatype=XSD.string)))
    # xsd:string throughout, never a language-tagged literal: the shapes pin the
    # datatype, and a lang string is a different RDF term to a SHACL engine.
    g.add((ds, DCTERMS.title,
           Literal(proj.get("title") or proj["name"], datatype=XSD.string)))
    g.add((ds, RDFS.label, Literal(proj["name"], datatype=XSD.string)))
    # `context` is the long form; the title is a description when there is no
    # longer one. Neither is invented — both are text somebody wrote.
    desc = (proj.get("context") or "").strip() or proj.get("title") or proj["name"]
    g.add((ds, DCTERMS.description, Literal(desc, datatype=XSD.string)))
    for kw in proj.get("keywords", []):
        g.add((ds, DCAT.keyword, Literal(str(kw), datatype=XSD.string)))
    # dcterms:type comes from the DCMI Type Vocabulary — twelve values, of which
    # `Dataset` is ours — and every controlled IRI needs its label.
    g.add((ds, DCTERMS.type, DCMITYPE.Dataset))
    g.add((DCMITYPE.Dataset, RDF.type, SKOS.Concept))
    g.add((DCMITYPE.Dataset, SKOS.prefLabel, Literal("Dataset", datatype=XSD.string)))
    if proj.get("license"):
        lic = URIRef(str(proj["license"]))
        g.add((ds, DCTERMS.license, lic))
        g.add((lic, RDF.type, DCTERMS.LicenseDocument))
    if proj.get("version"):
        g.add((ds, DCAT.version, Literal(str(proj["version"]), datatype=XSD.string)))
    if proj.get("date_published"):
        # xsd:dateTime, not xsd:date. IssuedPropertyShape pins it, and a date is
        # not a dateTime however much it reads like one.
        stamp = str(proj["date_published"])
        g.add((ds, DCTERMS.issued,
               Literal(stamp if "T" in stamp else stamp + "T00:00:00",
                       datatype=XSD.dateTime)))
    _editorial(g, ds, proj, meta)

    for d in ctx["datasets"]:
        for rel in d["files"]:
            p = root / rel
            dist = URIRef(f"{BASE}distribution/{rel.replace('/', '_')}")
            g.add((dist, RDF.type, DCAT.Distribution))
            # PSDI's DistributionPropertyShape has `sh:path dcat:Distribution` —
            # the CLASS IRI, where W3C DCAT's property is `dcat:distribution`.
            # Almost certainly an upstream typo. ONLY THEIRS IS EMITTED, and
            # that is measured rather than chosen: DatasetNodeShape is
            # `sh:closed`, so adding the correct property alongside makes the
            # record NON-conformant. You conform to what they published or not
            # at all. Worth reporting upstream; not worth guessing around.
            g.add((ds, DCAT.Distribution, dist))
            g.add((dist, DCTERMS.title,
                   Literal(Path(rel).name, datatype=XSD.string)))
            # An IRI, not a literal path: DownloadURLPropertyShape pins
            # `sh:nodeKind sh:IRI` and `sh:class rdfs:Resource`, so the relative
            # path `dcat.py` publishes is four violations here.
            access = URIRef(f"{BASE}artifact/{rel}")
            g.add((dist, DCAT.downloadURL, access))
            g.add((access, RDF.type, RDFS.Resource))
            if p.is_file():
                g.add((dist, DCAT.byteSize,
                       Literal(p.stat().st_size,
                               datatype=XSD.nonNegativeInteger)))
            # NO CHECKSUM. See the module docstring: their profile permits SHA-1
            # only, we compute SHA-256, and `spdx:checkSum` is optional.
    return g


def _editorial(g: Graph, ds: URIRef, proj: dict, meta: dict) -> None:
    """The optional metadata that makes a record findable rather than valid.

    Each field is modelled the way the shape asks for it. A catalogue profile
    pinning `sh:class vcard:Kind` is telling you the shape of the answer, not
    being fussy: a literal dropped there produces a record that says nothing a
    consumer can follow.
    """
    if meta.get("landing_page"):
        page = URIRef(str(meta["landing_page"]))
        g.add((ds, DCAT.landingPage, page))
        g.add((page, RDF.type, FOAF.Document))
        # FoafDocumentNodeShape is sh:closed and requires `rdfs:label`, so the
        # obvious two-line version — an IRI and its type — is non-conformant.
        g.add((page, RDFS.label,
               Literal(str(meta.get("landing_page_label") or meta["landing_page"]),
                       datatype=XSD.string)))

    if meta.get("access_rights"):
        raw = str(meta["access_rights"]).strip()
        iri = _ACCESS.get(raw.lower()) or (raw if raw.startswith("http") else None)
        if iri is None:
            raise UndeclarableValue(
                f"[psdi].access_rights = {raw!r} is not one of "
                f"{sorted(set(_ACCESS))} and is not a COAR IRI. This profile "
                f"permits four values.")
        node = URIRef(iri)
        g.add((ds, DCTERMS.accessRights, node))
        g.add((node, RDF.type, DCTERMS.RightsStatement))
        g.add((node, RDFS.label, Literal(raw, datatype=XSD.string)))

    # publisher: an IRI of class foaf:Agent. `dcat.py` publishes a name, which
    # is right for plain DCAT and is a nodeKind violation here.
    if meta.get("publisher_url") or proj.get("publisher"):
        url = meta.get("publisher_url")
        who = URIRef(str(url)) if url else URIRef(f"{ds}#publisher")
        g.add((ds, DCTERMS.publisher, who))
        g.add((who, RDF.type, FOAF.Agent))
        # FoafAgentTargetClassShape is sh:closed and permits foaf:name ONLY. An
        # rdfs:label alongside it — the obvious thing to add — makes the node
        # non-conformant.
        g.add((who, FOAF.name,
               Literal(str(proj.get("publisher") or url), datatype=XSD.string)))

    # THIS PROFILE'S CONTACT CARD HAS NO NAME ON IT, measured rather than
    # assumed: `VCardNodeShape` is `sh:closed` and permits `vcard:hasEmail` and
    # `vcard:hasURL`, full stop. `vcard:fn` — the obvious place for a name — is
    # rejected outright. Refused rather than dropped, because an author who
    # writes a name, sees exit 0, and never learns the field went nowhere is the
    # failure this class exists for.
    if meta.get("contact_name") and not meta.get("contact_url"):
        raise UndeclarableValue(
            "[psdi].contact_name cannot be carried: PSDI's VCardNodeShape is "
            "sh:closed and permits vcard:hasEmail and vcard:hasURL only, so a "
            "name on the card makes the record non-conformant. Put a page "
            "about the person in [psdi].contact_url, or drop the name.")
    if meta.get("contact_point") or meta.get("contact_url"):
        card = URIRef(f"{ds}#contact")
        g.add((ds, DCAT.contactPoint, card))
        g.add((card, RDF.type, VCARD.Kind))
        if meta.get("contact_point"):
            raw = str(meta["contact_point"]).strip()
            # HasEmailPropertyShape pins `sh:nodeKind sh:Literal` with datatype
            # xsd:string — a `mailto:` IRI, which is what DCAT-AP uses, is
            # refused.
            g.add((card, VCARD.hasEmail,
                   Literal(raw.removeprefix("mailto:"), datatype=XSD.string)))
        if meta.get("contact_url"):
            g.add((card, VCARD.hasURL, URIRef(str(meta["contact_url"]))))

    for c in meta.get("creators", []):
        orcid, nm = c.get("orcid"), c.get("name")
        if not (orcid or nm):
            continue
        # CreatorPropertyShape pins `sh:nodeKind sh:IRI`, so a name alone cannot
        # be the value. An ORCID is the right IRI for a person; without one the
        # creator gets a local IRI under this dataset, which is stable within
        # the record and honest about being no more than that.
        who = (URIRef(orcid if str(orcid).startswith("http")
                      else f"https://orcid.org/{orcid}") if orcid
               else URIRef(f"{ds}#creator-{_slug(nm)}"))
        g.add((ds, DCTERMS.creator, who))
        g.add((who, RDF.type, FOAF.Agent))
        g.add((who, FOAF.name, Literal(nm or str(who), datatype=XSD.string)))


def render(root: Path | str) -> str:
    """The emitter interface, shared with `croissant`, `graph` and `dcat`."""
    return build(root).serialize(format="turtle")


# ---------------------------------------------------------------------------
# conformance
# ---------------------------------------------------------------------------

@dataclass
class Conformance:
    conforms: bool
    exempt_violations: list = field(default_factory=list)
    real_violations: list = field(default_factory=list)
    shapes_commit: str = ""
    #: True when pyshacl CRASHED rather than reaching a verdict. Without it, an
    #: exception is indistinguishable from "your record does not conform" — a
    #: sentence about the data, when the true one is about the validator.
    crashed: bool = False
    detail: str = ""

    @property
    def passed(self) -> bool:
        """A verdict was reached and nothing outside the enumerated exemptions
        failed. **A crash is never a pass**: no violations were found because
        nothing looked."""
        return not self.crashed and not self.real_violations

    def as_dict(self) -> dict:
        return {
            "validator": "pyshacl against PSDI's published shapes",
            "shapes_commit": self.shapes_commit,
            "fully_conforms": self.conforms,
            "passed_excluding_exemptions": self.passed,
            "crashed": self.crashed,
            "detail": self.detail,
            "exemptions": EXEMPT,
            "exemption_scope": EXEMPT_CONSTRAINT,
            "exempt_violations": self.exempt_violations[:20],
            "real_violations": self.real_violations[:20],
        }


def shapes_commit() -> str:
    """The upstream commit the vendored shapes are pinned to, or ''."""
    f = SHAPES / "PROVENANCE.json"
    if not f.is_file():
        return ""
    return str(json.loads(f.read_text()).get("upstream_commit", ""))[:12]


def validate(data: Graph, exempt: dict | None = None) -> Conformance:
    """Validate against PSDI's shapes, separating exempt from real failures."""
    import pyshacl

    exempt = EXEMPT if exempt is None else exempt
    try:
        shapes = Graph().parse(SHAPES_FILE, format="turtle")
        conforms, results, _text = pyshacl.validate(
            data, shacl_graph=shapes, inference="none", abort_on_first=False)
    except Exception as e:                                   # noqa: BLE001
        # A crash is not a verdict.
        return Conformance(conforms=False, crashed=True,
                           detail=f"pyshacl did not reach a verdict — "
                                  f"{type(e).__name__}: {e}"[:400],
                           shapes_commit=shapes_commit())

    SH = Namespace("http://www.w3.org/ns/shacl#")
    exempt_hits: list[str] = []
    real: list[str] = []
    for r in results.subjects(RDF.type, SH.ValidationResult):
        src = results.value(r, SH.sourceShape)
        msg = results.value(r, SH.resultMessage)
        comp = results.value(r, SH.sourceConstraintComponent)
        name = str(src).rsplit("#", 1)[-1].rsplit("/", 1)[-1] if src else "?"
        kind = str(comp).rsplit("#", 1)[-1].rsplit("/", 1)[-1] if comp else "?"
        line = f"{name}: {msg}"
        if name == "KeywordPropertyShape" and kind == EXEMPT_CONSTRAINT:
            line += ("  [declare `keywords = [...]` under [project] in "
                     "context.toml — this one is not invented for you]")
        waived = name in exempt and kind == EXEMPT_CONSTRAINT
        if not waived and name in exempt:
            line = (f"{line}  [NOT EXEMPT: the exemption for {name} covers a "
                    f"missing value ({EXEMPT_CONSTRAINT}), and this is a "
                    f"{kind} violation of a value that is present]")
        (exempt_hits if waived else real).append(line)
    return Conformance(conforms=bool(conforms), exempt_violations=exempt_hits,
                       real_violations=real, shapes_commit=shapes_commit())


def check(root: Path | str) -> Conformance:
    """Build the record for a project and validate it in one call."""
    return validate(build(root))


def check_file(path: Path | str) -> Conformance:
    """Validate a Turtle record already on disk.

    Here rather than in the caller so that `claimcheck.conform` needs no graph
    library of its own: a module that decides *whether* to reach for rdflib must
    not import it to decide. `tests/test_dependencies.py` enforces that an
    extra's packages are imported only by the modules that declare the extra,
    and `conform` declares none.
    """
    try:
        g = Graph().parse(Path(path), format="turtle")
    except Exception as e:                                   # noqa: BLE001
        return Conformance(conforms=False, crashed=False,
                           real_violations=[f"will not parse as Turtle — "
                                            f"{type(e).__name__}: {e}"[:200]],
                           shapes_commit=shapes_commit())
    return validate(g)


def _selftest() -> int:
    """Two-sided: the checker must refuse a bad record and accept a good one.

    A conformance checker that cannot fail is the failure mode this whole
    package is written against, so the proof ships with the checker rather than
    only in the test suite — someone running an installed wheel has no tests.
    """
    ok = True
    good = Graph().parse(data=_KNOWN_GOOD, format="turtle")
    r = validate(good)
    print(f"  {'PASS' if r.passed else 'FAIL'}  a conformant record passes"
          f"{'' if r.passed else '  ' + str(r.real_violations[:3])}")
    ok &= r.passed
    # known-bad 1: an INVENTED identifier. The exemption is for an absent value;
    # if it also waives a Pattern violation, a made-up PSDI URL publishes clean.
    bad = Graph().parse(data=_KNOWN_GOOD, format="turtle")
    bad.add((URIRef(f"{BASE}dataset/selftest"), DCTERMS.identifier,
             Literal("https://resources.psdi.ac.uk/data/not-a-real-uuid",
                     datatype=XSD.string)))
    r = validate(bad)
    caught = not r.passed and any("NOT EXEMPT" in v for v in r.real_violations)
    print(f"  {'PASS' if caught else 'FAIL'}  an invented identifier is refused "
          f"by the exemption's scope")
    ok &= caught
    # known-bad 2: keywords removed. Required, and never defaulted.
    bare = Graph().parse(data=_KNOWN_GOOD, format="turtle")
    bare.remove((None, DCAT.keyword, None))
    r = validate(bare)
    caught = not r.passed and any("Keyword" in v for v in r.real_violations)
    print(f"  {'PASS' if caught else 'FAIL'}  a record with no keyword fails")
    ok &= caught
    print(f"\nselftest: {'PASS' if ok else 'FAIL'}  "
          f"(shapes {shapes_commit() or 'unpinned'})")
    return 0 if ok else 1


#: The smallest record that satisfies PSDI's shapes modulo the three exemptions.
#: Written out rather than built from a fixture project so that `--selftest`
#: works from an installed wheel with no example tree on disk.
_KNOWN_GOOD = f"""
@prefix dcat: <http://www.w3.org/ns/dcat#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix dcmitype: <http://purl.org/dc/dcmitype/> .
@prefix psdiShacl: <http://metadata.psdi.ac.uk/psdi-dcat-shacl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<{BASE}dataset/selftest> a dcat:Dataset ;
    dcterms:title "selftest"^^xsd:string ;
    dcterms:description "a record built to prove this checker can fail"^^xsd:string ;
    rdfs:label "selftest"^^xsd:string ;
    dcat:keyword "chemistry"^^xsd:string ;
    dcterms:type dcmitype:Dataset ;
    dcterms:conformsTo psdiShacl: ;
    dcat:Distribution <{BASE}distribution/x_csv> .

psdiShacl: a dcterms:Standard ; rdfs:label "PSDI DCAT profile (SHACL)"^^xsd:string .

dcmitype:Dataset a skos:Concept ; skos:prefLabel "Dataset"^^xsd:string .

<{BASE}distribution/x_csv> a dcat:Distribution ;
    dcterms:title "x.csv"^^xsd:string ;
    dcat:downloadURL <{BASE}artifact/x.csv> .

<{BASE}artifact/x.csv> a rdfs:Resource .
"""


def main(argv: list[str] | None = None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if "--selftest" in a:
        return _selftest()
    if "--check" in a:
        a.remove("--check")
        root = Path(a[0]) if a else Path.cwd()
        r = check(root)
        print(json.dumps(r.as_dict(), indent=2))
        if r.crashed:
            print("\npyshacl did not reach a verdict. This is a sentence about "
                  "the validator, not about your record.", file=sys.stderr)
        return 0 if r.passed else 1
    from claimcheck.emit import main as emit_main
    return emit_main(["psdi", *a])


if __name__ == "__main__":
    raise SystemExit(main())
