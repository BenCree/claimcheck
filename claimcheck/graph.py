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

## Mergeable, which it was not until 2026-11-20

**Every node that belongs to one experiment sits under `claimcheck/exp/<project
name>/`, and a relationship is named by its CLAIM ID rather than by its
position.** Before that change a relationship was `claimcheck/relationship/N`
counted from 1, so 340 of the 345 graphs emitted under `projects/` minted
`relationship/1` and the other 5 declared no relationship at all; a file was
`claimcheck/file/<name>`, and `file/edges_csv` was minted by 15 experiments
carrying 13 distinct sha256 digests; a column was
`claimcheck/records/<dataset>.<col>`, and `records/truncation.n_within_kept` by
9. All four counts are from the backup at
`/home/ben/code/shared/corpus/backup/graph_ttl_2026-11-20`, taken before
anything was re-emitted.
Loading two graphs into one store gave single nodes the verdicts, estimates
and digests of unrelated claims **and raised no error**, so the corpus the lab
reads to decide what to run next could not be assembled, and no query could ask
for a claim by the id that `context.toml`, the LOGs and people use.

| IRI | scope | why |
|---|---|---|
| `exp/<p>/relationship/<claim id>` | one experiment | the verdict, estimate and interval of one claim |
| `exp/<p>/evidence/<claim id>`, `exp/<p>/verdict/<claim id>` | one experiment | attached to that claim |
| `exp/<p>/file/<name>`, `exp/<p>/records/<ds>.<col>` | one experiment | the digest and the column are this project's |
| `exp/<p>/claim/<n>` | one experiment | a `[belief]` row, numbered by position in that table |
| `project/<p>` | one experiment | the experiment itself |
| `node/<id>`, `artefact/<name>` | **global, deliberately** | two experiments naming `heavy_atom_count` mean the same concept, and merging them is what joins the corpus |

The experiment segment is first so that a reader matching `/relationship/`,
`/claim/`, `/file/` or `/project/` keeps matching;
`shared/dashboard/build_dashboard.py` does exactly that, matching `/claim/`,
`/project/` and `/file/` by pattern, and its 155 tests pass unchanged against
the new IRIs. The two other readers of a graph's CONTENT in this lab are
`projects/z-ml/experiments/freeflow_intake/scripts/kg_queries.py`, which builds
`node/<id>` IRIs by hand and so depends on those staying global, and that
experiment's `tests/test_kg.py`; both were re-run and gave the same answers.

`claimcheck.corpus` merges the graphs and refuses quietly broken merges: it
reports any IRI minted by more than one experiment and any single-valued
predicate (`tl:verdict`, `tl:estimate`, `sc:sha256`) that comes back with two
values on one subject.

## The belief layer, in EVI's verbs rather than ours

A project with a `[belief]` table (`claimcheck.belief`) also gets the graph of
what rests on what, and the edges use **`evi:supports` and
`evi:directlyChallenges`**, not private predicates. With `claimcheck.evi`'s two
axioms loaded — `supports` transitive, `indirectlyChallenges` the chain
`( directlyChallenges supports )` — a reasoner derives from

    verdict --directlyChallenges--> relationship --supports--> claim
            --supports--> node --supports--> artefact

everything a refutation puts in doubt, with no traversal written here. EVI
intends `supports` to run through more than Claims: its own subproperty
hierarchy puts `generated`, `usedBy` and `derivedTo` under `directlySupports`
so that *"warrant can propagate through Activities and Agents"*.

**A reasoner gets a subset of `belief.caveats_on`, and both are right.** Two
things it cannot have, neither of them a bug:

* **Doubt with no challenger.** An `assumed` or `untested` claim carries a
  caveat because nobody has looked. Under open-world semantics no axiom
  entails that, and nothing here fakes one: those claims are emitted with
  their status and no `directlyChallenges` edge.
* **The peer edge.** `shares_approximation_with` is symmetric and not
  transitive, so it is not rendered as `evi:supports` — a transitive property
  would close over it and merge every peer into one class. It is in the graph
  as a claim; it is not in the chain.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, URIRef

from claimcheck.belief import BeliefGraph, bearers
from claimcheck.evi import EVI_ONTOLOGY
from claimcheck.namespace import BASE, NS
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

SC = Namespace("https://schema.org/")
CR = Namespace("http://mlcommons.org/croissant/")
EVI = Namespace("https://w3id.org/EVI#")
TL = Namespace(NS)


def _seg(s: object) -> str:
    """One path segment of an IRI, percent-encoded.

    Every claim id in this lab's corpus (19,592 of them on 2026-11-20) is
    already `[A-Za-z0-9_.-]`, so this changes nothing today. It is here so that
    the first id with a space or a slash in it emits a valid IRI instead of
    Turtle that will not reparse.
    """
    return quote(str(s), safe="")


def _num(v: object) -> float | None:
    """`v` as a float, or None when it is not a number.

    **An xsd:double literal that is not a double is worse than a missing one.**
    `projects/affinity-scoring/experiments/miel_mopac_verification_repo/outputs/miel_fepbench_4a/results/graph.ttl`
    carries `tl:claimed "None"^^xsd:double`, because its `claims.json` holds the
    STRING "None" and the emitter passed it to `Literal(..., datatype=double)`
    without looking. rdflib parses that file and warns `could not convert
    string to float: 'None'` once per load, and the triple is ill-typed: a
    SPARQL filter on it is undefined rather than false, so a query for "claims
    claimed above 0.5" silently omits it instead of reporting it. Here the
    triple is left out, which an open-world reader reads correctly as "no
    claimed value stated".
    """
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)                       # str included: "0.5" is a number
    except (TypeError, ValueError):
        return None
    return f


def _experiment(name: str) -> str:
    """The IRI prefix that every node belonging to ONE experiment sits under.

    **This is what makes the graphs mergeable, and it is the whole point of the
    `exp/` segment.** Until 2026-11-20 a relationship was
    `claimcheck/relationship/<position in context.toml>`, a file was
    `claimcheck/file/<name>` and a column `claimcheck/records/<dataset>.<col>`,
    none of which carried the experiment. Measured across the 345 emitted
    graphs under `projects/`: `relationship/1` was minted by 340 of them,
    `file/edges_csv` by 15 experiments with 13 distinct sha256 digests, and
    `records/truncation.n_within_kept` by 9. Loading two into one store
    therefore gave one node the verdicts, estimates and digests of unrelated
    claims AND RAISED NO ERROR, so the claim to column to file to digest
    traversal that `tests/test_graph.py` exists for returned nonsense in a
    merged store while passing on each graph alone.

    The segment goes BEFORE the kind (`exp/<name>/relationship/<id>`, not
    `relationship/<name>/<id>`) so that a reader matching on `/relationship/`,
    `/claim/` or `/file/` still matches. `shared/dashboard/build_dashboard.py`
    does exactly that.
    """
    return f"{BASE}exp/{_seg(name)}/"


def _claim_key(r: dict, used: set[str]) -> tuple[str, bool]:
    """The id a relationship's IRI is built from, and whether it was declared.

    A DECLARED `id` is used as it stands, because the point is that a query can
    ask for `A2_esr1_rigid_cnnaff_auc` by the name people, LOGs and
    `context.toml` use.

    Where none is declared the id is derived from the DECLARATION's content,
    never from its position: 25 of the 345 adjudicated directories predate
    `id` (`relate_sha256` is absent from their `claims.json`) and their
    `[[relationships]]` blocks carry no id at all. A content hash means that
    appending a claim leaves every other IRI untouched and inserting one does
    not silently rewrite history, which a position cannot promise. Two
    relationships whose declarations are identical in every hashed field do
    exist (`md-ensembles/experiments/t_equil` registers the same x, y, kind and
    method at `claimed = 1.0` and at `claimed = 0.0`), so `claimed` is hashed
    and a remaining exact duplicate is suffixed rather than silently merged.
    """
    cid = str(r.get("id") or "").strip()
    if cid:
        return cid, True
    # `baseline` joins the hashed fields ONLY when the claim has one, so that two three-column claims differing only
    # in what the score is compared against get different IRIs while every existing unnamed claim's IRI is unchanged
    # (appending to the list unconditionally would rewrite every one of them, 2026-11-20).
    key = json.dumps(
        [r.get("kind"), r.get("method"), r.get("x"), r.get("y"),
         r.get("claimed"), r.get("claimed_range")]
        + ([r["baseline"]] if r.get("baseline") else []),
        sort_keys=True, default=str)
    stem = "unnamed-" + hashlib.sha256(key.encode()).hexdigest()[:12]
    cid, n = stem, 1
    while cid in used:
        n += 1
        cid = f"{stem}-{n}"
    return cid, False


def build(root: Path | str,
          owner_roots: tuple[Path, ...] = ()) -> Graph:
    """`owner_roots`: extra directories in which a `defined_in` belief owner may
    be found, beside the project's own sibling directory. See
    `claimcheck.belief._owner_dir`."""
    root = Path(root)
    ctx = tomllib.loads((root / "context.toml").read_text())
    claims = json.loads((root / "results/claims.json").read_text())
    croissant = json.loads((root / "results/croissant.json").read_text())

    g = Graph()
    for pfx, n in (("sc", SC), ("cr", CR), ("evi", EVI), ("tl", TL)):
        g.bind(pfx, n)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)

    name = ctx["project"]["name"]
    #: Every node that belongs to this experiment and nothing else hangs off
    #: here. See `_experiment`.
    E = _experiment(name)

    proj = URIRef(f"{BASE}project/{_seg(name)}")
    g.add((proj, RDF.type, SC.Dataset))
    g.add((proj, SC.name, Literal(name)))

    # --- one node per FILE, carrying its digest and its ORIGIN --------------
    files = {}
    for dist in croissant["distribution"]:
        f = URIRef(f"{E}file/{_seg(dist['name'])}")
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
            u = URIRef(f"{E}records/{_seg(did)}.{_seg(col)}")
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
    #: `"x ~ y"` -> the relationship's IRI, so a belief claim that names one in
    #: `tested_by` can be joined to the recomputation that settled it.
    by_pair: dict[str, URIRef] = {}
    used: set[str] = set()
    for r in claims["relationships"]:
        cid, declared = _claim_key(r, used)
        used.add(cid)
        rel = URIRef(f"{E}relationship/{_seg(cid)}")
        by_pair[f"{r['x']} ~ {r['y']}"] = rel
        g.add((rel, RDF.type, TL.Relationship))
        # THE ID AS A LITERAL TOO, so a query can ask for a claim by the name
        # that humans, LOGs and `context.toml` use rather than by reproducing
        # the IRI construction, and so a merged store can say which experiment
        # a row came from without parsing a URI.
        g.add((rel, TL.claimId, Literal(cid)))
        g.add((rel, TL.claimIdDeclared,
               Literal(declared, datatype=XSD.boolean)))
        g.add((rel, TL.inProject, proj))
        # `baseline` is the third column of a `roc_auc_difference` claim (2026-11-20) and is emitted like x and y,
        # because without it two claims comparing one score against two different columns are the same triple set.
        for role, ref in (("x", r["x"]), ("y", r["y"]), ("baseline", r.get("baseline"))):
            if ref and ref in fields:
                g.add((rel, TL[role], fields[ref]))
        g.add((rel, TL.method, Literal(r["method"])))
        claimed = _num(r.get("claimed"))
        if claimed is not None:
            g.add((rel, TL.claimed, Literal(claimed, datatype=XSD.double)))
        # A range's open side is +-inf, which xsd:double carries as INF and -INF.
        if r.get("claimed_range") is not None:
            lo, hi = (_num(x) for x in r["claimed_range"][:2])
            if lo is not None:
                g.add((rel, TL.claimedLow, Literal(lo, datatype=XSD.double)))
            if hi is not None:
                g.add((rel, TL.claimedHigh, Literal(hi, datatype=XSD.double)))
        estimate = _num(r.get("estimate"))
        if estimate is not None:
            g.add((rel, TL.estimate, Literal(estimate, datatype=XSD.double)))
        # THE INTERVAL AND WHAT IT RESAMPLES. A verdict with no interval in the
        # graph cannot answer "which claims could not tell their effect from
        # zero", which is the question `RULES.md` calls UNRESOLVED and the one a
        # merged corpus is read for. `relate` has always written these to
        # `results/claims.json`; they simply never reached the RDF, so the
        # corpus could report a verdict and not the width behind it.
        for key, term in (("ci_lo", TL.ciLow), ("ci_hi", TL.ciHigh)):
            v = _num(r.get(key))
            if v is not None:
                g.add((rel, term, Literal(v, datatype=XSD.double)))
        if r.get("resampling_unit"):
            g.add((rel, TL.resamplingUnit, Literal(r["resampling_unit"])))
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
            e = URIRef(f"{E}evidence/{_seg(cid)}")
            g.add((e, RDF.type, EVI.Evidence))
            g.add((e, SC.identifier, Literal(ev["locator"])))
            g.add((e, RDFS.label, Literal(ev.get("source", ""))))
            g.add((e, TL.extractedBy, Literal(ev.get("extracted_by", "unknown"))))
            g.add((rel, TL.restsOn, e))
            g.add((e, EVI.directlySupports, rel))

        if r["verdict"] == "refuted":
            v = URIRef(f"{E}verdict/{_seg(cid)}")
            g.add((v, RDF.type, EVI.Method))
            g.add((v, RDFS.label, Literal(f"recomputed {r['method']} from the data")))
            g.add((v, EVI.directlyChallenges, rel))

    if ctx.get("belief"):
        _belief(g, root, by_pair, E, owner_roots)
    return g


def _belief(g: Graph, root: Path, by_pair: dict[str, URIRef], E: str,
            owner_roots: tuple[Path, ...] = ()) -> None:
    """The belief layer: nodes, claims, and the edges a reasoner walks.

    Errors are NOT swallowed. A `[belief]` table that does not load is a
    refusal a reader must see; emitting the graph without it would publish a
    description of the project with the doubt silently left out.

    **Two IRI kinds here stay global on purpose, and one is scoped.** A belief
    `node` is a concept (`heavy_atom_count`, `pocket_shape`, `freeflow`) and an
    `artefact` is a deliverable; two experiments naming the same one MEAN the
    same one, and merging them into a single node is what makes the merged
    store a corpus rather than 345 disconnected islands. The belief `claim`,
    by contrast, is numbered by position in one project's `[belief]` table and
    is scoped under `exp/`, because `claim/1` in two experiments is two
    different statements.
    """
    bg = BeliefGraph.from_project(root, owner_roots=owner_roots)

    # The document imports EVI, so a reasoner that resolves imports needs
    # nothing from `claimcheck.evi` — that file is for the one that does not.
    doc = URIRef(f"{BASE}graph")
    g.add((doc, RDF.type, OWL.Ontology))
    g.add((doc, OWL.imports, EVI_ONTOLOGY))

    nodes = {}
    for n in bg.nodes.values():
        u = URIRef(f"{BASE}node/{n.id}")
        nodes[n.id] = u
        g.add((u, RDF.type, TL.Node))
        g.add((u, TL.nodeKind, Literal(n.kind)))
        g.add((u, RDFS.label, Literal(n.label or n.id)))

    for i, c in enumerate(bg.claims, 1):
        u = URIRef(f"{E}claim/{i}")
        g.add((u, RDF.type, EVI.Claim))
        g.add((u, TL.subject, nodes[c.subject]))
        g.add((u, TL.predicate, Literal(c.predicate)))
        g.add((u, TL.object, nodes[c.object]))
        g.add((u, TL.status, Literal(c.status)))
        if c.evidence:
            g.add((u, RDFS.comment, Literal(c.evidence)))
        # `skos:note` and NOT a second `rdfs:comment`: two comments on one
        # subject are an unordered pair, and a consumer reading either as "the
        # evidence" is the exact mix-up `Claim.note` exists to prevent.
        if c.note:
            g.add((u, SKOS.note, Literal(c.note)))
        # A CLAIM SUPPORTS THE END WHOSE NUMBERS IT IS ABOUT, which for a
        # directed predicate is the dependent end and only that end.
        for b in bearers(c):
            g.add((u, EVI.supports, nodes[b]))
        # …and the recomputation that settled it supports it, so a `verdict`
        # node challenging the relationship reaches the claim, the node and the
        # artefact through one property chain.
        if c.tested_by and c.tested_by in by_pair:
            g.add((by_pair[c.tested_by], EVI.supports, u))

    # One hop of rests-on. Peers are absent on purpose — see the module
    # docstring and `BeliefGraph.supports_edges`.
    for a, b in bg.supports_edges():
        g.add((nodes[a], EVI.supports, nodes[b]))

    for name, nids in sorted(bg.rests_on.items()):
        art = URIRef(f"{BASE}artefact/{name}")
        g.add((art, RDF.type, TL.Artefact))
        g.add((art, SC.name, Literal(name)))
        # ONE EDGE PER NODE. An artefact resting on four methods is supported
        # by four, and a reasoner walking `supports` must reach all four
        # closures — dropping three would hand it a subset of `caveats_on`
        # that looks complete.
        for nid in nids:
            g.add((nodes[nid], EVI.supports, art))


def render(root: Path | str) -> str:
    """The emitter interface, shared with `croissant` and `dcat`."""
    return build(root).serialize(format="turtle")


def main() -> int:
    # `--owner-root <dir>`, repeatable: an extra directory in which to look for
    # a `defined_in` belief owner. AN ARGUMENT AND NOT AN ENVIRONMENT VARIABLE,
    # so the rule's command carries it and Snakemake's run record says where the
    # owner was resolved from (2026-11-22).
    argv, owner_roots = [], []
    it = iter(sys.argv[1:])
    for tok in it:
        if tok == "--owner-root":
            owner_roots.append(Path(next(it)))
        elif tok.startswith("--owner-root="):
            owner_roots.append(Path(tok.split("=", 1)[1]))
        else:
            argv.append(tok)
    root = Path(argv[0]) if argv else Path.cwd()
    out = Path(argv[1]) if len(argv) > 1 else root / "results/graph.ttl"
    g = build(root, owner_roots=tuple(owner_roots))
    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out), format="turtle")
    kinds = {str(o).split("#")[-1].split("/")[-1]
             for o in g.objects(None, RDF.type)}
    print(f"  {out}  —  {len(g)} triples, {len(kinds)} node kind(s): "
          f"{', '.join(sorted(kinds))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
