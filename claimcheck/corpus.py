"""Many experiments' graphs as ONE store, and the checks that a merge is real.

    python -m claimcheck.corpus <sources.tsv> <out.ttl> <report.json> [<table.csv>]

`claimcheck.graph` emits one `results/graph.ttl` per experiment. That is the
unit of work, and it is not the unit a question is asked of: "which refuted
claims has nothing re-examined", "which claims could not tell their effect from
zero", "what rests on a method we have since refuted" each cross experiments.
This module loads many graphs into one `rdflib.Graph` and **refuses a merge that
is quietly wrong**, which is the part that matters, because rdflib merges by IRI
and raises nothing when two unrelated nodes share one.

## What it checks, and why a merge needs checking at all

Measured on 2026-11-20, before `claimcheck.graph` grew its `exp/<project>/`
segment, and counted from the backup at
`/home/ben/code/shared/corpus/backup/graph_ttl_2026-11-20`: 340 of the 345
graphs under `/home/ben/code/projects` minted
`relationship/1` under the one base in `claimcheck.namespace` (written there and
nowhere else, so the two cannot drift), because a relationship was
named by its POSITION in `context.toml`, the other 5 declaring no relationship
at all. `file/edges_csv` was minted by 15 experiments carrying 13 distinct
sha256 digests and `records/truncation.n_within_kept` by 9. Merged, one node
then held the
verdicts, estimates and digests of unrelated claims, and no error was raised.

So two reports travel with every merge:

* **`shared_iris`**: every IRI minted by more than one experiment, with the
  experiments that minted it. Some sharing is the POINT: a belief `node/` or an
  `artefact/` is a concept two experiments may both name, and merging those is
  what joins the corpus. The report says which kind each shared IRI is, so a
  reader can tell a join from a collision rather than being told a number.
* **`conflated`**: any subject carrying two values for a predicate that is
  single-valued by meaning (`tl:verdict`, `tl:estimate`, `sc:sha256`). This is
  the general detector: it fires on every node kind at once, so a fix that
  scoped relationships and forgot files would still be caught.

## What it does not do

It does not adjudicate. The counts it writes go into a table that a project's
own `context.toml` registers claims against, and `claimcheck.relate` returns
the verdicts, as for any other number.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from claimcheck.namespace import BASE, NS

SC = "https://schema.org/"
TL = NS

#: Single-valued BY MEANING. A subject with two of any of these has been merged
#: with a subject it is not. Not an exhaustive list of the vocabulary: these are
#: the ones whose conflation changes what a reader would conclude.
SINGLE_VALUED = (
    f"{TL}verdict",
    f"{TL}estimate",
    f"{TL}claimed",
    f"{TL}claimId",
    f"{TL}method",
    f"{TL}resamplingUnit",
    f"{SC}sha256",
    f"{TL}origin",
    f"{TL}inDataset",
    f"{TL}unit",
    # THE TWO THAT CHECK THE DELIBERATE SHARING. `node/<id>` and
    # `artefact/<name>` are global on purpose, and that is only safe if two
    # experiments naming one id MEAN the same thing. A node arriving with two
    # `nodeKind` values, or two labels, is two concepts that happen to share a
    # name, and merging them is then exactly the defect this module exists to
    # catch, with nothing in `shared_iris` able to see it: ownership is keyed
    # by experiment, and sharing these is what the design intends.
    f"{TL}nodeKind",
    "http://www.w3.org/2000/01/rdf-schema#label",
)

#: The IRI kinds the 2026-11-20 change scoped under `exp/<project>/`, and so
#: the kinds a claim about that change is entitled to be adjudicated on.
#: `project/<name>` is NOT among them: it was already one IRI per experiment
#: and the change did not touch it, so scoring it would let the claim take
#: credit for a kind that was never broken. It stays in the table with a blank
#: indicator, which `binomial_count` drops and counts, so a collision between
#: two experiments declaring one name is still visible to a reader without
#: entering a verdict about the change.
SCOPED_KINDS = ("relationship", "evidence", "verdict", "file", "records",
                "claim")

#: IRI kinds that two experiments are MEANT to share. A belief node is a
#: concept and an artefact is a deliverable; `node/heavy_atom_count` is named by
#: 13 experiments in this lab's corpus (measured 2026-11-20), and collapsing it
#: to one node is how the corpus becomes connected rather than 346 islands.
#:
#: **It is only safe while the experiments AGREE about what the node is**, which
#: is why `nodeKind` and `rdfs:label` are in `SINGLE_VALUED`: `shared_iris`
#: cannot see a disagreement here, because sharing these IRIs is the design.
#: Four ids currently disagree, which is a real finding and not a merge bug:
#: `heavy_atom_count` is declared `method` by some of its 13 owners and
#: `quantity` by others, `ligand_size` the same across 8, `openfe_mm_ddg`
#: `quantity` against `reference`, and `truncated_pocket` `condition` against
#: `representation`.
SHARED_BY_DESIGN = (f"{BASE}node/", f"{BASE}artefact/", f"{BASE}graph")


def _kind(iri: str) -> str:
    """`exp/<project>/relationship/<id>` -> `relationship`; `node/<id>` -> `node`."""
    rest = iri[len(BASE):] if iri.startswith(BASE) else iri
    parts = rest.split("/")
    if parts[0] == "exp" and len(parts) >= 3:
        return parts[2]
    return parts[0] or "graph"


def read_sources(path: Path | str) -> list[tuple[str, Path]]:
    """`<project name>\\t<experiment directory>` per line, `#` comments skipped.

    The list is an INPUT and not something this module discovers, so that the
    rule which builds it is the one place that decides what is in the corpus and
    Snakemake can see every graph as a dependency.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, d = line.partition("\t")
        out.append((name.strip(), Path(d.strip())))
    return out


def _n_claims(d: Path) -> int:
    """How many relationships `relate` adjudicated in this directory. -1 when
    `results/claims.json` is missing or unreadable, so the row says "not known"
    rather than "none"."""
    try:
        return len(json.loads(
            (d / "results/claims.json").read_text())["relationships"])
    except Exception:                                 # noqa: BLE001 - fail soft
        return -1


def merge(sources: list[tuple[str, Path]]) -> tuple[Graph, dict]:
    """Load every `results/graph.ttl` into one store and report on the merge.

    A directory whose graph is missing or will not parse is QUARANTINED with its
    reason and the rest are merged (`shared/LAB.md`, failing soft): one
    unparseable file must not cost the corpus the other 344.
    """
    merged = Graph()
    minted: dict[str, set[str]] = defaultdict(set)
    per_project: dict[str, dict] = {}
    quarantine: list[dict] = []

    for name, d in sources:
        ttl = Path(d) / "results/graph.ttl"
        if not ttl.exists():
            quarantine.append({"project": name, "dir": str(d),
                               "reason": "no results/graph.ttl"})
            continue
        one = Graph()
        try:
            one.parse(str(ttl), format="turtle")
        except Exception as exc:                      # noqa: BLE001 - fail soft
            quarantine.append({"project": name, "dir": str(d),
                               "reason": f"will not parse: {exc}"})
            continue
        for s in set(one.subjects()):
            minted[str(s)].add(name)
        rels = set(one.subjects(RDF.type, URIRef(f"{TL}Relationship")))
        per_project[name] = {
            "dir": str(d),
            "triples": len(one),
            "relationships": len(rels),
            #: The count from `results/claims.json`, which is what `relate`
            #: adjudicated. The known answer the merge has to satisfy is that
            #: this many relationship nodes survive the merge; a collision
            #: loses rows by unioning them onto one subject, and a count is the
            #: cheapest detector of that.
            "claims_json_relationships": _n_claims(Path(d)),
            #: Kept so that `programme_table` can ask how many of this
            #: project's claims are still ITS OWN in the merged store. Counting
            #: relationships per graph cannot answer that: under the old scheme
            #: each graph held exactly the claims it adjudicated and the loss
            #: happened only when two were loaded together.
            "relationship_iris": sorted(str(r) for r in rels),
        }
        merged += one

    shared = {i: sorted(ps) for i, ps in minted.items() if len(ps) > 1}
    report = {
        "projects": len(per_project),
        "quarantined": quarantine,
        "triples": len(merged),
        "relationships": len(set(merged.subjects(
            RDF.type, URIRef(f"{TL}Relationship")))),
        "per_project": per_project,
        "shared_iris": [
            {"iri": i, "kind": _kind(i), "projects": ps,
             "shared_by_design": i.startswith(SHARED_BY_DESIGN)}
            for i, ps in sorted(shared.items())
        ],
        "conflated": conflated(merged),
        #: Every IRI with the experiments that minted it. Returned here rather
        #: than recounted by `table`, so the table and the report are one
        #: measurement and cannot disagree. `main` drops it before writing the
        #: report a person reads.
        "_minted": {i: sorted(ps) for i, ps in sorted(minted.items())},
    }
    return merged, report


def conflated(g: Graph) -> list[dict]:
    """Subjects carrying two values for a single-valued predicate.

    The general detector. A collision that this does not see is a collision
    that changed nothing a reader would read.
    """
    out = []
    for p in SINGLE_VALUED:
        pu = URIRef(p)
        for s in set(g.subjects(pu, None)):
            vals = sorted({str(o) for o in g.objects(s, pu)})
            if len(vals) > 1:
                out.append({"subject": str(s), "predicate": p, "values": vals})
    return sorted(out, key=lambda r: (r["predicate"], r["subject"]))


# ---------------------------------------------------------------------------
# The questions the corpus exists to answer
# ---------------------------------------------------------------------------

PFX = f"""\
PREFIX tl: <{TL}>
PREFIX sc: <{SC}>
PREFIX evi: <https://w3id.org/EVI#>
PREFIX cr: <http://mlcommons.org/croissant/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""


def questions() -> dict[str, tuple[str, str]]:
    """Name -> (what it asks, SPARQL). Each one CROSSES experiments, which is
    the thing that could not be asked before: every row carries the experiment
    and the claim id, so an answer can be cited and re-run."""
    return {
        "Q1_refuted_claims_by_experiment": (
            "Every refuted claim in the corpus, with its experiment and its id.",
            PFX + """SELECT ?project ?claim ?method ?estimate WHERE {
              ?rel tl:verdict "refuted" ; tl:claimId ?claim ;
                   tl:method ?method ; tl:inProject ?p .
              OPTIONAL { ?rel tl:estimate ?estimate }
              ?p sc:name ?project } ORDER BY ?project ?claim"""),
        "Q2_intervals_that_contain_zero": (
            "Claims whose 95% interval spans zero: UNRESOLVED, not absent.",
            PFX + """SELECT ?project ?claim ?lo ?hi WHERE {
              ?rel tl:claimId ?claim ; tl:ciLow ?lo ; tl:ciHigh ?hi ;
                   tl:inProject ?p .
              ?p sc:name ?project .
              FILTER (?lo <= 0.0 && ?hi >= 0.0) } ORDER BY ?project ?claim"""),
        "Q3_claims_on_one_column_across_experiments": (
            "Which experiments make a claim about a column of the same name, "
            "so a result quoted from one can be checked against the others.",
            PFX + """SELECT ?col ?project ?claim ?verdict WHERE {
              ?rel tl:claimId ?claim ; tl:verdict ?verdict ; tl:inProject ?p ;
                   tl:x ?f .
              ?f sc:name ?col . ?p sc:name ?project }
              ORDER BY ?col ?project ?claim"""),
        "Q4_what_a_refutation_puts_in_doubt": (
            "Belief nodes reached from a refuted claim, across experiments: "
            "the corpus question the per-experiment graph could not answer.",
            # `rdfs:label` AND NOT `sc:name`: `claimcheck.graph._belief` writes
            # a Node's text as `rdfs:label`. Asking for `sc:name` returned
            # nothing and looked like "no refutation reaches a belief node",
            # which is a finding, and it was a typo.
            PFX + """SELECT DISTINCT ?project ?claim ?node WHERE {
              ?v evi:directlyChallenges ?rel .
              ?rel tl:claimId ?claim ; tl:inProject ?p .
              ?p sc:name ?project .
              ?rel evi:supports+ ?n . ?n a tl:Node ; rdfs:label ?node }
              ORDER BY ?project ?claim ?node"""),
        "Q5_one_file_content_claimed_by_two_experiments": (
            "A sha256 that two experiments both distribute: real reuse, and "
            "unaskable while every experiment minted the same file IRIs.",
            # GROUPED, not listed. Without the HAVING this returned one row per
            # file in the corpus, which is a listing and answers nothing.
            PFX + """SELECT ?sha (COUNT(DISTINCT ?project) AS ?n_experiments)
                            (GROUP_CONCAT(DISTINCT ?project ; separator="; ")
                             AS ?experiments)
              WHERE {
              ?f sc:sha256 ?sha .
              ?p sc:distribution ?f ; sc:name ?project }
              GROUP BY ?sha
              HAVING (COUNT(DISTINCT ?project) > 1)
              ORDER BY DESC(?n_experiments) ?sha"""),
        "Q6_claims_with_no_claim_id_in_the_graph": (
            "Relationship nodes carrying no tl:claimId, which means a graph "
            "still emitted by the pre-2026-11-20 scheme. This is the count "
            "that says how much of the tree has been re-emitted, and it is "
            "here because Q1 and Q2 require a claimId and would otherwise "
            "drop those claims without saying so.",
            PFX + """SELECT ?rel WHERE {
              ?rel a tl:Relationship .
              FILTER NOT EXISTS { ?rel tl:claimId ?any } } ORDER BY ?rel"""),
    }


def ask(g: Graph) -> dict:
    """Run every question. A query that raises is reported with its error
    rather than stopping the others."""
    out = {}
    for name, (what, q) in questions().items():
        try:
            rows = [[str(v) if v is not None else "" for v in row]
                    for row in g.query(q)]
            out[name] = {"question": what, "n_rows": len(rows), "rows": rows}
        except Exception as exc:                      # noqa: BLE001 - fail soft
            out[name] = {"question": what, "error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# The table a context.toml registers claims against
# ---------------------------------------------------------------------------

def table(report: dict) -> list[dict]:
    """One row per IRI minted anywhere in the corpus.

    `one_experiment_only` is 1 when this IRI was minted by a single experiment,
    or when it is a kind two experiments are MEANT to share (`node/`,
    `artefact/`). Higher is better. `experiments` names the owners, so a shared
    IRI can be chased without a second query.

    This is the EVIDENCE table and no claim is registered on it: a share over
    roughly 20,000 IRIs cannot resolve one collision at any interval, which is
    why the claims sit on `kind_table` instead. The control is a second corpus
    rather than a column here; see `kind_table`.
    """
    rows = []
    for iri, ps in sorted(report["_minted"].items()):
        owners = set(ps)
        design = iri.startswith(SHARED_BY_DESIGN)
        rows.append({
            "iri": iri,
            "kind": _kind(iri),
            "n_experiments": len(owners),
            "experiments": ";".join(sorted(owners)),
            "shared_by_design": int(design),
            "one_experiment_only": int(design or len(owners) <= 1),
        })
    return rows


def kind_table(report: dict, control: dict | None = None) -> list[dict]:
    """One row per IRI KIND, and this is the table the claims are registered
    against.

    **The control is another corpus, not a reconstruction of one.** `control`
    is the report from merging a second set of graphs, in practice the backup
    of every `results/graph.ttl` as it stood before the emitter changed, and
    its per-kind collision counts land in the `*_control` columns. Stripping
    the `exp/<project>/` segment from a current IRI was tried first and is
    wrong: the old scheme named a relationship by its POSITION, so
    `exp/a/relationship/A1_one` unscopes to `relationship/A1_one`, which never
    existed and does not collide with anything, and the control silently
    reported the relationship kind as having been fine all along. It is only
    faithful for `file` and `records`, whose names the change did not touch.
    Reading the old graphs measures the old scheme instead of guessing it.

    **The unit is the kind because the share over IRIs cannot resolve a single
    collision, and saying so is better than registering a claim that cannot
    fail.** The corpus mints tens of thousands of IRIs (the exact count is
    `n_iris` in `results/corpus_report.json`). The Clopper-Pearson lower bound
    at k = n rises towards 1 as n grows, and the bound at k = n - 1 rises with
    it, so at a cohort that large the two sit closer together than any range a
    person would write: a range wide enough to pass the clean corpus passes a
    corpus with one collision in it too. `docs/binomial_count.md` shows the
    same effect at the small end, where 8 of 8 gives [0.631, 1] and
    `[0.75, 1]` can never agree at n = 8. The exact bounds at this corpus's n
    are not quoted here because nothing in this package computed them.
    `RULES.md` says the general rule plainly: cohort size decides resolution.

    There are seven kinds, and a kind is the thing the fix is actually about:
    the 2026-11-20 change scoped `relationship`, `evidence`, `verdict`, `file`,
    `records` and the belief `claim`, and deliberately left `node` and
    `artefact` global. At n = 7 the interval does separate the cases: 7 of 7
    gives [0.590, 1] and 6 of 7 gives [0.421, 0.996], so a claimed range of
    [0.5, 1.0] agrees on the first and is unverifiable on the second. **A kind
    with even one shared IRI cannot pass.** The offending count is on the row,
    so a reader gets the number and not only the verdict.
    """
    def counts(rep: dict) -> dict[str, dict]:
        minted = {i: set(ps) for i, ps in rep["_minted"].items()}
        per: dict[str, dict] = defaultdict(lambda: {"n_iris": 0, "n_shared": 0})
        for iri, owners in minted.items():
            if iri.startswith(SHARED_BY_DESIGN):
                continue                   # global on purpose, see SHARED_BY_DESIGN
            k = per[_kind(iri)]
            k["n_iris"] += 1
            k["n_shared"] += int(len(owners) > 1)
        return per

    now = counts(report)
    then = counts(control) if control else {}
    rows = []
    for kind, v in sorted(now.items()):
        scoped = kind in SCOPED_KINDS
        c = then.get(kind)
        # BLANK, NOT A SENTINEL, for a cell the claim must not score: a kind
        # the change never scoped, or a kind the control does not have at all.
        # `binomial_count` drops a blank and counts it in `dropped`, which is
        # the behaviour wanted; a -1 would REFUSE the claim (the column must be
        # an indicator), and a 0 or 1 would be a verdict about something not
        # measured. "The old corpus had none of these" and "the old corpus had
        # these and they were fine" are different statements and neither is a
        # number here.
        rows.append({
            "kind": kind,
            "is_scoped_kind": int(scoped),
            "n_iris": v["n_iris"],
            "n_shared": v["n_shared"],
            "scoped_cleanly": int(v["n_shared"] == 0) if scoped else "",
            "n_iris_control": c["n_iris"] if c else "",
            "n_shared_control": c["n_shared"] if c else "",
            "scoped_cleanly_control":
                (int(c["n_shared"] == 0) if (c and scoped) else ""),
        })
    return rows


def programme_table(report: dict, root: str = "projects") -> list[dict]:
    """One row per PROGRAMME: did every claim its experiments adjudicated
    survive the merge?

    The programme is the unit for the same reason the kind is: at 343
    experiments a single lost experiment is invisible to an interval, and at
    ten programmes it is not (10 of 10 gives [0.692, 1], 9 of 10 gives
    [0.555, 0.998], so a claimed range of [0.6, 1.0] separates them).
    """
    minted = report["_minted"]
    per: dict[str, dict] = defaultdict(
        lambda: {"n_experiments": 0, "n_claims_json": 0, "n_in_corpus": 0,
                 "n_mismatched": 0})
    for name, info in report["per_project"].items():
        d = info["dir"]
        after = d.split(f"/{root}/", 1)[-1] if f"/{root}/" in d else d
        programme = after.split("/")[0] or "unknown"
        p = per[programme]
        p["n_experiments"] += 1
        want = info["claims_json_relationships"]
        # HOW MANY OF THIS PROJECT'S CLAIMS ARE STILL ITS OWN, not how many
        # are in its own graph. Counting per graph answers nothing: under the
        # old scheme every graph held exactly the claims it adjudicated, and
        # the loss appeared only when two were loaded into one store, which is
        # the failure this is here to catch.
        got = sum(1 for iri in info["relationship_iris"]
                  if len(minted.get(iri, [name])) == 1)
        p["n_claims_json"] += max(want, 0)
        p["n_in_corpus"] += got
        p["n_mismatched"] += int(want < 0 or want != got)
    return [
        {"programme": k, **v,
         "all_claims_in_corpus": int(v["n_mismatched"] == 0)}
        for k, v in sorted(per.items())
    ]


def per_experiment_table(report: dict) -> list[dict]:
    """One row per experiment: how many relationships it adjudicated, and how
    many survived the merge.

    `counts_match` is the known answer. It is 1 when every claim `relate`
    adjudicated in that directory is a distinct node in the merged store, which
    is what a merge is supposed to mean and what a collision silently breaks:
    two claims sharing an IRI arrive as one node, so the corpus holds fewer
    claims than were tested and reports no error.
    """
    rows = []
    for name, info in sorted(report["per_project"].items()):
        want = info["claims_json_relationships"]
        got = info["relationships"]
        rows.append({
            "experiment": name,
            "dir": info["dir"],
            "n_claims_json": want,
            "n_in_corpus": got,
            "counts_match": int(want >= 0 and want == got),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    if len(sys.argv) < 8:
        print("usage: python -m claimcheck.corpus <sources.tsv> "
              "<control_sources.tsv> <out.ttl> <report.json> "
              "<iri_ownership.csv> <iri_kinds.csv> <programmes.csv> "
              "[<experiments.csv>]\n"
              "  <control_sources.tsv> may be the word 'none'.",
              file=sys.stderr)
        return 2
    sources = read_sources(sys.argv[1])
    out_ttl, out_json = Path(sys.argv[3]), Path(sys.argv[4])
    merged, report = merge(sources)
    report["queries"] = ask(merged)

    # THE CONTROL IS A SECOND CORPUS: the same experiments' graphs as they
    # stood before the emitter changed. Its store is not kept, only its
    # collision counts, which land in the `*_control` columns of the kind
    # table. See `kind_table` for why a reconstruction from the current IRIs
    # was wrong.
    control = None
    if sys.argv[2] != "none":
        _, control = merge(read_sources(sys.argv[2]))
        report["control"] = {k: v for k, v in control.items()
                             if k not in ("_minted", "per_project")}

    out_ttl.parent.mkdir(parents=True, exist_ok=True)
    merged.serialize(destination=str(out_ttl), format="turtle")

    rows = table(report)
    _write_csv(Path(sys.argv[5]), rows)
    _write_csv(Path(sys.argv[6]), kind_table(report, control))
    _write_csv(Path(sys.argv[7]), programme_table(report))
    if len(sys.argv) > 8:
        _write_csv(Path(sys.argv[8]), per_experiment_table(report))

    report["n_iris"] = len(rows)
    report["n_iris_one_experiment_only"] = sum(
        r["one_experiment_only"] for r in rows)
    del report["_minted"]
    for info in report["per_project"].values():
        info.pop("relationship_iris", None)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=False))

    print(f"  {out_ttl}: {report['triples']} triples, "
          f"{report['projects']} experiments, "
          f"{report['relationships']} relationships")
    print(f"  {report['n_iris_one_experiment_only']} of {report['n_iris']} "
          "IRIs belong to one experiment")
    if control:
        print(f"  control (the pre-2026-11-20 graphs): "
              f"{control['relationships']} relationship node(s) from "
              f"{control['projects']} experiments, "
              f"{len(control['conflated'])} conflated subject(s)")
    print(f"  {len(report['conflated'])} conflated subject(s), "
          f"{len(report['quarantined'])} quarantined directory(ies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
