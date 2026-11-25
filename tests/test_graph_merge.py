"""Two experiments' graphs in ONE store must keep their claims apart.

**This is the test that could not have existed before, and its absence is how
the defect survived.** `tests/test_graph.py` builds one graph and asks four
questions of it, and every one of those questions is answered correctly by a
graph whose IRIs collide with every other graph in the lab, because nothing in
it ever loads a second graph. Measured on 2026-11-20 across the 345 graphs
emitted under `/home/ben/code/projects`, counted from the backup at
`/home/ben/code/shared/corpus/backup/graph_ttl_2026-11-20`: 340 of them minted
`<https://w3id.org/claimcheck/relationship/1>`, `file/edges_csv` was minted by
15 experiments with 13 distinct digests, and `records/truncation.n_within_kept`
by 9. rdflib merges by IRI and raises nothing, so a merged corpus silently gave
one node the verdicts and digests of unrelated claims.

Each test is run against a PLANTED PAIR: two experiments built to collide as
hard as the format allows, sharing claim positions, file names, column names and
dataset ids. The fixtures are written here rather than produced by a Snakemake
run, because `tests/test_graph.py` skips when snakemake is absent and a skipped
test cannot fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, URIRef  # noqa: E402
from rdflib.namespace import RDF  # noqa: E402

from claimcheck.graph import build  # noqa: E402
from claimcheck.namespace import BASE, NS  # noqa: E402

TL = NS
SC = "https://schema.org/"

#: Single-valued by meaning: a subject carrying two of any of these has been
#: merged with a subject it is not.
SINGLE_VALUED = (
    f"{TL}verdict",
    f"{TL}estimate",
    f"{TL}claimed",
    f"{TL}method",
    f"{TL}claimId",
    f"{SC}sha256",
    f"{TL}inDataset",
)


def _write(root: Path, project: str, claims: list[dict], sha: str) -> Path:
    """One minimal adjudicated experiment on disk.

    Deliberately IDENTICAL between the two experiments in every field the old
    scheme used to build an IRI: the same dataset id, the same column names,
    the same file name, the same number of claims in the same order. The only
    things that differ are the project name, the claim ids and the numbers,
    which is exactly the case the old scheme could not represent.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    (root / "context.toml").write_text(
        f'[project]\nname = "{project}"\ntitle = "t"\nlicense = "CC0-1.0"\n')
    (root / "results/claims.json").write_text(json.dumps(
        {"project": project, "relationships": claims}))
    (root / "results/croissant.json").write_text(json.dumps({
        "distribution": [{"name": "edges_csv", "sha256": sha,
                          f"{TL}origin": "computed"}],
        "recordSet": [{
            f"{TL}dataset": "edges",
            "field": [
                {"name": "ddg_on", "description": "d",
                 f"{TL}unit": "kcal/mol",
                 "source": {"fileObject": {"@id": "edges_csv"}}},
                {"name": "ddg_off", "description": "d",
                 f"{TL}unit": "kcal/mol",
                 "source": {"fileObject": {"@id": "edges_csv"}}},
            ]}],
    }))
    return root


def _claim(cid: str, estimate: float, verdict: str, claimed: float,
           lo: float, hi: float) -> dict:
    return {"id": cid, "x": "edges.ddg_on", "y": "edges.ddg_off",
            "kind": "correlation", "method": "spearman",
            "claimed": claimed, "estimate": estimate,
            "ci_lo": lo, "ci_hi": hi, "resampling_unit": "edge",
            "verdict": verdict, "status": "tested",
            "why": f"{cid}: recomputed",
            "evidence": {"source": "s", "locator": f"{cid} :: row",
                         "extracted_by": "human"}}


@pytest.fixture(scope="module")
def planted(tmp_path_factory) -> tuple[Graph, Graph]:
    """Two experiments whose claims sit at the SAME POSITIONS, 1 and 2.

    Under the old scheme both owned `relationship/1` and `relationship/2`,
    `file/edges_csv` and `records/edges.ddg_on`.
    """
    d = tmp_path_factory.mktemp("planted")
    a = _write(d / "a", "alpha_experiment",
               [_claim("A1_a_known_answer", 1.0, "agrees", 1.0, 0.99, 1.0),
                _claim("A2_gcmc_reorders", 0.42, "refuted", 0.9, 0.31, 0.53)],
               "a" * 64)
    b = _write(d / "b", "beta_experiment",
               [_claim("B1_a_known_answer", -1.0, "refuted", 1.0, -1.0, -0.99),
                _claim("B2_water_helps", 0.01, "unverifiable", 0.5, -0.24, 0.38)],
               "b" * 64)
    return build(a), build(b)


@pytest.fixture(scope="module")
def merged(planted) -> Graph:
    """The store the corpus is. Serialised and reparsed, because that is how a
    consumer will load it: a merge of in-memory objects could pass while the
    Turtle on disk does not round-trip."""
    ga, gb = planted
    m = Graph()
    for g in (ga, gb):
        m.parse(data=g.serialize(format="turtle"), format="turtle")
    return m


# ---------------------------------------------------------------------------
# The two sides. Both must hold or the merge is not a merge.
# ---------------------------------------------------------------------------

def test_the_two_graphs_share_no_relationship_iri(planted):
    """SIDE ONE: distinct claims stay distinct.

    This is the assertion that fails on the pre-2026-11-20 emitter: it returned
    `{relationship/1, relationship/2}` for both experiments.
    """
    ga, gb = planted
    sa = {str(s) for s in ga.subjects(RDF.type, URIRef(f"{TL}Relationship"))}
    sb = {str(s) for s in gb.subjects(RDF.type, URIRef(f"{TL}Relationship"))}
    assert len(sa) == 2 and len(sb) == 2, (sa, sb)
    assert not (sa & sb), f"same IRI in two experiments: {sorted(sa & sb)}"


def test_the_merged_store_holds_every_claim_and_loses_none(merged):
    """Four claims in, four claims out. A collision loses rows by silently
    unioning them onto one subject, and the count is the cheapest detector."""
    rels = set(merged.subjects(RDF.type, URIRef(f"{TL}Relationship")))
    assert len(rels) == 4, sorted(str(r) for r in rels)
    ids = {str(o) for o in merged.objects(None, URIRef(f"{TL}claimId"))}
    assert ids == {"A1_a_known_answer", "A2_gcmc_reorders",
                   "B1_a_known_answer", "B2_water_helps"}, ids


def test_no_subject_in_the_merged_store_is_two_things_at_once(merged):
    """SIDE TWO, and the general form: a single-valued predicate with two
    values on one subject is a merge that happened and should not have.

    This catches every node kind at once, which is the point: fixing
    relationships alone would leave `file/edges_csv` carrying two digests and
    the claim to column to file to digest traversal returning both.
    """
    bad = {}
    for p in SINGLE_VALUED:
        for s in set(merged.subjects(URIRef(p), None)):
            vals = {str(o) for o in merged.objects(s, URIRef(p))}
            if len(vals) > 1:
                bad[f"{s} {p}"] = sorted(vals)
    assert not bad, f"conflated subjects: {bad}"


def test_each_file_and_column_keeps_its_own_digest(merged):
    """The traversal `tests/test_graph.py` exists for, in a merged store.

    Both experiments declare a file called `edges_csv` with a DIFFERENT
    sha256. Two digests on one node is not a detail: it is the graph claiming
    the claim is about bytes it is not about.
    """
    rows = list(merged.query(f"""
        PREFIX tl: <{TL}> PREFIX sc: <{SC}>
        SELECT ?cid ?sha WHERE {{
          ?rel tl:claimId ?cid ; tl:x ?x .
          ?x tl:inFile ?f . ?f sc:sha256 ?sha }}"""))
    by_claim = {}
    for cid, sha in rows:
        by_claim.setdefault(str(cid), set()).add(str(sha))
    assert len(by_claim) == 4, by_claim
    for cid, shas in by_claim.items():
        assert len(shas) == 1, f"{cid} reaches {len(shas)} digests: {shas}"
    assert {next(iter(s)) for s in by_claim.values()} == {"a" * 64, "b" * 64}


# ---------------------------------------------------------------------------
# The IRI is derivable and stable
# ---------------------------------------------------------------------------

def test_a_claim_can_be_found_by_the_id_people_use(merged):
    """Proposal P26's form of the defect: `A2_esr1_rigid_cnnaff_auc` was not
    askable, because the graph named claims by position and carried no id."""
    rows = list(merged.query(f"""
        PREFIX tl: <{TL}> PREFIX sc: <{SC}>
        SELECT ?project ?verdict ?estimate WHERE {{
          ?rel tl:claimId "A2_gcmc_reorders" ;
               tl:inProject ?p ; tl:verdict ?verdict ; tl:estimate ?estimate .
          ?p sc:name ?project }}"""))
    assert len(rows) == 1, rows
    project, verdict, estimate = rows[0]
    assert str(project) == "alpha_experiment"
    assert str(verdict) == "refuted"
    assert float(estimate) == pytest.approx(0.42)


def test_the_iri_is_derivable_from_the_experiment_and_the_claim_id(planted):
    """A cited IRI must be reconstructable without reading the emitter."""
    ga, _ = planted
    want = URIRef(f"{BASE}exp/alpha_experiment/relationship/A2_gcmc_reorders")
    assert (want, RDF.type, URIRef(f"{TL}Relationship")) in ga


def test_inserting_a_claim_does_not_rewrite_the_others(tmp_path):
    """Position-named IRIs mean that inserting a claim at the top silently
    makes every citation of a later one point at a different claim. 33 claims
    reached one graph on 2026-11-18 as `relationship/35` to `relationship/67`,
    and every one of those names would have moved.
    """
    old = [_claim("C1_first", 0.1, "agrees", 0.1, 0.0, 0.2),
           _claim("C2_second", 0.2, "agrees", 0.2, 0.1, 0.3)]
    new = [_claim("C0_inserted", 0.9, "agrees", 0.9, 0.8, 1.0), *old]
    before = build(_write(tmp_path / "before", "gamma", old, "c" * 64))
    after = build(_write(tmp_path / "after", "gamma", new, "c" * 64))

    def iris(g):
        return {str(s) for s in g.subjects(RDF.type, URIRef(f"{TL}Relationship"))}

    assert iris(before) < iris(after), (iris(before), iris(after))
    assert len(iris(after)) == 3
    # The sharp form, and the one a position cannot satisfy: the IRI of a claim
    # that did not move is the SAME STRING, so a citation of it still points at
    # it. Under positional naming `C1_first` was `relationship/1` before and
    # `relationship/2` after, while `relationship/1` became a different claim.
    for cid in ("C1_first", "C2_second"):
        want = URIRef(f"{BASE}exp/gamma/relationship/{cid}")
        assert (want, RDF.type, URIRef(f"{TL}Relationship")) in before
        assert (want, RDF.type, URIRef(f"{TL}Relationship")) in after


def test_an_undeclared_id_is_derived_from_content_not_position(tmp_path):
    """25 of the 345 adjudicated directories predate the `id` key. They still
    have to merge, and a positional fallback would reintroduce the defect for
    them, so the fallback hashes the declaration. Two relationships identical
    in every hashed field are suffixed rather than merged, which
    `md-ensembles/experiments/t_equil` needs: it registers the same x, y, kind
    and method at `claimed = 1.0` and at `claimed = 0.0`.
    """
    def anon(claimed, est):
        c = _claim("", est, "agrees", claimed, est - 0.1, est + 0.1)
        del c["id"]
        return c

    pair = [anon(1.0, 0.95), anon(0.0, 0.05)]
    g = build(_write(tmp_path / "d", "delta", pair, "d" * 64))
    iris = {str(s) for s in g.subjects(RDF.type, URIRef(f"{TL}Relationship"))}
    assert len(iris) == 2, iris
    assert not any(s.rsplit("/", 1)[-1].isdigit() for s in iris), iris
    assert all("unnamed-" in s for s in iris), iris
    declared = {str(o) for o in g.objects(None, URIRef(f"{TL}claimIdDeclared"))}
    assert declared == {"false"}, declared

    # Reordered, the SAME two IRIs come back: the name follows the declaration
    # and not its place in the list.
    g2 = build(_write(tmp_path / "d2", "delta", pair[::-1], "d" * 64))
    assert iris == {str(s) for s in
                    g2.subjects(RDF.type, URIRef(f"{TL}Relationship"))}


def test_identical_declarations_do_not_collapse_onto_one_node(tmp_path):
    """The one case the content hash alone cannot separate, so it is suffixed
    and counted rather than silently merged into one claim."""
    def anon():
        c = _claim("", 0.5, "agrees", 0.5, 0.4, 0.6)
        del c["id"]
        return c

    g = build(_write(tmp_path / "e", "epsilon", [anon(), anon()], "e" * 64))
    iris = {str(s) for s in g.subjects(RDF.type, URIRef(f"{TL}Relationship"))}
    assert len(iris) == 2, iris


# ---------------------------------------------------------------------------
# What the merge is FOR
# ---------------------------------------------------------------------------

def test_a_non_numeric_claimed_value_is_left_out_not_emitted_as_a_double(tmp_path):
    """An xsd:double literal that is not a double is worse than a missing one.

    One directory in the tree carries `tl:claimed "None"^^xsd:double`, from a
    `claims.json` holding the STRING "None": rdflib warns on every load, and a
    SPARQL filter over that literal is undefined rather than false, so a query
    for claims above a threshold omits it instead of reporting it.
    """
    bad = _claim("Z1_claimed_is_a_string", 0.5, "agrees", 0.5, 0.4, 0.6)
    bad["claimed"] = "None"
    bad["ci_lo"] = "None"
    g = build(_write(tmp_path / "z", "zeta", [bad], "f" * 64))
    ttl = g.serialize(format="turtle")
    assert '"None"' not in ttl, ttl
    assert (None, URIRef(f"{TL}claimed"), None) not in g
    assert (None, URIRef(f"{TL}ciLow"), None) not in g
    # The claim itself is still there, and so is the bound that IS a number.
    assert list(g.objects(None, URIRef(f"{TL}claimId"))), "the claim vanished"
    assert list(g.objects(None, URIRef(f"{TL}ciHigh")))
    # And it reparses, which an ill-typed literal makes rdflib complain about.
    assert len(Graph().parse(data=ttl, format="turtle")) == len(g)


def test_the_interval_travels_so_unresolved_is_askable(merged):
    """`RULES.md`: an interval containing zero is UNRESOLVED, not absent. The
    graph carried no bounds until 2026-11-20, so the corpus could report a
    verdict and not the width behind it."""
    rows = list(merged.query(f"""
        PREFIX tl: <{TL}>
        SELECT ?cid ?lo ?hi WHERE {{
          ?rel tl:claimId ?cid ; tl:ciLow ?lo ; tl:ciHigh ?hi .
          FILTER (?lo <= 0.0 && ?hi >= 0.0) }}"""))
    assert [str(r[0]) for r in rows] == ["B2_water_helps"], rows


def test_a_cross_experiment_query_names_the_experiment(merged):
    """The deliverable: every refuted claim in the store with its experiment
    and its id. Under the old scheme this query returned one row per position
    with the experiments conflated."""
    rows = list(merged.query(f"""
        PREFIX tl: <{TL}> PREFIX sc: <{SC}>
        SELECT ?project ?cid WHERE {{
          ?rel tl:verdict "refuted" ; tl:claimId ?cid ; tl:inProject ?p .
          ?p sc:name ?project }} ORDER BY ?project ?cid"""))
    assert [(str(a), str(b)) for a, b in rows] == [
        ("alpha_experiment", "A2_gcmc_reorders"),
        ("beta_experiment", "B1_a_known_answer"),
    ], rows
