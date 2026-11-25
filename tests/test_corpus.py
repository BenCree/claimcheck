"""The merge, and the detectors that say whether it was a merge or a collision.

`tests/test_graph_merge.py` tests the IRI scheme. This tests the thing that
reads many graphs: that the collision reports FIRE on a planted collision and
are silent on a clean corpus, that sharing which is the point is reported as
such, and that the claim tables carry the control column the corpus experiment's
C2 is registered against.

Every fixture is written here rather than produced by a Snakemake run, so
nothing skips: a skipped test cannot fail, which is how the defect these cover
survived.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rdflib")

from claimcheck.corpus import (  # noqa: E402
    SHARED_BY_DESIGN,
    _kind,
    ask,
    conflated,
    kind_table,
    merge,
    programme_table,
    questions,
    read_sources,
    table,
)
from claimcheck.namespace import BASE  # noqa: E402

TTL_HEAD = """\
@prefix cr: <http://mlcommons.org/croissant/> .
@prefix sc: <https://schema.org/> .
@prefix tl: <https://w3id.org/claimcheck/ns#> .
"""


def _experiment(root: Path, name: str, ttl_body: str, n_claims: int) -> Path:
    """One directory with a `results/graph.ttl` and a `results/claims.json`.

    The graph is written as Turtle rather than built, so a test can plant IRIs
    the current emitter would never mint: that is the only way to check that the
    collision detectors still work once the emitter is fixed.
    """
    d = root / name
    (d / "results").mkdir(parents=True, exist_ok=True)
    (d / "results/graph.ttl").write_text(TTL_HEAD + ttl_body)
    (d / "results/claims.json").write_text(json.dumps(
        {"project": name,
         "relationships": [{"id": f"{name}_{i}"} for i in range(n_claims)]}))
    return d


def _clean(name: str, claim: str, sha: str) -> str:
    """What the post-2026-11-20 emitter writes: everything under `exp/<name>/`."""
    e = f"{BASE}exp/{name}"
    return f"""\
<{BASE}project/{name}> a sc:Dataset ;
    sc:name "{name}" ;
    sc:distribution <{e}/file/edges_csv> .

<{e}/file/edges_csv> a cr:FileObject ;
    sc:name "edges_csv" ; sc:sha256 "{sha}" ; tl:origin "computed" .

<{e}/records/edges.ddg> a cr:Field ;
    sc:name "ddg" ; tl:inDataset "edges" ; tl:unit "kcal/mol" ;
    tl:inFile <{e}/file/edges_csv> .

<{e}/relationship/{claim}> a tl:Relationship ;
    tl:claimId "{claim}" ; tl:method "spearman" ; tl:verdict "agrees" ;
    tl:estimate 0.5 ; tl:inProject <{BASE}project/{name}> ;
    tl:x <{e}/records/edges.ddg> .
"""


def _old(name: str, claim: str, sha: str) -> str:
    """What the emitter wrote BEFORE: a position, and no experiment anywhere."""
    return f"""\
<{BASE}project/{name}> a sc:Dataset ;
    sc:name "{name}" ;
    sc:distribution <{BASE}file/edges_csv> .

<{BASE}file/edges_csv> a cr:FileObject ;
    sc:name "edges_csv" ; sc:sha256 "{sha}" ; tl:origin "computed" .

<{BASE}records/edges.ddg> a cr:Field ;
    sc:name "ddg" ; tl:inDataset "edges" ; tl:unit "kcal/mol" ;
    tl:inFile <{BASE}file/edges_csv> .

<{BASE}relationship/1> a tl:Relationship ;
    tl:claimId "{claim}" ; tl:method "spearman" ; tl:verdict "agrees" ;
    tl:estimate 0.5 ; tl:x <{BASE}records/edges.ddg> .
"""


#: SEPARATE SUBTREES PER FIXTURE. Both fixtures wrote
#: `<tmp>/projects/alpha/experiments/a_exp` when first written, so a test
#: requesting both got whichever ran second, in both variables, and the clean
#: case was quietly measured on the broken files. The programme name has to
#: stay the second path segment under `projects/`, because that is what
#: `programme_table` reads, so the fixture name goes above it.
@pytest.fixture
def clean(tmp_path):
    r = tmp_path / "clean/projects"
    a = _experiment(r / "alpha/experiments", "a_exp",
                    _clean("a_exp", "A1_one", "a" * 64), 1)
    b = _experiment(r / "beta/experiments", "b_exp",
                    _clean("b_exp", "B1_one", "b" * 64), 1)
    return [("a_exp", a), ("b_exp", b)]


@pytest.fixture
def broken(tmp_path):
    r = tmp_path / "broken/projects"
    a = _experiment(r / "alpha/experiments", "a_exp",
                    _old("a_exp", "A1_one", "a" * 64), 1)
    b = _experiment(r / "beta/experiments", "b_exp",
                    _old("b_exp", "B1_one", "b" * 64), 1)
    return [("a_exp", a), ("b_exp", b)]


# ---------------------------------------------------------------------------
# The detectors, both ways round
# ---------------------------------------------------------------------------

def test_a_clean_corpus_reports_no_collision(clean):
    _, report = merge(clean)
    assert report["projects"] == 2
    assert report["relationships"] == 2
    offenders = [s for s in report["shared_iris"] if not s["shared_by_design"]]
    assert not offenders, offenders
    assert not report["conflated"], report["conflated"]


def test_the_old_scheme_is_caught_and_named(broken):
    """THE OTHER SIDE. Without this the clean result above says nothing: a
    detector that never fires reports every corpus as sound."""
    g, report = merge(broken)
    offenders = {s["iri"]: s for s in report["shared_iris"]
                 if not s["shared_by_design"]}
    assert f"{BASE}relationship/1" in offenders, sorted(offenders)
    assert f"{BASE}file/edges_csv" in offenders, sorted(offenders)
    assert offenders[f"{BASE}relationship/1"]["projects"] == ["a_exp", "b_exp"]

    # And the general detector: one node with two digests and two claim ids.
    bad = {(c["subject"], c["predicate"]) for c in report["conflated"]}
    assert (f"{BASE}file/edges_csv", "https://schema.org/sha256") in bad, bad
    assert any(p.endswith("claimId") for _, p in bad), bad

    # The merge lost a claim, which is what a collision costs and what no
    # error ever announced: two adjudicated claims, one node.
    assert report["relationships"] == 1
    assert sum(i["claims_json_relationships"]
               for i in report["per_project"].values()) == 2


def test_conflated_is_silent_on_a_graph_that_is_merely_large(clean):
    g, _ = merge(clean)
    assert conflated(g) == []


# ---------------------------------------------------------------------------
# Sharing that is the point is not reported as a defect
# ---------------------------------------------------------------------------

def test_a_concept_two_experiments_name_is_a_join_not_a_collision(tmp_path):
    """`node/heavy_atom_count` is named by 13 experiments in this lab's corpus
    (measured 2026-11-20) and is one concept named 13 times. Scoping it would
    leave 346 islands, which is the opposite of a corpus, so it is global and
    the report labels it as shared by design rather than as a collision."""
    body = """
<https://w3id.org/claimcheck/node/heavy_atom_count> a tl:Node ;
    sc:name "heavy_atom_count" ; tl:nodeKind "descriptor" .
"""
    a = _experiment(tmp_path / "projects/alpha/experiments", "a_exp",
                    _clean("a_exp", "A1_one", "a" * 64) + body, 1)
    b = _experiment(tmp_path / "projects/beta/experiments", "b_exp",
                    _clean("b_exp", "B1_one", "b" * 64) + body, 1)
    _, report = merge([("a_exp", a), ("b_exp", b)])
    shared = {s["iri"]: s for s in report["shared_iris"]}
    node = f"{BASE}node/heavy_atom_count"
    assert node in shared, sorted(shared)
    assert shared[node]["shared_by_design"] is True
    assert not [s for s in report["shared_iris"] if not s["shared_by_design"]]
    assert node.startswith(SHARED_BY_DESIGN)
    # Agreeing about what the node IS, so nothing is flagged.
    assert not report["conflated"], report["conflated"]


def test_two_experiments_disagreeing_about_a_shared_node_is_caught(tmp_path):
    """The other side of the deliberate sharing, and the only thing that makes
    it safe: `node/heavy_atom_count` is global because two experiments naming
    it mean one concept. If they do NOT, merging them is the same defect in a
    place `shared_iris` cannot see, because sharing that IRI is the design.
    """
    def body(kind, label):
        return f"""
<{BASE}node/pocket_shape> a tl:Node ;
    rdfs:label "{label}" ; tl:nodeKind "{kind}" .
"""
    head = "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    a = _experiment(tmp_path / "projects/alpha/experiments", "a_exp",
                    head + _clean("a_exp", "A1_one", "a" * 64)
                    + body("descriptor", "pocket shape"), 1)
    b = _experiment(tmp_path / "projects/beta/experiments", "b_exp",
                    head + _clean("b_exp", "B1_one", "b" * 64)
                    + body("method", "a different thing entirely"), 1)
    _, report = merge([("a_exp", a), ("b_exp", b)])
    bad = {(c["subject"], c["predicate"].rsplit("#", 1)[-1])
           for c in report["conflated"]}
    assert (f"{BASE}node/pocket_shape", "nodeKind") in bad, bad
    assert (f"{BASE}node/pocket_shape", "label") in bad, bad


# ---------------------------------------------------------------------------
# The tables the claims are registered against
# ---------------------------------------------------------------------------

def test_the_kind_table_carries_the_control_from_the_old_graphs(clean, broken):
    """C1 reads `scoped_cleanly` and C2 reads `scoped_cleanly_control`. The
    control is the SAME experiments' graphs as they were before the emitter
    changed, so if it does not collide then C1 is crediting the change with
    repairing something that was never broken.

    Stripping the `exp/<project>/` segment off a current IRI was the first
    attempt and this test is why it was abandoned:
    `exp/a_exp/relationship/A1_one` strips to `relationship/A1_one`, which
    never existed, so the reconstruction reported the relationship kind as
    clean under the old scheme when every graph in the lab minted
    `relationship/1`.
    """
    _, report = merge(clean)
    _, control = merge(broken)
    rows = {r["kind"]: r for r in kind_table(report, control)}
    assert rows, "no kinds in the table"
    assert all(r["n_shared"] == 0 for r in rows.values()), rows
    for kind in ("relationship", "file", "records"):
        assert rows[kind]["is_scoped_kind"] == 1, rows[kind]
        assert rows[kind]["scoped_cleanly"] == 1, rows[kind]
        assert rows[kind]["scoped_cleanly_control"] == 0, rows[kind]
        assert rows[kind]["n_shared_control"] >= 1, rows[kind]
    # `project/<name>` was one per experiment under both schemes and the change
    # never scoped it, so BOTH its indicators are blank: the claim is not
    # credited with a kind that was never broken, and a reader still sees its
    # collision count on the row.
    assert rows["project"]["is_scoped_kind"] == 0, rows["project"]
    assert rows["project"]["scoped_cleanly"] == "", rows["project"]
    assert rows["project"]["scoped_cleanly_control"] == "", rows["project"]
    assert rows["project"]["n_shared_control"] == 0, rows["project"]


def test_a_kind_the_control_never_had_is_blank_not_clean(clean):
    """BLANK, not 1 and not a sentinel. "The old corpus had none of these" and
    "the old corpus had these and they were fine" are different statements, and
    a 1 here would let a kind invented after the control was taken pass C2 for
    free. A sentinel such as -1 is not the answer either: `binomial_count`
    drops a blank and counts it, while a number outside {0, 1} refuses the
    claim outright."""
    _, report = merge(clean)
    rows = {r["kind"]: r for r in kind_table(report, None)}
    assert all(r["scoped_cleanly_control"] == "" for r in rows.values()), rows
    assert all(r["n_shared_control"] == "" for r in rows.values()), rows


def test_the_kind_table_fails_on_a_corpus_that_still_collides(broken):
    _, report = merge(broken)
    rows = {r["kind"]: r for r in kind_table(report)}
    for kind in ("relationship", "file", "records"):
        assert rows[kind]["scoped_cleanly"] == 0, rows[kind]
        assert rows[kind]["n_shared"] >= 1, rows[kind]


def test_the_programme_table_detects_a_claim_lost_in_the_merge(clean, broken):
    ok = programme_table(merge(clean)[1])
    assert {r["programme"] for r in ok} == {"alpha", "beta"}, ok
    assert all(r["all_claims_in_corpus"] == 1 for r in ok), ok
    assert sum(r["n_claims_json"] for r in ok) == 2

    bad = programme_table(merge(broken)[1])
    assert sum(r["all_claims_in_corpus"] for r in bad) == 0, bad
    # TWO CLAIMS ADJUDICATED, NEITHER OF THEM STILL ITS OWN. Both minted
    # `relationship/1`, so after the merge that node belongs to no single
    # experiment and neither programme can point at its own claim. Reporting 1
    # here would be worse than reporting 0: it would say one of the two
    # survived, when what survived is a node that answers for both.
    assert sum(r["n_in_corpus"] for r in bad) == 0, bad
    assert sum(r["n_claims_json"] for r in bad) == 2, bad


def test_the_ownership_table_is_one_row_per_iri(clean):
    _, report = merge(clean)
    rows = table(report)
    assert len({r["iri"] for r in rows}) == len(rows)
    assert all(r["one_experiment_only"] == 1 for r in rows), rows
    assert {r["kind"] for r in rows} >= {"relationship", "file", "records"}
    assert all(r["experiments"] for r in rows), "an IRI with no owner named"


# ---------------------------------------------------------------------------
# Failing soft, and the small pieces
# ---------------------------------------------------------------------------

def test_an_unreadable_graph_is_quarantined_and_the_rest_merge(tmp_path):
    """`shared/LAB.md`: one bad item never exits a loop over many. One
    unparseable graph must not cost the corpus the other 344."""
    a = _experiment(tmp_path / "projects/alpha/experiments", "a_exp",
                    _clean("a_exp", "A1_one", "a" * 64), 1)
    b = _experiment(tmp_path / "projects/beta/experiments", "b_exp",
                    "this is not turtle <<<", 1)
    c = tmp_path / "projects/gamma/experiments/c_exp"
    c.mkdir(parents=True)
    _, report = merge([("a_exp", a), ("b_exp", b), ("c_exp", c)])
    assert report["projects"] == 1
    assert report["relationships"] == 1
    reasons = {q["project"]: q["reason"] for q in report["quarantined"]}
    assert set(reasons) == {"b_exp", "c_exp"}, reasons
    assert "will not parse" in reasons["b_exp"]
    assert "no results/graph.ttl" in reasons["c_exp"]


def test_kind_reads_the_kind_from_either_scheme():
    assert _kind(f"{BASE}exp/p/relationship/A1") == "relationship"
    assert _kind(f"{BASE}relationship/1") == "relationship"
    assert _kind(f"{BASE}node/x") == "node"
    assert _kind(f"{BASE}exp/p/records/ds.col") == "records"


def test_every_question_runs_and_none_of_them_errors(clean):
    """THE GAP THAT LET Q4 SHIP BROKEN. `ask` reports a failed query with its
    error rather than stopping the others, which is right for a sweep and means
    a query with a typo returns "no rows" to a reader. Q4 asked a `tl:Node` for
    `sc:name` when `claimcheck.graph` writes `rdfs:label`, so it returned
    nothing and read as "no refutation reaches a belief node", which is a
    finding. Nothing tested `ask` at all.
    """
    g, _ = merge(clean)
    answers = ask(g)
    assert set(answers) == set(questions()), sorted(answers)
    errors = {k: v["error"] for k, v in answers.items() if "error" in v}
    assert not errors, errors
    for name, a in answers.items():
        assert a["question"].strip(), name
        assert a["n_rows"] == len(a["rows"]), name


def test_the_questions_that_must_find_something_do(clean):
    """A query that cannot return a row is not a question. These two are
    answerable off the minimal fixture, so a change that breaks their shape
    fails here rather than reading as an empty corpus."""
    g, _ = merge(clean)
    answers = ask(g)
    # Q3 joins a claim to a named column; both fixtures have one.
    assert answers["Q3_claims_on_one_column_across_experiments"]["n_rows"] == 2
    # Q6 counts relationship nodes with no claim id: none here, by construction.
    assert answers["Q6_claims_with_no_claim_id_in_the_graph"]["n_rows"] == 0


def test_q6_counts_the_old_scheme_graphs(broken):
    """Q6 is what says how much of the tree has been re-emitted, and it exists
    because Q1 and Q2 require a claimId and would otherwise drop those claims
    without saying so."""
    g, _ = merge(broken)
    answers = ask(g)
    assert answers["Q6_claims_with_no_claim_id_in_the_graph"]["n_rows"] == 0, \
        "the planted old-scheme fixture carries a claimId literal already"
    # Q1 needs `inProject`, which the old emitter never wrote, so it is blind
    # to these graphs. That is the gap Q6 measures.
    assert answers["Q1_refuted_claims_by_experiment"]["n_rows"] == 0


def test_sources_are_read_with_comments_skipped(tmp_path):
    p = tmp_path / "sources.tsv"
    p.write_text("# a comment\n\nalpha\t/tmp/alpha\nbeta\t/tmp/beta\n")
    assert read_sources(p) == [("alpha", Path("/tmp/alpha")),
                               ("beta", Path("/tmp/beta"))]
