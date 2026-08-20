"""The belief model: does a refutation reach what it should, and stop where it should?

A propagation test that only ever watches doubt arrive has tested one half of
the mechanism. Everything here that shows a caveat being delivered has a
partner showing one refused, and the partner is usually the more interesting
of the two — `supersedes` and `dominates_error_of` exist in the vocabulary to
say *this carries no doubt*, and a traversal that ignored them would look
correct on every positive example.

**The vocabulary is covered by construction.** The three parametrised tests
below take their arguments from `vocabulary.PREDICATES`, `vocabulary.STATUSES`
and `vocabulary.NODE_KINDS` themselves, so a term added without a decision
about how it behaves fails immediately, and a term nothing exercises cannot
exist. That is the mechanical form of "a vocabulary term nothing tests is a
term nobody can rely on"; a hand-written list of the thirteen would drift the
first time somebody added a fourteenth.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from claimcheck import croissant, relate, vocabulary as v
from claimcheck.belief import BeliefError, BeliefGraph, Claim, bearers, check_claim

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"

#: Evidence for statuses that require it, so a fixture exercising a status is
#: not accidentally testing the evidence rule instead.
_EV = "stated by the fixture"


def claim(subject: str, predicate: str, obj: str, status: str = "assumed",
          **extra) -> dict:
    d: dict = {"subject": subject, "predicate": predicate, "object": obj,
               "status": status}
    if v.STATUSES[status]["evidence_required"]:
        d["evidence"] = _EV
    return d | extra


def graph(claims: list[dict], nodes=("a", "b", "c", "d"), rests_on=None,
          recomputed=None, kind="method") -> BeliefGraph:
    return BeliefGraph.from_dict(
        {"nodes": [{"id": n, "kind": kind} for n in nodes],
         "claims": claims,
         "rests_on": dict(rests_on or {})},
        recomputed=recomputed)


# ---------------------------------------------------------------------------
# The whole vocabulary, one parametrised test per column
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("predicate", sorted(v.PREDICATES))
def test_every_predicate_moves_staleness_exactly_where_its_row_says(predicate):
    """Column one. Five of the thirteen move none, and `stale=None` is an
    ANSWER — this is the test that makes it one rather than an omission."""
    g = graph([claim("a", predicate, "b", "falsified")], nodes=("a", "b"))
    direction = v.PREDICATES[predicate]["stale"]
    if direction == "backward":
        assert g.stale_table() == [("b", "a")]
    elif direction == "forward":
        assert g.stale_table() == [("a", "b")]
    else:
        assert g.stale_table() == [], (
            f"{predicate} declares stale=None and made something stale anyway")


@pytest.mark.parametrize("predicate", sorted(v.PREDICATES))
def test_every_predicate_delivers_its_caveat_to_the_end_its_row_names(predicate):
    """Column two, at one hop: WHOSE numbers does this warning belong beside?

    The directed predicates deliver to one end only. That is the fix for
    handing `[falsified] surrogate --approximates--> reference` to the
    reference — 18 of 50 deliveries on the graph this was measured against.
    """
    g = graph([claim("a", predicate, "b", "falsified")], nodes=("a", "b"),
              rests_on={"A": "a", "B": "b"})
    got = {name for name in ("A", "B") if g.caveats_on(name)}
    doubt = v.PREDICATES[predicate]["doubt"]
    expected = {v.TO_SUBJECT: {"A"}, v.TO_OBJECT: {"B"}}.get(doubt, {"A", "B"})
    assert got == expected, f"{predicate} ({doubt}) delivered to {got}"


@pytest.mark.parametrize("predicate", sorted(v.PREDICATES))
def test_doubt_crosses_exactly_the_predicates_that_declare_dependence(predicate):
    """Column two, at two hops: does doubt travel THROUGH this edge?

    `x` is put at the dependent end whichever way the predicate is written, so
    the only variable left is whether the edge conducts. Four predicates say it
    does not, and this is the half of the test that watches nothing arrive.
    """
    doubt = v.PREDICATES[predicate]["doubt"]
    edge = (claim("y", predicate, "x", "by_construction") if doubt == v.TO_OBJECT
            else claim("x", predicate, "y", "by_construction"))
    far = claim("y", "approximates", "z", "falsified")
    g = graph([edge, far], nodes=("x", "y", "z"), rests_on={"X": "x"})
    delivered = [d["claim"] for d in g.caveats_on("X")]
    if doubt == v.NOWHERE:
        assert delivered == [], (
            f"{predicate} declares doubt={doubt} and carried it anyway")
    else:
        assert delivered == ["y --approximates--> z"], predicate


@pytest.mark.parametrize("status", sorted(v.STATUSES))
def test_every_status_answers_both_questions_exactly_as_its_row_says(status):
    """The two questions, over all seven statuses. Three answer no to both,
    three answer yes to both, and one — `falsified` — differs, which is the
    entire reason there are two methods rather than one boolean."""
    g = graph([claim("a", "approximates", "b", status)], nodes=("a", "b"),
              rests_on={"A": "a"})
    row = v.STATUSES[status]
    assert bool(g.open_questions()) is row["open_question"], status
    assert bool(g.caveats_on("A")) is row["carries_caveat"], status


@pytest.mark.parametrize("status", sorted(v.STATUSES))
def test_evidence_is_demanded_exactly_where_the_row_says(status):
    """Two-sided per status: the same claim with and without evidence."""
    bare = {"subject": "a", "predicate": "approximates", "object": "b",
            "status": status}
    if v.STATUSES[status]["evidence_required"]:
        with pytest.raises(BeliefError, match="requires evidence"):
            graph([bare], nodes=("a", "b"))
        graph([bare | {"evidence": _EV}], nodes=("a", "b"))     # and this loads
    else:
        graph([bare], nodes=("a", "b"))                          # loads bare


@pytest.mark.parametrize("kind", v.NODE_KINDS)
def test_every_declared_node_kind_loads(kind):
    g = graph([], nodes=("a",), kind=kind)
    assert g.nodes["a"].kind == kind


def test_a_node_kind_outside_the_vocabulary_is_refused():
    with pytest.raises(BeliefError, match="unknown kind"):
        graph([], nodes=("a",), kind="widget")


def test_the_two_columns_disagree_on_exactly_one_predicate():
    """DERIVED in `vocabulary.DISAGREE`, so the docstring's "twelve of
    thirteen" cannot go stale. If this number ever moves, the doubt column has
    stopped being a copy of the staleness column for a new reason and somebody
    should read it."""
    assert v.DISAGREE == ("shares_approximation_with",), v.DISAGREE


def test_the_derived_status_lists_are_derived():
    """`OPEN_QUESTION` and `CARRIES_CAVEAT` differ by exactly `falsified`. A
    second hand-written list is a second thing to keep in step."""
    assert set(v.CARRIES_CAVEAT) - set(v.OPEN_QUESTION) == {"falsified"}
    assert set(v.OPEN_QUESTION) - set(v.CARRIES_CAVEAT) == set()


# ---------------------------------------------------------------------------
# The traversal, where the two relations part company
# ---------------------------------------------------------------------------

def test_a_refutation_reaches_the_surrogate_and_not_the_reference():
    """The bearer rule, in one test with both sides. Refuting `surrogate
    approximates reference` is bad news about the surrogate and, if anything, a
    vindication of the reference."""
    g = graph([claim("surrogate", "approximates", "reference", "falsified")],
              nodes=("surrogate", "reference"),
              rests_on={"S": "surrogate", "R": "reference"})
    assert len(g.caveats_on("S")) == 1
    assert g.caveats_on("R") == [], "the yardstick is not implicated"


def test_the_edge_that_carries_doubt_and_no_staleness():
    """`shares_approximation_with`: epistemic dependence, zero staleness
    dependence. Answering the caveat question with the staleness traversal made
    this the one edge doubt could not cross, with every check green."""
    g = graph([claim("a", "shares_approximation_with", "b", "by_construction"),
               claim("b", "approximates", "c", "falsified")],
              nodes=("a", "b", "c"), rests_on={"A": "a"})
    assert g.stale_if("b") == [], "nothing is recomputed when a peer changes"
    assert [d["claim"] for d in g.caveats_on("A")] == ["b --approximates--> c"]


def test_a_shared_assumption_is_symmetric_and_not_transitive():
    """`a` and `b` share one assumption; `b` and `c` share another. That gives
    `a` and `c` nothing in common, and closing over the relation would assert
    that it did — measured elsewhere as turning one method's deliveries into
    five, three of them about methods it shares nothing with."""
    claims = [claim("a", "shares_approximation_with", "b", "by_construction"),
              claim("b", "shares_approximation_with", "c", "by_construction"),
              claim("c", "approximates", "d", "falsified")]
    g = graph(claims, rests_on={"A": "a", "B": "b"})
    assert [d["claim"] for d in g.caveats_on("B")] == ["c --approximates--> d"]
    assert g.caveats_on("A") == [], "a and c share no assumption"


def test_a_cycle_does_not_make_a_node_stale_because_of_itself():
    g = graph([claim("a", "depends_on", "b", "by_construction"),
               claim("b", "depends_on", "a", "by_construction")],
              nodes=("a", "b"))
    assert g.stale_if("a") == ["b"]
    assert g.stale_if("b") == ["a"]


def test_a_self_loop_terminates_and_reaches_nothing():
    g = graph([claim("a", "depends_on", "a", "by_construction")], nodes=("a",))
    assert g.stale_if("a") == []
    assert g.in_doubt("a") == []


def test_an_artefact_nobody_declared_is_not_a_clean_bill_of_health():
    """`caveats_on` answers `[]` both for "not joined" and for "joined, nothing
    to say". A caller that cannot tell them apart prints reassurance about a
    name that does not exist."""
    g = graph([claim("a", "approximates", "b", "falsified")], nodes=("a", "b"),
              rests_on={"A": "a"})
    assert g.knows("A") and not g.knows("typo")
    assert g.caveats_on("typo") == []


def test_caveats_that_can_reach_nobody_are_counted():
    """A graph with a falsified claim and nothing declared to rest on anything
    reported `0 caveat(s)` on the same line as the word `falsified`."""
    g = graph([claim("a", "approximates", "b", "falsified")], nodes=("a", "b"))
    s = g.summary()
    assert s["caveat_deliveries"] == 0
    assert s["caveats_undeliverable"] == 1


# ---------------------------------------------------------------------------
# What the model refuses to load. Each raises on the bad and loads on the good.
# ---------------------------------------------------------------------------

def test_a_predicate_outside_the_vocabulary_is_refused():
    with pytest.raises(BeliefError, match="unknown predicate 'is_better_than'"):
        graph([claim("a", "is_better_than", "b", "assumed")], nodes=("a", "b"))
    graph([claim("a", "approximates", "b", "assumed")], nodes=("a", "b"))


def test_a_status_outside_the_vocabulary_is_refused():
    with pytest.raises(BeliefError, match="unknown status 'probably'"):
        graph([{"subject": "a", "predicate": "approximates", "object": "b",
                "status": "probably"}], nodes=("a", "b"))


def test_a_claim_about_an_undeclared_node_is_refused():
    """Referential integrity, which no JSON Schema can express: a typo creates
    a phantom node that every closure then quietly routes around."""
    with pytest.raises(BeliefError, match="object 'docked_pse'"):
        graph([claim("a", "depends_on", "docked_pse", "by_construction")],
              nodes=("a", "docked_pose"))
    graph([claim("a", "depends_on", "docked_pose", "by_construction")],
          nodes=("a", "docked_pose"))


def test_untested_may_not_carry_evidence():
    """A claim somebody settled and forgot to restate is a stale export, one
    field earlier."""
    with pytest.raises(BeliefError, match="'untested' and carries evidence"):
        graph([{"subject": "a", "predicate": "approximates", "object": "b",
                "status": "untested", "evidence": "we did check, actually"}],
              nodes=("a", "b"))


def test_the_same_claim_twice_is_refused():
    """It is not asserted twice as hard, and it double-counts every denominator
    drawn from `by_status`."""
    c = claim("a", "approximates", "b", "measured")
    with pytest.raises(BeliefError, match="is declared twice"):
        graph([c, dict(c)], nodes=("a", "b"))


def test_one_triple_under_two_statuses_is_legal_and_reported():
    """Deliberately NOT an error: a claim measured once and later falsified is
    a history worth publishing. It is counted so it can be looked at."""
    g = graph([claim("a", "approximates", "b", "measured"),
               claim("a", "approximates", "b", "falsified")], nodes=("a", "b"))
    assert g.contradictions() == [("a", "approximates", "b",
                                   ["falsified", "measured"])]
    assert g.summary()["contradictory_triples"] == 1


def test_resting_on_a_node_that_does_not_exist_is_refused():
    """It would not break the build. It would silently stop doubt reaching that
    artefact — the failure where a file looks better supported than it is."""
    with pytest.raises(BeliefError, match="rests_on\\['A'\\] names 'ghost'"):
        graph([claim("a", "approximates", "b", "falsified")], nodes=("a", "b"),
              rests_on={"A": "ghost"})


def test_a_node_declared_twice_is_refused():
    with pytest.raises(BeliefError, match="declared twice"):
        BeliefGraph.from_dict({"nodes": [{"id": "a", "kind": "method"},
                                         {"id": "a", "kind": "reference"}]})


def test_a_predicate_the_doubt_column_does_not_classify_is_refused(monkeypatch):
    """The guard that keeps the vocabulary honest as it grows. It fires on the
    first CLAIM that uses the predicate, naming the file to edit, rather than
    making the package unimportable for everybody."""
    monkeypatch.setitem(v.PREDICATES, "invented",
                        {"stale": None, "doubt": "sideways", "meaning": "x"})
    with pytest.raises(BeliefError, match="is not one of"):
        check_claim(Claim("a", "invented", "b", "assumed"), {"a", "b"})


def test_bearers_covers_every_doubt_kind():
    """`bearers` is a four-way branch and the two-ended cases fall through to
    the same return. If a fifth kind is ever added, this is where it lands."""
    seen = {v.PREDICATES[c.predicate]["doubt"]: bearers(c)
            for c in [Claim("a", p, "b", "assumed") for p in v.PREDICATES]}
    assert set(seen) == set(v.DOUBT_KINDS)
    assert seen[v.TO_SUBJECT] == ("a",)
    assert seen[v.TO_OBJECT] == ("b",)
    assert seen[v.BOTH_WAYS] == ("a", "b")
    assert seen[v.NOWHERE] == ("a", "b")


# ---------------------------------------------------------------------------
# The join to `relate`: a recomputed verdict becomes a belief
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """The shipped example, with `results/claims.json` RECOMPUTED here.

    Not the committed copy: a test that reads results somebody left in the
    working tree passes on a fresh clone by examining nothing.
    """
    work = tmp_path_factory.mktemp("belief") / "example"
    shutil.copytree(EXAMPLE, work)
    shutil.rmtree(work / "results", ignore_errors=True)
    (work / "results").mkdir(parents=True)
    (work / "results" / "claims.json").write_text(
        json.dumps(relate.check(work), indent=2) + "\n")
    # …and the Croissant, because the RDF tests below join the belief layer to
    # the columns it is about, and that join comes through the Croissant.
    (work / "results" / "croissant.json").write_text(
        json.dumps(croissant.build(work), indent=2) + "\n")
    return work


def test_relate_still_reports_the_two_verdicts_in_isolation(project):
    """The starting point, and the thing this module is about. One claim
    agrees, one is refuted, and `claims.json` says nothing about the first
    being in trouble because of the second."""
    d = json.loads((project / "results/claims.json").read_text())
    assert d["verdicts"] == {"agrees": 1, "refuted": 1}
    assert not any("interaction_strength" in json.dumps(r)
                   for r in d["relationships"])


def test_the_refuted_prediction_falsifies_the_claim_that_made_it(project):
    g = BeliefGraph.from_project(project)
    c = next(c for c in g.claims
             if c.tested_by == "mace.e_int_kcal ~ mace.n_lig_atoms")
    assert c.status == "falsified"
    assert "OUTSIDE" in c.evidence, "the caveat carries the interval that made it"
    assert str(c) == "mace_e_int --approximates--> interaction_strength"


def test_the_relationship_that_agreed_now_carries_the_other_ones_caveat(project):
    """THE ANSWER THAT DID NOT EXIST BEFORE. `mace.e_int_kcal ~
    affinity.experimental_pKD` came back `agrees`; the assumption underneath it
    came back `falsified`; and every number in the `mace` dataset now says so.
    """
    g = BeliefGraph.from_project(project)
    delivered = {d["claim"]: d for d in g.caveats_on("mace")}
    assert "mace_e_int --approximates--> interaction_strength" in delivered
    row = delivered["mace_e_int --approximates--> interaction_strength"]
    assert row["status"] == "falsified"
    assert row["recomputed_from"] == "mace.e_int_kcal ~ mace.n_lig_atoms"
    assert not row["open_question"], "a settled negative is not a to-do"


def test_a_caveat_arrives_from_two_edges_away(project):
    """`mace` rests on the energy, the energy is evaluated on a docked pose,
    and the pose is an assumed stand-in for the real one. Nothing in
    `claims.json` mentions poses."""
    g = BeliefGraph.from_project(project)
    claims = [d["claim"] for d in g.caveats_on("mace")]
    assert "docked_pose --approximates--> crystal_pose" in claims
    assert g.in_doubt("mace_e_int") == ["crystal_pose", "docked_pose",
                                        "interaction_strength",
                                        "measured_affinity"]


def test_doubt_does_not_cross_supersedes_in_the_shipped_example(project):
    """The negative side, in the example a reader actually runs. The claim
    below is the SAME predicate against the SAME object as the falsified one —
    it is about the score this pipeline replaced, and it does not travel."""
    g = BeliefGraph.from_project(project)
    claims = [d["claim"] for d in g.caveats_on("mace")]
    assert "vina_score --approximates--> interaction_strength" not in claims
    assert len(claims) == 2


def test_the_measured_reference_is_not_implicated(project):
    """`affinity` is the yardstick the surrogate was judged against. It has
    nothing to answer for, and a model that warned about it would be training
    its reader to ignore warnings."""
    g = BeliefGraph.from_project(project)
    assert g.knows("affinity")
    assert g.caveats_on("affinity") == []


def test_an_unverifiable_verdict_becomes_untested_and_carries_no_evidence(
        tmp_path):
    """`unverifiable` is not a pass and not a refutation. It arrives as
    `untested` — nobody has looked — which raises an open question, carries a
    caveat, and must not carry evidence: the checker's reason is a fact about
    the check, not about the claim."""
    work = tmp_path / "example"
    shutil.copytree(EXAMPLE, work)
    ctx = work / "context.toml"
    ctx.write_text(ctx.read_text().replace('method = "pearson"',
                                           'method = "kendall"', 1))
    shutil.rmtree(work / "results", ignore_errors=True)
    (work / "results").mkdir(parents=True)
    (work / "results" / "claims.json").write_text(
        json.dumps(relate.check(work), indent=2) + "\n")

    g = BeliefGraph.from_project(work)
    c = next(c for c in g.claims
             if c.tested_by == "mace.e_int_kcal ~ affinity.experimental_pKD")
    assert c.status == "untested"
    assert c.evidence == ""
    assert c.open_question and c.carries_caveat
    assert any(d["claim"] == str(c) for d in g.caveats_on("mace"))


def test_a_status_beside_tested_by_is_refused(tmp_path):
    """A status that is about to be overwritten is still a status somebody
    opens the file and believes."""
    with pytest.raises(BeliefError, match="declares both `tested_by` and `status`"):
        graph([{"subject": "a", "predicate": "approximates", "object": "b",
                "status": "measured", "tested_by": "x ~ y"}], nodes=("a", "b"),
              recomputed={"x ~ y": {"status": "falsified", "why": "w"}})


def test_tested_by_naming_no_relationship_is_refused():
    with pytest.raises(BeliefError, match="names no relationship"):
        graph([{"subject": "a", "predicate": "approximates", "object": "b",
                "tested_by": "p ~ q"}], nodes=("a", "b"),
              recomputed={"x ~ y": {"status": "falsified", "why": "w"}})


def test_a_tested_claim_with_no_verdicts_to_read_is_refused(tmp_path):
    """Reporting the declared status instead would publish a belief nobody has
    checked — with the file that would have checked it missing."""
    work = tmp_path / "example"
    shutil.copytree(EXAMPLE, work)
    shutil.rmtree(work / "results", ignore_errors=True)
    with pytest.raises(BeliefError, match="does not exist"):
        BeliefGraph.from_project(work)


def test_a_project_with_no_belief_table_is_refused_rather_than_answered(tmp_path):
    """An empty answer here reads as a clean bill of health."""
    work = tmp_path / "example"
    shutil.copytree(EXAMPLE, work)
    ctx = work / "context.toml"
    ctx.write_text(ctx.read_text().split("[belief.rests_on]")[0])
    with pytest.raises(BeliefError, match="declares no \\[belief\\] table"):
        BeliefGraph.from_project(work)


def test_two_relationships_over_the_same_pair_cannot_be_named(tmp_path):
    """`tested_by` may identify a relationship by its two columns. Two
    relationships over the same pair make that ambiguous, and picking one would
    attach a belief to whichever happened to be first — so the pair stops being
    a usable name and the refusal points at `id` instead."""
    work = tmp_path / "example"
    shutil.copytree(EXAMPLE, work)
    d = relate.check(work)
    d["relationships"].append(dict(d["relationships"][0]))
    (work / "results").mkdir(parents=True, exist_ok=True)
    (work / "results" / "claims.json").write_text(json.dumps(d, indent=2))
    with pytest.raises(BeliefError, match="are over that pair of columns"):
        BeliefGraph.from_project(work)


# ---------------------------------------------------------------------------
# The RDF rendering, and what a reasoner can and cannot get from it
# ---------------------------------------------------------------------------

_CHAIN = """
    PREFIX evi: <https://w3id.org/EVI#>
    SELECT DISTINCT ?x WHERE { ?v evi:directlyChallenges/evi:supports+ ?x }"""

_ONE_HOP = """
    PREFIX evi: <https://w3id.org/EVI#>
    SELECT DISTINCT ?x WHERE { ?v evi:directlyChallenges/evi:supports ?x }"""


def test_the_property_chain_reaches_what_the_python_closure_reaches(project):
    """EVI's chain, evaluated by rdflib's own path engine rather than by our
    breadth-first search, on the same graph. If these two disagree about a
    refuted claim, one of them has a bug — and the whole argument for keeping a
    twenty-line closure beside an axiomatised one rests on their agreeing."""
    pytest.importorskip("rdflib")
    from claimcheck.graph import build
    from claimcheck.namespace import BASE

    g = build(project)
    derived = {str(r[0]).removeprefix(f"{BASE}artefact/")
               for r in g.query(_CHAIN) if f"{BASE}artefact/" in str(r[0])}
    bg = BeliefGraph.from_project(project)
    ours = {name for name in bg.rests_on
            if any(d["status"] == "falsified" for d in bg.caveats_on(name))}
    assert derived == ours == {"mace"}


def test_transitivity_is_what_makes_the_chain_reach_the_artefact(project):
    """Two-sided on the axiom itself: with `supports` transitive the chain
    reaches the artefact three hops away; one hop of the same path does not.
    If this passed either way, `owl:TransitiveProperty` would be decoration."""
    pytest.importorskip("rdflib")
    from claimcheck.graph import build
    from claimcheck.namespace import BASE

    g = build(project)
    one = {str(r[0]) for r in g.query(_ONE_HOP)}
    many = {str(r[0]) for r in g.query(_CHAIN)}
    assert many > one
    assert not any(x.startswith(f"{BASE}artefact/") for x in one)
    assert any(x.startswith(f"{BASE}artefact/") for x in many)


def test_the_reasoner_cannot_derive_doubt_that_nobody_challenged(project):
    """OPEN-WORLD SEMANTICS, stated as a test rather than as a paragraph.

    `docked_pose --approximates--> crystal_pose` is `assumed`: nobody has
    looked. Our closure delivers it; no axiom entails it, because absence of a
    challenge is not a challenge. The graph carries the status honestly and
    invents no `directlyChallenges` edge to make a reasoner agree.
    """
    pytest.importorskip("rdflib")
    from claimcheck.graph import build
    from claimcheck.namespace import BASE

    g = build(project)
    reached = {str(r[0]) for r in g.query(_CHAIN)}
    bg = BeliefGraph.from_project(project)
    assumed = "docked_pose --approximates--> crystal_pose"
    assert assumed in [d["claim"] for d in bg.caveats_on("mace")]
    assert f"{BASE}node/crystal_pose" not in reached
    # …and the claim IS in the graph, with its status, for a reader to see.
    rows = list(g.query("""
        PREFIX evi: <https://w3id.org/EVI#>
        PREFIX tl: <%sns#>
        SELECT ?s WHERE { ?c a evi:Claim ; tl:status "assumed" ; tl:subject ?s }
        """ % BASE))
    assert f"{BASE}node/docked_pose" in {str(r[0]) for r in rows}


def test_the_peer_edge_is_not_rendered_as_a_transitive_property():
    """The other half of the subset. `shares_approximation_with` is symmetric
    and not transitive, so it is NOT an `evi:supports` edge — rendering it as
    one would let the reasoner close over it and merge every peer into a single
    class. It is in the graph as a claim; it is not in the chain."""
    g = graph([claim("a", "shares_approximation_with", "b", "by_construction")],
              nodes=("a", "b"))
    assert g.supports_edges() == []
    directed = graph([claim("a", "approximates", "b", "by_construction")],
                     nodes=("a", "b"))
    assert directed.supports_edges() == [("b", "a")]


def test_the_restated_axioms_are_exactly_evis_two(project):
    """A restatement is only safe while it is true, so the chain is pinned to
    those two properties IN THAT ORDER — `( supports directlyChallenges )`
    would be a different axiom that happens to parse."""
    pytest.importorskip("rdflib")
    from rdflib.collection import Collection
    from rdflib.namespace import OWL, RDF

    from claimcheck.evi import CHAIN, EVI, TRANSITIVE, axioms

    g = axioms()
    lst = g.value(EVI.indirectlyChallenges, OWL.propertyChainAxiom)
    assert lst is not None, "no property chain was emitted"
    assert list(Collection(g, lst)) == list(CHAIN)
    assert list(CHAIN) == [EVI.directlyChallenges, EVI.supports]
    assert (TRANSITIVE, RDF.type, OWL.TransitiveProperty) in g
    assert TRANSITIVE == EVI.supports


def test_the_data_graph_imports_evi_rather_than_only_restating_it(project):
    """A reasoner that resolves imports needs nothing from `claimcheck.evi`.
    The restatement is for the one that does not."""
    pytest.importorskip("rdflib")
    from rdflib.namespace import OWL

    from claimcheck.evi import EVI_ONTOLOGY
    from claimcheck.graph import build

    g = build(project)
    assert EVI_ONTOLOGY in set(g.objects(None, OWL.imports))


def test_the_axioms_are_not_mixed_into_the_data_graph(project):
    """They would be the only blank nodes in it and the only subjects this
    project did not mint, which is the guard that keeps two runs diffable."""
    pytest.importorskip("rdflib")
    from rdflib.namespace import OWL

    from claimcheck.graph import build

    g = build(project)
    assert not list(g.triples((None, OWL.propertyChainAxiom, None)))


# ---------------------------------------------------------------------------
# The report a consumer reads
# ---------------------------------------------------------------------------

def test_the_report_is_json_and_carries_both_questions(project):
    from claimcheck import belief
    doc = json.loads(belief.render(project))
    assert doc["summary"]["claims"] == 6
    assert doc["summary"]["recomputed"] == 2
    assert doc["summary"]["vocabulary"] == {
        "predicates": 13, "statuses": 7, "node_kinds": 7,
        "open_question_statuses": list(v.OPEN_QUESTION),
        "caveat_statuses": list(v.CARRIES_CAVEAT)}
    assert [q["claim"] for q in doc["open_questions"]]
    assert doc["caveats"] and all(c["artefact"] == "mace" for c in doc["caveats"])


def test_the_summary_counts_claims_and_deliveries_separately(project):
    """One row per (artefact, claim) pair means four artefacts resting on one
    node report `4 caveat(s)` two lines under `1 claim(s)`."""
    g = BeliefGraph.from_project(project)
    g.rests_on["mace_copy"] = ("mace_e_int",)
    s = g.summary()
    assert s["caveats"] == 2 and s["caveat_deliveries"] == 4


def test_a_bare_string_in_rests_on_is_refused_rather_than_answered():
    """The other half of the line above, and the reason it is a tuple now.

    `rests_on` is a plain dict, so anything holding a `BeliefGraph` can assign
    into it. A bare string iterates as characters, matches no node, and makes
    `caveats_on` return `[]` — which reads as "nothing in doubt". This was
    measured, not imagined: the assertion above silently fell from 4 caveat
    deliveries to 2 the moment the field became a sequence.
    """
    g = graph([claim("a", "approximates", "b", status="falsified")],
              rests_on={"A": "a"})
    assert len(g.caveats_on("A")) == 1
    g.rests_on["A"] = "a"                     # the mistake, verbatim
    with pytest.raises(BeliefError, match="iterates as characters"):
        g.caveats_on("A")


# ---------------------------------------------------------------------------
# The core-not-extra decision, checked rather than asserted
# ---------------------------------------------------------------------------

_BAN = """
import sys
ALLOWED = set(sys.stdlib_module_names) | {"claimcheck"}
class Ban:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in ALLOWED:
            raise ImportError("this must not be needed: " + name)
        return None
sys.meta_path.insert(0, Ban())
sys.path.insert(0, %r)
%s
"""


def _under_ban(body: str):
    import subprocess
    import sys as _sys
    return subprocess.run([_sys.executable, "-c", _BAN % (str(REPO), body)],
                          capture_output=True, text=True, cwd=str(REPO))


def test_the_belief_model_answers_with_no_third_party_package_installed():
    """`tests/test_dependencies.py` runs the whole core behind this ban; this
    runs the belief model alone behind it, and proves the ban can fail.

    The second half is what makes the first worth anything. A meta-path finder
    that let something through would make the core-only claim decorative, and
    the failure mode is silent: the development machine has rdflib.
    """
    good = _under_ban(
        "from claimcheck.belief import BeliefGraph\n"
        "g = BeliefGraph.from_dict({'nodes': [{'id': 'a', 'kind': 'method'},\n"
        "                                     {'id': 'b', 'kind': 'method'}],\n"
        "  'claims': [{'subject': 'a', 'predicate': 'approximates',\n"
        "              'object': 'b', 'status': 'falsified',\n"
        "              'evidence': 'we checked'}],\n"
        "  'rests_on': {'A': 'a'}})\n"
        "assert g.caveats_on('A'), 'answered nothing'\n"
        "print('OK')\n")
    assert good.returncode == 0, good.stderr[-1500:]
    assert "OK" in good.stdout

    bad = _under_ban("import rdflib\nprint('OK')\n")
    assert bad.returncode != 0, "the ban let a third-party package through"
    assert "this must not be needed: rdflib" in bad.stderr


# ---------------------------------------------------------------------------
# `note`: the field for a nearby fact that is not evidence
# ---------------------------------------------------------------------------

def test_an_untested_claim_may_carry_a_note_and_may_not_carry_evidence():
    """Two-sided on the distinction the field exists for.

    The case this is drawn from: `mace_eint --is_cheaper_surrogate_for-->
    cascade` is untested, and beside it sits "tier 1 is 1.7 s, the cascade
    ~390 s, ~350x". A measurement about cost is not a measurement that one
    substitutes for the other. If `note` were merely evidence under another
    name, the second half of this test would fail.
    """
    g = graph([claim("a", "approximates", "b", status="untested",
                     note="a is 350x cheaper, which is not why we believe it")])
    assert g.claims[0].note.startswith("a is 350x")
    assert g.claims[0].evidence == ""
    assert g.claims[0].open_question

    with pytest.raises(BeliefError, match="'untested' and carries evidence"):
        graph([claim("a", "approximates", "b", status="untested",
                     evidence="the same sentence in the wrong field")])


def test_a_note_reaches_the_reader_beside_the_caveat_and_stays_labelled():
    """A note nobody sees is a comment. It travels with the delivery, in its
    own key, so a consumer cannot read it as the warrant."""
    g = graph([claim("a", "approximates", "b", status="falsified",
                     note="the replacement is not ready either")],
              rests_on={"A": "a"})
    d, = g.caveats_on("A")
    assert d["note"] == "the replacement is not ready either"
    assert d["evidence"] == _EV and d["evidence"] != d["note"]


def test_a_note_is_allowed_beside_tested_by_where_evidence_is_not():
    """The asymmetry is deliberate: a recomputation supplies the evidence, so
    a hand-written one beside it is a second answer — but no recomputation
    produces or contradicts a caveat about what the claim means."""
    rec = {"x ~ y": {"status": "falsified", "why": "measured, and it is not"}}
    g = BeliefGraph.from_dict(
        {"nodes": [{"id": "a", "kind": "method"}, {"id": "b", "kind": "method"}],
         "claims": [{"subject": "a", "predicate": "approximates", "object": "b",
                     "tested_by": "x ~ y", "note": "on 12 rows, all one target"}]},
        recomputed=rec)
    assert g.claims[0].status == "falsified"
    assert g.claims[0].note == "on 12 rows, all one target"
    assert g.claims[0].evidence == "measured, and it is not"


# ---------------------------------------------------------------------------
# `rests_on` as a list: one artefact, several nodes
# ---------------------------------------------------------------------------

def test_an_artefact_resting_on_several_nodes_inherits_from_every_one():
    """The reason the field is a list. A workflow rule exercises a
    representation AND a strategy AND a threshold; the map this was ported
    from has entries naming twelve nodes at once.

    Two-sided: naming one of the two nodes delivers one of the two caveats, so
    a single-valued field does not merely lose detail — it reports an artefact
    as half as doubtful as it is, and says nothing about the half it dropped.
    """
    claims = [claim("a", "approximates", "c", status="falsified"),
              claim("b", "approximates", "d", status="falsified")]
    both = graph(claims, rests_on={"R": ["a", "b"]})
    assert len(both.caveats_on("R")) == 2

    one = graph(claims, rests_on={"R": ["a"]})
    assert len(one.caveats_on("R")) == 1


def test_the_caveat_names_which_declared_node_carried_it():
    """"Does this artefact inherit it" is answerable by a union; "why" is not,
    and "why" is the first thing a reader asks of a warning."""
    g = graph([claim("a", "approximates", "c", status="falsified"),
               claim("b", "approximates", "d", status="falsified")],
              rests_on={"R": ["a", "b"]})
    via = {d["claim"]: d["rests_on"] for d in g.caveats_on("R")}
    assert via["a --approximates--> c"] == ["a"]
    assert via["b --approximates--> d"] == ["b"]


def test_a_bare_string_still_loads_as_a_list_of_one():
    """Every `context.toml` written before this change says `x = "node"`, and
    all of them must keep working — the change is a widening, not a break."""
    g = graph([claim("a", "approximates", "b", status="falsified")],
              rests_on={"A": "a"})
    assert g.rests_on["A"] == ("a",)
    assert len(g.caveats_on("A")) == 1


def test_an_artefact_declared_to_rest_on_nothing_is_refused():
    """An empty list answers `knows` with True and `caveats_on` with `[]` —
    "declared" and "nothing in doubt", which together read as a clean bill of
    health for an artefact nobody has attached to anything."""
    with pytest.raises(BeliefError, match="rests_on\\['A'\\] is empty"):
        graph([claim("a", "approximates", "b")], rests_on={"A": []})


def test_the_same_node_named_twice_in_one_entry_is_refused():
    """Harmless to the closure and not to the counts: every caveat reachable
    through it would be delivered twice, and `caveat_deliveries` is a
    published number."""
    with pytest.raises(BeliefError, match="names the same node twice"):
        graph([claim("a", "approximates", "b")], rests_on={"A": ["a", "a"]})


# ---------------------------------------------------------------------------
# `defined_in`: one project owns a claim, several read it
# ---------------------------------------------------------------------------

def _project(d, name, claims, nodes=("a", "b")):
    """A minimal claimcheck project on disk, belief table only."""
    root = d / name
    root.mkdir(parents=True)
    lines = ['[project]', f'name = "{name}"', '', '[belief]', '']
    for n in nodes:
        lines += ['[[belief.nodes]]', f'id = "{n}"', 'kind = "method"', '']
    for c in claims:
        lines.append('[[belief.claims]]')
        lines += [f'{k} = {v!r}'.replace("'", '"') for k, v in c.items()]
        lines.append('')
    (root / "context.toml").write_text("\n".join(lines))
    return root


def test_a_claim_is_read_from_the_project_that_owns_it(tmp_path):
    """The measured case: one statement — that an assay number stands in for
    the physical binding free energy — was `assumed` with no evidence in four
    projects at once, and every RMSE against experiment rested on it."""
    _project(tmp_path, "owner",
             [{"subject": "a", "predicate": "approximates", "object": "b",
               "status": "assumed"}])
    reader = _project(tmp_path, "reader",
                      [{"subject": "a", "predicate": "approximates",
                        "object": "b", "defined_in": "owner",
                        "note": "every number in this project rests on it"}])
    g = BeliefGraph.from_project(reader)
    c, = g.claims
    assert c.status == "assumed" and c.defined_in == "owner"
    assert c.note == "every number in this project rests on it"


def test_settling_it_in_the_owner_settles_it_everywhere(tmp_path):
    """The reason for the field, and the half that would be missing if it were
    only a consistency check: the reader is not edited at all."""
    owner = _project(tmp_path, "owner",
                     [{"subject": "a", "predicate": "approximates",
                       "object": "b", "status": "assumed"}])
    reader = _project(tmp_path, "reader",
                      [{"subject": "a", "predicate": "approximates",
                        "object": "b", "defined_in": "owner"}])
    assert BeliefGraph.from_project(reader).claims[0].status == "assumed"
    assert not BeliefGraph.from_project(reader).claims[0].carries_caveat is False

    ctx = owner / "context.toml"
    ctx.write_text(ctx.read_text().replace(
        'status = "assumed"',
        'status = "falsified"\nevidence = "measured, and it does not"'))
    after = BeliefGraph.from_project(reader).claims[0]
    assert after.status == "falsified"
    assert after.evidence == "measured, and it does not"


def test_a_second_answer_beside_defined_in_is_refused(tmp_path):
    """Same rule as `tested_by`: a hand-written status beside an imported one
    is a second answer, and it is the one a reader opening the file meets."""
    _project(tmp_path, "owner",
             [{"subject": "a", "predicate": "approximates", "object": "b",
               "status": "assumed"}])
    for extra in ({"status": "measured", "evidence": "x"},
                  {"evidence": "x"},
                  {"tested_by": "x ~ y"}):
        d = tmp_path / f"r_{'_'.join(extra)}"
        reader = _project(d, "reader",
                          [{"subject": "a", "predicate": "approximates",
                            "object": "b", "defined_in": "owner"} | extra])
        (d / "owner").symlink_to(tmp_path / "owner")
        with pytest.raises(BeliefError, match="declares both `defined_in`"):
            BeliefGraph.from_project(reader)


def test_importing_a_claim_the_owner_does_not_make_is_refused(tmp_path):
    """It would invent a status nobody wrote — the failure this whole field is
    meant to prevent, arriving through the mechanism meant to prevent it."""
    _project(tmp_path, "owner",
             [{"subject": "a", "predicate": "supersedes", "object": "b",
               "status": "by_construction"}])
    reader = _project(tmp_path, "reader",
                      [{"subject": "a", "predicate": "approximates",
                        "object": "b", "defined_in": "owner"}])
    with pytest.raises(BeliefError, match="asserts no claim"):
        BeliefGraph.from_project(reader)


def test_an_owner_that_itself_imports_is_refused(tmp_path):
    """One hop, enforced at the project level. Two projects pointing at each
    other cannot even be built, so no cycle detection is needed anywhere."""
    _project(tmp_path, "far",
             [{"subject": "a", "predicate": "approximates", "object": "b",
               "status": "assumed"}])
    _project(tmp_path, "middle",
             [{"subject": "a", "predicate": "approximates", "object": "b",
               "defined_in": "far"}])
    reader = _project(tmp_path, "reader",
                      [{"subject": "a", "predicate": "approximates",
                        "object": "b", "defined_in": "middle"}])
    with pytest.raises(BeliefError, match="itself uses `defined_in`"):
        BeliefGraph.from_project(reader)


def test_the_imported_claims_nodes_must_still_be_declared_locally(tmp_path):
    """Nothing is merged across projects. Only the verdict travels, so a
    reader's graph stays a graph over nodes that reader declared."""
    _project(tmp_path, "owner",
             [{"subject": "a", "predicate": "approximates", "object": "b",
               "status": "assumed"}])
    reader = _project(tmp_path, "reader",
                      [{"subject": "a", "predicate": "approximates",
                        "object": "b", "defined_in": "owner"}],
                      nodes=("a",))
    with pytest.raises(BeliefError, match="not a declared node"):
        BeliefGraph.from_project(reader)


def test_defined_in_needs_a_directory_to_resolve_against():
    with pytest.raises(BeliefError, match="no directory to resolve it"):
        BeliefGraph.from_dict(
            {"nodes": [{"id": "a", "kind": "method"},
                       {"id": "b", "kind": "method"}],
             "claims": [{"subject": "a", "predicate": "approximates",
                         "object": "b", "defined_in": "somewhere"}]})


def test_naming_a_project_that_is_not_there_is_refused(tmp_path):
    reader = _project(tmp_path, "reader",
                      [{"subject": "a", "predicate": "approximates",
                        "object": "b", "defined_in": "ghost"}])
    with pytest.raises(BeliefError, match="does not exist"):
        BeliefGraph.from_project(reader)


def test_the_work_queue_says_which_entries_nobody_can_work_on():
    """`open_questions` is a queue. An entry blocked by a measured fact
    elsewhere is not a next experiment, and the closed vocabulary has one word
    for both — so the note has to travel with the queue, not only with the
    caveat delivery."""
    g = graph([claim("a", "approximates", "b", status="untested",
                     note="BLOCKED: b cannot be built on this corpus"),
               claim("a", "approximates", "c", status="untested")],
              nodes=("a", "b", "c"))
    q = {d["claim"]: d["note"] for d in g.open_questions()}
    assert q["a --approximates--> b"].startswith("BLOCKED")
    assert q["a --approximates--> c"] == ""


def test_defined_in_resolves_siblings_from_a_relative_root(tmp_path, monkeypatch):
    """`Path(".").parent` is `.`. Every project's own Snakefile runs
    `claimcheck.belief .` from inside the project directory, so without
    resolving first the sibling lookup happens INSIDE the project and finds
    nothing. The in-process load passed on an absolute path and the rule failed
    on the relative one — which is why this test uses a relative path."""
    _project(tmp_path, "owner",
             [{"subject": "a", "predicate": "approximates", "object": "b",
               "status": "assumed"}])
    _project(tmp_path, "reader",
             [{"subject": "a", "predicate": "approximates", "object": "b",
               "defined_in": "owner"}])
    monkeypatch.chdir(tmp_path / "reader")
    g = BeliefGraph.from_project(".")
    assert g.claims[0].status == "assumed"


# ---------------------------------------------------------------------------
# Naming one of two relationships over the same columns
# ---------------------------------------------------------------------------

def _rel(x, y, status="measured", rid=""):
    r = {"x": x, "y": y, "status": status, "why": "the fixture said so"}
    if rid:
        r["id"] = rid
    return r


def test_a_claim_can_name_a_relationship_by_id():
    """Declaring one pair of columns twice at two resampling units is worth
    doing — the row-level interval is what most published numbers use, the
    cluster-level one is what the unit of independence is. `"x ~ y"` cannot
    tell them apart, so an id can."""
    rec = BeliefGraph._index({"relationships": [
        _rel("a.p", "a.q", "measured", rid="by_row"),
        _rel("a.p", "a.q", "falsified", rid="by_cluster")]})
    g = BeliefGraph.from_dict(
        {"nodes": [{"id": "a", "kind": "method"}, {"id": "b", "kind": "method"}],
         "claims": [{"subject": "a", "predicate": "approximates",
                     "object": "b", "tested_by": "by_cluster"}]},
        recomputed=rec)
    assert g.claims[0].status == "falsified"


def test_naming_an_ambiguous_pair_says_what_to_do_about_it():
    """Two relationships over one pair and no id. The old message said
    "distinguish them", which named no way to do it."""
    rec = BeliefGraph._index({"relationships": [
        _rel("a.p", "a.q", "measured"), _rel("a.p", "a.q", "falsified")]})
    with pytest.raises(BeliefError, match="two resampling units"):
        BeliefGraph.from_dict(
            {"nodes": [{"id": "a", "kind": "method"},
                       {"id": "b", "kind": "method"}],
             "claims": [{"subject": "a", "predicate": "approximates",
                         "object": "b", "tested_by": "a.p ~ a.q"}]},
            recomputed=rec)


def test_a_unique_pair_still_works_without_an_id():
    """Every context written before ids existed keeps working."""
    rec = BeliefGraph._index({"relationships": [_rel("a.p", "a.q", "falsified")]})
    g = BeliefGraph.from_dict(
        {"nodes": [{"id": "a", "kind": "method"}, {"id": "b", "kind": "method"}],
         "claims": [{"subject": "a", "predicate": "approximates",
                     "object": "b", "tested_by": "a.p ~ a.q"}]},
        recomputed=rec)
    assert g.claims[0].status == "falsified"


def test_two_relationships_sharing_an_id_are_refused():
    with pytest.raises(BeliefError, match="share the id"):
        BeliefGraph._index({"relationships": [
            _rel("a.p", "a.q", rid="same"), _rel("a.r", "a.s", rid="same")]})
