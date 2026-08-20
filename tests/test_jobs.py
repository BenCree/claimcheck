"""Job records: the four refusals, and the one PROV distinction they exist for.

Every test here is two-sided in the sense the package requires — the refusals
are shown refusing *and* shown passing the neighbouring legal case, because a
refusal that fires on everything is the same defect as one that fires on
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimcheck.jobs import Job, JobError, JobLog, VERDICTS, check_job

REPO = Path(__file__).resolve().parents[1]


def rec(**over) -> dict:
    """A record that passes every check, so a test that changes one field is
    testing that field and not the four rules it did not mean to touch."""
    d = {"id": "67444", "question": "does a pocket-aware acquisition beat one "
                                    "that does not?",
         "expected": ["results/q22/mace_fp.csv"],
         "check": "python cluster/check_q22.py",
         "verdict": "agrees",
         "generated": ["results/q22/mace_fp.csv"],
         "started": "2026-08-04T09:12:00", "ended": "2026-08-04T11:48:00",
         "agent": "thira"}
    return d | over


def log(*records: dict, **kw) -> JobLog:
    return JobLog.from_lines([json.dumps(r) for r in records], **kw)


def project(tmp_path: Path, *records: dict, jobs_table: str | None = None) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "jobs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    table = '[jobs]\nrecords = "jobs.jsonl"\nagent = "thira"\n' \
        if jobs_table is None else jobs_table
    (root / "context.toml").write_text(
        '[project]\nname = "t"\n\n' + table)
    return root


# ---------------------------------------------------------------------------
# The vocabulary is not a new one
# ---------------------------------------------------------------------------

def test_the_verdicts_are_relates_verdicts_and_not_a_second_set():
    """Proven by identity, not by a comment saying so. If somebody adds a
    fourth word here, or a fourth there, this fails rather than letting two
    modules adjudicate the same evidence in two vocabularies."""
    from claimcheck.relate import STATUS
    assert VERDICTS == STATUS
    assert set(VERDICTS) == {"agrees", "refuted", "unverifiable"}
    assert set(VERDICTS.values()) == {"measured", "falsified", "untested"}


# ---------------------------------------------------------------------------
# The four refusals, each beside the legal case it must not refuse
# ---------------------------------------------------------------------------

def test_a_record_with_no_question_is_refused_and_one_with_a_question_is_not():
    """The field the whole module exists for. Without it the record documents
    that something happened, which a scheduler's accounting already does."""
    assert log(rec()).jobs[0].question
    with pytest.raises(JobError, match="has no 'question'"):
        log(rec(question=""))


def test_a_verdict_over_nothing_declared_is_refused():
    """112 of 190 jobs in the registry this came from were in this state and
    were reported as needing attention. Nothing had been declared, so no check
    on them could have failed.

    The other side: the same record with a `check` and no `expected` is legal —
    an adjudicating command is a declaration even when it names no paths.
    """
    with pytest.raises(JobError, match="could have failed"):
        log(rec(expected=[], check="", generated=[]))
    ok = log(rec(expected=[], check="python check.py", generated=[]))
    assert ok.jobs[0].adjudicable and ok.jobs[0].verdict == "agrees"


def test_a_pass_whose_declared_outputs_are_absent_is_refused():
    """30 jobs exited 0 with their outputs not there. A pass is a claim that
    the criterion was met; an exit status is not that claim.

    Two-sided twice over: the same absence under a `refuted` verdict is exactly
    what a failure record should look like, and is accepted.
    """
    with pytest.raises(JobError, match="passed, and 1 of its 1 declared"):
        log(rec(generated=[]))
    failed = log(rec(verdict="refuted", generated=[]))
    assert failed.jobs[0].missing == ("results/q22/mace_fp.csv",)


def test_confirming_outputs_and_recording_no_verdict_is_refused():
    """Something looked. A record that says what it saw and not what it
    concluded reads as neutral, and it is not."""
    with pytest.raises(JobError, match="records no verdict"):
        log(rec(verdict=""))
    pending = log(rec(verdict="", generated=[]))
    assert pending.jobs[0].verdict == "" and pending.jobs[0].adjudicable


def test_a_verdict_outside_the_vocabulary_is_refused():
    with pytest.raises(JobError, match="the vocabulary has 3"):
        log(rec(verdict="PASS"))
    assert log(rec(verdict="unverifiable", generated=[])).jobs[0].status \
        == "untested"


# ---------------------------------------------------------------------------
# A gap is not a defect
# ---------------------------------------------------------------------------

def test_nothing_declared_is_counted_apart_from_failures():
    """The distinction the registry could not draw. A job nobody wrote a
    criterion for has not failed — and a list where those sit beside real
    failures is a list that trains a reader to scroll past the failures."""
    lg = log(rec(id="a"),
             rec(id="b", verdict="refuted", generated=[]),
             rec(id="c", expected=[], check="", verdict="", generated=[]))
    s = lg.summary()
    assert s["jobs"] == 3
    assert s["by_verdict"] == {"agrees": 1, "refuted": 1, "unadjudicated": 1}
    assert s["nothing_declared_to_check"] == 1
    assert s["adjudicable"] == 2
    assert [j.id for j in lg.undeclared()] == ["c"]


# ---------------------------------------------------------------------------
# Reading the log
# ---------------------------------------------------------------------------

def test_a_truncated_last_line_stops_the_load_rather_than_being_skipped():
    """A JSONL log is append-only, so a half-written final line is the ordinary
    shape of an interrupted write — and that line is the job somebody is most
    likely asking about. Blank lines are not the same thing and are ignored."""
    good = json.dumps(rec())
    assert len(JobLog.from_lines([good, "", "  "]).jobs) == 1
    with pytest.raises(JobError, match="line 2 is not JSON"):
        JobLog.from_lines([good, '{"id": "trunc", "ques'])


def test_two_records_under_one_id_are_refused():
    """Publishing the later one silently would hide a contradiction between
    them — including two different verdicts on one run."""
    with pytest.raises(JobError, match="line 1 and line 2"):
        log(rec(), rec(verdict="refuted", generated=[]))


def test_the_table_default_agent_fills_in_and_a_record_may_override_it():
    lg = log(rec(agent=""), rec(id="x", agent="paros"), agent="thira")
    assert [j.agent for j in lg.jobs] == ["thira", "paros"]


def test_no_jobs_table_is_refused_rather_than_answered_empty(tmp_path):
    """"Nothing was submitted" and "nobody wrote it down" are different
    answers, and only one of them is good news."""
    root = project(tmp_path, rec())
    (root / "context.toml").write_text('[project]\nname = "t"\n')
    with pytest.raises(JobError, match="no \\[jobs\\] table"):
        JobLog.from_project(root)


def test_a_missing_records_file_is_refused_rather_than_read_as_empty(tmp_path):
    root = project(tmp_path, rec())
    (root / "jobs.jsonl").unlink()
    with pytest.raises(JobError, match="A missing log is not an empty one"):
        JobLog.from_project(root)


def test_from_project_reads_the_declared_file(tmp_path):
    lg = JobLog.from_project(project(tmp_path, rec(), rec(id="b")))
    assert [j.id for j in lg.jobs] == ["67444", "b"]
    assert lg.source == "jobs.jsonl"
    assert lg.report()["jobs"][0]["question"].startswith("does a pocket-aware")


# ---------------------------------------------------------------------------
# PROV — the distinction the mapping exists for
# ---------------------------------------------------------------------------

def _graph(tmp_path, *records):
    pytest.importorskip("rdflib")
    from claimcheck.jobs import build
    return build(project(tmp_path, *records))


def test_an_expected_output_is_not_asserted_to_exist_and_a_confirmed_one_is(
        tmp_path):
    """THE LOAD-BEARING TEST. `prov:generated` asserts that an entity came into
    existence. A failed job whose declared outputs were emitted there would
    publish, in the interchange format, the exact claim its check refused.

    Two-sided on the same path in the same graph: it appears on the plan under
    `dct:requires` for the failed job, and as a generated entity only for the
    one that produced it.
    """
    from rdflib import URIRef
    from rdflib.namespace import DCTERMS, PROV
    from claimcheck.namespace import BASE

    g = _graph(tmp_path,
               rec(id="ok"),
               rec(id="bad", verdict="refuted", generated=[],
                   expected=["results/q22/mace_fp.csv"]))
    path = "results/q22/mace_fp.csv"
    ent = URIRef(f"{BASE}entity/{path}")

    generators = {str(s) for s, _, _ in g.triples((None, PROV.generated, None))}
    assert f"{BASE}job/ok" in generators
    assert f"{BASE}job/bad" not in generators, (
        "a refuted job asserted that its missing output exists")

    required = {str(s) for s, _, o in g.triples((None, DCTERMS.requires, None))
                if str(o) == path}
    assert required == {f"{BASE}job/ok/criterion", f"{BASE}job/bad/criterion"}
    assert (None, PROV.generated, None) in g
    assert len(list(g.triples((None, PROV.generated, ent)))) == 1


def test_the_criterion_is_a_plan_reached_through_a_qualified_association(
        tmp_path):
    """Plain PROV-O, walked as a stranger's reasoner would walk it, rather than
    asserted by looking for our own triples."""
    from rdflib.namespace import PROV
    from claimcheck.namespace import BASE

    g = _graph(tmp_path, rec())
    from rdflib.namespace import RDF

    act = list(g.subjects(PROV.qualifiedAssociation, None))
    assert [str(a) for a in act] == [f"{BASE}job/67444"]
    assoc = next(g.objects(act[0], PROV.qualifiedAssociation))
    plan = next(g.objects(assoc, PROV.hadPlan))
    assert (plan, RDF.type, PROV.Plan) in g
    assert (assoc, RDF.type, PROV.Association) in g
    assert str(next(g.objects(assoc, PROV.agent))) == f"{BASE}agent/thira"
    assert (act[0], PROV.wasAssociatedWith, None) in g


def test_the_question_lands_where_a_catalogue_reads_it(tmp_path):
    from rdflib import URIRef
    from rdflib.namespace import DCTERMS, PROV, RDF
    from claimcheck.namespace import BASE

    g = _graph(tmp_path, rec())
    job = URIRef(f"{BASE}job/67444")
    assert (job, RDF.type, PROV.Activity) in g
    desc = str(next(g.objects(job, DCTERMS.description)))
    assert desc.startswith("does a pocket-aware")


def test_a_superseded_job_says_so_in_dublin_core_and_needs_no_verdict(tmp_path):
    """Not a fourth outcome. `dct:isReplacedBy` says more than "N/A" and says it
    in a term a stranger's tooling already understands."""
    from rdflib import URIRef
    from rdflib.namespace import DCTERMS
    from claimcheck.namespace import BASE

    g = _graph(tmp_path, rec(id="old", verdict="", generated=[],
                             superseded_by="new"))
    assert (URIRef(f"{BASE}job/old"), DCTERMS.isReplacedBy,
            URIRef(f"{BASE}job/new")) in g


def test_a_timestamp_it_cannot_interpret_is_carried_and_not_reformatted(
        tmp_path):
    """A record's own string survives. Restating a timestamp in a shape this
    module invented would make the published record disagree with the log."""
    from rdflib import URIRef
    from rdflib.namespace import PROV, XSD
    from claimcheck.namespace import BASE

    g = _graph(tmp_path, rec(started="2026-08-04T09:12:00", ended="Tue 4 Aug"))
    job = URIRef(f"{BASE}job/67444")
    assert next(g.objects(job, PROV.startedAtTime)).datatype == XSD.dateTime
    ended = next(g.objects(job, PROV.endedAtTime))
    assert ended.datatype is None and str(ended) == "Tue 4 Aug"


def test_a_job_with_nothing_declared_emits_no_plan(tmp_path):
    """The graph must not manufacture a criterion nobody wrote. An empty plan
    in the record would look like a declaration and adjudicate nothing."""
    from rdflib.namespace import PROV

    g = _graph(tmp_path, rec(expected=[], check="", verdict="", generated=[]))
    assert not list(g.subjects(PROV.hadPlan, None))
    assert list(g.triples((None, None, PROV.Activity)))


# ---------------------------------------------------------------------------
# It reads; it does not record
# ---------------------------------------------------------------------------

def test_it_writes_nothing_and_hashes_nothing():
    """The rule that shaped this package: Snakemake writes the run receipt.
    A module about jobs is the most tempting place to write a second one."""
    import ast

    tree = ast.parse((REPO / "claimcheck" / "jobs.py").read_text())
    # THE DOCSTRING IS STRIPPED, and the first version of this test did not
    # strip it: the module explains in prose that it never runs `sbatch` or
    # `ssh`, and the scan matched its own explanation. A checker that fails on
    # a file for saying what it does not do is a checker that gets deleted.
    if tree.body and isinstance(tree.body[0], ast.Expr) \
            and isinstance(tree.body[0].value, ast.Constant):
        tree.body.pop(0)
    code = ast.unparse(tree)
    for banned in ("hashlib", "sha256", "md5", "write_text", "mkdir",
                   "subprocess", "sbatch", "ssh", "Popen", "urlopen"):
        assert banned not in code, f"jobs.py reaches for {banned!r}"


def test_check_job_is_reachable_on_its_own():
    """The refusals are the part with no other home, so they must be usable
    without a file, a project or a graph library."""
    check_job(Job(id="a", question="why", expected=("x",), check="c",
                  verdict="agrees", generated=("x",)))
    with pytest.raises(JobError):
        check_job(Job(id="a", question=""))


# ---------------------------------------------------------------------------
# The shipped example carries all three states
# ---------------------------------------------------------------------------

def test_the_example_demonstrates_pass_refute_and_nothing_declared():
    """A format registered with no worked example is an undocumented one, and
    the three states are the whole reason this module exists — a reader who
    only ever sees a passing job cannot tell what the refusals are for."""
    lg = JobLog.from_project(REPO / "example")
    s = lg.summary()
    assert s["by_verdict"] == {"agrees": 1, "refuted": 1, "unadjudicated": 1}
    assert s["nothing_declared_to_check"] == 1
    assert s["questions_recorded"] == s["jobs"] == 3


def test_the_examples_refuted_job_publishes_no_entity_for_its_missing_output():
    """The same distinction, on the bytes that ship, rather than on a fixture
    built to make it true."""
    pytest.importorskip("rdflib")
    from rdflib import URIRef
    from rdflib.namespace import DCTERMS, PROV

    from claimcheck.jobs import build
    from claimcheck.namespace import BASE

    g = build(REPO / "example")
    refuted = URIRef(f"{BASE}job/score-2026-08-05")
    assert not list(g.objects(refuted, PROV.generated))
    plan = next(g.objects(next(g.objects(refuted, PROV.qualifiedAssociation)),
                          PROV.hadPlan))
    assert [str(o) for o in g.objects(plan, DCTERMS.requires)] \
        == ["results/ordering.csv"]
