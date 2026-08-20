"""The grade ladder, checked against a Snakemake store that Snakemake wrote.

Every fixture here is two-sided, and the reason is the module under test: a
grader is exactly the kind of code that passes by agreeing with itself. So each
property is watched to FAIL on a file that should not earn it and to PASS on one
that should, and the two live in the same test where the contrast is visible.

The store is never faked except in one place — the `incomplete` marker, which
only a killed job writes — and the encoding used to fake it is first checked
against a filename Snakemake itself produced.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from _snakemake import snakemake_cmd, why_not

from claimcheck import grading

#: One deterministic rule and one that is not, which is the whole difference
#: between EXECUTED and REPRODUCED. `wobbly` appends the clock in nanoseconds,
#: so a re-run cannot come back byte-identical.
SNAKEFILE = """
rule all:
    input: "results/stable.txt", "results/wobbly.txt"

rule stable:
    input: "in.txt"
    output: "results/stable.txt"
    shell: "cat {input} > {output}"

rule wobbly:
    input: "in.txt"
    output: "results/wobbly.txt"
    shell: "cat {input} > {output}; date +%s%N >> {output}"
"""


def _run_snakemake(work: Path, *targets: str) -> subprocess.CompletedProcess:
    sm = snakemake_cmd()
    assert sm is not None
    return subprocess.run([*sm, "-c1", *(targets or ("all",))], cwd=str(work),
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A real project, built by a real Snakemake run. Never reused across tests
    that mutate it — the mutating tests copy it first."""
    if snakemake_cmd() is None:
        pytest.skip(why_not())
    work = tmp_path_factory.mktemp("graded") / "proj"
    work.mkdir(parents=True)
    (work / "Snakefile").write_text(SNAKEFILE)
    (work / "in.txt").write_text("hello\n")
    r = _run_snakemake(work)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    return work


@pytest.fixture
def fresh(project, tmp_path):
    """A private copy, so a test may edit, touch and delete freely."""
    work = tmp_path / "proj"
    shutil.copytree(project, work, symlinks=True)
    return work


# ---------------------------------------------------------------------------
# Reading Snakemake's store — the thing this module refuses to duplicate
# ---------------------------------------------------------------------------

def test_the_record_is_found_where_snakemake_put_it_and_nowhere_else(fresh):
    """PASS side: an output Snakemake made. FAIL side: a file it did not.

    If the second returned a record too, the encoding would be matching by
    accident and every grade above NONE would be unearned.
    """
    rec = grading.run_record(fresh, "results/stable.txt")
    assert rec is not None, "no record for a file Snakemake definitely produced"
    assert rec["rule"] == "stable"
    assert any(k.endswith("in.txt") for k in rec["input_checksums"]), rec

    (fresh / "results" / "stray.txt").write_text("nobody made me\n")
    assert grading.run_record(fresh, "results/stray.txt") is None


def test_the_key_encoding_matches_a_filename_snakemake_itself_wrote(project):
    """The one cross-check that matters, because the rest of this file trusts it.

    `_record_path` is a reimplementation of Snakemake's private encoding. Asking
    it to reproduce a name Snakemake actually chose is the only way to know it
    still agrees; the alternative is a test that passes on both packages being
    wrong in the same way.
    """
    store = project / ".snakemake" / "metadata"
    written = {p.resolve() for p in store.rglob("*") if p.is_file()}
    ours = grading._record_path(store, "results/stable.txt").resolve()
    assert ours in written, f"{ours} is not among {sorted(written)}"


def test_a_long_output_path_is_split_the_way_snakemake_splits_it():
    """Names over the filesystem limit become directories, all but the last
    prefixed with `@`. Untested, this branch would be found by whoever first
    wrote a rule with a long output name, at the moment their grade read NONE."""
    p = grading._record_path(Path("/store"), "x" * 400)
    parts = p.relative_to("/store").parts
    assert len(parts) > 1, "a 400-character key was not split at all"
    assert all(s.startswith("@") for s in parts[:-1])
    assert not parts[-1].startswith("@")


def test_grading_writes_nothing_at_all(fresh):
    """THE DESIGN CLAIM, checked rather than asserted.

    The whole argument for this module is that Snakemake's record is the record
    and nothing here keeps a second one. A stray sidecar would make that false
    silently, so the tree is digested before and after a full grading pass.
    """
    def snapshot():
        return {str(p.relative_to(fresh)): grading.digest(p)
                for p in sorted(fresh.rglob("*")) if p.is_file()}

    before = snapshot()
    grading.report(fresh)
    grading.quotable(fresh)
    assert snapshot() == before


# ---------------------------------------------------------------------------
# The rungs
# ---------------------------------------------------------------------------

def test_executed_and_none_are_told_apart(fresh):
    """PASS side: produced by a completed run. FAIL side: on disk, unclaimed."""
    good = grading.grade(fresh, "results/stable.txt")
    assert (good.grade, good.quotable, good.rule) == ("EXECUTED", True, "stable")

    (fresh / "results" / "unclaimed.csv").write_text("a,b\n1,2\n")
    bad = grading.grade(fresh, "results/unclaimed.csv")
    assert (bad.grade, bad.quotable) == ("NONE", False)
    assert "no rule" in bad.why


def test_a_changed_input_makes_the_output_unquotable(fresh):
    """The number on disk was not computed from the data now present.

    Both sides in one test: the same file, quotable before the input moves and
    refused after, with the moved input named.
    """
    before = grading.grade(fresh, "results/stable.txt")
    assert before.quotable and not before.stale

    (fresh / "in.txt").write_text("something else entirely\n")

    after = grading.grade(fresh, "results/stable.txt")
    assert after.grade == "EXECUTED", "a moved input is not 'the rule never ran'"
    assert after.stale and not after.quotable
    assert [Path(c).name for c in after.changed_inputs] == ["in.txt"]
    assert "not computed from the data now present" in after.why


def test_an_output_touched_after_its_run_is_refused(fresh):
    """Snakemake records no digest for its own outputs, so mtime is the only
    signal there is. It found a real one: all twelve files under `results/` in
    the three projects at `al_mace/datasets/claimcheck/` are 1.1-1.3 s newer
    than their own recorded endtime, while every `data/` file matches exactly."""
    target = fresh / "results/stable.txt"
    assert grading.grade(fresh, target).quotable

    os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 600))

    after = grading.grade(fresh, target)
    assert after.grade == "EXECUTED" and after.stale and not after.quotable
    assert "newer than the endtime" in after.why


def test_a_job_that_started_and_never_finished_is_declared(fresh):
    """The marker Snakemake leaves when a run is killed. Faked here because the
    honest way to produce one is to kill a job mid-write, and the encoding used
    to fake it is the one checked against Snakemake's own filename above."""
    assert grading.grade(fresh, "results/stable.txt").grade == "EXECUTED"

    marker = grading._record_path(fresh / ".snakemake" / "incomplete",
                                  "results/stable.txt")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"external_jobid": None}))

    g = grading.grade(fresh, "results/stable.txt")
    assert (g.grade, g.quotable) == ("DECLARED", False)
    assert "never finished" in g.why


def test_a_record_that_outlived_its_file_is_declared_not_executed(fresh):
    """A deleted output with a surviving record. Reporting EXECUTED here would
    be the grader describing a file that is not there."""
    assert grading.grade(fresh, "results/stable.txt").grade == "EXECUTED"
    (fresh / "results/stable.txt").unlink()
    g = grading.grade(fresh, "results/stable.txt")
    assert (g.grade, g.quotable) == ("DECLARED", False)
    assert "not on disk" in g.why


def test_the_quotable_line_sits_where_the_ladder_says(fresh):
    """`DECLARED` is below the line and `EXECUTED` is on it, and staleness
    overrides both directions. Cheap, and it is the property every other test
    here is written in terms of."""
    mk = grading.Grade
    assert not mk("p", "NONE").quotable
    assert not mk("p", "DECLARED").quotable
    assert mk("p", "EXECUTED").quotable
    assert mk("p", "REPRODUCED").quotable
    assert mk("p", "VERIFIED").quotable
    assert not mk("p", "VERIFIED", stale=True).quotable


# ---------------------------------------------------------------------------
# The top two rungs, which cost a re-run
# ---------------------------------------------------------------------------

def test_reproduction_separates_a_deterministic_rule_from_one_that_is_not(fresh):
    """THE HEADLINE, and the reason the top two rungs are not free.

    Both files are `EXECUTED` and both are up to date, which is what a workflow
    engine can tell you. Only re-running says that one of them is a function of
    its inputs and the other is a function of the clock.
    """
    done = grading.reproduce(fresh)
    assert done["results/stable.txt"] is True
    assert done["results/wobbly.txt"] is False, (
        "a rule that appends the nanosecond clock came back identical, which "
        "means the re-run did not happen")

    graded = grading.grade_all(fresh, reproduced=[k for k, v in done.items() if v])
    assert graded["results/stable.txt"].grade == "VERIFIED"
    assert graded["results/wobbly.txt"].grade == "EXECUTED"


def test_reproduction_does_not_touch_the_tree_it_is_checking(fresh):
    """The recorded incident this is written against: the predecessor's
    equivalent promised a sandbox, ran a real `--force` in the real tree, and
    destroyed three result files. Bytes AND mtimes, because a re-run in place
    would leave the content identical for the deterministic rule and only the
    timestamps would show it."""
    def snapshot():
        return {str(p.relative_to(fresh)): (grading.digest(p), p.stat().st_mtime)
                for p in sorted(fresh.rglob("*")) if p.is_file()}

    before = snapshot()
    grading.reproduce(fresh)
    assert snapshot() == before


def test_a_reproduction_clears_an_mtime_doubt_but_never_a_moved_input(fresh):
    """The asymmetry is the point. Bytes that come back identical are the
    recipe's output whatever touched the file afterwards — but no re-run can
    make an input be the data it was."""
    target = fresh / "results/stable.txt"
    os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 600))
    assert not grading.grade(fresh, target).quotable

    g = grading.grade(fresh, target, reproduced=True)
    assert (g.grade, g.quotable) == ("VERIFIED", True)
    assert "supersedes" in g.why

    (fresh / "in.txt").write_text("moved\n")
    g = grading.grade(fresh, target, reproduced=True)
    assert (g.grade, g.quotable) == ("REPRODUCED", False)


def test_a_file_no_rule_produces_cannot_take_the_re_run_down_with_it(fresh):
    """Measured on this package's own `example/`: two declared CSVs that no rule
    makes were handed to Snakemake as targets, and the whole run died on
    `MissingRuleException` before reproducing anything. A checker that reports
    nothing looks the same as one with nothing to report."""
    (fresh / "results/handmade.csv").write_text("a\n1\n")
    done = grading.reproduce(fresh)
    assert "results/handmade.csv" not in done
    assert done["results/stable.txt"] is True


# ---------------------------------------------------------------------------
# The command line, whose exit status is the interface
# ---------------------------------------------------------------------------

def test_check_exits_nonzero_only_when_something_is_refused(fresh, capsys):
    """PASS side and FAIL side of the gate itself. A `--check` that always
    exited 0 would be a green light nobody had earned."""
    ok = grading.main([str(fresh), "results/stable.txt", "--check"])
    capsys.readouterr()
    assert ok == 0

    (fresh / "in.txt").write_text("changed\n")
    bad = grading.main([str(fresh), "results/stable.txt", "--check"])
    out = capsys.readouterr().out
    assert bad == 1
    assert "REFUSED" in out


def test_the_report_is_json_and_carries_the_evidence(fresh, tmp_path):
    """What a rule would write. The fields are Snakemake's own, which is the
    point — a reader can take the `shellcmd` and run it."""
    out = tmp_path / "grades.json"
    assert grading.main([str(fresh), "--json", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["files"] == d["quotable"] == 2
    one = d["grades"]["results/stable.txt"]
    assert one["grade"] == "EXECUTED" and one["rule"] == "stable"
    assert "cat" in one["shellcmd"] and one["endtime"]


def test_grading_a_project_that_never_ran_says_so_rather_than_crashing(tmp_path):
    """No `.snakemake` at all — a fresh clone, or somebody else's directory.
    The answer is NONE for everything, and it must not be a traceback."""
    (tmp_path / "results").mkdir()
    (tmp_path / "results/x.csv").write_text("a\n1\n")
    graded = grading.grade_all(tmp_path)
    assert [g.grade for g in graded.values()] == ["NONE"]
    assert grading.quotable(tmp_path) == {}
