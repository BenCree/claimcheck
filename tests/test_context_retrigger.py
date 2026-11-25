"""Editing a claimed value re-adjudicates the claim. Editing a sentence of prose does not.

BOTH HALVES OR NEITHER. The first half alone passes on a rule that re-runs
whenever `context.toml` is touched, which is the fix nobody would keep: this lab
adds an `evidence.note` constantly, and re-adjudicating
`cry1_screen_pose_and_apo` for a sentence costs 49 minutes of one core. So each
case here is run three ways and the two wrong rules are built and shown to fail:

| rule                                  | prose edit         | claimed edit       |
|---------------------------------------|--------------------|--------------------|
| `claims.smk` as shipped               | no re-adjudication | re-adjudication    |
| the `params` digest pinned (the bug)  | no re-adjudication | NONE, silently     |
| `ancient()` dropped, no digest        | re-adjudication    | re-adjudication    |

The defect this guards: `check_claims` takes `context.toml` as `ancient()`, so
Snakemake ignores its modification time, and before 2026-11-22 nothing else
read it either. An edited `claimed` value left `results/claims.json` in place
with a verdict computed from the previous declaration and a `context_sha256`
that no longer matched the file beside it.

The assertions are on the planned job list, the bytes of the verdict file and
`context_sha256` -- never on an exit status, which is what hid this in the first
place: `snakemake` exits 0 when it has decided there is nothing to do.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _snakemake import snakemake_cmd, why_not

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from claimcheck.workflow import (  # noqa: E402
    NON_VERDICT_SECTIONS, PROSE_KEYS, RULES, verdict_digest)

# ── the fixture ──────────────────────────────────────────────────────────────
# b = 2a exactly, so Spearman's rho is 1.0 and the verdict of a claim at 1.0 is
# `agrees` with no room for argument. Moving `claimed` to -1.0 must make it
# `refuted`: the point is that the verdict MOVES WHERE THE NEW VALUE DEMANDS,
# not merely that the file was rewritten.

CONTEXT = """
[project]
name = "context_retrigger_fixture"
title = "Does editing a claimed value re-adjudicate the claim?"

[[datasets]]
id = "runs"
origin = "computed"
level = "test_case"
files = ["data/rows.csv"]
title = "Twenty-four rows in six batches"

[[variables]]
dataset = "runs"
name = "a"
unit = "1"
description = "The monotone column."

[[variables]]
dataset = "runs"
name = "b"
unit = "1"
description = "Twice a, so the rank correlation is exactly 1."

[[variables]]
dataset = "runs"
name = "batch"
unit = "1"

[[relationships]]
id = "R1_b_tracks_a"
x = "runs.a"
y = "runs.b"
kind = "correlation"
method = "spearman"
claimed = 1.0
resampling_unit = "batch"
[relationships.evidence]
source = "b is 2a, so the ranks agree exactly and the claim must hold."
note = "FIRST NOTE."
"""

ROWS = "\n".join(["a,b,batch",
                  *(f"{i},{2 * i},batch{i % 6}" for i in range(24))]) + "\n"

SNAKEFILE = """
import sys
sys.path.insert(0, {repo!r})

include: {rules!r}


rule all:
    input:
        "results/claims.json",
"""


def _project(tmp_path: Path, rules: str) -> Path:
    proj = tmp_path / "fixture"
    (proj / "data").mkdir(parents=True)
    (proj / "context.toml").write_text(CONTEXT.lstrip())
    (proj / "data" / "rows.csv").write_text(ROWS)
    (proj / "Snakefile").write_text(
        SNAKEFILE.format(repo=str(REPO), rules=rules).lstrip())
    return proj


def _snakemake(proj: Path, *extra) -> subprocess.CompletedProcess:
    """Always with CLAIMCHECK_PY pinned: `interpreter()` otherwise probes every
    sibling environment on this machine for rdflib, which is a minute of
    subprocesses the test has no use for."""
    return subprocess.run(
        [*snakemake_cmd(), "-c1", "results/claims.json", *extra],
        capture_output=True, text=True, cwd=str(proj),
        env={**os.environ, "PYTHONPATH": str(REPO),
             "CLAIMCHECK_PY": sys.executable})


def _planned(proj: Path) -> str:
    """The job list of a dry run, which is where a missing re-run is visible."""
    r = _snakemake(proj, "-n")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    return r.stdout


def _edit(proj: Path, old: str, new: str) -> None:
    text = (proj / "context.toml").read_text()
    assert old in text, old
    (proj / "context.toml").write_text(text.replace(old, new))


def _verdict(proj: Path) -> tuple[str, str, bytes]:
    doc = json.loads((proj / "results" / "claims.json").read_text())
    row = next(r for r in doc["relationships"] if r["id"] == "R1_b_tracks_a")
    return (row["verdict"], doc["context_sha256"],
            (proj / "results" / "claims.json").read_bytes())


# ── the three rules ──────────────────────────────────────────────────────────
#: The shipped file, and the two mutants that each get one half wrong. Built by
#: editing the real `claims.smk`, so a mutant cannot drift away from it.
PINNED_PARAM = 'verdict_fields=_cw.verdict_digest(_PROJECT),'
ANCIENT_CONTEXT = 'context=ancient(_PROJECT / "context.toml"),'


def _mutant(tmp_path: Path, name: str) -> str:
    text = Path(RULES).read_text()
    assert PINNED_PARAM in text, "the params digest is gone from claims.smk"
    text = text.replace(PINNED_PARAM, 'verdict_fields="pinned",')
    if name == "mtime_only":
        # The first occurrence is check_claims'. The emitters keep theirs.
        assert ANCIENT_CONTEXT in text
        text = text.replace(ANCIENT_CONTEXT, 'context=_PROJECT / "context.toml",', 1)
    out = tmp_path / f"claims_{name}.smk"
    out.write_text(text)
    return str(out)


# ── half one: a prose edit must not re-adjudicate ────────────────────────────

def test_editing_an_evidence_note_does_not_re_adjudicate(tmp_path):
    """THE HALF A HEAVY-HANDED FIX FAILS. The verdict file is not rewritten, so
    its bytes are identical and no emitter downstream of it goes stale."""
    if snakemake_cmd() is None:
        pytest.skip(why_not())
    proj = _project(tmp_path, RULES)
    assert _snakemake(proj).returncode == 0
    before = _verdict(proj)

    _edit(proj, "FIRST NOTE.", "A SECOND NOTE, three words longer than the first.")
    assert "check_claims" not in _planned(proj)
    _snakemake(proj)
    assert _verdict(proj) == before


def test_dropping_ancient_instead_would_re_adjudicate_on_that_note(tmp_path):
    """The rejected alternative, built and measured rather than argued about."""
    if snakemake_cmd() is None:
        pytest.skip(why_not())
    proj = _project(tmp_path, _mutant(tmp_path, "mtime_only"))
    assert _snakemake(proj).returncode == 0
    _edit(proj, "FIRST NOTE.", "A SECOND NOTE, three words longer than the first.")
    assert "check_claims" in _planned(proj)


# ── half two: a verdict field must re-adjudicate ─────────────────────────────

def test_editing_a_claimed_value_re_adjudicates_and_the_verdict_moves(tmp_path):
    """A claim at +1.0 on b = 2a agrees; the same claim at -1.0 is refuted."""
    if snakemake_cmd() is None:
        pytest.skip(why_not())
    proj = _project(tmp_path, RULES)
    assert _snakemake(proj).returncode == 0
    was_verdict, was_digest, was_bytes = _verdict(proj)
    assert was_verdict == "agrees", was_verdict

    _edit(proj, "claimed = 1.0", "claimed = -1.0")
    assert "check_claims" in _planned(proj)
    assert _snakemake(proj).returncode == 0
    now_verdict, now_digest, now_bytes = _verdict(proj)
    assert now_verdict == "refuted", now_verdict
    assert now_digest != was_digest
    assert now_bytes != was_bytes


def test_without_the_digest_an_edited_claim_is_adjudicated_against_the_old_value(tmp_path):
    """THE DEFECT, reproduced. The rule is the shipped one with the digest
    pinned to a constant, which is what `check_claims` was until 2026-11-22: the
    claim is moved to -1.0, nothing re-runs, and the verdict on disk still reads
    `agrees` while `context_sha256` names a file that no longer exists."""
    if snakemake_cmd() is None:
        pytest.skip(why_not())
    proj = _project(tmp_path, _mutant(tmp_path, "no_digest"))
    assert _snakemake(proj).returncode == 0
    was = _verdict(proj)

    _edit(proj, "claimed = 1.0", "claimed = -1.0")
    assert "check_claims" not in _planned(proj)
    assert _snakemake(proj).returncode == 0          # exits 0 and does nothing
    assert _verdict(proj) == was
    assert was[0] == "agrees"                        # against a claim of -1.0


# ── the digest itself, field by field ────────────────────────────────────────

def _digest_of(tmp_path: Path, text: str) -> str:
    d = tmp_path / f"ctx{abs(hash(text))}"
    d.mkdir(exist_ok=True)      # the same text digested twice is the same directory
    (d / "context.toml").write_text(text)
    return verdict_digest(d)


@pytest.mark.parametrize("old,new", [
    ("claimed = 1.0", "claimed = 0.9"),
    ("claimed = 1.0", 'claimed_range = [0.9, 1.0]'),
    ("method = \"spearman\"", "method = \"kendall\""),
    ("kind = \"correlation\"", "kind = \"value\""),
    ("resampling_unit = \"batch\"", "resampling_unit = \"runs\""),
    ('x = "runs.a"', 'x = "runs.b"'),
    ('y = "runs.b"', 'y = "runs.a"'),
    # a key no estimator had when this was written: the denylist must digest it
    ("claimed = 1.0", "claimed = 1.0\nbaseline = \"runs.b\""),
    ("claimed = 1.0", "claimed = 1.0\ntie_tol = 1e-9"),
    ("claimed = 1.0", "claimed = 1.0\nalpha = 20.0"),
    ("claimed = 1.0", "claimed = 1.0\nsome_key_no_estimator_has_yet = 3"),
    # the dataset and variable tables a verdict also reads
    ('level = "test_case"', 'level = "reference"'),
    ('files = ["data/rows.csv"]', 'files = ["data/other.csv"]'),
    ('origin = "computed"', 'origin = "manual"'),
    ("name = \"a\"\nunit = \"1\"", "name = \"a\"\nunit = \"kcal/mol\""),
    ('name = "context_retrigger_fixture"', 'name = "renamed"'),
    # a whole claim added, and the project-wide figure style
    ("[[relationships]]", "[[relationships]]\nid = \"R0\"\nx = \"runs.a\"\n"
     "y = \"runs.b\"\nkind = \"correlation\"\nmethod = \"pearson\"\n"
     "claimed = 1.0\n\n[[relationships]]"),
    ('title = "Does editing a claimed value re-adjudicate the claim?"',
     'figstyle = "styles/lab.mplstyle"'),
])
def test_the_digest_moves_when_a_verdict_field_moves(tmp_path, old, new):
    assert _digest_of(tmp_path, CONTEXT) != _digest_of(
        tmp_path, CONTEXT.replace(old, new, 1)), old


@pytest.mark.parametrize("old,new", [
    ("FIRST NOTE.", "an entirely different note, written later"),
    ('title = "Twenty-four rows in six batches"', 'title = "Renamed dataset"'),
    ('description = "The monotone column."', 'description = "Rewritten."'),
    ('title = "Does editing a claimed value re-adjudicate the claim?"',
     'title = "Retitled."'),
    # a comment, a blank line and a reordered table: tomllib never reports them
    ("[[relationships]]", "# a comment nobody computes from\n\n[[relationships]]"),
    ("kind = \"correlation\"\nmethod = \"spearman\"",
     "method = \"spearman\"\nkind = \"correlation\""),
])
def test_the_digest_holds_still_for_prose(tmp_path, old, new):
    assert _digest_of(tmp_path, CONTEXT) == _digest_of(
        tmp_path, CONTEXT.replace(old, new, 1)), old


def test_the_digest_carries_its_own_name_and_the_packages_hash_width():
    d = verdict_digest(Path(RULES).parents[2] / "example")
    name, _, hexdigits = d.partition(":")
    assert name == "context-verdict-fields"
    assert len(hexdigits) == 16 and int(hexdigits, 16) >= 0


def test_a_date_in_the_declaration_does_not_raise(tmp_path):
    """`retrieved = 2026-08-15` is a TOML date, and `json.dumps` refuses one
    without `default=str`. The example's manual dataset carries one."""
    text = CONTEXT.replace('origin = "computed"',
                           'origin = "manual"\nretrieved = 2026-08-15')
    assert _digest_of(tmp_path, text)


def test_belief_and_jobs_are_the_only_tables_outside_the_digest(tmp_path):
    """They are the inputs of their own rules, which consume
    `results/claims.json` and so are recomputed whenever a verdict is. Digesting
    them here would re-adjudicate a 49-minute directory for a node label."""
    assert set(NON_VERDICT_SECTIONS) == {"belief", "jobs"}
    for text in (CONTEXT + '\n[belief.rests_on]\nruns = "a_node"\n',
                 CONTEXT + '\n[jobs]\nrecords = "jobs.jsonl"\n'):
        assert _digest_of(tmp_path, CONTEXT) == _digest_of(tmp_path, text)


def test_a_table_this_package_does_not_read_is_digested_anyway(tmp_path):
    """THE DENYLIST, AT TABLE LEVEL. Naming the four tables `relate` reads today
    would be an allowlist, and a table added later would go undigested, which is
    the defect being fixed. A `[units]` table already exists in two of the
    tree's declarations and nothing in this package reads it."""
    text = CONTEXT + '\n[units]\na = { unit = "kcal/mol" }\n'
    assert _digest_of(tmp_path, CONTEXT) != _digest_of(tmp_path, text)


def test_evidence_source_is_not_treated_as_prose():
    """`claimcheck.scrutiny` matches a control label at the start of it, and
    that warning travels in `results/claims.json`."""
    assert "source" not in PROSE_KEYS
    assert "note" in PROSE_KEYS


# ── the sweep that measures the damage already done ──────────────────────────

def test_the_sweep_tells_a_prose_edit_from_a_changed_claimed_value(tmp_path):
    """Two-sided, because a sweep that calls every mismatch stale would make the
    count meaningless: `context_sha256` differs after a note edit BY DESIGN."""
    if snakemake_cmd() is None:
        pytest.skip(why_not())
    from claimcheck.workflow.context_drift import inspect

    proj = _project(tmp_path, RULES)
    assert _snakemake(proj).returncode == 0
    assert inspect(proj)["verdict"] == "current"

    _edit(proj, "FIRST NOTE.", "a different note")
    assert inspect(proj)["verdict"] == "prose_only"

    _edit(proj, "claimed = 1.0", "claimed = -1.0")
    rec = inspect(proj)
    assert rec["verdict"] == "verdict_fields"
    assert "claimed" in rec["detail"] and "-1.0" in rec["detail"]


def test_a_changed_claim_is_reported_even_when_other_claims_were_added(tmp_path):
    """A SECOND CLAIM ADDED MUST NOT HIDE THE FIRST ONE MOVING. The sweep used
    to return at the first key-set difference, which would have reported
    `active_latent_real` (55 claims declared, 24 adjudicated) as merely
    `claims_changed` and never looked at whether those 24 verdicts still
    describe their claims."""
    if snakemake_cmd() is None:
        pytest.skip(why_not())
    from claimcheck.workflow.context_drift import inspect

    proj = _project(tmp_path, RULES)
    assert _snakemake(proj).returncode == 0

    _edit(proj, "[[relationships]]", "[[relationships]]\nid = \"R2_added_later\"\n"
          "x = \"runs.a\"\ny = \"runs.b\"\nkind = \"correlation\"\n"
          "method = \"pearson\"\nclaimed = 1.0\n\n[[relationships]]")
    rec = inspect(proj)
    assert rec["verdict"] == "claims_changed"
    assert rec["n_claims_declared"] == 2 and rec["n_verdicts"] == 1

    _edit(proj, "claimed = 1.0\nresampling_unit", "claimed = -1.0\nresampling_unit")
    rec = inspect(proj)
    assert rec["verdict"] == "verdict_fields", rec["detail"]
    assert "R1_b_tracks_a" in rec["detail"] and "declared with no verdict" in rec["detail"]


def test_the_sweep_reports_a_missing_or_broken_file_rather_than_failing(tmp_path):
    """Fail soft per item, loud in the row: a directory that cannot be read is a
    row with a reason, so one bad directory never ends the sweep."""
    from claimcheck.workflow.context_drift import inspect

    bare = tmp_path / "never_adjudicated"
    (bare / "data").mkdir(parents=True)
    (bare / "context.toml").write_text(CONTEXT.lstrip())
    assert inspect(bare)["verdict"] == "no_claims"

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "context.toml").write_text("this is not toml = = =\n")
    rec = inspect(broken)
    assert rec["verdict"] == "unreadable" and "context.toml" in rec["detail"]

    nodigest = tmp_path / "no_digest"
    (nodigest / "results").mkdir(parents=True)
    (nodigest / "context.toml").write_text(CONTEXT.lstrip())
    (nodigest / "results" / "claims.json").write_text('{"relationships": []}')
    assert inspect(nodigest)["verdict"] == "no_digest"


# ── the same defect one layer up: what a rule declared against what it read ──
#
# A stale `ancient()` edge and an incomplete input list are one shape: Snakemake
# can only notice a change to a file the rule declared. Rule `master_ledger` in
# /home/ben/code/shared/ledger/Snakefile declared only its own build script
# while the script walked four trees, and the ledger was found four days stale
# on 2026-11-22 with 29 experiments carrying a newer results/claims.json. The
# shape is reproduced here in a fixture rather than asserted against that live
# tree, whose state is somebody else's to change.

AGGREGATOR = """
rule aggregate:
    input:
        script="build.py",
    output:
        "summary.tsv",
    shell:
        "cat a.tsv b.tsv > summary.tsv"
"""


def test_a_rule_that_reads_what_it_never_declared_is_named(tmp_path):
    """THE LEDGER SHAPE. The rule declares its script, reads two tables, and
    Snakemake's record lists only the script, so neither table can ever make the
    summary stale and the rule reports nothing to be done for ever."""
    if snakemake_cmd() is None:
        pytest.skip(why_not())
    from claimcheck.workflow.context_drift import (recorded_inputs,
                                                   undeclared_sources)

    proj = tmp_path / "aggregator"
    proj.mkdir()
    (proj / "Snakefile").write_text(AGGREGATOR.lstrip())
    for name in ("build.py", "a.tsv", "b.tsv"):
        (proj / name).write_text(f"{name}\n")
    r = subprocess.run([*snakemake_cmd(), "-c1", "summary.tsv"],
                       capture_output=True, text=True, cwd=str(proj),
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]

    assert recorded_inputs(proj, "summary.tsv") == {
        str((proj / "build.py").resolve())}

    rec = undeclared_sources(proj, "summary.tsv", ["build.py", "a.tsv", "b.tsv"])
    assert rec["verdict"] == "inputs_incomplete"
    assert [Path(p).name for p in rec["undeclared"]] == ["a.tsv", "b.tsv"]

    # THE OTHER SIDE, and the one a check like this usually gets wrong: naming
    # only what the rule did declare must come back clean, or every rule on the
    # machine reads as defective and the count stops being read.
    clean = undeclared_sources(proj, "summary.tsv", ["build.py"])
    assert clean["verdict"] == "inputs_cover_sources" and clean["undeclared"] == []


def test_an_output_no_rule_produced_is_no_record_rather_than_a_defect(tmp_path):
    """A file with no record is a queue item, not a lie: nothing claims to have
    produced it, so there is no input list to be incomplete."""
    from claimcheck.workflow.context_drift import (recorded_inputs,
                                                   undeclared_sources)

    d = tmp_path / "unrecorded"
    d.mkdir()
    (d / "summary.tsv").write_text("by hand\n")
    assert recorded_inputs(d, "summary.tsv") is None
    assert undeclared_sources(d, "summary.tsv", ["a.tsv"])["verdict"] == "no_record"


def test_a_verdict_file_from_the_shared_rules_declares_its_whole_declaration(tmp_path):
    """The shared `check_claims` derives its input list from `context.toml` with
    `declared_files`, so this can only fail if that derivation breaks."""
    if snakemake_cmd() is None:
        pytest.skip(why_not())
    from claimcheck.workflow.context_drift import inspect

    proj = _project(tmp_path, RULES)
    assert _snakemake(proj).returncode == 0
    rec = inspect(proj)
    assert rec["inputs"] == "inputs_cover_sources", rec["inputs_undeclared"]
    assert rec["inputs_undeclared"] == []


def test_the_sweep_walks_a_tree_and_counts_once_per_directory(tmp_path):
    from claimcheck.workflow.context_drift import sweep

    for name in ("one", "two"):
        d = tmp_path / "tree" / name
        d.mkdir(parents=True)
        (d / "context.toml").write_text(CONTEXT.lstrip())
    recs = sweep([tmp_path / "tree"])
    assert len(recs) == 2 and {r["verdict"] for r in recs} == {"no_claims"}
    assert len(sweep([tmp_path / "tree", tmp_path / "tree" / "one"])) == 2
