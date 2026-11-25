"""`declared_figure_exists` means the figure the claim DECLARED is on disk, and nothing else.

Two-sided, because the field that this one replaces could only come back false. `relate` runs before
`claimcheck.figures` inside the shared `check_claims` rule (`claimcheck/workflow/claims.smk`), so at the
moment `results/claims.json` is written the default `figures/claims/<id>.png` either does not exist yet or
is the previous pass's file. `relate` therefore answers only the question it can answer honestly, and the
name says which question that is. Whether a figure was DRAWN for a claim is read from
`figures/claims/index.json` and the PNG beside it, which is what the master ledger does.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CONTEXT = """
[project]
name = "declared_figure_fixture"
title = "Does the declared-figure field test the declared figure?"

[[datasets]]
id = "runs"
origin = "computed"
level = "test_case"
files = ["data/rows.csv"]

[[variables]]
dataset = "runs"
name = "a"
unit = "1"

[[variables]]
dataset = "runs"
name = "b"
unit = "1"

[[variables]]
dataset = "runs"
name = "batch"
unit = "1"

[[relationships]]
id = "D1_declares_a_figure_that_is_there"
x = "runs.a"
y = "runs.b"
kind = "correlation"
method = "spearman"
claimed = 1.0
resampling_unit = "batch"
figure = "figures/present.png"
evidence.source = "The declared figure is written before relate runs, so the field must read true."

[[relationships]]
id = "D2_declares_a_figure_that_is_not_there"
x = "runs.a"
y = "runs.b"
kind = "correlation"
method = "spearman"
claimed = 1.0
resampling_unit = "batch"
figure = "figures/missing.png"
evidence.source = "The declared figure is never written, so the field must read false."

[[relationships]]
id = "D3_declares_no_figure_at_all"
x = "runs.a"
y = "runs.b"
kind = "correlation"
method = "spearman"
claimed = 1.0
resampling_unit = "batch"
evidence.source = "Nothing is declared, so the declared figure cannot be on disk and the field reads false."
"""

ROWS = "\n".join(["a,b,batch", *(f"{i},{2 * i},batch{i % 4}" for i in range(24))]) + "\n"


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "declared_figure_fixture"
    (proj / "data").mkdir(parents=True)
    (proj / "figures").mkdir()
    (proj / "context.toml").write_text(CONTEXT.lstrip())
    (proj / "data" / "rows.csv").write_text(ROWS)
    # the declared figure of D1 only. D2 declares one and it is never written; that is the other side.
    (proj / "figures" / "present.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return proj


def _claims(proj: Path) -> dict:
    subprocess.run([sys.executable, "-m", "claimcheck.relate", ".", "results/claims.json"],
                   cwd=proj, env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
                   check=True, capture_output=True)
    return json.loads((proj / "results" / "claims.json").read_text())


def test_a_declared_figure_on_disk_reads_present_and_one_that_is_absent_reads_absent(tmp_path):
    rows = {r["id"]: r for r in _claims(_project(tmp_path))["relationships"]}
    assert rows["D1_declares_a_figure_that_is_there"]["declared_figure_exists"] is True
    assert rows["D2_declares_a_figure_that_is_not_there"]["declared_figure_exists"] is False
    assert rows["D3_declares_no_figure_at_all"]["declared_figure_exists"] is False


def test_the_summary_lists_the_claims_whose_declared_figure_is_missing(tmp_path):
    doc = _claims(_project(tmp_path))
    assert doc["claims_without_a_declared_figure"] == [
        "D2_declares_a_figure_that_is_not_there", "D3_declares_no_figure_at_all"]


def test_no_row_and_no_summary_claims_more_than_the_declared_sense(tmp_path):
    """The overloaded names are gone. `figure_exists` and `claims_without_a_figure` read as "a figure
    exists for this claim", which is not what either tested: on 2026-11-18 the master ledger reported
    21,049 of 25,760 claims as having no figure while 20,128 claim PNGs were on disk."""
    doc = _claims(_project(tmp_path))
    assert "claims_without_a_figure" not in doc
    for row in doc["relationships"]:
        assert "figure_exists" not in row
