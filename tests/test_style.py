"""Lint and types, run by `pytest` so that one command is the whole gate.

The predecessor kept these in a pre-commit hook. A hook is opt-in: it does not
run for `--no-verify`, for a merge made in a web UI, or for anybody who has not
read the README — and its recorded incident is a must-level check that failed
for three commits while every summary looked clean.

Zero findings is the bar, not a baseline. The package is four modules; if that
stops being achievable the answer is the predecessor's growth ratchet, not a
quietly raised threshold.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run(tool: str, *args: str) -> subprocess.CompletedProcess:
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not installed")
    return subprocess.run([tool, *args], capture_output=True, text=True,
                          cwd=str(REPO))


def test_ruff_is_clean():
    r = _run("ruff", "check", ".", "--output-format", "concise")
    assert r.returncode == 0, r.stdout + r.stderr


def test_mypy_is_clean():
    """Config lives in pyproject, so a bare `mypy` gives the same answer this
    does. Two ways of running it that disagree is how the predecessor shipped a
    failing check three times."""
    r = _run("mypy", "--no-color-output")
    assert r.returncode == 0, r.stdout[-2000:]


def test_the_gate_is_one_command():
    """`pytest` must reach lint and types, or somebody will run the tests, see
    green, and ship a lint failure."""
    text = (REPO / "pyproject.toml").read_text()
    assert "[tool.ruff]" in text and "[tool.mypy]" in text
    assert (REPO / "tests" / "test_style.py").is_file()


def test_the_style_gate_can_fail(tmp_path):
    """Two-sided. A linter pointed at nothing passes."""
    bad = tmp_path / "bad.py"
    bad.write_text("import os\n")            # F401
    r = _run("ruff", "check", str(bad), "--select", "F", "--output-format",
             "concise")
    assert r.returncode != 0 and "F401" in r.stdout
