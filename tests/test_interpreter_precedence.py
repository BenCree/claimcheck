"""Which interpreter the included rules run, and in what order of precedence.

The rule file `claimcheck/workflow/claims.smk` chooses an interpreter for every
claimcheck job. Three things can name one, and the order matters:

1. `CLAIMCHECK_PY` in the environment, set deliberately for this run
2. a `PY` global defined by the including Snakefile, inherited by accident
3. `claimcheck.workflow.interpreter()`, which probes for the emit extras

The order was 2, 3 with no 1, which contradicted the comment in claims.smk and
the docstring of `interpreter`. Measured 2026-10-21 on
/home/ben/code/projects/affinity-scoring/experiments/hit_recovery_unseen_chemistry:
the project's own rules need a Python 3.10 environment, every claimcheck module
imports `tomllib` which is 3.11+, and `check_claims` therefore failed with
ModuleNotFoundError on a project whose data were complete. The only escape was
editing PY, which changes every rule's shell command and makes Snakemake treat
finished outputs as stale.

These tests read the precedence expression out of the .smk source rather than
executing it, because an .smk file is not importable Python: it is parsed by
Snakemake. A test that needs Snakemake to run is slower and tests Snakemake.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SMK = Path(__file__).resolve().parents[1] / "claimcheck" / "workflow" / "claims.smk"


@pytest.fixture(scope="module")
def source() -> str:
    return SMK.read_text()


@pytest.fixture(scope="module")
def assignment(source: str) -> str:
    """The single line assigning _PY, with no comment lines around it."""
    matches = [line.strip() for line in source.splitlines()
               if re.match(r"^_PY\s*=", line.strip())]
    assert len(matches) == 1, f"expected exactly one _PY assignment, got {matches}"
    return matches[0]


def test_claimcheck_py_is_consulted_at_all(assignment: str) -> None:
    """The documented override must appear in the expression that chooses."""
    assert "CLAIMCHECK_PY" in assignment


def test_claimcheck_py_beats_the_inherited_py_global(assignment: str) -> None:
    """CLAIMCHECK_PY must come before the PY global in the `or` chain.

    This is the whole point: a project cannot unset PY without rewriting every
    rule, so the environment variable has to win.
    """
    position_env = assignment.index("CLAIMCHECK_PY")
    position_global = assignment.index('globals().get("PY")')
    assert position_env < position_global, assignment


def test_the_py_global_is_still_honoured(assignment: str) -> None:
    """Removing the PY global would break every project that sets one.

    The fix narrows precedence; it does not remove a documented route.
    """
    assert 'globals().get("PY")' in assignment


def test_the_probe_remains_the_last_resort(assignment: str) -> None:
    """With neither override set, the package still probes for the emit extras."""
    assert "interpreter()" in assignment
    assert assignment.index("interpreter()") > assignment.index('globals().get("PY")')


def test_os_is_imported_under_the_underscore_convention(source: str) -> None:
    """An included .smk shares the project's global namespace.

    Every name claims.smk defines is visible to the including Snakefile, so the
    file imports under underscore-prefixed aliases to avoid colliding with a
    project's own `os` or `sys`. A bare `import os` here would be a name the
    project did not ask for.
    """
    assert "import os as _os" in source
    assert not re.search(r"^import os$", source, flags=re.MULTILINE)


def test_the_reason_is_recorded_beside_the_change(source: str) -> None:
    """The incident that makes this a rule rather than a preference.

    RULES.md: "Where a rule has a date or a measurement behind it, the incident
    is kept in one line, because that is what makes it a rule and not a
    preference." A later reader reversing the order should have to read why.
    """
    assert "tomllib" in source
    assert "2026-10-21" in source
