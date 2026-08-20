"""Snakemake's own per-rule suite, run as part of ours.

`snakemake --generate-unit-tests` writes one test per rule, each running that
rule alone in a temporary directory against a copied slice of the inputs. We did
not write those tests and we do not maintain them — which is the point. The rule
in this repository is *if Snakemake already does it, do not build it again*, and
per-rule isolation testing is squarely something Snakemake already does.

What this file adds is only the two things it cannot do for itself:

* **run them**, so they are part of one command rather than a thing somebody
  remembers, and
* **notice when the generated suite has gone out of date** — a rule added to the
  Snakefile with no generated test is a rule nothing isolates, and the generated
  directory gives no sign of it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from _snakemake import snakemake_cmd, why_not

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"
GENERATED = EXAMPLE / ".tests" / "unit"


def _rules_in_snakefile() -> set[str]:
    text = (EXAMPLE / "Snakefile").read_text()
    # `all` is excluded by Snakemake itself: it executes nothing, so there is
    # nothing to isolate. Its message says so when generating.
    return {m for m in re.findall(r"^rule (\w+):", text, re.M) if m != "all"}


def test_a_generated_test_exists_for_every_rule():
    """The staleness this cannot otherwise detect. Add a rule, forget to
    regenerate, and the suite still passes — over one rule fewer."""
    if not GENERATED.is_dir():
        pytest.skip("run `snakemake --generate-unit-tests` in example/")
    have = {p.stem[len("test_"):] for p in GENERATED.glob("test_*.py")}
    missing = _rules_in_snakefile() - have
    assert not missing, (
        f"rules with no generated test: {sorted(missing)}. Regenerate with "
        f"`cd example && snakemake --generate-unit-tests`, then re-pin the "
        f"interpreter — see this repository's README.")


def test_snakemakes_generated_rule_tests_pass():
    """Each rule, run alone in a temp directory on a copied input slice."""
    if not GENERATED.is_dir():
        pytest.skip("run `snakemake --generate-unit-tests` in example/")
    if snakemake_cmd() is None:
        pytest.skip(why_not())
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings",
         str(GENERATED)],
        capture_output=True, text=True, cwd=str(EXAMPLE),
        env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]


def test_the_generated_tests_use_this_interpreter():
    """As generated they invoke a bare `python`, which on this machine is a
    different environment without the package installed. The regeneration step
    is not complete until they are re-pinned, so that is checked rather than
    left in a README nobody re-reads."""
    if not GENERATED.is_dir():
        pytest.skip("run `snakemake --generate-unit-tests` in example/")
    for p in GENERATED.glob("test_*.py"):
        src = p.read_text()
        assert '"python",' not in src, (
            f"{p.name} still calls a bare `python`; re-pin it to "
            f"`sys.executable` after regenerating")
