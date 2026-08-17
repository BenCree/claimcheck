"""How the tests find Snakemake, and why they stopped believing `which`.

Every end-to-end test here needs to RUN the workflow, so each guarded itself
with

    if shutil.which("snakemake") is None:
        pytest.skip("snakemake is not installed")

and on 2026-08-17 that sentence was false while 20 of 69 tests skipped on it --
the whole of `test_end_to_end.py` and the whole of `test_graph.py`, which is to
say every test that exercises the package end to end rather than a function at a
time. Snakemake was installed. `python -m pytest` invoked through an absolute
interpreter path does not put that environment's `bin/` on `PATH`, so `which`
found nothing and the suite reported 49 passed and looked healthy.

A skip that states a false reason is worse than a failure, because a failure is
read. This asks the question the tests actually mean -- *can I run Snakemake?* --
and answers it the way that is true in both cases: prefer the executable, fall
back to `python -m snakemake` in the interpreter running the tests, and skip only
when neither exists.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys


def snakemake_cmd() -> list | None:
    """The argv prefix that runs Snakemake here, or None if it truly is absent."""
    exe = shutil.which("snakemake")
    if exe:
        return [exe]
    if importlib.util.find_spec("snakemake") is not None:
        return [sys.executable, "-m", "snakemake"]
    return None


def why_not() -> str:
    """The skip reason, phrased so it cannot be wrong."""
    return ("snakemake is neither on PATH nor importable by "
            f"{sys.executable} -- this is the only case in which these tests "
            f"may be skipped")
