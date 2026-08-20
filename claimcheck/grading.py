"""May this number be quoted? — the grade ladder, read out of Snakemake's record.

    python -m claimcheck.grading <project-dir>              # grade everything
    python -m claimcheck.grading <project-dir> --check      # exit 1 if any is not
    python -m claimcheck.grading <project-dir> --reproduce  # re-run in a COPY first

`claimcheck.relate` adjudicates a claim; it says nothing about the file the
claim was computed from. "r = 0.5, 95% CI [0.2, 0.7]" is worth nothing if the
CSV underneath was written by a script nobody can find. This answers the other
half of the question — **what is known about how this file came to exist?** —
and it is deliberately a different question from whether the claim holds.

## The ladder

    NONE        on disk, and no rule claims it
    DECLARED    a rule claims it, and no completed run left a file here
    EXECUTED    a completed run of a rule produced it
    REPRODUCED  re-run in a copy of the project, byte-identical
    VERIFIED    reproduced, and every recorded input digest still matches

**`EXECUTED` is not `REPRODUCED`.** A workflow engine reporting "nothing to be
done" has compared timestamps; it has not opened the file. Only `reproduce()`
below, which re-runs the recipe and compares digests, awards the top two rungs.

## The ladder says what ran. `stale` says whether the record still fits.

They are separate axes on purpose, and `Grade.quotable` needs both: at least
`EXECUTED`, **and** a record that still describes the file on disk. A record
stops fitting for two reasons, each stated in the verdict:

* a recorded input digest no longer matches the file of that name — the bytes
  here were not computed from the data now present. Snakemake's own summary
  calls this "updated input files" rather than "ok";
* the file is newer than the run that recorded it, so something wrote or
  touched it afterwards.

Collapsing either into `DECLARED` was the first design and it was wrong: it
reads as "the rule never ran", which is a different and much worse thing than
"it ran and then somebody touched the output". Measured on the three real
projects in `al_mace/datasets/claimcheck/`, where every one of the twelve files
under `results/` carries an mtime 1.1–1.3 s later than its own recorded
`endtime` while the `data/` files match to the microsecond — one post-run
operation over each `results/` directory, invisible to everything else.

A refusal on mtime is the weakest of the three, and `--reproduce` is what
settles it: bytes that come back identical from a re-run are the recipe's
output whatever touched the file afterwards, so a successful reproduction
clears the mtime doubt. It does not clear a moved input, because nothing but
the missing data could.

## Where the evidence comes from, and what is NOT built here

Snakemake already writes the run record. `.snakemake/metadata/<urlsafe-base64 of
the output path>` is JSON holding `rule`, `input`, `input_checksums` (sha256 per
input, as read at run time), `code`, `shellcmd`, `conda_env`,
`software_stack_hash`, `starttime` and `endtime`; `.snakemake/incomplete/<same
key>` marks a job that started and never finished. This module READS that store
and writes nothing into it.

That is the whole design. `tests/test_dependencies.py` mechanically refuses any
function here that hashes an input and writes a sidecar, because maintaining a
second copy of this record was the single largest piece of duplication in the
package this one replaces.

**Snakemake records no digest for its own outputs** — the output path is the
record's filename rather than a field. So "were these bytes edited after the run
that produced them?" is answered from the file's mtime against the recorded
`endtime` (which Snakemake sets *from* that mtime at completion). That is weaker
evidence than a digest and the verdict says so: it catches an edit, and it does
not catch an edit that preserved the mtime.

## What was kept from the two predecessors, and what was refused

Two retired packages did this job before this one. Both are archived and neither
is named here as a path, because a pointer into a zip file is not a reference —
what is worth having is the argument, so the argument is reproduced.

**Kept:** the ladder, the `Grade` record, and the shape of `grade()` — facts in,
grade out, so the grader cannot go and find evidence that agrees with itself.

**Refused: both of their evidence stores.** One wrote a `<output>.receipt.json`
sidecar beside every artifact and a `provenance/reproduction.json` remembering
old reproduction verdicts; the other kept a repository-wide `ledger.json`. All
three are parallel records of what Snakemake already holds, and the middle one
had to grow a defect fix for describing bytes that had changed since its verdict
was written. Here a reproduction verdict is computed inside the call that
reports it and never persisted, so it cannot go stale.

**Kept, and it is why `--check` is a flag rather than the default:** a check that
refuses everything stops being consulted. The default mode reports; only
`--check` exits 1. Their refusals also said what to do next ("Write the rule,
then run it") rather than only what was wrong, and that phrasing is kept.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

#: Weakest to strongest. A grade is a statement about *what is known*, not about
#: quality: `DECLARED` does not mean bad, it means nobody has checked.
GRADES: tuple[str, ...] = ("NONE", "DECLARED", "EXECUTED", "REPRODUCED", "VERIFIED")

#: The weakest grade that may be quoted. `DECLARED` is below the line: a rule
#: describing an output is a recipe, not a result.
QUOTABLE_FROM = "EXECUTED"

#: Snakemake's store, relative to the directory the workflow ran in.
STORE = ".snakemake"

#: mtime and the recorded `endtime` are both float seconds from different
#: sources, so they are compared with a second of slack. Anything larger would
#: hide a quick edit; anything smaller reports filesystem rounding as tampering.
MTIME_SLACK = 1.0


@dataclass(frozen=True)
class Grade:
    """One file's grade, and everything needed to argue with it."""

    path: str
    grade: str
    rule: str | None = None
    sha256: str | None = None
    why: str = ""
    #: The run record no longer describes the file on disk — see the module
    #: docstring for the two ways that happens. Never quotable while true.
    stale: bool = False
    #: Which recorded inputs moved, so the message names them rather than only
    #: reporting that something did. Empty when the staleness is the mtime kind.
    changed_inputs: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)

    @property
    def quotable(self) -> bool:
        """At least EXECUTED, and a record that still fits the file."""
        return GRADES.index(self.grade) >= GRADES.index(QUOTABLE_FROM) and not self.stale

    def render(self) -> str:
        head = "QUOTABLE  " if self.quotable else "REFUSED   "
        out = [f"{head}{self.path}",
               f"            grade: {self.grade}{'  (stale)' if self.stale else ''}"]
        if self.rule:
            out.append(f"            rule:  {self.rule}")
        out.append(f"            {self.why}")
        return "\n".join(out)


def digest(path: Path | str) -> str | None:
    """sha256 of a file, or None if it is not one. Never raises on a directory.

    None rather than an exception because callers hash whatever a rule declared,
    and a rule may legitimately declare a directory. An earlier version of this
    raised `IsADirectoryError` from inside a checker, turning a valid workflow
    into a crashing one.
    """
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Reading Snakemake's store. Nothing below writes to it.
# ---------------------------------------------------------------------------

def _record_path(store_dir: Path, key: str) -> Path:
    """Where Snakemake filed the record for output `key`.

    Reimplemented rather than imported because importing Snakemake to read four
    JSON files would make a stdlib-only package depend on a workflow engine.
    The encoding is `snakemake/persistence/file.py::_record_path`: urlsafe
    base64, split into chunks one shorter than the filesystem's maximum name
    length, every chunk but the last prefixed with `@` and used as a directory.
    Short paths — every real one — come out as a single component.
    """
    try:
        max_len = os.pathconf(str(store_dir), "PC_NAME_MAX") if os.name == "posix" else 255
    except (OSError, ValueError, AttributeError):
        max_len = 255
    if not max_len:
        max_len = 255
    b64 = base64.urlsafe_b64encode(key.encode()).decode()
    n = max(max_len - 1, 1)
    parts = [b64[i:i + n] for i in range(0, len(b64), n)]
    parts = ["@" + s for s in parts[:-1]] + [parts[-1]]
    return store_dir.joinpath(*parts)


def _keys(root: Path, path: Path) -> list[str]:
    """The names Snakemake could have filed this output under.

    A rule declaring `results/claims.json` keys on that relative string; a rule
    declaring an absolute path keys on the absolute string. Both are tried
    because both occur — `datasets/claimcheck/darby_rowan32` has one of each.
    """
    out = [str(path.resolve())]
    try:
        out.insert(0, path.resolve().relative_to(root.resolve()).as_posix())
    except ValueError:
        pass
    return out


def run_record(root: Path | str, path: Path | str) -> dict | None:
    """Snakemake's own metadata record for one output, or None if there is none.

    The record is the evidence for every grade above NONE, and it is Snakemake's
    file in Snakemake's format. Reading it is the whole mechanism.
    """
    root, p = Path(root), Path(path)
    p = p if p.is_absolute() else root / p
    store = root / STORE / "metadata"
    if not store.is_dir():
        return None
    for key in _keys(root, p):
        f = _record_path(store, key)
        if f.is_file():
            try:
                rec = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                # Snakemake tolerates a half-written record here and so does
                # this: a corrupt record is no record, which grades NONE.
                continue
            if isinstance(rec, dict) and rec:
                return rec
    return None


def unfinished(root: Path | str, path: Path | str) -> bool:
    """True if a job wrote this output's start marker and never cleared it.

    Snakemake marks every output incomplete when the job starts and unmarks it
    when the job finishes, so a marker still present means the run died in the
    middle. Older stores (record format < 6) carried the same fact as
    `incomplete: true` inside the record itself; both are checked.
    """
    root, p = Path(root), Path(path)
    p = p if p.is_absolute() else root / p
    store = root / STORE / "incomplete"
    if store.is_dir():
        for key in _keys(root, p):
            if _record_path(store, key).is_file():
                return True
    rec = run_record(root, p)
    return bool(rec and rec.get("incomplete"))


def _moved_inputs(root: Path, rec: dict) -> list[str]:
    """Recorded input digests that no longer describe the file on disk.

    `input_checksums` is Snakemake's, written at run time, and the values are
    `sha256:<hex>` (bare hex in older stores, which is why the prefix is
    stripped rather than required). An entry whose file has since vanished
    counts as moved: the run cannot be re-read from data that is not there.
    """
    moved = []
    for name, recorded in (rec.get("input_checksums") or {}).items():
        if not recorded:
            continue
        want = str(recorded).split(":", 1)[-1]
        q = Path(name)
        now = digest(q if q.is_absolute() else root / q)
        if now != want:
            moved.append(name)
    return sorted(moved)


def _edited_after(root: Path, path: Path, rec: dict) -> float | None:
    """Seconds by which the file is newer than the run that recorded it.

    Snakemake sets `endtime` from the output's own mtime at completion, so an
    unedited file matches it exactly. This is the only signal available for a
    post-run edit, because Snakemake stores no digest for its outputs.
    """
    end = rec.get("endtime")
    if not isinstance(end, int | float) or not path.is_file():
        return None
    drift = path.stat().st_mtime - float(end)
    return drift if drift > MTIME_SLACK else None


# ---------------------------------------------------------------------------
# The grade itself
# ---------------------------------------------------------------------------

def _manual_note(root: Path, rel: str) -> str:
    """What a `context.toml` says about a file no rule produced, if anything."""
    for d in _context(root).get("datasets", []):
        if rel in [str(f) for f in d.get("files", [])] and d.get("origin") == "manual":
            return (f" It is declared in context.toml as the manual dataset "
                    f"{d['id']!r}, whose evidence is the pinned sha256 that "
                    f"`claimcheck.datasets` checks on every load — which is a "
                    f"different and weaker thing from a run that can be repeated.")
    return ""


def grade(root: Path | str, path: Path | str, *, reproduced: bool = False) -> Grade:
    """Grade one file from Snakemake's record of the run that made it.

    `reproduced` is passed in rather than discovered, following the one idea
    worth keeping from the retired predecessors: a grader that also gathers its
    own evidence is a grader that can quietly agree with itself. `reproduce()` below
    is what establishes it.
    """
    root = Path(root).resolve()
    p = Path(path)
    p = (p if p.is_absolute() else root / p).resolve()
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError:
        rel = str(p)
    sha = digest(p)
    rec = run_record(root, p)
    started = unfinished(root, p)

    if rec is None and not started:
        return Grade(rel, "NONE", None, sha,
                     "on disk, and no rule in this workflow's Snakemake store "
                     "claims it. Write the rule that produces it, run it, then "
                     "re-ask." + _manual_note(root, rel))
    rec = rec or {}
    rule = rec.get("rule")
    moved = _moved_inputs(root, rec)
    extra = {k: rec.get(k) for k in ("shellcmd", "conda_env",
                                     "software_stack_hash", "starttime",
                                     "endtime") if rec.get(k) is not None}
    # Annotated so mypy can splat it. Every value is a different field type and
    # mypy cannot narrow a `**dict` against a typed dataclass; a TypedDict here
    # would be four lines to say what the dataclass already says.
    common: dict[str, Any] = {"rule": rule, "sha256": sha,
                              "changed_inputs": tuple(moved), "extra": extra}

    # --- DECLARED: a rule claims the path, and no completed run left a file.
    if started:
        return Grade(rel, "DECLARED", stale=True, why=(
            f"rule {rule or '?'} started and never finished — Snakemake still "
            f"holds the incomplete marker for this output, so whatever is on "
            f"disk is what an interrupted run left. Re-run it."), **common)
    if not p.is_file():
        return Grade(rel, "DECLARED", stale=True, why=(
            f"rule {rule!r} recorded a completed run producing this path, and "
            f"the file is not on disk now. The record has outlived what it "
            f"describes; re-run the rule."), **common)

    # --- The two ways a completed run stops describing the file.
    moved_why = (
        f"{len(moved)} recorded input digest(s) no longer match the file of "
        f"that name ({', '.join(moved[:3])}{', …' if len(moved) > 3 else ''}), "
        f"so these bytes were not computed from the data now present. Re-run "
        f"the rule." if moved else "")
    drift = _edited_after(root, p, rec)
    drift_why = (
        f"the file is {drift:,.1f} s newer than the endtime rule {rule!r} "
        f"recorded, so it was written or touched after the run that is supposed "
        f"to describe it. This is mtime and not a digest — Snakemake stores no "
        f"digest for its own outputs — so it catches an edit and would miss one "
        f"that preserved the mtime. `--reproduce` settles it either way."
        if drift is not None else "")

    if not reproduced:
        why = f"produced by a completed run of rule {rule!r}; not re-run and compared."
        tail = " ".join(x for x in (moved_why, drift_why) if x)
        return Grade(rel, "EXECUTED", stale=bool(tail),
                     why=f"{why} {tail}".strip(), **common)

    # --- Re-run and identical, which is the evidence an mtime cannot give.
    if moved:
        return Grade(rel, "REPRODUCED", stale=True, why=(
            f"re-ran to the same bytes under rule {rule!r}, but an input digest "
            f"has moved since: the output is reproducible and its inputs are "
            f"not what they were. " + moved_why), **common)
    return Grade(rel, "VERIFIED", stale=False, why=(
        f"re-ran to the same bytes under rule {rule!r} with every recorded "
        f"input digest unchanged." + (
            " The mtime was later than the recorded endtime, which the "
            "reproduction supersedes: whatever touched the file, the bytes are "
            "the ones the recipe produces." if drift is not None else "")),
        **common)


def _context(root: Path) -> dict:
    """`context.toml`, or `{}` where there is none.

    Missing is fine — grading works on any Snakemake project. Malformed is not
    swallowed: a `context.toml` that does not parse is a problem to be told
    about, not a reason to grade fewer files without saying so.
    """
    f = Path(root) / "context.toml"
    return tomllib.loads(f.read_text()) if f.is_file() else {}


def targets(root: Path | str) -> list[Path]:
    """Every file worth grading in a claimcheck project.

    Everything under the results directory, plus every file a `[[datasets]]`
    block names — the data is where the numbers come from, and a computed
    dataset with no rule behind it is exactly the thing worth noticing.
    """
    root = Path(root).resolve()
    ctx = _context(root)
    out: list[Path] = []
    results = root / ctx.get("project", {}).get("results", "results")
    if results.is_dir():
        out += [p for p in results.rglob("*") if p.is_file()]
    for d in ctx.get("datasets", []):
        out += [root / f for f in d.get("files", []) if (root / f).is_file()]
    # RESOLVED, so a caller can pass these straight back in. Returning
    # `root / rel` made `grade_all` join them to the root a second time and
    # every file in `example/` graded NONE under the path `example/example/...`
    # — a checker reporting "no rule claims it" about a path that does not
    # exist, which is the confident wrong answer this package exists to refuse.
    return sorted({p.resolve() for p in out})


def recorded_outputs(root: Path | str) -> list[str]:
    """Every output Snakemake has a record for, read back out of its own store.

    THE INDEX IS NOT BUILT HERE, IT IS DECODED. `.snakemake/metadata/` files an
    entry per output under the urlsafe-base64 of that output's path, so the
    store's own filenames ARE the list of everything the workflow has produced.
    This reverses `_record_path` and returns the keys.

    WHY THIS IS NOT A SECOND MANIFEST. The rule for this package is that if
    Snakemake already does it, we do not build it again, and a committed table
    of paths and digests is exactly the thing that rule refuses — al_mace's
    a retired predecessor kept 35,608 sha256s in git for this purpose and was
    refused a home here on those grounds. Reading Snakemake's record is not the
    same act as writing one: nothing below hashes anything, writes anything, or
    survives the call.
    """
    root = Path(root).resolve()
    store = root / STORE / "metadata"
    if not store.is_dir():
        return []
    keys: list[str] = []
    for f in store.rglob("*"):
        if not f.is_file():
            continue
        # Rebuild the base64 from the path components, which `_record_path`
        # split at the filesystem's name limit and prefixed with `@`. Short
        # paths -- every real one -- are a single component and this is a
        # no-op, but the long-path case is the one that would silently drop
        # records if it were assumed away.
        parts = list(f.relative_to(store).parts)
        if any(not p.startswith("@") for p in parts[:-1]):
            continue                      # not one of Snakemake's own records
        b64 = "".join(p.lstrip("@") for p in parts)
        try:
            keys.append(base64.urlsafe_b64decode(b64.encode()).decode())
        except (ValueError, UnicodeDecodeError):
            # A file in the store that is not a record. Skipped rather than
            # guessed at: inventing a path here would report a deletion that
            # never happened, and a false alarm is how a check stops being read.
            continue
    return sorted(set(keys))


def vanished(root: Path | str) -> list[dict]:
    """Outputs Snakemake recorded producing that are no longer on disk.

    THE COMPLEMENT OF `targets()`, AND ONLY THIS ONE CAN SEE A DELETION.
    `targets()` walks the filesystem, so it enumerates what is there; a file
    that has been removed is simply not in its list and nothing notices. The
    question "what did this workflow make that is now gone?" can only be
    answered against a record of what was made, and Snakemake keeps one.

    This matters where it is used: `results/` and `data/` are gitignored in the
    projects this serves, so git cannot protect them, and a deletion there is
    invisible to every other check in the package. The grade ladder would call
    such a file NONE -- "no rule claims it" -- which is the wrong sentence
    entirely, because a rule did claim it and the bytes are gone.

    Each entry carries the rule and the run that made it, so the answer to "can
    I get it back?" is in the same row as the loss.
    """
    root = Path(root).resolve()
    out = []
    for key in recorded_outputs(root):
        p = Path(key) if Path(key).is_absolute() else root / key
        if p.exists():
            continue
        rec = run_record(root, p) or {}
        out.append({
            "path": key,
            "rule": rec.get("rule"),
            "starttime": rec.get("starttime"),
            "endtime": rec.get("endtime"),
            # `code` in Snakemake's record is the RECIPE TEXT, not a digest.
            # Calling it a hash cost nothing here and would cost a reader
            # everything: the first value it produced was "{ENV} {PY} -".
            "recipe": (rec.get("code") or "").strip()[:60] or None,
            "regenerate": f"snakemake {key}" if rec.get("rule") else None,
        })
    return out


def grade_all(root: Path | str, paths=None, *, reproduced=()) -> dict[str, Grade]:
    """`{relative path: Grade}` for `paths`, or for everything worth grading."""
    root = Path(root).resolve()
    todo = [Path(p) for p in paths] if paths is not None else targets(root)
    done = set(reproduced)
    out: dict[str, Grade] = {}
    for p in todo:
        q = (p if p.is_absolute() else root / p).resolve()
        try:
            rel = q.relative_to(root).as_posix()
        except ValueError:
            rel = str(q)
        out[rel] = grade(root, q, reproduced=rel in done or str(p) in done)
    return out


def quotable(root: Path | str, paths=None, *, reproduced=()) -> dict[str, Grade]:
    """Only the ones that may be quoted."""
    return {k: v for k, v in grade_all(root, paths, reproduced=reproduced).items()
            if v.quotable}


def report(root: Path | str, paths=None, *, reproduced=()) -> dict:
    """The whole verdict as JSON, for a rule to write or a reader to diff."""
    graded = grade_all(root, paths, reproduced=reproduced)
    counts: dict[str, int] = {}
    for g in graded.values():
        counts[g.grade] = counts.get(g.grade, 0) + 1
    return {
        "root": str(Path(root).resolve()),
        "files": len(graded),
        "quotable": sum(1 for g in graded.values() if g.quotable),
        "by_grade": dict(sorted(counts.items())),
        "grades": {k: {"grade": v.grade, "rule": v.rule, "sha256": v.sha256,
                       "quotable": v.quotable, "stale": v.stale,
                       "changed_inputs": list(v.changed_inputs), "why": v.why,
                       **v.extra}
                   for k, v in sorted(graded.items())},
    }


# ---------------------------------------------------------------------------
# The top two rungs, which cost a re-run
# ---------------------------------------------------------------------------

def _snakemake_cmd() -> list[str] | None:
    """The argv prefix that runs Snakemake here, or None if it truly is absent.

    `shutil.which` alone is not the question: a package invoked through an
    absolute interpreter path does not have that environment's `bin/` on PATH,
    and answering "not installed" in that case is a false statement rather than
    a missing feature.
    """
    exe = shutil.which("snakemake")
    if exe:
        return [exe]
    try:
        import importlib.util
        if importlib.util.find_spec("snakemake") is not None:
            return [sys.executable, "-m", "snakemake"]
    except (ImportError, ValueError):
        pass
    return None


class ReproduceError(RuntimeError):
    """The re-run could not be attempted, which is not the same as failing."""


def reproduce(root: Path | str, paths=None, *, cores: int = 1,
              timeout: float = 1800) -> dict[str, bool]:
    """Re-run the workflow **in a copy** and report which outputs came back same.

    `{relative path: bytes were identical}`. Feed it to `grade_all(reproduced=)`
    — the verdict is used in the call that computed it and never written down,
    so unlike the store this replaces it cannot describe bytes that have since
    changed.

    IN A COPY, and that is not fastidiousness. The predecessor's equivalent had
    a `--cheap` mode that promised a sandbox, ran a real `snakemake --force` in
    the real tree, and destroyed three result files. The copy excludes
    `.snakemake`, so the re-run starts with no memory of the first one.

    The targets are DELETED in the copy before the run, because "nothing to be
    done" is the answer an engine gives after comparing timestamps and it is
    exactly what this must not accept as a reproduction. Outputs not asked for
    are left in place and act as inputs; to reproduce a chain, ask for the chain.
    """
    root = Path(root)
    sm = _snakemake_cmd()
    if sm is None:
        raise ReproduceError(
            "snakemake is neither on PATH nor importable by "
            f"{sys.executable}, so nothing can be re-run. The grades stop at "
            f"EXECUTED, which is a fact about this machine, not about the data.")
    todo = [Path(p) for p in paths] if paths is not None else targets(root)
    rels = []
    for p in todo:
        q = (p if p.is_absolute() else root / p).resolve()
        try:
            rel = q.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue                    # outside the project; not ours to re-run
        # ONLY WHAT A RULE HAS ACTUALLY PRODUCED. Asking Snakemake to rebuild a
        # file no rule declares is a `MissingRuleException` that aborts the
        # whole run, so one ungraded CSV sitting in `data/` would take the
        # reproduction of every real output down with it — which is how a
        # checker comes to report nothing and look fine. Files with no record
        # grade NONE and there is nothing to reproduce.
        if run_record(root, q) is not None:
            rels.append(rel)
    if not rels:
        return {}

    with tempfile.TemporaryDirectory(prefix="claimcheck-reproduce-") as tmp:
        work = Path(tmp) / root.resolve().name
        shutil.copytree(root, work, symlinks=True, ignore=shutil.ignore_patterns(
            STORE, "__pycache__", ".git", ".pytest_cache"))
        for rel in rels:
            (work / rel).unlink(missing_ok=True)
        r = subprocess.run([*sm, "-c", str(cores), *rels], cwd=str(work),
                           capture_output=True, text=True, timeout=timeout)
        out = {}
        for rel in rels:
            before, after = digest(root / rel), digest(work / rel)
            out[rel] = before is not None and before == after
        if not any(out.values()) and r.returncode != 0:
            raise ReproduceError(
                f"snakemake exited {r.returncode} in the copy and reproduced "
                f"nothing, so this says nothing about the data:\n"
                f"{(r.stderr or r.stdout)[-1200:]}")
        return out


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="May this number be quoted? Grade files against Snakemake's "
                    "own record of the runs that made them.")
    ap.add_argument("root", nargs="?", default=".", help="the project directory")
    ap.add_argument("paths", nargs="*",
                    help="files to grade (default: results/ and every declared "
                         "dataset)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any graded file is not quotable")
    ap.add_argument("--reproduce", action="store_true",
                    help="re-run the workflow in a COPY first, so REPRODUCED and "
                         "VERIFIED can be reached")
    ap.add_argument("--vanished", action="store_true",
                    help="list outputs Snakemake recorded producing that are no "
                         "longer on disk, and exit 1 if any are. This is the only "
                         "check here that can see a DELETION: everything else "
                         "walks the filesystem, so a removed file is simply "
                         "absent from the list and nothing notices.")
    ap.add_argument("--json", metavar="OUT", help="write the full report here")
    a = ap.parse_args(argv)

    root = Path(a.root)

    if a.vanished:
        gone = vanished(root)
        recorded = recorded_outputs(root)
        if not recorded:
            # UNTESTABLE, NOT CLEAN. No store means the workflow has never run
            # here, and "0 vanished" would be a pass earned by having nothing to
            # lose. Reported as the absence of evidence it is.
            print(f"no Snakemake record under {root}/{STORE}/metadata — nothing "
                  f"to compare against, which is not the same as nothing missing")
            return 1
        for g in gone:
            print(f"VANISHED  {g['path']}")
            print(f"            made by rule {g['rule']!r}"
                  + (f"; regenerate with `{g['regenerate']}`" if g["regenerate"] else ""))
        print(f"\n{len(gone)} of {len(recorded)} recorded output(s) are no longer "
              f"on disk")
        return 1 if gone else 0

    paths = a.paths or None
    done: dict[str, bool] = {}
    if a.reproduce:
        try:
            done = reproduce(root, paths)
        except (ReproduceError, subprocess.TimeoutExpired) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        for rel, reproduced in sorted(done.items()):
            if not reproduced:
                print(f"  NOT REPRODUCED  {rel} — the re-run produced different "
                      f"bytes, so the recipe does not determine the file")

    graded = grade_all(root, paths, reproduced=[k for k, v in done.items() if v])
    for grade in graded.values():
        print(grade.render())
    n_quotable = sum(1 for grade in graded.values() if grade.quotable)
    print(f"\n{n_quotable} of {len(graded)} quotable "
          f"({', '.join(f'{k}: {v}' for k, v in sorted(_counts(graded).items()))})")
    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            report(root, paths, reproduced=[k for k, v in done.items() if v]),
            indent=2) + "\n")
        print(f"  -> {out}")
    return 1 if (a.check and n_quotable < len(graded)) else 0


def _counts(graded: dict[str, Grade]) -> dict[str, int]:
    out: dict[str, int] = {}
    for g in graded.values():
        out[g.grade] = out.get(g.grade, 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
