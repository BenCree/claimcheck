"""Has a declaration moved under its verdicts, and did it move in a way that matters?

    python -m claimcheck.workflow.context_drift <dir>...          # a row per directory
    python -m claimcheck.workflow.context_drift <dir>... --check  # exit 1 on a stale verdict

`results/claims.json` records `context_sha256`, the first 16 hex characters of
the sha256 of the `context.toml` that produced its numbers. Until 2026-11-22 the
shared `check_claims` rule took that file as `ancient()` with no `params`
trigger, so an edited `claimed` value did not re-adjudicate and the field went
on naming a declaration nobody could read any more. This counts where that
happened and, more usefully, SAYS WHICH KIND OF EDIT IT WAS, because the raw
digest mismatch does not distinguish them:

| verdict            | what it means                                                      |
|--------------------|--------------------------------------------------------------------|
| `current`          | the digest matches; the verdicts were computed from this file       |
| `prose_only`       | the digest differs and every verdict field a row echoes is the same |
| `verdict_fields`   | a field a verdict is computed from differs; the verdicts are stale  |
| `claims_changed`   | claims were added or removed, so there is nothing to compare        |
| `no_claims`        | no `results/claims.json`: never adjudicated, which is a queue item  |
| `no_digest`        | no `context_sha256` recorded, and the echoed fields still compared  |
| `unreadable`       | the declaration or the verdicts would not parse; the reason is given|
| `mirror_disagrees` | THIS TOOL IS WRONG: fields differ where the file is byte-identical  |

`prose_only` is a BOUND AND NOT A PROOF. A row echoes `id`, `x`, `y`, `kind`,
`method`, `claimed`, `claimed_range` and `resampling_unit`, so a changed
`alpha`, `tie_tol`, `group`, `interval`, `baseline`, `join`, `event` or
`max_ci_width` is invisible here and lands in `prose_only`. Reading it the other
way is safe: `verdict_fields` is a definite finding, and the total of
`prose_only` plus `verdict_fields` is the upper bound on the damage.

Nothing is re-run and nothing is written. A stale verdict is reported, not
quietly repaired: re-adjudicating one changes published numbers, which is the
project owner's call.

## The same defect one layer up: what a rule declared against what it read

    python -m claimcheck.workflow.context_drift --coverage <output> --names-file <paths>

A stale `ancient()` edge and an incomplete input list are one shape: **Snakemake
can only notice a change to a file the rule declared, so what a rule reads and
never declares can move for ever and the rule reports "Nothing to be done".**
That is what happened to the master ledger, found 2026-11-22 four days stale
with 29 experiments carrying a newer `results/claims.json`: rule `master_ledger`
in `/home/ben/code/shared/ledger/Snakefile` declared only its own build script,
while the script walked four trees.

`--coverage` asks it directly, from Snakemake's own store: of the files this
output is known to have been built from, which does its record NOT list? The
answer is only as good as the list handed to it, which is why the list is an
argument. For a verdict file the names come from `context.toml`, and that case
is swept per directory as `inputs_incomplete` below without being asked for. For
an aggregator the names come out of its own output, since a ledger whose rows
each carry an experiment directory states what it read.

WHAT THIS CANNOT DO, so that nobody reads more into it: it does not discover
what a script reads. Nothing static can, short of tracing the process. It
compares a recorded input list against a list somebody else is responsible for,
and its value is that the comparison is then recorded rather than eyeballed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

#: The fields a verdict row echoes from the declaration, so the only ones whose
#: drift can be read back out of `results/claims.json`.
ECHOED = ("kind", "method", "claimed", "claimed_range", "resampling_unit")


def digest(path: Path) -> str:
    """The first 16 hex of a file's sha256, which is what `relate` records."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def recorded_inputs(root: Path | str, output: Path | str) -> set[str] | None:
    """The input list Snakemake stored for one output, as absolute paths.

    `None` when there is no record at all, which is a different statement from
    an empty list: no record means no rule is known to have produced the file,
    and an empty list means a rule produced it from nothing declared.

    The key lookup is `claimcheck.grading.run_record`, which already knows that
    a rule declaring `results/claims.json` keys on that relative string while
    one declaring an absolute path keys on the absolute string, and how a long
    key is split. Duplicating that here is how the two would disagree.
    """
    from claimcheck.grading import run_record

    root = Path(root)
    rec = run_record(root, output)
    if rec is None:
        return None
    out = set()
    for f in rec.get("input", []) or []:
        p = Path(f)
        out.add(str((p if p.is_absolute() else root / p).resolve()))
    return out


def undeclared_sources(root: Path | str, output: Path | str, names) -> dict:
    """Which of `names` the rule that built `output` never declared.

    The ledger shape: a rule can only be made stale by a file it declared, so a
    source that is read and not declared can move for ever in silence. `names`
    is the list of files the output is known to have been built from, and it is
    an argument because nothing static can derive it from a shell command.
    """
    root = Path(root)
    declared = recorded_inputs(root, output)
    named = {str((Path(n) if Path(n).is_absolute() else root / n).resolve())
             for n in names if str(n).strip()}
    if declared is None:
        return {"output": str(Path(output)), "verdict": "no_record",
                "n_named": len(named), "n_declared": 0, "undeclared": []}
    missing = sorted(named - declared)
    return {"output": str(Path(output)),
            "verdict": "inputs_incomplete" if missing else "inputs_cover_sources",
            "n_named": len(named), "n_declared": len(declared),
            "undeclared": missing}


def _declared(ctx: dict) -> dict:
    """The echoable fields of every declared claim, keyed by id and columns.

    `resampling_unit` falls back to the x dataset's, mirroring
    `claimcheck/relate.py:1841`. Nothing else is derived: `x` and `y` are
    already written `dataset.column` in the declaration, which is the form the
    row carries.
    """
    units = {d.get("id"): d.get("resampling_unit") for d in ctx.get("datasets", [])}
    out = {}
    for rel in ctx.get("relationships", []):
        x, y = str(rel.get("x", "")), str(rel.get("y", ""))
        key = (str(rel.get("id", "") or ""), x, y)
        out[key] = {
            "kind": rel.get("kind"),
            "method": rel.get("method"),
            "claimed": rel.get("claimed"),
            "claimed_range": rel.get("claimed_range"),
            "resampling_unit": (rel.get("resampling_unit")
                                or units.get(x.split(".")[0])),
        }
    return out


def _rows(doc: dict) -> dict:
    """The same fields, as `results/claims.json` recorded them."""
    out = {}
    for row in doc.get("relationships", []):
        key = (str(row.get("id", "") or ""), str(row.get("x", "")),
               str(row.get("y", "")))
        out[key] = {k: row.get(k) for k in ECHOED}
    return out


def _coverage_of_a_verdict_file(root: Path, ctx: dict) -> dict:
    """Does the record of `results/claims.json` list the declaration it read?

    The verdict file's own sources are known without being told: `context.toml`
    and every non-`results/` file the datasets declare, which is exactly what
    `declared_files` derives for the shared rule's input list. A project that
    hand-copied `check_claims` instead of including the shared rules can miss
    one, and then a changed dataset never re-adjudicates either.
    """
    from claimcheck.workflow import declared_files

    try:
        names = ["context.toml", *declared_files(root)]
    except Exception:                                  # noqa: BLE001 - fail soft
        names = ["context.toml"]
    return undeclared_sources(root, "results/claims.json", names)


def inspect(root: Path) -> dict:
    """One directory, as a record. Never raises: a failure is a row with a reason."""
    root = Path(root)
    rec: dict = {"dir": str(root.resolve()), "verdict": "", "detail": ""}
    ctx_file, claims_file = root / "context.toml", root / "results" / "claims.json"
    try:
        ctx = tomllib.loads(ctx_file.read_text())
        now = digest(ctx_file)
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        rec.update(verdict="unreadable", detail=f"context.toml: {e}")
        return rec
    rec["context_sha256_now"] = now
    if not claims_file.is_file():
        rec.update(verdict="no_claims", detail="never adjudicated")
        return rec
    try:
        doc = json.loads(claims_file.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        rec.update(verdict="unreadable", detail=f"results/claims.json: {e}")
        return rec
    # A SECOND, INDEPENDENT QUESTION, on its own field so it cannot blur the
    # staleness count: did the rule that wrote this verdict file declare the
    # files it read? A source never declared is one Snakemake can never see
    # change, which is the master ledger's defect in this directory's terms.
    cov = _coverage_of_a_verdict_file(root, ctx)
    rec["inputs"] = cov["verdict"]
    rec["inputs_undeclared"] = cov["undeclared"]

    was = doc.get("context_sha256")
    rec["context_sha256_recorded"] = was or ""

    declared, rows = _declared(ctx), _rows(doc)
    # THE CLAIMS IN BOTH ARE COMPARED EVEN WHEN SOME ARE ONLY IN ONE. Returning
    # at the first difference would have reported `claims_changed` for
    # `active_latent_real` (55 declared, 24 adjudicated) and said nothing about
    # whether those 24 verdicts still describe their claims, which is the more
    # serious of the two questions.
    moved = [f"{key[0] or key[1] + '~' + key[2]}.{f}: {rows[key][f]!r} -> "
             f"{declared[key][f]!r}"
             for key in sorted(set(declared) & set(rows)) for f in ECHOED
             if declared[key][f] != rows[key][f]]
    added, gone = sorted(set(declared) - set(rows)), sorted(set(rows) - set(declared))
    counted = (f"{len(added)} declared with no verdict, {len(gone)} verdicts "
               f"with no declaration")
    rec["n_claims_declared"], rec["n_verdicts"] = len(declared), len(rows)
    if was and was == now:
        # THE SELF-CHECK. The declaration is byte-identical to the one that
        # produced these verdicts, so nothing can have moved. Anything this
        # finds is a fault in the mirroring above (`_declared` reproduces
        # `relate`'s dataset fallback for `resampling_unit`), not drift in the
        # tree, and until it reads zero across a tree no count here is quotable.
        rec.update(verdict="mirror_disagrees" if moved else "current",
                   detail="; ".join(moved) if moved else "")
        return rec
    if moved:
        rec.update(verdict="verdict_fields",
                   detail="; ".join(moved) + (f" [and {counted}]" if added or gone
                                              else ""))
    elif not was:
        # The comparison still ran, so this is weaker than `current` and
        # stronger than nothing: no digest was recorded, and every claim the
        # verdict file and the declaration share is unchanged.
        rec.update(verdict="no_digest",
                   detail=f"no context_sha256 recorded; {counted}; every claim "
                          f"in both is unchanged")
    elif added or gone:
        rec.update(verdict="claims_changed",
                   detail=f"{counted}; every claim in both is unchanged")
    else:
        rec.update(verdict="prose_only",
                   detail=f"{len(rows)} claims, every echoed field unchanged")
    return rec


def sweep(roots) -> list[dict]:
    """Every directory under `roots` that declares a `context.toml`."""
    seen, out = set(), []
    for r in roots:
        r = Path(r)
        found = ([r / "context.toml"] if (r / "context.toml").is_file()
                 else sorted(r.rglob("context.toml")))
        for f in found:
            d = f.parent.resolve()
            if d not in seen:
                seen.add(d)
                out.append(inspect(d))
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m claimcheck.workflow.context_drift",
        description="whether a declaration moved under its verdicts, and whether "
                    "the move changed a number")
    ap.add_argument("roots", nargs="*", metavar="DIR",
                    help="an experiment directory, or a tree to walk for them")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any directory's verdicts are stale")
    ap.add_argument("--json", metavar="OUT", help="write every record here")
    ap.add_argument("--only", metavar="VERDICT", default="",
                    help="print only rows with this verdict")
    ap.add_argument("--coverage", metavar="OUTPUT",
                    help="one output file: which of --names-file does the rule "
                         "that built it not declare?")
    ap.add_argument("--names-file", metavar="PATHS",
                    help="the sources that output was built from, one path per line")
    a = ap.parse_args(argv)

    if a.coverage:
        if not a.names_file:
            ap.error("--coverage needs --names-file")
        out = Path(a.coverage)
        root = Path(a.roots[0]) if a.roots else out.parent
        names = Path(a.names_file).read_text().splitlines()
        rec = undeclared_sources(root, out, names)
        print(f"{rec['verdict']}\t{rec['output']}")
        print(f"  {rec['n_named']} sources named, {rec['n_declared']} declared "
              f"in Snakemake's record, {len(rec['undeclared'])} named and not "
              f"declared")
        for p in rec["undeclared"]:
            print(f"  undeclared: {p}")
        if a.json:
            Path(a.json).write_text(json.dumps(rec, indent=2) + "\n")
            print(f"\nwritten: {Path(a.json).resolve()}")
        return 1 if a.check and rec["verdict"] != "inputs_cover_sources" else 0

    if not a.roots:
        ap.error("give at least one directory, or use --coverage")
    records = sweep(a.roots)
    for rec in records:
        if a.only and rec["verdict"] != a.only:
            continue
        print(f"{rec['verdict']}\t{rec['dir']}\t{rec['detail']}")
    counts: dict[str, int] = {}
    inputs: dict[str, int] = {}
    for rec in records:
        counts[rec["verdict"]] = counts.get(rec["verdict"], 0) + 1
        if rec.get("inputs"):
            inputs[rec["inputs"]] = inputs.get(rec["inputs"], 0) + 1
    print(f"\n{len(records)} directories with a context.toml")
    for k in sorted(counts):
        print(f"  {counts[k]:5d}  {k}")
    if inputs:
        print("\nand, of those adjudicated, whether the rule declared what it read")
        for k in sorted(inputs):
            print(f"  {inputs[k]:5d}  {k}")
    if a.json:
        Path(a.json).write_text(json.dumps(records, indent=2) + "\n")
        print(f"\nwritten: {Path(a.json).resolve()}")
    return 1 if a.check and counts.get("verdict_fields") else 0


if __name__ == "__main__":
    sys.exit(main())
