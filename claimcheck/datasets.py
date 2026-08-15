"""Loading datasets, and the two ways a multi-dataset project goes wrong quietly.

Adding a dataset is three lines of `context.toml`. Relating it to another is
four more. Neither should be able to produce a confident wrong answer, and the
two ways they can are handled here rather than in the checker.

## 1. A dataset somebody put there by hand

`origin = "computed"` means a rule in this workflow made it, so Snakemake holds
its digest, its code and its environment. Nothing more is needed.

`origin = "manual"` means a person downloaded it and dropped it in. **Snakemake
cannot know anything about it** — no rule made it, so there is no record, and
the next person to replace it with a newer export leaves no trace at all.

So a manual dataset must declare `source`, `retrieved` and `sha256`, and the
digest is **checked on every load**. Replace the file and the run stops. That is
the entire protection, and it is deliberately a hard error rather than a
warning: a warning about a silently changed input is a warning nobody reads
until after they have published.

## 2. A join that fabricates rows

Relating columns in two datasets needs a key, and a key with duplicates on
either side produces a many-to-many join. Ten rows against ten rows sharing one
key is a hundred pairs, none of them measurements, and a correlation over them
is a number about the join rather than about the science.

This refuses that outright, and reports coverage either way — how many rows on
each side found a partner, and how many did not. A join that silently drops
half your data is the same defect one step quieter.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

COMPUTED, MANUAL = "computed", "manual"

#: Required of a manual dataset, and required for a reason each.
#:   source     — a file with no stated origin cannot be re-fetched or cited
#:   retrieved  — "the current version" is not a version
#:   sha256     — the only thing that notices a silent replacement
_MANUAL_REQUIRED = ("source", "retrieved", "sha256")


class DatasetError(ValueError):
    """Refused at load. Nothing downstream sees a half-loaded project."""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(root: Path | str, spec: dict) -> dict:
    """One dataset's rows, verified. Raises rather than returning a warning."""
    root = Path(root)
    did = spec.get("id")
    if not did:
        raise DatasetError("a dataset has no `id`, so nothing can refer to it")
    origin = spec.get("origin")
    if origin not in (COMPUTED, MANUAL):
        raise DatasetError(
            f"dataset {did!r} declares origin {origin!r}; it must be "
            f"'{COMPUTED}' (a rule in this workflow made it) or '{MANUAL}' "
            f"(a person put it there). There is no default, because the two "
            f"need different evidence and guessing would pick the weaker one.")

    if origin == MANUAL:
        missing = [k for k in _MANUAL_REQUIRED if not spec.get(k)]
        if missing:
            raise DatasetError(
                f"dataset {did!r} is manually imported and does not declare "
                f"{', '.join(missing)}. Snakemake knows nothing about a file no "
                f"rule produced, so these are the only record there is.")

    rows: list[dict] = []
    for rel in spec["files"]:
        p = root / rel
        if not p.is_file():
            raise DatasetError(f"dataset {did!r} names {rel}, which is not there")
        if origin == MANUAL:
            actual = sha256(p)
            if actual != spec["sha256"]:
                raise DatasetError(
                    f"dataset {did!r}: {rel} is not the file that was checked "
                    f"in.\n  declared sha256 {spec['sha256'][:16]}...\n"
                    f"  actual   sha256 {actual[:16]}...\n"
                    f"  A manually imported file that changed without anybody "
                    f"saying so is the one thing nothing else here can detect. "
                    f"If the new file is correct, update `sha256` in "
                    f"context.toml and say in the commit what changed.")
        with p.open(newline="", encoding="utf-8-sig") as fh:
            rows.extend(csv.DictReader(fh))

    if not rows:
        raise DatasetError(f"dataset {did!r} has a header and no rows")
    return {"id": did, "origin": origin, "rows": rows, "spec": spec,
            "digests": {rel: sha256(root / rel) for rel in spec["files"]}}


def join(left: dict, right: dict, key: str) -> tuple[list[tuple[dict, dict]], dict]:
    """Paired rows, and an honest report of what the join did.

    Returns `(pairs, report)`. `report` always carries the coverage; `pairs` is
    empty when the join was refused, and the reason is in `report["refused"]`.
    """
    lrows, rrows = left["rows"], right["rows"]
    report: dict = {"key": key, "left": left["id"], "right": right["id"],
                    "n_left": len(lrows), "n_right": len(rrows),
                    "refused": None}

    for side, ds in ((("left"), left), (("right"), right)):
        if key not in (ds["rows"][0] if ds["rows"] else {}):
            report["refused"] = (
                f"{ds['id']} has no column {key!r}, so the two cannot be "
                f"lined up at all")
            return [], report

    # DUPLICATE KEYS ARE REFUSED, not resolved. Ten rows against ten sharing one
    # key is a hundred pairs and none of them is a measurement.
    for name, ds in (("left", left), ("right", right)):
        dupes = [k for k, n in Counter(r[key] for r in ds["rows"]).items() if n > 1]
        if dupes:
            report["refused"] = (
                f"{ds['id']} has {len(dupes)} repeated value(s) of {key!r} "
                f"(e.g. {dupes[0]!r}). Joining on it would multiply rows into "
                f"pairs that were never measured together. Deduplicate the key, "
                f"or join on something that identifies a row.")
            report[f"n_duplicate_keys_{name}"] = len(dupes)
            return [], report

    index = {r[key]: r for r in rrows}
    pairs = [(l, index[l[key]]) for l in lrows if l[key] in index]
    report |= {
        "n_matched": len(pairs),
        "n_left_unmatched": len(lrows) - len(pairs),
        "n_right_unmatched": len(rrows) - len(pairs),
        # Stated so a reader does not have to divide. A join at 40% is not a
        # broken join, but it IS a different population from the one either
        # file describes, and the number belongs beside the result.
        "coverage": round(len(pairs) / max(len(lrows), 1), 4),
    }
    return pairs, report
