"""Croissant from a `context.toml`, plus the two things Croissant cannot say.

    python -m claimcheck.croissant <project-dir> [<out.json>]

**Units.** Croissant 1.1 has no unit mechanism — verified against the spec as
downloaded: no `unitText`, no `unitCode`, no `PropertyValue`, no
`variableMeasured`, no QUDT, and the single occurrence of the string `unit` in
103 kB is the gloss on `contentSize`. So a `Field` carries its unit as an
annotation in our own namespace, declared in `@context` rather than left as a
bare key.

**Claims.** Croissant describes what a dataset IS. It has nothing for what
somebody asserts ABOUT it, so the checked relationships ride alongside, each
with its verdict and the locator its evidence came from.

Both are additive. Strip the `twolayer:` namespace and what remains is ordinary
Croissant. Stdlib only — this imports nothing third-party, so it sits in `emit/`
by subject rather than by dependency.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

from claimcheck.namespace import NS

#: The official Croissant 1.1 context, copied verbatim into
#: `_croissant_context.py`. NOT hand-rolled — see that file for what happened
#: when it was.
from claimcheck._croissant_context import CONTEXT as _CROISSANT_CONTEXT


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(name: str) -> str:
    """One `@id`, with every character a reference resolver may choke on gone.

    `@id`s ARE SLUGGED. A dot in a node id makes `mlcroissant` report
    "reference to node with UUID 'x.csv' ... but this node doesn't exist" — the
    dot breaks reference resolution. `contentUrl` keeps the real path.
    """
    return "".join(c if c.isalnum() else "_" for c in str(name))


def file_id(rel: str) -> str:
    """The `@id` of one declared file: THE WHOLE DECLARED PATH, not its basename.

    **THE DEFECT THIS FIXES (2026-11-22).** Until this date the id was
    `slug(Path(rel).name)`, so two declared files whose basenames agree got ONE
    node: an experiment declaring eight distinct `edges_plot.csv` under
    different directories published a single `edges_plot_csv` carrying eight
    different `sha256` values, and the corpus merge counted 69 such conflated
    subjects inside single experiments
    (`/home/ben/code/shared/corpus/results/queries.md`). `claimcheck.graph`
    mints its file node from this id, so the same collapse reached the graph.
    rdflib and `mlcroissant` both merge by id and raise nothing, so a package
    validated while describing the wrong bytes.

    The id is derived from the declared path UNCONDITIONALLY rather than
    disambiguated only where a collision is noticed: an id that depends on which
    other files happen to sit beside it changes when an unrelated dataset is
    added, and a reader cannot tell which rule produced it.

    **IT IS NOT INJECTIVE AND `build` REFUSES THE CASE IT CANNOT NAME.** Every
    non-alphanumeric character becomes `_`, so `arm_a/edges.csv` and
    `arm/a_edges.csv` are two different files with one id. Narrowing the
    character class would be a second incompatible change to every id, so
    instead `build` refuses a project declaring such a pair and says which two
    paths collide; measured over the corpus on 2026-11-22, the count of
    affected experiments is in
    `/home/ben/code/shared/corpus/data/file_ids.csv`.
    """
    return slug(str(rel))


#: Croissant 1.1's atomic types, in full. The spec's table lists exactly these
#: five -- there is no Time, and there is no unit mechanism anywhere in the
#: format, which is why units go in our own namespace beside the field.
_ATOMIC = ("sc:Boolean", "sc:Date", "sc:Float", "sc:Integer", "sc:Text")


def _content_url(root: Path, rel: str, out_dir: Path | str | None) -> str:
    """The path a CONSUMER will resolve, relative to where the JSON-LD lands."""
    if out_dir is None:
        return rel
    try:
        return os.path.relpath(root / rel, Path(out_dir))
    except ValueError:
        # Different drives on Windows; an absolute path still loads.
        return str((root / rel).resolve())


def _data_type(root: Path, files: list, column: str) -> str:
    """`sc:Integer` / `sc:Float` / `sc:Text`, read off the column itself.

    Croissant exists so a loader knows what a column IS -- claimcheck's own
    `dcat.py` puts it as "column 3 is a float you extract from data.csv". Every
    field was emitted as `sc:Text` regardless, so `dG_exp` reached a consumer as
    b'-8.546824' and every downstream cast was theirs to get right.

    Inferred rather than declared, because a `type =` key in `context.toml`
    would be a second place for the truth to live and the CSV is the first.
    An EMPTY column stays `sc:Text`: nothing about it is known, and guessing
    Float for a column with no values is the confident-wrong answer.
    """
    vals = []
    for rel in files:
        p = root / rel
        if not p.is_file():
            continue
        with p.open(newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                v = str(r.get(column, "")).strip()
                if v:
                    vals.append(v)
    if not vals:
        return "sc:Text"
    try:
        for v in vals:
            int(v)
        return "sc:Integer"
    except ValueError:
        pass
    try:
        for v in vals:
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):
                return "sc:Text"      # non-finite is not a Float to a loader
        return "sc:Float"
    except ValueError:
        return "sc:Text"


def _json_safe(obj):
    """The document with every non-finite float replaced by None, so it is JSON and not Python's superset of it.

    WHY (2026-11-12). `claimed_range = [1.0, inf]` is a legal claim, and `json.dumps` writes its open side as the bare token
    `Infinity`. RFC 8259 has no such token. Python's own `json` and `jq` read it back, so nothing here noticed; `orjson` refuses it,
    and `orjson` is what `mlcroissant` reads with. The package validated in some readers and raised JSONDecodeError in the one whose
    job is to get the data out (found by projects/z-ml/experiments/hq_decoy_testset reading its own output back).

    AN OPEN SIDE IS WRITTEN AS null, in place: `"claimedRange": [1.0, null]` is "at least 1.0". The alternatives were weighed. A
    large sentinel (1.8e308) is a number the claim never stated, and a consumer comparing against it gets an answer that looks
    measured. Omitting the open side turns a two-element range into a one-element list whose remaining element no longer says
    which side it is. null keeps the position, cannot be mistaken for a bound, and is what a reader of any JSON library receives
    as "no value". A NaN, which no field should carry, becomes null by the same rule and not a token that breaks the file.

    THE POSITION SURVIVES A JSON READER AND NOT A JSON-LD ONE, so each claim also carries `claimedLow` and `claimedHigh`. Read as
    JSON-LD, an array is an unordered set and a null inside it is dropped: `[1.0, null]` and `[null, 1.0]` both arrive as the single
    value 1.0 (measured with rdflib's JSON-LD parser). The two named keys say which side a bound is on; an open side is null there
    too, which JSON-LD reads as the property being absent, and absent means unbounded. They are the names `graph.ttl` already uses.

    `render` and `main` then serialise with `allow_nan=False`, so a non-finite value that reaches them by any other route raises
    here and does not publish."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def build(root: Path | str, claims_path: Path | None = None,
          out_dir: Path | str | None = None) -> dict:
    """`out_dir` is where the JSON-LD will be WRITTEN, and it matters.

    `mlcroissant` resolves a relative `contentUrl` against the folder holding
    the JSON-LD file (`operations/download.py:48`, `filepath = ctx.folder /
    url`), not against the project root. Emitting the project-root-relative
    path put `data/x.csv` in a file at `results/croissant.json`, so every
    consumer looked for `results/data/x.csv` and found nothing.

    The result validated cleanly, which is what made it worth fixing rather
    than noticing: `mlc.Dataset(jsonld=...)` accepted the metadata and only
    `ds.records()` failed, so a test that validates and never reads passes on a
    Croissant that no one can load. Measured on this package's own example,
    2026-08-17 -- 0 records; with the file moved beside the data, 32.
    """
    root = Path(root)          # same reason as `relate.check`
    ctx = tomllib.loads((root / "context.toml").read_text())
    from claimcheck.emit import project_table
    proj = project_table(ctx)
    claims_file = claims_path or (root / "results/claims.json")
    claims = (json.loads(claims_file.read_text())
              if claims_file.is_file() else {"relationships": []})

    # TWO DECLARED PATHS, ONE `@id` IS REFUSED (2026-11-22). `file_id` is not
    # injective, so a project could still publish one node for two files and
    # that is the whole defect this change exists to remove. It is refused here
    # rather than silently disambiguated, because the author knows which of the
    # two paths should be renamed and this code does not.
    seen_ids: dict[str, str] = {}
    for d_ in ctx["datasets"]:
        for rel_ in d_["files"]:
            fid = file_id(rel_)
            prior = seen_ids.setdefault(fid, str(rel_))
            if prior != str(rel_):
                raise ValueError(
                    f"{root / 'context.toml'} declares {prior!r} and {rel_!r}, "
                    f"which are different files with one Croissant @id "
                    f"({fid!r}): every character that is not a letter or a "
                    f"digit becomes '_'. One node would carry both digests and "
                    f"no reader could tell. Rename one of the two paths")

    by_ds: dict[str, list[dict]] = {}
    for v in ctx.get("variables", []):
        by_ds.setdefault(v["dataset"], []).append(v)

    dist, record_sets = [], []
    for d in ctx["datasets"]:
        first = file_id(d["files"][0])
        for rel in d["files"]:
            p_ = root / rel
            fo = {"@type": "cr:FileObject", "@id": file_id(rel),
                  "name": file_id(rel),
                  "contentUrl": _content_url(root, rel, out_dir),
                  "encodingFormat": "text/csv", "sha256": sha256(p_)}
            # THE ORIGIN TRAVELS WITH THE FILE. A consumer who cannot tell a
            # computed file from one somebody dropped in by hand has been given
            # the same confidence in both, and they do not deserve the same.
            fo[f"{NS}origin"] = d["origin"]
            if d.get("level"):
                fo[f"{NS}level"] = d["level"]
            if d["origin"] == "manual":
                fo[f"{NS}manualSource"] = d.get("source")
                fo[f"{NS}retrieved"] = d.get("retrieved")
                fo[f"{NS}pinnedSha256"] = d.get("sha256")
            dist.append(fo)

        fields = []
        for v in by_ds.get(d["id"], []):
            f = {"@type": "cr:Field",
                 "@id": f"{first}_records/{v['name']}", "name": v["name"],
                 "description": v.get("description", ""),
                 "dataType": _data_type(root, d["files"], v["name"]),
                 "source": {"fileObject": {"@id": first},
                            "extract": {"column": v["name"]}}}
            if v.get("unit") is not None:
                f[f"{NS}unit"] = v["unit"]
                f[f"{NS}unitSystem"] = "UDUNITS-2"
            fields.append(f)
        if fields:
            rs = {"@type": "cr:RecordSet",
                  "@id": f"{first}_records",
                  "name": f"{first}_records",
                  f"{NS}dataset": d["id"],
                  "field": fields}
            # A dataset's `title` says in one sentence what a row of it IS, and
            # until 2026-08-20 nothing read it. Every one of 61 dataset blocks
            # in the project this was found in declared one, and three in this
            # package's own example — all of it written carefully and arriving
            # nowhere. It belongs in the record set's description, which is the
            # field a Croissant consumer shows beside the table.
            if d.get("title"):
                rs["description"] = str(d["title"]).strip()
            record_sets.append(rs)

    return _json_safe({
        "@context": {**_CROISSANT_CONTEXT, "@language": "en", "cc": NS},
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": proj["name"],
        "description": proj.get("title", proj["name"]),
        f"{NS}context": proj.get("context", "").strip(),
        **{k: v for k, v in (("license", proj.get("license")),
                             ("version", proj.get("version")),
                             ("datePublished", proj.get("date_published")),
                             ("citation", proj.get("citation"))) if v},
        "distribution": dist,
        "recordSet": record_sets,
        # The join report rides alongside: a correlation over a partial join is
        # a number about a different population, and the coverage says which.
        f"{NS}joins": claims.get("joins", []),
        f"{NS}claims": [
            {"x": r["x"], "y": r["y"], "xUnit": r.get("x_unit"),
             # THE THIRD COLUMN, where a method has one (`roc_auc_difference`, 2026-11-20). Added only when the claim
             # declares it, so every existing package stays byte-identical. Without it two claims that differ ONLY in
             # what the score was compared against (a CNN affinity against heavy-atom count, and the same affinity
             # against cLogP) are indistinguishable in the published metadata: same x, same y, two different numbers.
             **({"baseline": r["baseline"], "baselineUnit": r.get("baseline_unit")} if r.get("baseline") else {}),
             "yUnit": r.get("y_unit"), "crossDataset": r.get("cross_dataset"),
             "method": r["method"], "claimed": r.get("claimed"),
             "claimedRange": r.get("claimed_range"),
             "claimedLow": (r.get("claimed_range") or [None, None])[0],
             "claimedHigh": (r.get("claimed_range") or [None, None])[1],
             "estimate": r.get("estimate"), "ciLow": r.get("ci_lo"),
             "ciHigh": r.get("ci_hi"), "nUnits": r.get("n_units"),
             "verdict": r["verdict"], "status": r["status"], "why": r["why"],
             "evidenceSource": r.get("evidence", {}).get("source"),
             "evidenceLocator": r.get("evidence", {}).get("locator"),
             "extractedBy": r.get("evidence", {}).get("extracted_by"),
             "level": r.get("level")}
            for r in claims["relationships"]
        ],
    })


def render(root: Path | str) -> str:
    """The emitter interface, shared with `graph` and `dcat`."""
    return json.dumps(build(root), indent=2, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    root = Path(a[0]) if a else Path.cwd()
    out = Path(a[1]) if len(a) > 1 else root / "results/croissant.json"
    d = build(root, out_dir=out.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    claims = d[f"{NS}claims"]
    refuted = sum(1 for c in claims if c["verdict"] == "refuted")
    unver = sum(1 for c in claims if c["verdict"] == "unverifiable")
    n_fields = sum(len(rs["field"]) for rs in d["recordSet"])
    print(f"  {out}  —  {len(d['recordSet'])} record set(s), {n_fields} field(s), "
          f"{len(claims)} claim(s), {refuted} refuted, {unver} unverifiable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
