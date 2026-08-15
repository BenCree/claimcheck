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

import hashlib
import json
import sys
import tomllib
from pathlib import Path

NS = "https://w3id.org/provchem/two-layer#"

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


def build(root: Path | str, claims_path: Path | None = None) -> dict:
    root = Path(root)          # same reason as `relate.check`
    ctx = tomllib.loads((root / "context.toml").read_text())
    ds = ctx["dataset"]
    claims_file = claims_path or (root / "results/claims.json")
    claims = (json.loads(claims_file.read_text())
              if claims_file.is_file() else {"relationships": []})

    # THE `@id` IS A SLUG, NOT A FILENAME. `measurements.csv` as an `@id` makes
    # `mlcroissant` report "there is a reference to node with UUID
    # 'measurements.csv' ... but this node doesn't exist" — the dot breaks
    # reference resolution. `emit/croissant.py` already slugs for this reason;
    # `contentUrl` carries the real path.
    def slug(name: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in name)

    first = slug(Path(ds["files"][0]).name)
    fields = []
    for v in ctx.get("variables", []):
        f = {"@type": "cr:Field", "@id": f"{first}_records/{v['name']}",
             "name": v["name"], "description": v.get("description", ""),
             # REQUIRED, and only discoverable once the official `@context` is
             # in place. With a hand-rolled minimal context `mlcroissant` could
             # not resolve `field` at all and reported the file clean; with the
             # real one it checks the Field and refuses this if it is absent.
             # A validator that cannot resolve your terms is not validating.
             "source": {"fileObject": {"@id": first},
                        "extract": {"column": v["name"]}},
             # The spec's atomic table lists exactly sc:Boolean, sc:Date,
             # sc:Float, sc:Integer, sc:Text. sc:Text is the conservative legal
             # choice for a column whose type nobody has declared.
             "dataType": "sc:Text"}
        if v.get("unit") is not None:
            # Absent stays absent: an omitted key means nobody has said, and
            # `unit = ""` means somebody declared it dimensionless. Emitting a
            # null for both would collapse the distinction the file exists for.
            f[f"{NS}unit"] = v["unit"]
            f[f"{NS}unitSystem"] = "UDUNITS-2"
        fields.append(f)

    dist = []
    for rel in ds["files"]:
        p = root / rel
        # `sc:FileObject`, NOT `cr:FileObject`. With `cr` bound to the Croissant
        # namespace the prefixed form resolves to
        # `http://mlcommons.org/croissant/FileObject` and `mlcroissant`
        # rejects it: the class lives in schema.org. Found by validating the
        # real output — the error does not appear until a validator sees it.
        # `cr:FileObject` is correct WITH the official `@context`. It was
        # briefly changed to `sc:FileObject` while a hand-rolled minimal
        # context was in use, which resolved the prefix wrongly — the fix for
        # the symptom, not the cause.
        dist.append({"@type": "cr:FileObject", "@id": slug(Path(rel).name),
                     "name": slug(Path(rel).name), "contentUrl": rel,
                     "encodingFormat": "text/csv", "sha256": sha256(p)})

    return {
        # `@language` IS NOT DECORATION. Without it `mlcroissant validate`
        # dies with `KeyError: '@language'` in `json_ld.py:187`, which reads as
        # "your file is broken" and is really "the validator indexes a key it
        # did not check for". Our conformance ledger has carried a `crashed`
        # verdict from this same bug; one declared key avoids it, and a crash
        # is never a pass.
        "@context": {**_CROISSANT_CONTEXT, "@language": "en", "twolayer": NS},
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": ds["name"],
        "description": ds.get("title", ds["name"]),
        f"{NS}context": ds.get("context", "").strip(),
        # The four `mlcroissant` reports as RECOMMENDED-but-absent. Emitted only
        # when declared: inventing a licence or a date to silence a warning is
        # the defect `emit/dcat.py` refuses against PSDI's shapes — "a value
        # invented to satisfy a validator is the defect, not the fix".
        **{k: v for k, v in (("license", ds.get("license")),
                             ("version", ds.get("version")),
                             ("datePublished", ds.get("date_published")),
                             ("citation", ds.get("citation"))) if v},
        f"{NS}resamplingUnit": ds.get("resampling_unit"),
        "distribution": dist,
        "recordSet": [{"@type": "cr:RecordSet", "@id": f"{first}_records",
                       "name": f"{first}_records", "field": fields}],
        f"{NS}claims": [
            {"x": r["x"], "y": r["y"], "xUnit": r.get("x_unit"),
             "yUnit": r.get("y_unit"), "method": r["method"],
             "claimed": r["claimed"], "estimate": r.get("estimate"),
             "ciLow": r.get("ci_lo"), "ciHigh": r.get("ci_hi"),
             "nUnits": r.get("n_units"),
             "verdict": r["verdict"], "status": r["status"], "why": r["why"],
             "evidenceSource": r.get("evidence", {}).get("source"),
             "evidenceLocator": r.get("evidence", {}).get("locator"),
             "extractedBy": r.get("evidence", {}).get("extracted_by")}
            for r in claims["relationships"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    root = Path(a[0]) if a else Path.cwd()
    out = Path(a[1]) if len(a) > 1 else root / "results/croissant.json"
    d = build(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2) + "\n")
    claims = d[f"{NS}claims"]
    refuted = sum(1 for c in claims if c["verdict"] == "refuted")
    unver = sum(1 for c in claims if c["verdict"] == "unverifiable")
    print(f"  {out}  —  {len(d['recordSet'][0]['field'])} field(s), "
          f"{len(claims)} claim(s), {refuted} refuted, {unver} unverifiable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
