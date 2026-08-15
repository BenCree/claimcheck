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
    proj = ctx["project"]
    claims_file = claims_path or (root / "results/claims.json")
    claims = (json.loads(claims_file.read_text())
              if claims_file.is_file() else {"relationships": []})

    def slug(name: str) -> str:
        # `@id`s ARE SLUGGED. A dot in a node id makes `mlcroissant` report
        # "reference to node with UUID 'x.csv' ... but this node doesn't
        # exist" — the dot breaks reference resolution. `contentUrl` keeps the
        # real path.
        return "".join(c if c.isalnum() else "_" for c in name)

    by_ds = {}
    for v in ctx.get("variables", []):
        by_ds.setdefault(v["dataset"], []).append(v)

    dist, record_sets = [], []
    for d in ctx["datasets"]:
        first = slug(Path(d["files"][0]).name)
        for rel in d["files"]:
            p_ = root / rel
            fo = {"@type": "cr:FileObject", "@id": slug(Path(rel).name),
                  "name": slug(Path(rel).name), "contentUrl": rel,
                  "encodingFormat": "text/csv", "sha256": sha256(p_)}
            # THE ORIGIN TRAVELS WITH THE FILE. A consumer who cannot tell a
            # computed file from one somebody dropped in by hand has been given
            # the same confidence in both, and they do not deserve the same.
            fo[f"{NS}origin"] = d["origin"]
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
                 "dataType": "sc:Text",
                 "source": {"fileObject": {"@id": first},
                            "extract": {"column": v["name"]}}}
            if v.get("unit") is not None:
                f[f"{NS}unit"] = v["unit"]
                f[f"{NS}unitSystem"] = "UDUNITS-2"
            fields.append(f)
        if fields:
            record_sets.append({"@type": "cr:RecordSet",
                                "@id": f"{first}_records",
                                "name": f"{first}_records",
                                f"{NS}dataset": d["id"],
                                "field": fields})

    return {
        "@context": {**_CROISSANT_CONTEXT, "@language": "en", "twolayer": NS},
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
             "yUnit": r.get("y_unit"), "crossDataset": r.get("cross_dataset"),
             "method": r["method"], "claimed": r["claimed"],
             "estimate": r.get("estimate"), "ciLow": r.get("ci_lo"),
             "ciHigh": r.get("ci_hi"), "nUnits": r.get("n_units"),
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
    n_fields = sum(len(rs["field"]) for rs in d["recordSet"])
    print(f"  {out}  —  {len(d['recordSet'])} record set(s), {n_fields} field(s), "
          f"{len(claims)} claim(s), {refuted} refuted, {unver} unverifiable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
