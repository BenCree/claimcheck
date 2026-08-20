"""RO-Crate 1.1 — built by the reference implementation, and really validated.

    python -m claimcheck.emit rocrate <project-dir>          # to stdout
    python -m claimcheck.rocrate <project-dir>               # writes the crate
    python -m claimcheck.rocrate --check <crate-dir>         # roc-validator
    python -m claimcheck.rocrate --selftest                  # the checker can fail

## What a crate adds that Croissant and DCAT do not

Croissant says *how do I read this*, DCAT says *how do I find this*, and a crate
says *here is the whole directory, packaged, with every file named and hashed*.
It is the format a repository deposit takes, and unlike the other two it
describes the payload as a **tree** rather than as a list of columns.

This is a real dependency and it was refused once on measured grounds — a
survey found five workflow engines already emitting Workflow Run Crate with
`runcrate` validating it, so a *sixth* description of a workflow run would have
been duplication. What is emitted here is not that. It describes the project's
files, and it leans on `rocrate` (the reference implementation) and
`roc-validator` (CRS4, Apache-2.0) rather than assembling JSON-LD by hand,
which is the only reason the module is 200 lines and not 700.

## The version is passed explicitly, always

`ROCrate()` defaults to **1.2** in ro-crate-py 0.15.1. Constructing it bare
moves the published format without a decision being taken and without a diff
anywhere showing it. `RO_CRATE_VERSION` below is the decision; changing that
line is how the format changes, and `tests/test_rocrate.py` pins it.

## `sha256` is DECLARED, or it is not published at all

The RO-Crate 1.1 context has 2,627 terms and **not one of them is a checksum**.
An undeclared key in a compacted JSON-LD document expands to nothing, so a
digest emitted as a bare key is silently dropped by every conformant processor
while `json.loads` — which is how anyone would look — still shows it present.
`EXTRA_TERMS` goes through ro-crate-py's sanctioned `extra_terms` hook and maps
it onto SPDX, which is a real vocabulary for exactly this.

## The metadata file defines the crate root

`ro-crate-metadata.json` is not a document that can live anywhere; the directory
holding it **is** the crate, and every data entity `@id` is relative to that
directory. Writing it into `results/` while describing the project root is the
Croissant `contentUrl` defect one format over — validates cleanly, resolves to
nothing. `write()` refuses that outright, and `claimcheck.conform` catches it
anywhere in a tree by dereferencing every entity.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from claimcheck.croissant import sha256

RO_CRATE_VERSION = "1.1"
RO_CRATE_CONFORMS = f"https://w3id.org/ro/crate/{RO_CRATE_VERSION}"

#: Terms RO-Crate 1.1 does not define and this crate needs.
EXTRA_TERMS = {"sha256": "http://spdx.org/rdf/terms#checksumValue"}

#: Guessing a media type from a suffix is a guess. The table is explicit so an
#: unknown suffix yields `application/octet-stream` rather than something
#: plausible and wrong.
MEDIA_TYPES = {
    ".csv": "text/csv", ".json": "application/json",
    ".jsonld": "application/ld+json", ".ttl": "text/turtle",
    ".tsv": "text/tab-separated-values", ".md": "text/markdown",
    ".toml": "application/toml", ".txt": "text/plain",
    ".yaml": "application/yaml", ".png": "image/png",
}

#: Never described: they are machinery, not data, and hashing a lock file that
#: changes on every run makes the crate differ from itself.
SKIP_DIRS = {".snakemake", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
             ".mypy_cache", ".tests"}

METADATA_NAME = "ro-crate-metadata.json"


class CrateError(RuntimeError):
    """The crate cannot be written where it was asked to go."""


def root_dataset(doc: dict) -> dict:
    """The root data entity, found by `@id` and never by position.

    Reading `doc["@graph"][1]` is true of a hand-assembled graph and false of
    ro-crate-py's, which puts the root first and the descriptor second. Indexing
    a graph by position asserts an ordering the format does not promise.
    """
    for e in doc.get("@graph", []):
        if e.get("@id") == "./":
            return e
    raise KeyError("no root data entity with @id './' in this crate")


def payload(crate_dir: Path) -> list[Path]:
    """Every file the crate will describe, sorted, machinery excluded."""
    out = []
    for p in sorted(crate_dir.rglob("*")):
        if not p.is_file() or p.name == METADATA_NAME:
            continue
        if set(p.relative_to(crate_dir).parts) & SKIP_DIRS:
            continue
        out.append(p)
    return out


def build(root: Path | str, name: str | None = None,
          description: str | None = None, license_url: str | None = None,
          exclude: tuple[str, ...] = ()) -> dict:
    """The crate document for the directory `root`, which becomes the crate.

    The structure — the `@context`, the descriptor, `conformsTo`, the
    trailing-slash root — is the library's. Only the content is ours.

    **This reads the tree it describes, so WHEN it runs is part of what it
    says.** Every `sha256` is a statement about bytes at the moment of the call,
    so a caller that writes more files into the same tree afterwards must name
    them in `exclude` or build the crate last. Describing a file that is about
    to be replaced is a digest for bytes that will not be there, and RO-Crate
    1.1 does not verify checksums at REQUIRED, so no validator would say so.

    Title, description and licence come from `context.toml` when it exists, so
    the crate agrees with the Croissant rather than restating it differently.
    """
    from rocrate.model.contextentity import ContextEntity
    from rocrate.rocrate import ROCrate

    root = Path(root)
    meta: dict = {}
    ctx_file = root / "context.toml"
    if ctx_file.is_file():
        import tomllib
        meta = tomllib.loads(ctx_file.read_text()).get("project", {})

    crate = ROCrate(version=RO_CRATE_VERSION)
    crate.metadata.extra_terms.update(EXTRA_TERMS)
    crate.name = name or meta.get("title") or meta.get("name") or root.name
    # NOTHING HERE ASSERTS ANYTHING ABOUT VALIDATORS. What this crate conforms
    # to is decided by `validate()` and recorded beside it, never claimed in
    # prose by the thing being validated.
    crate.description = (description or (meta.get("context") or "").strip()
                         or crate.name)
    # Day granularity, not the library's second-resolution default: a stamp
    # finer than the question needs makes a rebuild and a real change the same
    # diff.
    crate.datePublished = (meta.get("date_published")
                           or date.today().isoformat())
    lic = license_url or meta.get("license")
    if lic and str(lic).startswith("http"):
        crate.root_dataset["license"] = crate.add(ContextEntity(
            crate, str(lic), properties={"@type": "CreativeWork"}))
    elif lic:
        crate.root_dataset["license"] = str(lic)

    for p in payload(root):
        rel = p.relative_to(root).as_posix()
        if rel in exclude:
            continue
        props = {"name": p.name, "contentSize": p.stat().st_size,
                 "encodingFormat": MEDIA_TYPES.get(p.suffix,
                                                   "application/octet-stream"),
                 "sha256": sha256(p)}
        crate.add_file(source=p, dest_path=rel, properties=props)

    return crate.metadata.generate()


def render(root: Path | str) -> str:
    """The emitter interface, shared with the other formats.

    Returns the crate for `root` itself. Where it is written matters — see
    `write()` — which is why `python -m claimcheck.rocrate` exists beside
    `python -m claimcheck.emit rocrate`.
    """
    return json.dumps(build(root), indent=2) + "\n"


def write(root: Path | str, out: Path | str | None = None) -> Path:
    """Write `ro-crate-metadata.json` into the directory the crate describes.

    Refuses any other location. The directory holding the metadata file IS the
    crate root and every `@id` is relative to it, so a crate describing `.` from
    inside `results/` names files that are not there — clean JSON-LD pointing at
    nothing, which is the shape of the Croissant defect this package already has
    on record.
    """
    root = Path(root)
    out = Path(out) if out is not None else root / METADATA_NAME
    if out.name != METADATA_NAME:
        raise CrateError(
            f"an RO-Crate descriptor must be named {METADATA_NAME!r}; the "
            f"specification identifies it by that name, not by its content")
    if out.resolve().parent != root.resolve():
        raise CrateError(
            f"refusing to write a crate for {root} into {out.parent}. The "
            f"directory holding {METADATA_NAME} IS the crate root, and every "
            f"entity @id is relative to it — written there, this crate names "
            f"files that are not in that directory. Write it to "
            f"{root / METADATA_NAME} instead.")
    out.write_text(json.dumps(build(root), indent=2) + "\n")
    return out


# ---------------------------------------------------------------------------
# conformance
# ---------------------------------------------------------------------------

@dataclass
class Conformance:
    profile: str
    passed: bool
    checks_run: int
    checks_failed: int
    issues: list
    #: True when `roc-validator` CRASHED rather than reaching a verdict.
    #: `checks_run: 0` beside `passed: False` is the tell, and a report without
    #: this field makes "the validator broke" read as "your crate is invalid".
    crashed: bool = False

    def as_dict(self) -> dict:
        return {"profile": self.profile, "passed": self.passed,
                "crashed": self.crashed, "checks_run": self.checks_run,
                "checks_failed": self.checks_failed, "issues": self.issues[:20]}


def validate(crate_dir: Path | str, profile: str = "ro-crate-1.1",
             severity: str = "REQUIRED") -> Conformance:
    """Run `roc-validator` against a crate directory. Really runs.

    Offline by construction: the validator is told not to fetch remote crates,
    because a conformance report that depends on the network says something
    different on a train.

    **A pass at REQUIRED is not a statement of completeness.** A crate with no
    digests and no agent passes at REQUIRED with zero issues; it says the crate
    is well formed, not that it is informative. `readable()` is the check that
    asks whether anything can be got out of it.
    """
    try:
        from rocrate_validator import services
        from rocrate_validator.models import Severity, ValidationSettings

        r = services.validate(ValidationSettings(
            rocrate_uri=str(Path(crate_dir).resolve()),
            profile_identifier=profile,
            requirement_severity=getattr(Severity, severity),
            disable_remote_crate_download=True,
            abort_on_first=False,
        ))
        issues = [f"{i.severity.name}: {i.message}" for i in r.get_issues()]
        return Conformance(profile=profile, passed=bool(r.passed()),
                           checks_run=len(r.executed_checks),
                           checks_failed=len(r.failed_checks), issues=issues)
    except Exception as e:                                   # noqa: BLE001
        # Zero checks ran, so nothing about this crate has been agreed to.
        return Conformance(profile=profile, passed=False, crashed=True,
                           checks_run=0, checks_failed=0,
                           issues=[f"roc-validator did not reach a verdict — "
                                   f"{type(e).__name__}: {e}"[:400]])


def readable(metadata: Path | str) -> tuple[list[str], int]:
    """Can a consumer actually GET the files this crate names?

    Returns `(complaints, dereferenced)`. **The count is not decoration.** A
    crate whose every entity is a remote URI has nothing here to open, so an
    empty complaint list would mean "nothing was checked" while reading as a
    clean pass. The caller reports UNKNOWN on a zero.

    Validation cannot answer this. RO-Crate 1.1 requires a descriptor, a root
    and well-formed JSON-LD; it does not require the entities to denote
    anything, and does not verify checksums at REQUIRED. So a crate whose file
    entities point one directory up passes `roc-validator` and yields nothing to
    whoever unpacks it — the Croissant `contentUrl` defect, one format over.

    Three questions, each of which a validator leaves unanswered:
    the entity resolves to a file, the recorded size is the size on disk, and
    the recorded digest is the digest of those bytes.
    """
    metadata = Path(metadata)
    crate_dir = metadata.parent
    try:
        doc = json.loads(metadata.read_text())
    except (OSError, ValueError, RecursionError, UnicodeDecodeError) as e:
        return ([f"the crate will not parse — {type(e).__name__}: {e}"[:200]], 0)

    bad: list[str] = []
    n = 0
    local = 0
    for entity in doc.get("@graph", []):
        types = entity.get("@type", [])
        types = types if isinstance(types, list) else [types]
        if "File" not in types:
            continue
        n += 1
        eid = str(entity.get("@id", ""))
        if "://" in eid:
            continue                     # a remote entity; nothing local to check
        local += 1
        p = crate_dir / eid
        if not p.is_file():
            bad.append(f"{eid!r} denotes nothing — no file at {p}. An entity "
                       f"@id is relative to the directory holding "
                       f"{METADATA_NAME}, not to the project root")
            continue
        size = entity.get("contentSize")
        if size is not None and int(size) != p.stat().st_size:
            bad.append(f"{eid!r} records contentSize {size} and is "
                       f"{p.stat().st_size} bytes on disk")
        want = entity.get("sha256")
        if want and want != sha256(p):
            bad.append(f"{eid!r} records a sha256 that is not the digest of "
                       f"the bytes at {p}")
    if not n:
        bad.append("this crate names no File entity, so there is nothing in it "
                   "for a consumer to read")
    return bad, local


def _selftest() -> int:
    """Two-sided, and self-contained: no example tree, no network.

    Builds a real crate in a temporary directory, checks that it reads back,
    then breaks it in the exact way the measured Croissant defect broke — the
    metadata one directory away from what it describes — and checks that
    `readable()` says so. A checker nobody has watched fail is a checker nobody
    should trust.
    """
    import tempfile
    ok = True
    gone = Path(tempfile.gettempdir()) / "claimcheck-no-such-crate-dir"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "proj"
        (root / "data").mkdir(parents=True)
        (root / "data" / "x.csv").write_text("a,b\n1,2\n")
        doc = build(root)

        good = root / METADATA_NAME
        good.write_text(json.dumps(doc, indent=2) + "\n")
        c, _n = readable(good)
        print(f"  {'PASS' if not c else 'FAIL'}  a crate at the root of what it "
              f"describes reads back{'' if not c else '  ' + str(c[:1])}")
        ok &= not c

        moved = root / "results"
        moved.mkdir()
        (moved / METADATA_NAME).write_text(json.dumps(doc, indent=2) + "\n")
        c, _n = readable(moved / METADATA_NAME)
        hit = any("denotes nothing" in x for x in c)
        print(f"  {'PASS' if hit else 'FAIL'}  the same crate one directory "
              f"away is caught")
        ok &= hit

        try:
            write(root, moved / METADATA_NAME)
            print("  FAIL  writing a crate away from its root was permitted")
            ok = False
        except CrateError:
            print("  PASS  writing a crate away from its root is refused")

        tampered = json.loads(good.read_text())
        for e in tampered["@graph"]:
            if e.get("sha256"):
                e["sha256"] = "0" * 64
        (root / "t.json").write_text(json.dumps(tampered))
        # Read it back from the crate root so only the digest differs.
        good.write_text(json.dumps(tampered, indent=2) + "\n")
        c, _n = readable(good)
        hit = any("sha256" in x for x in c)
        print(f"  {'PASS' if hit else 'FAIL'}  a digest that does not describe "
              f"the bytes is caught")
        ok &= hit

    # A directory that does not exist. The validator must not report a pass for
    # a crate nothing looked at, and a crash must be labelled as one.
    v = validate(gone)
    print(f"  {'PASS' if not v.passed else 'FAIL'}  a missing crate is not a pass"
          f" (crashed={v.crashed})")
    ok &= not v.passed
    print(f"\nselftest: {'PASS' if ok else 'FAIL'}  "
          f"(RO-Crate {RO_CRATE_VERSION})")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if "--selftest" in a:
        return _selftest()
    if "--check" in a:
        a.remove("--check")
        crate = Path(a[0]) if a else Path.cwd()
        r = validate(crate)
        print(json.dumps(r.as_dict(), indent=2))
        complaints, _n = readable(crate / METADATA_NAME)
        for c in complaints:
            print(f"  UNREADABLE  {c}", file=sys.stderr)
        if r.crashed:
            print("\nroc-validator did not reach a verdict. That is a sentence "
                  "about the validator, not about your crate.", file=sys.stderr)
        return 0 if (r.passed and not complaints) else 1
    root = Path(a[0]) if a else Path.cwd()
    out = Path(a[1]) if len(a) > 1 else None
    try:
        written = write(root, out)
    except CrateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    doc = json.loads(written.read_text())
    files = [e for e in doc["@graph"]
             if "File" in (e.get("@type") if isinstance(e.get("@type"), list)
                           else [e.get("@type")])]
    print(f"  {written}  —  RO-Crate {RO_CRATE_VERSION}, {len(files)} file(s), "
          f"{len(doc['@graph'])} entities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
