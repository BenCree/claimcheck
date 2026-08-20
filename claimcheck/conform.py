"""Sweep a tree for published metadata, validate every piece, and READ it.

    python -m claimcheck.conform <tree-or-file>
    python -m claimcheck.conform --json <tree>
    python -m claimcheck.conform --selftest

    from claimcheck.conform import sweep
    report = sweep(Path("datasets/"))

## Why a sweep and not a check per file

Metadata accumulates. A project publishes a manifest, then twenty more, and each
was correct on the day it was written against whatever the validator said then.
Nobody re-runs the first one. The measured case: a repository with 23 Croissant
manifests, **all 23 declaring 1.0** when 1.1 had been current for over a year,
and `prov:` — the whole reason to move — appearing in **1 of 23**.

## VALIDATING IS NOT READING, and that is the whole point of this module

On 2026-08-17 this package's own emitted Croissant validated cleanly under
`mlcroissant` and yielded **zero records** to a consumer, because `contentUrl`
was written relative to the project root while `mlcroissant` resolves it against
the folder holding the JSON-LD (`operations/download.py:48`). Every check that
existed passed. `tests/test_croissant_loads.py` is the miniature fix — one
format, one project, one test — and this is the general form:

> **for every format, the question is not "does it validate" but "can somebody
> get the data out of it".**

So each finding carries two verdicts, never folded together:

    validates   a validator ran and the document conformed
    reads       a consumer really got the payload through it

`validates=PASS reads=FAIL` is the defect above, and it is a line you can see.
A single status would have printed PASS.

## Four outcomes, and `UNKNOWN` is not one of the good ones

    PASS       both verdicts reached and both good
    FAIL       something ran and said no
    UNKNOWN    no verdict — unclassifiable, uninstalled validator, or a crash
    SKIP       the file is not metadata

`UNKNOWN` is counted separately, because folding it into PASS or SKIP is how a
sweep comes to report a clean bill of health over files nothing opened. A
summary reading `18 PASS, 0 FAIL` across 23 files has not told you about the
other five.

**A crash is UNKNOWN, never FAIL.** Measured 2026-08-12: `mlcroissant` 1.1.0
raises `KeyError: '@language'` on any manifest whose `@context` omits it — 13 of
23 in the first real tree this was run against. Reported as FAIL that reads
"your manifests are invalid"; as UNKNOWN it reads "the validator broke on them",
and only one of those sends you to the right place.

## Detection is structural, never by filename

`croissant.json` is a convention, not a contract, and a project that names its
manifest something else is not thereby exempt. A file is a Croissant manifest if
it declares `conformsTo` naming mlcommons or carries `recordSet`; an RO-Crate if
it has the `@id: ro-crate-metadata.json` descriptor the specification requires;
a DCAT record if the Turtle names the DCAT namespace. All three read the
document.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

PASS, FAIL, UNKNOWN, SKIP = "PASS", "FAIL", "UNKNOWN", "SKIP"

CROISSANT, ROCRATE, DCAT, NOTHING = "croissant", "ro-crate", "dcat", "?"

#: The extensions worth opening while sweeping a tree. Classification still
#: reads the document — the predecessor swept `*.json` only, while its own
#: stated principle was "never by filename", and Croissant *is* JSON-LD, so
#: identical bytes named `.jsonld` were not even counted in the denominator.
CANDIDATE_SUFFIXES = (".json", ".jsonld", ".ttl")


@dataclass
class Finding:
    path: str
    kind: str
    declared_version: str
    #: The two verdicts, kept apart. Folding them is the defect this exists for.
    validates: str = UNKNOWN
    reads: str = UNKNOWN
    detail: str = ""
    issues: list = field(default_factory=list)
    unreadable: list = field(default_factory=list)

    @property
    def status(self) -> str:
        # SKIP is set explicitly by the classifier, never inferred from `kind`.
        # Inferring it made a truncated document — kind `?` because nothing
        # could classify it — read as "not metadata", which is the one file
        # somebody needs told about.
        if SKIP in (self.validates, self.reads):
            return SKIP
        if FAIL in (self.validates, self.reads):
            return FAIL
        if UNKNOWN in (self.validates, self.reads):
            return UNKNOWN
        return PASS

    def as_dict(self) -> dict:
        return {"path": self.path, "kind": self.kind,
                "declared_version": self.declared_version,
                "status": self.status, "validates": self.validates,
                "reads": self.reads, "detail": self.detail,
                "issues": self.issues[:10], "unreadable": self.unreadable[:10]}


class Unreadable(Exception):
    """It looks like metadata and will not parse. Not the same as not being it."""


def _load_json(p: Path):
    """Parse, or raise.

    Returning `None` on a JSON error makes a truncated crate classify as "not
    metadata", which drops it out of the denominator entirely — a sweep then
    reports `0 metadata document(s) of 1 file(s)`. A file that will not parse is
    exactly the file somebody needs told about.

    `json` raises a BARE ValueError past the 4300-digit integer limit and
    RecursionError on deep nesting. Uncaught, ONE bad file aborts the sweep for
    every other file: this module's own failure mode, one level up.
    """
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError, RecursionError, UnicodeDecodeError) as e:
        raise Unreadable(f"{type(e).__name__}: {e}"[:160]) from None


def classify(p: Path) -> tuple[str, str]:
    """`(kind, declared_version)` — read off the document, never the filename."""
    if p.suffix.lower() == ".ttl":
        try:
            text = p.read_text(errors="replace")
        except OSError as e:
            raise Unreadable(f"{type(e).__name__}: {e}"[:160]) from None
        # A namespace scan, not a parse. Deciding whether to load a graph
        # library should not require loading a graph library, and the DCAT
        # namespace IRI in the bytes is as structural a fact as any.
        if "http://www.w3.org/ns/dcat#" in text:
            # No version to read: DCAT records do not carry one, and which
            # PROFILE this claims is decided by `_check_dcat` from the record's
            # own `dcterms:conformsTo` rather than guessed at here.
            return DCAT, ""
        return NOTHING, ""

    d = _load_json(p)
    if isinstance(d, list):
        # Expanded JSON-LD is a list. Rejecting it for not being a dict made a
        # perfectly good expanded document count as "not metadata".
        for node in d:
            if isinstance(node, dict) and any(
                    "mlcommons.org/croissant" in str(v) for v in node.values()):
                return CROISSANT, "(expanded)"
        return NOTHING, ""
    if not isinstance(d, dict):
        return NOTHING, ""
    graph = d.get("@graph")
    if isinstance(graph, list):
        for node in graph:
            if isinstance(node, dict) and node.get("@id") == "ro-crate-metadata.json":
                c = node.get("conformsTo")
                v = c.get("@id") if isinstance(c, dict) else (c or "")
                return ROCRATE, str(v).rstrip("/").rsplit("/", 1)[-1]
    conforms = d.get("dct:conformsTo") or d.get("conformsTo") or ""
    if isinstance(conforms, dict):
        conforms = conforms.get("@id", "")
    if "mlcommons.org/croissant" in str(conforms):
        return CROISSANT, str(conforms).rstrip("/").rsplit("/", 1)[-1]
    if "recordSet" in d or "cr:recordSet" in d:
        return CROISSANT, "(undeclared)"
    return NOTHING, ""


# ---------------------------------------------------------------------------
# one checker per format: each returns (validates, reads, detail, issues, unread)
# ---------------------------------------------------------------------------

def _check_croissant(p: Path) -> Finding:
    f = Finding(str(p), CROISSANT, "", detail="mlcroissant")
    try:
        import mlcroissant as mlc
    except ImportError:
        f.detail = ("mlcroissant is not installed — no verdict reached. "
                    "pip install 'claimcheck[conform]'")
        return f
    try:
        ds = mlc.Dataset(jsonld=str(p))
    except Exception as e:                                   # noqa: BLE001
        # Two ways to reach no verdict and they need different work. A refusal
        # by the validator is FAIL; a crash inside it is UNKNOWN. mlcroissant
        # raises the same exception type for both, so the string is all there
        # is — and when in doubt this reports UNKNOWN, because claiming a
        # verdict nobody reached is the worse error.
        msg = f"{type(e).__name__}: {e}"[:300]
        if "ValidationError" in type(e).__name__:
            f.validates, f.detail = FAIL, "mlcroissant refused the manifest"
            f.issues = [msg]
        else:
            f.detail = f"mlcroissant did not reach a verdict — {msg}"
        return f
    f.validates = PASS

    # AND NOW THE HALF THAT VALIDATION CANNOT DO.
    sets = list(ds.metadata.record_sets)
    if not sets:
        f.reads = FAIL
        f.unreadable = ["the manifest validates and declares no recordSet, so "
                        "there is nothing in it for a consumer to load"]
        return f
    try:
        empty = [rs.id for rs in sets
                 if not next(iter(ds.records(record_set=rs.id)), None)]
    except Exception as e:                                   # noqa: BLE001
        f.reads = FAIL
        f.unreadable = [
            f"the manifest validates and a consumer cannot read it — "
            f"{type(e).__name__}: {e}"[:300],
            "check contentUrl is relative to the JSON-LD, not the project root"]
        return f
    if empty:
        f.reads = FAIL
        f.unreadable = [f"record set {rid!r} validates and yields no rows"
                        for rid in empty]
        return f
    f.reads = PASS
    f.detail = f"mlcroissant, {len(sets)} record set(s) read"
    return f


def _check_rocrate(p: Path) -> Finding:
    f = Finding(str(p), ROCRATE, "", detail="roc-validator")
    try:
        from claimcheck import rocrate
    except ImportError as e:
        f.detail = f"the 'rocrate' extra is not installed — {e}"[:200]
        return f
    # An RO-Crate is validated as a DIRECTORY, not a file: the payload it
    # describes has to be there for the checks about it to mean anything.
    r = rocrate.validate(p.parent)
    if r.crashed:
        f.detail = "roc-validator crashed — no verdict reached"
        f.issues = r.issues[:10]
    else:
        f.validates = PASS if r.passed else FAIL
        f.detail = f"roc-validator, {r.checks_run} checks"
        f.issues = r.issues[:10]
    f.unreadable, n = rocrate.readable(p)
    f.reads = _verdict(f, f.unreadable, n,
                       "every entity in this crate is a remote URI, so "
                       "nothing local was opened")
    return f


def _verdict(f: Finding, complaints: list, dereferenced: int, why: str) -> str:
    """FAIL on a complaint, PASS on a check that ran, UNKNOWN on zero checks.

    **An empty complaint list is not a pass on its own.** A readback that
    dereferenced nothing produces exactly the same empty list as one that
    dereferenced forty files and found them all — and a checker that reports a
    pass over zero checks is what `iris_resolve` did in the predecessor for
    every call ever made, with a `root` parameter three call sites passed and
    nothing read.
    """
    if complaints:
        return FAIL
    if not dereferenced:
        f.detail = f"{f.detail}; not read — {why}".lstrip("; ")
        return UNKNOWN
    f.detail = f"{f.detail}; {dereferenced} file(s) really opened".lstrip("; ")
    return PASS


#: The one DCAT profile with vendored shapes here. A record is checked against a
#: profile it DECLARES and never against one somebody guessed at: reporting a
#: plain DCAT catalogue card as "fails PSDI" is an accusation about a standard
#: its author never invoked, which is the mirror image of claiming a profile you
#: do not meet.
_PSDI_MARKERS = ("metadata.psdi.ac.uk", "psdiDcatExt")


def _check_dcat(p: Path) -> Finding:
    # NO GRAPH LIBRARY IS IMPORTED HERE. Parsing lives in `dcat.readback` and
    # `psdi.check_file`, which are the modules that declare those extras; a
    # module deciding *whether* a graph library is needed must not import one to
    # decide. `test_dependencies.py` checks it rather than trusting it.
    f = Finding(str(p), DCAT, "plain", detail="")
    try:
        from claimcheck import dcat
    except ImportError as e:
        f.detail = f"the 'graph' extra is not installed — {e}"[:200]
        return f

    text = p.read_text(errors="replace")
    if any(m in text for m in _PSDI_MARKERS):
        f.declared_version = "psdi"
        from claimcheck import psdi
        r = psdi.check_file(p)
        if r.crashed:
            f.detail = f"pyshacl did not reach a verdict — {r.detail}"[:300]
            f.issues = [r.detail]
        else:
            f.validates = PASS if r.passed else FAIL
            f.detail = (f"PSDI shapes @{r.shapes_commit or 'unpinned'}, "
                        f"{len(r.exempt_violations)} exempt")
            f.issues = r.real_violations[:10]
    else:
        # UNKNOWN, and it is the true answer. W3C DCAT ships no normative
        # shapes, so for a record claiming no profile there is no validator to
        # run and nothing has agreed to anything. Saying PASS here would be a
        # clean bill of health signed by nobody.
        f.detail = ("no profile declared, and DCAT itself ships no normative "
                    "shapes — nothing validated this record. `claimcheck.emit "
                    "psdi` emits one that declares a profile and is checked "
                    "against it")
    f.unreadable, n = dcat.readback(p)
    if any("will not parse" in u for u in f.unreadable):
        f.validates = FAIL

    f.reads = _verdict(f, f.unreadable, n,
                       "every dcat:downloadURL here is an absolute IRI, so "
                       "nothing was fetched")
    return f


_CHECKERS = {CROISSANT: _check_croissant, ROCRATE: _check_rocrate,
             DCAT: _check_dcat}


def check_one(p: Path) -> Finding:
    try:
        kind, version = classify(p)
    except Unreadable as e:
        return Finding(str(p), NOTHING, "",
                       detail=f"looks like metadata and will not parse — {e}")
    if kind == NOTHING:
        return Finding(str(p), NOTHING, "", validates=SKIP, reads=SKIP,
                       detail="not a metadata document")
    try:
        f = _CHECKERS[kind](p)
    except Exception as e:                                   # noqa: BLE001
        # A crash is not a pass, and one bad file must not end the sweep.
        return Finding(str(p), kind, version,
                       detail=f"{type(e).__name__}: {e}"[:200])
    f.declared_version = version or f.declared_version
    return f


def sweep(root: Path | str, suffixes=None) -> dict:
    """Every metadata document under `root`, each read by its own consumer.

    **`root` may be one file.** `rglob` on a file matches nothing, so a sweep
    that only ever globbed printed `0 metadata document(s) of 0 file(s)` and
    exited 0 for a named file — an affirmative sentence about a document nothing
    had opened. A named file is a request for a verdict on that file, whatever
    its suffix; the suffix list decides what to *open* while sweeping a tree.
    """
    root = Path(root)
    suffixes = tuple(suffixes) if suffixes else CANDIDATE_SUFFIXES
    if root.is_file():
        cands = [root]
    else:
        cands = sorted(p for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in suffixes
                       and not set(p.parts) & {".snakemake", ".git",
                                               "__pycache__"})
    findings = [check_one(p) for p in cands]
    considered = [f for f in findings if f.status != SKIP]
    counts: dict[str, int] = {}
    versions: dict[str, int] = {}
    for f in considered:
        counts[f.status] = counts.get(f.status, 0) + 1
        key = f"{f.kind} {f.declared_version}"
        versions[key] = versions.get(key, 0) + 1
    return {
        "root": str(root),
        "target": "file" if root.is_file() else "directory",
        "files_seen": len(findings),
        "metadata_documents": len(considered),
        "skipped_not_metadata": len(findings) - len(considered),
        "by_status": dict(sorted(counts.items())),
        # The count with no cheaper substitute: documents a validator blessed
        # and a consumer could not use.
        "validated_but_unreadable": sum(
            1 for f in considered if f.validates == PASS and f.reads == FAIL),
        "by_declared_version": dict(sorted(versions.items())),
        "findings": [f.as_dict() for f in considered],
        # What was passed over, so a report with an empty denominator can say
        # what it declined to open rather than leaving the reader to guess
        # whether anything was there at all.
        "skipped": [f.as_dict() for f in findings if f.status == SKIP][:20],
    }


def format_report(rep: dict) -> str:
    lines = []
    for f in rep["findings"]:
        lines.append(f"{f['status']:<7} {f['kind']:<9} "
                     f"validates={f['validates']:<7} reads={f['reads']:<7} "
                     f"{f['path']}")
        if f["detail"]:
            lines.append(f"          {f['detail'][:100]}")
        for i in f["issues"][:2]:
            lines.append(f"          ! {i[:100]}")
        for u in f["unreadable"][:2]:
            lines.append(f"          UNREADABLE {u[:96]}")
    lines.append("")
    if not rep["files_seen"]:
        lines.append(f"NOTHING WAS EXAMINED under {rep['root']} — no files were "
                     f"opened, so nothing here is a pass.")
        return "\n".join(lines)
    lines.append(f"{rep['metadata_documents']} metadata document(s) of "
                 f"{rep['files_seen']} file(s) — {rep['by_status']}")
    if not rep["metadata_documents"]:
        for s in rep.get("skipped", [])[:5]:
            lines.append(f"        not a metadata document: {s['path']}")
    lines.append(f"declared versions: {rep['by_declared_version']}")
    if rep["validated_but_unreadable"]:
        lines.append(
            f"{rep['validated_but_unreadable']} document(s) VALIDATE AND CANNOT "
            f"BE READ. That combination is why this command exists: every check "
            f"that existed on 2026-08-17 passed on a Croissant yielding zero "
            f"rows.")
    if rep["by_status"].get(UNKNOWN):
        lines.append("UNKNOWN is not a pass — no verdict was reached for those "
                     "files. Read `detail`: a classification failure and a "
                     "validator crash need different work.")
    return "\n".join(lines)


def _selftest() -> int:
    """Two-sided, on documents built here: the sweep must PASS good metadata,
    FAIL the defect it exists for, and never call a crash a pass."""
    import shutil
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "proj"
        (root / "data").mkdir(parents=True)
        (root / "data" / "x.csv").write_text("a,b\n1,2\n3,4\n")

        good = _croissant_stub(root, "data/x.csv")
        (root / "croissant.json").write_text(json.dumps(good))
        f = check_one(root / "croissant.json")
        hit = f.validates == PASS and f.reads == PASS
        print(f"  {'PASS' if hit else 'FAIL'}  a Croissant whose data is where "
              f"it says reads  ({f.validates}/{f.reads}) {f.unreadable[:1]}")
        ok &= hit

        # THE DEFECT. Identical metadata, one directory down: still valid, and
        # a consumer gets nothing.
        (root / "results").mkdir()
        shutil.copy(root / "croissant.json", root / "results" / "croissant.json")
        f = check_one(root / "results" / "croissant.json")
        hit = f.validates == PASS and f.reads == FAIL
        print(f"  {'PASS' if hit else 'FAIL'}  the same Croissant one directory "
              f"away VALIDATES AND CANNOT BE READ  ({f.validates}/{f.reads})")
        ok &= hit

        (root / "truncated.json").write_text('{"recordSet": [')
        f = check_one(root / "truncated.json")
        hit = f.status == UNKNOWN and f.kind == NOTHING
        print(f"  {'PASS' if hit else 'FAIL'}  a truncated document is UNKNOWN, "
              f"not silently skipped  ({f.status})")
        ok &= hit

        (root / "notes.json").write_text('{"hello": "world"}')
        f = check_one(root / "notes.json")
        hit = f.status == SKIP
        print(f"  {'PASS' if hit else 'FAIL'}  ordinary JSON is SKIP, not a "
              f"pass  ({f.status})")
        ok &= hit

        rep = sweep(root)
        hit = rep["validated_but_unreadable"] == 1 and rep["files_seen"] >= 4
        print(f"  {'PASS' if hit else 'FAIL'}  the sweep counts the "
              f"validates-but-unreadable case separately "
              f"({rep['validated_but_unreadable']} of "
              f"{rep['metadata_documents']})")
        ok &= hit

        rep = sweep(Path(td) / "empty-tree")
        hit = "NOTHING WAS EXAMINED" in format_report(rep)
        print(f"  {'PASS' if hit else 'FAIL'}  an empty sweep says nothing was "
              f"examined rather than reporting health")
        ok &= hit
    print(f"\nselftest: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _croissant_stub(root: Path, rel: str) -> dict:
    """The smallest Croissant a consumer can really read, built here so that
    `--selftest` works from an installed wheel with no example tree.

    The digest is computed, not typed. `mlcroissant` verifies `sha256` at READ
    time and not at validation time — a third thing this module catches and a
    validator does not.
    """
    from claimcheck._croissant_context import CONTEXT
    from claimcheck.croissant import sha256
    return {
        "@context": {**CONTEXT, "@language": "en"},
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": "selftest", "description": "built to prove this can fail",
        "distribution": [{"@type": "cr:FileObject", "@id": "x_csv",
                          "name": "x_csv", "contentUrl": rel,
                          "encodingFormat": "text/csv",
                          "sha256": sha256(root / rel)}],
        "recordSet": [{"@type": "cr:RecordSet", "@id": "rows", "name": "rows",
                       "field": [
                           {"@type": "cr:Field", "@id": "rows/a", "name": "a",
                            "dataType": "sc:Integer",
                            "source": {"fileObject": {"@id": "x_csv"},
                                       "extract": {"column": "a"}}}]}],
    }


def main(argv: list[str] | None = None) -> int:
    a = list(argv if argv is not None else sys.argv[1:])
    if "--selftest" in a:
        return _selftest()
    as_json = "--json" in a
    if as_json:
        a.remove("--json")
    rep = sweep(Path(a[0]) if a else Path.cwd())
    print(json.dumps(rep, indent=2) if as_json else format_report(rep))
    bad = rep["by_status"].get(FAIL, 0) + rep["by_status"].get(UNKNOWN, 0)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
