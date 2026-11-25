"""One place to ask for an output format, and the seam a new one plugs into.

    python -m claimcheck.emit <format> <project-dir> [<out>]
    python -m claimcheck.emit --list

## Why a registry for three things

Because the alternative is that adding a fourth format means editing the CLI,
the Snakefile template, the README and the tests — and because a format
somebody else needs should not require forking this. An emitter is a function
`build(root) -> str`, registered by name. Third-party ones are discovered
through the `claimcheck.emitters` entry-point group, so a separate package can
add one and it appears in `--list` with no change here.

## An unavailable format says why

`graph` and `dcat` need `rdflib`, `psdi` needs `pyshacl` on top of it, and
`rocrate` needs the reference implementation — all extras. Asking for one
without it must not be an `ImportError` traceback from three frames down:
`--list` marks it unavailable with the install command, and `emit` refuses with
the same sentence. A missing optional dependency is a fact about the
installation, not a crash.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

#: name -> (module path, the extra it needs, one-line description).
#: Built-ins are declared rather than imported, so listing the formats never
#: pays the cost of importing a graph library nobody asked for.
_BUILTIN = {
    "croissant": ("claimcheck.croissant", None,
                  "MLCommons Croissant 1.1 — how to READ the dataset"),
    "belief": ("claimcheck.belief", None,
               "what rests on what, and what a refutation puts in doubt"),
    "jobs": ("claimcheck.jobs", "graph",
             "PROV-O: why a run was launched, and what would have counted"),
    "graph": ("claimcheck.graph", "graph",
              "the claims and columns as RDF, joined to the digests"),
    "evi": ("claimcheck.evi", "graph",
            "EVI's two propagation axioms, for a reasoner with no network"),
    "dcat": ("claimcheck.dcat", "graph",
             "W3C DCAT — how to FIND the dataset, for a catalogue"),
    "psdi": ("claimcheck.psdi", "psdi",
             "DCAT in PSDI's profile, checked against PSDI's own SHACL"),
    "rocrate": ("claimcheck.rocrate", "rocrate",
                "RO-Crate 1.1 — the whole directory, named and hashed"),
}

#: module -> the extra whose packages it may import, for modules that are NOT
#: formats. `_BUILTIN` is the registry of things `emit` can render, and every
#: entry in it must have a `render(root)`; a module that CONSUMES an extra's
#: package without emitting a format has no entry there and still has to be
#: allowed to import it. `tests/test_dependencies.py` reads both.
#:
#: `claimcheck.corpus` merges many projects' graphs into one store, so it needs
#: rdflib for the same reason `graph` does, and `render(root)` means nothing for
#: it: a format writes one project's file and this one reads hundreds.
_EXTRA_CONSUMERS = {
    "claimcheck.corpus": "graph",
}

#: extra -> the module whose presence proves the extra is installed. Declared
#: rather than inferred: `psdi` needs `pyshacl` AND `rdflib`, and `find_spec` on
#: `claimcheck.psdi` would say yes on an install that has neither, because our
#: own module is always importable.
_EXTRA_PROBE = {
    "graph": ("rdflib",),
    "psdi": ("rdflib", "pyshacl"),
    "rocrate": ("rocrate", "rocrate_validator"),
}

ENTRY_POINT_GROUP = "claimcheck.emitters"


class EmitError(RuntimeError):
    """Asked for a format that is not installed, or does not exist."""


def _third_party() -> dict[str, Callable]:
    from importlib.metadata import entry_points
    return {ep.name: ep.load() for ep in entry_points(group=ENTRY_POINT_GROUP)}


def available() -> dict[str, str | None]:
    """format -> None if usable, or the reason it is not.

    Never raises. Listing what you could do must work when some of it is
    uninstalled, or the list is only useful to people who need it least.
    """
    from importlib.util import find_spec
    out: dict[str, str | None] = {}
    for name, (mod, extra, _) in _BUILTIN.items():
        try:
            ok = find_spec(mod.rsplit(".", 1)[0]) is not None
            for probe in _EXTRA_PROBE.get(extra or "", ()):
                ok = ok and find_spec(probe) is not None
        except (ImportError, ValueError):
            ok = False
        out[name] = None if ok else (
            f"needs the '{extra}' extra — pip install 'claimcheck[{extra}]'")
    for name in _third_party():
        out[name] = None
    return out


def get(name: str) -> Callable[[Path], str]:
    """The build function for one format, or a refusal that says what to do."""
    third = _third_party()
    if name in third:
        return third[name]
    if name not in _BUILTIN:
        known = ", ".join(sorted(set(_BUILTIN) | set(third)))
        raise EmitError(f"no format named {name!r}. Known: {known}")
    why = available()[name]
    if why:
        raise EmitError(f"the {name!r} format {why}")
    mod = __import__(_BUILTIN[name][0], fromlist=["render"])
    if not hasattr(mod, "render"):
        raise EmitError(f"{_BUILTIN[name][0]} defines no `render(root)`")
    return mod.render


def emit(name: str, root: Path | str) -> str:
    return get(name)(Path(root))


def main(argv: list[str] | None = None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    if not a or a[0] in ("--list", "-l"):
        for name, why in sorted(available().items()):
            desc = _BUILTIN.get(name, (None, None, "third-party"))[2]
            mark = "  " if why is None else "! "
            print(f"{mark}{name:12} {desc}")
            if why:
                print(f"               {why}")
        return 0
    name, root = a[0], Path(a[1]) if len(a) > 1 else Path.cwd()
    try:
        text = emit(name, root)
    except EmitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if len(a) > 2:
        out = Path(a[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"  {out}  —  {len(text.splitlines())} lines of {name}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------- project table


def project_table(ctx: dict) -> dict:
    """`ctx["project"]`, with the British spelling of `licence` accepted.

    Every emitter here reads `license`, the US spelling, because that is the key
    Croissant, DCAT and RO-Crate all use. A project that writes `licence` was
    silently publishing NO licence at all: the declaration sat in context.toml,
    nothing read it, and the package went out unlicensed. Found 2026-11-13 in 13
    experiments at once, which is what a silent failure looks like.

    So the alias is accepted and a warning is printed naming the file to fix. It
    is accepted rather than merely warned about because the intent is
    unambiguous and the alternative is publishing an unlicensed package, which
    is the worse of the two failures. Declaring BOTH with different values is an
    error, not a merge: nothing here may choose which licence a project meant.
    """
    proj = dict(ctx.get("project") or {})
    uk = proj.pop("licence", None)
    if uk is None:
        return proj
    us = proj.get("license")
    if us is not None and str(us).strip() != str(uk).strip():
        raise ValueError(
            f"context.toml declares license = {us!r} and licence = {uk!r}. "
            "They differ, and nothing here may choose between them. Keep one.")
    if us is None:
        print(f"warning: context.toml declares `licence`; the emitters read "
              f"`license`. Using {uk!r}, but rename the key: every other reader "
              "of this file, and every tool downstream, uses the US spelling.",
              file=sys.stderr)
        proj["license"] = uk
    return proj
