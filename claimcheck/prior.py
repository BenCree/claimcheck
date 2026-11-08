"""Has somebody already declared this? Search prior projects before writing a new one.

    python -m claimcheck.prior <tree> "water A/B, UMA, our own pocket cut"
    python -m claimcheck.prior <tree> --like <proposed-context.toml>
    python -m claimcheck.prior <tree> --json "gcmc occupancy"
    python -m claimcheck.prior --selftest

    from claimcheck.prior import search
    hits = search(Path("datasets/claimcheck"), "uma charge")

## Why this module exists

A project's declaration is the only place its question is written down in full,
and declarations accumulate faster than anyone re-reads them. The measured case
that produced this module: a request for "miel A/B plots with a different model
and our own starting structures" was already answered by `pla15_crosscheck`,
which declares `mace_asis_kcal_mol`, `mace_cut6_kcal_mol`, `uma_charged_kcal_mol`
and `uma_neutral_kcal_mol`, carries two adjudicated figures, and had already
settled that the 6 A cut is harmless at r = 0.989 [0.916, 1.000] over 15 systems.
Re-running it would have cost GPU hours to reproduce a verdict already on disk.

**The check is cheap and the duplicate is expensive, so the check comes first.**
This reads `context.toml` files and nothing else. It never opens a dataset, runs
a rule, or touches the network.

## What it reports, and why each field

For every project whose declaration overlaps the query it prints:

* **which terms matched and where** (title, context prose, a dataset, a variable
  name or description, a relationship id or its evidence). A match on a variable
  NAME is the strong signal, because that is an arm somebody already computed; a
  match on context prose is weak and often just shared vocabulary.
* **the state of the project**, which decides what to do with the hit:
  `adjudicated` (a `results/claims.json` exists, so there are verdicts to read),
  `declared` (claims registered, never run: the work is designed and pending),
  `undeclared` (a Snakefile with no `context.toml`, which computes and claims
  nothing), and whether any figure exists.

A hit that is `adjudicated` means the question is answered and you should read
the verdict. A hit that is `declared` means somebody already wrote the claim and
running it is cheaper than writing a second one. Those are different actions,
which is why the state is printed beside every hit rather than left to be looked
up.

## What it deliberately does not do

It does not decide whether your question is the same as theirs. Two projects can
share every keyword and ask different things, and no string match settles that.
It surfaces the candidates and the reader adjudicates, which is why the matched
FIELD is printed rather than only a score.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import tomllib  # standard library from Python 3.11, the floor pyproject.toml declares

# Words that match everything in a scientific repository and so carry no signal.
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "is", "it", "this", "that", "from", "by", "as", "be", "are", "was", "were",
    "one", "two", "per", "each", "every", "any", "all", "not", "no", "than",
    "data", "value", "values", "result", "results", "run", "runs", "project",
    "claim", "claims", "test", "number", "numbers", "set", "sets", "using",
    "our", "we", "its", "their", "which", "what", "how", "why", "if", "then",
}
STRONG = {"variable", "dataset", "relationship"}   # a match here is an arm, not prose


def _terms(text: str) -> list[str]:
    """Lower-case alphanumeric tokens of length 2 or more, minus the stop list."""
    return [t for t in re.findall(r"[a-z0-9_]{2,}", text.lower()) if t not in STOP]


def _fields(doc: dict) -> list[tuple[str, str]]:
    """Every searchable (field-kind, text) pair in one declaration."""
    out: list[tuple[str, str]] = []
    proj = doc.get("project", {})
    for k in ("name", "title", "context"):
        if proj.get(k):
            out.append(("title" if k != "context" else "context", str(proj[k])))
    for d in doc.get("datasets", []) or []:
        out.append(("dataset", f"{d.get('id', '')} {d.get('title', '')}"))
    for v in doc.get("variables", []) or []:
        out.append(("variable", str(v.get("name", ""))))
        if v.get("description"):
            out.append(("variable_description", str(v["description"])))
    for r in doc.get("relationships", []) or []:
        out.append(("relationship", f"{r.get('id', '')} {r.get('x', '')} {r.get('y', '')}"))
        ev = r.get("evidence", {})
        if isinstance(ev, dict) and ev.get("source"):
            out.append(("evidence", str(ev["source"])))
    return out


def _state(proj_dir: Path) -> dict:
    """What has actually happened in this project, which decides what a hit means."""
    claims = proj_dir / "results" / "claims.json"
    figs = sorted((proj_dir / "figures").glob("*.png")) if (proj_dir / "figures").is_dir() else []
    verdicts: dict[str, int] = {}
    if claims.is_file():
        try:
            d = json.loads(claims.read_text())
            rows = d if isinstance(d, list) else d.get("relationships", d.get("claims", []))
            for r in rows:
                if isinstance(r, dict) and r.get("verdict"):
                    verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        except Exception:                                    # noqa: BLE001
            pass
    if claims.is_file():
        state = "adjudicated"
    elif (proj_dir / "context.toml").is_file():
        state = "declared"
    else:
        state = "undeclared"
    return {"state": state, "verdicts": verdicts, "n_figures": len(figs),
            "has_snakefile": (proj_dir / "Snakefile").is_file()}


def search(tree: Path, query: str, limit: int = 12) -> list[dict]:
    """Rank sibling projects by declaration overlap with `query`. Read-only."""
    want = set(_terms(query))
    hits: list[dict] = []
    for toml in sorted(tree.glob("*/context.toml")):
        try:
            doc = tomllib.loads(toml.read_text())
        except Exception as exc:                             # noqa: BLE001
            hits.append({"project": toml.parent.name, "score": 0.0,
                         "unreadable": f"{type(exc).__name__}: {exc}"})
            continue
        matched: dict[str, set[str]] = {}
        for kind, text in _fields(doc):
            common = want & set(_terms(text))
            if common:
                matched.setdefault(kind, set()).update(common)
        if not matched:
            continue
        # A variable name is an arm somebody computed; prose is shared vocabulary.
        score = sum(len(v) * (3.0 if k in STRONG else 1.0) for k, v in matched.items())
        hits.append({"project": toml.parent.name, "score": score,
                     "matched": {k: sorted(v) for k, v in matched.items()},
                     "title": (doc.get("project", {}) or {}).get("title", ""),
                     **_state(toml.parent)})
    # A Snakefile with no context.toml computes and declares nothing. Those are
    # invisible to a declaration search by construction, so name them separately
    # rather than letting the search imply they do not exist.
    for snake in sorted(tree.glob("*/Snakefile")):
        if not (snake.parent / "context.toml").is_file():
            hits.append({"project": snake.parent.name, "score": 0.0,
                         "matched": {}, "title": "(no context.toml)",
                         **_state(snake.parent)})
    hits.sort(key=lambda h: -h.get("score", 0.0))
    return hits[:limit] if limit else hits


def _render(hits: list[dict], query: str) -> str:
    out = [f'prior work matching "{query}"', ""]
    if not any(h.get("score", 0) > 0 for h in hits):
        out.append("  NOTHING MATCHED. That is a result: write the project.")
    for h in hits:
        if h.get("unreadable"):
            out.append(f"  {h['project']:<34} UNREADABLE  {h['unreadable']}")
            continue
        if h.get("score", 0) <= 0 and h.get("state") != "undeclared":
            continue
        v = ", ".join(f"{k} {n}" for k, n in sorted(h.get("verdicts", {}).items()))
        tail = (f"{h['state']}"
                + (f" [{v}]" if v else "")
                + (f", {h['n_figures']} figure(s)" if h.get("n_figures") else ""))
        out.append(f"  {h['project']:<34} score {h.get('score', 0):>5.1f}  {tail}")
        if h.get("title"):
            out.append(f"      {h['title'][:104]}")
        for kind, terms in sorted(h.get("matched", {}).items()):
            mark = "**" if kind in STRONG else "  "
            out.append(f"      {mark} {kind:<22} {', '.join(terms[:10])}")
    out += ["", "A match on a variable or relationship is an arm somebody already",
            "computed. A match on context prose is shared vocabulary and weak.",
            "`adjudicated` means read the verdict; `declared` means the claim is",
            "written and running it beats writing a second one."]
    return "\n".join(out)


def selftest() -> int:
    """Two-sided: a query that must match and one that must not.

    A one-sided test passes on a function that returns every project every time,
    which is the failure mode this module is most likely to have.
    """
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tree = Path(td)
        a = tree / "pla15_crosscheck"
        (a / "results").mkdir(parents=True)
        (a / "context.toml").write_text(
            '[project]\nname = "pla15_crosscheck"\ntitle = "Our scorers on their inputs"\n'
            '[[variables]]\ndataset = "arms"\nname = "uma_charged_kcal_mol"\n'
            'unit = "kcal/mol"\ndescription = "UMA with real formal charges"\n')
        (a / "results" / "claims.json").write_text('[{"id": "P1", "verdict": "agrees"}]')
        b = tree / "unrelated_torsions"
        b.mkdir()
        (b / "context.toml").write_text(
            '[project]\nname = "unrelated_torsions"\ntitle = "Torsion barriers"\n')

        hit = search(tree, "uma charged")
        got = [h["project"] for h in hit if h.get("score", 0) > 0]
        print(f"  known-good query 'uma charged' matched {got} (want pla15_crosscheck)")
        ok &= got == ["pla15_crosscheck"]

        miss = search(tree, "ribosome cryoem")
        got2 = [h["project"] for h in miss if h.get("score", 0) > 0]
        print(f"  known-bad query 'ribosome cryoem' matched {got2} (want none)")
        ok &= got2 == []

        stopped = search(tree, "the and of our")
        got3 = [h["project"] for h in stopped if h.get("score", 0) > 0]
        print(f"  stop-words-only query matched {got3} (want none)")
        ok &= got3 == []

        st = [h for h in hit if h["project"] == "pla15_crosscheck"][0]
        print(f"  state read back: {st['state']} (want adjudicated)")
        ok &= st["state"] == "adjudicated"

        c = tree / "computes_nothing_declared"
        c.mkdir()
        (c / "Snakefile").write_text("rule all:\n    input: []\n")
        named = [h["project"] for h in search(tree, "uma charged")]
        print(f"  a Snakefile with no context.toml is still named: "
              f"{'computes_nothing_declared' in named}")
        ok &= "computes_nothing_declared" in named

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Has somebody already declared this? Search sibling "
                    "projects' context.toml before writing a new one.")
    ap.add_argument("tree", nargs="?", default="datasets/claimcheck",
                    help="directory holding the projects (default: datasets/claimcheck)")
    ap.add_argument("query", nargs="*", help="words describing the intended experiment")
    ap.add_argument("--like", type=Path,
                    help="read the query out of a proposed context.toml instead")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    if a.like:
        doc = tomllib.loads(a.like.read_text())
        query = " ".join(t for _, text in _fields(doc) for t in _terms(text))
    else:
        query = " ".join(a.query)
    if not query.strip():
        ap.error("give a query, or --like a proposed context.toml")

    hits = search(Path(a.tree), query, a.limit)
    if a.json:
        print(json.dumps(hits, indent=2))
    else:
        print(_render(hits, query if not a.like else f"--like {a.like}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
