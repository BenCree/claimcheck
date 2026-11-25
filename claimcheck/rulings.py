"""Decisions already made — the register, and the check that nothing re-asks one.

    python -m claimcheck.rulings <register.toml>                     # list them
    python -m claimcheck.rulings <register.toml> --open <open.toml>  # check
    python -m claimcheck.rulings <register.toml> --open <open.toml> --check

A claim is a prediction about data and `relate` adjudicates it. A **ruling** is a
different thing: a decision a person made, which no amount of data will settle,
and which stays made. Which protonation tool is standard. What licence the inputs
carry. Whether a pocket may be docked at all. These are not claims, they have no
interval and no verdict, and before this module they lived wherever the
conversation happened to leave them.

**The failure this exists to stop.** An answered question is written down in one
place, and a later question in another place asks it again. Nobody notices,
because nothing compares the two, so the person who already decided is asked to
decide again. Measured on one project, 2026-11-13: of the decisions standing on
the open list, two had already been ruled on by the project owner the same day,
and one of those had the owner's own words recorded one file away. The register
is the single place, and `--check` is what makes it get read, because a list
nobody consults is not a register.

## The register

    [[rulings]]
    id = "R-protonation-tool"
    date = "2026-11-13"                 # when it was decided, not when written
    question = "Which protonation tool is the standing choice?"
    ruling = "dimorphite-dl, at pH 7.4."
    quote = "it stays, at pH 7.4"       # their own words, verbatim, where there are any
    decided_by = "project owner"
    answers = ["D-protonation", "D-protonation-tool"]   # open-question ids it settles
    status = "standing"                 # or "superseded"
    superseded_by = "R-protonation-tool-2"
    note = "Why, and what it does not cover."

## What the check does, and what it deliberately will not do

Two kinds of duplicate, and they are not treated alike.

**An exact duplicate blocks.** An open question whose id is named in a standing
ruling's `answers`, or which carries `settled_by = "<ruling id>"`, is a question
that is already answered. `--check` exits 1 and names both. This is exact, so it
is safe to enforce.

**A reworded duplicate is reported and never enforced.** The same question asked
in different words cannot be recognised reliably, and a checker that guesses
would either miss them or block real work. So overlap in the question text raises
a WARNING naming the ruling it resembles, and the exit status does not change.
A warning is for a person to read, which is the honest limit of what text
comparison buys. The threshold is `SIMILAR`, and it is our choice with no source:
it was set by hand so that the two real duplicates on the project it was built
for are both named.

A superseded ruling never blocks anything: that is what superseding means.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

#: Share of the shorter question's words that must also appear in the other
#: before the pair is called similar. Our choice, no source. Reported only.
SIMILAR = 0.6

#: Words carried by almost every question here, so they say nothing about
#: whether two questions are the same one.
STOPWORDS = frozenset("""
a an and are as at be been but by can do does for from had has have how if in
into is it its not of on or our should so than that the their then there these
this to was we were what when where which who why will with would
""".split())


class RegisterError(ValueError):
    """The register itself is malformed, which is never a soft failure."""


def _words(text):
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in STOPWORDS and len(w) > 2}


def similarity(a: str, b: str) -> float:
    """Share of the shorter question's meaningful words found in the longer.

    Deliberately crude. It exists to raise a flag for a person, never to decide
    anything, so a cleverer measure would buy precision the result does not use.
    """
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def load(path) -> list[dict]:
    """Read the register. A missing file is an error, not an empty register:
    'no rulings' and 'the register is not where you said' must not look alike."""
    path = Path(path)
    if not path.is_file():
        raise RegisterError(f"no register at {path}")
    try:
        doc = tomllib.load(path.open("rb"))
    except tomllib.TOMLDecodeError as exc:
        raise RegisterError(f"{path}: {exc}") from exc
    rulings = doc.get("rulings")
    if not isinstance(rulings, list):
        raise RegisterError(f"{path}: no [[rulings]] blocks")
    seen = {}
    for i, r in enumerate(rulings):
        rid = r.get("id")
        if not rid:
            raise RegisterError(f"{path}: ruling {i} has no id")
        if rid in seen:
            raise RegisterError(f"{path}: two rulings share the id {rid!r}")
        seen[rid] = True
        for key in ("question", "ruling", "date"):
            if not r.get(key):
                raise RegisterError(f"{path}: ruling {rid!r} has no {key}")
        status = r.get("status", "standing")
        if status not in ("standing", "superseded"):
            raise RegisterError(
                f"{path}: ruling {rid!r} has status {status!r}, "
                "which is neither 'standing' nor 'superseded'")
        if status == "superseded" and not r.get("superseded_by"):
            raise RegisterError(
                f"{path}: ruling {rid!r} is superseded by nothing named")
    for r in rulings:
        by = r.get("superseded_by")
        if by and by not in seen:
            raise RegisterError(
                f"{path}: ruling {r['id']!r} names superseded_by={by!r}, "
                "which is not in the register")
    return rulings


def standing(rulings) -> list[dict]:
    return [r for r in rulings if r.get("status", "standing") == "standing"]


def load_open(path) -> list[dict]:
    """Open questions to check. Any TOML with a list of tables carrying `id`
    and `question`; the key is `decisions` or `questions`."""
    path = Path(path)
    if not path.is_file():
        raise RegisterError(f"no open-questions file at {path}")
    doc = tomllib.load(path.open("rb"))
    for key in ("decisions", "questions", "open"):
        if isinstance(doc.get(key), list):
            return [d for d in doc[key] if isinstance(d, dict)]
    return []


def check(rulings, open_questions):
    """-> (settled, similar). `settled` blocks; `similar` is for a person."""
    live = standing(rulings)
    by_id = {r["id"]: r for r in live}
    answers = {}
    for r in live:
        for qid in r.get("answers", []) or []:
            answers.setdefault(qid, r)

    settled, similar = [], []
    for q in open_questions:
        qid, text = q.get("id", ""), q.get("question", "")
        hit = answers.get(qid)
        named = q.get("settled_by")
        if hit is None and named and named in by_id:
            hit = by_id[named]
        if hit is not None:
            settled.append((q, hit))
            continue
        for r in live:
            s = similarity(text, r.get("question", ""))
            if s >= SIMILAR:
                similar.append((q, r, s))
                break
    return settled, similar


def report(rulings, open_questions, out=None):
    # Resolved at call time, never bound at import: a default of
    # sys.stdout captures whatever stream existed when this module was
    # imported and writes past any later redirect.
    out = sys.stdout if out is None else out
    settled, similar = check(rulings, open_questions)
    for q, r in settled:
        print(f"SETTLED  {q.get('id','?')}: {q.get('question','')}", file=out)
        print(f"         already ruled {r['date']} by {r.get('decided_by','?')}: "
              f"{r['ruling']}", file=out)
        if r.get("quote"):
            print(f"         their words: \"{r['quote']}\"", file=out)
        print(f"         ruling {r['id']}. Strike the question or say what is "
              "distinct about it.", file=out)
    for q, r, s in similar:
        print(f"WARNING  {q.get('id','?')} resembles ruling {r['id']} "
              f"({s:.2f} of its words): {r['ruling']}", file=out)
        print("         Not enforced. Read both and decide whether it is the "
              "same question.", file=out)
    if not settled and not similar:
        print(f"ok: {len(open_questions)} open question(s), none already ruled on",
              file=out)
    return settled, similar


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="The register of decisions already made, and the check that "
                    "no open question re-asks one.")
    ap.add_argument("register", help="TOML file of [[rulings]]")
    ap.add_argument("--open", dest="open_path",
                    help="TOML file of open [[decisions]] to check against it")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if an open question is already ruled on")
    args = ap.parse_args(argv)

    try:
        rulings = load(args.register)
    except RegisterError as exc:
        print(f"register error: {exc}", file=sys.stderr)
        return 2

    if not args.open_path:
        live = standing(rulings)
        print(f"{len(live)} standing ruling(s), "
              f"{len(rulings) - len(live)} superseded, in {args.register}")
        for r in sorted(live, key=lambda r: r["date"], reverse=True):
            print(f"\n{r['id']}  ({r['date']}, {r.get('decided_by','?')})")
            print(f"  Q: {r['question']}")
            print(f"  A: {r['ruling']}")
            if r.get("quote"):
                print(f"  \"{r['quote']}\"")
            if r.get("answers"):
                print(f"  settles: {', '.join(r['answers'])}")
        return 0

    try:
        open_questions = load_open(args.open_path)
    except RegisterError as exc:
        print(f"open-questions error: {exc}", file=sys.stderr)
        return 2
    settled, _ = report(rulings, open_questions)
    return 1 if (settled and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
