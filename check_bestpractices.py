#!/usr/bin/env python
"""Run every published best-practice checklist against this project.

    python check_bestpractices.py                    # everything
    python check_bestpractices.py --paper uncertainty
    python check_bestpractices.py --severity must    # hard requirements only
    python check_bestpractices.py --list             # what is encoded
    python check_bestpractices.py --selftest         # prove the runner detects

**Why this exists.** Our adversarial reviewer should measure us against a
**published standard**, not against its own judgement. `bestpractices/` encodes
the explicit checklists of the LiveCoMS best-practices literature as executable
code; this runs them.

**Exit codes:** 0 if no `must` check FAILed, 1 otherwise. `should` and
`consider` failures are reported and do not gate — they are advice, and
conflating advice with a requirement is how a checklist stops being read.

**`UNTESTABLE` is never counted as a pass.** A check that cannot fail has not
checked anything; this project has already shipped one vacuous PASS.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bestpractices import Context, FAIL, NA, PASS, UNTESTABLE  # noqa: E402

SEV_ORDER = {"must": 0, "should": 1, "consider": 2}
MARK = {PASS: "  ok  ", FAIL: " FAIL ", UNTESTABLE: " untst", NA: "  n/a "}


def load_modules(only=None):
    import bestpractices
    mods = []
    for m in pkgutil.iter_modules(bestpractices.__path__):
        if m.name.startswith("_"):
            continue
        if only and m.name != only:
            continue
        mod = importlib.import_module(f"bestpractices.{m.name}")
        if not hasattr(mod, "PAPER") or not hasattr(mod, "CHECKS"):
            print(f"  ! bestpractices/{m.name}.py has no PAPER/CHECKS — skipped")
            continue
        mods.append(mod)
    return sorted(mods, key=lambda m: m.PAPER.slug)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper")
    p.add_argument("--root", type=Path, default=None,
                   help="Root directory of the project to check (default: current working directory)")
    p.add_argument("--severity", choices=("must", "should", "consider"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def selftest():
    """The runner must actually distinguish the four outcomes. A runner that
    silently turns FAIL into PASS is worse than no runner."""
    from bestpractices import Check, Result
    ok = True
    cases = [(PASS, 0), (FAIL, 1), (UNTESTABLE, 0), (NA, 0)]
    for status, want_gate in cases:
        c = Check(id="t", question="q", severity="must", section="§0",
                  fn=lambda ctx, s=status: Result(s, "synthetic"))
        r = c.fn(Context())
        gated = 1 if (r.status == FAIL and c.severity == "must") else 0
        good = r.status == status and gated == want_gate
        print(f"  {status:<10} -> gates={gated} {'ok' if good else 'BROKEN'}")
        ok &= good
    try:
        Result(NA, "")
        print("  N/A-without-reason was accepted — BROKEN"); ok = False
    except ValueError:
        print("  N/A without a reason correctly rejected")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    args = parse_args()
    if args.selftest:
        sys.exit(selftest())

    mods = load_modules(args.paper)
    if not mods:
        print("No checklist modules found in bestpractices/.")
        print("Each must define PAPER (a Paper) and CHECKS (a list of Check).")
        sys.exit(0)

    if args.list:
        for m in mods:
            p = m.PAPER
            n = {s: sum(1 for c in m.CHECKS if c.severity == s)
                 for s in SEV_ORDER}
            print(f"\n{p.slug} — {p.title}")
            print(f"  {p.authors} · {p.url}")
            print(f"  scope: {p.scope}")
            print(f"  {len(m.CHECKS)} checks "
                  f"({n['must']} must, {n['should']} should, {n['consider']} consider)")
        sys.exit(0)

    root = args.root.resolve() if args.root else Path.cwd().resolve()
    ctx = Context(root=root)
    tally = {PASS: 0, FAIL: 0, UNTESTABLE: 0, NA: 0}
    must_failed = []

    for m in mods:
        p = m.PAPER
        print(f"\n{'='*78}\n{p.slug} — {p.title}\n{p.url}\n{'='*78}")
        checks = sorted(m.CHECKS, key=lambda c: (SEV_ORDER[c.severity], c.id))
        for c in checks:
            if args.severity and c.severity != args.severity:
                continue
            try:
                r = c.fn(ctx)
            except Exception as exc:                        # noqa: BLE001
                from bestpractices import Result
                r = Result(UNTESTABLE, f"check raised {type(exc).__name__}: "
                                       f"{str(exc)[:70]}")
            tally[r.status] += 1
            if r.status == FAIL and c.severity == "must":
                must_failed.append((p.slug, c))
            print(f" [{MARK[r.status]}] {c.severity:<8} {c.id:<10} {c.question}")
            if r.detail:
                print(f"            -> {r.detail}")
            if args.verbose and r.evidence:
                print(f"            evidence: {r.evidence}")
            print(f"            {c.section}")

    total = sum(tally.values())
    print(f"\n{'='*78}\nSUMMARY over {len(mods)} paper(s), {total} checks")
    print(f"  PASS {tally[PASS]} · FAIL {tally[FAIL]} · "
          f"UNTESTABLE {tally[UNTESTABLE]} · N/A {tally[NA]}")
    if tally[UNTESTABLE]:
        print(f"  NOTE: {tally[UNTESTABLE]} untestable — these are NOT passes.")
    if must_failed:
        print(f"\n  {len(must_failed)} MUST-level failure(s):")
        for slug, c in must_failed:
            print(f"    {slug}/{c.id} ({c.section}) — {c.question}")
        sys.exit(1)
    print("\n  No must-level failures.")
    sys.exit(0)


if __name__ == "__main__":
    main()
