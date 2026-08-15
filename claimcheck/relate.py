"""Recompute a claimed relationship, with an interval, and say whether it holds.

    python -m claimcheck.relate <project-dir> [<out.json>]

## Why an interval and not a tolerance

The first version of this compared two point estimates against a hand-picked
tolerance. On five rows it "falsified" both claims it was given — and it would
have done that to a *true* claim just as readily, because a point comparison at
n=5 has no power. A checker that cannot distinguish "the claim is wrong" from
"the data cannot tell" is not a checker; it is a coin with an opinion.

So the verdict is now **does the claimed value lie inside the interval**, and
the interval is computed from the data. A claim outside a tight interval is
refuted. A claim inside a wide one is not confirmed — it is `unverifiable`,
which is a third outcome and is **never a pass**.

## The resampling unit is declared, never assumed

Bootstrapping over rows assumes rows are independent. In a sweep they are not:
every row of one configuration shares its seeds, its library and its budget, so
resampling rows would treat 29 configurations as 870 independent observations
and report an interval roughly a fifth of its true width.

`resampling_unit` names the column to cluster on and **must be stated** when the
data has repeated structure. `provchem` will not guess it: guessing it is how a
confidence interval becomes decoration.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import tomllib
from pathlib import Path

#: Three outcomes. `unverifiable` means the check could not reach a verdict,
#: which is a different fact from the claim being wrong. Folding them together
#: is how a column of "could not run" becomes a clean bill of health.
AGREES, REFUTED, UNVERIFIABLE = "agrees", "refuted", "unverifiable"
_STATUS = {AGREES: "measured", REFUTED: "falsified", UNVERIFIABLE: "untested"}

#: Enough that the percentile endpoints are stable to ~2 decimal places, and
#: cheap enough to run on every build. Fixed seed: an interval that moves
#: between two runs of the same data is not a measurement.
N_BOOT, SEED = 2000, 20260815

#: An interval wider than this admits so much of the possible range that
#: "the claim lies inside it" says nothing. r spans [-1, 1], so a width of 1.0
#: is half of everything — an interval that broad cannot discriminate a strong
#: negative relationship from a strong positive one, and reporting `agrees`
#: against it would be the reassurance this package exists to refuse.
#:
#: FOUND BY RUNNING IT. On the five-row example the CI came out [-0.683, 1.000]
#: and both claims were reported as HOLDING — while this module's own docstring
#: already said a claim inside a wide interval "is not confirmed, it is
#: unverifiable". The prose was right and the code did not implement it, which
#: is `docs/MISTAKES.md` §4 committed inside the file that argues against it.
MAX_INFORMATIVE_WIDTH = 1.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """r, or None where it is undefined rather than zero.

    A constant column has no correlation defined on it. Returning 0.0 would be
    a number where there is no answer, and 0.0 reads as "no relationship" —
    a conclusion that has not been earned.
    """
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / math.sqrt(sxx * syy)


def bootstrap_ci(pairs: list[tuple[float, float]], clusters: list[str] | None,
                 level: float = 0.95) -> tuple[float, float, int] | None:
    """Percentile CI for r, resampling CLUSTERS when they are given.

    Returns `(lo, hi, n_effective)` where `n_effective` is the number of
    independent units actually resampled — the number a reader needs to judge
    the interval, and the one that differs from the row count by 30x here.
    """
    rng = random.Random(SEED)
    if clusters is None:
        groups = [[p] for p in pairs]
    else:
        by: dict[str, list[tuple[float, float]]] = {}
        for p, c in zip(pairs, clusters, strict=True):
            by.setdefault(c, []).append(p)
        groups = list(by.values())
    if len(groups) < 3:
        return None
    stats = []
    for _ in range(N_BOOT):
        drawn: list[tuple[float, float]] = []
        for _ in range(len(groups)):
            drawn.extend(rng.choice(groups))
        r = pearson([p[0] for p in drawn], [p[1] for p in drawn])
        if r is not None:
            stats.append(r)
    if len(stats) < N_BOOT // 2:
        return None                    # too often undefined to be an interval
    stats.sort()
    a = (1 - level) / 2
    return (stats[int(a * len(stats))],
            stats[min(int((1 - a) * len(stats)), len(stats) - 1)],
            len(groups))


def _rows(root: Path, ctx: dict) -> list[dict]:
    rows: list[dict] = []
    for rel in ctx["dataset"]["files"]:
        with (root / rel).open(newline="", encoding="utf-8-sig") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _paired(rows: list[dict], x: str, y: str, unit: str | None):
    """Rows usable for BOTH columns, kept together, with the cluster label.

    Paired deliberately: filtering each column separately would compare
    `x[i]` against `y[j]`, which is a correlation between two different
    subsets and is not a number about anything.
    """
    pairs, clusters, dropped = [], [], 0
    for r in rows:
        try:
            a, b = float(r[x]), float(r[y])
        except (ValueError, KeyError):
            dropped += 1
            continue
        if a != a or b != b or abs(a) == float("inf") or abs(b) == float("inf"):
            dropped += 1               # non-finite is not missing and not a value
            continue
        pairs.append((a, b))
        if unit:
            clusters.append(str(r.get(unit, "")))
    return pairs, (clusters if unit else None), dropped


def check(root: Path | str) -> dict:
    # COERCED, not assumed. The annotation said `Path` and the body did
    # `root / "context.toml"`, so passing the obvious thing — a string — failed
    # with `TypeError: unsupported operand type(s) for /: 'str' and 'str'`,
    # which tells the caller nothing about what they did wrong.
    root = Path(root)
    ctx = tomllib.loads((root / "context.toml").read_text())
    declared = {v["name"]: v for v in ctx.get("variables", [])}
    rows = _rows(root, ctx)
    default_unit = ctx["dataset"].get("resampling_unit")

    out = []
    for rel in ctx.get("relationships", []):
        x, y = rel["x"], rel["y"]
        unit = rel.get("resampling_unit", default_unit)
        row: dict = {
            "x": x, "y": y, "kind": rel["kind"], "method": rel["method"],
            "claimed": rel["claimed"], "evidence": rel.get("evidence", {}),
            "x_unit": declared.get(x, {}).get("unit"),
            "y_unit": declared.get(y, {}).get("unit"),
            "resampling_unit": unit,
        }

        # `base=row` binds at definition. The closure is called inside its own
        # iteration so it is correct either way today — but a late-binding
        # closure over a loop variable is a trap that costs one word to remove,
        # and the next person to hoist this out of the loop gets no warning.
        # Same defect ruff caught in `emit/rdf.py`; second occurrence.
        def verdict(v: str, why: str, base: dict = row, **extra) -> dict:
            return base | {"verdict": v, "why": why, "status": _STATUS[v],
                           **extra}

        absent = [c for c in (x, y) if not rows or c not in rows[0]]
        if absent:
            out.append(verdict(UNVERIFIABLE,
                               f"the data has no column {', '.join(absent)}"))
            continue
        undeclared = [c for c in (x, y)
                      if declared.get(c, {}).get("unit") is None]
        if undeclared:
            out.append(verdict(
                UNVERIFIABLE,
                f"no unit declared for {', '.join(undeclared)} — a relationship "
                f"between undeclared quantities cannot be compared with anyone "
                f"else's"))
            continue
        if unit and rows and unit not in rows[0]:
            out.append(verdict(
                UNVERIFIABLE,
                f"resampling unit {unit!r} is not a column, so the interval "
                f"would assume an independence the data does not have"))
            continue

        pairs, clusters, dropped = _paired(rows, x, y, unit)
        row |= {"n_rows": len(pairs), "dropped": dropped}
        r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        if r is None:
            out.append(verdict(UNVERIFIABLE,
                               f"r is undefined on {len(pairs)} usable pair(s)"))
            continue
        row["estimate"] = round(r, 6)
        ci = bootstrap_ci(pairs, clusters)
        if ci is None:
            out.append(verdict(
                UNVERIFIABLE,
                f"fewer than 3 independent units to resample"
                f"{f' over {unit}' if unit else ''}, so no interval exists"))
            continue
        lo, hi, n_eff = ci
        width = hi - lo
        row |= {"ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                "ci_level": 0.95, "n_units": n_eff, "ci_width": round(width, 6)}
        inside = rel["claimed"] is not None and lo <= rel["claimed"] <= hi
        where = (f"the 95% CI [{lo:.3f}, {hi:.3f}] for r = {r:.3f}, "
                 f"resampling {n_eff} {unit or 'row'}(s)")
        limit = rel.get("max_ci_width", MAX_INFORMATIVE_WIDTH)
        if not inside:
            # A refutation survives a wide interval: lying outside an interval
            # this generous is a stronger result, not a weaker one.
            out.append(verdict(REFUTED, f"claimed {rel['claimed']} lies "
                                        f"OUTSIDE {where}"))
        elif width > limit:
            out.append(verdict(
                UNVERIFIABLE,
                f"claimed {rel['claimed']} lies within {where} — but that "
                f"interval is {width:.3f} wide, over the {limit} at which it "
                f"stops discriminating. The data cannot tell, which is not the "
                f"same as the claim holding"))
        else:
            out.append(verdict(AGREES,
                               f"claimed {rel['claimed']} lies within {where}"))

    # `res`, not `r` — `r` is the correlation coefficient forty lines up, and
    # binding one name to a float and then to a result row in one function is
    # how a reader (and a type checker) loses track of which is which.
    verdicts: dict[str, int] = {}
    for res in out:
        verdicts[res["verdict"]] = verdicts.get(res["verdict"], 0) + 1
    return {"dataset": ctx["dataset"]["name"], "rows_read": len(rows),
            "relationships": out, "verdicts": verdicts,
            "any_refuted": any(res["verdict"] == REFUTED for res in out),
            # UNVERIFIABLE IS NOT A PASS, so it is reported at the top level
            # rather than left to be counted out of the rows.
            "any_unverifiable": any(res["verdict"] == UNVERIFIABLE
                                    for res in out)}


def main(argv: list[str] | None = None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    root = Path(a[0]) if a else Path.cwd()
    out = Path(a[1]) if len(a) > 1 else root / "results/claims.json"
    d = check(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2) + "\n")
    mark = {AGREES: "HOLDS  ", REFUTED: "REFUTED", UNVERIFIABLE: "?      "}
    for r in d["relationships"]:
        print(f"  {mark[r['verdict']]} {r['x']} ~ {r['y']}   claimed "
              f"{r['claimed']}  ->  {r['why']}")
        print(f"          {r['evidence'].get('source', 'no source stated')} "
              f"[{r['evidence'].get('extracted_by', 'unknown')}]"
              f"  -> {r['status']}")
    print(f"  {d['verdicts']} over {d['rows_read']} row(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
