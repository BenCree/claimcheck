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
data has repeated structure. `claimcheck` will not guess it: guessing it is how a
confidence interval becomes decoration.
"""

from __future__ import annotations

import json
import math
import random
import sys
import tomllib
from pathlib import Path

from claimcheck.datasets import DatasetError, join
from claimcheck.datasets import load as load_dataset

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


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Pearson on ranks: monotonic association without assuming a straight line.

    Worth having in chemistry specifically. Binding energies against measured
    affinity are routinely monotonic and curved, and Pearson understates that
    while Spearman does not — so "which estimator" is a scientific choice and
    belongs in `context.toml`, not in whichever one the code happens to call.
    """
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            # TIES TAKE THE AVERAGE RANK. Assigning them arbitrary distinct
            # ranks invents an ordering the data does not have, and on data with
            # many repeated values that is most of the correlation.
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    return pearson(rank(xs), rank(ys))


#: THE EXTENSION POINT FOR STATISTICS. `method` was written into every output
#: and never dispatched on, so `method = "spearman"` computed Pearson and
#: labelled it Spearman — a silent wrong answer of exactly the kind this package
#: exists to refuse, in its own core.
#:
#: Adding one is a function and an entry here. Each declares the `kind` it
#: implements so a mismatched pair is refused rather than quietly ignored.
ESTIMATORS: dict[str, tuple] = {
    "pearson": (pearson, "correlation"),
    "spearman": (spearman, "correlation"),
}


def bootstrap_ci(pairs: list[tuple[float, float]], clusters: list[str] | None,
                 level: float = 0.95, estimate=pearson
                 ) -> tuple[float, float, int] | None:
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
        r = estimate([p[0] for p in drawn], [p[1] for p in drawn])
        if r is not None:
            stats.append(r)
    if len(stats) < N_BOOT // 2:
        return None                    # too often undefined to be an interval
    stats.sort()
    a = (1 - level) / 2
    return (stats[int(a * len(stats))],
            stats[min(int((1 - a) * len(stats)), len(stats) - 1)],
            len(groups))


def _ref(text: str, only: str | None) -> tuple[str, str]:
    """`"affinity.pKD"` -> `("affinity", "pKD")`.

    A bare name is allowed only when the project has ONE dataset. With two, a
    bare column name is ambiguous and the ambiguity would be resolved by
    whichever happened to be first — so it is refused instead.
    """
    if "." in text:
        did, _, col = text.partition(".")
        return did, col
    if only is None:
        raise DatasetError(
            f"{text!r} does not say which dataset it is in. With more than one "
            f"dataset a bare column name is ambiguous; write `dataset.column`.")
    return only, text


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

    # Every dataset is loaded and verified BEFORE any claim is checked. A
    # manually imported file whose digest has moved stops the run here, rather
    # than producing verdicts about bytes nobody meant to ship.
    sets = {d["id"]: load_dataset(root, d) for d in ctx["datasets"]}
    only = next(iter(sets)) if len(sets) == 1 else None
    # A LEGIBLE REFUSAL, not a KeyError. A variable with no `dataset` used to
    # raise `KeyError: 'dataset'` from inside a comprehension, which tells the
    # person editing a TOML file nothing about which block is wrong.
    declared = {}
    for v in ctx.get("variables", []):
        if "dataset" not in v:
            raise DatasetError(
                f"variable {v.get('name', '?')!r} does not say which dataset it "
                f"belongs to. Add `dataset = \"<id>\"` to its block; with more "
                f"than one dataset the name alone is ambiguous.")
        if v["dataset"] not in sets:
            raise DatasetError(
                f"variable {v['name']!r} names dataset {v['dataset']!r}, which "
                f"is not declared. Known: {', '.join(sorted(sets))}")
        declared[(v["dataset"], v["name"])] = v

    out, joins = [], []
    for rel in ctx.get("relationships", []):
        xd, x = _ref(rel["x"], only)
        yd, y = _ref(rel["y"], only)
        unit = rel.get("resampling_unit") or sets[xd]["spec"].get("resampling_unit")
        row: dict = {
            "x": f"{xd}.{x}", "y": f"{yd}.{y}",
            "x_dataset": xd, "y_dataset": yd,
            "cross_dataset": xd != yd,
            "kind": rel["kind"], "method": rel["method"],
            "claimed": rel["claimed"], "evidence": rel.get("evidence", {}),
            "x_unit": declared.get((xd, x), {}).get("unit"),
            "y_unit": declared.get((yd, y), {}).get("unit"),
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

        # THE DECLARED METHOD IS THE ONE THAT RUNS, or the run stops.
        if rel["method"] not in ESTIMATORS:
            out.append(verdict(
                UNVERIFIABLE,
                f"unknown method {rel['method']!r}. Known: "
                f"{', '.join(sorted(ESTIMATORS))}"))
            continue
        estimate, implements = ESTIMATORS[rel["method"]]
        if rel["kind"] != implements:
            out.append(verdict(
                UNVERIFIABLE,
                f"{rel['method']!r} computes a {implements}, but this claim "
                f"declares kind {rel['kind']!r}"))
            continue

        unknown = [d for d in (xd, yd) if d not in sets]
        if unknown:
            out.append(verdict(
                UNVERIFIABLE,
                f"no dataset with id {', '.join(repr(u) for u in unknown)}. "
                f"Known: {', '.join(sorted(sets))}"))
            continue

        # ONE DATASET, or two joined on a stated key. The join is refused
        # outright on duplicate keys and its coverage travels with the verdict —
        # a correlation over a 40%-covered join is a number about a different
        # population from the one either file describes.
        if xd == yd:
            rows = sets[xd]["rows"]
        else:
            key = rel.get("join")
            if not key:
                out.append(verdict(
                    UNVERIFIABLE,
                    f"{xd} and {yd} are different datasets and no `join` key is "
                    f"declared, so their rows cannot be lined up"))
                continue
            pairs_rows, jr = join(sets[xd], sets[yd], key)
            joins.append(jr)
            row["join"] = jr
            if jr["refused"]:
                out.append(verdict(UNVERIFIABLE, jr["refused"]))
                continue
            rows = [{**r_, **l_} for l_, r_ in pairs_rows]

        absent = [c for c in (x, y) if not rows or c not in rows[0]]
        if absent:
            out.append(verdict(UNVERIFIABLE,
                               f"the data has no column {', '.join(absent)}"))
            continue
        undeclared = [c for (d_, c) in ((xd, x), (yd, y))
                      if declared.get((d_, c), {}).get("unit") is None]
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
        # A COLUMN THAT EXISTS AND IS EMPTY is not the same as a column with
        # few clusters, and the bootstrap cannot tell them apart -- both arrive
        # as "fewer than 3 independent units". The difference matters: one says
        # the data is small, the other says the declared unit is not in the file
        # at all. Measured on al_mace's `darby_predictions_rowan32_k4.csv`,
        # where `butina_cluster` is a header over 32 empty rows and is named as
        # that file's resampling unit in two places downstream.
        if unit and clusters and not any(c.strip() for c in clusters):
            out.append(verdict(
                UNVERIFIABLE,
                f"resampling unit {unit!r} is a column of the data but is EMPTY "
                f"in all {len(clusters)} usable row(s), so there is nothing to "
                f"cluster on. The unit was declared and the file does not carry "
                f"it -- which is a different problem from having too few groups, "
                f"and is not fixed by collecting more rows"))
            continue
        row |= {"n_rows": len(pairs), "dropped": dropped}
        r = estimate([p[0] for p in pairs], [p[1] for p in pairs])
        if r is None:
            out.append(verdict(UNVERIFIABLE,
                               f"r is undefined on {len(pairs)} usable pair(s)"))
            continue
        row["estimate"] = round(r, 6)
        ci = bootstrap_ci(pairs, clusters, estimate=estimate)
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
    return {"project": ctx["project"]["name"],
            "datasets": {d: {"origin": s_["origin"], "rows": len(s_["rows"]),
                             "digests": s_["digests"]}
                         for d, s_ in sets.items()},
            "joins": joins,
            "rows_read": sum(len(s_["rows"]) for s_ in sets.values()),
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
