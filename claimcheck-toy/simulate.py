"""Make the toy dataset. In a real project a lab would hand you this file.

One hundred samples in ten batches. `cheap_score` is a noisy proxy for
`slow_measure_kJ`; `coin_flip` is a seeded uniform draw and is the control.

TEN batches, not four, and the number matters. The batch is the resampling unit,
so it sets how wide every interval is. A first version of this used four and the
noise control came back "correlated" at r = 0.404 with an interval excluding
zero. That was not a bug: four independent units cannot separate noise from
signal, and the control is what showed it.
"""

import argparse
import csv
import random
from pathlib import Path

N_BATCHES = 10
PER_BATCH = 10
PROXY_NOISE = 1.2          # standard deviation added to the truth, arbitrary units


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/measurements.csv")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows = []
    for b in range(N_BATCHES):
        for i in range(PER_BATCH):
            truth = rng.uniform(0, 10)
            rows.append({
                "sample": f"b{b}s{i}",
                "batch": f"batch{b}",
                "cheap_score": round(truth + rng.gauss(0, PROXY_NOISE), 3),
                "slow_measure_kJ": round(truth, 3),
                "coin_flip": round(rng.uniform(0, 1), 3),
            })

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows, {N_BATCHES} batches, seed {a.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
