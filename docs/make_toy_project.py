"""Build the toy project the tutorial in docs/TUTORIAL.md walks through.

    python docs/make_toy_project.py widget_proxy
    cd widget_proxy
    python -m claimcheck.relate .

Ten batches of ten samples. `cheap_score` is a noisy proxy for
`slow_measure_kJ`; `coin_flip` is a seeded uniform draw and is the control. The
batch is the resampling unit, and ten of them is the smallest number that
separates the control from the signal: an earlier version used four and the noise
came back "correlated" at r = 0.404, which was the design being underpowered
rather than a bug.
"""

import csv
import hashlib
import random
import sys
from pathlib import Path

CONTEXT = '''[project]
name = "widget_proxy"
title = "Does a cheap score predict the slow measurement?"
context = """
One hundred samples in ten batches. Each has a cheap_score that takes a second
and a slow_measure_kJ that takes a day. The question is whether the cheap one can
stand in for the slow one. coin_flip is a control: it is pure noise and must not
correlate with anything.
"""

[[datasets]]
id = "runs"
origin = "manual"
files = ["data/measurements.csv"]
source = "https://example.org/widget-lab/measurements.csv"
retrieved = "2026-09-07"
sha256 = "%(sha)s"
title = "One row per sample: the cheap score, the slow measurement and a control"

[[variables]]
dataset = "runs"
name = "cheap_score"
unit = "1"
description = "The fast proxy, arbitrary units."

[[variables]]
dataset = "runs"
name = "slow_measure_kJ"
unit = "kJ/mol"
description = "The expensive measurement this project is trying to avoid making."

[[variables]]
dataset = "runs"
name = "coin_flip"
unit = "1"
description = "A seeded uniform draw. The control."

[[variables]]
dataset = "runs"
name = "batch"
unit = "1"
description = "Which batch a sample came from. The resampling unit: samples in one batch are not independent of each other."

[[relationships]]
id = "R1_proxy_predicts"
x = "runs.cheap_score"
y = "runs.slow_measure_kJ"
kind = "correlation"
method = "pearson"
claimed = 0.9
resampling_unit = "batch"
evidence.source = "Claimed at 0.9 before looking at the data."
evidence.locator = "data/measurements.csv :: cheap_score, slow_measure_kJ"

[[relationships]]
id = "R2_control_finds_nothing"
x = "runs.coin_flip"
y = "runs.slow_measure_kJ"
kind = "correlation"
method = "pearson"
claimed = 0.0
resampling_unit = "batch"
evidence.source = "Claimed at 0.0 and REQUIRED to hold. If noise correlates here the construction is broken and R1 cannot be read."
evidence.locator = "data/measurements.csv :: coin_flip, slow_measure_kJ"
'''


def main(argv):
    out = Path(argv[1] if len(argv) > 1 else "widget_proxy")
    (out / "data").mkdir(parents=True, exist_ok=True)

    rng = random.Random(7)
    rows = []
    for b in range(10):
        for i in range(10):
            truth = rng.uniform(0, 10)
            rows.append({
                "sample": f"b{b}s{i}",
                "batch": f"batch{b}",
                "cheap_score": round(truth + rng.gauss(0, 1.2), 3),
                "slow_measure_kJ": round(truth, 3),
                "coin_flip": round(rng.uniform(0, 1), 3),
            })

    csv_path = out / "data" / "measurements.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # The digest is written into the declaration rather than left for the reader,
    # because a `manual` dataset without one is refused on the first load.
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (out / "context.toml").write_text(CONTEXT % {"sha": sha})

    print(f"{out}/ built: {len(rows)} rows, 10 batches")
    print(f"  cd {out} && python -m claimcheck.relate .")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
