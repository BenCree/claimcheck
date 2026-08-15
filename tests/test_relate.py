"""`provchem.relate` — the three verdicts, each seen to happen.

**A checker that has only ever returned one verdict has not been shown capable
of the others.** Run against the real sweep in `example`, this
returned `refuted` three times and `unverifiable` once and `agrees` never — so
`agrees` is exercised here on constructed data whose true correlation is known,
rather than by tuning a claim about real data until it passed. Choosing the
claimed value after seeing the interval is the thing this package exists to
refuse; doing it to make a test go green would be worse, not better.

The other half is `reg-06`'s rule in a statistical setting: the interval must be
narrow enough to refute a wrong claim AND wide enough to admit a right one. A
check that refuses everything is not strict, it is broken — and the first
version of this, comparing point estimates against a hand-picked tolerance, was
exactly that.
"""

from __future__ import annotations

import math
import random
import shutil
import tomllib
from pathlib import Path

import pytest

from claimcheck.relate import AGREES, REFUTED, UNVERIFIABLE, bootstrap_ci, check, pearson

REPO = Path(__file__).resolve().parents[1]


def _project(tmp_path: Path, rows: list[dict], relationships: str,
             variables: str, resampling_unit: str = "") -> Path:
    cols = list(rows[0])
    lines = [",".join(cols)]
    lines += [",".join(str(r[c]) for c in cols) for r in rows]
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    ru = f'resampling_unit = "{resampling_unit}"\n' if resampling_unit else ""
    (tmp_path / "context.toml").write_text(
        f'[dataset]\nname = "t"\nfiles = ["d.csv"]\n{ru}\n'
        f'{variables}\n{relationships}\n')
    return tmp_path


_VARS = ('[[variables]]\nname = "x"\nunit = "1"\n\n'
         '[[variables]]\nname = "y"\nunit = "1"\n')


def _rel(claimed: float) -> str:
    return (f'[[relationships]]\nx = "x"\ny = "y"\nkind = "correlation"\n'
            f'method = "pearson"\nclaimed = {claimed}\n')


def _correlated(n: int = 200, rho: float = 0.8, seed: int = 1) -> list[dict]:
    """Pairs with a known population correlation, so the truth is not in doubt."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a = rng.gauss(0, 1)
        b = rho * a + math.sqrt(1 - rho ** 2) * rng.gauss(0, 1)
        out.append({"x": round(a, 6), "y": round(b, 6), "g": i % 20})
    return out


# ---------------------------------------------------------------------------
# All three verdicts, each seen
# ---------------------------------------------------------------------------

def test_a_true_claim_agrees(tmp_path):
    """The verdict the real data never produced. Constructed at rho = 0.8 and
    claimed at the sample estimate's own value, so `agrees` is reachable."""
    rows = _correlated()
    r = pearson([x["x"] for x in rows], [x["y"] for x in rows])
    d = check(_project(tmp_path, rows, _rel(round(r, 3)), _VARS))
    got = d["relationships"][0]
    assert got["verdict"] == AGREES, got["why"]
    assert got["status"] == "measured"
    assert got["ci_lo"] <= got["claimed"] <= got["ci_hi"]


def test_a_wrong_claim_is_refuted(tmp_path):
    d = check(_project(tmp_path, _correlated(), _rel(0.10), _VARS))
    got = d["relationships"][0]
    assert got["verdict"] == REFUTED and got["status"] == "falsified"
    assert "OUTSIDE" in got["why"]
    assert d["any_refuted"] is True


def test_an_undeclared_unit_is_unverifiable_not_a_pass(tmp_path):
    """`unit` absent means nobody has said, and a relationship between
    undeclared quantities is not comparable with anybody else's."""
    only_x = '[[variables]]\nname = "x"\nunit = "1"\n'
    d = check(_project(tmp_path, _correlated(), _rel(0.8), only_x))
    got = d["relationships"][0]
    assert got["verdict"] == UNVERIFIABLE and got["status"] == "untested"
    assert "no unit declared for y" in got["why"]
    assert d["any_unverifiable"] is True


def test_the_interval_is_not_so_wide_it_admits_anything(tmp_path):
    """The other side of `reg-06`. A check that cannot refuse is not a check —
    and a percentile interval on 200 points must exclude 0 for rho = 0.8."""
    d = check(_project(tmp_path, _correlated(), _rel(0.0), _VARS))
    assert d["relationships"][0]["verdict"] == REFUTED


# ---------------------------------------------------------------------------
# The resampling unit, which is the part that would be decoration if guessed
# ---------------------------------------------------------------------------

def test_clustering_widens_the_interval(tmp_path):
    """The whole reason the unit must be declared. Rows inside a cluster are
    not independent, so resampling rows reports an interval too narrow — and a
    too-narrow interval refutes true claims, which is the expensive direction.
    """
    rows = _correlated(n=200)
    # Each cluster repeated: the row count triples, the information does not.
    inflated = [dict(r) for r in rows for _ in range(3)]
    flat = bootstrap_ci([(r["x"], r["y"]) for r in inflated], None)
    clustered = bootstrap_ci([(r["x"], r["y"]) for r in inflated],
                             [str(r["g"]) for r in inflated])
    assert flat and clustered
    assert (clustered[1] - clustered[0]) > (flat[1] - flat[0]), \
        "clustering must not report MORE precision than resampling rows"
    assert clustered[2] == 20 and flat[2] == 600, \
        "n_units is the number actually resampled, not the row count"


def test_a_resampling_unit_that_is_not_a_column_is_refused(tmp_path):
    rows = _correlated()
    d = check(_project(tmp_path, rows, _rel(0.8), _VARS,
                       resampling_unit="no_such_column"))
    got = d["relationships"][0]
    assert got["verdict"] == UNVERIFIABLE
    assert "independence the data does not have" in got["why"]


def test_too_few_units_gives_no_interval(tmp_path):
    """Two clusters cannot support a percentile interval, and saying so is a
    different answer from saying the claim is wrong."""
    rows = [{"x": i, "y": i * 2 + (i % 3), "g": i % 2} for i in range(30)]
    d = check(_project(tmp_path, rows, _rel(0.9), _VARS, resampling_unit="g"))
    assert d["relationships"][0]["verdict"] == UNVERIFIABLE


# ---------------------------------------------------------------------------
# Determinism, and the arithmetic
# ---------------------------------------------------------------------------

def test_two_runs_of_one_dataset_give_one_interval(tmp_path):
    """An interval that moves between runs of the same data is not a
    measurement. The bootstrap seed is fixed for that reason."""
    p = _project(tmp_path, _correlated(), _rel(0.8), _VARS)
    a, b = check(p)["relationships"][0], check(p)["relationships"][0]
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])


def test_r_is_none_rather_than_zero_where_it_is_undefined():
    """0.0 reads as 'no relationship', which is a conclusion. A constant column
    supports no conclusion at all."""
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert pearson([1.0, 2.0], [1.0, 2.0]) is None          # n < 3
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_the_shipped_example_reaches_both_verdicts():
    """The example is the first thing anyone runs. If a first run showed only
    one verdict, a reader would not learn the other exists."""
    d = check(REPO / "example")
    assert d["verdicts"].get(AGREES) and d["verdicts"].get(REFUTED), d["verdicts"]
    assert d["rows_read"] == 870
