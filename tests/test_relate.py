"""`claimcheck.relate` — the three verdicts, each seen to happen.

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
from pathlib import Path

import pytest

from claimcheck.relate import AGREES, ESTIMATORS, REFUTED, UNVERIFIABLE, _mean, _median, bootstrap_ci, check, pearson

REPO = Path(__file__).resolve().parents[1]


def _project(tmp_path: Path, rows: list[dict], relationships: str,
             variables: str, resampling_unit: str = "") -> Path:
    cols = list(rows[0])
    lines = [",".join(cols)]
    lines += [",".join(str(r[c]) for c in cols) for r in rows]
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    ru = f'resampling_unit = "{resampling_unit}"\n' if resampling_unit else ""
    (tmp_path / "context.toml").write_text(
        f'[project]\nname = "t"\n\n'
        f'[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n{ru}\n'
        f'{variables}\n{relationships}\n')
    return tmp_path


_VARS = ('[[variables]]\ndataset = "d"\nname = "x"\nunit = "1"\n\n'
         '[[variables]]\ndataset = "d"\nname = "y"\nunit = "1"\n')


def _rel(claimed: float) -> str:
    return (f'[[relationships]]\nx = "d.x"\ny = "d.y"\nkind = "correlation"\n'
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
    only_x = '[[variables]]\ndataset = "d"\nname = "x"\nunit = "1"\n'
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
    assert set(d["datasets"]) == {"mace", "affinity"}


def test_an_empty_resampling_unit_says_so(tmp_path):
    """"Fewer than 3 independent units" is true of an empty column and of a
    genuinely small one, and only one of those is fixed by more data.

    From al_mace's `darby_predictions_rowan32_k4.csv`: `butina_cluster` is a
    header over 32 empty rows, and two documents downstream name it as that
    file's resampling unit.
    """
    import csv
    from claimcheck.relate import check
    (tmp_path / "data").mkdir()
    with (tmp_path / "data" / "d.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "grp"])
        for i in range(12):
            w.writerow([i, 2 * i + (i % 3), ""])
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "t"\n'
        '[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["data/d.csv"]\n'
        '[[variables]]\ndataset = "d"\nname = "x"\nunit = "1"\n'
        '[[variables]]\ndataset = "d"\nname = "y"\nunit = "1"\n'
        '[[relationships]]\nx = "d.x"\ny = "d.y"\nkind = "correlation"\n'
        'method = "pearson"\nclaimed = 0.9\nresampling_unit = "grp"\n')
    r = check(tmp_path)["relationships"][0]
    assert r["verdict"] == "unverifiable"
    assert "EMPTY" in r["why"], r["why"]
    assert "too few groups" in r["why"], r["why"]


def test_a_populated_resampling_unit_still_works(tmp_path):
    """The other side: the same shape with the column filled in must NOT take
    the branch above."""
    import csv
    from claimcheck.relate import check
    (tmp_path / "data").mkdir()
    with (tmp_path / "data" / "d.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "grp"])
        for i in range(12):
            w.writerow([i, 2 * i + (i % 3), f"c{i % 4}"])
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "t"\n'
        '[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["data/d.csv"]\n'
        '[[variables]]\ndataset = "d"\nname = "x"\nunit = "1"\n'
        '[[variables]]\ndataset = "d"\nname = "y"\nunit = "1"\n'
        '[[relationships]]\nx = "d.x"\ny = "d.y"\nkind = "correlation"\n'
        'method = "pearson"\nclaimed = 0.99\nresampling_unit = "grp"\n')
    r = check(tmp_path)["relationships"][0]
    assert "EMPTY" not in r["why"], r["why"]
    assert r.get("n_units") == 4, r


# ---------------------------------------------------------------------------
# A known answer of +-1 from two independent computations
# ---------------------------------------------------------------------------

def _known_answer_rows(rel_noise: float, n: int = 60, seed: int = 3) -> list[dict]:
    """y is x computed a second way, agreeing to `rel_noise` relative, the shape
    of an engine derivative against its finite difference. Written with 17
    significant figures so the CSV does not round the disagreement away."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a = rng.uniform(-150, 150)
        out.append({"x": f"{a:.17g}", "y": f"{a * (1 + rel_noise * rng.gauss(0, 1)):.17g}", "g": i})
    return out


def test_a_known_answer_of_one_agrees_at_numerical_precision(tmp_path):
    """Two computations agreeing to 1e-7 relative, as an engine derivative and a
    central finite difference do, have 1 - r of about 5e-15: a real difference,
    and a bootstrap interval tight enough to exclude 1.0. Refuted before the
    known-answer tolerance existed (mace_lambda/alkanes KF2a, 2026-09-25,
    1 - r = 2.6e-15 over 378 frames)."""
    d = check(_project(tmp_path, _known_answer_rows(1e-7), _rel(1.0), _VARS))
    got = d["relationships"][0]
    assert got["verdict"] == AGREES, got["why"]
    assert "numerical precision" in got["why"]


def test_the_known_answer_tolerance_does_not_rescue_a_real_disagreement(tmp_path):
    """The other side: agreement to 1e-5 relative gives 1 - r of about 5e-11,
    fifty times the tolerance, so a claim of exactly 1.0 is still refuted."""
    d = check(_project(tmp_path, _known_answer_rows(1e-5), _rel(1.0), _VARS))
    got = d["relationships"][0]
    assert got["verdict"] == REFUTED, got["why"]


# ---------------------------------------------------------------------------
# Verification levels: toy, test_case, reference, real
# ---------------------------------------------------------------------------

def test_relationship_inherits_verification_level_from_dataset(tmp_path):
    """A claim inherits verification level from its dataset when not overridden."""
    rows = _correlated(n=30, rho=0.9)
    cols = list(rows[0])
    lines = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "lvl_test"\n\n'
        '[[datasets]]\nid = "d"\norigin = "computed"\nlevel = "toy"\nfiles = ["d.csv"]\n\n'
        f'{_VARS}\n{_rel(0.9)}\n'
    )
    d = check(tmp_path)
    rel = d["relationships"][0]
    assert rel["level"] == "toy"
    assert "levels" in d
    assert d["levels"]["toy"]["total"] == 1


def test_relationship_explicit_verification_level_override(tmp_path):
    """A claim can explicitly declare its verification level."""
    rows = _correlated(n=30, rho=0.9)
    cols = list(rows[0])
    lines = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    rel_toml = (
        '[[relationships]]\nx = "d.x"\ny = "d.y"\nkind = "correlation"\n'
        'method = "pearson"\nclaimed = 0.9\nlevel = "test_case"\n'
    )
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "lvl_test"\n\n'
        '[[datasets]]\nid = "d"\norigin = "computed"\nlevel = "reference"\nfiles = ["d.csv"]\n\n'
        f'{_VARS}\n{rel_toml}\n'
    )
    d = check(tmp_path)
    rel = d["relationships"][0]
    assert rel["level"] == "test_case"
    assert d["levels"]["test_case"]["total"] == 1



# ---------------------------------------------------------------------------
# `median` (2026-10-21): the typical resampling unit, a value method added for the median per-target rank correlation of
# miel_outliers (amendment 5). Two-sided: it is the median and cannot be the mean, and it fails soft on a single unit or an empty column.
# ---------------------------------------------------------------------------

def test_median_is_registered_as_a_value_method_beside_the_mean():
    assert ESTIMATORS["median"] == (_median, "value")
    assert ESTIMATORS["mean"] == (_mean, "value"), "adding the median left the mean as it was"


def test_median_estimator_odd_even_tie_single_and_empty():
    assert _median([], [3.0, 1.0, 2.0]) == 2.0                         # odd count: the middle value
    assert _median([], [4.0, 1.0, 3.0, 2.0]) == 2.5                    # even count: the mean of the middle two
    assert _median([], [1.0, 2.0, 2.0, 3.0]) == 2.0                    # a tie across the middle of an even count
    assert _median([], [2.0, 2.0, 2.0, 2.0]) == 2.0                    # all tied
    assert _median([], [1.0, 2.0, 2.0, 10.0]) == 2.0                   # a tie, and an outlier that does not move it
    assert _median([], [7.0]) == 7.0                                   # a single unit
    assert _median([], []) is None                                     # the empty case


def test_median_is_not_the_mean_on_a_skewed_set():
    skew = [0.0, 0.0, 0.0, 0.0, 100.0]
    assert _median([], skew) == 0.0 and _mean([], skew) == 20.0 and _median([], skew) != _mean([], skew)


_MVARS = ('[[variables]]\ndataset = "d"\nname = "unit"\nunit = "1"\n\n'
          '[[variables]]\ndataset = "d"\nname = "v"\nunit = "1"\n')


def _median_claim(rid: str, method: str, lo: float | None, hi: float | None) -> str:
    rng = ", ".join(("-inf" if lo is None else str(lo), "inf" if hi is None else str(hi)))
    return (f'[[relationships]]\nid = "{rid}"\nx = "d.v"\ny = "d.v"\nkind = "value"\nmethod = "{method}"\n'
            f'resampling_unit = "unit"\nclaimed_range = [{rng}]\n\n')


def _verdicts(tmp_path: Path, values: list, rels: str) -> dict:
    rows = [{"unit": f"u{i}", "v": v} for i, v in enumerate(values)]
    d = check(_project(tmp_path, rows, rels, _MVARS, resampling_unit="unit"))
    return {r["id"]: r for r in d["relationships"]}


def test_a_median_claim_agrees_refutes_or_straddles_on_data_whose_median_is_known(tmp_path):
    values = [0.10 + 0.001 * (i % 7) for i in range(80)]                # every unit between 0.100 and 0.106, median 0.103
    v = _verdicts(tmp_path, values,
                  _median_claim("in", "median", 0.05, 0.2) + _median_claim("out", "median", 0.5, 1.0)
                  + _median_claim("edge", "median", 0.103, 1.0) + _median_claim("open_above", "median", None, 0.05))
    assert v["in"]["verdict"] == AGREES and v["out"]["verdict"] == REFUTED
    assert v["edge"]["verdict"] == UNVERIFIABLE, "the interval straddles the bound at the median itself"
    assert v["open_above"]["verdict"] == REFUTED, "a one-sided range with an open end: every unit is above 0.05"
    assert 0.100 <= v["in"]["estimate"] <= 0.106


def test_a_few_huge_units_move_the_mean_claim_and_not_the_median_claim(tmp_path):
    values = [(-1) ** i * 0.01 * (i % 5) for i in range(57)] + [100.0, 100.0, 100.0]       # median 0.0, mean about 5
    v = _verdicts(tmp_path, values, _median_claim("med", "median", -0.2, 0.2) + _median_claim("avg", "mean", -0.2, 0.2))
    assert v["med"]["verdict"] == AGREES and abs(v["med"]["estimate"]) < 0.05
    assert v["avg"]["verdict"] != AGREES and v["avg"]["estimate"] > 4.0, "the same range on the mean cannot agree: the method is not the mean"


def test_a_median_of_one_unit_or_of_an_empty_column_reads_unverifiable_and_does_not_raise(tmp_path):
    (tmp_path / "one").mkdir()
    one = _verdicts(tmp_path / "one", [0.1], _median_claim("m", "median", 0.0, 1.0))
    assert one["m"]["verdict"] == UNVERIFIABLE                          # one unit: nothing to resample, so no interval
    (tmp_path / "none").mkdir()
    rows = [{"unit": f"u{i}", "v": ""} for i in range(5)]               # five units whose value is missing: the median of nothing
    d = check(_project(tmp_path / "none", rows, _median_claim("m", "median", 0.0, 1.0), _MVARS, resampling_unit="unit"))
    got = d["relationships"][0]
    assert got["verdict"] == UNVERIFIABLE and got.get("estimate") is None


# ---------------------------------------------------------------------------
# Added 2026-11-12 from an independent review of the rank transform and the bootstrap
# ---------------------------------------------------------------------------

def test_spearman_gives_tied_values_their_average_rank():
    """Hand-ranked: x has ranks 1.5, 1.5, 3, 4.5, 4.5 and y has 1, 2.5, 2.5, 4, 5, so Spearman is Pearson on those. Ordering the
    ties by position instead would give a different number, which the second assertion requires."""
    from claimcheck.relate import spearman
    xs, ys = [1.0, 1.0, 2.0, 3.0, 3.0], [0.0, 5.0, 5.0, 7.0, 9.0]
    expected = pearson([1.5, 1.5, 3.0, 4.5, 4.5], [1.0, 2.5, 2.5, 4.0, 5.0])
    assert spearman(xs, ys) == pytest.approx(expected, abs=1e-12)
    assert spearman(xs, ys) != pytest.approx(pearson([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]), abs=1e-6)
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(31)
    for _ in range(10):
        a = [float(rng.randrange(6)) for _ in range(200)]
        b = [float(round(0.4 * v + rng.gauss(0, 1))) for v in a]
        assert spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic, abs=1e-12)


def test_no_bootstrap_draw_ever_splits_a_unit():
    """Every row carries a unique id in x and units hold 1 to 5 rows. In each of the 2,000 draws all rows of a unit must appear the
    same number of times. The control is the same detector with no unit declared, where it must fire on nearly every draw."""
    from collections import Counter
    rng = random.Random(8)
    clusters, pairs = [], []
    for u in range(30):
        for _ in range(rng.randint(1, 5)):
            clusters.append(f"u{u}")
            pairs.append((float(len(pairs)), rng.gauss(0, 1)))
    unit_of = {int(p[0]): c for p, c in zip(pairs, clusters, strict=True)}
    size_of = Counter(clusters)
    seen = {"draws": 0, "split": 0, "sizes": set()}

    def recording(xs, ys):
        per_unit: dict[str, list[int]] = {}
        for row_id, m in Counter(int(v) for v in xs).items():
            per_unit.setdefault(unit_of[row_id], []).append(m)
        whole = all(len(ms) == size_of[u] and len(set(ms)) == 1 for u, ms in per_unit.items())
        seen["draws"] += 1
        seen["split"] += not whole
        seen["sizes"].add(sum(ms[0] for ms in per_unit.values()))
        return pearson(xs, ys)

    _, _, n_units = bootstrap_ci(pairs, clusters, estimate=recording)
    assert n_units == 30 and seen["split"] == 0 and seen["sizes"] == {30} and seen["draws"] > 1000
    seen.update(draws=0, split=0, sizes=set())
    bootstrap_ci(pairs, None, estimate=recording)
    assert seen["split"] > 0.9 * seen["draws"]
