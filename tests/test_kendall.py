"""Kendall tau-b (method = "kendall"): the tie-corrected rank correlation, adjudicated by the same bootstrap over `resampling_unit` as
`spearman` and `pearson` (added 2026-10-27).

Each case has an answer that can be worked out by hand, each verdict is tested in both directions, and the estimator is checked against
`scipy.stats.kendalltau(variant="b")` where scipy is installed (the package itself needs nothing but the standard library).

tau-b = (C - D) / sqrt((C + D + Tx) * (C + D + Ty)), where C and D count concordant and discordant pairs, Tx the pairs tied in x only
and Ty the pairs tied in y only. A pair tied in both columns is in none of the four counts.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from claimcheck.relate import AGREES, ESTIMATORS, REFUTED, UNVERIFIABLE, bootstrap_ci, check, kendall, kendall_tie_tol

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "kendall_claims"

VARS = ('[[variables]]\ndataset = "d"\nname = "unit"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "a"\nunit = "kcal/mol"\n\n'
        '[[variables]]\ndataset = "d"\nname = "b"\nunit = "kcal/mol"\n')


def project(tmp_path: Path, rows: list[tuple], rels: str) -> Path:
    """A one-dataset project: columns unit, a, b, one row per tuple (unit, a, b)."""
    (tmp_path / "d.csv").write_text("unit,a,b\n" + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "t"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + VARS + "\n" + rels)
    return tmp_path


def rel(claimed: str = "claimed = 1.0", extra: str = "", unit: str | None = "unit", method: str = "kendall", kind: str = "correlation") -> str:
    ru = f'resampling_unit = "{unit}"\n' if unit else ""
    return f'[[relationships]]\nid = "k"\nx = "d.a"\ny = "d.b"\nkind = "{kind}"\nmethod = "{method}"\n{claimed}\n{ru}{extra}\n'


def verdict(tmp_path: Path, rows: list[tuple], **kw) -> dict:
    return check(project(tmp_path, rows, rel(**kw)))["relationships"][0]


def per_row(pairs: list[tuple[float, float]]) -> list[tuple]:
    """Each pair as its own resampling unit."""
    return [(f"u{i}", a, b) for i, (a, b) in enumerate(pairs)]


def independent_pairs(n: int, seed: int = 11) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n)]


# ---------------------------------------------------------------------------
# The estimator alone
# ---------------------------------------------------------------------------

def test_registered_as_a_correlation_estimator():
    assert "kendall" in ESTIMATORS and ESTIMATORS["kendall"][1] == "correlation" and ESTIMATORS["kendall"][0] is kendall


def test_a_perfect_monotone_relation_is_one_even_when_curved():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert kendall(xs, [x ** 3 for x in xs]) == 1.0


def test_a_reversed_relation_is_minus_one():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert kendall(xs, [-x for x in xs]) == -1.0


def test_independent_data_is_near_zero():
    pairs = independent_pairs(400)
    t = kendall([p[0] for p in pairs], [p[1] for p in pairs])
    assert t is not None and abs(t) < 0.1                    # sd of tau under independence is about 0.037 at n = 400


def test_hand_computed_table_with_ties_in_both_columns():
    """Seven points, A(1,1) B(1,1) C(1,2) D(1,2) E(2,1) F(2,3) G(2,3).

    21 pairs. Tied in x: the four points at x = 1 give 6 pairs, the three at x = 2 give 3, so 9. Tied in y: y = 1 holds A, B, E (3 pairs),
    y = 2 holds C, D (1), y = 3 holds F, G (1), so 5. Tied in both: AB, CD, FG, so 3. Hence Tx = 9 - 3 = 6 and Ty = 5 - 3 = 2.
    The pairs across the two x groups that are not tied in y: E (y = 1) against C and D (y = 2) is discordant, 2 pairs; F and G (y = 3)
    against A, B, C, D (all lower) are concordant, 8 pairs; E against A and B is tied in y. So C = 8 and D = 2, and
    C + D + Tx + Ty + joint ties = 8 + 2 + 6 + 2 + 3 = 21.
    tau-b = (8 - 2) / sqrt((10 + 6) * (10 + 2)) = 6 / sqrt(192) = 0.4330127.
    """
    pts = [(1, 1), (1, 1), (1, 2), (1, 2), (2, 1), (2, 3), (2, 3)]
    assert kendall([p[0] for p in pts], [p[1] for p in pts]) == pytest.approx(6 / math.sqrt(192), abs=1e-12)


def test_hand_computed_asymmetric_table():
    """Four points (1,1) (1,2) (1,3) (2,3). Pairs: three tied in x (the x = 1 group), one tied in y ((1,3) with (2,3)), and two concordant
    ((1,1) and (1,2) each against (2,3)). So C = 2, D = 0, Tx = 3, Ty = 1 and 2 + 0 + 3 + 1 = 6 = 4 choose 2.
    tau-b = 2 / sqrt((2 + 3) * (2 + 1)) = 2 / sqrt(15) = 0.5163978. Tau-a would give 2/6 = 0.333, so this separates tau-b from tau-a."""
    assert kendall([1, 1, 1, 2], [1, 2, 3, 3]) == pytest.approx(2 / math.sqrt(15), abs=1e-12)
    assert kendall([1, 2, 3, 3], [1, 1, 1, 2]) == pytest.approx(2 / math.sqrt(15), abs=1e-12)      # symmetric in its arguments


def test_it_does_not_depend_on_the_order_of_the_rows():
    pts = [(1, 1), (1, 1), (1, 2), (1, 2), (2, 1), (2, 3), (2, 3)]
    base = kendall([p[0] for p in pts], [p[1] for p in pts])
    rng = random.Random(3)
    for _ in range(10):
        shuffled = pts[:]
        rng.shuffle(shuffled)
        assert kendall([p[0] for p in shuffled], [p[1] for p in shuffled]) == pytest.approx(base, abs=1e-12)


def test_agrees_with_scipy_tau_b_on_heavily_tied_random_data():
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(5)
    for n in (3, 8, 50, 300):
        xs = [float(rng.randint(0, 6)) for _ in range(n)]            # few levels, so most pairs tie
        ys = [x + rng.randint(-2, 2) for x in xs]
        got = kendall(xs, ys)
        want = stats.kendalltau(xs, ys, variant="b").statistic
        if math.isnan(want):
            assert got is None
        else:
            assert got == pytest.approx(want, abs=1e-12)
    xs = [rng.gauss(0, 1) for _ in range(200)]                       # and with no ties at all
    ys = [x + rng.gauss(0, 1) for x in xs]
    assert kendall(xs, ys) == pytest.approx(stats.kendalltau(xs, ys, variant="b").statistic, abs=1e-12)


def test_undefined_where_a_column_is_constant_or_there_are_too_few_rows():
    assert kendall([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]) is None
    assert kendall([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]) is None
    assert kendall([1.0, 2.0], [1.0, 2.0]) is None                   # fewer than 3 pairs, as for pearson
    assert kendall([], []) is None


def test_it_is_fast_enough_to_sit_inside_a_2000_draw_bootstrap():
    """A pure O(n^2) loop costs about 40 ms at n = 642, and 4,000 calls (bootstrap plus permutation null) would take minutes. The
    n log n count must stay well under that: 100 calls at n = 642 in under a second (about 0.3 s measured), and n = 5000 in one call."""
    pairs = independent_pairs(642)
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    start = time.perf_counter()
    for _ in range(100):
        kendall(xs, ys)
    assert time.perf_counter() - start < 1.0
    big = independent_pairs(5000)
    start = time.perf_counter()
    kendall([p[0] for p in big], [p[1] for p in big])
    assert time.perf_counter() - start < 1.0


# ---------------------------------------------------------------------------
# tie_tol
# ---------------------------------------------------------------------------

def noisy_ties(groups: int = 4, seeds: int = 4) -> list[tuple]:
    """One truth per group, estimates equal in exact arithmetic but ordered by 1e-9 noise, as in test_tie_tol.py."""
    rng = random.Random(7)
    return [(f"p{g}", g, g + rng.uniform(-1e-9, 1e-9)) for g in range(groups) for _ in range(seeds)]


def test_tie_tol_estimator_merges_values_within_the_tolerance():
    rows = noisy_ties()
    xs, ys = [r[1] for r in rows], [r[2] for r in rows]
    raw, snapped = kendall(xs, ys), kendall_tie_tol(1e-6)(xs, ys)
    # xs tie within each group of 4 and ys tie only once snapped, so each group's 6 pairs are tied in both columns
    assert raw is not None and raw < 1.0
    assert snapped == 1.0


def test_tie_tol_does_not_merge_values_that_really_differ():
    assert kendall_tie_tol(1e-6)([0, 1, 2, 3], [0.0, 2.0, 1.999, 3.0]) < 1.0
    assert kendall_tie_tol(0.1)([0, 1, 2, 3], [0.0, 1.0, 1.001, 3.0]) < 1.0       # 1.0 and 1.001 now tie: 5 of 6 pairs concordant, 1 tied


def test_tie_tol_in_a_claim_turns_a_noise_refutation_into_agreement(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with_tol = verdict(tmp_path / "a", noisy_ties(), extra="tie_tol = 1e-6")
    without = verdict(tmp_path / "b", noisy_ties())
    assert with_tol["estimate"] == 1.0 and with_tol["verdict"] == AGREES and with_tol["tie_tol"] == 1e-6
    assert without["estimate"] < 1.0 and without["verdict"] == REFUTED and "tie_tol" not in without


def test_tie_tol_is_still_refused_for_pearson_and_for_bad_values(tmp_path):
    rows = noisy_ties()
    for name, extra, method in (("p", "tie_tol = 1e-6", "pearson"), ("z", "tie_tol = 0.0", "kendall"), ("n", "tie_tol = -1.0", "kendall")):
        (tmp_path / name).mkdir()
        v = verdict(tmp_path / name, rows, extra=extra, method=method)
        assert v["verdict"] == UNVERIFIABLE and "tie_tol" in v["why"]


# ---------------------------------------------------------------------------
# Verdicts, in both directions
# ---------------------------------------------------------------------------

def test_perfect_relation_claimed_one_agrees_and_claimed_minus_one_is_refuted(tmp_path):
    rows = per_row([(float(i), float(i) ** 3) for i in range(12)])
    (tmp_path / "up").mkdir()
    (tmp_path / "down").mkdir()
    v = verdict(tmp_path / "up", rows)
    assert v["estimate"] == 1.0 and v["verdict"] == AGREES and v["method"] == "kendall"
    w = verdict(tmp_path / "down", rows, claimed="claimed = -1.0")
    assert w["verdict"] == REFUTED


def test_reversed_relation_claimed_minus_one_agrees(tmp_path):
    v = verdict(tmp_path, per_row([(float(i), -float(i)) for i in range(12)]), claimed="claimed = -1.0")
    assert v["estimate"] == -1.0 and v["verdict"] == AGREES


def test_independent_data_has_an_interval_containing_zero_and_a_large_p(tmp_path):
    (tmp_path / "zero").mkdir()
    (tmp_path / "high").mkdir()
    rows = per_row(independent_pairs(80))
    v = verdict(tmp_path / "zero", rows, claimed="claimed = 0.0")
    assert v["ci_lo"] < 0.0 < v["ci_hi"] and abs(v["estimate"]) < 0.2
    assert v["verdict"] == AGREES and v["p_value"] > 0.05
    assert verdict(tmp_path / "high", rows, claimed="claimed = 0.8")["verdict"] == REFUTED


def test_a_strong_relation_has_a_small_p(tmp_path):
    rng = random.Random(2)
    rows = per_row([(x, x + rng.gauss(0, 0.5)) for x in (rng.gauss(0, 1) for _ in range(80))])
    v = verdict(tmp_path, rows, claimed="claimed_range = [0.4, 1.0]")
    assert v["estimate"] > 0.5 and v["p_value"] <= 0.01 and v["verdict"] == AGREES


def test_claimed_range_agrees_inside_refutes_outside_and_is_unverifiable_across_a_bound(tmp_path):
    for name in ("in", "out", "across"):
        (tmp_path / name).mkdir()
    rows = per_row(independent_pairs(80))
    inside = verdict(tmp_path / "in", rows, claimed="claimed_range = [-0.5, 0.5]")
    outside = verdict(tmp_path / "out", rows, claimed="claimed_range = [0.6, 1.0]")
    across = verdict(tmp_path / "across", rows, claimed="claimed_range = [0.0, 1.0]")
    assert (inside["verdict"], outside["verdict"], across["verdict"]) == (AGREES, REFUTED, UNVERIFIABLE)


def test_r_squared_is_tau_squared_and_says_it_is_not_variance_explained(tmp_path):
    v = verdict(tmp_path, per_row(independent_pairs(80)), claimed="claimed = 0.0")
    assert v["r_squared"] == pytest.approx(v["estimate"] ** 2, abs=1e-5)
    assert v["r_squared_is_on"] == "tau-b" and "not a share of variance" in v["r_squared_note"]


def test_a_constant_column_is_unverifiable_not_an_error(tmp_path):
    v = verdict(tmp_path, per_row([(float(i), 5.0) for i in range(12)]))
    assert v["verdict"] == UNVERIFIABLE and "undefined" in v["why"]


def test_fewer_than_three_units_is_unverifiable(tmp_path):
    rows = [("u1", 1.0, 1.0), ("u1", 2.0, 2.0), ("u1", 3.0, 3.0), ("u2", 4.0, 4.0), ("u2", 5.0, 5.0)]
    v = verdict(tmp_path, rows)
    assert v["verdict"] == UNVERIFIABLE and "fewer than 3 independent units" in v["why"]


def test_kind_mismatch_and_unknown_method_are_refused_and_name_kendall(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    rows = per_row([(float(i), float(i)) for i in range(12)])
    assert "computes a correlation" in verdict(tmp_path / "a", rows, kind="difference")["why"]
    got = verdict(tmp_path / "b", rows, method="kendal")
    assert got["verdict"] == UNVERIFIABLE and "kendall" in got["why"]


# ---------------------------------------------------------------------------
# The bootstrap: reproducible, and over units
# ---------------------------------------------------------------------------

def test_the_interval_is_reproducible(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    rows = per_row(independent_pairs(60))
    one, two = verdict(tmp_path / "a", rows, claimed="claimed = 0.0"), verdict(tmp_path / "b", rows, claimed="claimed = 0.0")
    assert (one["ci_lo"], one["ci_hi"], one["p_value"]) == (two["ci_lo"], two["ci_hi"], two["p_value"])


def test_the_bootstrap_resamples_whole_units_not_rows(tmp_path):
    """Five units of 20 rows share one relation each: inside a unit every row is an exact copy of the unit's single (a, b) point plus a
    tiny tie-breaking offset, so the information is five points, not 100 rows. Declaring the unit must report 5 units and a wider interval
    than treating the 100 rows as independent."""
    (tmp_path / "unit").mkdir()
    (tmp_path / "row").mkdir()
    rng = random.Random(9)
    centres = [(1.0, 2.0), (2.0, 1.0), (3.0, 5.0), (4.0, 3.0), (5.0, 4.0)]
    rows = [(f"u{k}", c[0] + rng.uniform(-0.01, 0.01), c[1] + rng.uniform(-0.01, 0.01)) for k, c in enumerate(centres) for _ in range(20)]
    by_unit = verdict(tmp_path / "unit", rows, claimed="claimed_range = [-1.0, 1.0]")
    by_row = verdict(tmp_path / "row", rows, claimed="claimed_range = [-1.0, 1.0]", unit=None)
    assert by_unit["n_units"] == 5 and by_row["n_units"] == 100
    assert by_unit["ci_hi"] - by_unit["ci_lo"] > by_row["ci_hi"] - by_row["ci_lo"]


def test_bootstrap_ci_accepts_the_estimator_directly_and_counts_units():
    pairs = [(float(i), float(i) ** 2) for i in range(10)]
    lo, hi, n = bootstrap_ci(pairs, [f"u{i // 2}" for i in range(10)], estimate=kendall)
    assert n == 5 and lo == 1.0 and hi == 1.0


def test_642_rows_with_the_full_bootstrap_and_permutation_run_in_reasonable_time(tmp_path):
    rows = [(f"m{i}", a, a + b) for i, (a, b) in enumerate(independent_pairs(642))]
    start = time.perf_counter()
    v = verdict(tmp_path, rows, claimed="claimed_range = [-1.0, 1.0]", unit="unit")
    elapsed = time.perf_counter() - start
    assert v["n_units"] == 642 and v["p_value"] is not None
    assert elapsed < 30.0, f"{elapsed:.0f} s for 2,000 bootstrap draws and 2,000 permutations at n = 642"


# ---------------------------------------------------------------------------
# End to end: context.toml -> python -m claimcheck.relate -> claims.json
# ---------------------------------------------------------------------------

def test_end_to_end_from_a_context_toml_through_the_command_line(tmp_path):
    work = tmp_path / "kendall_claims"
    shutil.copytree(FIXTURE, work)
    out = tmp_path / "claims.json"
    done = subprocess.run([sys.executable, "-m", "claimcheck.relate", str(work), str(out)], capture_output=True, text=True,
                          cwd=str(REPO), env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stdout + done.stderr
    got = {r["id"]: r for r in json.loads(out.read_text())["relationships"]}
    assert set(got) == {"K1_monotone_is_one", "K2_reversed_is_minus_one", "K3_wrong_sign_is_refuted", "K4_constant_is_unverifiable",
                        "K5_range_agrees"}
    assert got["K1_monotone_is_one"]["verdict"] == AGREES and got["K1_monotone_is_one"]["estimate"] == 1.0
    assert got["K2_reversed_is_minus_one"]["verdict"] == AGREES and got["K2_reversed_is_minus_one"]["estimate"] == -1.0
    assert got["K3_wrong_sign_is_refuted"]["verdict"] == REFUTED
    assert got["K4_constant_is_unverifiable"]["verdict"] == UNVERIFIABLE
    assert got["K5_range_agrees"]["verdict"] == AGREES
    assert all(r["method"] == "kendall" and r["n_units"] == 12 for r in got.values() if r["verdict"] != UNVERIFIABLE)
    assert len(json.loads(out.read_text())["relate_sha256"]) == 16
