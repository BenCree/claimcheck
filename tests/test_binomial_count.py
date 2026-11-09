"""`binomial_count`: "k of n units pass a rule", with the exact Clopper-Pearson interval in place of the bootstrap.

Added 2026-11-09 for the z-ml paper's open check O38: a share of 8 ligands within a tolerance, adjudicated with the percentile
bootstrap, collapses to [1, 1] at 8 of 8 and to [0.625, 1] at 7 of 8, so "at least 0.75" could agree only at 8 of 8 and the verdict
reported the panel size. The exact interval says how little 8 of 8 shows ([0.631, 1]). Every fixture is checked against closed
forms or against the binomial tail that defines the interval, in both directions.
"""

from __future__ import annotations

import math
from pathlib import Path

from claimcheck.relate import AGREES, ESTIMATORS, REFUTED, UNVERIFIABLE, beta_ppf, check, clopper_pearson, sign_test_p

VARS = ('[[variables]]\ndataset = "d"\nname = "unit"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "v"\nunit = "1"\n')


def project(tmp_path: Path, values: list, rels: str, units: list | None = None) -> Path:
    units = units or [f"u{i}" for i in range(len(values))]
    lines = ["unit,v"] + [f"{u},{v}" for u, v in zip(units, values, strict=True)]
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "b"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + VARS + "\n" + rels)
    return tmp_path


def rel(rid: str, claimed=None, rng=None, unit: str | None = "unit", x: str = "d.v") -> str:
    s = f'[[relationships]]\nid = "{rid}"\nx = "{x}"\ny = "d.v"\nkind = "value"\nmethod = "binomial_count"\n'
    if unit:
        s += f'resampling_unit = "{unit}"\n'
    if claimed is not None:
        s += f"claimed = {claimed}\n"
    if rng is not None:
        s += f"claimed_range = [{rng[0]}, {rng[1]}]\n"
    return s + "\n"


def verdicts(tmp_path: Path, values: list, rels: str, units: list | None = None) -> dict:
    return {r["id"]: r for r in check(project(tmp_path, values, rels, units))["relationships"]}


def tail_ge(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def tail_le(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


# ── the interval itself ──────────────────────────────────────────────────────

def test_closed_forms_at_the_edges():
    """k = n: lower = 0.025 ** (1/n), upper exactly 1. k = 0: lower exactly 0, upper = 1 - 0.025 ** (1/n). k = 1: lower = 1 - 0.975 ** (1/n).
    k = n - 1: upper = 0.975 ** (1/n). 8 of 8 is the textbook [0.6306, 1]."""
    for n in (1, 3, 6, 8, 10, 24):
        lo, hi = clopper_pearson(n, n)
        assert abs(lo - 0.025 ** (1 / n)) < 1e-12 and hi == 1.0
        lo, hi = clopper_pearson(0, n)
        assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / n))) < 1e-12
    for n in (3, 6, 8, 10):
        assert abs(clopper_pearson(1, n)[0] - (1 - 0.975 ** (1 / n))) < 1e-12
        assert abs(clopper_pearson(n - 1, n)[1] - 0.975 ** (1 / n)) < 1e-12
    lo, hi = clopper_pearson(8, 8)
    assert round(lo, 4) == 0.6306 and hi == 1.0


def test_interior_bounds_solve_the_binomial_tails():
    """The definition: at the lower bound P(X >= k) = 0.025, at the upper P(X <= k) = 0.025. A bound that is off in either direction
    moves the tail away from 0.025, so this fails on a wrong bound of either sign."""
    for k, n in ((5, 10), (7, 8), (6, 8), (1, 8), (9, 10), (5, 6), (3, 24)):
        lo, hi = clopper_pearson(k, n)
        assert abs(tail_ge(k, n, lo) - 0.025) < 1e-10
        assert abs(tail_le(k, n, hi) - 0.025) < 1e-10
        assert lo < k / n < hi
        assert abs(tail_ge(k, n, lo + 1e-4) - 0.025) > 1e-6       # a shifted bound does not pass


def test_published_values():
    """Values printed to four places in the standard tables (and by scipy.stats.binomtest(k, n).proportion_ci('exact'))."""
    expect = {(5, 10): (0.1871, 0.8129), (7, 8): (0.4735, 0.9968), (6, 8): (0.3491, 0.9681), (9, 10): (0.5550, 0.9975),
              (6, 6): (0.5407, 1.0), (5, 6): (0.3588, 0.9958), (0, 10): (0.0, 0.3085),
              (5, 8): (0.2449, 0.9148), (4, 6): (0.2228, 0.9567), (3, 6): (0.1181, 0.8819)}       # the last three: docs/binomial_count.md table
    for (k, n), (elo, ehi) in expect.items():
        lo, hi = clopper_pearson(k, n)
        assert round(lo, 4) == elo and round(hi, 4) == ehi, (k, n, lo, hi)


def test_symmetry_and_monotonicity():
    for n in (6, 8, 13):
        prev_lo = -1.0
        for k in range(n + 1):
            lo, hi = clopper_pearson(k, n)
            lo2, hi2 = clopper_pearson(n - k, n)
            assert abs(lo - (1 - hi2)) < 1e-12 and abs(hi - (1 - lo2)) < 1e-12
            assert lo > prev_lo
            prev_lo = lo


def test_out_of_range_inputs():
    assert clopper_pearson(0, 0) is None and clopper_pearson(3, 2) is None and clopper_pearson(-1, 4) is None
    assert beta_ppf(0.0, 2, 3) == 0.0 and beta_ppf(1.0, 2, 3) == 1.0


# ── the claim ─────────────────────────────────────────────────────────────────

def test_eight_of_eight(tmp_path):
    """8 of 8: estimate 1, exact interval [0.6306, 1], p = 2/256. [0.5, 1] agrees; [0.75, 1] is unverifiable (a bootstrap would have
    said [1, 1] and agreed); a point claim of 0.6305 is refuted and 0.6307 agrees, which pins the lower bound to 4 places."""
    v = verdicts(tmp_path, [1] * 8, rel("most", rng=[0.5, 1.0]) + rel("q", rng=[0.75, 1.0]) + rel("below", claimed=0.6305)
                 + rel("at", claimed=0.6307))
    for r in v.values():
        assert r["estimate"] == 1.0 and r["n_success"] == 8 and r["n_units"] == 8 and r["dropped"] == 0
        assert r["ci_lo"] == round(0.025 ** 0.125, 6) and r["ci_hi"] == 1.0
        assert r["p_value"] == 2 / 256 and r["p_null"] == 0.5 and r["interval"] == "clopper_pearson"
        assert r["r_squared"] is None
    assert v["most"]["verdict"] == AGREES
    assert v["q"]["verdict"] == UNVERIFIABLE and "straddles" in v["q"]["why"]
    assert v["below"]["verdict"] == REFUTED and v["at"]["verdict"] == AGREES
    assert "8 of 8 unit(s) pass" in v["most"]["why"] and "Clopper-Pearson" in v["most"]["why"]


def test_six_of_eight_and_seven_of_eight(tmp_path):
    """6 of 8 and 7 of 8 both straddle 0.5: the exact interval cannot place them above a majority at n = 8."""
    v = verdicts(tmp_path, [1] * 6 + [0] * 2, rel("most", rng=[0.5, 1.0]))
    assert v["most"]["verdict"] == UNVERIFIABLE and v["most"]["n_success"] == 6
    assert v["most"]["p_value"] == sign_test_p(6, 8)
    v = verdicts(tmp_path, [1] * 7 + [0], rel("most", rng=[0.5, 1.0]) + rel("weak", rng=[0.45, 1.0]))
    assert v["most"]["verdict"] == UNVERIFIABLE and v["weak"]["verdict"] == AGREES


def test_refuted_when_the_interval_lies_below(tmp_path):
    v = verdicts(tmp_path, [0] * 10, rel("most", rng=[0.5, 1.0]) + rel("none", claimed=0.0))
    assert v["most"]["verdict"] == REFUTED
    assert v["none"]["verdict"] == AGREES and v["none"]["ci_lo"] == 0.0


def test_booleans_and_missing_values(tmp_path):
    v = verdicts(tmp_path, ["true", "False", "yes", "no", "", "nan", 1, 0], rel("c", claimed=0.5))["c"]
    assert v["n_success"] == 3 and v["n_units"] == 6 and v["dropped"] == 2 and v["estimate"] == 0.5


def test_a_number_that_is_not_an_indicator_is_refused(tmp_path):
    v = verdicts(tmp_path, [1, 0, 0.7, 1], rel("c", claimed=0.5))["c"]
    assert v["verdict"] == UNVERIFIABLE and "neither 0 nor 1" in v["why"]
    v = verdicts(tmp_path, [1, 0, "maybe", 1], rel("c", claimed=0.5))["c"]
    assert v["verdict"] == UNVERIFIABLE and "not an indicator" in v["why"]


def test_several_rows_per_unit_are_refused(tmp_path):
    """Three runs of one ligand are not three independent trials; the claim names the units and refuses, and the same rows with one row
    per unit adjudicate."""
    units = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    v = verdicts(tmp_path, [1, 1, 0, 1, 1, 1, 0, 1, 1], rel("c", rng=[0.5, 1.0]), units)["c"]
    assert v["verdict"] == UNVERIFIABLE and "more than one row" in v["why"] and "a, b, c" in v["why"]
    v = verdicts(tmp_path, [1, 1, 0, 1, 1, 1, 0, 1, 1], rel("c", rng=[0.5, 1.0], unit=None))["c"]
    assert v["n_units"] == 9 and v["verdict"] == UNVERIFIABLE


def test_degenerate_cases_are_unverifiable(tmp_path):
    """An all-missing column, an empty unit column, and x != y each come back unverifiable, never a number."""
    v = verdicts(tmp_path, ["", "", "nan"], rel("empty", rng=[0.0, 1.0]))["empty"]
    assert v["verdict"] == UNVERIFIABLE and "no unit has a usable value" in v["why"]
    v = verdicts(tmp_path, [1, 0, 1], rel("blank_units", rng=[0.0, 1.0]), units=["", "", ""])["blank_units"]
    assert v["verdict"] == UNVERIFIABLE and "EMPTY" in v["why"]
    v = verdicts(tmp_path, [1, 0, 1], rel("two", rng=[0.0, 1.0], x="d.unit"))["two"]
    assert v["verdict"] == UNVERIFIABLE and "one-column" in v["why"]


def test_one_unit_still_has_an_interval(tmp_path):
    """Unlike the bootstrap, which needs 3 units, the exact interval exists at n = 1: 1 of 1 is [0.025, 1], which says almost nothing."""
    v = verdicts(tmp_path, [1], rel("one", rng=[0.5, 1.0]))["one"]
    assert v["verdict"] == UNVERIFIABLE and v["ci_lo"] == 0.025 and v["ci_hi"] == 1.0


def test_registered_as_a_value_method_and_the_fraction_is_unchanged(tmp_path):
    assert ESTIMATORS["binomial_count"][1] == "value"
    v = verdicts(tmp_path, [1] * 8, '[[relationships]]\nid = "f"\nx = "d.v"\ny = "d.v"\nkind = "value"\nmethod = "fraction"\n'
                 'resampling_unit = "unit"\nclaimed_range = [0.75, 1.0]\n')["f"]
    assert v["ci_lo"] == 1.0 and v["ci_hi"] == 1.0 and v["verdict"] == AGREES and v["p_value"] is None


def test_kind_must_be_value(tmp_path):
    v = verdicts(tmp_path, [1, 0, 1], '[[relationships]]\nid = "k"\nx = "d.v"\ny = "d.v"\nkind = "correlation"\nmethod = "binomial_count"\n'
                 'claimed = 0.5\n')["k"]
    assert v["verdict"] == UNVERIFIABLE and "computes a value" in v["why"]
