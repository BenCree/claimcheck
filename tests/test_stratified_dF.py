"""`stratified_dF`: a reference free energy re-weighted over two conformational wells, adjudicated by relate (2026-11-09).

    dF - dF_0 = -kT ln (1 - p + p exp(-delta / kT)),   p the share of the second well among frames in either well,
                                                        delta = dF_1 - dF_0 from per-unit estimates in a second dataset.

Every fixture is checked in both directions on data whose truth is known: exact populations and exact deltas give a degenerate
interval at the analytic value (agrees on a tight range around it, refuted on a range around a wrong value), the two limits p = 0 and
p = 1 return the offset and offset + delta, delta = 0 returns the offset for any p, frames in neither well do not move the estimate,
the interval from each source alone is degenerate when that source has no spread, and every malformed declaration is `unverifiable`.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from claimcheck.relate import AGREES, REFUTED, UNVERIFIABLE, check, stratified_shift

KT = 0.5924837   # kcal/mol at 298.15 K

VARS = ('[[variables]]\ndataset = "frames"\nname = "w"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "frames"\nname = "rep"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "deltas"\nname = "d"\nunit = "kcal/mol"\n\n'
        '[[variables]]\ndataset = "deltas"\nname = "inc"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "deltas"\nname = "off"\nunit = "kcal/mol"\n')


def project(tmp_path: Path, frames: list[tuple[str, object]], deltas: list[tuple[str, float, float]], rels: str, vars_: str = VARS) -> Path:
    (tmp_path / "frames.csv").write_text("rep,w\n" + "".join(f"{u},{w}\n" for u, w in frames))
    (tmp_path / "deltas.csv").write_text("inc,d,off\n" + "".join(f"{u},{d!r},{o!r}\n" for u, d, o in deltas))
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "s"\n\n'
        '[[datasets]]\nid = "frames"\norigin = "computed"\nfiles = ["frames.csv"]\n\n'
        '[[datasets]]\nid = "deltas"\norigin = "computed"\nfiles = ["deltas.csv"]\n\n' + vars_ + "\n" + rels)
    return tmp_path


def rel(rid: str, claimed=None, rng=None, extra: str = "", kT=KT, offset: str | None = "off", yunit: str | None = "inc",
        x: str = "frames.w", y: str = "deltas.d") -> str:
    s = (f'[[relationships]]\nid = "{rid}"\nx = "{x}"\ny = "{y}"\nkind = "value"\nmethod = "stratified_dF"\n'
         f'resampling_unit = "rep"\n')
    if kT is not None:
        s += f"kT = {kT!r}\n"
    if yunit is not None:
        s += f'y_resampling_unit = "{yunit}"\n'
    if offset is not None:
        s += f'offset = "{offset}"\n'
    if claimed is not None:
        s += f"claimed = {claimed!r}\n"
    if rng is not None:
        s += f"claimed_range = [{rng[0]!r}, {rng[1]!r}]\n"
    return s + extra + "\n"


def verdicts(tmp_path, frames, deltas, rels, vars_=VARS) -> dict:
    return {r["id"]: r for r in check(project(tmp_path, frames, deltas, rels, vars_))["relationships"]}


def frames_exact(n_rep: int, n1: int, n0: int, n_other: int = 0) -> list[tuple[str, object]]:
    """Every replica holds n1 frames in well 1, n0 in well 0 and n_other in neither: p is the same in every bootstrap draw."""
    out = []
    for r in range(n_rep):
        out += [(f"r{r}", 1)] * n1 + [(f"r{r}", 0)] * n0 + [(f"r{r}", -1)] * n_other
    return out


def deltas_exact(n: int, d: float, off: float = 0.0) -> list[tuple[str, float, float]]:
    return [(f"i{k}", d, off) for k in range(n)]


def direct(p: float, d: float, kT: float = KT) -> float:
    """The textbook form, written independently of the log-sum-exp in relate."""
    return -kT * math.log((1 - p) + p * math.exp(-d / kT))


# ── the function ──────────────────────────────────────────────────────────────

def test_the_shift_is_the_textbook_formula_and_is_bounded_by_its_limits():
    rng = random.Random(7)
    for _ in range(500):
        p, d = rng.random(), rng.uniform(-3, 3)
        s = stratified_shift(p, d, KT)
        assert s == pytest.approx(direct(p, d), abs=1e-12)
        assert min(0.0, d) - 1e-12 <= s <= max(0.0, d) + 1e-12      # a population-weighted log-sum-exp lies between its two terms
    assert stratified_shift(0.0, 1.7, KT) == 0.0
    assert stratified_shift(1.0, 1.7, KT) == pytest.approx(1.7, abs=1e-15)
    assert stratified_shift(0.3, 0.0, KT) == pytest.approx(0.0, abs=1e-15)


def test_no_overflow_far_from_kT():
    s = stratified_shift(0.25, -1000 * KT, KT)                         # exp(+1000) would overflow in the direct form
    assert math.isfinite(s) and s == pytest.approx(-1000 * KT - KT * math.log(0.25), rel=1e-12)
    assert stratified_shift(0.25, 1000 * KT, KT) == pytest.approx(-KT * math.log(0.75), rel=1e-12)


def test_the_function_refuses_impossible_inputs():
    for p, d, kT in ((-0.1, 0.0, KT), (1.1, 0.0, KT), (0.5, 0.0, 0.0), (0.5, float("nan"), KT)):
        with pytest.raises(ValueError):
            stratified_shift(p, d, kT)


# ── known answers through check() ─────────────────────────────────────────────

def test_exact_populations_give_the_analytic_value_both_ways(tmp_path):
    """p = 0.3 in every replica and delta = 0.5 at every increment: every draw is the same, the interval is a point at the analytic
    value. A tight range around it agrees; a tight range around the value for p = 0.5 is refuted."""
    v_true = 0.2 + direct(0.3, 0.5)
    v_wrong = 0.2 + direct(0.5, 0.5)
    v = verdicts(tmp_path, frames_exact(6, 3, 7), deltas_exact(5, 0.5, 0.2),
                 rel("right", rng=[v_true - 1e-9, v_true + 1e-9]) + rel("wrong", rng=[v_wrong - 1e-9, v_wrong + 1e-9]))
    r = v["right"]
    assert r["verdict"] == AGREES, r["why"]
    assert v["wrong"]["verdict"] == REFUTED
    assert r["estimate"] == pytest.approx(v_true, abs=1e-6) and r["ci_lo"] == pytest.approx(v_true, abs=1e-6)
    assert r["ci_hi"] == pytest.approx(v_true, abs=1e-6)
    assert r["p_second_well"] == 0.3 and r["delta"] == 0.5 and r["offset"] == 0.2
    assert r["n_units"] == 6 and r["n_units_y"] == 5 and r["n_first_well"] == 42 and r["n_second_well"] == 18 and r["n_other"] == 0
    assert r["p_value"] is None and r["r_squared"] is None
    assert "the population alone gives" in r["why"]


def test_the_limits_p0_p1_and_delta0(tmp_path):
    (tmp_path / "a").mkdir()
    v = verdicts(tmp_path / "a", frames_exact(4, 0, 10), deltas_exact(4, 0.8, 1.5), rel("p0", rng=[1.5 - 1e-12, 1.5 + 1e-12]))
    assert v["p0"]["verdict"] == AGREES and v["p0"]["shift"] == 0.0
    (tmp_path / "b").mkdir()
    v = verdicts(tmp_path / "b", frames_exact(4, 10, 0), deltas_exact(4, 0.8, 1.5), rel("p1", rng=[2.3 - 1e-9, 2.3 + 1e-9]))
    assert v["p1"]["verdict"] == AGREES and v["p1"]["shift"] == pytest.approx(0.8, abs=1e-6)
    (tmp_path / "c").mkdir()
    rng = random.Random(3)
    fr = [(f"r{r}", 1 if rng.random() < 0.2 + 0.1 * r else 0) for r in range(6) for _ in range(50)]
    v = verdicts(tmp_path / "c", fr, deltas_exact(4, 0.0, -0.3), rel("d0", rng=[-0.3 - 1e-12, -0.3 + 1e-12]))
    assert v["d0"]["verdict"] == AGREES, v["d0"]["why"]           # equal wells: the populations do not matter, however they vary



def test_frames_in_neither_well_are_counted_and_left_out(tmp_path):
    a = verdicts(tmp_path, frames_exact(5, 3, 7), deltas_exact(4, 0.4), rel("a", rng=[-10, 10]))["a"]
    (tmp_path / "o").mkdir()
    b = verdicts(tmp_path / "o", frames_exact(5, 3, 7, n_other=4), deltas_exact(4, 0.4), rel("a", rng=[-10, 10]))["a"]
    assert a["estimate"] == b["estimate"] and b["n_other"] == 20 and b["share_other"] == pytest.approx(20 / 70, abs=1e-6)
    assert a["n_other"] == 0


def test_each_source_alone(tmp_path):
    """p varies across replicas and delta is exact: the delta-only interval is a point and the population-only interval equals the
    joint one. And the other way round."""
    fr = [(f"r{r}", 1 if k < 2 * r else 0) for r in range(1, 7) for k in range(20)]
    r = verdicts(tmp_path, fr, deltas_exact(5, 0.6), rel("a", rng=[-10, 10]))["a"]
    assert r["ci_delta_only"][0] == r["ci_delta_only"][1] == r["estimate"]
    assert r["ci_population_only"] == [r["ci_lo"], r["ci_hi"]] and r["ci_hi"] > r["ci_lo"]
    (tmp_path / "d").mkdir()
    de = [(f"i{k}", 0.2 * k, 0.0) for k in range(6)]
    r = verdicts(tmp_path / "d", frames_exact(5, 4, 6), de, rel("b", rng=[-10, 10]))["b"]
    assert r["ci_population_only"][0] == r["ci_population_only"][1] == r["estimate"]
    assert r["ci_delta_only"] == [r["ci_lo"], r["ci_hi"]] and r["ci_hi"] > r["ci_lo"]
    assert r["delta"] == pytest.approx(0.5, abs=1e-12)


def test_resampling_both_is_wider_than_either(tmp_path):
    fr = [(f"r{r}", 1 if k < 3 * r else 0) for r in range(1, 9) for k in range(30)]
    de = [(f"i{k}", 0.1 * k - 0.2, 0.0) for k in range(10)]
    r = verdicts(tmp_path, fr, de, rel("a", rng=[-10, 10]))["a"]
    w = r["ci_hi"] - r["ci_lo"]
    assert w > r["ci_population_only"][1] - r["ci_population_only"][0]
    assert w > r["ci_delta_only"][1] - r["ci_delta_only"][0]


def test_a_range_straddled_is_unverifiable(tmp_path):
    de = [(f"i{k}", 0.3 * k - 0.6, 0.0) for k in range(5)]
    r = verdicts(tmp_path, frames_exact(5, 5, 5), de, rel("a", rng=[0.0, float("inf")]))["a"]
    assert r["ci_lo"] < 0 < r["ci_hi"] and r["verdict"] == UNVERIFIABLE and "straddles" in r["why"]


# ── refusals ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("extra_rel,needle", [
    (dict(kT=None), "needs `kT`"),
    (dict(kT=0.0), "needs `kT`"),
    (dict(kT=-1.0), "needs `kT`"),
    (dict(y="frames.w"), "two different datasets"),
    (dict(offset="nope"), "offset = 'nope' is not a column"),
    (dict(yunit="nope"), "y_resampling_unit"),
    (dict(y="deltas.missing"), "has no column deltas.missing"),
])
def test_malformed_declarations_are_unverifiable(tmp_path, extra_rel, needle):
    r = verdicts(tmp_path, frames_exact(5, 3, 7), deltas_exact(4, 0.4), rel("a", rng=[-10, 10], **extra_rel))["a"]
    assert r["verdict"] == UNVERIFIABLE and needle in r["why"], r["why"]


def test_a_failed_delta_is_not_dropped(tmp_path):
    de = deltas_exact(4, 0.4) + [("i9", float("nan"), 0.0)]
    r = verdicts(tmp_path, frames_exact(5, 3, 7), de, rel("a", rng=[-10, 10]))["a"]
    assert r["verdict"] == UNVERIFIABLE and "failed cell" in r["why"]


def test_an_offset_that_is_not_constant_is_refused(tmp_path):
    de = [("i0", 0.4, 0.0), ("i1", 0.4, 0.1), ("i2", 0.4, 0.0)]
    r = verdicts(tmp_path, frames_exact(5, 3, 7), de, rel("a", rng=[-10, 10]))["a"]
    assert r["verdict"] == UNVERIFIABLE and "one finite value" in r["why"]


def test_no_frame_in_either_well(tmp_path):
    r = verdicts(tmp_path, frames_exact(5, 0, 0, n_other=10), deltas_exact(4, 0.4), rel("a", rng=[-10, 10]))["a"]
    assert r["verdict"] == UNVERIFIABLE and "neither well is populated" in r["why"]


def test_fewer_than_three_units_on_either_side(tmp_path):
    r = verdicts(tmp_path, frames_exact(2, 3, 7), deltas_exact(4, 0.4), rel("a", rng=[-10, 10]))["a"]
    assert r["verdict"] == UNVERIFIABLE and "fewer than 3 independent units" in r["why"]
    (tmp_path / "y").mkdir()
    r = verdicts(tmp_path / "y", frames_exact(5, 3, 7), deltas_exact(2, 0.4), rel("a", rng=[-10, 10]))["a"]
    assert r["verdict"] == UNVERIFIABLE and "fewer than 3 independent units" in r["why"]
    assert r["estimate"] == pytest.approx(direct(0.3, 0.4), abs=1e-6)      # the point estimate is still on the row


def test_an_undeclared_unit_is_unverifiable(tmp_path):
    vars_ = VARS.replace('name = "d"\nunit = "kcal/mol"', 'name = "d"')
    r = verdicts(tmp_path, frames_exact(5, 3, 7), deltas_exact(4, 0.4), rel("a", rng=[-10, 10]), vars_)["a"]
    assert r["verdict"] == UNVERIFIABLE and "no unit declared" in r["why"]


def test_the_interval_is_reproducible(tmp_path):
    fr = [(f"r{r}", 1 if k < 3 * r else 0) for r in range(1, 9) for k in range(30)]
    de = [(f"i{k}", 0.1 * k - 0.2, 0.0) for k in range(10)]
    a = verdicts(tmp_path, fr, de, rel("a", rng=[-10, 10]))["a"]
    b = verdicts(tmp_path, fr, de, rel("a", rng=[-10, 10]))["a"]
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])
