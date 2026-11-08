"""`pooled_sd` and `pooled_se`: the within-group sd of replicates pooled over groups (2026-10-28), docs/pooled_sd.md.

Two-sided throughout: a known answer from synthetic groups with a stated true sd, a hand-computed case, the degenerate cases that must come
back `unverifiable`, and the diagnostics that decide whether pooling is justified, checked on groups that share one sd and on groups that
do not. The distribution functions are stdlib only (a test elsewhere blocks any other import in relate.py), so they are checked against
published table values and, where scipy is installed, against scipy.

COVERAGE TOLERANCES WERE FIXED BEFORE THE FIRST RUN, from the binomial standard error sqrt(0.95 * 0.05 / n): the exact chi-square interval
is asserted within 3 standard errors of 0.95 over 2000 draws; the percentile bootstrap, which is known to undercover with few groups, is
asserted only to be at least 0.88 at 40 groups of 3, and its coverage at 8 groups of 3 (the PD1 design) is recorded, not asserted tight.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from claimcheck.relate import (
    AGREES,
    ESTIMATORS,
    REFUTED,
    UNVERIFIABLE,
    _pooled_groups,
    POOL_BOOTSTRAP_MIN_GROUPS,
    _pooled_sd,
    _pooled_se,
    _pooled_se_arithmetic,
    _sd,
    bootstrap_ci,
    brown_forsythe,
    effective_replicates,
    pool_check_outcome,
    check,
    chi2_cdf,
    chi2_ppf,
    f_sf,
    hartley_fmax_critical,
    pooled_chi2_interval,
    pooling_trigger,
)

VARS = ('[[variables]]\ndataset = "d"\nname = "ligand"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "dg"\nunit = "kcal/mol"\n\n'
        '[[variables]]\ndataset = "d"\nname = "ts"\nunit = "fs"\n')


def project(tmp_path: Path, groups: dict, rels: str, protocol: dict | None = None, extra_rows: list | None = None) -> Path:
    lines = ["ligand,dg,ts"]
    for g, vs in groups.items():
        for v in vs:
            lines.append(f"{g},{v},{(protocol or {}).get(g, 2)}")
    lines += extra_rows or []
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "p"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + VARS + "\n" + rels)
    return tmp_path


def rel(rid: str, method: str = "pooled_sd", claimed=None, rng=None, extra: str = "", group: str | None = "ligand",
        unit: str | None = "ligand", y: str = "dg") -> str:
    s = f'[[relationships]]\nid = "{rid}"\nx = "d.{y}"\ny = "d.{y}"\nkind = "value"\nmethod = "{method}"\n'
    if group:
        s += f'group = "{group}"\n'
    if unit:
        s += f'resampling_unit = "{unit}"\n'
    if claimed is not None:
        s += f"claimed = {claimed}\n"
    if rng is not None:
        s += f"claimed_range = [{rng[0]}, {rng[1]}]\n"
    return s + extra + "\n"


def one(tmp_path: Path, groups: dict, rels: str, rid: str = "c", **kw) -> dict:
    """The row of claim `rid`; when the project holds a single claim it is returned whatever its id."""
    rows = {r["id"]: r for r in check(project(tmp_path, groups, rels, **kw))["relationships"]}
    return rows[rid] if rid in rows or len(rows) != 1 else next(iter(rows.values()))


def fresh(tmp_path: Path, name: str) -> Path:
    """A new empty sub-directory, for a second project in one test."""
    p = tmp_path / name
    p.mkdir()
    return p


def draw(n_groups: int, n_rep: int, sd, seed: int, mean_spread: float = 5.0) -> dict:
    """Synthetic groups: each its own mean (so the between-group spread is not mistaken for noise) and replicates with the stated sd."""
    rng = random.Random(seed)
    sds = sd if isinstance(sd, list) else [sd] * n_groups
    out = {}
    for i in range(n_groups):
        mu = rng.uniform(-mean_spread, mean_spread)             # ONE mean per group, drawn once
        out[f"L{i:02d}"] = [round(rng.gauss(mu, sds[i]), 6) for _ in range(n_rep)]
    return out


# ------------------------------------------------------------------------------------- the distribution functions

@pytest.mark.parametrize("df,lo,hi", [(1, 0.000982, 5.024), (2, 0.0506, 7.378), (10, 3.247, 20.483), (16, 6.908, 28.845)])
def test_chi_square_quantiles_match_the_table(df, lo, hi):
    assert chi2_ppf(0.025, df) == pytest.approx(lo, rel=2e-3)
    assert chi2_ppf(0.975, df) == pytest.approx(hi, rel=2e-3)


def test_chi_square_cdf_inverts_and_bounds():
    assert chi2_cdf(0.0, 4) == 0.0
    assert chi2_cdf(chi2_ppf(0.3, 7), 7) == pytest.approx(0.3, abs=1e-12)
    assert chi2_cdf(2.0, 2) == pytest.approx(1 - math.exp(-1.0), abs=1e-12)     # chi-square on 2 df is exponential
    with pytest.raises(ValueError):
        chi2_ppf(1.0, 3)


def test_f_tail_matches_closed_form_and_table():
    assert f_sf(39.0, 2, 2) == pytest.approx(0.025, abs=1e-12)                  # on (2, 2) the tail is exactly 1 / (1 + f)
    assert f_sf(3.326, 5, 10) == pytest.approx(0.05, abs=5e-4)                  # F(0.95; 5, 10) = 3.326
    assert f_sf(0.0, 3, 3) == 1.0


def test_distribution_functions_against_scipy_when_installed():
    st = pytest.importorskip("scipy.stats")
    for df in (1, 2, 3, 7, 16, 40):
        for p in (0.01, 0.025, 0.5, 0.975, 0.999):
            assert chi2_ppf(p, df) == pytest.approx(st.chi2.ppf(p, df), rel=1e-9)
    for f, d1, d2 in ((0.3, 2, 5), (1.0, 4, 4), (5.5, 7, 20), (40.0, 2, 3)):
        assert f_sf(f, d1, d2) == pytest.approx(st.f.sf(f, d1, d2), rel=1e-9)
    rng = random.Random(5)
    gs = [[rng.gauss(0, s) for _ in range(n)] for s, n in ((1, 5), (2, 6), (4, 7), (1.5, 4))]
    ref = st.levene(*gs, center="median")
    bf = brown_forsythe(gs)
    assert bf["statistic"] == pytest.approx(ref.statistic, rel=1e-9)
    assert bf["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)


def test_pooled_interval_is_the_chi_square_interval():
    lo, hi = pooled_chi2_interval(1.0, 16)
    assert (lo, hi) == (pytest.approx(0.745, abs=1e-3), pytest.approx(1.522, abs=1e-3))     # 16 df: 8 ligands, 3 runs each
    lo, hi = pooled_chi2_interval(1.0, 2)
    assert (lo, hi) == (pytest.approx(0.52, abs=5e-3), pytest.approx(6.29, abs=1e-2))       # one ligand, 3 runs
    assert pooled_chi2_interval(3.0, 16) == pytest.approx(tuple(3.0 * v for v in pooled_chi2_interval(1.0, 16)))


def test_hartley_fmax_for_two_groups_is_the_f_quantile():
    """For k = 2, max/min exceeds r with probability 2 P(F > r), so the 5% critical value is the 0.975 quantile of F(df, df)."""
    for df in (3, 10, 24):
        crit = hartley_fmax_critical(2, df, 0.05)
        assert f_sf(crit, df, df) == pytest.approx(0.025, abs=2e-4)
    assert hartley_fmax_critical(2, 10, 0.05) == pytest.approx(3.717, abs=0.01)             # F(0.975; 10, 10)


def test_hartley_fmax_orders_with_groups_and_df():
    base = hartley_fmax_critical(4, 6)
    assert hartley_fmax_critical(8, 6) > base               # more groups, a larger extreme ratio
    assert hartley_fmax_critical(4, 20) < base              # more degrees of freedom, a tighter ratio
    assert hartley_fmax_critical(8, 2) > 50                 # 8 groups of 3 runs: the bound is so wide the criterion almost never fails
    # a published table gives about 8.4 for 3 groups on 6 df at 5% (a secondary source, see docs/pooled_sd.md)
    assert 8.2 < hartley_fmax_critical(3, 6) < 8.6


def test_numbers_quoted_in_the_docs_are_reproduced():
    """Every F-max bound and chi-square factor in docs/pooled_sd.md (df = 2 means 3 runs per group)."""
    for k, bound in ((2, 39.0), (3, 87.49), (5, 202.38), (6, 266.18), (8, 403.08), (12, 704.41)):
        assert hartley_fmax_critical(k, 2) == pytest.approx(bound, rel=1e-3)
    assert math.sqrt(hartley_fmax_critical(8, 2)) == pytest.approx(20.08, abs=0.02)
    for df, lo, hi in ((2, 0.52, 6.29), (4, 0.60, 2.87), (16, 0.745, 1.522), (40, 0.82, 1.28)):
        assert pooled_chi2_interval(1.0, df) == (pytest.approx(lo, abs=6e-3), pytest.approx(hi, abs=6e-3))


# ------------------------------------------------------------------------------------- the estimator

def test_hand_computed_pooled_sd():
    """[1,2,3] has SS 2 on 2 df and [2,4,6] has SS 8 on 2 df, so the pooled sd is sqrt(10 / 4)."""
    pairs, info = _pooled_groups([{"g": "a", "v": v} for v in (1, 2, 3)] + [{"g": "b", "v": v} for v in (2, 4, 6)], "v", "g", None)
    assert pairs == [(2.0, 2.0), (2.0, 8.0)]
    assert info["df"] == 4 and info["n_groups_used"] == 2
    assert _pooled_sd([p[0] for p in pairs], [p[1] for p in pairs]) == pytest.approx(math.sqrt(2.5), abs=1e-12)
    assert info["group_sd"] == {"a": 1.0, "b": 2.0}


def test_duplicates_give_dahlbergs_method_error():
    """With 2 values per group a group's variance is d^2 / 2 on 1 df, so the pooled sd is sqrt(sum d^2 / 2n), the duplicate-measurement formula
    known as Dahlberg's method error (an algebraic identity; the 1940 book was not read, docs/pooled_sd.md)."""
    pairs_in = [(1.0, 1.4), (5.0, 4.1), (-2.0, -2.9), (0.3, 0.35)]
    rows = [{"g": f"p{i}", "v": v} for i, p in enumerate(pairs_in) for v in p]
    pairs, info = _pooled_groups(rows, "v", "g", None)
    dahlberg = math.sqrt(sum((a - b) ** 2 for a, b in pairs_in) / (2 * len(pairs_in)))
    assert _pooled_sd([p[0] for p in pairs], [p[1] for p in pairs]) == pytest.approx(dahlberg, abs=1e-12)
    assert info["df"] == 4 and info["group_mean"]["p0"] == pytest.approx(1.2)


def test_one_group_pooled_sd_is_the_plain_sd():
    vals = [0.3, 1.9, 2.2, 4.8, 3.1]
    pairs, _ = _pooled_groups([{"g": "a", "v": v} for v in vals], "v", "g", None)
    assert _pooled_sd([p[0] for p in pairs], [p[1] for p in pairs]) == pytest.approx(_sd([], vals), abs=1e-12)


def test_pooled_se_is_the_pooled_sd_over_root_mean_replicates():
    rows = [{"g": g, "v": v} for g, vs in (("a", (1, 2, 3)), ("b", (2, 4, 6)), ("c", (5, 5.5, 6.5))) for v in vs]
    pairs, _ = _pooled_groups(rows, "v", "g", None)
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    assert _pooled_se(xs, ys) == pytest.approx(_pooled_sd(xs, ys) / math.sqrt(3), abs=1e-12)          # three replicates in every group
    # UNBALANCED: groups of 2 and 4. The arithmetic mean is 3 and the harmonic mean 2 / (1/2 + 1/4) = 8/3. The default is the harmonic mean
    # (the n whose 1/n is the average of the groups' 1/n_g), the arithmetic mean is the documented alternative.
    rows = [{"g": g, "v": v} for g, vs in (("a", (1, 2)), ("b", (2, 4, 6, 7))) for v in vs]
    pairs, info = _pooled_groups(rows, "v", "g", None)
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    assert info["mean_replicates"] == 3.0 and info["harmonic_replicates"] == pytest.approx(8.0 / 3.0)
    assert effective_replicates(xs, "harmonic") == pytest.approx(8.0 / 3.0) and effective_replicates(xs, "arithmetic") == 3.0
    assert _pooled_se(xs, ys) == pytest.approx(_pooled_sd(xs, ys) / math.sqrt(8.0 / 3.0), abs=1e-12)
    assert _pooled_se_arithmetic(xs, ys) == pytest.approx(_pooled_sd(xs, ys) / math.sqrt(3.0), abs=1e-12)
    assert _pooled_se(xs, ys) > _pooled_se_arithmetic(xs, ys)           # the harmonic mean is never the larger, so its se never the smaller
    # the identity behind the default: the mean over groups of the variance of a group mean, sigma^2 / n_g, is sigma^2 / harmonic mean
    ns = [x + 1 for x in xs]
    assert sum(1.0 / n for n in ns) / len(ns) == pytest.approx(1.0 / effective_replicates(xs, "harmonic"))


def test_a_group_drawn_twice_keeps_its_own_sum_of_squares():
    """The reason each group is collapsed before the bootstrap: a group repeated in a resample must count twice with ITS degrees of freedom."""
    pairs, _ = _pooled_groups([{"g": g, "v": v} for g, vs in (("a", (0, 2)), ("b", (10, 14))) for v in vs], "v", "g", None)
    twice = [pairs[0], pairs[0], pairs[1]]
    assert _pooled_sd([p[0] for p in twice], [p[1] for p in twice]) == pytest.approx(math.sqrt((2 + 2 + 8) / 3), abs=1e-12)


def test_singleton_groups_are_dropped_and_counted(tmp_path):
    groups = draw(5, 3, 1.0, seed=2) | {"S1": [9.9], "S2": [-4.0]}
    r = one(tmp_path, groups, rel("c", claimed=1.0))
    assert r["n_groups_used"] == 5 and r["n_groups_dropped"] == 2 and r["groups_dropped"] == ["S1", "S2"]
    assert r["df"] == 10 and r["n_rows"] == 17 and r["n_rows_used"] == 15


def test_the_verdict_row_carries_every_new_field(tmp_path):
    r = one(tmp_path, draw(8, 3, 1.0, seed=3), rel("c", claimed=1.0))
    for key in ("estimate", "ci_lo", "ci_hi", "ci_chi2_lo", "ci_chi2_hi", "ci_chi2_note", "df", "df_satterthwaite", "n_groups_used",
                "n_groups_dropped", "group_sd", "group_n", "sd_ratio_max_min", "brown_forsythe", "pool_justified", "pool_criteria",
                "pool_failed", "pool_unchecked", "pooled_note", "mean_replicates", "harmonic_replicates", "n_units", "resampling_unit", "group",
                "interval", "interval_source", "ci_bootstrap_lo", "ci_bootstrap_hi", "ci_satterthwaite_lo", "ci_satterthwaite_hi"):
        assert key in r, key
    assert r["df"] == 16 and r["n_units"] == 8 and r["resampling_unit"] == "ligand" and r["group"] == "ligand"
    assert r["r_squared"] is None and r["p_value"] is None
    assert r["ci_chi2_lo"] < r["estimate"] < r["ci_chi2_hi"]
    assert "chi2 interval decides" in r["why"] and "bootstrap interval is" in r["why"] and "trigger rule" in r["why"]
    assert r["interval"] == "chi2" and "default: 8 groups, below 35" in r["interval_source"]
    assert (r["ci_lo"], r["ci_hi"]) == (r["ci_chi2_lo"], r["ci_chi2_hi"])            # the verdict reads the interval used, both stay in the row
    for c in r["pool_criteria"].values():
        assert c["result"] in ("pass", "fail", "not_computed")
    assert r["pool_criteria"]["fmax"]["threshold"] > 1 and r["pool_criteria"]["fmax"]["df_used"] == 2
    assert "low_power_note" in r["brown_forsythe"]


def test_the_known_answer_agrees_and_a_wrong_claim_is_refuted(tmp_path):
    groups = draw(30, 3, 1.0, seed=4)
    ok = one(tmp_path, groups, rel("ok", claimed=1.0), rid="ok")
    assert ok["verdict"] == AGREES and ok["ci_lo"] <= 1.0 <= ok["ci_hi"]
    bad = one(tmp_path, groups, rel("bad", claimed=3.0), rid="bad")
    assert bad["verdict"] == REFUTED
    assert one(tmp_path, groups, rel("in", rng=[0.5, 2.0]), rid="in")["verdict"] == AGREES
    assert one(tmp_path, groups, rel("out", rng=[2.0, 5.0]), rid="out")["verdict"] == REFUTED


def test_pooled_se_row_is_the_se_not_the_sd(tmp_path):
    groups = draw(30, 3, 1.0, seed=4)
    sd = one(tmp_path, groups, rel("c", "pooled_sd", claimed=1.0))
    se = one(tmp_path, groups, rel("c", "pooled_se", claimed=1.0 / math.sqrt(3)))
    assert se["estimate"] == pytest.approx(sd["estimate"] / math.sqrt(3), abs=2e-6)
    assert se["pooled_sd"] == pytest.approx(sd["estimate"], abs=2e-6)
    assert se["ci_chi2_hi"] == pytest.approx(sd["ci_chi2_hi"] / math.sqrt(3), abs=2e-6)
    assert se["verdict"] == AGREES


# ------------------------------------------------------------------------------------- coverage, and the trigger rule on both kinds of data

def test_the_exact_chi_square_interval_covers_at_the_nominal_rate():
    """Equal-variance normal groups, true sd 1.7, 8 groups of 3: the exact interval covers 0.95 +- 3 standard errors over 4 x 4000 draws.

    RECORD. The first run used one seed (11) and 2000 draws and measured 0.966, 3.3 standard errors high, so the 3-standard-error tolerance
    written beforehand failed. Seeds 12, 13 and 14 at 4000 draws gave 0.953, 0.951 and 0.947, and seed 11 at 4000 draws gave 0.958: a chance
    excursion of one seed, not a bias in `chi2_ppf` (whose table values pass). The test therefore pools four seeds; the tolerance was not widened.
    """
    n_draw, hit = 4000, 0
    for seed in (11, 12, 13, 14):
        rng = random.Random(seed)
        for _ in range(n_draw):
            ss = df = 0
            for _g in range(8):
                v = [rng.gauss(0, 1.7) for _ in range(3)]
                m = sum(v) / 3
                ss += sum((x - m) ** 2 for x in v)
                df += 2
            lo, hi = pooled_chi2_interval(math.sqrt(ss / df), df)
            hit += lo <= 1.7 <= hi
    total = 4 * n_draw
    se = math.sqrt(0.95 * 0.05 / total)
    assert abs(hit / total - 0.95) <= 3 * se, hit / total


def _bootstrap_coverage(n_groups: int, n_rep: int, n_draw: int, seed: int) -> float:
    rng = random.Random(seed)
    hit = 0
    for _ in range(n_draw):
        pairs = []
        for _g in range(n_groups):
            v = [rng.gauss(0, 1.7) for _ in range(n_rep)]
            m = sum(v) / n_rep
            pairs.append((float(n_rep - 1), sum((x - m) ** 2 for x in v)))
        ci = bootstrap_ci(pairs, None, estimate=_pooled_sd)
        hit += ci is not None and ci[0] <= 1.7 <= ci[1]
    return hit / n_draw


def test_the_bootstrap_interval_covers_with_many_groups_and_is_recorded_with_few():
    many = _bootstrap_coverage(40, 3, 150, seed=21)
    assert 0.88 <= many <= 0.995, many
    few = _bootstrap_coverage(8, 3, 300, seed=22)          # the PD1 design; a percentile bootstrap over 8 groups undercovers
    print(f"percentile-bootstrap coverage of the true pooled sd, 8 groups of 3: {few:.3f}; 40 groups of 3: {many:.3f}")
    assert 0.6 < few < 0.995, few


def test_equal_variance_groups_are_poolable(tmp_path):
    r = one(tmp_path, draw(6, 6, 1.0, seed=1), rel("c", claimed=1.0))
    assert r["pool_justified"] is True and r["pool_failed"] == []
    assert r["pool_criteria"]["fmax"]["result"] == "pass" and r["pool_criteria"]["bf"]["result"] == "pass"
    assert r["brown_forsythe"]["p_value"] > 0.10
    assert r["pool_unchecked"] == ["protocol"]              # no protocol column declared, and the row says so


def test_very_different_variances_are_flagged_and_the_verdict_is_not_changed(tmp_path):
    groups = draw(6, 6, [0.2, 0.4, 1.0, 2.0, 4.0, 8.0], seed=1)
    r = one(tmp_path, groups, rel("c", rng=[0.0, 100.0]))
    assert r["pool_justified"] is False
    assert set(r["pool_failed"]) == {"fmax", "bf"}
    assert r["brown_forsythe"]["p_value"] < 0.05
    assert r["sd_ratio_max_min"] > 5
    assert r["verdict"] == AGREES                            # the flag does not touch the verdict
    assert "NOT justified" in r["why"] and "the verdict is unchanged" in r["why"]


def test_brown_forsythe_cannot_be_computed_at_two_replicates_and_says_so():
    two = [[1.0, 2.0], [3.0, 5.0], [0.0, 9.0]]               # both deviations from a 2-value median are equal: no within-group spread
    assert "not_computed" in brown_forsythe(two)
    assert "not_computed" in brown_forsythe([[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]])          # only 2 groups
    mixed = brown_forsythe([[1.0, 2.0], [3.0, 5.0], [0.0, 9.0, 2.0, 4.0]])
    assert "p_value" in mixed and 0.0 <= mixed["p_value"] <= 1.0
    # the trigger rule carries the reason instead of a p
    t = pooling_trigger({"a": 1.0, "b": 1.2, "c": 0.9}, {"a": 1, "b": 1, "c": 1}, brown_forsythe(two))
    assert t["pool_criteria"]["bf"]["result"] == "not_computed" and "bf" in t["pool_unchecked"]


def test_a_zero_group_sd_fails_the_fmax_criterion():
    t = pooling_trigger({"a": 0.0, "b": 1.0, "c": 2.0}, {"a": 2, "b": 2, "c": 2}, {"not_computed": "x"})
    assert t["pool_criteria"]["fmax"]["result"] == "fail" and t["pool_justified"] is False
    t0 = pooling_trigger({"a": 0.0, "b": 0.0}, {"a": 2, "b": 2}, {"not_computed": "x"})
    assert t0["pool_criteria"]["fmax"]["result"] == "not_computed"


def test_protocol_column_is_checked_when_declared(tmp_path):
    groups = draw(6, 3, 1.0, seed=6)
    same = one(tmp_path, groups, rel("c", claimed=1.0, extra='protocol = "ts"\n'), protocol={g: 2 for g in groups})
    assert same["pool_criteria"]["protocol"]["result"] == "pass" and "protocol" not in same["pool_unchecked"]
    mixed = one(tmp_path, groups, rel("c", claimed=1.0, extra='protocol = "ts"\n'), protocol={g: (2 if i < 3 else 4) for i, g in enumerate(groups)})
    assert mixed["pool_criteria"]["protocol"]["result"] == "fail" and "protocol" in mixed["pool_failed"] and mixed["pool_justified"] is False
    assert mixed["pool_criteria"]["protocol"]["distinct_values"] == ["2", "4"]
    missing = one(tmp_path, groups, rel("c", claimed=1.0, extra='protocol = "nope"\n'))
    assert missing["verdict"] == UNVERIFIABLE and "nope" in missing["why"]


def test_satterthwaite_df_equals_the_nominal_df_for_equal_group_sds(tmp_path):
    groups = {f"L{i}": [i, i + 1.0, i + 2.0] for i in range(5)}      # every group has sd exactly 1
    r = one(tmp_path, groups, rel("c", claimed=1.0))
    assert r["df_satterthwaite"] == pytest.approx(10.0, abs=1e-3) and r["sd_ratio_max_min"] == pytest.approx(1.0)
    assert r["estimate"] == pytest.approx(1.0)
    unequal = one(tmp_path, draw(6, 3, [0.1, 0.1, 0.1, 5.0, 5.0, 5.0], seed=8), rel("c", rng=[0, 100]))
    assert unequal["df_satterthwaite"] < unequal["df"]


# ------------------------------------------------------------------------------------- degenerate cases and refusals

def test_a_constant_column_is_unverifiable_with_estimate_zero(tmp_path):
    r = one(tmp_path, {f"L{i}": [1.5, 1.5, 1.5] for i in range(4)}, rel("c", claimed=0.0))
    assert r["verdict"] == UNVERIFIABLE and r["estimate"] == 0.0 and "exactly 0" in r["why"]


def test_fewer_than_two_groups_with_replicates_is_unverifiable(tmp_path):
    r = one(tmp_path, {"A": [1.0, 2.0, 3.0], "B": [4.0], "C": [5.0]}, rel("c", claimed=1.0))
    assert r["verdict"] == UNVERIFIABLE and "fewer than 2" in r["why"] and r["n_groups_used"] == 1 and r["n_groups_dropped"] == 2
    none = one(tmp_path, {"A": [1.0], "B": [4.0]}, rel("c", claimed=1.0))
    assert none["verdict"] == UNVERIFIABLE and none["n_groups_used"] == 0


def test_two_usable_groups_have_the_chi_square_interval_but_no_bootstrap(tmp_path):
    groups = {"A": [1.0, 2.0, 4.0], "B": [3.0, 3.5, 5.5]}
    r = one(tmp_path, groups, rel("c", rng=[0.0, 100.0]))                # the default interval below 35 groups is the chi-square one: a verdict exists
    assert r["verdict"] == AGREES and r["interval"] == "chi2" and r["ci_bootstrap_lo"] is None and "not available" in r["why"]
    assert r["ci_chi2_lo"] < r["estimate"] < r["ci_chi2_hi"]
    b = one(tmp_path, groups, rel("c", rng=[0.0, 100.0], extra='interval = "bootstrap"\n'))
    assert b["verdict"] == UNVERIFIABLE and "fewer than 3 independent units" in b["why"] and "ci_lo" not in b


def test_the_interval_key_chooses_which_interval_decides_and_both_stay_in_the_row(tmp_path):
    groups = draw(8, 3, 1.0, seed=3)
    c = one(tmp_path, groups, rel("c", rng=[0.0, 100.0], extra='interval = "chi2"\n'))
    b = one(tmp_path, groups, rel("c", rng=[0.0, 100.0], extra='interval = "bootstrap"\n'))
    assert c["interval"] == "chi2" and b["interval"] == "bootstrap"
    assert c["interval_source"] == b["interval_source"] == "declared in the claim"
    assert (c["ci_lo"], c["ci_hi"]) == (c["ci_chi2_lo"], c["ci_chi2_hi"]) and (b["ci_lo"], b["ci_hi"]) == (b["ci_bootstrap_lo"], b["ci_bootstrap_hi"])
    for r in (c, b):                                                       # the same two intervals are in both rows
        assert r["ci_chi2_lo"] == c["ci_chi2_lo"] and r["ci_bootstrap_hi"] == b["ci_bootstrap_hi"]
    assert (b["ci_hi"] - b["ci_lo"]) < (c["ci_hi"] - c["ci_lo"])           # the bootstrap is the narrower at 8 groups, which is why it is not the default there
    # the verdict follows the interval chosen: a range whose upper bound lies between the two intervals' upper ends
    mid = (b["ci_hi"] + c["ci_hi"]) / 2
    assert one(tmp_path, groups, rel("c", rng=[0.0, mid], extra='interval = "bootstrap"\n'))["verdict"] == AGREES
    assert one(tmp_path, groups, rel("c", rng=[0.0, mid], extra='interval = "chi2"\n'))["verdict"] == UNVERIFIABLE
    bad = one(tmp_path, groups, rel("c", claimed=1.0, extra='interval = "bayes"\n'))
    assert bad["verdict"] == UNVERIFIABLE and "'bayes'" in bad["why"]


def test_the_default_interval_switches_to_the_bootstrap_at_the_cutoff(tmp_path):
    assert POOL_BOOTSTRAP_MIN_GROUPS == 35
    below = one(tmp_path, draw(POOL_BOOTSTRAP_MIN_GROUPS - 1, 3, 1.0, seed=5), rel("c", rng=[0.0, 100.0]))
    at = one(tmp_path, draw(POOL_BOOTSTRAP_MIN_GROUPS, 3, 1.0, seed=5), rel("c", rng=[0.0, 100.0]))
    assert below["interval"] == "chi2" and at["interval"] == "bootstrap"
    assert "at or above 35" in at["interval_source"] and "below 35" in below["interval_source"]


def test_pooled_se_replicate_mean_key(tmp_path):
    """Unbalanced groups (2 and 4 runs alternating): harmonic by default, arithmetic on request, and the row says which."""
    groups = {f"L{i}": ([1.0 + i, 2.5 + i] if i % 2 else [1.0 + i, 2.0 + i, 3.5 + i, 3.0 + i]) for i in range(6)}
    h = one(tmp_path, groups, rel("c", "pooled_se", rng=[0.0, 100.0]))
    a = one(tmp_path, groups, rel("c", "pooled_se", rng=[0.0, 100.0], extra='replicate_mean = "arithmetic"\n'))
    assert h["replicate_mean"] == "harmonic" and a["replicate_mean"] == "arithmetic"
    assert h["pooled_sd"] == pytest.approx(a["pooled_sd"], abs=1e-5)
    assert h["effective_replicates"] == pytest.approx(8 / 3, abs=1e-5) and a["effective_replicates"] == pytest.approx(3.0)
    assert h["estimate"] == pytest.approx(h["pooled_sd"] / math.sqrt(8 / 3), abs=2e-6) and a["estimate"] == pytest.approx(a["pooled_sd"] / math.sqrt(3), abs=2e-6)
    assert h["estimate"] > a["estimate"]
    bad = one(tmp_path, groups, rel("c", "pooled_se", claimed=1.0, extra='replicate_mean = "median"\n'))
    assert bad["verdict"] == UNVERIFIABLE and "'median'" in bad["why"]


def test_a_missing_group_column_is_named(tmp_path):
    r = one(tmp_path, draw(4, 3, 1.0, seed=1), rel("c", claimed=1.0, group="batch", unit="batch"))
    assert r["verdict"] == UNVERIFIABLE and "'batch'" in r["why"] and "not a column" in r["why"]


def test_group_is_required_and_must_be_the_resampling_unit(tmp_path):
    groups = draw(5, 3, 1.0, seed=1)
    nog = one(tmp_path, groups, rel("c", claimed=1.0, group=None, unit="ligand"))
    assert nog["verdict"] == UNVERIFIABLE and "group" in nog["why"]
    other = one(tmp_path, groups, rel("c", claimed=1.0, group="ligand", unit="ts"))
    assert other["verdict"] == UNVERIFIABLE and "'ligand'" in other["why"] and "'ts'" in other["why"]
    implied = one(tmp_path, groups, rel("c", claimed=1.0, unit=None))        # the group is the unit when none is stated
    assert implied["resampling_unit"] == "ligand" and implied["verdict"] != "error" and "ci_lo" in implied


def test_x_must_equal_y(tmp_path):
    ctx = rel("c", claimed=1.0).replace('x = "d.dg"', 'x = "d.ts"')
    r = one(tmp_path, draw(5, 3, 1.0, seed=1), ctx)
    assert r["verdict"] == UNVERIFIABLE and "one-column" in r["why"]


def test_a_non_finite_value_makes_the_claim_unverifiable_naming_the_count_and_groups(tmp_path):
    """A non-finite value is a failed cell with a reason, never a number: the claim is not computed on the rest."""
    groups = draw(5, 3, 1.0, seed=7)
    for method in ("pooled_sd", "pooled_se", "pool_check"):
        r = one(tmp_path, groups, rel("c", method, claimed=1.0), extra_rows=["L00,nan,2", "L02,inf,2", "L02,-inf,2"])
        assert r["verdict"] == UNVERIFIABLE and "estimate" not in r and r["n_nonfinite"] == 3 and r["n_not_a_number"] == 0
        assert "3 non-finite" in r["why"] and "L00 (1)" in r["why"] and "L02 (2)" in r["why"] and "failed cell" in r["why"]
    blank = one(tmp_path, groups, rel("c", claimed=1.0), extra_rows=["L01,,2"])
    assert blank["verdict"] == UNVERIFIABLE and blank["n_not_a_number"] == 1 and "blank or not a number" in blank["why"] and "L01 (1)" in blank["why"]
    clean = one(tmp_path, groups, rel("c", rng=[0.0, 100.0]))
    assert clean["verdict"] == AGREES and clean["n_nonfinite"] == 0


def test_empty_group_labels_are_still_dropped_and_counted(tmp_path):
    r = one(tmp_path, draw(5, 3, 1.0, seed=7), rel("c", rng=[0.0, 100.0]), extra_rows=[",3.3,2", ",4.4,2"])
    assert r["dropped_no_group"] == 2 and r["n_rows"] == 15 and r["df"] == 10 and r["verdict"] == AGREES


def test_kind_must_be_value(tmp_path):
    bad = rel("c", claimed=1.0).replace('kind = "value"', 'kind = "correlation"')
    r = one(tmp_path, draw(5, 3, 1.0, seed=1), bad)
    assert r["verdict"] == UNVERIFIABLE and "computes a value" in r["why"]


WORKED = {"L1": [10.2, 10.9, 10.5], "L2": [-3.1, -2.6, -2.9], "L3": [4.0, 4.8, 4.3], "L4": [7.7, 7.9, 8.4], "L5": [1.2, 0.6, 1.0],
          "L6": [-6.0, -5.5, -6.2]}


def test_the_worked_example_in_the_docs_is_reproduced(tmp_path):
    """The six-ligand, three-run example of docs/pooled_sd.md, number for number (the bootstrap is seeded, so its interval is exact)."""
    r = one(tmp_path, WORKED, rel("c", rng=[0.0, 1.0]))
    assert r["verdict"] == AGREES and r["interval"] == "chi2"          # 6 groups: the chi-square interval decides by default
    assert r["estimate"] == pytest.approx(0.342377, abs=1e-6)
    assert (r["ci_bootstrap_lo"], r["ci_bootstrap_hi"]) == (pytest.approx(0.305505, abs=2e-6), pytest.approx(0.375648, abs=2e-6))
    assert (r["ci_chi2_lo"], r["ci_chi2_hi"]) == (pytest.approx(0.245514, abs=2e-6), pytest.approx(0.565174, abs=2e-6))
    assert (r["ci_lo"], r["ci_hi"]) == (r["ci_chi2_lo"], r["ci_chi2_hi"])
    assert r["df"] == 12 and r["n_units"] == 6 and r["df_satterthwaite"] == pytest.approx(11.19, abs=0.01)
    assert sorted(r["group_sd"].values()) == pytest.approx([0.251661, 0.305505, 0.351188, 0.360555, 0.360555, 0.404145], abs=1e-6)
    assert r["sd_ratio_max_min"] == pytest.approx(1.61, abs=0.005)
    assert r["brown_forsythe"]["p_value"] == pytest.approx(0.995, abs=0.001)
    fm = r["pool_criteria"]["fmax"]
    assert fm["variance_ratio"] == pytest.approx(2.58, abs=0.01) and fm["threshold"] == pytest.approx(266.18, rel=1e-3) and fm["k"] == 6
    assert r["pool_justified"] is True and r["pool_unchecked"] == ["protocol"]
    # the aside in the docs: a range of [0, 0.5] agrees by the bootstrap interval and straddles the bound by the chi-square one
    assert r["ci_bootstrap_hi"] < 0.5 < r["ci_chi2_hi"]
    assert one(tmp_path, WORKED, rel("c", rng=[0.0, 0.5]))["verdict"] == UNVERIFIABLE
    assert one(tmp_path, WORKED, rel("c", rng=[0.0, 0.5], extra='interval = "bootstrap"\n'))["verdict"] == AGREES


# ------------------------------------------------------------------------------------- illustrative examples of docs/pooled_sd.md
# ILLUSTRATIVE SYNTHETIC NUMBERS, drawn by `draw` with the seeds below (one mean per group; normal replicates). The docs quote the results.

SD_8 = [0.10 + 0.20 * i / 7 for i in range(8)]                    # true run-to-run sds, 0.10 to 0.30 kcal/mol


def example_a():
    """(a) 8 ligands, 3 runs each, true sds 0.10 to 0.30, one time step (2 fs)."""
    d = draw(8, 3, SD_8, seed=11)
    return d, {g: 2 for g in d}


def example_b():
    """(b) 8 ligands, 2 runs each, the same sds, one time step."""
    d = draw(8, 2, SD_8, seed=12)
    return d, {g: 2 for g in d}


def example_c():
    """(c) 8 ligands, 3 runs each: the first four at 2 fs (true sds 0.10 to 0.20), the last four at 4 fs (0.40 to 0.60)."""
    d = draw(8, 3, [0.10, 0.13, 0.16, 0.20, 0.40, 0.48, 0.55, 0.60], seed=21)
    return d, {g: (2 if i < 4 else 4) for i, g in enumerate(d)}


PROTO = 'protocol = "ts"\n'


def test_example_a_all_three_criteria_computed_and_passing(tmp_path):
    d, proto = example_a()
    r = one(tmp_path, d, rel("c", rng=[0.0, 0.5], extra=PROTO), protocol=proto)
    assert r["pool_justified"] is True and r["pool_unchecked"] == [] and r["pool_failed"] == []
    assert {n: c["result"] for n, c in r["pool_criteria"].items()} == {"fmax": "pass", "bf": "pass", "protocol": "pass"}
    assert r["estimate"] == pytest.approx(0.144906, abs=1e-6) and r["df"] == 16 and r["interval"] == "chi2"
    assert (r["ci_chi2_lo"], r["ci_chi2_hi"]) == (pytest.approx(0.107921, abs=2e-6), pytest.approx(0.220536, abs=2e-6))
    assert (r["ci_bootstrap_lo"], r["ci_bootstrap_hi"]) == (pytest.approx(0.081612, abs=2e-6), pytest.approx(0.201484, abs=2e-6))
    assert r["brown_forsythe"]["p_value"] == pytest.approx(0.698, abs=0.001)
    fm = r["pool_criteria"]["fmax"]
    assert fm["variance_ratio"] == pytest.approx(36.43, abs=0.01) and fm["threshold"] == pytest.approx(403.08, rel=1e-3)
    assert r["sd_ratio_max_min"] == pytest.approx(6.04, abs=0.01)         # the sample sds range over six-fold at 3 runs although the true sds range threefold
    assert r["verdict"] == AGREES


def test_example_b_two_runs_brown_forsythe_not_computable(tmp_path):
    d, proto = example_b()
    r = one(tmp_path, d, rel("c", rng=[0.0, 1.0], extra=PROTO), protocol=proto)
    assert r["pool_criteria"]["bf"]["result"] == "not_computed" and "within-group spread" in r["pool_criteria"]["bf"]["reason"]
    assert r["pool_criteria"]["fmax"]["result"] == "pass" and r["pool_criteria"]["fmax"]["informative"] is False
    assert r["pool_criteria"]["protocol"]["result"] == "pass"
    assert r["pool_justified"] is True and r["pool_unchecked"] == ["bf"]           # as built: true when nothing computed failed
    assert r["pool_criteria"]["fmax"]["threshold"] == pytest.approx(47243.7, rel=1e-3)      # 8 groups on 1 df: a 217-fold sd ratio
    assert r["df"] == 8 and r["estimate"] == pytest.approx(0.12621, abs=1e-5)
    assert (r["ci_chi2_lo"], r["ci_chi2_hi"]) == (pytest.approx(0.085249, abs=2e-6), pytest.approx(0.241789, abs=2e-6))


def test_example_c_mixed_time_steps_fail_the_protocol_criterion_and_pool_within_each(tmp_path):
    d, proto = example_c()
    r = one(tmp_path, d, rel("c", rng=[0.0, 0.5], extra=PROTO), protocol=proto)
    assert r["pool_justified"] is False and r["pool_failed"] == ["protocol"]
    assert r["pool_criteria"]["protocol"]["distinct_values"] == ["2", "4"]
    assert r["pool_criteria"]["fmax"]["result"] == "pass" and r["pool_criteria"]["bf"]["result"] == "pass"     # the numbers alone do not show it
    assert r["estimate"] == pytest.approx(0.464595, abs=1e-6) and r["df"] == 16
    names = list(d)
    for steps, sub, est, lo, hi in ((2, names[:4], 0.110064, 0.074344, 0.210858), (4, names[4:], 0.647752, 0.437529, 1.240944)):
        s = one(fresh(tmp_path, f"fs{steps}"), {g: d[g] for g in sub}, rel("c", rng=[0.0, 2.0], extra=PROTO), protocol={g: proto[g] for g in sub})
        assert s["pool_justified"] is True and s["df"] == 8
        assert s["estimate"] == pytest.approx(est, abs=1e-6)
        assert (s["ci_chi2_lo"], s["ci_chi2_hi"]) == (pytest.approx(lo, abs=2e-6), pytest.approx(hi, abs=2e-6))


def test_brown_forsythe_is_not_computed_when_every_group_has_two_values_even_with_rounding(tmp_path):
    """Regression: with 2 values the two absolute deviations from the median are equal in exact arithmetic but differ by rounding in floats,
    and a fixed tiny threshold read that as spread (p = 1e-114). The threshold is relative to the deviations."""
    for seed in range(5):
        bf = brown_forsythe([list(v) for v in draw(8, 2, SD_8, seed=seed).values()])
        assert "not_computed" in bf and "p_value" not in bf, (seed, bf)


# ------------------------------------------------------------------------------------- pool_check: "pooling is justified" as a claim

def check_rel(rid="PC", claimed=1, extra="", **kw) -> str:
    return rel(rid, "pool_check", claimed=claimed, extra=extra, **kw)


def test_pool_check_agrees_for_a_known_good_case(tmp_path):
    d, proto = example_a()
    r = one(tmp_path, d, check_rel(extra=PROTO), rid="PC", protocol=proto)
    assert r["verdict"] == AGREES and r["estimate"] == 1.0 and r["claimed"] == 1.0
    assert "confirmed by fmax, bf" in r["why"] and r["pool_justified"] is True and r["pool_unchecked"] == []
    assert "ci_lo" not in r and r["p_value"] is None and r["r_squared"] is None and r["n_units"] == 8
    for n, c in r["pool_criteria"].items():                              # every criterion carries its value, threshold and source
        assert c["result"] == "pass" and c["source"]
    assert r["pool_criteria"]["fmax"]["threshold"] > r["pool_criteria"]["fmax"]["variance_ratio"]
    assert r["pool_criteria"]["bf"]["threshold"] == 0.10 and "our choice" in r["pool_criteria"]["bf"]["source"]


def test_pool_check_is_refuted_by_a_mixed_protocol_naming_the_values(tmp_path):
    d, proto = example_c()
    r = one(tmp_path, d, check_rel(extra=PROTO), protocol=proto)
    assert r["verdict"] == REFUTED and r["estimate"] == 0.0 and r["pool_failed"] == ["protocol"]
    assert "protocol: 2 distinct values ['2', '4']" in r["why"] and "NOT justified" in r["why"]


def test_pool_check_is_refuted_by_wildly_unequal_variances_naming_both_numbers(tmp_path):
    d = draw(6, 6, [0.2, 0.4, 1.0, 2.0, 4.0, 8.0], seed=1)
    r = one(tmp_path, d, check_rel())
    assert r["verdict"] == REFUTED and r["estimate"] == 0.0 and set(r["pool_failed"]) == {"fmax", "bf"}
    assert "fmax: variance ratio" in r["why"] and "exceeds the F-max bound" in r["why"] and "bf: Brown-Forsythe p = " in r["why"]
    assert r["brown_forsythe"]["p_value"] < 0.05 and r["pool_criteria"]["fmax"]["variance_ratio"] > r["pool_criteria"]["fmax"]["threshold"]


def test_pool_check_with_two_runs_per_group_cannot_be_confirmed_only_refuted(tmp_path):
    d, proto = example_b()
    r = one(tmp_path, d, check_rel(extra=PROTO), protocol=proto)
    assert r["verdict"] == UNVERIFIABLE and r["estimate"] == 1.0
    assert "refuted here but not confirmed" in r["why"] and "bf" in r["pool_unchecked"]
    # the same two-run groups with a mixed protocol ARE refuted: the protocol criterion can fail even where nothing can confirm
    mixed = one(fresh(tmp_path, "m"), d, check_rel(extra=PROTO), protocol={g: (2 if i < 4 else 4) for i, g in enumerate(d)})
    assert mixed["verdict"] == REFUTED and mixed["pool_failed"] == ["protocol"]
    # two-run groups whose sds differ 30-fold stay unverifiable (F-max at 1 df cannot reject that), the documented limit; an extreme ratio
    # (500-fold) is still refuted, so F-max can fail without being able to confirm
    mild = {f"L{i}": [0.0, s] for i, s in enumerate([0.1, 0.2, 0.5, 1.0, 2.0, 3.0])}
    w = one(fresh(tmp_path, "w"), mild, check_rel())
    assert w["verdict"] == UNVERIFIABLE and w["pool_criteria"]["fmax"]["informative"] is False and w["pool_criteria"]["fmax"]["result"] == "pass"
    wild = {f"L{i}": [0.0, s] for i, s in enumerate([0.01, 0.02, 0.05, 0.1, 1.0, 5.0])}
    x = one(fresh(tmp_path, "x"), wild, check_rel())
    assert x["verdict"] == REFUTED and x["pool_failed"] == ["fmax"]


def test_pool_check_without_a_protocol_column_confirms_and_lists_it_unchecked(tmp_path):
    d, _ = example_a()
    r = one(tmp_path, d, check_rel())
    assert r["verdict"] == AGREES and r["pool_unchecked"] == ["protocol"] and "not computed: protocol" in r["why"]


def test_pool_check_with_nothing_computable_is_unverifiable_and_has_no_estimate(tmp_path):
    r = one(tmp_path, {f"L{i}": [1.5, 1.5, 1.5] for i in range(4)}, check_rel())
    assert r["verdict"] == UNVERIFIABLE and "estimate" not in r and "no criterion could be computed" in r["why"]


def test_pool_check_refusals(tmp_path):
    d, _ = example_a()
    assert "claimed = 0" in one(tmp_path, d, check_rel(claimed=0))["why"]
    assert one(tmp_path, d, check_rel(claimed=0))["verdict"] == UNVERIFIABLE
    rr = rel("PC", "pool_check", rng=[0, 1])
    assert one(tmp_path, d, rr)["verdict"] == UNVERIFIABLE and "claimed_range" in one(tmp_path, d, rr)["why"]
    assert one(tmp_path, d, check_rel(group=None, unit="ligand"))["verdict"] == UNVERIFIABLE
    assert one(tmp_path, d, check_rel(group="batch", unit="batch"))["verdict"] == UNVERIFIABLE
    one_group = one(tmp_path, {"A": [1.0, 2.0, 4.0], "B": [3.0]}, check_rel())
    assert one_group["verdict"] == UNVERIFIABLE and "fewer than 2" in one_group["why"]


def test_pool_check_outcome_function_both_ways():
    ok = {"fmax": {"result": "pass", "informative": True}, "bf": {"result": "not_computed"}, "protocol": {"result": "not_computed"}}
    assert pool_check_outcome(ok)[:2] == (AGREES, 1.0)
    weak = {"fmax": {"result": "pass", "informative": False}, "bf": {"result": "not_computed"}, "protocol": {"result": "pass"}}
    assert pool_check_outcome(weak)[:2] == (UNVERIFIABLE, 1.0)
    bad = {"fmax": {"result": "fail", "variance_ratio": None, "threshold": 5.0, "k": 4, "df_used": 2}, "bf": {"result": "not_computed"},
           "protocol": {"result": "not_computed"}}
    v, est, why = pool_check_outcome(bad)
    assert (v, est) == (REFUTED, 0.0) and "infinite (smallest group sd is 0)" in why
    assert pool_check_outcome({n: {"result": "not_computed"} for n in ("fmax", "bf", "protocol")}) == (UNVERIFIABLE, None, "no criterion could be computed (not computed: fmax, bf, protocol)")


def test_a_pooled_claim_shows_its_linked_check_and_keeps_its_own_verdict(tmp_path):
    """The check is reported beside the estimate and never enforced: the same pooled claim with and without `pool_check` has the same verdict."""
    d, proto = example_c()
    both = (rel("SD", "pooled_sd", rng=[0.0, 1.0], extra=PROTO + 'pool_check = "PC"\n') + check_rel(extra=PROTO))     # the pooled claim comes FIRST
    project(tmp_path, d, both, protocol=proto)
    rows = {r["id"]: r for r in check(tmp_path)["relationships"]}
    sd, pc = rows["SD"], rows["PC"]
    assert sd["verdict"] == AGREES and pc["verdict"] == REFUTED                    # both verdicts, side by side
    assert sd["pool_check_verdict"] == REFUTED and sd["pool_check_same_data"] is True and sd["pool_check"] == "PC"
    assert "Pooling check 'PC': REFUTED" in sd["why"] and "does not change its verdict" in sd["why"]
    assert sd["pool_justified"] is False                                           # the pooled row's own flag agrees with the linked claim
    alone = one(fresh(tmp_path, "alone"), d, rel("SD", "pooled_sd", rng=[0.0, 1.0], extra=PROTO), rid="SD", protocol=proto)
    assert alone["verdict"] == sd["verdict"] and (alone["ci_lo"], alone["ci_hi"]) == (sd["ci_lo"], sd["ci_hi"]) and "pool_check_verdict" not in alone


def test_a_link_to_a_missing_or_wrong_claim_is_said_not_silently_dropped(tmp_path):
    d, proto = example_a()
    miss = one(tmp_path, d, rel("SD", "pooled_sd", rng=[0.0, 1.0], extra='pool_check = "NOPE"\n'), rid="SD")
    assert miss["verdict"] == AGREES and miss["pool_check_verdict"] is None and "no pool_check claim with that id" in miss["why"]
    wrong = one(fresh(tmp_path, "w"), d, rel("SD", "pooled_sd", rng=[0.0, 1.0], extra='pool_check = "OTHER"\n') + rel("OTHER", "pooled_se", rng=[0, 1]), rid="SD")
    assert wrong["pool_check_verdict"] is None and "not pool_check" in wrong["why"]
    other_col = one(fresh(tmp_path, "o"), d, rel("SD", "pooled_sd", rng=[0.0, 1.0], extra='pool_check = "PC"\n') + check_rel(y="ts"), rid="SD")
    assert other_col["pool_check_same_data"] is False and "reads a different column or group" in other_col["why"]


def test_default_figures_draw_a_pool_check_claim(tmp_path):
    pytest.importorskip("matplotlib")
    from claimcheck import figures
    from claimcheck.relate import main

    d, proto = example_c()
    root = project(tmp_path, d, check_rel(extra=PROTO) + rel("SD", "pooled_sd", rng=[0.0, 1.0], extra=PROTO + 'pool_check = "PC"\n'), protocol=proto)
    assert main([str(root)]) == 0
    index = figures.build(root, root / "figures" / "claims")
    assert sorted(i["id"] for i in index) == ["PC", "SD"] and (root / "figures" / "claims" / "PC.png").stat().st_size > 5000


# ------------------------------------------------------------------------------------- which interval: recorded coverage, normal and heavy-tailed

def _coverage(n_groups: int, n_rep: int, n_draw: int, seed: int, tails: str) -> tuple[float, float]:
    """(exact chi-square, percentile bootstrap) coverage of the true pooled sd 1.7. `tails` is "normal" or "t5" (Student t on 5 df, scaled to sd 1.7)."""
    rng = random.Random(seed)
    scale = 1.7 / math.sqrt(5.0 / 3.0)

    def draw_one() -> float:
        if tails == "normal":
            return rng.gauss(0.0, 1.7)
        return scale * rng.gauss(0.0, 1.0) / math.sqrt(rng.gammavariate(2.5, 2.0) / 5.0)         # t5 = z / sqrt(chi2_5 / 5)
    hit_c = hit_b = 0
    for _ in range(n_draw):
        pairs = []
        for _g in range(n_groups):
            v = [draw_one() for _ in range(n_rep)]
            m = sum(v) / n_rep
            pairs.append((float(n_rep - 1), sum((x - m) ** 2 for x in v)))
        sd = _pooled_sd([p[0] for p in pairs], [p[1] for p in pairs])
        lo, hi = pooled_chi2_interval(sd, int(sum(p[0] for p in pairs)))
        hit_c += lo <= 1.7 <= hi
        ci = bootstrap_ci(pairs, None, estimate=_pooled_sd)
        hit_b += ci is not None and ci[0] <= 1.7 <= ci[1]
    return hit_c / n_draw, hit_b / n_draw


def test_interval_coverage_with_normal_and_heavy_tailed_replicates_is_recorded():
    """RECORDED, not a pass criterion except for the loose bounds: coverage of the true pooled sd by each interval, with Student-t(5) replicates
    standing for non-normal run-to-run noise. docs/pooled_sd.md quotes the printed values (pytest -s)."""
    out = {}
    for tails in ("normal", "t5"):
        for ng in (8, 40):
            out[(tails, ng)] = _coverage(ng, 3, 300 if ng == 8 else 150, seed=31, tails=tails)
            print(f"coverage of the true pooled sd, {ng} groups of 3, {tails} replicates: chi-square {out[(tails, ng)][0]:.3f}, bootstrap {out[(tails, ng)][1]:.3f}")
    for v in out.values():
        assert all(0.5 < c <= 1.0 for c in v)
    # the one relation the default rests on, asserted: at 8 groups the exact interval is closer to 0.95 than the bootstrap, for normal replicates
    assert abs(out[("normal", 8)][0] - 0.95) < abs(out[("normal", 8)][1] - 0.95)
    # and for heavy-tailed replicates at 8 groups (a 0.1 difference, several standard errors)
    assert out[("t5", 8)][0] > out[("t5", 8)][1]


def test_methods_are_registered():
    assert ESTIMATORS["pooled_sd"][1] == "value" and ESTIMATORS["pooled_se"][1] == "value" and ESTIMATORS["pool_check"][1] == "value"


def test_default_figures_draw_a_pooled_claim(tmp_path):
    """check_claims draws every claim; a pooled one must not crash the figure step."""
    pytest.importorskip("matplotlib")
    import json

    from claimcheck import figures
    from claimcheck.relate import main

    root = project(tmp_path, draw(6, 3, 1.0, seed=9), rel("c", claimed=1.0))
    assert main([str(root)]) == 0
    index = figures.build(root, root / "figures" / "claims")
    assert [i["id"] for i in index] == ["c"] and (root / "figures" / "claims" / "c.png").stat().st_size > 5000
    assert json.loads((root / "results" / "claims.json").read_text())["relationships"][0]["pool_justified"] in (True, False)
