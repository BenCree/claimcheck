"""`topn_recovery`: "k of the K actives are in the N-compound shortlist, out of M screened", with the exact hypergeometric p.

Added 2026-11-12. A top-N recovery count had been headed for `binomial_count` on a table of the actives alone, whose p is the
two-sided binomial against 0.5. For 7 actives and the first 50 of 357 that p is 0.016 at 0 of 7, the likeliest count under chance,
and 1.0 at 4 of 7, where the chance of 4 or more is 0.0087. The chance share is 50 / 357 = 0.1401, and the count under chance is
hypergeometric, not binomial, because the actives compete for the same 50 places: a binomial tail at 0.1401 is 3% too large at 3 of 7
and 46% too large at 7 of 7. Every fixture is checked against integer arithmetic, and against scipy where it is installed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from claimcheck.relate import (
    AGREES,
    ESTIMATORS,
    REFUTED,
    UNVERIFIABLE,
    _topn_recovery_share,
    check,
    clopper_pearson,
    hypergeom_tails,
    sign_test_p,
)

K, N, M = 7, 50, 357                                           # actives, shortlist places, library


def pmf(i: int, k_marked: int = K, n_drawn: int = N, n_total: int = M) -> float:
    return math.comb(k_marked, i) * math.comb(n_total - k_marked, n_drawn - i) / math.comb(n_total, n_drawn)


# ── the tails ────────────────────────────────────────────────────────────────

def test_tails_are_the_sums_of_the_exact_probabilities_and_complement_each_other():
    assert sum(pmf(i) for i in range(K + 1)) == pytest.approx(1.0, abs=1e-12)
    for k in range(K + 1):
        lower, upper = hypergeom_tails(k, K, N, M)
        assert lower == pytest.approx(sum(pmf(i) for i in range(k + 1)), abs=1e-14)
        assert upper == pytest.approx(sum(pmf(i) for i in range(k, K + 1)), abs=1e-14)
        assert lower + upper - pmf(k) == pytest.approx(1.0, abs=1e-12)
    assert hypergeom_tails(0, K, N, M)[1] == 1.0 and hypergeom_tails(K, K, N, M)[0] == 1.0
    assert hypergeom_tails(3, K, N, M)[1] == pytest.approx(0.060181, abs=1e-6)
    assert hypergeom_tails(4, K, N, M)[1] == pytest.approx(0.008713, abs=1e-6)
    assert hypergeom_tails(K, K, N, M)[1] == pytest.approx(math.comb(350, 43) / math.comb(357, 50), rel=1e-12)


def test_tails_agree_with_scipy_hypergeom_and_fisher_exact():
    stats = pytest.importorskip("scipy.stats")
    for k_marked, n_drawn, n_total in [(K, N, M), (29, 73, 1479), (5, 5, 12), (40, 50, 200)]:
        for k in range(max(0, n_drawn - (n_total - k_marked)), min(k_marked, n_drawn) + 1):
            lower, upper = hypergeom_tails(k, k_marked, n_drawn, n_total)
            assert upper == pytest.approx(stats.hypergeom.sf(k - 1, n_total, k_marked, n_drawn), rel=1e-9, abs=1e-300)
            assert lower == pytest.approx(stats.hypergeom.cdf(k, n_total, k_marked, n_drawn), rel=1e-9, abs=1e-300)
            tab = [[k, k_marked - k], [n_drawn - k, n_total - k_marked - n_drawn + k]]
            assert upper == pytest.approx(stats.fisher_exact(tab, alternative="greater").pvalue, rel=1e-9, abs=1e-300)
            assert lower == pytest.approx(stats.fisher_exact(tab, alternative="less").pvalue, rel=1e-9, abs=1e-300)


def test_outside_the_support_or_an_impossible_draw_is_none():
    assert hypergeom_tails(8, K, N, M) is None and hypergeom_tails(-1, K, N, M) is None
    assert hypergeom_tails(0, 5, 10, 12) is None               # 10 drawn from 12 with 5 marked: at least 3 are marked
    assert hypergeom_tails(3, 5, 10, 12) is not None
    assert hypergeom_tails(1, 8, 3, 7) is None and hypergeom_tails(0, 0, 0, 0) is None


def test_the_binomial_at_the_chance_share_overstates_the_upper_tail():
    """The actives are not independent trials. The binomial tail at 50/357 against the exact one: within 0.5% at k = 1 and 2, then
    3% too large at 3 of 7, 8% at 4, and 46% at 7 of 7. In the other direction the fixture would fail on a function returning the
    binomial."""
    p0 = N / M
    ratio = {}
    for k in range(1, K + 1):
        binom = sum(math.comb(K, i) * p0 ** i * (1 - p0) ** (K - i) for i in range(k, K + 1))
        ratio[k] = binom / hypergeom_tails(k, K, N, M)[1]
    assert ratio[1] == pytest.approx(0.995, abs=0.001) and ratio[2] == pytest.approx(1.002, abs=0.001)
    assert ratio[3] == pytest.approx(1.031, abs=0.001) and ratio[4] == pytest.approx(1.084, abs=0.001)
    assert ratio[7] == pytest.approx(1.463, abs=0.001)


def test_the_p_against_one_half_answers_a_different_question():
    """Why this method exists: the p `binomial_count` reports for the same counts."""
    assert sign_test_p(0, 7) == pytest.approx(0.015625) and pmf(0) == pytest.approx(0.3444, abs=5e-4)   # "significant", and the mode
    assert sign_test_p(4, 7) == 1.0 and hypergeom_tails(4, K, N, M)[1] < 0.01


def test_the_share_estimator_and_its_registration():
    assert _topn_recovery_share([1, 1, 1, 0, 0], [1, 0, 1, 1, 0]) == pytest.approx(2 / 3)
    assert _topn_recovery_share([0, 0], [1, 0]) is None
    fn, kind = ESTIMATORS["topn_recovery"]
    assert fn is _topn_recovery_share and kind == "value"


# ── through `check` ──────────────────────────────────────────────────────────

VARS = ('[[variables]]\ndataset = "d"\nname = "is_hit"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "in_top"\nunit = "1"\n')


def library(k: int, n_actives: int = K, n_top: int = N, n_total: int = M) -> list[tuple[str, object, object]]:
    """A library table with k of the actives shortlisted: rows are (compound, is_hit, in_top)."""
    rows = []
    for i in range(n_total):
        hit = i < n_actives
        top = (i < k) if hit else (i - n_actives < n_top - k)
        rows.append((f"c{i}", int(hit), int(top)))
    return rows


def verdicts(tmp_path: Path, rows: list[tuple], rels: str) -> dict:
    (tmp_path / "d.csv").write_text("\n".join(["compound,is_hit,in_top"] + [",".join(str(v) for v in r) for r in rows]) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "t"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + VARS + "\n" + rels)
    return {r["id"]: r for r in check(tmp_path)["relationships"]}


def rel(rid: str, extra: str, x: str = "d.is_hit", y: str = "d.in_top", unit: str | None = "compound") -> str:
    s = f'[[relationships]]\nid = "{rid}"\nx = "{x}"\ny = "{y}"\nkind = "value"\nmethod = "topn_recovery"\n{extra}\n'
    return s + (f'resampling_unit = "{unit}"\n\n' if unit else "\n")


def test_three_of_seven_in_the_first_50_of_357(tmp_path):
    row = verdicts(tmp_path, library(3), rel("r", "claimed_range = [0.05, 0.9]"))["r"]
    assert (row["n_success"], row["n_actives"], row["n_shortlist"], row["n_library"]) == (3, 7, 50, 357)
    assert row["estimate"] == pytest.approx(3 / 7, abs=1e-6) and row["n_units"] == 7
    lo, hi = clopper_pearson(3, 7)
    assert (row["ci_lo"], row["ci_hi"]) == pytest.approx((lo, hi), abs=1e-6) and row["interval"] == "clopper_pearson"
    assert row["p_null"] == pytest.approx(50 / 357, abs=1e-6)
    assert row["enrichment_factor"] == pytest.approx((3 / 7) / (50 / 357), abs=1e-5)
    assert row["p_enrichment"] == pytest.approx(0.060181, abs=1e-6)
    assert row["p_depletion"] == pytest.approx(1 - hypergeom_tails(4, K, N, M)[1], abs=1e-12)
    assert row["p_value"] == pytest.approx(2 * 0.060181, abs=2e-6)          # twice the smaller tail
    assert "hypergeometric" in row["p_method"] and "twice the smaller tail" in row["p_method"]
    assert "conservative" in row["interval_source"]
    assert row["r_squared"] is None and row["max_share"] == 1.0
    assert row["verdict"] == AGREES and "3 of 7 actives" in row["why"] and "50 shortlisted of 357" in row["why"]


def test_the_verdict_is_the_interval_against_the_claim(tmp_path):
    out = verdicts(tmp_path, library(6),
                   rel("chance", f"claimed = {50 / 357:.6f}") + rel("high", "claimed_range = [0.4, 1.0]") + rel("straddle", "claimed_range = [0.5, 1.0]"))
    lo, hi = clopper_pearson(6, 7)
    assert lo > 50 / 357
    assert out["chance"]["verdict"] == REFUTED                 # 6 of 7: the chance share lies outside [0.421, 0.996]
    assert out["high"]["verdict"] == AGREES
    assert out["straddle"]["verdict"] == UNVERIFIABLE
    assert out["chance"]["p_enrichment"] == pytest.approx(hypergeom_tails(6, K, N, M)[1], rel=1e-9)


def test_zero_of_seven_is_not_significant(tmp_path):
    row = verdicts(tmp_path, library(0), rel("r", "claimed_range = [0.0, 0.5]"))["r"]
    assert row["estimate"] == 0.0 and row["p_enrichment"] == 1.0
    assert row["p_depletion"] == pytest.approx(pmf(0), abs=1e-12) and row["p_value"] == pytest.approx(2 * pmf(0), abs=1e-12)
    assert row["p_value"] > 0.6


def test_the_counts_come_from_the_table_and_missing_rows_are_dropped_and_counted(tmp_path):
    rows = library(3) + [("m1", "", 1), ("m2", 0, "NA"), ("m3", "nan", 0)]
    row = verdicts(tmp_path, rows, rel("r", "claimed_range = [0.0, 1.0]"))["r"]
    assert row["dropped"] == 3 and row["n_library"] == 357 and row["n_rows"] == 357
    words = [(c, "yes" if h else "no", "TRUE" if t else "false") for c, h, t in library(3)]
    (tmp_path / "w").mkdir()
    assert verdicts(tmp_path / "w", words, rel("r", "claimed_range = [0.0, 1.0]"))["r"]["n_success"] == 3


def test_swapping_label_and_shortlist_keeps_the_p_and_changes_the_estimate(tmp_path):
    """The 2 x 2 table is symmetric in its margins, so the hypergeometric p does not care which column is which. The estimate does:
    k / K is recall, k / N is precision. The row carries all four counts so a swapped declaration is visible."""
    out = verdicts(tmp_path, library(3), rel("ok", "claimed_range = [0.0, 1.0]")
                   + rel("swapped", "claimed_range = [0.0, 1.0]", x="d.in_top", y="d.is_hit"))
    assert out["swapped"]["p_enrichment"] == pytest.approx(out["ok"]["p_enrichment"], rel=1e-9)
    assert out["ok"]["estimate"] == pytest.approx(3 / 7, abs=1e-6) and out["swapped"]["estimate"] == pytest.approx(3 / 50, abs=1e-6)
    assert out["swapped"]["n_actives"] == 50 and out["swapped"]["n_shortlist"] == 7


def test_a_shortlist_smaller_than_the_actives_caps_the_share(tmp_path):
    row = verdicts(tmp_path, library(4, n_actives=20, n_top=10, n_total=100), rel("r", "claimed_range = [0.0, 0.5]"))["r"]
    assert row["max_share"] == 0.5 and row["estimate"] == pytest.approx(0.2)
    assert row["p_enrichment"] == pytest.approx(hypergeom_tails(4, 20, 10, 100)[1], rel=1e-9)


def test_refusals_name_the_reason(tmp_path):
    cases = {
        "one_col": (library(3), rel("r", "claimed = 0.5", y="d.is_hit"), "two-column"),
        "not_ind": ([("a", 1, 0.7)] + library(3), rel("r", "claimed = 0.5"), "not an indicator"),
        "dup": (library(3) + [("c0", 1, 1)], rel("r", "claimed = 0.5"), "more than one row"),
        "no_act": ([(c, 0, t) for c, _, t in library(3)], rel("r", "claimed = 0.5"), "nothing to recover"),
        "no_top": ([(c, h, 0) for c, h, _ in library(3)], rel("r", "claimed = 0.5"), "cannot show recovery"),
        "all_top": ([(c, h, 1) for c, h, _ in library(3)], rel("r", "claimed = 0.5"), "cannot show recovery"),
        "kind": (library(3), rel("r", "claimed = 0.5").replace('kind = "value"', 'kind = "correlation"'), "computes a value"),
        "no_claim": (library(3), rel("r", ""), "neither"),
    }
    for name, (rows, rels, needle) in cases.items():
        (tmp_path / name).mkdir()
        row = verdicts(tmp_path / name, rows, rels)["r"]
        assert row["verdict"] == UNVERIFIABLE and needle in row["why"], (name, row["why"])


def test_without_a_declared_unit_each_row_is_a_compound(tmp_path):
    row = verdicts(tmp_path, library(3), rel("r", "claimed_range = [0.0, 1.0]", unit=None))["r"]
    assert row["n_success"] == 3 and row["n_library"] == 357 and row["verdict"] == AGREES
