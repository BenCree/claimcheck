"""`sign_test`, the one value claim that carries a p: "the correction lowers the error on most targets" adjudicated by relate.

One +1/-1 per resampling unit, the estimate is the share of units that are the event, the interval is the bootstrap over
units, and the row carries the exact two-sided binomial p against 0.5. Every fixture is checked in both directions on data
whose truth is known, and the p values are asserted exactly: they are integer arithmetic, not a simulation.
"""

from __future__ import annotations

from pathlib import Path

from claimcheck.relate import AGREES, REFUTED, UNVERIFIABLE, check, sign_test_p

VARS = ('[[variables]]\ndataset = "d"\nname = "unit"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "v"\nunit = "1"\n')


def project(tmp_path: Path, values: list, rels: str, units: list | None = None) -> Path:
    units = units or [f"u{i}" for i in range(len(values))]
    lines = ["unit,v"] + [f"{u},{v}" for u, v in zip(units, values, strict=True)]
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "s"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + VARS + "\n" + rels)
    return tmp_path


def rel(rid: str, claimed=None, rng=None, extra: str = "", unit: str = "unit") -> str:
    s = f'[[relationships]]\nid = "{rid}"\nx = "d.v"\ny = "d.v"\nkind = "value"\nmethod = "sign_test"\n'
    if unit:
        s += f'resampling_unit = "{unit}"\n'
    if claimed is not None:
        s += f"claimed = {claimed}\n"
    if rng is not None:
        s += f"claimed_range = [{rng[0]}, {rng[1]}]\n"
    return s + extra + "\n"


def verdicts(tmp_path: Path, values: list, rels: str, units: list | None = None) -> dict:
    return {r["id"]: r for r in check(project(tmp_path, values, rels, units))["relationships"]}


def ones_and_zeros(k: int, n: int) -> list[int]:
    return [1] * k + [0] * (n - k)


# ── the exact p ───────────────────────────────────────────────────────────────

def test_the_exact_p_is_integer_arithmetic():
    assert sign_test_p(9, 10) == 22 / 1024                        # 2 * (C(10,9) + C(10,10)) / 2**10 = 0.021484375
    assert sign_test_p(1, 10) == 22 / 1024                        # two-sided: the other tail is the same number
    assert sign_test_p(5, 10) == 1.0                              # the doubled tail exceeds 1 and is capped
    assert sign_test_p(14, 15) == 32 / 32768                      # 2 * (C(15,14) + C(15,15)) / 2**15 = 0.0009765625
    assert sign_test_p(10, 10) == 2 / 1024
    assert sign_test_p(0, 0) is None and sign_test_p(3, 2) is None


# ── 9 of 10 ──────────────────────────────────────────────────────────────────

def test_nine_of_ten(tmp_path):
    """p = 0.0215 exactly. The verdict on `claimed_range = [0.75, 1.0]` is UNVERIFIABLE, not agrees: the percentile bootstrap
    of a fraction over 10 units has its 2.5th percentile at 0.7 (for Bin(10, 0.9), P(X <= 6) = 0.013 < 0.025 < P(X <= 7) =
    0.070), so the interval [0.7, 1.0] straddles 0.75 and this design cannot tell whether the fraction is above it. The
    claim this method exists for, "most targets", is `[0.5, 1.0]`, and that agrees. A claim of 0.5 is refuted: 0.5 lies
    below the interval, which is the same fact the p states."""
    v = verdicts(tmp_path, ones_and_zeros(9, 10),
                 rel("quarter", rng=[0.75, 1.0]) + rel("most", rng=[0.5, 1.0]) + rel("half", claimed=0.5))
    for r in v.values():
        assert r["estimate"] == 0.9 and r["n_success"] == 9 and r["n_units"] == 10 and r["n_ties_dropped"] == 0
        assert r["p_value"] == 22 / 1024 and "exact two-sided binomial" in r["p_method"]
        assert r["ci_lo"] == 0.7 and r["ci_hi"] == 1.0
        assert r["event"] == "true" and r["event_source"] == "detected"
        assert r["r_squared"] is None
    assert v["quarter"]["verdict"] == UNVERIFIABLE and "straddles" in v["quarter"]["why"]
    assert v["most"]["verdict"] == AGREES
    assert v["half"]["verdict"] == REFUTED
    assert "(9 of 10 unit(s) are the event, exact two-sided binomial p = 0.0215 against 0.5)" in v["most"]["why"]


# ── 5 of 10 ──────────────────────────────────────────────────────────────────

def test_five_of_ten(tmp_path):
    """p = 1.0: both tails contain everything. `[0.75, 1.0]` is UNVERIFIABLE rather than refuted: the bootstrap over 10
    units at 0.5 runs [0.2, 0.8] (2.5th and 97.5th percentiles of Bin(10, 0.5) / 10), whose upper end is above 0.75, so the
    interval straddles the bound. Refuting would need the whole interval below 0.75, which ten units cannot give. The
    control claim of 0.5 agrees."""
    v = verdicts(tmp_path, ones_and_zeros(5, 10), rel("quarter", rng=[0.75, 1.0]) + rel("half", claimed=0.5))
    for r in v.values():
        assert r["estimate"] == 0.5 and r["p_value"] == 1.0 and r["n_success"] == 5 and r["n_units"] == 10
        assert r["ci_lo"] == 0.2 and r["ci_hi"] == 0.8
    assert v["quarter"]["verdict"] == UNVERIFIABLE and "straddles" in v["quarter"]["why"]
    assert v["half"]["verdict"] == AGREES


# ── 14 of 15, the lab's own example ───────────────────────────────────────────

def test_fourteen_of_fifteen_agrees_with_most(tmp_path):
    """p = 0.00098 exactly (32/32768). With 15 units the bootstrap interval is [0.8, 1.0], which lies inside [0.75, 1.0]:
    agrees. A claim that the correction helps on at most half the targets is refuted."""
    v = verdicts(tmp_path, ones_and_zeros(14, 15), rel("quarter", rng=[0.75, 1.0]) + rel("at_most_half", rng=[0.0, 0.5]))
    r = v["quarter"]
    assert r["estimate"] == round(14 / 15, 6) and r["p_value"] == 32 / 32768 and r["n_success"] == 14 and r["n_units"] == 15
    assert r["ci_lo"] == 0.8 and r["ci_hi"] == 1.0
    assert r["verdict"] == AGREES
    assert v["at_most_half"]["verdict"] == REFUTED


def test_the_other_direction_is_the_same_test(tmp_path):
    """1 of 15 (the correction raises the error on 14 of 15): the same p, the mirrored estimate and interval."""
    v = verdicts(tmp_path, ones_and_zeros(1, 15), rel("few", rng=[0.0, 0.25]))
    r = v["few"]
    assert r["estimate"] == round(1 / 15, 6) and r["p_value"] == 32 / 32768 and r["n_success"] == 1
    assert r["ci_lo"] == 0.0 and r["ci_hi"] == 0.2
    assert r["verdict"] == AGREES


# ── signed columns, ties and units ────────────────────────────────────────────

def test_a_signed_column_counts_its_sign_and_drops_zeros_as_ties(tmp_path):
    """Values: 12 positive, 2 negative, 3 exact zeros. The zeros are ties, dropped and counted; the test runs on 14 units."""
    values = [0.4] * 12 + [-1.5] * 2 + [0.0] * 3
    v = verdicts(tmp_path, values, rel("pos", rng=[0.5, 1.0]))
    r = v["pos"]
    assert r["event"] == "positive" and r["event_source"] == "detected"
    assert r["n_ties_dropped"] == 3 and r["n_units"] == 14 and r["n_success"] == 12 and r["n_rows"] == 17
    assert r["estimate"] == round(12 / 14, 6) and r["p_value"] == sign_test_p(12, 14)
    assert r["verdict"] == AGREES


def test_several_rows_per_unit_resolve_to_the_majority_sign(tmp_path):
    """Three rows per unit, 12 units: ten units with signs (+, +, -) are the event, two with (-, -, +) are not. The estimate is
    10/12, not the row fraction 24/36, and n_units is 12. A thirteenth unit with rows (+, -) splits evenly and is dropped
    as a unit tie, counted apart from the row ties."""
    values, units = [], []
    for u in range(12):
        signs = (1.0, 1.0, -1.0) if u < 10 else (-1.0, -1.0, 1.0)
        values += list(signs)
        units += [f"t{u}"] * 3
    values += [2.0, -2.0]
    units += ["t12", "t12"]
    v = verdicts(tmp_path, values, rel("maj", rng=[0.5, 1.0]), units)
    r = v["maj"]
    assert r["n_rows"] == 38 and r["n_units"] == 12 and r["n_success"] == 10
    assert r["n_unit_ties_dropped"] == 1 and r["n_ties_dropped"] == 0
    assert r["estimate"] == round(10 / 12, 6) and r["p_value"] == sign_test_p(10, 12)
    assert "majority sign" in r["unit_value"] and "up to 3 rows per unit" in r["unit_value"]
    assert r["verdict"] == AGREES


def test_all_ties_is_unverifiable_with_zero_units(tmp_path):
    """Every unit splits evenly, so nothing is left to count: UNVERIFIABLE with n_units 0, the counts on the row, and no p.
    This is relate's answer to a degenerate input elsewhere too (r undefined, fewer than 3 units): a verdict row that says
    why, never an exception that would stop the other claims."""
    values = [1.0, -1.0] * 6
    units = [f"t{i // 2}" for i in range(12)]
    v = verdicts(tmp_path, values, rel("tied", rng=[0.5, 1.0]), units)
    r = v["tied"]
    assert r["verdict"] == UNVERIFIABLE and "n_units is 0" in r["why"]
    assert r["n_units"] == 0 and r["n_unit_ties_dropped"] == 6 and r["n_success"] == 0
    assert r.get("p_value") is None and r.get("estimate") is None


def test_all_zeros_declared_signed_is_all_ties(tmp_path):
    """A column of nothing but zeros reads as boolean by detection (no events, not ties). Declaring `event = "positive"`
    makes them ties, and then n_units is 0 and n_ties_dropped is the row count."""
    v = verdicts(tmp_path, [0] * 8, rel("as_bool", rng=[0.0, 0.5]) + rel("as_sign", rng=[0.0, 0.5], extra='event = "positive"\n'))
    assert v["as_bool"]["event"] == "true" and v["as_bool"]["n_units"] == 8 and v["as_bool"]["n_success"] == 0
    assert v["as_bool"]["verdict"] == AGREES
    assert v["as_sign"]["event"] == "positive" and v["as_sign"]["event_source"] == "declared"
    assert v["as_sign"]["n_units"] == 0 and v["as_sign"]["n_ties_dropped"] == 8 and v["as_sign"]["verdict"] == UNVERIFIABLE


def test_boolean_words_are_read_and_missing_values_are_dropped(tmp_path):
    v = verdicts(tmp_path, ["True"] * 9 + ["false"] + ["", "NA"], rel("words", rng=[0.5, 1.0]))
    r = v["words"]
    assert r["event"] == "true" and r["n_units"] == 10 and r["n_success"] == 9 and r["dropped"] == 2
    assert r["p_value"] == 22 / 1024


def test_a_column_that_is_neither_boolean_nor_numeric_is_refused(tmp_path):
    v = verdicts(tmp_path, [1, 1, "cat", 0, 1], rel("text", rng=[0.5, 1.0]))
    r = v["text"]
    assert r["verdict"] == UNVERIFIABLE
    assert "neither boolean nor numeric" in r["why"] and "'cat'" in r["why"] and "row 3" in r["why"]
    assert r.get("estimate") is None


def test_an_unknown_event_is_refused(tmp_path):
    v = verdicts(tmp_path, ones_and_zeros(9, 10), rel("bad", rng=[0.5, 1.0], extra='event = "negative"\n'))
    assert v["bad"]["verdict"] == UNVERIFIABLE and "event = 'negative'" in v["bad"]["why"]


def test_two_units_have_no_interval(tmp_path):
    v = verdicts(tmp_path, [1, 1], rel("two", rng=[0.5, 1.0]))
    assert v["two"]["verdict"] == UNVERIFIABLE and "fewer than 3 independent units" in v["two"]["why"]
    assert v["two"]["n_units"] == 2


def test_without_a_resampling_unit_each_row_is_a_unit(tmp_path):
    v = verdicts(tmp_path, ones_and_zeros(9, 10), rel("rows", rng=[0.5, 1.0], unit=""))
    r = v["rows"]
    assert r["resampling_unit"] is None and r["n_units"] == 10 and r["p_value"] == 22 / 1024
    assert "over 10 row(s)" in r["p_method"]


def test_kind_must_be_value(tmp_path):
    rels = ('[[relationships]]\nid = "mismatch"\nx = "d.v"\ny = "d.v"\nkind = "correlation"\nmethod = "sign_test"\n'
            'claimed = 0.9\nresampling_unit = "unit"\n')
    v = verdicts(tmp_path, ones_and_zeros(9, 10), rels)
    assert v["mismatch"]["verdict"] == UNVERIFIABLE and "computes a value" in v["mismatch"]["why"]


# ── a zero-variance fraction must not read as unverifiable ────────────────────

def test_a_fraction_of_all_true_agrees_with_a_range_of_exactly_one(tmp_path):
    """`fraction` on an all-true column: every bootstrap draw is 1.0, the interval is [1.0, 1.0], and `claimed_range =
    [1.0, 1.0]` must agree. A zero-variance bootstrap is a tight interval, not an absent one."""
    rels = ('[[relationships]]\nid = "all"\nx = "d.v"\ny = "d.v"\nkind = "value"\nmethod = "fraction"\n'
            'resampling_unit = "unit"\nclaimed_range = [1.0, 1.0]\n\n'
            '[[relationships]]\nid = "all_sign"\nx = "d.v"\ny = "d.v"\nkind = "value"\nmethod = "sign_test"\n'
            'resampling_unit = "unit"\nclaimed_range = [1.0, 1.0]\n')
    v = verdicts(tmp_path, [1] * 12, rels)
    assert v["all"]["ci_lo"] == 1.0 and v["all"]["ci_hi"] == 1.0 and v["all"]["verdict"] == AGREES
    assert v["all_sign"]["verdict"] == AGREES and v["all_sign"]["p_value"] == 2 / 4096
