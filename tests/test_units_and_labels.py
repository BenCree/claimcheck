"""The two modules ported from `provchem` on 2026-08-17, both sides of each.

`claimcheck` could already say a column was a float in kcal/mol. It could not
say that kcal/mol and kJ/mol are the same quantity a factor apart, nor what made
a row a hit. Both gaps are recorded in `archive/PORTING_LOG.md` in the al_mace
repository, items 1 and 2.
"""

from __future__ import annotations

import pytest

from claimcheck.labels import LabelRule, UnusableLabelRule, rate
from claimcheck.units import UNITS, UnknownUnit, compatible, convert


# --------------------------------------------------------------------- units

def test_a_checked_conversion_converts():
    assert convert(1.0, "kcal/mol", "kJ/mol") == pytest.approx(4.184)
    assert convert(4.184, "kJ/mol", "kcal/mol") == pytest.approx(1.0)
    assert convert(1.0, "nm", "0.1 nm") == pytest.approx(10.0)
    assert convert(2.5, "ns", "ps") == pytest.approx(2500.0)


def test_the_same_unit_is_the_identity():
    assert convert(7.0, "kcal/mol", "kcal/mol") == 7.0


def test_an_unknown_unit_raises_rather_than_passing_the_value_through():
    """The whole point. A conversion that silently returns its input turns a
    units bug into a plausible number, which is the characteristic failure of
    computational chemistry -- not a crash."""
    with pytest.raises(UnknownUnit):
        convert(1.0, "furlong", "nm")
    with pytest.raises(UnknownUnit):
        convert(1.0, "nm", "furlong")


def test_a_change_of_quantity_is_refused_by_name():
    """eV to kcal/mol needs Avogadro's number: per particle to per mole. It is
    absent on purpose and the refusal has to say why."""
    with pytest.raises(UnknownUnit) as e:
        convert(1.0, "eV", "kcal/mol")
    assert "change of quantity" in str(e.value)


def test_a_declared_pair_with_no_checked_factor_is_refused():
    """Both are lengths and both are declared, but nobody has written the
    factor -- so it raises rather than inventing one."""
    UNITS["fm"] = UNITS["nm"].__class__("fm", "length")
    try:
        with pytest.raises(UnknownUnit) as e:
            convert(1.0, "nm", "fm")
        assert "no checked factor" in str(e.value)
    finally:
        del UNITS["fm"]


def test_compatible_distinguishes_the_three_cases():
    ok, why = compatible("kcal/mol", "kJ/mol")
    assert ok and "factor" in why
    ok, why = compatible("kcal/mol", "0.1 nm")
    # NOT refused: correlating an energy against a length is ordinary. What
    # matters is that the reader is told they are different quantities.
    assert ok and "different quantities" in why
    ok, why = compatible(None, "kcal/mol")
    assert not ok and "no unit declared" in why


# -------------------------------------------------------------------- labels

def test_a_rule_missing_its_assay_is_refused():
    """`assay` is the field people leave out, and it is the one that decides
    whether two rates may be compared."""
    with pytest.raises(UnusableLabelRule) as e:
        LabelRule(name="hit", assay="", endpoint="pKD", statement="pKD >= 6")
    assert "assay" in str(e.value)


def test_a_complete_rule_describes_itself():
    r = LabelRule(name="hit", assay="biophysical_binding", endpoint="pKD",
                  statement="pKD >= 6.0 by SPR at 25 C")
    assert r.describe() == "hit (biophysical_binding, pKD)"


def test_rules_from_different_assays_are_not_comparable():
    """The recorded incident: a pK_D threshold published beside an HTS
    dose-response call, in one column."""
    spr = LabelRule(name="a", assay="biophysical_binding", endpoint="pKD",
                    statement="pKD >= 6")
    hts = LabelRule(name="b", assay="hts_dose_response", endpoint="pKD",
                    statement="pKD >= 6")
    ok, why = spr.comparable_to(hts)
    assert not ok and "different assays" in why


def test_rules_thresholding_different_endpoints_are_not_comparable():
    a = LabelRule(name="a", assay="cell_viability", endpoint="pIC50",
                  statement="pIC50 >= 6")
    b = LabelRule(name="b", assay="cell_viability", endpoint="pKD",
                  statement="pKD >= 6")
    ok, why = a.comparable_to(b)
    assert not ok and "thresholds" in why


def test_matching_rules_are_comparable():
    a = LabelRule(name="a", assay="biophysical_binding", endpoint="pKD",
                  statement="pKD >= 6")
    b = LabelRule(name="b", assay="biophysical_binding", endpoint="pKD",
                  statement="pKD >= 6.5")
    ok, why = a.comparable_to(b)
    assert ok and "biophysical_binding" in why


def test_a_rate_cannot_be_produced_without_its_rule():
    """There is no plain-float form of `rate`, deliberately: a float can be
    pulled out of context by the next line of code."""
    r = LabelRule(name="hit", assay="biophysical_binding", endpoint="pKD",
                  statement="pKD >= 6.0")
    out = rate(113, 621, r)
    assert out["rate"] == pytest.approx(113 / 621)
    assert out["label_rule"] == "hit (biophysical_binding, pKD)"
    assert out["label_assay"] == "biophysical_binding"
    assert out["label_statement"]


def test_a_rate_over_nothing_is_refused():
    r = LabelRule(name="hit", assay="a", endpoint="b", statement="c")
    with pytest.raises(ValueError):
        rate(0, 0, r)
    with pytest.raises(ValueError):
        rate(5, 3, r)
