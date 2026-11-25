"""`unvalidated_distance`: a distance claim with nothing beside it proving the ruler measures distance.

Two-sided throughout. Every test that asserts the warning FIRES has a partner asserting it does NOT, because a scrutiny
code that cannot stay quiet is noise and a code that cannot fire is decoration.

The incidents behind the code, both in this lab. RDKit's `GetBestRMS` superposes before measuring, so a ligand displaced
25 angstrom scored 0.0 and every receptor in a selection passed. And a constrained embed returned conformers 38 angstrom
from their reference while every composition and charge check on them passed; two further placements gave 1.4 and 6.2
angstrom, and the only reason any was caught is that a known displacement was measured alongside.
"""

from __future__ import annotations

from claimcheck.scrutiny import (
    UNVALIDATED_DISTANCE,
    is_distance,
    scrutinise,
    validates_a_distance,
)


def distance_row(**kw) -> dict:
    row = dict(id="P1_the_pose_is_close_to_the_crystal", y_unit="angstrom",
               estimate=0.42, verdict="agrees", why="")
    row.update(kw)
    return row


def validator_row(**kw) -> dict:
    row = dict(id="K1_a_translated_copy_reads_the_translation_KNOWN_ANSWER", y_unit="angstrom",
               estimate=25.0, verdict="agrees",
               evidence_source="a copy translated 25 angstrom must read 25 angstrom in place")
    row.update(kw)
    return row


def codes(rows):
    return [w["code"] for w in scrutinise(rows)]


# --------------------------------------------------------------------- is_distance

def test_a_distance_unit_is_recognised():
    for unit in ("angstrom", "Angstrom", " Å ", "nm", "A", "rmsd"):
        assert is_distance(distance_row(y_unit=unit)), unit


def test_a_non_distance_unit_is_not():
    for unit in ("kcal/mol", "kJ/mol", "fraction", "dimensionless", "compounds", "", None):
        assert not is_distance(distance_row(y_unit=unit)), unit


# --------------------------------------------------------------------- validates_a_distance

def test_a_translation_known_answer_validates():
    assert validates_a_distance(validator_row())


def test_a_validator_that_reads_zero_does_not_validate():
    """THE CENTRAL CASE. A displaced copy reading 0.0 and agreeing with a claimed 0.0 is the DEFECT, not the check for
    it: that is exactly what a superposing RMSD does. It must not count as validation."""
    assert not validates_a_distance(validator_row(estimate=0.0, verdict="agrees"))


def test_a_validator_that_was_refuted_does_not_validate():
    assert not validates_a_distance(validator_row(verdict="refuted"))


def test_a_claim_that_says_nothing_about_displacement_does_not_validate():
    assert not validates_a_distance(validator_row(
        id="K2_the_same_pose_scored_twice", evidence_source="scoring one pose twice gives one number"))


# --------------------------------------------------------------------- the code firing, and staying quiet

def test_it_fires_when_a_distance_has_no_validator():
    assert codes([distance_row()]) == [UNVALIDATED_DISTANCE]


def test_it_stays_quiet_when_a_validator_is_present():
    assert UNVALIDATED_DISTANCE not in codes([distance_row(), validator_row()])


def test_it_stays_quiet_when_nothing_reports_a_distance():
    rows = [distance_row(y_unit="kcal/mol"), distance_row(id="P2", y_unit="fraction", estimate=0.5)]
    assert UNVALIDATED_DISTANCE not in codes(rows)


def test_it_fires_on_every_distance_claim_not_just_the_first():
    rows = [distance_row(id="P1"), distance_row(id="P2"), distance_row(id="P3")]
    fired = [w for w in scrutinise(rows) if w["code"] == UNVALIDATED_DISTANCE]
    assert [w["id"] for w in fired] == ["P1", "P2", "P3"]
    assert all(w["n_distance_claims"] == 3 for w in fired)


def test_a_zero_reading_validator_does_not_silence_the_code():
    """The two halves together: the superposing-ruler row is present, agrees, and mentions a translation, and the code
    must STILL fire, because that row is the bug rather than the guard."""
    rows = [distance_row(), validator_row(estimate=0.0)]
    assert UNVALIDATED_DISTANCE in codes(rows)


def test_the_warning_carries_the_unit_and_the_estimate_it_is_about():
    w = [x for x in scrutinise([distance_row(estimate=0.42, y_unit="angstrom")])
         if x["code"] == UNVALIDATED_DISTANCE][0]
    assert w["unit"] == "angstrom" and w["estimate"] == 0.42 and w["id"] == "P1_the_pose_is_close_to_the_crystal"


def test_a_non_dict_row_does_not_raise():
    """Junk in the row list is skipped and the real rows are still judged. The first version of this test asserted the
    code stays QUIET here, which was wrong: the list holds a genuine distance claim with no validator, so firing is the
    correct behaviour. The test was the thing at fault, not the code, and it is kept as the case it meant to be."""
    fired = codes([distance_row(), None, "not a row", 7])
    assert fired == [UNVALIDATED_DISTANCE]


def test_junk_alone_warns_about_nothing():
    assert codes([None, "not a row", 7]) == []


def test_every_warning_carries_the_two_keys_relate_prints():
    """`relate.main` prints `w['code']`, `w['claim']` and `w['why']` for every warning it writes, so a code that omits
    either of the last two kills the whole adjudication AFTER every verdict has been computed and before anything is
    written. This code emitted `id` and `detail` instead and did exactly that on 2026-11-25, on an experiment whose pKa
    claims report residue-to-ligand distances: `KeyError: 'claim'`, no results/claims.json, and eight correct verdicts
    lost. The contract is asserted here rather than at the printer because the printer is right and the emitter was not.
    """
    for w in scrutinise([distance_row()]):
        assert "claim" in w and "why" in w, f"{w['code']} omits a key relate.main prints"
        assert w["claim"] and w["why"]


def test_the_unvalidated_distance_warning_names_the_claim_in_both_keys():
    """`id` is kept beside `claim` rather than renamed away, so anything already reading it still works."""
    w = [x for x in scrutinise([distance_row()]) if x["code"] == UNVALIDATED_DISTANCE][0]
    assert w["claim"] == w["id"] == "P1_the_pose_is_close_to_the_crystal"
    assert w["why"] == w["detail"]
