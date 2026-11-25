"""`fixed_by_construction`: an estimate that no resample could move, at a value the estimator could have left.

Two-sided throughout. Every test that asserts the warning FIRES has a partner asserting it does NOT, because a
scrutiny code that cannot stay quiet is noise and a code that cannot fire is decoration.

The incident, 2026-11-25. A lab's relative free-energy method rested on one known answer: transform a ligand into an
identical copy and the result must be 0. It agreed at exactly 0.0 over 3 repeats with an interval of exactly [0, 0].
That agreement was worth nothing, because when both end states are the same molecule the Hamiltonian does not change
with lambda, every reduced potential is identical, and MBAR returns exactly zero whatever samples were drawn. The
claim could not have failed, and its own registered text claimed it would catch four distinct defects.
"""

from __future__ import annotations

from claimcheck.scrutiny import FIXED_BY_CONSTRUCTION, scrutinise


def row(**kw) -> dict:
    r = dict(id="K1_self_edge_returns_zero", method="mean", verdict="agrees",
             estimate=0.0, ci_lo=0.0, ci_hi=0.0, n_units=3, resampling_unit="repeat", why="")
    r.update(kw)
    return r


def codes(rows: list) -> list[str]:
    return [w["code"] for w in scrutinise(rows) if w["code"] == FIXED_BY_CONSTRUCTION]


def test_the_self_edge_shape_fires():
    """The incident itself: a mean of exactly 0.0 with a zero-width interval over 3 repeats."""
    assert codes([row()]) == [FIXED_BY_CONSTRUCTION]


def test_a_real_interval_is_quiet():
    """The partner. The same claim with a genuine spread across repeats says nothing."""
    assert codes([row(estimate=0.0005, ci_lo=-0.2652, ci_hi=0.3998)]) == []


def test_one_unit_is_quiet_because_one_unit_cannot_produce_a_width():
    """`relate` already reports a single unit as unverifiable with its own reason, so this code must not
    double-report it as something more sinister."""
    assert codes([row(n_units=1)]) == []
    assert codes([row(n_units=None)]) == []


def test_an_estimate_at_an_estimator_bound_is_left_to_cannot_fail():
    """A fraction of 1.0 with a zero-width interval is the `cannot_fail` diagnosis, not this one, and a row must not
    carry both: the repairs differ, and two warnings on one row trains a reader to skim."""
    at_bound = row(id="I1_every_pose_touches_the_pocket", method="fraction",
                   estimate=1.0, ci_lo=1.0, ci_hi=1.0, n_units=40)
    assert codes([at_bound]) == []


def test_it_fires_on_a_refuted_row_too():
    """Unlike `cannot_fail`, this shape is not about passing. A refuted claim whose estimate no resample could move
    is just as uninformative, and `full_restraint_control` has several."""
    assert codes([row(verdict="refuted", estimate=0.0, ci_lo=0.0, ci_hi=0.0)]) == [FIXED_BY_CONSTRUCTION]


def test_the_paired_self_comparison_known_answer_is_quiet():
    """The one designed zero width. A paired-difference claim comparing a column against a copy of itself MUST return
    exactly 0.0 with an interval of exactly [0, 0]: an unpaired resample would give the same 0.0 with a wide
    interval, so the zero width is the entire evidence that the bootstrap draws each unit with both of its columns.
    Warning there would fire on every correct use of the documented pattern, and `test_bedroc_difference.py` already
    asserts that no warning reaches such a row. A declared `baseline` is what marks the family."""
    paired = row(id="P40_6kx4_a_column_against_a_copy_of_itself_KNOWN_ANSWER",
                 method="roc_auc_difference", baseline="panel.fraction_reference_shared",
                 estimate=0.0, ci_lo=0.0, ci_hi=0.0, n_units=1597)
    assert codes([paired]) == []


def test_a_paired_difference_without_a_baseline_key_still_fires():
    """The exclusion is driven by the declared `baseline`, not by the method name, so a row that somehow reaches
    scrutiny without one is still judged on its shape rather than waved through on its method."""
    assert codes([row(method="roc_auc_difference", n_units=1597)]) == [FIXED_BY_CONSTRUCTION]


def test_a_missing_or_null_interval_is_quiet():
    assert codes([row(ci_lo=None, ci_hi=None)]) == []
    assert codes([row(estimate=None)]) == []
    assert codes([{"id": "no_interval_at_all", "method": "mean", "verdict": "unverifiable"}]) == []


def test_the_warning_carries_the_two_keys_relate_prints():
    """`relate.main` prints w['code'], w['claim'] and w['why'] for every warning. A code omitting either of the last
    two kills the whole adjudication after every verdict is computed, which happened to `unvalidated_distance`."""
    for w in scrutinise([row()]):
        assert "claim" in w and "why" in w and w["claim"] and w["why"]


def test_the_warning_names_the_unit_and_the_count():
    w = [x for x in scrutinise([row()]) if x["code"] == FIXED_BY_CONSTRUCTION][0]
    assert w["n_units"] == 3 and w["resampling_unit"] == "repeat"
    assert "repeat(s)" in w["why"] and "zero width" in w["why"]


def test_a_non_dict_row_does_not_raise():
    assert codes([row(), None, "not a row", 7]) == [FIXED_BY_CONSTRUCTION]


def test_junk_alone_warns_about_nothing():
    assert codes([None, "not a row", 7]) == []
