"""The shapes `relate` is right about and a reader still has to be warned of.

Every check here has a two-sided fixture: a known-bad row it must fire on and a
known-good row it must stay silent on. That is this lab's rule for a new
estimator and it applies to a check just as much, because "the warning list is
empty" is itself a claim.

The bad rows are the real ones, reduced to the fields that matter, from
2026-11-17: `I1_every_shortlist_pose_touches_the_pocket` (1.0 over 40 units with
80 dropped), `A1_recomputed_sha256_matches_inventory` (65 of 65 with 4 dropped),
`W2_a_cut_above_the_maximum_keeps_nothing_DEGENERATE` (refuted at 0.0 against a
claimed 1.0) and the tyk2 line of `flexdock_cadence/LOG.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

from claimcheck import relate
from claimcheck.scrutiny import (
    CANNOT_FAIL,
    PROSE_CONFLICT,
    REFUTED_CONTROL,
    audit,
    is_control,
    prose_conflicts,
    scrutinise,
)


def row(**kw) -> dict:
    """A verdict row with the fields every check reads, overridden by `kw`."""
    base = {"id": "X1_a_claim", "x": "d.c", "y": "d.c", "kind": "value",
            "method": "fraction", "estimate": 0.5, "ci_lo": 0.4, "ci_hi": 0.6,
            "claimed": None, "claimed_range": None, "verdict": "agrees",
            "why": "", "n_units": 100, "n_rows": 100, "dropped": 0,
            "resampling_unit": "pose", "evidence": {}}
    return base | kw


def codes(rows: list[dict]) -> list[str]:
    return [w["code"] for w in scrutinise(rows)]


# ---------------------------------------------------------------------------
# 1. A claim that could not have failed
# ---------------------------------------------------------------------------

def test_the_real_shortlist_claim_fires_and_names_what_the_drops_could_have_done():
    """I1: 1.0 over 40 units with 80 dropped, against a claimed range of [0.9, 1.0]."""
    bad = row(id="I1_every_shortlist_pose_touches_the_pocket", estimate=1.0,
              ci_lo=1.0, ci_hi=1.0, claimed_range=[0.9, 1.0], n_units=40,
              n_rows=40, dropped=80, n_success=40)
    found = scrutinise([bad])
    assert [w["code"] for w in found] == [CANNOT_FAIL]
    w = found[0]
    assert w["dropped"] == 80 and w["rows_in"] == 120
    assert abs(w["dropped_share"] - 2 / 3) < 1e-6 and w["material"] is True
    # The sharper form: one column, so every drop is a drop of the measured property.
    assert w["dropped_for_the_measured_property"] is True
    # 40 passes over the 120 that went in is 0.333, outside [0.9, 1.0].
    assert abs(w["worst_case_share"] - 1 / 3) < 1e-6
    assert w["drops_alone_decided_it"] is True
    assert "80 of the 120" in w["why"] and "0.3333" in w["why"]


def test_the_same_claim_with_nothing_dropped_is_silent():
    """THE OTHER SIDE. An estimate pinned at a bound is not by itself a defect:
    a perfect known answer is supposed to sit there, and this check must not
    fire on one."""
    good = row(id="I1_every_shortlist_pose_touches_the_pocket", estimate=1.0,
               ci_lo=1.0, ci_hi=1.0, claimed_range=[0.9, 1.0], n_units=120,
               n_rows=120, dropped=0, n_success=120)
    assert codes([good]) == []


def test_drops_under_an_estimate_that_is_not_at_a_bound_are_silent():
    """A claim with drops whose estimate sits in the middle could have come out
    either way, so the drops did not decide it."""
    good = row(estimate=0.62, ci_lo=0.55, ci_hi=0.70, claimed_range=[0.5, 0.8],
               n_units=40, dropped=80)
    assert codes([good]) == []


def test_an_integrity_claim_fires_on_four_missing_artifacts_and_says_the_verdict_would_have_held():
    """A1: 65 of 65 pass with 4 dropped as missing. The drops could NOT have
    changed the verdict, and four artifacts are still absent from a claim about
    whether the artifacts match their digests. Both facts are in the text."""
    bad = row(id="A1_recomputed_sha256_matches_inventory", method="binomial_count",
              estimate=1.0, ci_lo=0.945, ci_hi=1.0, claimed_range=[0.5, 1.0],
              n_units=65, n_rows=65, dropped=4, n_success=65,
              resampling_unit="artifact")
    found = scrutinise([bad])
    assert [w["code"] for w in found] == [CANNOT_FAIL]
    w = found[0]
    assert w["material"] is True                      # 4 of 69 is 5.8%
    assert abs(w["worst_case_share"] - 65 / 69) < 1e-6
    assert w["drops_alone_decided_it"] is False
    assert "could not have changed the verdict" in w["why"]


def test_a_two_column_claim_says_it_cannot_tell_which_column_the_drops_came_from():
    bad = row(id="K1_two_columns", x="d.a", y="d.b", kind="correlation",
              method="spearman", estimate=1.0, ci_lo=1.0, ci_hi=1.0, claimed=1.0,
              n_units=40, dropped=12)
    w = scrutinise([bad])[0]
    assert w["dropped_for_the_measured_property"] is False
    assert "does not say which of them" in w["why"]


def test_a_refuted_claim_at_a_bound_with_drops_is_not_this_warning():
    """This warning is about a PASS that could not have failed. A refutation at a
    bound refuted itself; the drops did not buy it anything."""
    good = row(estimate=1.0, ci_lo=1.0, ci_hi=1.0, claimed=0.0, verdict="refuted",
               n_units=40, dropped=80)
    assert CANNOT_FAIL not in codes([good])


def test_an_estimate_at_the_low_bound_counts_the_drops_the_other_way():
    """The failing twin of a pass-at-1.0 is a claim that nothing passed, and the
    drops there would have been passes."""
    bad = row(estimate=0.0, ci_lo=0.0, ci_hi=0.09, claimed_range=[0.0, 0.1],
              n_rows=20, n_units=20, dropped=20, n_success=0)
    w = scrutinise([bad])[0]
    assert abs(w["worst_case_share"] - 0.5) < 1e-9 and w["drops_alone_decided_it"] is True


def test_the_share_is_taken_over_rows_because_that_is_what_dropped_counts():
    """ROWS, NOT UNITS. `relate` increments `dropped` once per row it cannot read,
    while `n_units` counts resampling units, so on a clustered claim the two are
    different numbers: 1,000 rows in 40 clusters with 80 rows dropped is 7.4%
    dropped, and `n_units + dropped` would have called it 67%."""
    bad = row(estimate=1.0, ci_lo=0.99, ci_hi=1.0, claimed_range=[0.9, 1.0],
              n_rows=1000, n_units=40, dropped=80)
    w = scrutinise([bad])[0]
    assert w["rows_in"] == 1080 and abs(w["dropped_share"] - 80 / 1080) < 1e-6
    assert w["n_units"] == 40 and "over 40 pose(s)" in w["why"]
    # 1,000 passes over the 1,080 rows that went in is 0.926, outside [0.9, 1.0].
    assert abs(w["worst_case_share"] - 1000 / 1080) < 1e-6


def test_a_point_claim_is_told_that_the_worst_case_cannot_be_judged():
    """THE FALSE REASSURANCE THIS AVOIDS. A point `claimed` has no range for the
    worst case to fall inside, and folding "no range" together with "inside the
    range" made the text say the drops could not have changed the verdict, which
    nothing had established."""
    bad = row(estimate=1.0, ci_lo=0.9, ci_hi=1.0, claimed=1.0, claimed_range=None,
              n_rows=40, n_units=40, dropped=20)
    w = scrutinise([bad])[0]
    assert w["drops_alone_decided_it"] is None
    assert "could not have changed the verdict" not in w["why"]
    assert "an interval nobody has" in w["why"]


def test_a_sign_test_gets_the_warning_and_no_counterfactual():
    """`relate._sign_units` counts its successes and its units over UNITS, while
    `dropped` counts ROWS, because a unit takes the majority sign of its rows. A
    worst case mixing the two would be a number about nothing, so the drops are
    reported and no counterfactual is."""
    bad = row(method="sign_test", estimate=1.0, ci_lo=0.9, ci_hi=1.0,
              claimed_range=[0.75, 1.0], n_rows=120, n_units=15, dropped=30,
              n_success=15)
    w = scrutinise([bad])[0]
    assert w["code"] == CANNOT_FAIL and w["worst_case_share"] is None
    assert w["drops_alone_decided_it"] is None and "would be" not in w["why"]


def test_pool_check_is_never_called_a_claim_that_cannot_fail():
    """It is deterministic, it has no interval, and 1 is what it reports when
    pooling is justified: an estimate of 1 there is the answer, not a bound."""
    good = row(method="pool_check", estimate=1.0, ci_lo=None, ci_hi=None,
               claimed=1.0, n_units=8, dropped=3)
    assert codes([good]) == []


# ---------------------------------------------------------------------------
# 2. A control, degenerate case or known answer that was refuted
# ---------------------------------------------------------------------------

def test_the_real_degenerate_control_fires():
    """W2: refuted at 0.0 against claimed 1.0, where 0.0 is the correct behaviour
    of the control and the claimed value was written the wrong way round."""
    bad = row(id="W2_a_cut_above_the_maximum_keeps_nothing_DEGENERATE",
              estimate=0.0, ci_lo=0.0, ci_hi=0.0, claimed=1.0, verdict="refuted",
              why="claimed 1.0 lies OUTSIDE the 95% CI [0, 0]")
    found = scrutinise([bad])
    assert [w["code"] for w in found] == [REFUTED_CONTROL]
    assert "the wrong way round" in found[0]["why"]


def test_every_label_in_use_is_recognised_and_an_ordinary_claim_is_not():
    for ident in ("A1_6kx4_aposcore_is_deterministic_KNOWN_ANSWER",
                  "P2_displaced_draws_fail_CONTROL",
                  "A3_6kx4_random_label_control",
                  "W2_a_cut_keeps_nothing_DEGENERATE",
                  "V6_6kx4_degenerate_control"):
        assert is_control(row(id=ident)), ident
    for ident in ("D4_tyk2_repacking_does_not_degrade_recovery",
                  "G1_gcmc_does_not_reorder",
                  # MEASURED AGAINST a control, which is not the same as BEING one.
                  # An infix match on `_control_` called this one a control
                  # (cry1_shortlist_rungs, 2026-11-17), so only an ending counts.
                  "T1_most_of_the_set_beats_its_control_on_both_receptors",
                  "T1_aposcore_costs_well_under_a_second_a_pose"):
        assert not is_control(row(id=ident)), ident


def test_an_evidence_note_that_opens_with_the_label_counts():
    bad = row(id="Q7_plain_name", verdict="refuted", claimed=1.0,
              evidence={"source": "KNOWN ANSWER. The same poses scored twice must agree."})
    assert codes([bad]) == [REFUTED_CONTROL]


def test_the_word_control_inside_an_evidence_note_does_not_count():
    """THE FALSE POSITIVE THIS AVOIDS. Half the evidence notes in the corpus
    discuss the controls beside them; matching the word anywhere would warn on
    every refutation in the file."""
    good = row(id="A2_7ej9_aposcore_does_not_reorder_gnina", verdict="refuted",
               claimed=1.0,
               evidence={"source": "A refutation means the score reorders. The random "
                                   "control A3 says whether the design can resolve anything."})
    assert codes([good]) == []


def test_a_control_that_agrees_is_silent():
    good = row(id="P2_displaced_draws_fail_CONTROL", estimate=0.0, ci_lo=0.0,
               ci_hi=0.168, claimed_range=[0.0, 0.2], verdict="agrees")
    assert codes([good]) == []


# ---------------------------------------------------------------------------
# 3. Prose against the adjudicated row
# ---------------------------------------------------------------------------

TYK2 = {"id": "D4_tyk2_repacking_does_not_degrade_recovery", "x": "d.a", "y": "d.b",
        "method": "mean", "estimate": 0.220397, "ci_lo": 0.146105, "ci_hi": 0.282901,
        "verdict": "agrees", "dropped": 0, "n_units": 46}
PTP1B = {"id": "D4_ptp1b_repacking_does_not_degrade_recovery", "x": "d.a", "y": "d.b",
         "method": "mean", "estimate": 0.301994, "ci_lo": 0.200952, "ci_hi": 0.402535,
         "verdict": "agrees", "dropped": 0, "n_units": 154}


def test_the_real_log_line_fires_on_tyk2_and_stays_silent_on_ptp1b(tmp_path):
    """The two halves of one real sentence. Same file, same shape, same
    precision: the tyk2 interval in the prose is nearly twice the adjudicated
    one and the ptp1b interval is the adjudicated one."""
    (tmp_path / "LOG.md").write_text(
        "The paired difference is +0.220 [+0.045, +0.412] at tyk2 and "
        "+0.302 [+0.201, +0.403] at ptp1b.\n")
    found = prose_conflicts(tmp_path, [TYK2, PTP1B])
    assert [w["code"] for w in found] == [PROSE_CONFLICT]
    assert found[0]["claim"] == TYK2["id"] and found[0]["line"] == 1
    assert found[0]["prose"] == "+0.220 [+0.045, +0.412]"
    assert "0.146105" in found[0]["why"]


def test_a_triple_broken_across_a_hard_wrapped_line_is_still_found(tmp_path):
    """THE MISS THAT HID TWO OF THE FIVE REAL CASES. A LOG file is hard-wrapped at
    a fixed width, so the estimate can end one line with its bracket beginning the
    next. A line-by-line scan saw neither half and reported nothing, which reads
    exactly like agreement."""
    (tmp_path / "LOG.md").write_text(
        "The paired difference (repacked minus rigid, per ligand) is +0.220\n"
        "[+0.045, +0.412] at tyk2 and +0.302 [+0.201, +0.403] at ptp1b.\n")
    found = prose_conflicts(tmp_path, [TYK2, PTP1B])
    assert [w["code"] for w in found] == [PROSE_CONFLICT]
    assert found[0]["claim"] == TYK2["id"] and found[0]["line"] == 1


def test_an_integer_triple_is_not_read_as_a_quoted_result(tmp_path):
    """A number with no decimal point matches any row within half a unit of it,
    which on a correlation is most of them. `(2, 3)` in a sentence about array
    shapes is not a quoted interval."""
    (tmp_path / "LOG.md").write_text("the block is 0 [1, 2] in shape, and 1 (2, 3) deep\n")
    assert prose_conflicts(tmp_path, [TYK2, PTP1B]) == []


def test_a_triple_matching_no_adjudicated_estimate_is_silent(tmp_path):
    """THE SCOPE THAT KEEPS THIS READABLE. A LOG quotes numbers from other
    experiments, from the literature and from runs that are not claims. Only a
    triple whose estimate matches a row in THIS directory is checked at all."""
    (tmp_path / "LOG.md").write_text(
        "Published FEP reaches -0.611 [-0.700, -0.500] on this set.\n")
    assert prose_conflicts(tmp_path, [TYK2, PTP1B]) == []


def test_the_declaration_is_prose_too_and_results_are_not(tmp_path):
    """`gnina_flexres` carries its stale intervals in `context.toml`, so the
    declaration is scanned; `results/` was written by a rule and is not."""
    (tmp_path / "context.toml").write_text(
        'evidence.source = "measured at +0.220 [+0.045, +0.412]"\n')
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "notes.md").write_text("+0.220 [+0.045, +0.412]\n")
    found = prose_conflicts(tmp_path, [TYK2])
    assert len(found) == 1 and found[0]["file"].endswith("context.toml")


def test_an_interval_quoted_to_fewer_decimals_than_the_row_still_matches(tmp_path):
    """Rounding is not a conflict. +0.22 [+0.15, +0.28] is the tyk2 row to two
    decimals and must not be reported."""
    (tmp_path / "LOG.md").write_text("+0.22 [+0.15, +0.28]\n")
    assert prose_conflicts(tmp_path, [TYK2]) == []


def test_a_bound_truncated_or_rounded_the_wrong_way_is_not_a_conflict(tmp_path):
    """MEASURED IN THE CORPUS SWEEP, 2026-11-17. A bound written by hand can be a
    whole unit in the last place away from the row: -0.285674 truncated to
    -0.2856, 0.534591 written as 0.54. Reporting those taught a reader to scroll
    past the ones that matter, where the gap is a hundred times larger."""
    rows = [{"id": "H1_halves_agree", "x": "d.a", "y": "d.b", "method": "pearson",
             "estimate": -0.187879, "ci_lo": -0.692308, "ci_hi": 0.534591,
             "verdict": "agrees", "dropped": 0, "n_units": 8}]
    (tmp_path / "LOG.md").write_text("the halves agree at -0.19 [-0.69, 0.54]\n")
    assert prose_conflicts(tmp_path, rows) == []
    # The real conflict is still reported: +0.045 against +0.146105 is not rounding.
    (tmp_path / "LOG.md").write_text("tyk2 +0.220 [+0.045, +0.412]\n")
    assert len(prose_conflicts(tmp_path, [TYK2])) == 1


def test_an_estimate_matching_several_rows_is_left_alone(tmp_path):
    """The premise is that this sentence is quoting that row. `0.00` at two
    decimals matches every near-zero estimate in the file (four rows in one real
    case, two of them 0 [0, 0]), so there is no row it can be said to quote."""
    near_zero = [{"id": f"Z{i}", "x": "d.a", "y": "d.b", "method": "spearman",
                  "estimate": 0.0005 * i, "ci_lo": -0.2, "ci_hi": 0.2,
                  "verdict": "agrees", "dropped": 0, "n_units": 20} for i in range(4)]
    (tmp_path / "LOG.md").write_text("the themes sit at 0.00 [-0.12, 0.19] throughout\n")
    assert prose_conflicts(tmp_path, near_zero) == []
    # Three or fewer and the match is specific enough to report.
    assert len(prose_conflicts(tmp_path, near_zero[:3])) == 1


# ---------------------------------------------------------------------------
# 4. Through `relate`, on a project, and the sweep over an adjudicated directory
# ---------------------------------------------------------------------------

def _project(tmp_path: Path, rows: str, claim: str) -> Path:
    (tmp_path / "d.csv").write_text(rows)
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "t"\n\n[[datasets]]\nid = "d"\norigin = "computed"\n'
        'files = ["d.csv"]\n\n'
        '[[variables]]\ndataset = "d"\nname = "touches"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "a"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "b"\nunit = "1"\n\n' + claim)
    return tmp_path


def test_relate_puts_the_warnings_in_the_verdict_file_without_changing_a_verdict(tmp_path):
    """A `fraction` claim on a column where two thirds of the rows are blank:
    the pass is real, the warning travels with it, and the verdict is untouched."""
    body = "g,touches,a,b\n"
    for i in range(40):                       # 40 poses that touch the pocket
        body += f"u{i},1,{i},{i}\n"
    for i in range(80):                       # 80 with no measurable contact at all
        body += f"v{i},,{i},{i}\n"
    proj = _project(tmp_path, body,
                    '[[relationships]]\nid = "I1_every_pose_touches_the_pocket"\n'
                    'x = "d.touches"\ny = "d.touches"\nkind = "value"\nmethod = "fraction"\n'
                    'claimed_range = [0.9, 1.0]\nresampling_unit = "g"\n')
    d = relate.check(proj)
    assert d["relationships"][0]["verdict"] == "agrees"
    assert [w["code"] for w in d["warnings"]] == [CANNOT_FAIL]
    assert d["warnings"][0]["dropped"] == 80
    assert d["warnings"][0]["drops_alone_decided_it"] is True


def test_a_clean_project_carries_an_empty_warning_list(tmp_path):
    body = "g,touches,a,b\n" + "".join(f"u{i},1,{i},{i}\n" for i in range(40))
    proj = _project(tmp_path, body,
                    '[[relationships]]\nid = "I1_every_pose_touches_the_pocket"\n'
                    'x = "d.touches"\ny = "d.touches"\nkind = "value"\nmethod = "fraction"\n'
                    'claimed_range = [0.9, 1.0]\nresampling_unit = "g"\n')
    assert relate.check(proj)["warnings"] == []


def test_the_sweep_reads_an_adjudicated_directory_and_its_prose(tmp_path):
    """`audit` is the second call site: it reads the verdicts on disk rather than
    re-adjudicating them, so a directory already checked can be swept."""
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "claims.json").write_text(json.dumps({
        "relationships": [TYK2, row(id="P2_displaced_draws_fail_CONTROL",
                                    verdict="refuted", claimed=1.0, estimate=0.0,
                                    ci_lo=0.0, ci_hi=0.0)]}))
    (tmp_path / "LOG.md").write_text("tyk2 is +0.220 [+0.045, +0.412].\n")
    rep = audit(tmp_path)
    assert rep["counts"][REFUTED_CONTROL] == 1 and rep["counts"][PROSE_CONFLICT] == 1
    assert rep["total"] == 2 and rep["claims_read"] == 2


def test_the_shared_check_claims_rule_scrutinises_every_adjudication_and_cannot_fail_on_it():
    """THE WHOLE POINT: it fires without anyone remembering to run it. The prose
    scan needs the files beside the verdicts, so it cannot live in `relate` and
    is a step of the shared rule instead, chained with `&&` after the verdicts
    are written and wrapped so that a failure of the scan prints a warning
    rather than costing a project its verdicts."""
    text = (Path(__file__).resolve().parents[1] / "claimcheck" / "workflow" / "claims.smk").read_text()
    rule = text[text.index("rule check_claims:"):text.index("rule belief:")]
    assert "claimcheck.scrutiny ." in rule
    scan = rule[rule.index("claimcheck.scrutiny ."):]
    assert "||" in scan and "WARNING" in scan
    # Chained after the verdicts, never in a NEW STATEMENT: `relate && (...); scrutiny`
    # would let the rule exit 0 on the status of the scan after `relate` had failed, so
    # the step that reports a problem would be the step that hid one.
    shell = rule[rule.index("shell:"):]
    assert ">&2) && " in shell[:shell.index("claimcheck.scrutiny .")]
    assert ">&2); " not in shell and '>&2) ; ' not in shell


def test_a_new_experiment_citing_another_ones_rows_is_checked_against_those_rows(tmp_path):
    """THE CASE THAT STARTED THIS. A new experiment's declaration quoted four
    intervals from the experiment before it, three of them stale, and it had no
    verdicts of its own to be checked against. `--claims` points the sweep at the
    directory the numbers came from, which is the only place they can be checked;
    nothing here reaches across the tree by itself, because this package must not
    know where a corpus is kept."""
    source = tmp_path / "flexdock_cadence"
    (source / "results").mkdir(parents=True)
    (source / "results" / "claims.json").write_text(json.dumps({"relationships": [TYK2, PTP1B]}))
    citing = tmp_path / "gnina_flexres"
    citing.mkdir()
    (citing / "context.toml").write_text(
        'evidence.source = "the ensemble route HARMED tyk2 at +0.220 [+0.045, +0.412]"\n'
        'evidence.other = "and ptp1b at +0.302 [+0.201, +0.403]"\n')
    rep = audit(citing, source / "results" / "claims.json")
    assert rep["counts"][PROSE_CONFLICT] == 1
    assert rep["warnings"][0]["claim"] == TYK2["id"] and rep["warnings"][0]["line"] == 1
