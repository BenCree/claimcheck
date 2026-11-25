"""A verdict at the edge of its interval, and a refutation that says how far outside.

On 2026-11-17 six known-answer claims were read as a defect in this checker:
all six say "claimed 1.0 lies OUTSIDE the 95% CI [1.000, 1.000]" with the
estimate, both bounds and `claimed` all printing as 1.0, and identical claims on
a smaller cohort agree. The reading was that the comparison fails at large n.

IT DOES NOT, and the first two tests here are that property pinned down: a
claimed value exactly equal to a zero-width interval at that value agrees, at
+1, at -1, at an ordinary value, and for a `claimed_range` whose bound it is.
The comparison runs on unrounded numbers (`lo <= claimed <= hi`) and has always
been right about them.

What was wrong is the LINE, which rounds to three decimals and so reported a
refutation of 4e-7 as a refutation of nothing. The last tests are the fix: a
real near-identity refutation, printing as [1.000, 1.000], now carries its exact
margin and the known-answer tolerance beside it.
"""

from __future__ import annotations

from pathlib import Path

from claimcheck import relate


def _project(tmp_path: Path, rows: str, claim: str) -> Path:
    (tmp_path / "d.csv").write_text(rows)
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "t"\n\n[[datasets]]\nid = "d"\norigin = "computed"\n'
        'files = ["d.csv"]\n\n'
        '[[variables]]\ndataset = "d"\nname = "a"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "b"\nunit = "1"\n\n' + claim)
    return tmp_path


def _one(d: dict) -> dict:
    assert len(d["relationships"]) == 1
    return d["relationships"][0]


# ---------------------------------------------------------------------------
# The property: a zero-width interval exactly at the claimed value agrees
# ---------------------------------------------------------------------------

def test_a_known_answer_of_one_on_identical_columns_agrees(tmp_path):
    body = "g,a,b\n" + "".join(f"u{i},{i},{i}\n" for i in range(60))
    proj = _project(tmp_path, body,
                    '[[relationships]]\nid = "KA_identity"\nx = "d.a"\ny = "d.b"\n'
                    'kind = "correlation"\nmethod = "spearman"\nclaimed = 1.0\n'
                    'resampling_unit = "g"\n')
    r = _one(relate.check(proj))
    assert (r["estimate"], r["ci_lo"], r["ci_hi"]) == (1.0, 1.0, 1.0)
    assert r["verdict"] == "agrees"


def test_a_known_answer_of_minus_one_on_a_reversed_column_agrees(tmp_path):
    """THE OTHER BOUND, which the 2026-11-17 reading asked about: the same
    comparison at -1, where the interval is zero-width at -1."""
    body = "g,a,b\n" + "".join(f"u{i},{i},{-i}\n" for i in range(60))
    proj = _project(tmp_path, body,
                    '[[relationships]]\nid = "KA_reversed"\nx = "d.a"\ny = "d.b"\n'
                    'kind = "correlation"\nmethod = "spearman"\nclaimed = -1.0\n'
                    'resampling_unit = "g"\n')
    r = _one(relate.check(proj))
    assert (r["estimate"], r["ci_lo"], r["ci_hi"]) == (-1.0, -1.0, -1.0)
    assert r["verdict"] == "agrees"


def test_a_value_claim_on_a_constant_column_agrees_at_its_zero_width_interval(tmp_path):
    """Not only at +-1. A `mean` of a constant column has a zero-width interval
    at that constant, and a claim of exactly that value agrees."""
    body = "g,a,b\n" + "".join(f"u{i},7.5,7.5\n" for i in range(20))
    proj = _project(tmp_path, body,
                    '[[relationships]]\nid = "V_constant"\nx = "d.a"\ny = "d.a"\n'
                    'kind = "value"\nmethod = "mean"\nclaimed = 7.5\n'
                    'resampling_unit = "g"\n')
    r = _one(relate.check(proj))
    assert (r["estimate"], r["ci_lo"], r["ci_hi"]) == (7.5, 7.5, 7.5)
    assert r["verdict"] == "agrees"


def test_a_claimed_range_whose_bound_the_zero_width_interval_sits_on_agrees(tmp_path):
    """`claimed_range` is judged by where the whole interval lies, and a closed
    bound includes itself: [1.0, 1.0] inside [0.9, 1.0] agrees."""
    body = "g,a,b\n" + "".join(f"u{i},{i},{i}\n" for i in range(60))
    for claim_range, expect in (("[0.9, 1.0]", "agrees"), ("[-1.0, 1.0]", "agrees"),
                                ("[0.9, 0.99]", "refuted")):
        proj = _project(tmp_path, body,
                        f'[[relationships]]\nid = "R_bound"\nx = "d.a"\ny = "d.b"\n'
                        f'kind = "correlation"\nmethod = "spearman"\n'
                        f'claimed_range = {claim_range}\nresampling_unit = "g"\n')
        assert _one(relate.check(proj))["verdict"] == expect, claim_range


# ---------------------------------------------------------------------------
# The real defect: a refutation whose margin the printed line swallowed
# ---------------------------------------------------------------------------

def _near_identity(tmp_path: Path, n: int = 200, swaps: int = 12) -> Path:
    """Two columns that are the same measurement twice, a few adjacent pairs out of order.

    This is the real shape of `A1_6kx4_aposcore_is_deterministic_KNOWN_ANSWER`,
    reduced: the two ApoScore passes differ in the seventh decimal
    (20.82521521591532 against 20.825217903571236), so over 34,447 compounds
    adjacent pairs swap rank. 12 swaps in 200 rows put rho at 1 - 1.8e-5, below
    1 by more than KNOWN_ANSWER_TOL and by less than the three decimals the line
    prints, and leave no bootstrap replicate in 2000 free of a swapped pair, so
    the whole interval is below 1.0 as it is on the real cohort.
    """
    xs = list(range(n))
    ys = list(range(n))
    for s in range(swaps):
        i = 10 + 10 * s
        ys[i], ys[i + 1] = ys[i + 1], ys[i]
    body = "g,a,b\n" + "".join(f"u{i},{x},{y}\n" for i, (x, y) in enumerate(zip(xs, ys, strict=True)))
    return _project(tmp_path, body,
                    '[[relationships]]\nid = "A1_is_deterministic_KNOWN_ANSWER"\n'
                    'x = "d.a"\ny = "d.b"\nkind = "correlation"\nmethod = "spearman"\n'
                    'claimed = 1.0\nresampling_unit = "g"\n')


def test_a_real_refutation_that_prints_as_one_point_zero_now_states_its_margin(tmp_path):
    r = _one(relate.check(_near_identity(tmp_path)))
    # THE SENTENCE THAT WAS READ AS A CONTRADICTION, reproduced exactly: every
    # number in it prints as 1.000 and the verdict is a refutation. (On the real
    # 34,447 compounds the row's own 6-place fields round to 1.0 as well; 200
    # rows here keep the test to a second, and it is the printed line that did
    # the damage.)
    assert r["verdict"] == "refuted"
    assert "claimed 1.0 lies OUTSIDE the 95% CI [1.000, 1.000] for r = 1.000" in r["why"]
    assert r["estimate"] < 1.0 and r["ci_lo"] < 1.0
    # And it no longer reads as one: the margin is spelled out, with the
    # known-answer tolerance it is being judged against.
    assert "printed numbers are rounded and this refutation is real" in r["why"]
    # The known-answer allowance is compared against the ESTIMATE, which is the distance the
    # rescue above actually tests, never against the gap to the interval bound: comparing the
    # gap could print "inside the allowance" on a row the rescue had declined.
    assert "|1 - r| = 1.8e-05" in r["why"] and "times that allowance" in r["why"]
    assert "1e-12" in r["why"]


def test_a_refutation_the_printed_line_already_shows_is_left_alone(tmp_path):
    """THE OTHER SIDE. The margin sentence appears only where the rounding hid
    the gap. An ordinary refutation reads exactly as it always has."""
    body = "g,a,b\n" + "".join(f"u{i},{i},{(i * 7) % 61}\n" for i in range(60))
    proj = _project(tmp_path, body,
                    '[[relationships]]\nid = "A2_does_not_reorder"\nx = "d.a"\ny = "d.b"\n'
                    'kind = "correlation"\nmethod = "spearman"\nclaimed = 1.0\n'
                    'resampling_unit = "g"\n')
    r = _one(relate.check(proj))
    assert r["verdict"] == "refuted"
    assert "lies OUTSIDE" in r["why"]
    assert "printed numbers are rounded" not in r["why"]


def test_tie_tol_does_not_tie_a_pair_that_straddles_its_grid_boundary(tmp_path):
    """THE LIMIT OF `tie_tol`, pinned as a test because it was found twice by hand.

    `_snap` rounds each value to the nearest multiple of the tolerance before ranking, so two values
    closer together than the tolerance are tied only when they fall in the SAME cell. A pair that
    straddles a boundary is ranked apart however large the tolerance is made, and enlarging it
    shrinks that share without ever reaching zero. That is why
    `KA1b_cry1_aposcore_reproduces_itself_with_tie_tolerance` is refuted at `tie_tol` 1e-4 on
    differences near 2e-6, and the experiment's own evidence note says so:
    `/home/ben/code/projects/targets/cry1/experiments/funnel_rerank_and_top1k/context.toml`,
    "a PAIR THAT STRADDLES A GRID BOUNDARY is still ranked apart ... roughly two per cent of 34,386
    rows straddle a boundary whatever the tolerance". The note's conclusion is that the claim design
    is at fault rather than the data or the checker, and the registered repair is a `claimed_range`.
    """
    tol = 1e-4
    # Both sides of one boundary, 1e-6 apart: not tied, which is the whole finding.
    straddling = relate._snap([1.000049999, 1.000050999], tol)
    assert straddling[0] != straddling[1]
    # The same 1e-6 apart inside one cell: tied, so the tolerance does what it promises there.
    inside = relate._snap([1.000020000, 1.000021000], tol)
    assert inside[0] == inside[1]


def test_the_claimed_range_that_replaces_a_known_answer_of_exactly_one_agrees(tmp_path):
    """The constructive half. A rank correlation claimed at exactly 1.0 on a stochastic estimate is
    refuted by any noise at all; the same data against the `claimed_range` [0.9999, 1.0] that the
    funnel experiment registered in its `context.pending.toml` agrees."""
    ys = list(range(200))
    for s in range(12):                                   # the same 12 swaps as _near_identity
        i = 10 + 10 * s
        ys[i], ys[i + 1] = ys[i + 1], ys[i]
    body = "g,a,b\n" + "".join(f"u{i},{i},{y}\n" for i, y in enumerate(ys))
    proj = _project(tmp_path, body,
                    '[[relationships]]\nid = "KA1c_range"\nx = "d.a"\ny = "d.b"\n'
                    'kind = "correlation"\nmethod = "spearman"\n'
                    'claimed_range = [0.9999, 1.0]\nresampling_unit = "g"\n')
    r = _one(relate.check(proj))
    assert r["verdict"] == "agrees" and r["ci_hi"] <= 1.0
    # The same data against a claim of exactly 1.0 is refuted, which is the contrast.
    assert _one(relate.check(_near_identity(tmp_path)))["verdict"] == "refuted"


def test_the_tie_tolerance_still_rescues_a_known_answer_that_is_equal_in_exact_arithmetic(tmp_path):
    """`tie_tol` is the key that makes a known answer of 1.0 hold when the two
    columns differ only by rounding noise. It is not what was happening to the
    six claims: their columns differ by 2.7e-6, three orders above the 1e-9 they
    declare, so the snapping cannot tie them and the refutation stands."""
    body = "g,a,b\n" + "".join(f"u{i},{i}.0000000001,{i}.0000000002\n" for i in range(60))
    claim = ('[[relationships]]\nid = "KA_noise"\nx = "d.a"\ny = "d.b"\n'
             'kind = "correlation"\nmethod = "spearman"\nclaimed = 1.0\n'
             'tie_tol = 1.0e-9\nresampling_unit = "g"\n')
    assert _one(relate.check(_project(tmp_path, body, claim)))["verdict"] == "agrees"
