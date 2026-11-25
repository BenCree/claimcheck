"""Shapes that make a verdict untrustworthy, read off the finished rows.

    python -m claimcheck.scrutiny <experiment-dir>
    python -m claimcheck.scrutiny <experiment-dir> --check
    python -m claimcheck.scrutiny <experiment-dir> --json out.json

    from claimcheck.scrutiny import scrutinise, prose_conflicts
    rows = json.loads(Path("results/claims.json").read_text())["relationships"]
    warnings = scrutinise(rows)

`relate` answers one question: does the claimed value lie inside the interval.
It answers it correctly even when the claim could not have come out any other
way, even when the claim was registered with the wrong expectation, and even
when the prose beside it says something else. Each of those was found on
2026-11-17 by a person reading rows one at a time, which is the reading that
does not happen when nobody is looking. This module is that reading, run every
time `relate` does.

## Warnings never decide anything

A warning here never changes a `verdict`, never changes an exit status unless
`--check` is passed, and never appears on a verdict row. It is a separate list,
addressed to a person, naming the claim and the numbers that raised it. The
reason is the one `rulings` gives for its reworded-duplicate warning: a checker
that guesses intent either blocks real work or is switched off, and both end
with nobody reading it. Every warning below is a shape that is *usually* a
defect and is sometimes the point, so each is reported with the numbers that
would tell the two apart.

## The three shapes

**`cannot_fail`**: a claim that passed with its estimate pinned at the end of
what the estimator can produce, with rows dropped on the way. Measured on
`/home/ben/code/projects/targets/cry1/experiments/cry1_shortlist_rungs/results/claims.json`:
`I1_every_shortlist_pose_touches_the_pocket` reads 1.0 [1.0, 1.0] over 40 units
against the claimed range [0.9, 1.0], with **80 dropped**, two thirds of the
rows that went in, and they are the poses with no measurable contact, which is
the property the claim is about. Had all 80 been counted as failures the share
would be 0.333, far outside the claimed range: the drops alone decided the
verdict. Same shape, weaker, in
`/home/ben/code/projects/targets/cry1/experiments/structure_audit/results/claims.json`:
`A1_recomputed_sha256_matches_inventory` passes 65 of 65 with 4 dropped as
missing, an integrity claim reporting a pass while four artifacts are absent.

The test is on `dropped`, not on a mismatch between `n_units` and `n_rows`. The
two differ by construction on every clustered claim, so that form fires on
legitimate rows and does not discriminate (a reviewer checked it against the
master ledger on 2026-11-17 and reported 223 of 24,562 rows flagged, nearly all
of them honest cluster units; that count is theirs and is not re-measured here).

**`refuted_control`**: a claim whose id token or evidence label marks it as a
control, a degenerate case or a known answer, coming back `refuted`. That
combination is nearly always one of two things and a person has to say which:
a registration error, as in `W2_a_cut_above_the_maximum_keeps_nothing_DEGENERATE`,
refuted at 0.0 against `claimed = 1.0` where 0.0 is the correct behaviour of the
control and the claimed value was simply written the wrong way round; or a
genuine failure of the thing every other claim in the file rests on. Nothing
here tries to tell them apart: reading intent from a claim id is exactly the
guess that would make this checker ignorable. It reads the id and the label the
evidence note opens with, and nothing else, so a control whose claim text says
only "THE DATED REPLACEMENT FOR ..." is not recognised and is not meant to be.

**`unvalidated_distance`**: an adjudication that reports a distance and nowhere
demonstrates that its measure recovers a known displacement. A distance is the
one quantity whose instrument can fail by returning ZERO, and zero passes every
upper bound placed on it, so a wrong answer reads as an excellent one. Two
incidents: RDKit's `GetBestRMS` SUPERPOSES before measuring, so a ligand
displaced 25 angstrom scored 0.0 and every receptor in a selection passed, which
would have chosen a receptor that cannot reproduce its own crystal pose; and on
2026-11-23 a constrained embed returned conformers 38 angstrom from their
reference while every composition and charge check on them passed, with two
further placements giving 1.4 and 6.2 angstrom, all three caught only because a
known displacement was measured beside them. The code fires on each distance
claim when no claim in the same adjudication (a) speaks of a displacement, (b)
agreed, and (c) has a NON-ZERO estimate. The third condition is the point: a
claim that a displaced copy reads 0.0, agreeing with a claimed 0.0, is the defect
rather than the guard against it, and it does not silence the warning. The repair
costs no simulation: one claim that moves a copy by a declared offset and
recovers it, measured through the same code path.

**`prose_conflict`**: a number in the experiment's own prose that contradicts
the adjudicated row it is quoting. Measured on
`/home/ben/code/projects/targets/cry1/experiments/flexdock_cadence/LOG.md`,
which gives the tyk2 paired difference as +0.220 [+0.045, +0.412] while
`results/claims.json` adjudicated +0.220397 [+0.146105, +0.282901]: the point
estimate matches and the interval is nearly twice as wide. The matcher is
deliberately narrow, an estimate and an interval written as one triple whose
estimate matches an adjudicated row at the precision it is printed to while no
such row's interval does, because a loose number-matcher over a LOG file reports
everything and is read by nobody. The prose of an experiment with no verdicts of
its own is checked by naming the verdicts it quotes (`--claims`): nothing here
searches the tree, because this package must not know where a corpus is kept.

## What is deliberately not here

**An interval narrower than the spread of the values it summarises.** Proposed
on 2026-11-17 after a jackknife over 19 walkers produced an interval 0.58 wide
on values spanning 20 to 23. It is not added, because the ratio has no
defensible threshold: an interval on a mean is *supposed* to be about sqrt(n)
narrower than the raw spread, so the stated form fires on every correct mean of
more than a few units, and `relate` has no jackknife estimator to misuse. The
defect was in the code that produced that interval, which is not this package.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from claimcheck.relate import AGREES, REFUTED

#: The three codes. A closed set, so a reader can be told what the whole
#: vocabulary is and a consumer can switch on it.
CANNOT_FAIL = "cannot_fail"
REFUTED_CONTROL = "refuted_control"
PROSE_CONFLICT = "prose_conflict"
UNVALIDATED_DISTANCE = "unvalidated_distance"
FIXED_BY_CONSTRUCTION = "fixed_by_construction"
CODES: tuple[str, ...] = (CANNOT_FAIL, REFUTED_CONTROL, PROSE_CONFLICT, UNVALIDATED_DISTANCE,
                          FIXED_BY_CONSTRUCTION)

#: Width at or below which a resampled interval is treated as having no width at all. The bootstrap resamples whole
#: units, so if the units genuinely differ at all some resample must differ; a width under this is the units being
#: identical rather than the estimate being precise. It is deliberately tighter than BOUND_TOL: this test is about
#: exact equality, not about nearness to a bound.
ZERO_WIDTH_TOL = 1e-12

#: Units that mean "this claim reports a distance". Lowercased and stripped before the test. A distance is singled out
#: from every other unit because a distance measure can silently return ZERO for a wrong answer, which no other kind of
#: quantity does: a superposing RMSD scores a ligand displaced 25 angstrom as 0.0 and passes every bound beneath it.
DISTANCE_UNITS: frozenset[str] = frozenset({
    "angstrom", "angstroms", "a", "å", "ang",
    "nm", "nanometre", "nanometres", "nanometer", "nanometers",
    "pm", "picometre", "picometres", "picometer", "picometers",
    "rmsd", "rmsd (angstrom)", "distance",
})

#: What a claim that VALIDATES a distance measure looks like: it moves something by a known amount and recovers it. The
#: words are deliberately broad, because the point is to notice the absence of such a claim, not to police its wording.
_DISPLACEMENT_WORDS = re.compile(
    r"\b(translat\w*|displac\w*|shift\w*|moved?|offset|box.?length|known.?distance|in.?place)\b", re.I)

#: How close to a bound counts as sitting on it. Absolute, and far below the
#: gap between two attainable values of any share over a realistic number of
#: units (1 / n), so it recognises an estimate that IS the bound and never one
#: that is merely near it.
BOUND_TOL = 1e-12

#: What each estimator can produce, as (low, high); None where the end is not
#: bounded. An estimate at a bound here is one no further evidence could have
#: pushed further, which is what makes a drop able to decide the verdict.
#: `pool_check` is left out on purpose: it is deterministic, it has no interval,
#: and 1 is simply what it reports when pooling is justified.
METHOD_RANGE: dict[str, tuple[float | None, float | None]] = {
    "pearson": (-1.0, 1.0), "spearman": (-1.0, 1.0), "kendall": (-1.0, 1.0),
    "roc_auc": (0.0, 1.0), "bedroc": (0.0, 1.0),
    # A paired difference of two AUROCs spans [-1, 1], and both ends are reachable: a perfect score against its own
    # mirror is +1. Its known answer sits at 0.0, which is INTERIOR, so a self-difference claim is not flagged for
    # sitting where it must sit (a `claimed_range` with 0.0 as a bound would be, which is why that known answer is
    # registered as a point `claimed`).
    "roc_auc_difference": (-1.0, 1.0),
    # A paired difference of two BEDROCs spans [-1, 1] on the same reasoning: BEDROC is bounded 0 to 1 at any active
    # fraction (it does not saturate, `docs/bedroc.md`) and +1 is reached by a perfect ranking against its own inversion.
    # Its known answer is 0.0, interior, so a self-difference claim is not flagged for sitting where it must sit.
    "bedroc_difference": (-1.0, 1.0),
    "fraction": (0.0, 1.0), "f1": (0.0, 1.0), "sign_test": (0.0, 1.0),
    "binomial_count": (0.0, 1.0), "topn_recovery": (0.0, 1.0),
    "var_share": (0.0, None),
    "sd": (0.0, None), "pooled_sd": (0.0, None), "pooled_se": (0.0, None),
    "mean": (None, None), "median": (None, None),
}

#: The methods whose estimate is a count over THE SAME THING `dropped` counts, so
#: the share that would have been reported had every dropped row come back the
#: other way is arithmetic rather than a guess. Everything else reports the drops
#: and no counterfactual.
#:
#: `sign_test` is deliberately absent although it is a share: `relate._sign_units`
#: counts `n_success` and `n_units` over UNITS and `dropped` over ROWS (one unit
#: takes the majority sign of its rows), so a counterfactual mixing the two would
#: be a number about nothing. `topn_recovery` is absent for the same reason: its
#: denominator is the actives, and the drops are compounds.
SHARE_METHODS = frozenset({"fraction", "binomial_count"})

#: A drop share at or above this is called material in the warning text. OUR
#: CHOICE, NO SOURCE: `bestpractices/sampling_uncertainty.py`
#: (`_bootstrap_does_not_drop_silently`, Grossfield et al. LiveCoMS 1(1):5067
#: §2 item 8) requires that `n_kept` and `n_drop` be recorded and sets no
#: threshold on their ratio, and nothing else under `bestpractices/` does
#: either. It changes no verdict and no exit status: it chooses one word in a
#: sentence a person reads. The warning fires on ANY drop under a pinned
#: estimate, because `A1_recomputed_sha256_matches_inventory` drops 4 of 69
#: (5.8%) and is the case this was built for.
MATERIAL_DROP = 0.05

#: Uppercase tokens in a claim id that mark it as a control, a degenerate case
#: or a known answer, matched as whole `_`-separated tokens. The lowercase
#: `_control` suffix is the other convention in use and is matched too. Matching
#: the word "control" anywhere in the evidence text is NOT done: half the
#: evidence notes in the corpus discuss the controls beside them.
CONTROL_TOKENS: tuple[str, ...] = ("CONTROL", "CONTROLS", "DEGENERATE",
                                   "KNOWN_ANSWER", "KNOWNANSWER", "SANITY")

#: The same labels, at the START of an evidence note, where the convention is to
#: open with the kind of claim this is ("KNOWN ANSWER. The same poses ...").
CONTROL_LABEL = re.compile(
    r"^\s*(KNOWN[ _-]?ANSWER|NEGATIVE CONTROL|POSITIVE CONTROL|RANDOM CONTROL|"
    r"CONTROL|DEGENERATE(?: CONTROL)?)\b", re.IGNORECASE)

#: An estimate and its interval written as one triple: `+0.220 [+0.045, +0.412]`.
#: Both brackets and both signs optional, the estimate adjacent to the bracket
#: (which is what makes this a quoted result and not two numbers sharing a
#: line), and EVERY NUMBER CARRIES A DECIMAL POINT. An integer estimate would
#: match any row within half a unit of it, which on a correlation is almost
#: every row; `\s*` spans the newline a hard-wrapped LOG puts between the
#: estimate and its bracket, so the scan runs over the whole text and takes the
#: line from the match offset (two of the five real cases are wrapped).
PROSE_TRIPLE = re.compile(
    r"(?<![\d.])([+-]?\d+\.\d+)\s*[\[(]\s*([+-]?\d+\.\d+)\s*,\s*"
    r"([+-]?\d+\.\d+)\s*[\])]")

#: Prose files scanned beside an experiment's verdicts. `results/` is excluded
#: because what is in there was written by a rule, not by a person; `data`,
#: `work` and `logs` because a table of numbers is not prose and reading one is
#: how this check becomes slow and noisy.
PROSE_GLOBS: tuple[str, ...] = ("*.md", "*.txt", "context.toml", "context.pending.toml")
PROSE_SKIP: tuple[str, ...] = ("results", "figures", "data", "work", "logs", "scripts",
                               ".snakemake", ".git", "__pycache__", ".pytest_cache")

#: A prose file larger than this is not read. A generated table of ten thousand
#: rows is not prose and scanning it is how this check becomes slow and noisy.
MAX_PROSE_BYTES = 2_000_000

#: How many adjudicated rows a quoted estimate may match before the match stops
#: being evidence that the prose is quoting any of them. OUR CHOICE, NO SOURCE,
#: measured: `0.00 [-0.12, 0.19]` in one experiment's prose matched four rows,
#: two of them 0 [0, 0], because every near-zero estimate rounds to 0.00 at two
#: decimals. The check's whole premise is "this sentence is quoting that row",
#: and with several rows sharing the estimate the premise is not established, so
#: the triple is left alone. Every real conflict found so far matches one row.
MAX_CANDIDATES = 3


def _finite(v) -> float | None:
    """`v` as a float when it is one and is finite, else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def at_a_bound(row: dict) -> str | None:
    """Which bound the estimate sits on, or None.

    Two kinds of bound, and both mean the same thing for this warning: no
    further evidence could have moved the estimate further in that direction.
    The end of what the estimator can produce (a share of 1, a correlation of
    -1), and a finite bound of the claim's own `claimed_range`, which is the
    line the verdict was decided against.
    """
    v = _finite(row.get("estimate"))
    if v is None:
        return None
    lo, hi = METHOD_RANGE.get(str(row.get("method", "")), (None, None))
    if hi is not None and abs(v - hi) <= BOUND_TOL:
        return f"the highest value {row['method']} can produce ({hi:g})"
    if lo is not None and abs(v - lo) <= BOUND_TOL:
        return f"the lowest value {row['method']} can produce ({lo:g})"
    rng = row.get("claimed_range")
    if isinstance(rng, list) and len(rng) == 2:
        for end, name in ((rng[1], "upper"), (rng[0], "lower")):
            b = _finite(end)
            if b is not None and abs(v - b) <= BOUND_TOL:
                return f"the {name} bound of its own claimed range ({b:g})"
    return None


def kept_rows(row: dict) -> int | None:
    """How many ROWS the estimate was computed on.

    `dropped` counts rows (`relate._paired` increments it per row it cannot
    read), so the share has to be taken against rows as well. `n_units` counts
    resampling units, and on a clustered claim the two are different numbers:
    1,000 rows in 40 clusters with 80 rows dropped is 7.4% dropped, not the 67%
    that `n_units + dropped` would report. `n_rows` is the kept-row count and is
    what this reads; `n_units` is the fallback for a row that carries no
    `n_rows`, where they are the same thing.
    """
    for key in ("n_rows", "n_units"):
        v = row.get(key)
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            return v
    return None


def dropped_share(row: dict) -> tuple[int, int, float] | None:
    """`(dropped, rows_in, share)` for a row that reports both, else None.

    `n_rows` is what survived and `dropped` is what did not, so the rows that
    went in are their sum. Counting the share against the kept rows alone would
    report 80 dropped of 40 as 200%.
    """
    d = row.get("dropped")
    n = kept_rows(row)
    if not isinstance(d, int) or isinstance(d, bool) or d <= 0 or n is None:
        return None
    rows_in = n + d
    if rows_in == 0:
        return None
    return d, rows_in, d / rows_in


def worst_case_share(row: dict, dropped: int) -> float | None:
    """The share this claim would report had every dropped unit come back the
    other way, for the methods whose estimate is exactly a count over units.

    A POINT, NOT AN INTERVAL, and it is reported as one. It says how much room
    the drops left, never what the verdict would have been: a verdict needs an
    interval and this has none.
    """
    if str(row.get("method", "")) not in SHARE_METHODS:
        return None
    v = _finite(row.get("estimate"))
    n = kept_rows(row)
    if v is None or n is None or n <= 0:
        return None
    # `fraction` carries no `n_success` (measured on I1, 2026-11-17), so the count is
    # recovered from the estimate and the ROWS it was taken over, never from the units.
    n_success = row.get("n_success")
    k = float(n_success) if isinstance(n_success, int) and not isinstance(n_success, bool) else v * n
    # Dropped units counted the way that moves the estimate AWAY from the bound
    # it is sitting on: against a high estimate they all failed, against a low
    # one they all passed.
    if v >= 0.5:
        return k / (n + dropped)
    return (k + dropped) / (n + dropped)


def _in_claimed_range(row: dict, v: float) -> bool | None:
    """Whether a point lies inside the claim's `claimed_range`, or None when the
    claim states a point `claimed` instead and there is no range to lie in.

    THE NONE MATTERS. `None is False` is False, so folding the two together made
    a point claim read "the drops could not have changed the verdict", which is
    a reassurance nothing had established (found 2026-11-17 before this was ever
    run on a point claim with drops). A point `claimed` is judged against an
    interval, and the worst case has none, so there is nothing to say."""
    rng = row.get("claimed_range")
    if not (isinstance(rng, list) and len(rng) == 2):
        return None
    lo, hi = rng[0], rng[1]
    lo = -math.inf if _finite(lo) is None else float(lo)
    hi = math.inf if _finite(hi) is None else float(hi)
    return bool(lo <= v <= hi)


def one_column(row: dict) -> bool:
    """Whether the claim reads ONE column (`x = y`), which is the convention for
    every one-column method.

    It is what makes the sharper form of `cannot_fail` decidable from the row:
    where there is one column, a dropped row is a row whose value for that
    column was missing or non-finite, so it was dropped FOR THE PROPERTY THE
    CLAIM MEASURES. With two columns the drop may have come from either, and the
    row does not say which."""
    x, y = row.get("x"), row.get("y")
    return bool(x) and x == y


def _fixed_by_construction(row: dict) -> dict | None:
    """An estimate that did not vary across ANY resample, at a value the estimator could have moved away from.

    THE SHAPE. The bootstrap draws whole resampling units with replacement. If the units differ from one another at
    all, some draw must differ, so the interval has width. A ZERO-WIDTH interval over two or more units means every
    unit returned the identical value, and for anything measured from a stochastic simulation that does not happen.
    The quantity was fixed by how it is produced, not by what was observed.

    THE INCIDENT, 2026-11-25. `rbfe_known_answer`'s `K1_self_edge_returns_zero` is the known answer a lab's whole
    relative free-energy method rested on: transform a ligand into an identical copy and the answer must be 0. It
    agreed at exactly 0.0 over 3 repeats with an interval of exactly [0, 0], and the agreement was worth nothing. When
    the two end states are the same molecule the Hamiltonian does not change with lambda, so every reduced potential
    is identical and MBAR returns exactly zero WHATEVER SAMPLES WERE DRAWN. Twenty-five minutes of dynamics could not
    have moved it. The claim's own registered text asserted that a refutation would catch a wrong unit conversion, a
    wrong charge assignment or a broken gather; a wrong unit conversion multiplies zero, wrong charges apply to both
    states alike, and a gather returning zero is indistinguishable from success. It could catch one of the four
    failures it claimed to catch, and nobody noticed for the ten days the experiment sat unadjudicated.

    WHY IT IS NOT `cannot_fail`. That code looks for an estimate pinned at the END of what the estimator can produce,
    with units dropped on the way: a share of 1.0, a correlation of -1. This estimate is INTERIOR to its estimator's
    range and no units were dropped. The two diagnoses are different and so is the repair: `cannot_fail` is usually
    fixed by a cohort or a threshold, this one only by a different experiment.

    THE ONE ZERO WIDTH THAT IS DESIGNED, and is therefore excluded rather than reported. The paired-difference
    methods (`roc_auc_difference`, `bedroc_difference`, anything else declaring a `baseline`) have a documented known
    answer in which a column is compared against a copy of itself: it must return exactly 0.0 AND its interval must
    be exactly [0, 0], because a resample that drew the two columns independently would give the same 0.0 estimate
    with a WIDE interval. There the zero width is not a symptom, it is the entire evidence that the bootstrap is
    paired, and a warning on it would fire on every correct use of the pattern. A claim declaring a `baseline` is
    therefore left alone, which also keeps an existing contract: `tests/test_bedroc_difference.py` asserts that no
    warning reaches such a row.

    Fires only where the estimate is interior to the estimator's range, so the share-of-1.0 and correlation-of-1.0
    cases stay with `cannot_fail` and are not reported twice.
    """
    lo, hi = _finite(row.get("ci_lo")), _finite(row.get("ci_hi"))
    est = _finite(row.get("estimate"))
    if lo is None or hi is None or est is None:
        return None
    if abs(hi - lo) > ZERO_WIDTH_TOL:
        return None
    n = row.get("n_units")
    if not isinstance(n, int) or n < 2:
        # One unit cannot produce a width, and `relate` already reports that as unverifiable with its own reason.
        return None
    if at_a_bound(row) is not None:
        return None                                    # `cannot_fail`'s territory; do not report the same row twice.
    if str(row.get("baseline") or "").strip():
        # A paired-difference claim. Its self-comparison known answer is REQUIRED to have a zero-width interval, and
        # that width is the proof the bootstrap is paired. See the docstring.
        return None
    # NOT RESTRICTED TO is_control(), AND THE ATTEMPT TO DO SO IS WHY. Restricting to claims that name themselves a
    # control or known answer cut the corpus-wide count usefully and fired ZERO times on the experiment this code was
    # written for: `K1_self_edge_returns_zero` carries no KNOWN_ANSWER token in its id and its evidence note opens
    # "THE KNOWN ANSWER ...", which the CONTROL_LABEL regex does not match because the label must start the note. A
    # narrowing that silences the motivating case is not a narrowing, it is a different and useless check.
    unit = row.get("resampling_unit") or "unit"
    return {"claim": row.get("id") or f"{row.get('x')} ~ {row.get('y')}",
            "id": row.get("id", ""),
            "code": FIXED_BY_CONSTRUCTION,
            "method": row.get("method"), "verdict": row.get("verdict", ""),
            "estimate": est, "n_units": n, "resampling_unit": row.get("resampling_unit"),
            "why": (
                f"{row.get('method')} = {est:g} came back with an interval of exactly zero width over {n} "
                f"{unit}(s), at a value interior to what {row.get('method')} can produce. The bootstrap draws whole "
                f"{unit}s, so a zero width means EVERY {unit} returned the identical value. For a quantity measured "
                f"from a stochastic run that does not happen, and the estimate is fixed by how it is produced rather "
                f"than by what was observed: such a claim passes whether or not the thing it is supposed to validate "
                f"works. Check what could have made this number differ, and if nothing could, the repair is a "
                f"different experiment rather than a different bound. It IS the point for a column compared against "
                f"a copy of itself, where the zero width is what proves the resampling is paired."),
            "detail": "zero-width resampled interval over 2 or more units at an interior value"}


def _cannot_fail(row: dict) -> dict | None:
    """A claim that passed with its estimate pinned at a bound and units dropped."""
    if row.get("verdict") != AGREES:
        return None
    bound = at_a_bound(row)
    if bound is None:
        return None
    share = dropped_share(row)
    if share is None:
        return None
    dropped, rows_in, frac = share
    same_property = one_column(row)
    worst = worst_case_share(row, dropped)
    inside = None if worst is None else _in_claimed_range(row, worst)
    decided = None if inside is None else (not inside)
    unit = row.get("resampling_unit")
    over = f", over {row['n_units']} {unit or 'unit'}(s)" if isinstance(row.get("n_units"), int) else ""
    why = (f"{row.get('method')} = {row['estimate']} sits on {bound} and "
           f"{dropped} of the {rows_in} rows that went in were dropped "
           f"({frac:.1%}{', which is material' if frac >= MATERIAL_DROP else ''}{over}). ")
    if same_property:
        why += ("The claim reads one column (x = y), so every dropped row was dropped for the very "
                "property it measures: the rows that could have failed are the rows that are gone. ")
    else:
        why += ("The claim reads two columns, so the row does not say which of them the drops came "
                "from; read the rule that wrote the table. ")
    if worst is not None:
        why += (f"Had every dropped row come back the other way the share would be {worst:.4g}, a "
                f"point and not an interval")
        if decided is True:
            why += ", outside the claimed range: the drops alone decided this verdict. "
        elif decided is False:
            why += (", still inside the claimed range, so the drops could not have changed the "
                    "verdict, which is not the same as the dropped rows being accounted for. ")
        else:
            why += (", and this claim states a point value rather than a range, so whether that "
                    "would have changed the verdict is a question about an interval nobody has. ")
    why += "A pass that the dropped rows could not have prevented is not evidence that it would have held."
    return {"claim": row.get("id") or f"{row.get('x')} ~ {row.get('y')}",
            "code": CANNOT_FAIL, "method": row.get("method"), "verdict": row.get("verdict"),
            "estimate": row.get("estimate"), "bound": bound,
            "dropped": dropped, "rows_in": rows_in, "n_units": row.get("n_units"),
            "dropped_share": round(frac, 6),
            "material": bool(frac >= MATERIAL_DROP),
            "dropped_for_the_measured_property": same_property,
            "worst_case_share": None if worst is None else round(worst, 6),
            "drops_alone_decided_it": decided,
            "why": why}


def is_control(row: dict) -> bool:
    """Whether the claim says of itself that it is a control, a degenerate case
    or a known answer, by an id token or by the label its evidence note opens
    with. Nothing else is read: see CONTROL_TOKENS."""
    ident = str(row.get("id") or "")
    tokens = [t for t in re.split(r"[_\s-]+", ident) if t]
    # CASE-SENSITIVE, because the convention is a shouted label at the end of the id.
    # Case-folding first made `T1_most_of_the_set_beats_its_control_on_both_receptors`
    # match the CONTROL token, and a claim measured against a control is not one.
    joined = "_".join(tokens)
    for tok in CONTROL_TOKENS:
        if re.search(rf"(?:^|_){re.escape(tok)}(?:_|$)", joined):
            return True
    # The other convention in the corpus is a lowercase `..._control` ENDING, and
    # only an ending: `T1_most_of_the_set_beats_its_control_on_both_receptors` is a
    # claim measured against a control and is not one, and an infix match called it
    # one (measured on cry1_shortlist_rungs, 2026-11-17).
    if ident.lower().endswith("_control"):
        return True
    ev = row.get("evidence")
    if isinstance(ev, dict):
        return bool(CONTROL_LABEL.match(str(ev.get("source", "") or "")))
    return False


def _refuted_control(row: dict) -> dict | None:
    """A control, degenerate case or known answer that came back refuted."""
    if row.get("verdict") != REFUTED or not is_control(row):
        return None
    return {"claim": row.get("id") or f"{row.get('x')} ~ {row.get('y')}",
            "code": REFUTED_CONTROL, "method": row.get("method"), "verdict": REFUTED,
            "estimate": row.get("estimate"), "claimed": row.get("claimed"),
            "claimed_range": row.get("claimed_range"),
            "why": (f"this claim calls itself a control, a degenerate case or a known answer, and it is "
                    f"REFUTED: {row.get('why', '')} Either the claimed value was registered the wrong way "
                    f"round for the behaviour the control is meant to show, or the thing every claim beside "
                    f"it rests on has failed. Both need a person; neither is read from the id.")}


def scrutinise(rows: list[dict]) -> list[dict]:
    """Every warning the finished verdict rows raise, in the order the rows came.

    Pure: it reads rows and returns records. It is called at the end of
    `relate.check` so it fires on every adjudication, and from this module's
    command line so a directory already adjudicated can be swept without
    re-running the checker over it."""
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for check in (_cannot_fail, _refuted_control, _fixed_by_construction):
            found = check(row)
            if found is not None:
                out.append(found)
    #: A SET-LEVEL check, unlike the two above: it asks whether ANY claim in this adjudication validates the distance
    #: measure the others rely on, so it cannot be decided from one row.
    out.extend(_unvalidated_distance([r for r in rows if isinstance(r, dict)]))
    return out


def is_distance(row: dict) -> bool:
    """Does this row report a distance? Decided from the declared unit, which is the only place a unit is recorded."""
    for key in ("y_unit", "unit", "x_unit"):
        u = str(row.get(key) or "").strip().lower()
        if u in DISTANCE_UNITS:
            return True
    return False


def validates_a_distance(row: dict) -> bool:
    """Does this row prove that the distance measure reports a known displacement?

    Three things together, and all three are needed. It must be about a displacement (the id or the evidence says so), it
    must have AGREED (a validator that failed validates nothing), and its estimate must be NON-ZERO. The last is the whole
    point: a claim that a displaced copy reads 0.0 and agrees with a claimed 0.0 is the defect, not the check for it.
    """
    text = f"{row.get('id', '')} {row.get('evidence_source', '')} {row.get('evidence', '')} {row.get('why', '')}"
    if not _DISPLACEMENT_WORDS.search(text):
        return False
    if str(row.get("verdict", "")).lower() != AGREES:
        return False
    est = _finite(row.get("estimate"))
    return est is not None and abs(est) > BOUND_TOL


def _unvalidated_distance(rows: list[dict]) -> list[dict]:
    """Every distance claim in an adjudication that has no claim beside it proving the ruler measures distance.

    WHY THIS CODE EXISTS. A distance is the one quantity whose instrument can fail by returning ZERO, and zero passes
    every upper bound placed on it. This lab has been caught twice. RDKit's `GetBestRMS` SUPERPOSES before measuring, so a
    ligand displaced 25 angstrom scored 0.0 angstrom and every receptor in a selection passed, which would have chosen a
    receptor that cannot reproduce its own crystal pose. And on 2026-11-23 a constrained embed silently returned conformers
    38 angstrom from their reference while every composition and charge check on them passed; two further attempts to place
    them gave 1.4 and 6.2 angstrom, and the only reason any of the three was caught is that a known displacement was
    measured alongside.

    So: if an adjudication reports a distance and nowhere demonstrates that its measure recovers a known displacement, the
    distances in it are unreadable, however tight their intervals. This does not change a verdict or an exit status; it
    says the number has no established ruler behind it.

    The fix an experiment makes is one extra claim: translate a copy by a declared offset, measure it with the same code
    path, and claim that offset. It costs no simulation.
    """
    distances = [r for r in rows if is_distance(r)]
    if not distances:
        return []
    if any(validates_a_distance(r) for r in rows):
        return []
    # `claim` AND `why` ARE THE TWO KEYS EVERY WARNING CARRIES, because `relate.main` prints exactly those two for each
    # warning it writes. This code emitted `id` and `detail` instead, so any adjudication that triggered it computed
    # every verdict and then died with KeyError: 'claim' while printing, writing no results/claims.json at all (found
    # 2026-11-25 on h08_polar_deletion, whose pKa claims report residue-to-ligand distances). `id` and `detail` are kept
    # beside the contract keys rather than renamed away, so anything already reading them still does.
    return [dict(code=UNVALIDATED_DISTANCE, id=r.get("id", ""), claim=r.get("id", ""),
                 why=("this claim reports a distance and no claim in this adjudication demonstrates that the measure "
                      "recovers a known displacement. A superposing or mis-framed distance returns 0.0 for a wrong "
                      "answer and passes every upper bound; add one claim that moves a copy by a declared offset and "
                      "recovers it"),
                 detail=("this claim reports a distance and no claim in this adjudication demonstrates that the measure "
                         "recovers a known displacement. A superposing or mis-framed distance returns 0.0 for a wrong "
                         "answer and passes every upper bound; add one claim that moves a copy by a declared offset and "
                         "recovers it"),
                 unit=str(r.get("y_unit") or r.get("unit") or ""),
                 estimate=r.get("estimate"), verdict=r.get("verdict", ""),
                 n_distance_claims=len(distances))
            for r in distances]


# ---------------------------------------------------------------------------
# Prose against the adjudicated row
# ---------------------------------------------------------------------------

def _decimals(text: str) -> int:
    """How many decimal places a number was printed to."""
    _, _, frac = text.partition(".")
    return len(frac)


def _matches(printed: str, value) -> bool:
    """Whether a row's value is the number as printed, at the precision it was printed to.

    ONE UNIT IN THE LAST PLACE, not half of one. Half a unit is what rounding
    costs; a number written out by hand, truncated, or copied from a line that
    rounded the other way can be a whole unit off, and treating that as a
    contradiction reported two real pairs that agree: `-0.2856` against
    -0.285674 (truncated) and `0.54` against 0.534591 (rounded the wrong way),
    both measured in the corpus sweep on 2026-11-17. The conflicts this check
    exists for are orders of magnitude larger: +0.045 against +0.146105."""
    v = _finite(value)
    if v is None:
        return False
    d = _decimals(printed)
    return abs(v - float(printed)) <= (10.0 ** -d) + 1e-9


def prose_files(root: Path) -> list[Path]:
    """The prose beside an experiment: its Markdown, its text notes and its
    declaration, excluding everything a rule wrote."""
    found: list[Path] = []
    for pattern in PROSE_GLOBS:
        for p in sorted(root.rglob(pattern)):
            rel = p.relative_to(root).parts
            if any(part in PROSE_SKIP for part in rel[:-1]):
                continue
            if p.is_file() and p.stat().st_size <= MAX_PROSE_BYTES:
                found.append(p)
    return found


def prose_conflicts(root: Path | str, rows: list[dict]) -> list[dict]:
    """Numbers in an experiment's own prose that contradict its adjudicated rows.

    The matcher is narrow on purpose. It looks only for an estimate written
    immediately before its interval, `+0.220 [+0.045, +0.412]`, and reports it
    only when at least one adjudicated row's estimate matches the printed
    estimate at the precision it is printed to AND no such row's interval
    matches. A triple matching nothing in this directory is silent: it is a
    number about something else, and a check that reads it is the loose matcher
    that drowns everyone.
    """
    root = Path(root)
    usable = [r for r in rows if isinstance(r, dict)
              and _finite(r.get("estimate")) is not None
              and _finite(r.get("ci_lo")) is not None
              and _finite(r.get("ci_hi")) is not None]
    out: list[dict] = []
    for path in prose_files(root):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue                    # unreadable prose is not a conflict
        # OVER THE WHOLE TEXT, not line by line: a hard-wrapped LOG puts the estimate
        # at the end of one line and its bracket at the start of the next, and a
        # line-by-line scan silently missed two of the five real cases.
        for m in PROSE_TRIPLE.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            est, lo, hi = m.group(1), m.group(2), m.group(3)
            same_estimate = [r for r in usable if _matches(est, r["estimate"])]
            if not same_estimate or len(same_estimate) > MAX_CANDIDATES:
                continue                # nothing to quote, or too many to be quoting one
            if any(_matches(lo, r["ci_lo"]) and _matches(hi, r["ci_hi"])
                   for r in same_estimate):
                continue
            names = ", ".join(str(r.get("id") or f"{r.get('x')} ~ {r.get('y')}")
                              for r in same_estimate[:4])
            shown = "; ".join(f"{r['estimate']:.6g} [{r['ci_lo']:.6g}, {r['ci_hi']:.6g}]"
                              for r in same_estimate[:4])
            out.append({
                "claim": same_estimate[0].get("id") or "",
                "code": PROSE_CONFLICT,
                "file": str(path.resolve()), "line": line_no,
                "prose": f"{est} [{lo}, {hi}]",
                "candidates": [str(r.get("id") or "") for r in same_estimate[:4]],
                "why": (f"{path.resolve()}:{line_no} quotes {est} [{lo}, {hi}]. The estimate matches "
                        f"the adjudicated row(s) {names}, whose interval is not that one: {shown}. "
                        f"The verdict is what the verdict file says, and prose is not a source."),
            })
    return out


# ---------------------------------------------------------------------------
# Reading an adjudicated directory
# ---------------------------------------------------------------------------

def audit(root: Path | str, claims: Path | str | None = None) -> dict:
    """Every warning for one adjudicated experiment directory.

    Reads the verdicts that are on disk rather than re-running `relate`: this is
    the sweep, and a sweep that re-adjudicates would be a second answer to a
    question already answered.
    """
    root = Path(root)
    path = Path(claims) if claims is not None else root / "results" / "claims.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no adjudicated verdicts at {path}. Run the `check_claims` rule "
            f"(python -m claimcheck.relate {root}) before scrutinising them.")
    doc = json.loads(path.read_text())
    rows = doc.get("relationships") or []
    found = scrutinise(rows) + prose_conflicts(root, rows)
    counts: dict[str, int] = {c: 0 for c in CODES}
    for w in found:
        counts[w["code"]] = counts.get(w["code"], 0) + 1
    return {"root": str(root.resolve()), "claims": str(path.resolve()),
            "claims_read": len(rows), "warnings": found, "counts": counts,
            "total": len(found)}


def render(w: dict) -> str:
    """One warning as the two lines a person reads."""
    where = f" {w['file']}:{w['line']}" if w.get("file") else ""
    return f"  {w['code'].upper():16} {w.get('claim') or '?'}{where}\n      {w['why']}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Shapes that make a verdict untrustworthy: a claim that could "
                    "not have failed, a control that was refuted, prose that "
                    "contradicts the row it quotes. Warnings only; they change no "
                    "verdict.")
    ap.add_argument("root", nargs="?", default=".", help="the experiment directory")
    ap.add_argument("--claims", help="the verdict file (default: <root>/results/claims.json)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if anything was raised, for a caller that wants a status")
    ap.add_argument("--json", metavar="OUT", help="write the full report here")
    a = ap.parse_args(argv)
    try:
        rep = audit(a.root, a.claims)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for w in rep["warnings"]:
        print(render(w))
    print(f"\n{rep['total']} warning(s) over {rep['claims_read']} claim(s) in {rep['claims']}"
          + (": " + ", ".join(f"{k} {v}" for k, v in rep["counts"].items() if v) if rep["total"] else ""))
    if rep["total"]:
        print("  None of these changes a verdict. Each is a shape that is usually a defect and is "
              "sometimes the point, so each needs a person.")
    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2) + "\n")
        print(f"  -> {out}")
    return 1 if (a.check and rep["total"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
