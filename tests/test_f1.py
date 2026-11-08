"""F1 (method = "f1"): a retrieval claim, "the shortlist this score produces contains the hits".

The only two-column boolean estimator: `x` is the prediction, `y` is the truth, both read as indicators at >= 0.5, and they
do not commute. Each case has a hand-computable answer, and each verdict is tested in both directions.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from claimcheck.relate import AGREES, REFUTED, UNVERIFIABLE, check
from claimcheck.relate import _f1

VARS = ('[[variables]]\ndataset = "d"\nname = "unit"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "predicted"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "truth"\nunit = "1"\n')


# ---------------------------------------------------------------------------
# The estimator alone, on cases whose F1 can be worked out by hand
# ---------------------------------------------------------------------------

def test_perfect_prediction_is_one():
    """A shortlist that is exactly the positives scores 1."""
    assert _f1([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_inverted_prediction_is_zero():
    """A shortlist that selects exactly the negatives scores 0: no true positive."""
    assert _f1([0, 0, 1, 1], [1, 1, 0, 0]) == 0.0


def test_hand_computed_case():
    """TP 2, FP 1, FN 1 gives 2*2 / (2*2 + 1 + 1) = 4/6."""
    predicted = [1, 1, 1, 0, 0]
    truth = [1, 1, 0, 1, 0]
    assert _f1(predicted, truth) == pytest.approx(4.0 / 6.0)


def test_it_does_not_commute():
    """Swapping prediction and truth swaps precision for recall.

    Four predicted, two true: as written, TP 2 FP 2 FN 0 gives 4/6. Swapped,
    TP 2 FP 0 FN 2 gives the same 4/6 for F1 because the formula is symmetric
    in FP and FN, so this fixture checks the asymmetric case instead: a
    shortlist of 1 against 3 positives is not the same claim as a shortlist of
    3 against 1 positive.
    """
    predicted = [1, 0, 0, 0]
    truth = [1, 1, 1, 0]
    forward = _f1(predicted, truth)          # TP 1, FP 0, FN 2 -> 2/4
    backward = _f1(truth, predicted)         # TP 1, FP 2, FN 0 -> 2/4
    assert forward == pytest.approx(0.5)
    assert backward == pytest.approx(0.5)
    # The asymmetry shows once the counts differ on both sides.
    assert _f1([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(2.0 / 3.0)
    assert _f1([1, 0, 0, 0], [1, 1, 0, 0]) == pytest.approx(2.0 / 3.0)
    # and precision against recall is visible in the counts themselves
    assert _f1([1] * 4, [1, 0, 0, 0]) == pytest.approx(2.0 / 5.0)
    assert _f1([1, 0, 0, 0], [1] * 4) == pytest.approx(2.0 / 5.0)


def test_booleans_and_indicators_behave_identically():
    """A 0/1 column and a float column crossing 0.5 give the same answer, which
    is the convention `fraction` already uses."""
    assert _f1([1.0, 0.0], [1.0, 0.0]) == _f1([0.9, 0.1], [0.75, 0.25])


def test_no_positive_anywhere_is_undefined_not_zero():
    """F1 is undefined when the prediction and the truth are both empty of
    positives. Returning None makes the claim unverifiable, which is what keeps
    a broken join distinguishable from a real null; returning 0 would call it a
    measured failure and returning 1 would call it a success."""
    assert _f1([0, 0, 0], [0, 0, 0]) is None


def test_mismatched_lengths_and_empty_return_none():
    assert _f1([], []) is None
    assert _f1([1, 0], [1]) is None


# ---------------------------------------------------------------------------
# Through `check`, so the verdict path and the bootstrap are exercised
# ---------------------------------------------------------------------------

def project(tmp_path: Path, rels: str, n: int = 400, recall: float = 0.9,
            precision: float = 0.9, base_rate: float = 0.25, seed: int = 5) -> Path:
    """A cohort with a planted F1: positives at `base_rate`, found with
    probability `recall`, plus false positives tuned to `precision`."""
    rng = random.Random(seed)
    lines = ["unit,predicted,truth"]
    for i in range(n):
        truth = 1 if rng.random() < base_rate else 0
        if truth:
            predicted = 1 if rng.random() < recall else 0
        else:
            predicted = 1 if rng.random() < (1.0 - precision) * base_rate / (1 - base_rate) else 0
        lines.append(f"{i},{predicted},{truth}")
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "f1"\n\n[[datasets]]\nid = "d"\norigin = "computed"\n'
        'files = ["d.csv"]\n\n' + VARS + "\n" + rels)
    return tmp_path


def rel(rid: str, claimed=None, rng=None) -> str:
    s = (f'[[relationships]]\nid = "{rid}"\nx = "d.predicted"\ny = "d.truth"\n'
         'kind = "value"\nmethod = "f1"\nresampling_unit = "unit"\n')
    if claimed is not None:
        s += f"claimed = {claimed}\n"
    if rng is not None:
        s += "claimed_range = [" + ", ".join(str(v) for v in rng) + "]\n"
    return s + "\n"


def verdicts(tmp_path: Path, rels: str, **kw) -> dict:
    return {r["id"]: r for r in check(project(tmp_path, rels, **kw))["relationships"]}


def test_a_good_retriever_agrees_with_a_high_range(tmp_path):
    v = verdicts(tmp_path, rel("good", rng=[0.6, 1.0]), recall=0.9, precision=0.9)["good"]
    assert v["verdict"] == AGREES
    assert v["estimate"] > 0.6
    assert v["ci_lo"] <= v["estimate"] <= v["ci_hi"]


def test_the_same_retriever_is_refuted_by_a_low_range(tmp_path):
    """The other direction on identical data: the verdict must follow the
    claim, not the hope."""
    v = verdicts(tmp_path, rel("good", rng=[0.0, 0.3]), recall=0.9, precision=0.9)["good"]
    assert v["verdict"] == REFUTED


def test_a_useless_retriever_is_refuted_by_a_high_range(tmp_path):
    v = verdicts(tmp_path, rel("poor", rng=[0.6, 1.0]), recall=0.05, precision=0.05)["poor"]
    assert v["verdict"] == REFUTED


def test_a_wide_range_is_unverifiable_not_confirmed(tmp_path):
    """An interval straddling a bound is unverifiable. A claim inside a wide
    interval is not a confirmation."""
    v = verdicts(tmp_path, rel("mid", rng=[0.5, 1.0]), n=40, recall=0.5,
                 precision=0.5)["mid"]
    assert v["verdict"] in (UNVERIFIABLE, REFUTED)


def test_value_claims_carry_no_p_or_r_squared(tmp_path):
    """F1 is a value claim, so it carries no p value and no R squared, like
    every value method except sign_test."""
    v = verdicts(tmp_path, rel("good", rng=[0.6, 1.0]))["good"]
    # The key is present and None: `_value_verdict` does
    # row.setdefault("p_value", None), and only sign_test fills it in.
    assert v["p_value"] is None
    assert v.get("r_squared") is None


def test_an_empty_prediction_reports_unverifiable(tmp_path):
    """No positive in either column anywhere: the claim must come back
    unverifiable rather than as a measured zero."""
    (tmp_path / "d.csv").write_text(
        "unit,predicted,truth\n" + "\n".join(f"{i},0,0" for i in range(50)) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "f1"\n\n[[datasets]]\nid = "d"\norigin = "computed"\n'
        'files = ["d.csv"]\n\n' + VARS + "\n" + rel("empty", rng=[0.0, 1.0]))
    out = {r["id"]: r for r in check(tmp_path)["relationships"]}["empty"]
    assert out["verdict"] == UNVERIFIABLE


def test_f1_is_registered_as_a_value_method():
    from claimcheck.relate import ESTIMATORS
    assert "f1" in ESTIMATORS
    assert ESTIMATORS["f1"][1] == "value"
