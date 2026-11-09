"""`mean_diff`: mean(y | x = 1) - mean(y | x = 0) adjudicated by relate (2026-11-09), for "two groups of replicas agree within a
tolerance". Two-sided known answers (an exact difference agrees on a range around it and is refuted on a range around 0), a random
split of one population straddles a narrow range, and a group column that is not 0/1, or an empty group, is `unverifiable`.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from claimcheck.relate import AGREES, REFUTED, UNVERIFIABLE, _mean_diff, check

VARS = ('[[variables]]\ndataset = "d"\nname = "unit"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "g"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "v"\nunit = "1"\n')


def project(tmp_path: Path, rows: list[tuple[str, object, float]], rels: str) -> Path:
    (tmp_path / "d.csv").write_text("unit,g,v\n" + "".join(f"{u},{g},{v!r}\n" for u, g, v in rows))
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "m"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + VARS + "\n" + rels)
    return tmp_path


def rel(rid: str, rng) -> str:
    return (f'[[relationships]]\nid = "{rid}"\nx = "d.g"\ny = "d.v"\nkind = "value"\nmethod = "mean_diff"\n'
            f'resampling_unit = "unit"\nclaimed_range = [{rng[0]!r}, {rng[1]!r}]\n\n')


def verdicts(tmp_path, rows, rels) -> dict:
    return {r["id"]: r for r in check(project(tmp_path, rows, rels))["relationships"]}


def test_the_function():
    assert _mean_diff([1, 1, 0, 0], [3.0, 5.0, 1.0, 1.0]) == 3.0
    assert _mean_diff([1, 1], [3.0, 5.0]) is None and _mean_diff([0], [1.0]) is None


def test_an_exact_difference_both_ways(tmp_path):
    """Every unit of group 1 holds 0.9 and every unit of group 0 holds 0.15, ten frames each: 0.75 in every draw that has both groups."""
    rows = [(f"a{u}", 1, 0.9) for u in range(6) for _ in range(10)] + [(f"b{u}", 0, 0.15) for u in range(6) for _ in range(10)]
    v = verdicts(tmp_path, rows, rel("right", [0.75 - 1e-9, 0.75 + 1e-9]) + rel("zero", [-0.1, 0.1]))
    assert v["right"]["verdict"] == AGREES, v["right"]["why"]
    assert v["zero"]["verdict"] == REFUTED
    assert v["right"]["estimate"] == pytest.approx(0.75) and v["right"]["n_group_1"] == 60 and v["right"]["n_group_0"] == 60
    assert v["right"]["p_value"] is None and v["right"]["r_squared"] is None


def test_a_random_split_of_one_population_is_not_resolved_on_a_narrow_range(tmp_path):
    rng = random.Random(11)
    rows = []
    for u in range(12):
        share = rng.choice([0.0, 0.05, 0.5, 0.97])             # bimodal per-unit values, as replicas of a slow torsion are
        g = u % 2
        rows += [(f"u{u}", g, 1.0 if k < 20 * share else 0.0) for k in range(20)]
    r = verdicts(tmp_path, rows, rel("tight", [-0.02, 0.02]) + rel("loose", [-1.0, 1.0]))
    assert r["tight"]["verdict"] == UNVERIFIABLE and "straddles" in r["tight"]["why"]
    assert r["loose"]["verdict"] == AGREES


def test_a_group_column_other_than_0_or_1_is_refused(tmp_path):
    rows = [(f"u{u}", u % 3, 1.0) for u in range(9)]
    r = verdicts(tmp_path, rows, rel("a", [-1, 1]))["a"]
    assert r["verdict"] == UNVERIFIABLE and "x = 0 or 1" in r["why"]


def test_an_empty_group_is_undefined(tmp_path):
    rows = [(f"u{u}", 1, 1.0) for u in range(9)]
    r = verdicts(tmp_path, rows, rel("a", [-1, 1]))["a"]
    assert r["verdict"] == UNVERIFIABLE and "undefined" in r["why"]
