"""`tie_tol`: values of a spearman claim within the tolerance of one another rank as ties. A known-answer claim of 1.0 whose estimates tie in
exact arithmetic (eight seeds sharing one dF) must not be refuted by the ordering of 1e-9 noise."""

from __future__ import annotations

import random
from pathlib import Path

from claimcheck.relate import AGREES, REFUTED, UNVERIFIABLE, check


def project(tmp_path: Path, rels: str, step: float = 1.0, noise: float = 1e-9, groups: int = 3, seeds: int = 4) -> Path:
    rng = random.Random(7)
    lines = ["problem,truth,estimate"]
    for g in range(groups):
        for _s in range(seeds):
            lines.append(f"p{g},{g * step},{g * step + rng.uniform(-noise, noise)}")
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "t"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n'
        '[[variables]]\ndataset = "d"\nname = "truth"\nunit = "kT"\n\n[[variables]]\ndataset = "d"\nname = "estimate"\nunit = "kT"\n\n' + rels)
    return tmp_path


def rel(extra: str = "", method: str = "spearman") -> str:
    return (f'[[relationships]]\nid = "ka"\nx = "d.truth"\ny = "d.estimate"\nkind = "correlation"\nmethod = "{method}"\nclaimed = 1.0\n'
            f'resampling_unit = "problem"\n{extra}\n')


def verdict(tmp_path: Path, rels: str, **kw) -> dict:
    return check(project(tmp_path, rels, **kw))["relationships"][0]


def test_with_tie_tol_noise_ties_give_rho_one_and_agree(tmp_path):
    v = verdict(tmp_path, rel("tie_tol = 1e-6"))
    assert v["estimate"] == 1.0 and v["verdict"] == AGREES and v["tie_tol"] == 1e-6


def test_without_tie_tol_the_same_data_is_refuted_below_one(tmp_path):
    v = verdict(tmp_path, rel())
    assert v["estimate"] < 1.0 and v["verdict"] == REFUTED and "tie_tol" not in v


def test_values_that_really_differ_are_not_merged(tmp_path):
    # the estimate of each group is off by 1e-3 in a way that reverses the order of groups 1 and 2: a real discordance, not noise
    (tmp_path / "d.csv").write_text("problem,truth,estimate\n" + "\n".join(
        f"p{g},{g},{e}" for g, e in [(0, 0.0), (0, 0.0), (1, 2.0), (1, 2.0), (2, 1.999), (2, 1.999), (3, 3.0), (3, 3.0)]) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "t"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n'
        '[[variables]]\ndataset = "d"\nname = "truth"\nunit = "kT"\n\n[[variables]]\ndataset = "d"\nname = "estimate"\nunit = "kT"\n\n'
        + rel("tie_tol = 1e-6"))
    v = check(tmp_path)["relationships"][0]
    assert v["estimate"] < 1.0 and v["tie_tol"] == 1e-6          # 1e-3 is far above tie_tol = 1e-6, so groups 1 and 2 stay swapped


def test_a_tie_tol_larger_than_the_difference_does_merge_it(tmp_path):
    (tmp_path / "d.csv").write_text("problem,truth,estimate\n" + "\n".join(
        f"p{g},{g},{e}" for g, e in [(0, 0.0), (1, 1.0), (1, 1.0), (2, 1.001), (2, 1.001), (3, 3.0), (4, 4.0)]) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "t"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n'
        '[[variables]]\ndataset = "d"\nname = "truth"\nunit = "kT"\n\n[[variables]]\ndataset = "d"\nname = "estimate"\nunit = "kT"\n\n'
        + rel("tie_tol = 0.1"))
    v = check(tmp_path)["relationships"][0]
    assert v["tie_tol"] == 0.1 and v["estimate"] < 1.0           # 1.0 and 1.001 now tie, which is not the perfect order of 1 < 2


def test_tie_tol_is_refused_for_pearson_and_for_nonpositive_values(tmp_path):
    assert verdict(tmp_path, rel("tie_tol = 1e-6", method="pearson"))["verdict"] == UNVERIFIABLE
    assert verdict(tmp_path, rel("tie_tol = 0.0"))["verdict"] == UNVERIFIABLE
    assert verdict(tmp_path, rel("tie_tol = -1.0"))["verdict"] == UNVERIFIABLE
