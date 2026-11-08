"""Correlation claims with a `claimed_range`, judged by the rule value claims use: the whole interval inside the range agrees, the
whole interval outside refutes, an interval across a bound is unverifiable. Before this, the correlation path read `rel["claimed"]`
and a range claim raised KeyError, so one such claim stopped every verdict in the file (hit_recovery_unseen_chemistry, 2026-10-21,
71 range claims). Each verdict is tested on data whose correlation is known."""

from __future__ import annotations

import random
from pathlib import Path

from claimcheck.relate import AGREES, REFUTED, UNVERIFIABLE, check

VARS = ('[[variables]]\ndataset = "d"\nname = "unit"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "a"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "b"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "noise"\nunit = "1"\n')


def project(tmp_path: Path, rels: str, n: int = 400, seed: int = 5) -> Path:
    """b = a + small noise, so rho(a, b) is close to 0.99; `noise` is independent of `a`, so rho(a, noise) is close to 0."""
    rng = random.Random(seed)
    lines = ["unit,a,b,noise"]
    for i in range(n):
        a = rng.gauss(0.0, 1.0)
        lines.append(f"{i},{a},{a + rng.gauss(0.0, 0.1)},{rng.gauss(0.0, 1.0)}")
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "c"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + VARS + "\n" + rels)
    return tmp_path


def rel(rid: str, y: str, claimed=None, rng=None) -> str:
    s = (f'[[relationships]]\nid = "{rid}"\nx = "d.a"\ny = "d.{y}"\nkind = "correlation"\nmethod = "spearman"\n'
         f'resampling_unit = "unit"\n')
    if claimed is not None:
        s += f"claimed = {claimed}\n"
    if rng is not None:
        s += "claimed_range = [" + ", ".join(("-inf" if k == 0 else "inf") if v is None else str(v) for k, v in enumerate(rng)) + "]\n"
    return s + "\n"


def verdicts(tmp_path: Path, rels: str) -> dict:
    return {r["id"]: r for r in check(project(tmp_path, rels))["relationships"]}


def test_range_agrees_refutes_or_straddles(tmp_path):
    v = verdicts(tmp_path, rel("in", "b", rng=[0.9, 1.0]) + rel("out", "b", rng=[0.0, 0.5])
                 + rel("edge", "noise", rng=[0.0, 1.0]))
    assert v["in"]["verdict"] == AGREES
    assert v["out"]["verdict"] == REFUTED
    assert v["edge"]["verdict"] == UNVERIFIABLE


def test_open_sided_ranges(tmp_path):
    v = verdicts(tmp_path, rel("above", "b", rng=[0.5, None]) + rel("below", "b", rng=[None, 0.5]))
    assert v["above"]["verdict"] == AGREES
    assert v["below"]["verdict"] == REFUTED


def test_a_range_claim_keeps_p_and_r_squared(tmp_path):
    """A correlation is a relationship, so a range claim on one still reports p and R squared, unlike a value claim."""
    r = verdicts(tmp_path, rel("in", "b", rng=[0.9, 1.0]))["in"]
    assert r["p_value"] is not None and r["r_squared"] is not None
    assert r["claimed_range"] == [0.9, 1.0] and r["claimed"] is None


def test_range_claims_do_not_stop_point_claims_beside_them(tmp_path):
    v = verdicts(tmp_path, rel("range", "b", rng=[0.9, 1.0]) + rel("point_ok", "noise", claimed=0.0)
                 + rel("point_bad", "b", claimed=0.0))
    assert v["range"]["verdict"] == AGREES
    assert v["point_ok"]["verdict"] == AGREES
    assert v["point_bad"]["verdict"] == REFUTED


def test_neither_claimed_nor_range_is_unverifiable(tmp_path):
    v = verdicts(tmp_path, rel("bare", "b"))
    assert v["bare"]["verdict"] == UNVERIFIABLE


def test_a_malformed_range_is_unverifiable(tmp_path):
    v = verdicts(tmp_path, '[[relationships]]\nid = "bad"\nx = "d.a"\ny = "d.b"\nkind = "correlation"\nmethod = "spearman"\n'
                           'resampling_unit = "unit"\nclaimed_range = [0.9]\n')
    assert v["bad"]["verdict"] == UNVERIFIABLE
