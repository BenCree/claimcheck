"""Value claims (kind = "value"): means, sds, variance shares and fractions adjudicated by relate, with a point `claimed` or a
`claimed_range`, bootstrapped over the declared resampling unit. Each verdict is tested in both directions on data whose truth is known."""

from __future__ import annotations

import random
from pathlib import Path

from claimcheck.relate import AGREES, REFUTED, UNVERIFIABLE, check

VARS = ('[[variables]]\ndataset = "d"\nname = "unit"\nunit = "1"\n\n'
        '[[variables]]\ndataset = "d"\nname = "total"\nunit = "kT"\n\n'
        '[[variables]]\ndataset = "d"\nname = "part"\nunit = "kT"\n\n'
        '[[variables]]\ndataset = "d"\nname = "ok"\nunit = "1"\n')


def project(tmp_path: Path, rels: str, n: int = 400, sd_part: float = 1.0, sd_rest: float = 2.0, p_ok: float = 0.9, seed: int = 3) -> Path:
    rng = random.Random(seed)
    lines = ["unit,total,part,ok"]
    for i in range(n):
        part = rng.gauss(5.0, sd_part)
        rest = rng.gauss(0.0, sd_rest)
        lines.append(f"{i},{part + rest},{part},{1 if rng.random() < p_ok else 0}")
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "v"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + VARS + "\n" + rels)
    return tmp_path


def rel(rid: str, method: str, x: str, y: str, claimed=None, rng=None) -> str:
    s = f'[[relationships]]\nid = "{rid}"\nx = "d.{x}"\ny = "d.{y}"\nkind = "value"\nmethod = "{method}"\nresampling_unit = "unit"\n'
    if claimed is not None:
        s += f"claimed = {claimed}\n"
    if rng is not None:
        s += "claimed_range = [" + ", ".join(("-inf" if k == 0 else "inf") if v is None else str(v) for k, v in enumerate(rng)) + "]\n"
    return s + "\n"


def verdicts(tmp_path: Path, rels: str, **kw) -> dict:
    return {r["id"]: r for r in check(project(tmp_path, rels, **kw))["relationships"]}


def test_point_claims_both_ways(tmp_path):
    v = verdicts(tmp_path, rel("m_ok", "mean", "part", "part", claimed=5.0) + rel("m_bad", "mean", "part", "part", claimed=6.0)
                 + rel("sd_ok", "sd", "part", "part", claimed=1.0) + rel("sd_bad", "sd", "part", "part", claimed=1.5))
    assert v["m_ok"]["verdict"] == AGREES and v["m_bad"]["verdict"] == REFUTED
    assert v["sd_ok"]["verdict"] == AGREES and v["sd_bad"]["verdict"] == REFUTED
    assert v["m_ok"]["p_value"] is None and v["m_ok"]["r_squared"] is None


def test_ranges_agree_refute_or_straddle(tmp_path):
    # var(part)/var(total) = 1 / (1 + 4) = 0.2
    v = verdicts(tmp_path, rel("in", "var_share", "total", "part", rng=[0.1, 0.3]) + rel("out", "var_share", "total", "part", rng=[0.5, 1.0])
                 + rel("edge", "var_share", "total", "part", rng=[0.2, 1.0]))
    assert v["in"]["verdict"] == AGREES
    assert v["out"]["verdict"] == REFUTED
    assert v["edge"]["verdict"] == UNVERIFIABLE


def test_one_sided_ranges_and_fractions(tmp_path):
    v = verdicts(tmp_path, rel("at_least", "fraction", "ok", "ok", rng=[0.8, None]) + rel("at_most", "fraction", "ok", "ok", rng=[None, 0.5]))
    assert v["at_least"]["verdict"] == AGREES
    assert v["at_most"]["verdict"] == REFUTED


def test_a_value_claim_with_no_claim_is_unverifiable(tmp_path):
    v = verdicts(tmp_path, rel("none", "mean", "part", "part"))
    assert v["none"]["verdict"] == UNVERIFIABLE


def test_kind_and_method_must_match(tmp_path):
    rels = ('[[relationships]]\nid = "mismatch"\nx = "d.part"\ny = "d.part"\nkind = "correlation"\nmethod = "mean"\nclaimed = 5.0\n'
            'resampling_unit = "unit"\n')
    v = verdicts(tmp_path, rels)
    assert v["mismatch"]["verdict"] == UNVERIFIABLE
