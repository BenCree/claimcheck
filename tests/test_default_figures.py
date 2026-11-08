"""Figures by default: the shared `check_claims` rule runs relate and then claimcheck.figures, so every project gets figures/claims/<id>.png and
index.json without writing a rule. The figure step must never fail the verdicts, and must not be a declared output of the shared rule."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from claimcheck import figures

REPO = Path(__file__).resolve().parents[1]
TOY = REPO / "claimcheck-toy"
SMK = REPO / "claimcheck" / "workflow" / "claims.smk"


def test_relate_then_figures_lists_every_claim(tmp_path):
    pytest.importorskip("matplotlib")
    proj = tmp_path / "toy"
    shutil.copytree(TOY, proj, ignore=shutil.ignore_patterns("figures", "results"))
    env = {"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"}
    # the two commands of the rule's shell, in order
    subprocess.run([sys.executable, "-m", "claimcheck.relate", ".", "results/claims.json"], cwd=proj, env=env, check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "claimcheck.figures", ".", "figures/claims"], cwd=proj, env=env, check=True, capture_output=True)
    ids = [r["id"] for r in json.loads((proj / "results" / "claims.json").read_text())["relationships"]]
    index = json.loads((proj / "figures" / "claims" / "index.json").read_text())
    assert ids and [r["id"] for r in index] == ids
    for i in ids:
        assert (proj / "figures" / "claims" / f"{i}.png").stat().st_size > 5000


def test_without_matplotlib_figures_warns_and_exits_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "matplotlib", None)         # makes `import matplotlib` raise ImportError
    assert figures.main([str(tmp_path), str(tmp_path / "figures" / "claims")]) == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "matplotlib" in err and "verdicts are unaffected" in err
    assert not (tmp_path / "figures").exists()


def test_the_shared_rule_runs_figures_but_does_not_declare_the_directory():
    text = SMK.read_text()
    rule = text[text.index("rule check_claims:"):text.index("rule belief:")]
    assert "claimcheck.relate" in rule and "claimcheck.figures . figures/claims" in rule
    assert '"results/claims.json",' in rule
    # a declared output would collide with a project's own `claim_figures` rule (AmbiguousRuleException)
    outputs = rule[rule.index("output:"):rule.index("shell:")]
    assert "figures" not in outputs.replace("# ", "#").split("#")[0]
    assert "||" in rule                                          # a failing figure step prints a warning and cannot fail the verdicts
