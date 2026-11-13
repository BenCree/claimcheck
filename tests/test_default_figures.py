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


def test_the_screening_methods_each_get_a_figure(tmp_path):
    """Added 2026-11-12 with `bedroc` and `topn_recovery`: a new method must not break the figure step, which runs inside the shared
    check_claims rule. One claim of each, one roc_auc on the same table, and a bedroc claim with no alpha (unverifiable, no estimate)."""
    pytest.importorskip("matplotlib")
    import random

    from claimcheck import figures
    from claimcheck.relate import main as relate_main

    rng = random.Random(3)
    rows = [(f"c{i}", int(i < 9), rng.gauss(2.0 * (i < 9), 1.0)) for i in range(150)]
    cut = sorted((s for _, _, s in rows), reverse=True)[19]
    lines = ["compound,is_hit,score,in_top20"] + [f"{c},{h},{s:.6f},{int(s >= cut)}" for c, h, s in rows]
    (tmp_path / "d.csv").write_text("\n".join(lines) + "\n")
    variables = "".join(f'[[variables]]\ndataset = "d"\nname = "{n}"\nunit = "1"\n\n' for n in ("is_hit", "score", "in_top20"))
    head = '[[relationships]]\nx = "d.is_hit"\nresampling_unit = "compound"\n'
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "s"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + variables
        + head + 'id = "auc"\ny = "d.score"\nkind = "correlation"\nmethod = "roc_auc"\nclaimed_range = [0.5, 1.0]\n\n'
        + head + 'id = "bedroc20"\ny = "d.score"\nkind = "correlation"\nmethod = "bedroc"\nalpha = 20\nclaimed_range = [0.0, 1.0]\n\n'
        + head + 'id = "bedroc_no_alpha"\ny = "d.score"\nkind = "correlation"\nmethod = "bedroc"\nclaimed_range = [0.0, 1.0]\n\n'
        + head + 'id = "top20"\ny = "d.in_top20"\nkind = "value"\nmethod = "topn_recovery"\nclaimed_range = [0.0, 1.0]\n')
    assert relate_main([str(tmp_path), str(tmp_path / "results" / "claims.json")]) == 0
    verdicts = {r["id"]: r for r in json.loads((tmp_path / "results" / "claims.json").read_text())["relationships"]}
    assert verdicts["bedroc_no_alpha"]["verdict"] == "unverifiable" and verdicts["top20"]["n_shortlist"] == 20
    assert verdicts["bedroc20"]["alpha"] == 20.0 and verdicts["top20"]["n_actives"] == 9
    index = figures.build(tmp_path, tmp_path / "figures" / "claims")
    assert [r["id"] for r in index] == ["auc", "bedroc20", "bedroc_no_alpha", "top20"]
    for r in index:
        assert (tmp_path / "figures" / "claims" / f"{r['id']}.png").stat().st_size > 5000
    for cid in ("bedroc20", "top20"):
        assert figures.result_line(verdicts[cid]).strip()
        print(figures.result_line(verdicts[cid])[:400])
