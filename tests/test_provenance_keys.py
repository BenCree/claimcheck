"""Every verdict file says which checker and which declaration produced it: first 16 hex characters of sha256 of relate.py and context.toml."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from claimcheck import relate


def project(tmp_path: Path) -> Path:
    (tmp_path / "d.csv").write_text("g,x,y\n" + "\n".join(f"{i},{i},{i * 2 + (i % 3)}" for i in range(12)) + "\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "t"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n'
        '[[variables]]\ndataset = "d"\nname = "x"\nunit = "1"\n\n[[variables]]\ndataset = "d"\nname = "y"\nunit = "1"\n\n'
        '[[relationships]]\nid = "r"\nx = "d.x"\ny = "d.y"\nkind = "correlation"\nmethod = "spearman"\nclaimed = 1.0\nresampling_unit = "g"\n')
    return tmp_path


def h(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def test_results_json_records_the_checker_and_the_declaration(tmp_path):
    proj = project(tmp_path)
    out = tmp_path / "results" / "claims.json"
    assert relate.main([str(proj), str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["relate_sha256"] == h(Path(relate.__file__)) and len(d["relate_sha256"]) == 16
    assert d["context_sha256"] == h(proj / "context.toml")


def test_the_context_digest_moves_when_the_declaration_changes(tmp_path):
    proj = project(tmp_path)
    before = relate.check(proj)["context_sha256"]
    (proj / "context.toml").write_text((proj / "context.toml").read_text() + "\n# edited\n")
    after = relate.check(proj)
    assert after["context_sha256"] != before and after["relate_sha256"] == relate.check(proj)["relate_sha256"]
