"""Deletion detection, both sides of every claim it makes.

The check exists because every other check in this package walks the
filesystem. `grading.targets()` enumerates what is there, so a file that has
been deleted is not in its list and nothing says a word — and in the projects
this serves, `results/` and `data/` are gitignored, so git will not say a word
either. The only record of what SHOULD be there is Snakemake's own, and these
tests hold `vanished()` to reading it rather than inventing one.

Two-sided throughout, and the second side is the one that matters: a deletion
detector that reports nothing on a healthy tree has proved nothing until it has
been watched reporting something on a broken one.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from claimcheck.grading import recorded_outputs, vanished


def _store(root: Path, key: str, rec: dict) -> Path:
    """File a record the way Snakemake files it: urlsafe base64 of the path."""
    d = root / ".snakemake" / "metadata"
    d.mkdir(parents=True, exist_ok=True)
    f = d / base64.urlsafe_b64encode(key.encode()).decode()
    f.write_text(json.dumps(rec))
    return f


REC = {"rule": "make_thing", "starttime": 1.0, "endtime": 2.0, "code": "python x.py"}


def test_present_output_is_not_reported(tmp_path):
    """KNOWN-GOOD. The file is there; nothing may be claimed missing."""
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "a.csv").write_text("x\n")
    _store(tmp_path, "results/a.csv", REC)
    assert recorded_outputs(tmp_path) == ["results/a.csv"]
    assert vanished(tmp_path) == []


def test_deleted_output_is_reported(tmp_path):
    """KNOWN-BAD. Snakemake recorded making it; it is gone."""
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "a.csv").write_text("x\n")
    _store(tmp_path, "results/a.csv", REC)
    (tmp_path / "results" / "a.csv").unlink()

    gone = vanished(tmp_path)
    assert [g["path"] for g in gone] == ["results/a.csv"]
    assert gone[0]["rule"] == "make_thing"
    # The row must carry the way back, not just the bad news.
    assert gone[0]["regenerate"] == "snakemake results/a.csv"


def test_the_recipe_field_is_not_called_a_hash(tmp_path):
    """Snakemake's `code` is the recipe TEXT. The first version of this called
    it `code_hash` and its first real value was `{ENV} {PY} -`, which is a
    mislabel of exactly the kind this package exists to refuse."""
    (tmp_path / "results").mkdir()
    _store(tmp_path, "results/a.csv", REC)
    g = vanished(tmp_path)[0]
    assert "code_hash" not in g
    assert g["recipe"] == "python x.py"


def test_no_store_is_not_a_clean_bill_of_health(tmp_path):
    """A tree Snakemake has never run in has nothing to lose, and reporting
    zero there would be a pass earned by absence of evidence. `vanished` returns
    empty, and the CLI turns that into a refusal rather than an OK — asserted
    here so the two cannot drift apart."""
    assert recorded_outputs(tmp_path) == []
    assert vanished(tmp_path) == []

    from claimcheck.grading import main
    assert main([str(tmp_path), "--vanished"]) == 1


def test_absolute_and_relative_keys_both_resolve(tmp_path):
    """Snakemake keys on whatever string the rule declared, and both forms
    occur in the wild -- `datasets/claimcheck/darby_rowan32` has one of each.
    An absolute key that is gone must still be reported."""
    (tmp_path / "results").mkdir()
    abs_key = str((tmp_path / "results" / "b.csv").resolve())
    _store(tmp_path, abs_key, REC)
    assert vanished(tmp_path)[0]["path"] == abs_key


def test_a_non_record_in_the_store_is_skipped_not_guessed(tmp_path):
    """Something in `.snakemake/metadata/` that is not base64 must not become a
    reported deletion. A false alarm is how a check stops being read -- the
    failure this package's own history calls `check_crashes.sh`."""
    d = tmp_path / ".snakemake" / "metadata"
    d.mkdir(parents=True)
    (d / "not-a-record!!").write_text("{}")
    assert recorded_outputs(tmp_path) == []
    assert vanished(tmp_path) == []


def test_it_writes_nothing(tmp_path):
    """The package rule is that nothing here writes a run record. This reads
    Snakemake's; it must leave no trace of its own."""
    (tmp_path / "results").mkdir()
    _store(tmp_path, "results/a.csv", REC)
    before = {p for p in tmp_path.rglob("*")}
    vanished(tmp_path)
    recorded_outputs(tmp_path)
    assert {p for p in tmp_path.rglob("*")} == before
