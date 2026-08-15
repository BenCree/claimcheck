"""Adding a dataset, and the two ways a multi-dataset project fails quietly.

Adding one is three lines. Relating two is four more. Neither may be able to
produce a confident wrong answer, and the two ways they can are the subject
here — **each seen to fire**, because a safety check nobody has watched fail is
a safety check nobody should rely on.

1. **A manually imported file that changed.** No rule made it, so Snakemake has
   no record of it at all — not its digest, not when it arrived, not whether
   this is the copy anybody looked at. If it is silently replaced, nothing else
   in the system can tell.
2. **A join on a key with duplicates.** Ten rows against ten sharing one key is
   a hundred pairs, none of them measured together, and a correlation over them
   is a fact about the join.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from claimcheck.datasets import DatasetError, join
from claimcheck.datasets import load as load_dataset
from claimcheck.relate import check

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"


def _csv(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The manual import: the one thing nothing else can detect
# ---------------------------------------------------------------------------

def test_a_manual_import_must_say_where_it_came_from(tmp_path):
    """`source`, `retrieved` and `sha256`, each required for its own reason: a
    file with no stated origin cannot be re-fetched, "the current version" is
    not a version, and only the digest notices a replacement."""
    _csv(tmp_path, "d.csv", "a,b\n1,2\n")
    with pytest.raises(DatasetError, match="source, retrieved, sha256"):
        load_dataset(tmp_path, {"id": "d", "origin": "manual",
                                "files": ["d.csv"]})


def test_a_changed_manual_file_stops_the_run(tmp_path):
    """THE GUARD THAT MATTERS, watched failing.

    Someone drops in a newer export. Nothing else in the system can tell — no
    rule produced it, so Snakemake has no prior digest to compare. This does.
    """
    p = _csv(tmp_path, "d.csv", "a,b\n1,2\n")
    spec = {"id": "d", "origin": "manual", "files": ["d.csv"],
            "source": "a database", "retrieved": "2026-08-15",
            "sha256": _sha(p)}
    assert load_dataset(tmp_path, spec)["rows"]        # the good case first

    p.write_text("a,b\n1,3\n")                          # one digit changed
    with pytest.raises(DatasetError) as e:
        load_dataset(tmp_path, spec)
    msg = str(e.value)
    assert "not the file that was checked in" in msg
    assert "declared sha256" in msg and "actual" in msg, "print both digests"
    assert "update `sha256`" in msg, "say how to accept the new file on purpose"


def test_a_computed_dataset_needs_no_digest(tmp_path):
    """The other side. A rule made it, so Snakemake already holds the digest,
    the code and the environment — requiring a second copy here would be the
    duplication this repository exists to avoid."""
    _csv(tmp_path, "d.csv", "a,b\n1,2\n")
    ds = load_dataset(tmp_path, {"id": "d", "origin": "computed",
                                 "files": ["d.csv"]})
    assert ds["origin"] == "computed" and ds["rows"]


def test_an_undeclared_origin_is_refused_rather_than_defaulted(tmp_path):
    """There is no default on purpose. The two origins need different evidence,
    and any default would silently pick the weaker one."""
    _csv(tmp_path, "d.csv", "a,b\n1,2\n")
    for origin in (None, "", "downloaded", "COMPUTED"):
        spec = {"id": "d", "files": ["d.csv"]}
        if origin is not None:
            spec["origin"] = origin
        with pytest.raises(DatasetError, match="origin"):
            load_dataset(tmp_path, spec)


def test_the_example_pins_its_manual_dataset(tmp_path):
    """The shipped example must actually use the mechanism, or the mechanism is
    documentation. Copy it, corrupt the manual file, and the whole check
    stops — no verdicts are produced from bytes nobody meant to ship."""
    work = tmp_path / "example"
    shutil.copytree(EXAMPLE, work)
    shutil.rmtree(work / "results", ignore_errors=True)
    assert check(work)["relationships"], "the untouched copy must work"

    manual = work / "data" / "experimental_affinity.csv"
    manual.write_text(manual.read_text() + "SPURIOUS,9.9\n")
    with pytest.raises(DatasetError, match="not the file that was checked in"):
        check(work)


# ---------------------------------------------------------------------------
# The join: refused, not resolved
# ---------------------------------------------------------------------------

def _ds(did, rows):
    return {"id": did, "origin": "computed", "rows": rows, "spec": {},
            "digests": {}}


def test_a_clean_join_reports_its_coverage():
    left = _ds("l", [{"k": "a", "x": "1"}, {"k": "b", "x": "2"},
                     {"k": "c", "x": "3"}])
    right = _ds("r", [{"k": "a", "y": "9"}, {"k": "b", "y": "8"}])
    pairs, rep = join(left, right, "k")
    assert len(pairs) == 2 and rep["refused"] is None
    assert rep["n_matched"] == 2 and rep["n_left_unmatched"] == 1
    assert rep["coverage"] == pytest.approx(2 / 3, abs=1e-4)  # reported to 4dp


def test_duplicate_keys_are_refused_not_multiplied():
    """The silent disaster. Two rows sharing a key on each side is four pairs,
    and none of the four is a pair of measurements that were taken together."""
    left = _ds("l", [{"k": "a", "x": "1"}, {"k": "a", "x": "2"}])
    right = _ds("r", [{"k": "a", "y": "9"}, {"k": "a", "y": "8"}])
    pairs, rep = join(left, right, "k")
    assert pairs == []
    assert "repeated value" in rep["refused"]
    assert "multiply rows into pairs that were never measured" in rep["refused"]


def test_a_missing_join_column_is_named():
    left = _ds("l", [{"k": "a"}])
    right = _ds("r", [{"other": "a"}])
    pairs, rep = join(left, right, "k")
    assert pairs == [] and "has no column 'k'" in rep["refused"]


def test_a_cross_dataset_relationship_without_a_join_key_is_unverifiable(tmp_path):
    """Not guessed from shared column names. Two datasets that happen to share
    a column called `id` are not thereby joinable, and picking one would be the
    kind of convenience that produces a confident wrong number."""
    _csv(tmp_path, "a.csv", "k,x\n1,2\n2,3\n3,4\n")
    _csv(tmp_path, "b.csv", "k,y\n1,5\n2,6\n3,7\n")
    (tmp_path / "context.toml").write_text('''
[project]
name = "t"
[[datasets]]
id = "A"
origin = "computed"
files = ["a.csv"]
[[datasets]]
id = "B"
origin = "computed"
files = ["b.csv"]
[[variables]]
dataset = "A"
name = "x"
unit = "1"
[[variables]]
dataset = "B"
name = "y"
unit = "1"
[[relationships]]
x = "A.x"
y = "B.y"
kind = "correlation"
method = "pearson"
claimed = 0.5
''')
    got = check(tmp_path)["relationships"][0]
    assert got["verdict"] == "unverifiable"
    assert "no `join` key is declared" in got["why"]


def test_a_bare_column_name_is_ambiguous_with_two_datasets(tmp_path):
    """With one dataset a bare name is fine. With two it would be resolved by
    whichever happened to be declared first."""
    _csv(tmp_path, "a.csv", "x\n1\n")
    _csv(tmp_path, "b.csv", "x\n2\n")
    (tmp_path / "context.toml").write_text('''
[project]
name = "t"
[[datasets]]
id = "A"
origin = "computed"
files = ["a.csv"]
[[datasets]]
id = "B"
origin = "computed"
files = ["b.csv"]
[[relationships]]
x = "x"
y = "x"
kind = "correlation"
method = "pearson"
claimed = 0.5
''')
    with pytest.raises(DatasetError, match="does not say which dataset"):
        check(tmp_path)


# ---------------------------------------------------------------------------
# The shipped example, as the thing a reader copies
# ---------------------------------------------------------------------------

def test_the_example_joins_across_two_datasets_and_says_how_well():
    d = check(EXAMPLE)
    cross = [r for r in d["relationships"] if r["cross_dataset"]]
    assert cross, "the example must exercise a cross-dataset relationship"
    j = cross[0]["join"]
    assert j["key"] == "complex_name"
    assert j["n_matched"] == 637 and j["coverage"] == 1.0
    assert d["datasets"]["affinity"]["origin"] == "manual"
    assert d["datasets"]["mace"]["origin"] == "computed"
