"""Croissant is for READING the data, so these tests read it.

**The gap this file closes.** `test_croissant.py` asks `mlcroissant` to validate
the JSON-LD, and it passed for as long as the package has existed. On 2026-08-17
the emitted Croissant for the package's own example was measured to load **zero**
records:

    GenerationError ... In node "mace_energies_csv", file "data/mace_energies.csv"
    is either an invalid URL or an invalid path.

`contentUrl` was written relative to the PROJECT ROOT, and `mlcroissant` resolves
it relative to the folder holding the JSON-LD (`operations/download.py:48`,
`filepath = ctx.folder / url`). The Snakefile writes `results/croissant.json`, so
every consumer looked for `results/data/...`.

Validation could not catch it: the metadata was correct in every respect except
the one that mattered, and the failure lived in `ds.records()`, which nothing
called. **A format whose purpose is "how do I read this?" needs a test that
reads it.**

The second half is the types. Every field was emitted `sc:Text`, so a float
column arrived as `b'-8.546824'`. That is not a validation error either --
`sc:Text` is a legal dataType for any column -- it just moves every cast onto
the consumer and silently discards what the producer knew.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from _snakemake import snakemake_cmd, why_not

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "example"

mlc = pytest.importorskip("mlcroissant")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The example, built from scratch in a copy."""
    sm = snakemake_cmd()
    if sm is None:
        pytest.skip(why_not())
    work = tmp_path_factory.mktemp("croissant") / "example"
    shutil.copytree(EXAMPLE, work)
    shutil.rmtree(work / "results", ignore_errors=True)
    shutil.rmtree(work / ".snakemake", ignore_errors=True)
    import os
    r = subprocess.run([*sm, "-c1"], capture_output=True, text=True, cwd=str(work),
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stderr[-2000:]
    return work


# ---------------------------------------------------------------------------
# the one that would have caught it
# ---------------------------------------------------------------------------

def test_a_consumer_can_actually_load_the_rows(built):
    """Not "does it validate" -- does it READ. This is the whole format."""
    ds = mlc.Dataset(jsonld=str(built / "results" / "croissant.json"))
    rs = ds.metadata.record_sets[0]
    records = list(ds.records(record_set=rs.id))
    assert records, ("the Croissant validates but yields no rows -- check "
                     "contentUrl is relative to the JSON-LD, not the project root")
    import csv
    with (built / "data" / "mace_energies.csv").open(newline="") as fh:
        expected = sum(1 for _ in csv.DictReader(fh))
    assert len(records) == expected, f"{len(records)} loaded, {expected} in the CSV"


def test_the_content_url_is_relative_to_the_json_ld(built):
    """The precise defect, asserted as a property rather than as a load.

    Stated separately so a failure says WHICH of the two things broke: a load
    failure has many causes and this has one.
    """
    d = json.loads((built / "results" / "croissant.json").read_text())
    out_dir = built / "results"
    for fo in d["distribution"]:
        url = fo["contentUrl"]
        assert not Path(url).is_absolute(), f"{url} is absolute; it will not travel"
        assert (out_dir / url).resolve().is_file(), (
            f"contentUrl {url!r} does not resolve from {out_dir}, which is where "
            f"mlcroissant looks (operations/download.py: ctx.folder / url)")


# ---------------------------------------------------------------------------
# the types
# ---------------------------------------------------------------------------

def test_numeric_columns_are_not_shipped_as_text(built):
    """`sc:Text` for a float column is legal, validates, and throws away what
    the producer knew. The consumer then gets bytes."""
    d = json.loads((built / "results" / "croissant.json").read_text())
    types = {f["name"]: f["dataType"]
             for rs in d["recordSet"] for f in rs["field"]}
    assert types, "no fields emitted"
    assert types.get("e_int_kcal") == "sc:Float", types
    assert types.get("n_lig_atoms") == "sc:Integer", types
    assert all(t in ("sc:Boolean", "sc:Date", "sc:Float", "sc:Integer", "sc:Text")
               for t in types.values()), types


def test_the_loaded_values_are_numbers(built):
    """The consumer-visible consequence of the line above."""
    ds = mlc.Dataset(jsonld=str(built / "results" / "croissant.json"))
    rs = ds.metadata.record_sets[0]
    row = next(iter(ds.records(record_set=rs.id)))
    vals = {k.rsplit("/", 1)[-1]: v for k, v in row.items()}
    assert isinstance(vals["e_int_kcal"], float), (
        f"e_int_kcal came back as {type(vals['e_int_kcal']).__name__} "
        f"({vals['e_int_kcal']!r})")
    assert isinstance(vals["n_lig_atoms"], int), type(vals["n_lig_atoms"]).__name__


# ---------------------------------------------------------------------------
# the other side: inference must not be confident about nothing
# ---------------------------------------------------------------------------

def test_an_empty_column_stays_text(tmp_path):
    """The two-sided half. Guessing `sc:Float` for a column with no values is
    the confident-wrong answer, and this repository has one on record: a header
    that names `butina_cluster` over 32 empty rows, described downstream as the
    resampling unit the data supports."""
    from claimcheck.croissant import _data_type
    (tmp_path / "d.csv").write_text("a,b,c\n1,,2.5\n2,,3.5\n")
    assert _data_type(tmp_path, ["d.csv"], "a") == "sc:Integer"
    assert _data_type(tmp_path, ["d.csv"], "c") == "sc:Float"
    assert _data_type(tmp_path, ["d.csv"], "b") == "sc:Text", (
        "an empty column was given a numeric type, which claims knowledge of a "
        "column that holds none")


def test_a_non_finite_value_is_not_a_float(tmp_path):
    """`float('inf')` parses. A loader casting to Float then hands a consumer an
    infinity that no arithmetic downstream expects, so it stays Text."""
    from claimcheck.croissant import _data_type
    (tmp_path / "d.csv").write_text("a\n1.0\ninf\n")
    assert _data_type(tmp_path, ["d.csv"], "a") == "sc:Text"


def test_a_text_column_is_still_text(tmp_path):
    from claimcheck.croissant import _data_type
    (tmp_path / "d.csv").write_text("name\nx6738a\nx6832a\n")
    assert _data_type(tmp_path, ["d.csv"], "name") == "sc:Text"


def test_the_old_behaviour_is_detectably_wrong(built):
    """The two-sided half for the path fix.

    `build()` with no `out_dir` reproduces exactly what the package emitted
    before 2026-08-17: `contentUrl` relative to the project root. Asserting
    here that this DOES NOT resolve from `results/` is what makes the test
    above a regression test rather than a restatement of current behaviour.
    """
    from claimcheck.croissant import build
    old = build(built)                                   # no out_dir
    new = build(built, out_dir=built / "results")
    old_url = old["distribution"][0]["contentUrl"]
    new_url = new["distribution"][0]["contentUrl"]
    assert old_url != new_url, "the fix changed nothing"
    assert not (built / "results" / old_url).exists(), (
        "the pre-fix contentUrl resolves after all, so this test proves nothing")
    assert (built / "results" / new_url).is_file()


def _two_files_one_basename(root: Path) -> None:
    """A project declaring two DIFFERENT files that share a basename.

    This is not contrived: a project with one table per arm writes
    `arm_a/edges_plot.csv` and `arm_b/edges_plot.csv`, and one with eight arms
    was what produced the 69 conflated subjects the corpus merge counted.
    """
    for arm, lo in (("arm_a", 1), ("arm_b", 100)):
        (root / arm).mkdir(parents=True, exist_ok=True)
        rows = "\n".join(f"{lo + i},{(lo + i) * 2}" for i in range(4))
        (root / arm / "edges.csv").write_text(f"x,y\n{rows}\n")
    (root / "context.toml").write_text(
        '[project]\nname = "basename_collision"\n'
        'title = "Two files with one basename."\nlicense = "CC0-1.0"\n'
        'context = "A fixture for the 2026-11-22 file_id change."\n'
        '\n[[datasets]]\nid = "a"\norigin = "computed"\nlevel = "toy"\n'
        'files = ["arm_a/edges.csv"]\n'
        '\n[[datasets]]\nid = "b"\norigin = "computed"\nlevel = "toy"\n'
        'files = ["arm_b/edges.csv"]\n'
        '\n[[variables]]\ndataset = "a"\nname = "x"\nunit = "count"\n'
        '\n[[variables]]\ndataset = "a"\nname = "y"\nunit = "count"\n'
        '\n[[variables]]\ndataset = "b"\nname = "x"\nunit = "count"\n'
        '\n[[variables]]\ndataset = "b"\nname = "y"\nunit = "count"\n')


def test_two_files_with_one_basename_are_two_nodes(tmp_path):
    """THE DEFECT OF 2026-11-22: eight digests on one node.

    `@id` was `slug(Path(rel).name)`, so every declared file whose basename
    agreed collapsed onto one FileObject and the LAST `sha256` written won.
    rdflib and `mlcroissant` both merge by id and raise nothing, so the
    package validated while describing the wrong bytes. The two files here
    have different contents, so their digests differ and a collapse is
    visible.
    """
    from claimcheck.croissant import build
    _two_files_one_basename(tmp_path)
    d = build(tmp_path, out_dir=tmp_path / "results")
    ids = [fo["@id"] for fo in d["distribution"]]
    assert len(ids) == len(set(ids)) == 2, (
        f"two declared files collapsed onto {len(set(ids))} node(s): {ids}")
    digests = {fo["@id"]: fo["sha256"] for fo in d["distribution"]}
    assert len(set(digests.values())) == 2, (
        f"one digest for two different files: {digests}")
    # The record sets must point at DIFFERENT file objects, or the fields of
    # one arm read the other arm's rows.
    sources = [f["source"]["fileObject"]["@id"]
               for rs in d["recordSet"] for f in rs["field"]]
    assert set(sources) == set(ids), (
        f"fields reference {set(sources)}, not the two file objects {set(ids)}")


def test_a_consumer_reads_both_same_basename_files(tmp_path):
    """And it must READ, not merely validate: the records of each arm come back
    with that arm's own numbers.

    Distinct `@id`s are not the property wanted. The property wanted is that a
    consumer asking for arm B's record set gets arm B's rows, which is what a
    collapsed FileObject silently breaks.
    """
    from claimcheck.croissant import build
    _two_files_one_basename(tmp_path)
    out = tmp_path / "results"
    out.mkdir(exist_ok=True)
    (out / "croissant.json").write_text(
        json.dumps(build(tmp_path, out_dir=out), indent=2))
    ds = mlc.Dataset(jsonld=str(out / "croissant.json"))
    seen = {}
    for rs in ds.metadata.record_sets:
        rows = list(ds.records(record_set=rs.id))
        assert rows, f"record set {rs.id} yielded no rows"
        seen[rs.id] = {int(r[f"{rs.id}/x"]) for r in rows}
    assert len(seen) == 2, seen
    lows, highs = sorted(seen.values(), key=min)
    assert lows == {1, 2, 3, 4}, lows
    assert highs == {100, 101, 102, 103}, highs


def test_the_basename_id_would_have_collapsed_them():
    """The known-bad side. Without it the two tests above restate current
    behaviour instead of pinning a repair: this asserts that the rule they
    replaced really does collide on this fixture.

    `file_id` is IMPORTED, not re-implemented. A test carrying its own copy of
    the thing under test passes while the shipped code is wrong.
    """
    from claimcheck.croissant import file_id
    declared = ["arm_a/edges.csv", "arm_b/edges.csv"]

    def old_id(rel: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in Path(rel).name)

    assert len({old_id(r) for r in declared}) == 1, (
        "the fixture does not reproduce the defect, so it guards nothing")
    assert len({file_id(r) for r in declared}) == 2


def test_two_paths_with_one_id_are_refused(tmp_path):
    """`file_id` is NOT injective, and the case it cannot name is refused.

    Every character that is not a letter or a digit becomes `_`, so
    `arm_a/edges.csv` and `arm/a_edges.csv` are two different files with one
    id. Narrowing the character class would change every id a second time, so
    `build` refuses instead and names the two paths. Without this test the fix
    above would still let one node carry two digests, which is the entire
    defect.
    """
    from claimcheck.croissant import build, file_id
    assert file_id("arm_a/edges.csv") == file_id("arm/a_edges.csv"), (
        "the fixture no longer collides, so this guards nothing")
    for rel in ("arm_a/edges.csv", "arm/a_edges.csv"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x,y\n1,2\n")
    (tmp_path / "context.toml").write_text(
        '[project]\nname = "id_collision"\ntitle = "Two paths, one id."\n'
        'license = "CC0-1.0"\ncontext = "A fixture for the refusal."\n'
        '\n[[datasets]]\nid = "a"\norigin = "computed"\nlevel = "toy"\n'
        'files = ["arm_a/edges.csv", "arm/a_edges.csv"]\n'
        '\n[[variables]]\ndataset = "a"\nname = "x"\nunit = "count"\n')
    with pytest.raises(ValueError, match="one Croissant @id"):
        build(tmp_path, out_dir=tmp_path / "results")


def test_a_datasets_title_reaches_the_record_set():
    """`[[datasets]].title` was declared on every dataset block in a 61-block
    project and read by nothing. It says what a row is, which is what a
    Croissant consumer shows beside the table."""
    from claimcheck.croissant import build
    d = build(REPO / "example")
    described = {rs["@id"]: rs.get("description", "") for rs in d["recordSet"]}
    assert any(v for v in described.values()), described
