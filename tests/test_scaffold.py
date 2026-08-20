"""Scaffolding a project, and the things it must refuse to make up.

The claim under test is not "it writes four files". It is that what it writes
RUNS — `snakemake -c1` on the generated Snakefile, a Croissant a consumer could
read, and a grade from Snakemake's own record afterwards — and that the parts
nobody can infer from a CSV are left visibly blank rather than filled in.

Both halves are two-sided. The blanks are shown to make the checker say
`unverifiable`, and then filled in and shown to stop, because a blank that
changed no verdict would be decoration.
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest

from _snakemake import snakemake_cmd, why_not

from claimcheck import grading, relate, scaffold
from claimcheck.datasets import DatasetError

REPO = Path(__file__).resolve().parents[1]

CSV = ("key,dG_exp,n_atoms,label,never_filled\n"
       "L01,-8.5,32,alpha,\n"
       "L02,-7.1,28,beta,\n"
       "L03,-9.2,41,gamma,\n"
       "L04,-6.4,25,delta,\n")

OTHER = ("key,rmsd\n"
         "L01,1.2\n"
         "L02,2.4\n"
         "L03,0.9\n"
         "L04,3.1\n")


@pytest.fixture
def csv(tmp_path):
    p = tmp_path / "measurements.csv"
    p.write_text(CSV)
    return p


@pytest.fixture
def built(tmp_path, csv):
    """A scaffolded project, and the summary the call returned."""
    out = tmp_path / "study"
    return out, scaffold.scaffold(out, csv)


# ---------------------------------------------------------------------------
# What it writes
# ---------------------------------------------------------------------------

def test_it_writes_a_project_that_parses(built):
    out, summary = built
    for name in ("context.toml", "Snakefile", "README.md",
                 "data/measurements.csv"):
        assert (out / name).is_file(), name
    ctx = tomllib.loads((out / "context.toml").read_text())
    assert ctx["project"]["name"] == "study"
    assert [d["id"] for d in ctx["datasets"]] == ["measurements"]
    assert {v["name"] for v in ctx["variables"]} == {
        "key", "dG_exp", "n_atoms", "label", "never_filled"}
    assert summary["rows"] == 4 and summary["columns"] == 5


def test_the_column_types_are_read_off_the_data(built):
    """Through `croissant._data_type`, so a scaffolded project and a published
    one cannot disagree about what a column is. The empty column stays Text:
    guessing Float for a column with no values is the confident wrong answer."""
    _, summary = built
    assert summary["datasets"][0]["types"] == {
        "key": "sc:Text", "dG_exp": "sc:Float", "n_atoms": "sc:Integer",
        "label": "sc:Text", "never_filled": "sc:Text"}


def test_a_column_that_is_only_a_header_is_called_out(built):
    """`never_filled` has four rows and no values. The recorded consequence of
    not saying so: `darby_rowan32` names an empty `butina_cluster` as its
    resampling unit in two places downstream, and a reader selecting it gets 32
    NaNs rather than a refusal."""
    out, _ = built
    text = (out / "context.toml").read_text()
    assert "EMPTY: 0 of the 4 rows carry a value" in text


def test_the_data_is_pinned_by_digest_and_the_pin_is_live(built):
    """PASS side: the project loads. FAIL side: replace the CSV behind its back
    and the load stops. Scaffolding `manual` rather than `computed` is what buys
    this — it is the origin that requires source, retrieved and sha256."""
    out, _ = built
    ctx = tomllib.loads((out / "context.toml").read_text())
    spec = ctx["datasets"][0]
    assert spec["origin"] == "manual" and len(spec["sha256"]) == 64
    assert relate.check(out)["datasets"]["measurements"]["rows"] == 4

    (out / "data/measurements.csv").write_text(CSV.replace("-8.5", "-99.9"))
    with pytest.raises(DatasetError, match="not the file that was checked in"):
        relate.check(out)


# ---------------------------------------------------------------------------
# The two things it refuses to invent
# ---------------------------------------------------------------------------

def test_no_unit_is_declared_and_the_blank_changes_the_verdict(built):
    """THE RULE, and its consequence, in one test.

    Left side: not one variable carries a `unit`, and a relationship over two of
    them comes back `unverifiable`. Right side: declare the units and the same
    relationship reaches a real verdict. A blank that changed nothing would be a
    comment pretending to be a safeguard.
    """
    out, _ = built
    ctx = tomllib.loads((out / "context.toml").read_text())
    assert all("unit" not in v for v in ctx["variables"]), (
        "the scaffolder declared a unit it could not know")

    claim = ('\n[[relationships]]\nx = "measurements.dG_exp"\n'
             'y = "measurements.n_atoms"\nkind = "correlation"\n'
             'method = "pearson"\nclaimed = 0.0\n')
    ctx_file = out / "context.toml"
    ctx_file.write_text(ctx_file.read_text() + claim)

    r = relate.check(out)["relationships"][0]
    assert r["verdict"] == "unverifiable"
    assert "no unit declared" in r["why"]

    ctx_file.write_text(ctx_file.read_text().replace(
        'name = "dG_exp"', 'name = "dG_exp"\nunit = "kcal/mol"').replace(
        'name = "n_atoms"', 'name = "n_atoms"\nunit = "1"'))
    r = relate.check(out)["relationships"][0]
    assert r["verdict"] in ("agrees", "refuted"), r["why"]


def test_no_claim_is_invented_and_the_template_names_real_columns(built):
    """A `[[relationships]]` block needs `claimed` — the number somebody
    asserted — and a CSV never says what anyone claimed. So the generated file
    carries none, and the commented template points at columns that exist."""
    out, summary = built
    ctx = tomllib.loads((out / "context.toml").read_text())
    assert ctx.get("relationships", []) == []
    assert summary["relationships"] == 0
    text = (out / "context.toml").read_text()
    assert '# x = "measurements.dG_exp"' in text
    assert '# claimed = 0.0' in text and "\nclaimed" not in text


# ---------------------------------------------------------------------------
# The refusals, each measured against the version that would have been accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad,text,match", [
    ("header_only", "a,b\n", "header and no data rows"),
    ("duplicate", "a,a\n1,2\n", "repeats the column name"),
    ("blank_name", "a,,b\n1,2,3\n", "blank column name"),
    ("semicolons", "a;b;c\n1;2;3\n", "not comma-separated"),
    ("no_columns", "\n\n", "no columns"),
])
def test_it_refuses_a_csv_that_would_scaffold_a_lie(tmp_path, bad, text, match):
    """Each of these produces a project that either fails on the first run or,
    worse, succeeds while describing something the data does not contain."""
    src = tmp_path / f"{bad}.csv"
    src.write_text(text)
    with pytest.raises(scaffold.ScaffoldError, match=match):
        scaffold.scaffold(tmp_path / bad, src)
    assert not (tmp_path / bad).exists(), (
        "a refusal left a half-written project behind, which somebody will run")


def test_the_same_data_with_the_defect_removed_is_accepted(tmp_path):
    """The other side of every case above: the refusals are about the specific
    defect and not about being cautious."""
    src = tmp_path / "fine.csv"
    src.write_text("a,b\n1,2\n")
    assert scaffold.scaffold(tmp_path / "fine", src)["rows"] == 1


def test_it_will_not_write_into_a_directory_that_has_things_in_it(tmp_path, csv):
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "important.txt").write_text("do not lose me\n")
    with pytest.raises(scaffold.ScaffoldError, match="not empty"):
        scaffold.scaffold(out, csv)
    assert (out / "important.txt").read_text() == "do not lose me\n"
    scaffold.scaffold(out, csv, force=True)
    assert (out / "context.toml").is_file()
    assert (out / "important.txt").is_file(), "force is not permission to delete"


def test_a_missing_file_says_so_rather_than_scaffolding_an_empty_project(tmp_path):
    with pytest.raises(scaffold.ScaffoldError, match="no data file"):
        scaffold.scaffold(tmp_path / "x", tmp_path / "absent.csv")


def test_two_files_with_the_same_name_are_refused(tmp_path):
    """Both would be copied to `data/x.csv` and one would silently win."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for d in ("a", "b"):
        (tmp_path / d / "x.csv").write_text("k,v\n1,2\n")
    with pytest.raises(scaffold.ScaffoldError, match="share a basename"):
        scaffold.scaffold(tmp_path / "out",
                          [tmp_path / "a/x.csv", tmp_path / "b/x.csv"])


def test_a_column_name_that_breaks_toml_quoting_is_caught_not_shipped(tmp_path):
    """This module writes TOML by hand, because `tomllib` only reads. The
    round-trip check is what stands between a quote in a header and a
    context.toml that parses into a column name the data does not have."""
    src = tmp_path / "nasty.csv"
    src.write_text('he said "hi",back\\slash,plain\n1,2,3\n')
    out = tmp_path / "nasty"
    scaffold.scaffold(out, src)
    ctx = tomllib.loads((out / "context.toml").read_text())
    assert {v["name"] for v in ctx["variables"]} == {
        'he said "hi"', "back\\slash", "plain"}
    assert '\\"hi\\"' in (out / "context.toml").read_text(), (
        "the quotes were not escaped, so this passed for another reason")


# ---------------------------------------------------------------------------
# More than one file, which is where a project starts needing a join
# ---------------------------------------------------------------------------

def test_two_csvs_become_two_datasets_and_a_join_suggestion(tmp_path, csv):
    other = tmp_path / "geometry.csv"
    other.write_text(OTHER)
    out = tmp_path / "pair"
    summary = scaffold.scaffold(out, [csv, other])
    ctx = tomllib.loads((out / "context.toml").read_text())
    assert [d["id"] for d in ctx["datasets"]] == ["measurements", "geometry"]
    assert len(summary["datasets"]) == 2
    text = (out / "context.toml").read_text()
    # `key` is the only column in both files, so it is the only candidate — and
    # it is offered commented out, because a shared name is not a promise that
    # it identifies a row.
    assert '# join = "key"' in text


# ---------------------------------------------------------------------------
# The claim that matters: the thing it writes actually works
# ---------------------------------------------------------------------------

def test_the_generated_project_runs_under_snakemake_and_can_then_be_graded(built):
    """END TO END, and the acceptance test for the whole module.

    A bare CSV in, `snakemake -c1` out, then a grade read back from Snakemake's
    own store — which is also the only place both new modules meet.
    """
    sm = snakemake_cmd()
    if sm is None:
        pytest.skip(why_not())
    out, _ = built
    r = subprocess.run([*sm, "-c1"], cwd=str(out), capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]

    claims = json.loads((out / "results/claims.json").read_text())
    assert claims["relationships"] == [], "the scaffolder invented a claim"
    assert claims["datasets"]["measurements"]["rows"] == 4

    croissant = json.loads((out / "results/croissant.json").read_text())
    assert len(croissant["recordSet"][0]["field"]) == 5

    graded = grading.grade_all(out)
    assert graded["results/claims.json"].grade == "EXECUTED"
    assert graded["results/claims.json"].rule == "check_claims"
    assert graded["results/croissant.json"].quotable
    # The data was put there by a person, so no run describes it and it grades
    # NONE — with the pinned digest named as the evidence that does exist.
    data = graded["data/measurements.csv"]
    assert (data.grade, data.quotable) == ("NONE", False)
    assert "manual dataset" in data.why


def test_the_generated_croissant_declares_no_unit_it_was_not_given(built):
    """The blank has to survive all the way to the published description, or it
    is a blank in the source file and a guess in the artifact."""
    out, _ = built
    from claimcheck.croissant import build
    from claimcheck.namespace import NS
    doc = build(out)
    fields = [f for rs in doc["recordSet"] for f in rs["field"]]
    assert fields, "nothing was described"
    assert all(f"{NS}unit" not in f for f in fields)
    assert {f["dataType"] for f in fields} == {"sc:Text", "sc:Float", "sc:Integer"}
