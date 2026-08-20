"""Turn a bare CSV into a working project — context, workflow and all.

    python -m claimcheck.scaffold <new-dir> data.csv [more.csv ...]

Everything else in this package assumes a `context.toml` already exists. Writing
one by hand is the whole adoption barrier for anybody outside the project that
grew it: the format is small, but "small" and "obvious from nothing" are
different things, and the three real projects it is modelled on were each
written by someone who already knew the answer.

## What it writes

    <dir>/context.toml    a dataset block per file, a variable block per column
    <dir>/data/<file>.csv the data, copied in, so the project is self-contained
    <dir>/Snakefile       check_claims, croissant, and the two behind the extra
    <dir>/README.md       the two commands, and the two things left to do

`snakemake -c1` works on the result immediately, and
`python -m claimcheck.grading .` then grades what it produced.

## The two things it will not do

**Guess a unit.** A numeric column arrives with no `unit` key and a comment
saying so. `claimcheck.relate` returns `unverifiable` for any relationship
touching an undeclared column, so an unfilled unit is loud rather than assumed —
and that is the point. Inferring that a column called `energy` means kcal/mol
would manufacture exactly the fact this package exists to notice is missing. The
comment is an instruction rather than a commented-out `unit = ""`, because a
placeholder that can be uncommented into a wrong answer is worse than a blank.

**Invent a claim.** A `[[relationships]]` block needs `claimed`: the number
somebody asserted. Nothing in a CSV records what anyone claimed about it, so the
generated file carries a commented template naming real columns from your data
and nothing more. Filling `claimed = 0.0` in would look like a null hypothesis
and would in fact be a number this program made up.

## What it infers, because the data does say

Column types, through `claimcheck.croissant._data_type` — the same function that
labels a field for a Croissant consumer, so a scaffolded project and a published
one cannot disagree about what a column is. Row and value counts. The digest of
every file it copies.

## Where the shape comes from

`datasets/claimcheck/{darby_rowan32,md_vs_affinity,md10ns_vs_affinity}` in the
al_mace repository, which are three projects that work. Two departures from
them, each deliberate:

* `origin = "manual"`, not `"computed"`. Those three compute their data with a
  `prepare` rule from a source elsewhere in that repository. A CSV handed to
  this program was put there by a person, which is what `manual` means, and it
  is the origin that requires `source`, `retrieved` and `sha256` — all three of
  which are known here and are written in, so `claimcheck.datasets` will refuse
  the project the moment the file is silently replaced.
* `rule all` asks only for the two stdlib-only outputs. `graph` and `dcat` need
  the `graph` extra, and a first run that fails on an optional dependency
  teaches the reader that the tool is broken.
"""

from __future__ import annotations

import csv
import datetime
import shutil
import sys
import tomllib
from pathlib import Path

from claimcheck.croissant import _data_type
from claimcheck.datasets import sha256

#: Types `_data_type` returns for a column whose values are all numbers. Only
#: these get the "declare a unit" comment — a unit on a label column is noise,
#: and noise is what stops the real instruction being read.
_NUMERIC = ("sc:Integer", "sc:Float")


class ScaffoldError(ValueError):
    """Refused before anything was written. A half-written project is worse."""


def _toml_str(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _slug(s: str) -> str:
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(s).lower())
    return out.strip("_") or "dataset"


def _read(path: Path) -> tuple[list[str], int, dict[str, int]]:
    """Header, row count, and non-empty values per column — or a refusal.

    Every refusal here is about a file that would scaffold into a project which
    then fails, or worse, succeeds while describing something else.
    """
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ScaffoldError(f"{path.name} is empty — there is nothing to "
                                f"describe.") from None
        rows = [r for r in reader if any(c.strip() for c in r)]

    if not header:
        raise ScaffoldError(f"{path.name} has no columns; it may not be a CSV.")
    # `csv.DictReader` with its default dialect is what `claimcheck.datasets`
    # reads with, so a semicolon- or tab-separated file would scaffold happily
    # and then load as one column named after the whole header line. Refused
    # here rather than discovered on the first `snakemake` run.
    if len(header) == 1 and any(c in header[0] for c in ("\t", ";", "|")):
        raise ScaffoldError(
            f"{path.name} parses as ONE column named {header[0][:60]!r}, so it "
            f"is not comma-separated. `claimcheck.datasets` reads with "
            f"`csv.DictReader`'s default dialect and would see the same single "
            f"column. Convert it to CSV first.")
    blank = [i for i, c in enumerate(header) if not c.strip()]
    if blank:
        raise ScaffoldError(
            f"{path.name} has a blank column name at position "
            f"{', '.join(str(i + 1) for i in blank)}. A variable with no name "
            f"cannot be referred to in a claim; name it or drop the column.")
    dupes = sorted({c for c in header if header.count(c) > 1})
    if dupes:
        raise ScaffoldError(
            f"{path.name} repeats the column name(s) {', '.join(dupes)}. "
            f"`csv.DictReader` keeps only the last of each, so a project "
            f"scaffolded from this would describe columns nobody would be "
            f"reading. Rename them.")
    # DENOMINATOR FIRST. A header with no rows scaffolds a whole project that
    # then carries claims about an empty table, and every count in it is 0 over
    # a confident exit 0.
    if not rows:
        raise ScaffoldError(
            f"{path.name} has a header and no data rows, so there is nothing "
            f"to describe. A project scaffolded from it would carry variables "
            f"about an empty table.")

    filled = {c: 0 for c in header}
    for r in rows:
        for c, v in zip(header, r, strict=False):
            if str(v).strip():
                filled[c] += 1
    return header, len(rows), filled


def _dataset_id(path: Path, taken: set[str]) -> str:
    base = _slug(path.stem)
    out, n = base, 2
    while out in taken:
        out, n = f"{base}_{n}", n + 1
    return out


def _variables(ds_id: str, rel: str, header, n_rows, filled, types) -> list[str]:
    """One `[[variables]]` block per column, with the unit left undeclared."""
    lines: list[str] = []
    for col in header:
        t = types[col]
        lines += ["[[variables]]",
                  f"dataset = {_toml_str(ds_id)}",
                  f"name = {_toml_str(col)}",
                  "description = " + _toml_str(
                      f"{t}, inferred from {filled[col]} of {n_rows} non-empty "
                      f"value(s) in {rel}. Replace this with what the column "
                      f"MEANS.")]
        if filled[col] == 0:
            # THE COLUMN THAT IS A HEADER AND NOTHING ELSE. Worth its own line
            # because it is invisible to every other check until far too late:
            # `darby_rowan32` names an empty `butina_cluster` as its resampling
            # unit in two places downstream, and a reader selecting it in pandas
            # gets 32 NaNs and a correlation of NaN rather than a refusal.
            lines += [
                f"# EMPTY: 0 of the {n_rows} rows carry a value, so this is a "
                f"header and not",
                "# data. Naming it as a `resampling_unit`, or in a claim, gets "
                "back an",
                "# `unverifiable` verdict that says exactly that.",
            ]
        elif t in _NUMERIC:
            lines += [
                "# NO UNIT DECLARED. claimcheck does not guess one: add a "
                "`unit` line in",
                "# UDUNITS-2 syntax (kcal/mol, kJ/mol, nm, s, K, … or \"1\" for "
                "a genuinely",
                "# dimensionless quantity). Until then every relationship "
                "touching this",
                "# column comes back `unverifiable`, which is not a pass.",
            ]
        else:
            lines += [
                "# Inferred as text, so no unit is offered. If it is a quantity "
                "rather",
                "# than a label, declare `unit` in UDUNITS-2 syntax.",
            ]
        lines.append("")
    return lines


def _relationship_template(datasets: list[dict]) -> list[str]:
    """A commented block naming real columns, and the argument for leaving it so."""
    numeric = [(d["id"], c) for d in datasets for c in d["header"]
               if d["types"][c] in _NUMERIC]
    join = ""
    if len(datasets) > 1:
        shared = set(datasets[0]["header"])
        for d in datasets[1:]:
            shared &= set(d["header"])
        keys = sorted(shared)
        join = (f'# join = {_toml_str(keys[0])}'
                f'   # a column present in every file; it must identify a row,'
                if keys else
                '# join = "?"   # NO COLUMN IS COMMON TO ALL THE FILES, so'
                ' nothing lines their rows up yet')
    x = f"{numeric[0][0]}.{numeric[0][1]}" if numeric else "dataset.column_x"
    y = (f"{numeric[1][0]}.{numeric[1][1]}" if len(numeric) > 1
         else "dataset.column_y")

    out = ["# " + "-" * 74,
           "# RELATIONSHIPS — none written, and that is not an oversight.",
           "# " + "-" * 74,
           "#",
           "# A relationship needs `claimed`: the number SOMEBODY ASSERTED. A CSV",
           "# records what was measured and never what anyone claimed about it, so",
           "# filling that in would be this program inventing the one thing the",
           "# package exists to check. `claimed = 0.0` would read as a null",
           "# hypothesis and would in fact be a made-up number.",
           "#",
           "# The template below names real columns from your data. Uncomment it,",
           "# put your own number in `claimed`, and declare the units of both",
           "# columns above.",
           "#",
           "# [[relationships]]",
           f"# x = {_toml_str(x)}",
           f"# y = {_toml_str(y)}"]
    if join:
        out.append(join)
    out += [
        '# kind = "correlation"',
        '# method = "pearson"        # or "spearman" for monotonic but curved',
        "# claimed = 0.0             # <- YOUR number. There is no sensible default.",
        "#",
        "# THE RESAMPLING UNIT IS DECLARED, NEVER ASSUMED. Leave it out and the",
        "# interval treats every row as independent. If your rows share a series,",
        "# a subject, a seed or a configuration, name that column here:",
        "# resampling_unit = \"cluster_column\"",
        "#",
        "# [relationships.evidence]",
        '# source = "where the claim comes from — a paper, a premise, a README"',
        '# locator = "context.toml#relationship-1"',
        '# extracted_by = "human"',
        ""]
    if not numeric:
        out.insert(3, "# NOTE: no column in your data was inferred numeric, so the")
        out.insert(4, "#       template below names placeholders rather than columns.")
        out.insert(5, "#")
    dotted = sorted({c for d in datasets for c in d["header"] if "." in c})
    if dotted:
        out.insert(3, f"# NOTE: {', '.join(dotted[:4])} contain a dot, and `x`/`y` "
                      f"are split on the")
        out.insert(4, "#       FIRST dot into dataset and column, so those columns "
                      "cannot be")
        out.insert(5, "#       referenced. Rename them if you want to make a claim "
                      "about one.")
        out.insert(6, "#")
    return out


def build_context(name: str, datasets: list[dict]) -> str:
    """The `context.toml`, derived from the actual columns rather than a template."""
    total_rows = sum(d["n_rows"] for d in datasets)
    total_cols = sum(len(d["header"]) for d in datasets)
    n_num = sum(1 for d in datasets for c in d["header"]
                if d["types"][c] in _NUMERIC)
    L = [f"# Written by `python -m claimcheck.scaffold` on "
         f"{datetime.date.today().isoformat()} from "
         f"{', '.join(d['source_name'] for d in datasets)}.",
         "#",
         f"# {len(datasets)} dataset(s), {total_rows} row(s), {total_cols} "
         f"column(s), {n_num} numeric.",
         "#",
         "# Everything below that is a FACT was read from the data: the row and",
         "# value counts, the inferred types, the digests. Everything that is a",
         "# JUDGEMENT — what a column means, what unit it is in, what anybody",
         "# claims about it — was left for you, because a scaffolder that",
         "# guessed those would be manufacturing the evidence.",
         "",
         "[project]",
         f"name = {_toml_str(name)}",
         "# One sentence saying what this dataset IS. It becomes the",
         "# `description` of the published Croissant.",
         f"title = {_toml_str(name)}",
         "# Optional and omitted rather than faked: license, version,",
         "# date_published, citation, keywords, and `context` — the free text",
         "# that says what a reader has to know before believing any of it.",
         ""]

    for d in datasets:
        L += ["# " + "-" * 74,
              f"# DATASET {d['id']} — put here by a person, which is what `manual` means.",
              "# No rule made it, so Snakemake can record nothing about where it came",
              "# from; `source`, `retrieved` and `sha256` are the only provenance there",
              "# is, and the digest is checked on EVERY load. Replace the file without",
              "# saying so and the run stops with both digests printed.",
              "# " + "-" * 74,
              "[[datasets]]",
              f"id = {_toml_str(d['id'])}",
              'origin = "manual"',
              f"files = [{_toml_str(d['rel'])}]",
              "title = " + _toml_str(
                  f"{d['source_name']}: {d['n_rows']} rows x "
                  f"{len(d['header'])} columns."),
              "source = " + _toml_str(
                  f"Copied by `claimcheck.scaffold` from {d['source_path']}. "
                  f"Replace this with where the data actually came from."),
              f"retrieved = {_toml_str(datetime.date.today().isoformat())}",
              f"sha256 = {_toml_str(d['sha256'])}",
              ""]

    L += ["# " + "-" * 74,
          "# VARIABLES — one per column, with the unit LEFT UNDECLARED.",
          "# " + "-" * 74,
          ""]
    for d in datasets:
        L += _variables(d["id"], d["rel"], d["header"], d["n_rows"],
                        d["filled"], d["types"])
    L += _relationship_template(datasets)
    return "\n".join(L)


SNAKEFILE = '''# Layer 1: Snakemake, and nothing is added to it.
#
#   snakemake -n                        # always first: read the job list
#   snakemake -c1
#   python -m claimcheck.grading .      # what may now be quoted, and why not
#
# Snakemake already records, per output, the input digests as sha256, the code,
# the shell command, the conda environment, a software-stack hash and the times.
# That IS the run record, so this project keeps none of its own — `grading`
# reads Snakemake's store rather than a second copy of it.
#
# EVERY DATASET IS AN INPUT to the rules that describe it, even though they read
# the paths out of context.toml. Consuming a file is what puts its digest into
# that store.
#
# `grading` is deliberately not a rule here: it would read the store that its
# own run is still writing, and grade its own output as an unfinished job.

import sys

PY = sys.executable

DATA = [
%%DATA%%
]


rule all:
    input:
        "results/claims.json",
        "results/croissant.json",


rule check_claims:
    input:
        data=DATA,
        context="context.toml",
    output:
        "results/claims.json",
    params:
        py=PY,
    shell:
        "{params.py} -m claimcheck.relate . {output}"


rule croissant:
    input:
        data=DATA,
        claims="results/claims.json",
        context="context.toml",
    output:
        "results/croissant.json",
    params:
        py=PY,
    shell:
        "{params.py} -m claimcheck.croissant . {output}"


# THE TWO BELOW NEED THE `graph` EXTRA, so they are NOT in `rule all`: a first
# run that fails on an optional dependency teaches the reader that the tool is
# broken.
#
#     pip install 'claimcheck[graph]'
#     snakemake -c1 results/graph.ttl results/dcat.ttl

rule graph:
    input:
        croissant="results/croissant.json",
        claims="results/claims.json",
        context="context.toml",
    output:
        "results/graph.ttl",
    params:
        py=PY,
    shell:
        "{params.py} -m claimcheck.graph . {output}"


rule dcat:
    input:
        data=DATA,
        claims="results/claims.json",
        context="context.toml",
    output:
        "results/dcat.ttl",
    params:
        py=PY,
    shell:
        "{params.py} -m claimcheck.emit dcat . {output}"


# WHY THIS RULE IS COMMENTED OUT RATHER THAN ABSENT. `jobs` publishes why each
# run was launched and what would have counted as success -- the half of a
# cluster submission that Snakemake's own store has no slot for. It needs a
# `[jobs]` table naming a JSONL log, and it REFUSES a project that has none,
# because "nothing was submitted" and "nobody wrote it down" are different
# answers. A new project has neither yet, so uncommenting this is the second
# step, not the first.
#
#     [jobs]
#     records = "jobs.jsonl"
#
# rule jobs:
#     input:
#         records="jobs.jsonl",
#         context="context.toml",
#     output:
#         "results/jobs.ttl",
#     params:
#         py=PY,
#     shell:
#         "{params.py} -m claimcheck.emit jobs . {output}"
'''

README = '''# %%NAME%%

Scaffolded by `python -m claimcheck.scaffold` from %%SOURCES%% —
%%ROWS%% row(s), %%COLS%% column(s), %%NUMERIC%% numeric.

```bash
snakemake -n                      # always first: read the job list
snakemake -c1                     # check the claims, describe the data
python -m claimcheck.grading .    # which of those files may be quoted, and why not
```

## What is true in here already

Read from the data: the row and value counts, the inferred column types, and the
sha256 of every file under `data/`. That digest is checked on every load, so
replacing a file without saying so stops the run instead of changing an answer.

## What is missing, on purpose

1. **The units.** Every numeric column is declared with no `unit`, and a comment
   saying so. `claimcheck` does not guess: a relationship touching an undeclared
   column comes back `unverifiable`, which is **not a pass**. Declaring a unit is
   a judgement about what was measured, and a scaffolder that made it would be
   inventing the fact you are here to record.

2. **The claims.** `context.toml` ends with a commented `[[relationships]]`
   template naming real columns from your data. A relationship needs `claimed` —
   the number somebody asserted — and nothing in a CSV says what anyone claimed.
   Uncomment it, put your own number in, and run `snakemake -c1` again.

Until you do (2) the workflow runs and checks nothing, which is honest: zero
claims checked, and it says so.

## What each file is for

    context.toml   what the columns mean, and what is claimed about them
    Snakefile      the recipe. Snakemake records the digests, code and times
    data/          the data, copied in so the project is self-contained
    results/       written by the workflow: claims.json, croissant.json
'''


def scaffold(directory: Path | str, data, *, name: str | None = None,
             force: bool = False) -> dict:
    """Write a working claimcheck project around one or more CSVs.

    Nothing is written until every file has been read and accepted, so a
    refusal leaves no half-project behind for somebody to run by mistake.
    """
    directory = Path(directory)
    sources = ([Path(data)] if isinstance(data, str | Path)
               else [Path(d) for d in data])
    if not sources:
        raise ScaffoldError("no data file given; a project needs at least one")
    for s in sources:
        if not s.is_file():
            raise ScaffoldError(f"no data file at {s}")
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ScaffoldError(
            f"two of the given files share a basename ({', '.join(sorted(names))}), "
            f"so copying both into data/ would leave one of them. Rename one.")
    if directory.exists() and any(directory.iterdir()) and not force:
        raise ScaffoldError(f"{directory} is not empty (pass force=True to write "
                            f"into it anyway)")

    name = name or _slug(directory.resolve().name)
    parsed: list[dict] = []
    taken: set[str] = set()
    for s in sources:
        header, n_rows, filled = _read(s)
        did = _dataset_id(s, taken)
        taken.add(did)
        parsed.append({"id": did, "source_path": str(s.resolve()),
                       "source_name": s.name, "rel": f"data/{s.name}",
                       "header": header, "n_rows": n_rows, "filled": filled})

    # --- everything accepted; now write
    (directory / "data").mkdir(parents=True, exist_ok=True)
    for s, d in zip(sources, parsed, strict=True):
        shutil.copy(s, directory / d["rel"])
        d["sha256"] = sha256(directory / d["rel"])
        # TYPED FROM THE COPY, through the same function that types a field for
        # a Croissant consumer. Two answers to "what is this column" is how a
        # published description comes to disagree with the project that made it.
        d["types"] = {c: _data_type(directory, [d["rel"]], c) for c in d["header"]}

    text = build_context(name, parsed)
    # ROUND-TRIPPED BEFORE IT IS TRUSTED. This module writes TOML by hand —
    # `tomllib` only reads — so a column name containing a quote or a backslash
    # is one escaping bug away from a file that no longer parses, or worse,
    # parses into a different column name than the data has.
    try:
        back = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ScaffoldError(
            f"the generated context.toml does not parse ({e}). This is a bug in "
            f"claimcheck.scaffold, not in your data — please report the column "
            f"names that caused it.") from e
    written = {(v["dataset"], v["name"]) for v in back.get("variables", [])}
    expected = {(d["id"], c) for d in parsed for c in d["header"]}
    if written != expected:
        raise ScaffoldError(
            f"the generated context.toml describes different columns from the "
            f"data: {sorted(expected ^ written)}. This is a bug in "
            f"claimcheck.scaffold.")

    (directory / "context.toml").write_text(text)
    (directory / "Snakefile").write_text(SNAKEFILE.replace(
        "%%DATA%%", "\n".join(f'    "{d["rel"]}",' for d in parsed)))
    n_num = sum(1 for d in parsed for c in d["header"]
                if d["types"][c] in _NUMERIC)
    (directory / "README.md").write_text(
        README.replace("%%NAME%%", name)
              .replace("%%SOURCES%%", ", ".join(d["source_name"] for d in parsed))
              .replace("%%ROWS%%", str(sum(d["n_rows"] for d in parsed)))
              .replace("%%COLS%%", str(sum(len(d["header"]) for d in parsed)))
              .replace("%%NUMERIC%%", str(n_num)))

    return {"name": name, "directory": str(directory),
            "datasets": [{"id": d["id"], "file": d["rel"], "rows": d["n_rows"],
                          "columns": len(d["header"]), "sha256": d["sha256"],
                          "types": d["types"]} for d in parsed],
            "rows": sum(d["n_rows"] for d in parsed),
            "columns": sum(len(d["header"]) for d in parsed),
            "numeric_columns": n_num,
            "undeclared_units": [f"{d['id']}.{c}" for d in parsed
                                 for c in d["header"]
                                 if d["types"][c] in _NUMERIC],
            "relationships": 0}


def main(argv: list[str] | None = None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    force = "--force" in a
    a = [x for x in a if x != "--force"]
    if len(a) < 2:
        print("usage: python -m claimcheck.scaffold <new-dir> <data.csv> "
              "[more.csv ...] [--force]", file=sys.stderr)
        return 2
    try:
        d = scaffold(a[0], a[1:], force=force)
    except ScaffoldError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"  {d['directory']}  —  {len(d['datasets'])} dataset(s), "
          f"{d['rows']} row(s), {d['columns']} column(s)")
    for ds in d["datasets"]:
        print(f"    {ds['file']:<32} {ds['rows']:>6} rows  "
              f"sha256:{ds['sha256'][:12]}…")
    if d["undeclared_units"]:
        print(f"  {len(d['undeclared_units'])} numeric column(s) with NO "
              f"declared unit: {', '.join(d['undeclared_units'][:6])}"
              f"{' …' if len(d['undeclared_units']) > 6 else ''}")
        print("  -> every relationship touching one returns `unverifiable` "
              "until you declare it.")
    print("  0 relationships: the template in context.toml is commented out, "
          "because\n     `claimed` is a number only you can supply.")
    print(f"\n  cd {d['directory']} && snakemake -n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
