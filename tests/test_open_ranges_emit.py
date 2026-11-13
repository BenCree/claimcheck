"""An open-sided `claimed_range` must survive every emitted format (2026-11-12).

`claimed_range = [1.0, inf]` is a legal claim. `croissant.py` carried the bound into `json.dumps`, which writes the bare token
`Infinity` for it. RFC 8259 has no such token. Python's own `json` and `jq` read it back, so nothing here noticed; `orjson` refuses
it, and `orjson` is what `mlcroissant` reads with, so the package validated in some readers and raised JSONDecodeError in the one
whose job is to get the data out. Found by projects/z-ml/experiments/hq_decoy_testset reading its own output back.

The tests parse every emitted JSON document STRICTLY. A test that only used `json.loads` with its defaults passed throughout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claimcheck import emit
from claimcheck.relate import main as relate_main


def _strict(text: str):
    """RFC 8259: `Infinity`, `-Infinity` and `NaN` are not JSON. The standard library accepts them unless told otherwise."""
    def refuse(token: str):
        raise ValueError(f"{token} is not JSON (RFC 8259)")
    return json.loads(text, parse_constant=refuse)


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("open_range")
    rows = ["unit,a,b"] + [f"u{i},{i},{2 * i + (i % 3)}" for i in range(30)]
    (root / "d.csv").write_text("\n".join(rows) + "\n")
    variables = "".join(f'[[variables]]\ndataset = "d"\nname = "{n}"\nunit = "1"\n\n' for n in ("a", "b"))
    (root / "context.toml").write_text(
        '[project]\nname = "open"\n\n[[datasets]]\nid = "d"\norigin = "computed"\nfiles = ["d.csv"]\n\n' + variables
        + '[[relationships]]\nid = "mean_at_least_1"\nx = "d.b"\ny = "d.b"\nkind = "value"\nmethod = "mean"\n'
          'claimed_range = [1.0, inf]\nresampling_unit = "unit"\n\n'
        + '[[relationships]]\nid = "r_at_most"\nx = "d.a"\ny = "d.b"\nkind = "correlation"\nmethod = "spearman"\n'
          'claimed_range = [-inf, 0.5]\nresampling_unit = "unit"\n\n'
        + '[[relationships]]\nid = "closed"\nx = "d.a"\ny = "d.b"\nkind = "correlation"\nmethod = "spearman"\n'
          'claimed_range = [0.9, 1.0]\nresampling_unit = "unit"\n\n'
        + '[belief.rests_on]\nb = "qb"\n\n[[belief.nodes]]\nid = "qa"\nkind = "quantity"\nlabel = "a"\n\n'
          '[[belief.nodes]]\nid = "qb"\nkind = "quantity"\nlabel = "b"\n\n'
          '[[belief.claims]]\nsubject = "qa"\npredicate = "approximates"\nobject = "qb"\ntested_by = "mean_at_least_1"\n')
    assert relate_main([str(root), str(root / "results" / "claims.json")]) == 0
    from claimcheck import croissant
    assert croissant.main([str(root), str(root / "results" / "croissant.json")]) == 0      # the Turtle emitters read this file
    return root


def test_the_fixture_does_contain_an_infinite_bound(project):
    """Two-sided: the verdict file the emitters read carries the open sides, so an emitter that passes below did something."""
    rows = {r["id"]: r for r in json.loads((project / "results" / "claims.json").read_text())["relationships"]}
    assert rows["mean_at_least_1"]["claimed_range"] == [1.0, float("inf")] and rows["mean_at_least_1"]["verdict"] == "agrees"
    assert rows["r_at_most"]["claimed_range"] == [float("-inf"), 0.5] and rows["r_at_most"]["verdict"] == "refuted"
    with pytest.raises(ValueError, match="Infinity"):
        _strict(json.dumps({"claimedRange": [1.0, float("inf")]}))          # the strict parser does fail on the defect


def test_croissant_is_strict_json_and_an_open_side_is_null(project):
    from claimcheck import croissant
    text = croissant.render(project)
    doc = _strict(text)
    assert "Infinity" not in text and "NaN" not in text
    claims = {(c["x"], c["method"]): c for c in doc[f"{croissant.NS}claims"]}
    assert claims[("d.b", "mean")]["claimedRange"] == [1.0, None]            # open above
    by_range = [c["claimedRange"] for c in doc[f"{croissant.NS}claims"]]
    assert [None, 0.5] in by_range and [0.9, 1.0] in by_range                # open below; a closed range is untouched
    # the file `main` writes, not only the string `render` returns
    out = project / "results" / "croissant.json"
    assert croissant.main([str(project), str(out)]) == 0
    written = _strict(out.read_text())                                       # contentUrl differs by output folder; the claims do not
    assert written[f"{croissant.NS}claims"] == doc[f"{croissant.NS}claims"]


def test_orjson_reads_the_croissant_file(project):
    """The reader that refused it. Skipped only where orjson is absent; the strict parse above runs everywhere."""
    orjson = pytest.importorskip("orjson")
    from claimcheck import croissant
    out = project / "results" / "croissant_orjson.json"
    assert croissant.main([str(project), str(out)]) == 0
    doc = orjson.loads(out.read_bytes())
    assert any(c["claimedRange"] == [1.0, None] for c in doc[f"{croissant.NS}claims"])
    with pytest.raises(orjson.JSONDecodeError):
        orjson.loads(json.dumps({"claimedRange": [1.0, float("inf")]}))      # and it does refuse the old output


def test_every_json_emitter_is_strict_json(project):
    seen = []
    for name, why in sorted(emit.available().items()):
        if why:
            continue
        try:
            text = emit.emit(name, project)
        except Exception as exc:                                             # an emitter that needs more than this project has
            seen.append((name, f"not emitted: {type(exc).__name__}"))
            continue
        if text.lstrip().startswith(("{", "[")):
            _strict(text)
            seen.append((name, "strict JSON"))
        else:
            seen.append((name, "not JSON"))
    print(seen)
    assert ("croissant", "strict JSON") in seen and ("belief", "strict JSON") in seen
    assert not emit.available()["rocrate"] is None or ("rocrate", "strict JSON") in seen      # where its library is installed


def test_turtle_emitters_spell_an_open_side_as_xsd_spells_it(project):
    """`graph.ttl` carries the bounds as xsd:double, whose lexical space has `INF` and `-INF` and not Python's `inf`. Checked on the
    emitted TEXT, because rdflib reads either spelling back as the same number. It was already correct on 2026-11-12 (rdflib's Turtle
    writer spells it `INF`); this pins it. The other Turtle emitters carry no bound and are parsed to show an open range does not
    break them."""
    rdflib = pytest.importorskip("rdflib")
    import re
    spellings = {}
    for name in ("graph", "dcat", "evi", "psdi"):
        if emit.available().get(name):
            continue
        text = emit.emit(name, project)
        g = rdflib.Graph().parse(data=text, format="turtle")
        assert len(g) > 0
        spellings[name] = sorted(set(re.findall(r'"([^"]*)"\^\^xsd:double', text)))
    print(spellings)
    assert "INF" in spellings["graph"] and "-INF" in spellings["graph"]
    assert not {"inf", "-inf", "nan", "Infinity", "-Infinity"} & {s for v in spellings.values() for s in v}
    bounds = [o.toPython() for _, p, o in rdflib.Graph().parse(data=emit.emit("graph", project), format="turtle")
              if str(p).endswith(("claimedLow", "claimedHigh"))]
    assert float("inf") in bounds and float("-inf") in bounds and 0.5 in bounds       # and they still read back as the numbers


def test_mlcroissant_loads_the_rows_of_a_project_with_an_open_range(project):
    """The reader the format is written for, reading: not "does it validate" but "do the rows come out"."""
    mlc = pytest.importorskip("mlcroissant")
    ds = mlc.Dataset(jsonld=str(project / "results" / "croissant.json"))
    records = list(ds.records(record_set=ds.metadata.record_sets[0].id))
    assert len(records) == 30


def test_read_as_json_ld_the_side_of_an_open_bound_is_still_known(project):
    """Croissant is JSON-LD. There an array is an unordered set and a null inside it is dropped, so `claimedRange` alone gives an
    RDF reader `[1.0, null]` and `[null, 1.0]` as the same single value. `claimedLow` and `claimedHigh` carry the side."""
    rdflib = pytest.importorskip("rdflib")
    from claimcheck import croissant
    doc = _strict(croissant.render(project))
    by_method_x = {(c["method"], c["x"], c["claimedRange"][0]): c for c in doc[f"{croissant.NS}claims"]}
    at_least = by_method_x[("mean", "d.b", 1.0)]
    assert at_least["claimedLow"] == 1.0 and at_least["claimedHigh"] is None
    at_most = by_method_x[("spearman", "d.a", None)]
    assert at_most["claimedLow"] is None and at_most["claimedHigh"] == 0.5
    closed = by_method_x[("spearman", "d.a", 0.9)]
    assert (closed["claimedLow"], closed["claimedHigh"]) == (0.9, 1.0)
    # the array alone, as RDF: the side is lost. This is the property of JSON-LD the two keys exist for.
    def objects(fragment: dict) -> dict:
        g = rdflib.Graph().parse(data=json.dumps({"@context": {"@vocab": "https://example.org/"}, "@id": "https://example.org/c", **fragment}),
                                 format="json-ld")
        return {str(p).rsplit("/", 1)[-1]: sorted(float(o) for _, q, o in g if q == p) for _, p, _ in g}
    assert objects({"claimedRange": [1.0, None]}) == objects({"claimedRange": [None, 1.0]}) == {"claimedRange": [1.0]}
    assert objects({k: at_least[k] for k in ("claimedLow", "claimedHigh")}) == {"claimedLow": [1.0]}
    assert objects({k: at_most[k] for k in ("claimedLow", "claimedHigh")}) == {"claimedHigh": [0.5]}
