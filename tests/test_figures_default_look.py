"""The default look of a claim figure is a fixed point: with no figstyle, `claimcheck.figures` draws what it drew before styles existed.

The strongest statement available is a stored figure. `fixtures/figstyle/golden/MANIFEST.json` holds the sha256 of every figure below, of the PNG bytes
and of the decoded pixels, drawn by `claimcheck/figures.py` as it stood BEFORE the figstyle hook (the manifest records that file's sha256). These tests
draw the same inputs again and compare. One case per branch of `draw` (points coloured by unit with a legend and y = x; many points in one colour;
few points in one colour; no points), and a three-claim project through `build`, whose `index.json` is compared too.

PNG bytes depend on matplotlib, FreeType, Pillow and zlib, so the comparison runs only where all four match the manifest and is skipped, with the reason,
anywhere else. The tests that hold in every environment are in `test_figstyle.py`: an empty style, a missing style and no style give the same bytes in
one session, and `matplotlib.style.context` is never entered when no style is set.

To pin a new default look on purpose (never to make a failing test pass), draw it from the code you mean to keep:

    CLAIMCHECK_UPDATE_GOLDEN=1 pytest tests/test_figures_default_look.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import zlib
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from _figproject import BASE, pairs, write_project  # noqa: E402
from claimcheck import figures  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MANIFEST = HERE / "fixtures" / "figstyle" / "golden" / "MANIFEST.json"
UPDATE = os.environ.get("CLAIMCHECK_UPDATE_GOLDEN") == "1"

FILES = {"d": ["data/d.csv"]}


def cases() -> dict[str, tuple[dict, list[tuple[float, float]], list[str] | None]]:
    """name -> (claim, pairs, cluster labels): one per branch of `figures.draw`."""
    return {
        "units": ({**BASE, "id": "G_units"}, pairs(60), [f"t{i % 5}" for i in range(60)]),
        "dense": ({**BASE, "id": "G_dense", "verdict": "refuted", "y_unit": "kJ/mol", "n_rows": 240,
                   "why": "claimed 0.5 lies outside the 95% CI"}, pairs(240), None),
        "sparse": ({**BASE, "id": "G_sparse", "verdict": "unverifiable", "n_rows": 40, "ci_lo": -0.3, "ci_hi": 0.9,
                    "why": "the interval is wider than max_ci_width"}, pairs(40), None),
        "nopoints": ({**BASE, "id": "G_nopoints", "verdict": "unverifiable", "estimate": None, "n_rows": None, "ci_lo": None, "ci_hi": None,
                      "why": "the data has no column b"}, [], None),
    }


def environment() -> dict[str, str]:
    """What the bytes of a PNG depend on besides claimcheck."""
    import matplotlib
    from matplotlib import ft2font

    return {"matplotlib": matplotlib.__version__, "freetype": str(getattr(ft2font, "__freetype_version__", "unknown")),
            "pillow": importlib.metadata.version("pillow"), "zlib": zlib.ZLIB_VERSION}


def digests(png: Path) -> dict[str, str]:
    """sha256 of the file's bytes and of its decoded pixels (shape included)."""
    import matplotlib.image

    pixels = matplotlib.image.imread(png)
    return {"png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
            "pixels_sha256": hashlib.sha256(repr(pixels.shape).encode() + pixels.tobytes()).hexdigest()}


def drawn_now(tmp_path: Path) -> dict[str, dict]:
    """Every figure of this file, drawn with the code under test and no style, as digests."""
    out: dict[str, dict] = {"draw": {}, "build": {}}
    for name, (claim, points, clusters) in cases().items():
        png = tmp_path / f"{name}.png"
        figures.draw(claim, points, clusters, "d.a", "d.b", FILES, png)
        again = tmp_path / f"{name}_again.png"
        figures.draw(claim, points, clusters, "d.a", "d.b", FILES, again)
        assert digests(png) == digests(again), f"{name}: two draws of one input differ, so no comparison with a stored figure can mean anything"
        out["draw"][name] = digests(png)
    proj = tmp_path / "proj"
    write_project(proj)
    index = figures.build(proj, proj / "figures" / "claims")
    for entry in index:
        out["build"][entry["id"]] = digests(proj / entry["figure"])
    out["build"]["index.json"] = {"sha256": hashlib.sha256((proj / "figures" / "claims" / "index.json").read_bytes()).hexdigest()}
    return out


def test_without_a_style_the_figures_are_the_ones_drawn_before_styles_existed(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAIMCHECK_FIGSTYLE", raising=False)
    now = drawn_now(tmp_path)
    if UPDATE:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({
            "about": "sha256 of claim figures drawn with no style. Written by CLAIMCHECK_UPDATE_GOLDEN=1 pytest tests/test_figures_default_look.py",
            "figures_py_sha256": hashlib.sha256((REPO / "claimcheck" / "figures.py").read_bytes()).hexdigest(),
            "environment": environment(), **now}, indent=1) + "\n")
        return
    assert MANIFEST.is_file(), f"{MANIFEST} is missing: draw it from the code whose default look is to be kept (see this file's docstring)"
    golden = json.loads(MANIFEST.read_text())
    if golden["environment"] != environment():
        pytest.skip(f"the stored figures were drawn with {golden['environment']}, this environment is {environment()}; PNG bytes are not comparable")
    wrong = [f"{group}/{name}: {now[group][name]} != stored {golden[group].get(name)}"
             for group in ("draw", "build") for name in now[group] if now[group][name] != golden[group].get(name)]
    assert not wrong, "the default look changed:\n" + "\n".join(wrong)
    assert set(now["draw"]) == set(golden["draw"]) and set(now["build"]) == set(golden["build"])
