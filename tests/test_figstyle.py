"""claimcheck.figures drawn in a shared matplotlib style: `[project] figstyle = "<path>"` in context.toml, or the environment variable CLAIMCHECK_FIGSTYLE.

What is pinned: the key wins over the variable; a relative key is read from the directory holding context.toml and a relative variable from the working
directory; an unusable path prints one warning and the figures are drawn as if no style were set, without falling back to the variable; the style is in force
while each figure is drawn and not after; with no style `matplotlib.style` is never entered; an empty style changes nothing; and a missing matplotlib still
warns, exits 0 and leaves the verdicts alone. The stored figures drawn before the hook existed are in `test_figures_default_look.py`.

What a style does NOT reach is also true and is not tested here: `figures.draw` passes the verdict colours, the unit colours, the sizes of the caption, verdict
word and legend, the figure size, `dpi` and the grid itself, so rcParams cannot change them (see the module docstring of claimcheck.figures).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")                         # before anything reads rcParams["backend"], which would resolve a GUI backend on a desktop

from _figproject import write_project  # noqa: E402
from claimcheck import figures  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TOY = REPO / "claimcheck-toy"
STAND_IN = HERE / "fixtures" / "figstyle" / "stand_in.mplstyle"
CYCLE = ["#ff00ff", "#00ffff", "#ffff00"]


@pytest.fixture(autouse=True)
def _the_host_has_no_style(monkeypatch):
    monkeypatch.delenv(figures.STYLE_ENV, raising=False)


def rc_seen() -> dict:
    """The rcParams that matter to a style, as they are in force at this moment."""
    return {"font.size": matplotlib.rcParams["font.size"], "font.family": list(matplotlib.rcParams["font.family"]),
            "colours": matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]}


def rc_now() -> dict:
    """All rcParams but the backend (`matplotlib.use` sets it and the style context deliberately does not restore it)."""
    return {k: matplotlib.rcParams[k] for k in matplotlib.rcParams if k != "backend"}


@pytest.fixture
def rc_in_draw(monkeypatch):
    """What rcParams were in force inside each call of figures.draw."""
    seen: list[dict] = []
    real_draw = figures.draw

    def spy(*args, **kwargs):
        seen.append(rc_seen())
        return real_draw(*args, **kwargs)

    monkeypatch.setattr(figures, "draw", spy)
    return seen


@pytest.fixture
def kept(monkeypatch):
    """The Figures that `draw` would close, kept open so a test can read what was drawn on them."""
    import matplotlib.pyplot as plt

    real_close, figs = plt.close, []
    monkeypatch.setattr(plt, "close", lambda fig=None: figs.append(fig))
    yield figs
    for fig in figs:
        real_close(fig)


def project(tmp_path: Path, name: str, *project_lines: str, n_claims: int = 3) -> Path:
    proj = tmp_path / name
    write_project(proj, project_lines, n_claims)
    return proj


def build(proj: Path) -> list[dict]:
    return figures.build(proj, proj / "figures" / "claims")


def hashes(proj: Path, index: list[dict]) -> dict[str, str]:
    return {r["id"]: hashlib.sha256((proj / r["figure"]).read_bytes()).hexdigest() for r in index}


def quoted(path: Path) -> str:
    """A TOML basic string for a path."""
    return json.dumps(str(path))


def style_file(path: Path, font_size: int) -> Path:
    path.write_text(f"font.size: {font_size}\n")
    return path


def warnings_in(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("WARNING claimcheck.figures")]


# --- the stand-in style, and what it does to a figure -------------------------------------------------------------------------------------------------

def test_the_stand_in_style_sets_what_the_tests_read_and_differs_from_what_is_in_force():
    params = matplotlib.rc_params_from_file(STAND_IN, use_default_template=False)
    assert params["font.size"] == 17 and list(params["font.family"]) == ["monospace"]
    assert params["axes.prop_cycle"].by_key()["color"] == CYCLE and params["axes.spines.top"] is False
    outside = rc_seen()
    assert outside["font.size"] != 17 and outside["colours"] != CYCLE and outside["font.family"] != ["monospace"]


def test_a_figstyle_key_is_in_force_while_each_figure_is_drawn_and_not_after(tmp_path, rc_in_draw):
    proj = project(tmp_path, "styled", 'figstyle = "stand_in.mplstyle"')
    shutil.copy(STAND_IN, proj / "stand_in.mplstyle")
    before = rc_now()
    index = build(proj)
    assert len(rc_in_draw) == len(index) == 3
    assert all(seen == {"font.size": 17, "font.family": ["monospace"], "colours": CYCLE} for seen in rc_in_draw)
    assert rc_now() == before, "the style leaked out of the drawing context"


def test_without_a_figstyle_the_figures_are_drawn_under_the_rcparams_already_in_force(tmp_path, rc_in_draw):
    outside = rc_seen()
    build(project(tmp_path, "plain"))
    assert len(rc_in_draw) == 3 and all(seen == outside for seen in rc_in_draw)


def test_a_style_reaches_the_text_and_axes_that_draw_leaves_to_rcparams(tmp_path, kept):
    plain = project(tmp_path, "plain", n_claims=1)
    build(plain)
    plain_ax = kept[0].axes[0]
    kept.clear()
    styled = project(tmp_path, "styled", 'figstyle = "stand_in.mplstyle"', n_claims=1)
    shutil.copy(STAND_IN, styled / "stand_in.mplstyle")
    build(styled)
    styled_ax = kept[0].axes[0]
    # the style's own values on the styled figure ...
    assert styled_ax.xaxis.label.get_fontsize() == 19 and styled_ax.yaxis.label.get_fontsize() == 19
    assert {t.get_fontsize() for t in styled_ax.get_xticklabels() if t.get_text()} == {15}
    assert styled_ax.xaxis.label.get_fontfamily() == ["monospace"]
    assert not styled_ax.spines["top"].get_visible() and not styled_ax.spines["right"].get_visible()
    # ... and not on the plain one
    assert plain_ax.xaxis.label.get_fontsize() != 19
    assert {t.get_fontsize() for t in plain_ax.get_xticklabels() if t.get_text()} != {15}
    assert plain_ax.xaxis.label.get_fontfamily() != ["monospace"]
    assert plain_ax.spines["top"].get_visible() and plain_ax.spines["right"].get_visible()


# --- nothing set: the code that ran before styles ------------------------------------------------------------------------------------------------------

def test_without_a_style_matplotlib_style_is_never_entered(tmp_path, monkeypatch):
    import matplotlib.style

    entered: list[object] = []
    real_context = matplotlib.style.context

    def counting_context(style, *args, **kwargs):
        entered.append(style)
        return real_context(style, *args, **kwargs)

    monkeypatch.setattr(matplotlib.style, "context", counting_context)
    build(project(tmp_path, "plain", n_claims=2))
    assert entered == []
    styled = project(tmp_path, "styled", 'figstyle = "s.mplstyle"', n_claims=2)
    style_file(styled / "s.mplstyle", 17)
    build(styled)
    assert entered, "the counter does not see a style being entered, so the empty list above proves nothing"


def test_an_empty_style_changes_nothing_and_a_real_one_changes_every_figure(tmp_path):
    plain = project(tmp_path, "plain")
    plain_index = build(plain)
    empty = project(tmp_path, "empty", 'figstyle = "empty.mplstyle"')
    (empty / "empty.mplstyle").write_text("# a style that sets nothing\n")
    empty_index = build(empty)
    real = project(tmp_path, "real", 'figstyle = "stand_in.mplstyle"')
    shutil.copy(STAND_IN, real / "stand_in.mplstyle")
    real_index = build(real)
    plain_hashes = hashes(plain, plain_index)
    assert hashes(empty, empty_index) == plain_hashes
    real_hashes = hashes(real, real_index)
    assert all(real_hashes[i] != plain_hashes[i] for i in plain_hashes)
    # index.json: the four keys it always had; a style adds the two that name it
    assert all(set(r) == {"id", "verdict", "n_points", "figure", "commentary"} for r in plain_index)
    assert all(set(r) == {"id", "verdict", "n_points", "figure", "commentary", "figstyle", "figstyle_sha256"} for r in empty_index + real_index)


def test_importing_figures_or_asking_for_no_style_does_not_import_matplotlib(tmp_path):
    script = tmp_path / "lazy.py"
    script.write_text("import sys\nfrom pathlib import Path\nfrom claimcheck import figures\n"
                      "assert 'matplotlib' not in sys.modules, 'the import'\n"
                      "assert figures.load_style(Path('.'), {'project': {'name': 'x'}}) == (None, None)\n"
                      "assert 'matplotlib' not in sys.modules, 'asking for no style'\n")
    r = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --- a style that cannot be used ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("what", ["missing", "directory", "undecodable"])
def test_a_style_that_cannot_be_used_warns_once_and_the_figures_are_drawn_without_one(tmp_path, capsys, rc_in_draw, what):
    outside = rc_seen()
    plain = project(tmp_path, "plain")
    plain_hashes = hashes(plain, build(plain))
    capsys.readouterr()
    proj = project(tmp_path, "proj", 'figstyle = "bad.mplstyle"')
    if what == "directory":
        (proj / "bad.mplstyle").mkdir()
    elif what == "undecodable":
        (proj / "bad.mplstyle").write_bytes(b"\xff\xfe\x00\x80 this is not text")
    rc_in_draw.clear()
    index = build(proj)
    one = warnings_in(capsys.readouterr().err)
    assert len(one) == 1, f"three claims, so one warning per figure would be three: {one}"
    assert "bad.mplstyle" in one[0] and "[project].figstyle" in one[0] and "verdicts are unaffected" in one[0]
    assert hashes(proj, index) == plain_hashes
    assert all("figstyle" not in r for r in index)
    assert len(rc_in_draw) == 3 and all(seen == outside for seen in rc_in_draw)


def test_a_key_that_is_not_a_path_is_reported_like_a_missing_file(tmp_path, capsys):
    proj = project(tmp_path, "proj", "figstyle = 3")
    index = build(proj)
    assert len(warnings_in(capsys.readouterr().err)) == 1 and all("figstyle" not in r for r in index)


def test_a_bad_line_in_the_style_is_reported_by_matplotlib_once_and_not_once_per_figure(tmp_path, caplog):
    path = tmp_path / "s.mplstyle"
    path.write_text("font.size: 17\nnot.a.key: 1\n")
    caplog.clear()
    matplotlib.rc_params_from_file(path, use_default_template=False)
    assert len([r for r in caplog.records if "not.a.key" in r.getMessage()]) == 1, "matplotlib no longer logs a bad key, so this test cannot see it"
    proj = project(tmp_path, "proj", f"figstyle = {quoted(path)}")
    caplog.clear()
    build(proj)
    assert len([r for r in caplog.records if "not.a.key" in r.getMessage()]) == 1


def test_a_style_cannot_change_the_backend(tmp_path):
    """`matplotlib.style` ignores the keys that are not about style, with a warning; the README says so, so this pins it."""
    proj = project(tmp_path, "proj", 'figstyle = "b.mplstyle"', n_claims=1)
    (proj / "b.mplstyle").write_text("backend: svg\nfont.size: 17\n")
    with pytest.warns(UserWarning, match="not related to style"):
        build(proj)
    assert matplotlib.get_backend().lower() == "agg"      # the style context does not restore the backend, so a style that set it would have left svg here


# --- which of the two wins, and where a relative path is read from -----------------------------------------------------------------------------------

def test_the_key_beats_the_variable_and_a_bad_key_does_not_fall_back_to_it(tmp_path, monkeypatch, rc_in_draw, capsys):
    outside = rc_seen()
    a, b = style_file(tmp_path / "a.mplstyle", 17), style_file(tmp_path / "b.mplstyle", 23)
    monkeypatch.setenv(figures.STYLE_ENV, str(a))

    def sizes(proj: Path) -> set:
        rc_in_draw.clear()
        build(proj)
        return {seen["font.size"] for seen in rc_in_draw}

    assert sizes(project(tmp_path, "both", f"figstyle = {quoted(b)}", n_claims=1)) == {23}
    assert sizes(project(tmp_path, "only_variable", n_claims=1)) == {17}
    assert sizes(project(tmp_path, "empty_key", 'figstyle = ""', n_claims=1)) == {17}, "an empty key is no key"
    capsys.readouterr()
    assert sizes(project(tmp_path, "bad_key", f"figstyle = {quoted(tmp_path / 'missing.mplstyle')}", n_claims=1)) == {outside["font.size"]}
    assert len(warnings_in(capsys.readouterr().err)) == 1
    monkeypatch.delenv(figures.STYLE_ENV)
    assert sizes(project(tmp_path, "neither", n_claims=1)) == {outside["font.size"]}


def test_a_relative_key_is_read_from_the_project_and_a_relative_variable_from_the_working_directory(tmp_path, monkeypatch, rc_in_draw):
    proj = project(tmp_path, "proj", 'figstyle = "styles/s.mplstyle"', n_claims=1)
    (proj / "styles").mkdir()
    style_file(proj / "styles" / "s.mplstyle", 17)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "styles").mkdir(parents=True)
    style_file(elsewhere / "styles" / "s.mplstyle", 31)           # a decoy: the key must not be read from the working directory
    monkeypatch.chdir(elsewhere)
    build(proj)
    assert {seen["font.size"] for seen in rc_in_draw} == {17}
    rc_in_draw.clear()
    style_file(elsewhere / "v.mplstyle", 23)
    monkeypatch.setenv(figures.STYLE_ENV, "v.mplstyle")
    build(project(tmp_path, "plain", n_claims=1))
    assert {seen["font.size"] for seen in rc_in_draw} == {23}


# --- the paths a rule takes: main, and the command in a subprocess ---------------------------------------------------------------------------------

def test_main_applies_the_variable_and_reports_the_style(tmp_path, monkeypatch, capsys):
    proj = project(tmp_path, "proj")
    monkeypatch.setenv(figures.STYLE_ENV, str(STAND_IN))
    assert figures.main([str(proj), str(proj / "figures" / "claims")]) == 0
    assert f"(style {STAND_IN})" in capsys.readouterr().out
    index = json.loads((proj / "figures" / "claims" / "index.json").read_text())
    assert len(index) == 3
    for r in index:
        assert r["figstyle"] == str(STAND_IN) and r["figstyle_sha256"] == hashlib.sha256(STAND_IN.read_bytes()).hexdigest()[:16]


def test_the_command_the_shared_rule_runs_sees_the_variable(tmp_path):
    proj = project(tmp_path, "proj", n_claims=1)
    env = {"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    command = [sys.executable, "-m", "claimcheck.figures", ".", "figures/claims"]
    plain = subprocess.run(command, cwd=proj, env=env, capture_output=True, text=True)
    assert plain.returncode == 0, plain.stderr
    plain_index = json.loads((proj / "figures" / "claims" / "index.json").read_text())
    assert all("figstyle" not in r for r in plain_index) and "(style" not in plain.stdout
    styled = subprocess.run([*command[:-1], "figures/styled"], cwd=proj, env={**env, figures.STYLE_ENV: str(STAND_IN)}, capture_output=True, text=True)
    assert styled.returncode == 0, styled.stderr
    styled_index = json.loads((proj / "figures" / "styled" / "index.json").read_text())
    assert all(r["figstyle"] == str(STAND_IN) for r in styled_index) and f"(style {STAND_IN})" in styled.stdout
    assert (proj / "figures" / "claims" / "P1_by_unit.png").read_bytes() != (proj / "figures" / "styled" / "P1_by_unit.png").read_bytes()


# --- matplotlib missing: the figure step warns and the verdicts stand ---------------------------------------------------------------------------------

def test_without_matplotlib_a_configured_style_changes_nothing_but_the_warning_you_already_had(tmp_path, monkeypatch, capsys):
    proj = project(tmp_path, "proj", 'figstyle = "nope.mplstyle"')
    monkeypatch.setenv(figures.STYLE_ENV, str(STAND_IN))
    monkeypatch.setitem(sys.modules, "matplotlib", None)          # makes `import matplotlib` raise ImportError
    assert figures.main([str(proj), str(proj / "figures" / "claims")]) == 0
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1 and "matplotlib is not importable" in err and "verdicts are unaffected" in err
    assert not (proj / "figures").exists()


def test_verdicts_are_written_and_left_alone_when_matplotlib_is_missing_and_a_style_is_set(tmp_path):
    if not (TOY / "context.toml").exists():
        pytest.skip("the toy project is not here")
    proj = tmp_path / "toy"
    shutil.copytree(TOY, proj, ignore=shutil.ignore_patterns("figures", "results", ".snakemake"))
    poison = tmp_path / "poison"
    (poison / "matplotlib").mkdir(parents=True)
    (poison / "matplotlib" / "__init__.py").write_text('raise ImportError("matplotlib is poisoned for this test")\n')
    env = {"PYTHONPATH": os.pathsep.join([str(poison), str(REPO)]), "PATH": "/usr/bin:/bin", figures.STYLE_ENV: str(STAND_IN)}
    # the two commands of the shared rule, in order
    subprocess.run([sys.executable, "-m", "claimcheck.relate", ".", "results/claims.json"], cwd=proj, env=env, check=True, capture_output=True)
    verdicts = (proj / "results" / "claims.json").read_bytes()
    assert json.loads(verdicts)["relationships"]
    r = subprocess.run([sys.executable, "-m", "claimcheck.figures", ".", "figures/claims"], cwd=proj, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "matplotlib is not importable" in r.stderr, r.stderr
    assert (proj / "results" / "claims.json").read_bytes() == verdicts
    assert not (proj / "figures").exists()
