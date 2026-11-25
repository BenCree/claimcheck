"""The British spelling of `licence` must not publish an unlicensed package.

Found 2026-11-13: 13 experiments declared `licence` in context.toml and every
emitter reads `license`, so the declaration reached nothing and the packages
went out with no licence at all. That is a silent failure, which is the kind
worth a test.
"""
import pytest

from claimcheck.emit import project_table


def test_the_us_spelling_passes_through_untouched():
    proj = project_table({"project": {"name": "x", "license": "CC0-1.0"}})
    assert proj["license"] == "CC0-1.0"
    assert "licence" not in proj


def test_the_british_spelling_is_accepted_and_warned_about(capsys):
    """Accepted rather than only warned about: the intent is unambiguous and
    the alternative is shipping an unlicensed package, the worse failure."""
    proj = project_table({"project": {"name": "x", "licence": "CC-BY-4.0"}})
    assert proj["license"] == "CC-BY-4.0"
    assert "licence" not in proj, "the alias must not survive into the emitters"
    err = capsys.readouterr().err
    assert "licence" in err and "license" in err


def test_both_spellings_agreeing_is_fine_and_the_alias_is_dropped(capsys):
    proj = project_table({"project": {"license": "MIT", "licence": "MIT"}})
    assert proj["license"] == "MIT" and "licence" not in proj


def test_both_spellings_disagreeing_is_an_error_not_a_merge():
    """Nothing here may choose which licence a project meant."""
    with pytest.raises(ValueError, match="differ"):
        project_table({"project": {"license": "CC0-1.0", "licence": "CC-BY-4.0"}})


def test_no_licence_at_all_stays_absent():
    proj = project_table({"project": {"name": "x"}})
    assert "license" not in proj and "licence" not in proj


def test_a_missing_project_table_is_an_empty_dict_not_a_crash():
    assert project_table({}) == {}
