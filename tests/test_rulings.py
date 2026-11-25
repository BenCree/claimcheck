"""The register of decisions already made.

Two-sided throughout: a duplicate must be caught AND a genuinely new question
must pass, because a checker that flags everything is the same as one that flags
nothing.
"""
import pytest

from claimcheck import rulings as R

REGISTER = '''
[[rulings]]
id = "R-protonation-tool"
date = "2026-11-13"
question = "Which protonation tool is the standing choice?"
ruling = "dimorphite-dl, at pH 7.4."
quote = "it stays, at pH 7.4"
decided_by = "project owner"
answers = ["D-protonation", "D-protonation-tool"]

[[rulings]]
id = "R-licence"
date = "2026-11-13"
question = "What licence do the input archives carry?"
ruling = "CC0-1.0."
decided_by = "project owner"

[[rulings]]
id = "R-old-pocket"
date = "2026-11-01"
question = "May the secondary pocket be docked at all?"
ruling = "No, not until the loop is modelled."
status = "superseded"
superseded_by = "R-pocket"

[[rulings]]
id = "R-pocket"
date = "2026-11-13"
question = "May the secondary pocket be docked at all?"
ruling = "Yes: protonated structure, ligand in the pocket, gnina then ApoDock."
decided_by = "project owner"
answers = ["D-pocket-dockable"]
'''


@pytest.fixture
def register(tmp_path):
    p = tmp_path / "RULINGS.toml"
    p.write_text(REGISTER)
    return p


def _open(tmp_path, body):
    p = tmp_path / "open.toml"
    p.write_text(body)
    return p


# ------------------------------------------------------------------ loading


def test_a_missing_register_is_an_error_not_an_empty_one(tmp_path):
    """'No rulings' and 'the register is not where you said' must not look
    alike: the second silently turns the check into a no-op."""
    with pytest.raises(R.RegisterError, match="no register at"):
        R.load(tmp_path / "nowhere.toml")


def test_the_register_refuses_to_load_malformed_rulings(tmp_path):
    bad = [
        ('[[rulings]]\nid = "A"\nquestion = "q"\n', "no ruling"),          # no ruling
        ('[[rulings]]\ndate = "2026-01-01"\nquestion = "q"\nruling = "r"\n', "no id"),
        ('[[rulings]]\nid = "A"\ndate = "d"\nquestion = "q"\nruling = "r"\n'
         '[[rulings]]\nid = "A"\ndate = "d"\nquestion = "q"\nruling = "r"\n',
         "share the id"),
        ('[[rulings]]\nid = "A"\ndate = "d"\nquestion = "q"\nruling = "r"\n'
         'status = "maybe"\n', "neither"),
        ('[[rulings]]\nid = "A"\ndate = "d"\nquestion = "q"\nruling = "r"\n'
         'status = "superseded"\n', "superseded by nothing"),
        ('[[rulings]]\nid = "A"\ndate = "d"\nquestion = "q"\nruling = "r"\n'
         'status = "superseded"\nsuperseded_by = "ghost"\n', "not in the register"),
        ('title = "not a register"\n', r"no \[\[rulings\]\] blocks"),
    ]
    for body, msg in bad:
        p = tmp_path / "r.toml"
        p.write_text(body)
        with pytest.raises(R.RegisterError, match=msg):
            R.load(p)


def test_a_well_formed_register_loads_and_separates_standing_from_superseded(register):
    rs = R.load(register)
    assert len(rs) == 4
    assert {r["id"] for r in R.standing(rs)} == {
        "R-protonation-tool", "R-licence", "R-pocket"}


# ------------------------------------------------------ the check, both sides


def test_a_question_already_ruled_on_is_caught_by_id(register, tmp_path):
    op = _open(tmp_path, '[[decisions]]\nid = "D-protonation-tool"\n'
                         'question = "Which protonation tool is standard?"\n')
    settled, _ = R.check(R.load(register), R.load_open(op))
    assert len(settled) == 1
    q, r = settled[0]
    assert q["id"] == "D-protonation-tool" and r["id"] == "R-protonation-tool"


def test_a_genuinely_new_question_passes(register, tmp_path):
    """The other side: a checker that flags everything is one that flags
    nothing, because people stop reading it."""
    op = _open(tmp_path, '[[decisions]]\nid = "D-cohort-size"\n'
                         'question = "How many congeneric ligands per '
                         'target does the next campaign need?"\n')
    settled, similar = R.check(R.load(register), R.load_open(op))
    assert settled == [] and similar == []


def test_a_question_may_name_the_ruling_that_settles_it(register, tmp_path):
    op = _open(tmp_path, '[[decisions]]\nid = "D-new-id"\nquestion = "anything"\n'
                         'settled_by = "R-licence"\n')
    settled, _ = R.check(R.load(register), R.load_open(op))
    assert len(settled) == 1 and settled[0][1]["id"] == "R-licence"


def test_a_superseded_ruling_never_blocks(register, tmp_path):
    """That is what superseding means. The old pocket ruling said no; the
    standing one says yes, and only the standing one may settle anything."""
    op = _open(tmp_path, '[[decisions]]\nid = "D-pocket-dockable"\n'
                         'question = "May the secondary pocket be docked at all?"\n')
    settled, _ = R.check(R.load(register), R.load_open(op))
    assert len(settled) == 1
    assert settled[0][1]["id"] == "R-pocket"
    assert settled[0][1]["ruling"].startswith("Yes")


def test_a_reworded_duplicate_warns_and_does_not_block(register, tmp_path):
    """Text overlap cannot be trusted to decide, so it reports and never
    enforces. Both halves are the claim: it must warn, and it must not settle."""
    op = _open(tmp_path, '[[decisions]]\nid = "D-brand-new"\n'
                         'question = "What licence should the input archives carry?"\n')
    settled, similar = R.check(R.load(register), R.load_open(op))
    assert settled == [], "a text match must never block"
    assert len(similar) == 1 and similar[0][1]["id"] == "R-licence"
    assert similar[0][2] >= R.SIMILAR


def test_similarity_is_zero_for_unrelated_questions_and_high_for_a_rewording():
    assert R.similarity("What licence do the input archives carry?",
                        "What licence should the input archives carry?") >= 0.6
    assert R.similarity("Which protonation tool is the standing choice?",
                        "How many ligands per target does the campaign need?") == 0.0
    # a question made only of stopwords carries no signal and must not match
    assert R.similarity("and the of it", "What licence do the archives carry?") == 0.0


# ------------------------------------------------------------------- the CLI


def test_check_exits_one_on_a_settled_question_and_zero_otherwise(register, tmp_path,
                                                                  capsys):
    dup = _open(tmp_path, '[[decisions]]\nid = "D-protonation"\nquestion = "q"\n')
    assert R.main([str(register), "--open", str(dup), "--check"]) == 1
    assert "SETTLED" in capsys.readouterr().out

    fresh = tmp_path / "fresh.toml"
    fresh.write_text('[[decisions]]\nid = "D-fresh"\n'
                     'question = "Which GPU partition runs the next array?"\n')
    assert R.main([str(register), "--open", str(fresh), "--check"]) == 0
    assert "ok:" in capsys.readouterr().out


def test_without_check_a_settled_question_is_reported_but_the_exit_is_zero(
        register, tmp_path, capsys):
    """Fail soft by default, loud always: the report is printed either way, and
    only --check turns it into a blocking gate for a pre-commit hook."""
    dup = _open(tmp_path, '[[decisions]]\nid = "D-protonation"\nquestion = "q"\n')
    assert R.main([str(register), "--open", str(dup)]) == 0
    assert "SETTLED" in capsys.readouterr().out


def test_listing_the_register_prints_the_standing_rulings(register, capsys):
    assert R.main([str(register)]) == 0
    out = capsys.readouterr().out
    assert "3 standing ruling(s), 1 superseded" in out
    assert "it stays, at pH 7.4" in out
    assert "R-old-pocket" not in out


def test_a_broken_register_exits_two_rather_than_passing_silently(tmp_path, capsys):
    p = tmp_path / "r.toml"
    p.write_text("not a register\n")
    assert R.main([str(p)]) == 2
    assert "register error" in capsys.readouterr().err
