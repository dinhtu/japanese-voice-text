"""Tests for app/services/ja_scoring.py (per-mora Japanese score)."""

import pytest

from app.services.ja_scoring import (
    FREE_INSERTIONS,
    INSERTION_PENALTY,
    MAX_INSERTION_PENALTY,
    OVERLONG_CAP,
    score_japanese,
)


def grid(result) -> str:
    return "".join("o" if ok else "x" for ok in result.mora_ok)


def test_exact_match_scores_100():
    result = score_japanese("がっぺい", "がっぺい")
    assert result.score == 100 and grid(result) == "oooo"
    assert result.errors == [] and result.distance == 0 and result.cer == 0.0
    assert result.level == "excellent"


def test_score_is_the_share_of_ok_morae():
    # One mora heard as another: 3 of 4 morae ok.
    result = score_japanese("がっぺい", "がっぱい")
    assert grid(result) == "ooxo" and result.score == 75


def test_score_and_grid_always_agree_without_charged_insertions():
    for heard in ["", "が", "がっ", "がっぺ", "かっぺい", "がっぺいた", "たがっぺい"]:
        result = score_japanese("がっぺい", heard)
        assert result.score == round(100 * sum(result.mora_ok) / 4), heard


@pytest.mark.parametrize("heard", ["がっぺー", "がっぺえ"])
def test_long_vowel_spellings_are_one_sound(heard):
    assert score_japanese("がっぺい", heard).score == 100


@pytest.mark.parametrize("heard", ["とーきょー", "とおきょお", "とうきょー"])
def test_long_o_spellings_are_one_sound(heard):
    assert score_japanese("とうきょう", heard).score == 100


def test_long_vowel_needs_the_same_vowel_before_it():
    # ぱー is "paa", not "pei": ぺ is wrong and い does not match ー.
    result = score_japanese("がっぺい", "がっぱー")
    assert grid(result) == "ooxx"


def test_yoon_mora_counts_once():
    result = score_japanese("しゃしん", "さしん")
    assert grid(result) == "xoo" and result.score == 67
    assert [(e.type, e.target, e.recognized) for e in result.errors] == [("sub", "しゃ", "さ")]


def test_a_few_extra_sounds_are_free():
    for extra in range(FREE_INSERTIONS + 1):
        result = score_japanese("がっぺい", "がっ" + "つ" * extra + "ぺい")
        assert result.score == 100 and grid(result) == "oooo"
        assert sum(e.type == "ins" for e in result.errors) == extra


def test_more_extra_sounds_cost_a_little_and_are_capped():
    four = score_japanese("がっぺい", "がっぺいたつけん")
    assert four.score == 100 - (4 - FREE_INSERTIONS) * INSERTION_PENALTY
    many = score_japanese("ちょっとまってください", "ちょっとまってください" + "あいうえおかきくけこ")
    assert many.score == 100 - MAX_INSERTION_PENALTY
    assert grid(many) == "o" * 10


def test_babbling_far_past_the_target_is_capped():
    result = score_japanese("がっぺい", "がっぺい" + "たちつてとなにぬねの")
    assert result.score <= OVERLONG_CAP


def test_reported_case_extra_sound_no_longer_hides_behind_a_green_sokuon():
    # The screenshot: 合併 read so that っ alone aligned. っ has no ✓
    # neighbour, so it is "weak", the grid is all ✗ and matches the 0.
    result = score_japanese("がっぺい", "あったけん")
    assert result.score == 0 and grid(result) == "xxxx"
    assert ("weak", "っ") in [(e.type, e.target) for e in result.errors]


def test_sokuon_kept_when_a_neighbour_is_right():
    result = score_japanese("がっぺい", "がっけえ")
    assert result.mora_ok[1] is True


def test_bare_sokuon_scores_zero():
    assert score_japanese("がっぺい", "っ").score == 0


def test_ties_go_to_the_learner():
    # Swapped っ/ぺ: several 2-edit alignments exist; the chosen one keeps
    # the most morae ok (3), where the plain backtrace kept 2.
    result = score_japanese("がっぺい", "がぺっい")
    assert sum(result.mora_ok) == 3 and result.score == 75


def test_empty_recognition_scores_zero_with_every_mora_missing():
    result = score_japanese("きょう", "")
    assert result.score == 0 and grid(result) == "xx"
    assert [e.type for e in result.errors] == ["del", "del"]


def test_error_positions_are_character_offsets_of_the_mora():
    result = score_japanese("ちょっと", "ちょっこ")
    assert [(e.type, e.target, e.position) for e in result.errors] == [("sub", "と", 3)]


def test_empty_target_raises():
    with pytest.raises(ValueError):
        score_japanese("", "あ")
