"""Scoring logic tests (no model inference involved)."""

import pytest

from app.services.normalization import to_hiragana
from app.services.scoring import score_pronunciation

TARGET = "げんきょうもいちにちがんばりましょう"


def evaluate(target: str, recognized: str):
    """Normalize both sides the way the use case does, then score."""
    return score_pronunciation(to_hiragana(target), to_hiragana(recognized))


def test_exact_match_scores_100():
    result = evaluate(TARGET, TARGET)
    assert result.score == 100
    assert result.cer == 0.0
    assert result.distance == 0
    assert result.errors == []


def test_long_vowel_variant_is_absorbed_by_normalization():
    """しょう vs しょー is a writing difference, not a pronunciation error."""
    result = evaluate(TARGET, "げんきょーもいちにちがんばりましょー")
    assert result.score == 100
    assert result.cer == 0.0


def test_small_difference_scores_high_but_not_perfect():
    result = evaluate(TARGET, "げんきょうもいちにちがんばりましょ")
    assert 0 < result.cer < 0.2
    assert 80 <= result.score < 100
    assert result.errors


def test_completely_different_text_scores_low():
    close = evaluate(TARGET, "げんきょうもいちにちがんばりましょ")
    far = evaluate(TARGET, "こんにちは")
    assert far.score < 50
    assert far.score < close.score
    assert far.cer > 0.5


def test_kanji_target_normalizes_to_same_reading():
    """Kanji target vs kana speech must still match."""
    result = evaluate("今日も一日頑張りましょう", "きょーもいちにちがんばりましょー")
    assert result.score == 100


def test_empty_target_raises():
    with pytest.raises(ValueError):
        score_pronunciation("", "こんにちは")


def test_empty_recognition_scores_zero():
    result = score_pronunciation(to_hiragana(TARGET), "")
    assert result.score == 0
    assert result.cer == 1.0


def test_score_is_clamped_to_zero_for_long_hypotheses():
    result = score_pronunciation("あい", "こんにちはさようなら")
    assert result.score == 0
    assert result.cer > 1.0


def test_feedback_levels_track_the_score():
    assert evaluate(TARGET, TARGET).level == "excellent"
    assert evaluate(TARGET, "こんにちは").level == "mismatch"


def test_errors_report_position_and_characters():
    result = score_pronunciation("あいうえお", "あいえお")
    assert result.distance == 1
    assert result.errors[0].type == "del"
    assert result.errors[0].target == "う"
    assert result.errors[0].position == 2
