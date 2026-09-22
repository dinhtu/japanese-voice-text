"""GOPT-style aspect scores -- pure arithmetic, no ASR / Ollama."""

from types import SimpleNamespace

from app.services.aspect_scoring import (
    combine_overall,
    score_aspects,
    score_fluency,
    score_intonation,
    score_rhythm,
)
from app.services.prosody_issues import DurationIssue
from app.services.scoring import PronunciationError


def _mora(text: str, pitch: str = "L") -> SimpleNamespace:
    return SimpleNamespace(mora=text, pitch=pitch, phrase=0)


def test_fluency_plateau_for_careful_reading_pace():
    # 16 morae in 3.2s = 5.0 mora/s, inside the 3.8--7.2 band.
    assert score_fluency(3.2, 16, pronunciation=100.0) == 100.0


def test_fluency_drops_when_rushed_or_dragged():
    rushed = score_fluency(1.0, 16, pronunciation=100.0)  # 16 mora/s
    dragged = score_fluency(12.0, 16, pronunciation=100.0)  # 1.33 mora/s
    assert rushed < 50
    assert dragged < 50


def test_fluency_penalizes_incomplete_reading():
    full = score_fluency(3.2, 16, pronunciation=100.0)
    skipped = score_fluency(3.2, 16, pronunciation=40.0)
    assert skipped < full


def test_rhythm_uses_edit_ops_when_windows_missing():
    errors = [
        PronunciationError(type="del", target="っ", recognized="", position=1),
        PronunciationError(type="ins", target="", recognized="ん", position=2),
    ]
    score, measured = score_rhythm(None, None, [], errors)
    assert measured is False
    assert score < 100
    assert score == 89.0  # 100 - 6 del - 5 ins


def test_rhythm_penalizes_uneven_windows_and_short_sokuon():
    moras = [_mora("あ"), _mora("っ"), _mora("と"), _mora("う")]
    even = [(0.0, 0.18), (0.18, 0.36), (0.36, 0.54), (0.54, 0.72)]
    even_score, measured = score_rhythm(even, moras, [], [])
    assert measured is True
    assert even_score == 100.0

    uneven = [(0.0, 0.08), (0.08, 0.12), (0.12, 0.55), (0.55, 0.90)]
    issue = DurationIssue(
        kind="short_sokuon",
        mora_index=1,
        mora="っ",
        prev_mora="あ",
        next_mora="と",
        measured_seconds=0.04,
        expected_seconds=0.20,
    )
    uneven_score, _ = score_rhythm(uneven, moras, [issue], [])
    assert uneven_score < even_score


def test_intonation_none_without_voiced_frames():
    score, measured = score_intonation(None, None)
    assert score is None
    assert measured is False
    assert score_intonation(0, 0) == (None, False)


def test_intonation_scales_with_direction_match():
    perfect, ok = score_intonation(4, 4)
    half, _ = score_intonation(2, 4)
    none, _ = score_intonation(0, 4)
    assert ok is True
    assert perfect == 100.0
    assert half == 60.0
    assert none == 20.0


def test_overall_renormalizes_when_intonation_missing():
    with_i = combine_overall(100, 100, 100, 20)
    without_i = combine_overall(100, 100, 100, None)
    assert without_i == 100.0
    assert with_i < without_i


def test_score_aspects_perfect_bundle():
    moras = [_mora(m, p) for m, p in (("あ", "H"), ("い", "L"), ("う", "H"), ("え", "L"))]
    windows = [(0.0, 0.18), (0.18, 0.36), (0.36, 0.54), (0.54, 0.72)]
    result = score_aspects(
        pronunciation_cer_score=100,
        n_morae=4,
        audio_duration=0.72,
        speech_duration=0.72,
        windows=windows,
        moras=moras,
        duration_issues=[],
        errors=[],
        pitch_matched=4,
        pitch_total=4,
    )
    assert result.method == "local-aspect"
    assert result.pronunciation_score == 100.0
    assert result.fluency_score == 100.0
    assert result.rhythm_score == 100.0
    assert result.intonation_score == 100.0
    assert result.overall_score == 100.0
    assert result.rhythm_measured is True
    assert result.intonation_measured is True


def test_score_aspects_leaves_intonation_null_when_unvoiced():
    result = score_aspects(
        pronunciation_cer_score=90,
        n_morae=8,
        audio_duration=1.6,
    )
    assert result.pronunciation_score == 90.0
    assert result.intonation_score is None
    assert result.intonation_measured is False
    assert result.rhythm_measured is False
    assert 0 <= result.overall_score <= 100
