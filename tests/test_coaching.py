"""Tests for the pure, non-Ollama parts of app/services/coaching.py:
pitch_direction_match(), build_facts(), _format_facts(). generate_comment()
itself (the actual Ollama HTTP call) has no test here -- it needs a running
Ollama server and is exercised manually instead (see README)."""

from app.services.pitch_accent import MoraPitch
from app.services.prosody_issues import DurationIssue
from app.services.scoring import PronunciationError
from app.services.coaching import (
    PronunciationFacts,
    _format_facts,
    build_facts,
    pitch_direction_match,
)


def _moras(pitches: list[str]) -> list[MoraPitch]:
    return [MoraPitch(mora="x", pitch=p, phrase=0) for p in pitches]


def _point(semitone: float | None, voiced: bool = True) -> dict:
    return {"semitone": semitone, "voiced": voiced}


# --------------------------------------------------------- pitch_direction_match


def test_pitch_direction_match_counts_only_direction_agreement():
    moras = _moras(["H", "L", "H", "L"])
    points = [_point(2.0), _point(-1.0), _point(-0.5), _point(-3.0)]
    # mora0 H vs +2.0 -> match; mora1 L vs -1.0 -> match;
    # mora2 H vs -0.5 -> mismatch; mora3 L vs -3.0 -> match
    matched, total = pitch_direction_match(moras, points)
    assert (matched, total) == (3, 4)


def test_pitch_direction_match_ignores_unvoiced_or_missing_semitone():
    moras = _moras(["H", "L", "H"])
    points = [_point(2.0), _point(None), _point(1.0, voiced=False)]
    matched, total = pitch_direction_match(moras, points)
    # Only mora0 counts -- the other two are unvoiced / have no semitone.
    assert (matched, total) == (1, 1)


def test_pitch_direction_match_returns_zero_on_length_mismatch():
    moras = _moras(["H", "L"])
    points = [_point(1.0)]
    assert pitch_direction_match(moras, points) == (0, 0)


def test_pitch_direction_match_empty_input():
    assert pitch_direction_match([], []) == (0, 0)


# --------------------------------------------------------------------- build_facts


def test_build_facts_collects_wrong_morae_from_errors():
    moras = _moras(["H", "L", "H"])
    errors = [
        PronunciationError(type="sub", target="と", recognized="ど", position=1),
        PronunciationError(type="del", target="す", recognized="", position=2),
        PronunciationError(type="ins", target="", recognized="ん", position=2),
    ]
    facts = build_facts(
        text="てすと",
        score=72,
        level="fair",
        moras=moras,
        errors=errors,
        duration_issues=[],
        pitch_points=None,
    )
    assert facts.total_morae == 3
    assert facts.wrong_morae == ["と→ど", "す→(mất âm)", "(âm thừa)ん"]
    assert facts.pitch_matched is None and facts.pitch_total is None


def test_build_facts_with_no_errors_computes_pitch_when_points_given():
    moras = _moras(["H", "L"])
    points = [_point(2.0), _point(-2.0)]
    facts = build_facts(
        text="はい",
        score=100,
        level="excellent",
        moras=moras,
        errors=[],
        duration_issues=[],
        pitch_points=points,
    )
    assert facts.wrong_morae == []
    assert (facts.pitch_matched, facts.pitch_total) == (2, 2)


# ------------------------------------------------------------------- _format_facts


def test_format_facts_includes_duration_issue_line():
    facts = PronunciationFacts(
        text="ちょっと",
        score=80,
        level="good",
        total_morae=4,
        wrong_morae=[],
        duration_issues=[
            DurationIssue(
                kind="short_sokuon",
                mora_index=1,
                mora="っ",
                prev_mora="ちょ",
                next_mora="と",
                measured_seconds=0.03,
                expected_seconds=0.15,
            )
        ],
        pitch_matched=None,
        pitch_total=None,
    )
    text = _format_facts(facts)
    assert "âm ngắt 「っ」" in text
    assert "ちょ" in text and "と" in text
    assert "20%" in text  # 0.03 / 0.15
    assert "Không có mora nào đọc sai" in text
    assert "Không có đủ dữ liệu cao độ" in text


def test_format_facts_omits_pitch_line_when_pitch_total_is_zero_or_none():
    facts = PronunciationFacts(
        text="はい", score=100, level="excellent", total_morae=2,
    )
    text = _format_facts(facts)
    assert "Không có đủ dữ liệu cao độ để đánh giá." in text
    assert "Cao độ:" not in text


def test_format_facts_reports_pitch_match_when_available():
    facts = PronunciationFacts(
        text="はい", score=100, level="excellent", total_morae=2,
        pitch_matched=1, pitch_total=2,
    )
    text = _format_facts(facts)
    assert "Cao độ: 1/2 mora" in text
