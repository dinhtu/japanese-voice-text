"""Tests for the pure, non-Ollama parts of app/services/coaching.py:
pitch_direction_match(), build_facts(), _format_facts(), _parse_comment().
generate_comment() itself (the actual Ollama HTTP call) has no test here --
it needs a running Ollama server and is exercised manually instead (see
README)."""

from app.services.pitch_accent import MoraPitch
from app.services.prosody_issues import DurationIssue
from app.services.scoring import PronunciationError
from app.services.coaching import (
    COMMENT_JSON_SCHEMA,
    SUPPORTED_LANGUAGES,
    CoachingComment,
    PronunciationFacts,
    _format_facts,
    _parse_comment,
    build_facts,
    pitch_direction_match,
    resolve_system_prompt,
)


def _moras(pitches: list[str]) -> list[MoraPitch]:
    return [MoraPitch(mora="x", pitch=p, phrase=0) for p in pitches]


def _point(semitone: float | None, voiced: bool = True) -> dict:
    return {"semitone": semitone, "voiced": voiced}


# --------------------------------------------------------- pitch_direction_match


def test_pitch_direction_match_counts_only_direction_agreement():
    moras = _moras(["H", "L", "H", "L"])
    # Phrase mean of 3, 1, -1, -3 is 0. H must sit above that mean.
    points = [_point(3.0), _point(1.0), _point(-1.0), _point(-3.0)]
    # mora0 H vs +3 match; mora1 L vs +1 miss; mora2 H vs -1 miss; mora3 L vs -3 match
    matched, total = pitch_direction_match(moras, points)
    assert (matched, total) == (2, 4)


def test_pitch_direction_match_uses_phrase_local_mean():
    """A later High can be negative vs the clip mean and still match if it
    is the high mora of its own accent phrase."""
    moras = [
        MoraPitch(mora="x", pitch="H", phrase=0),
        MoraPitch(mora="x", pitch="L", phrase=0),
        MoraPitch(mora="x", pitch="H", phrase=1),
        MoraPitch(mora="x", pitch="L", phrase=1),
    ]
    points = [_point(3.0), _point(1.0), _point(-1.0), _point(-3.0)]
    matched, total = pitch_direction_match(moras, points)
    assert (matched, total) == (4, 4)


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
# --------------------------------------------------------- resolve_system_prompt


def test_supported_languages_include_vi_and_en():
    assert "vi" in SUPPORTED_LANGUAGES
    assert "en" in SUPPORTED_LANGUAGES


def test_resolve_system_prompt_returns_different_text_per_language():
    vi_prompt = resolve_system_prompt("vi")
    en_prompt = resolve_system_prompt("en")
    assert vi_prompt != en_prompt
    assert len(vi_prompt) > 0 and len(en_prompt) > 0
    # Each prompt should at least tell the model to write in that language.
    assert "tiếng Việt" in vi_prompt
    assert "English" in en_prompt


def test_resolve_system_prompt_rejects_unsupported_language():
    try:
        resolve_system_prompt("fr")
    except ValueError as e:
        assert "fr" in str(e)
        assert "vi" in str(e) and "en" in str(e)
    else:
        raise AssertionError("expected ValueError for unsupported lang")


# --------------------------------------------------------- _parse_comment


def test_parse_comment_splits_valid_json_into_assessment_and_suggestion():
    comment = _parse_comment(
        '{"assessment": "Rat tot.", "suggestion": "Chu y am ngat."}'
    )
    assert comment == CoachingComment(
        assessment="Rat tot.", suggestion="Chu y am ngat."
    )


def test_parse_comment_allows_empty_suggestion():
    comment = _parse_comment('{"assessment": "Rat tot.", "suggestion": ""}')
    assert comment.assessment == "Rat tot."
    assert comment.suggestion == ""


def test_parse_comment_strips_whitespace_from_both_fields():
    comment = _parse_comment('{"assessment": "  Rat tot.  ", "suggestion": "  "}')
    assert comment.assessment == "Rat tot."
    assert comment.suggestion == ""


def test_parse_comment_falls_back_to_plain_text_when_not_json():
    # A local model can still ignore the `format` JSON-schema constraint
    # occasionally -- this is the graceful degrade path (see the
    # docstring's Limitation section): treat the whole reply as the
    # assessment instead of losing the comment.
    comment = _parse_comment("Day la mot cau nhan xet binh thuong.")
    assert comment.assessment == "Day la mot cau nhan xet binh thuong."
    assert comment.suggestion == ""


def test_parse_comment_raises_when_response_is_completely_empty():
    try:
        _parse_comment('{"assessment": "", "suggestion": ""}')
    except Exception as e:  # CoachingUnavailableError
        assert "rỗng" in str(e) or "empty" in str(e).lower()
    else:
        raise AssertionError("expected an error for an empty comment")


def test_comment_json_schema_requires_both_keys():
    assert COMMENT_JSON_SCHEMA["type"] == "object"
    assert set(COMMENT_JSON_SCHEMA["required"]) == {"assessment", "suggestion"}
    assert set(COMMENT_JSON_SCHEMA["properties"]) == {"assessment", "suggestion"}
