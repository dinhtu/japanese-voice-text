"""Tests for app/services/prosody_issues.py (no ASR/torch involved -- works
directly on fabricated MoraPitch + window pairs, the same shape
pitch_accent_pattern() / mora_time_windows() produce)."""

from app.services.pitch_accent import MoraPitch
from app.services.prosody_issues import detect_duration_issues


def _moras(text_pattern: list[tuple[str, str]]) -> list[MoraPitch]:
    """Build MoraPitch list from [(mora, pitch), ...] pairs, all phrase 0."""
    return [MoraPitch(mora=m, pitch=p, phrase=0) for m, p in text_pattern]


def test_dropped_sokuon_is_flagged():
    """'ちょっと' read with a rushed っ (much shorter than the other morae) --
    exactly the 'choto' vs 'chotto' mistake from the feature request."""
    moras = _moras([("ちょ", "H"), ("っ", "L"), ("と", "L")])
    # ちょ and と each ~0.15s (ordinary pace); っ only 0.03s -- clearly rushed.
    windows = [(0.0, 0.15), (0.15, 0.18), (0.18, 0.33)]
    issues = detect_duration_issues(moras, windows)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.kind == "short_sokuon"
    assert issue.mora == "っ"
    assert issue.prev_mora == "ちょ" and issue.next_mora == "と"
    assert issue.measured_seconds == 0.03


def test_properly_held_sokuon_is_not_flagged():
    """っ held close to a full mora's worth of time -- no issue."""
    moras = _moras([("ちょ", "H"), ("っ", "L"), ("と", "L")])
    windows = [(0.0, 0.15), (0.15, 0.29), (0.29, 0.44)]  # っ ~0.14s, close to 0.15s baseline
    assert detect_duration_issues(moras, windows) == []


def test_dropped_chouon_is_flagged():
    """'きょう' -> きょ + ー, with the long vowel cut short."""
    moras = _moras([("きょ", "L"), ("ー", "H"), ("わ", "H")])
    windows = [(0.0, 0.15), (0.15, 0.18), (0.18, 0.33)]
    issues = detect_duration_issues(moras, windows)
    assert len(issues) == 1
    assert issues[0].kind == "short_chouon"
    assert issues[0].mora == "ー"


def test_no_sokuon_or_chouon_at_all_returns_no_issues():
    moras = _moras([("あ", "H"), ("り", "L"), ("が", "L"), ("と", "L"), ("う", "L")])
    windows = [(i * 0.15, (i + 1) * 0.15) for i in range(5)]
    assert detect_duration_issues(moras, windows) == []


def test_no_ordinary_morae_to_baseline_against_returns_no_issues():
    """Degenerate input: every mora is a sokuon/chouon -- nothing to
    compare durations against, so it must not crash or false-flag."""
    moras = _moras([("っ", "L"), ("ー", "H")])
    windows = [(0.0, 0.02), (0.02, 0.04)]
    assert detect_duration_issues(moras, windows) == []


def test_mismatched_lengths_returns_no_issues():
    moras = _moras([("ちょ", "H"), ("っ", "L"), ("と", "L")])
    windows = [(0.0, 0.15), (0.15, 0.18)]  # one short
    assert detect_duration_issues(moras, windows) == []


def test_multiple_issues_in_one_sentence():
    moras = _moras([("ちょ", "H"), ("っ", "L"), ("と", "L"), ("ー", "L"), ("さ", "H")])
    windows = [(0.0, 0.15), (0.15, 0.17), (0.17, 0.32), (0.32, 0.34), (0.34, 0.49)]
    issues = detect_duration_issues(moras, windows)
    kinds = {(i.mora_index, i.kind) for i in issues}
    assert kinds == {(1, "short_sokuon"), (3, "short_chouon")}
