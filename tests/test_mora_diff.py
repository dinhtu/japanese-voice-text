"""Tests for app/services/mora_diff.py."""

from app.services.mora_diff import _split_mora_spans, mora_status
from app.services.scoring import collect_errors

TARGET = "ちょっとまってください"


def test_split_mora_spans_fuses_small_yoon():
    spans = _split_mora_spans(TARGET)
    assert [text for text, _start, _end in spans] == [
        "ちょ", "っ", "と", "ま", "っ", "て", "く", "だ", "さ", "い",
    ]


def test_split_mora_spans_tracks_char_offsets():
    spans = _split_mora_spans("きょう")
    assert spans == [("きょ", 0, 2), ("う", 2, 3)]


def test_perfect_recognition_marks_every_mora_ok():
    errors = collect_errors(TARGET, TARGET)
    statuses = mora_status(TARGET, errors)
    assert [s.mora for s in statuses] == [
        "ちょ", "っ", "と", "ま", "っ", "て", "く", "だ", "さ", "い",
    ]
    assert all(s.ok for s in statuses)


def test_substitution_flags_only_its_own_mora():
    recognized = "ちょっとたってください"  # "ま" heard as "た"
    errors = collect_errors(TARGET, recognized)
    statuses = mora_status(TARGET, errors)
    bad = [s.mora for s in statuses if not s.ok]
    assert bad == ["ま"]


def test_deletion_flags_the_dropped_mora():
    recognized = "ちょっとまってださい"  # "く" dropped
    errors = collect_errors(TARGET, recognized)
    statuses = mora_status(TARGET, errors)
    bad = [s.mora for s in statuses if not s.ok]
    assert bad == ["く"]


def test_empty_errors_list_marks_everything_ok():
    statuses = mora_status(TARGET, [])
    assert all(s.ok for s in statuses)
    assert len(statuses) == 10
