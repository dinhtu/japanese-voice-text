"""Tests for app/services/mora_timing.py (no ASR model involved -- these
work directly on fabricated recognized-kana / char_spans pairs, the same
shape KanaRecognizer.transcribe(with_timing=True) produces)."""

import math

from app.services.mora_timing import _full_alignment, mora_time_windows

TARGET = "ちょっとまってください"  # 10 morae: ちょ っ と ま っ て く だ さ い


def _uniform_spans(kana: str, step: float = 0.1) -> list[tuple[float, float]]:
    return [(i * step, (i + 1) * step) for i in range(len(kana))]


def test_full_alignment_reports_matches_not_just_edits():
    pairs = _full_alignment(list("あいう"), list("あいう"))
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_full_alignment_marks_deletion_and_insertion():
    pairs = _full_alignment(list("あいう"), list("あう"))
    # 'い' deleted; 'あ' and 'う' still matched.
    assert (1, None) in pairs
    assert (0, 0) in pairs
    assert (2, 1) in pairs


def test_perfect_recognition_gives_real_merged_timing_per_mora():
    recognized = TARGET
    char_spans = _uniform_spans(recognized)
    duration = len(recognized) * 0.1
    windows = mora_time_windows(TARGET, recognized, char_spans, duration)
    assert len(windows) == 10
    # "ちょ" is two characters (small yoon fusion) -> its window should
    # span both characters' real timing, not just one.
    assert math.isclose(windows[0][0], 0.0)
    assert math.isclose(windows[0][1], 0.2)


def test_substitution_still_borrows_real_timing_and_stays_well_formed():
    recognized = "ちょっとたってください"  # "ま" misheard as "た"
    char_spans = _uniform_spans(recognized)
    duration = len(recognized) * 0.1
    windows = mora_time_windows(TARGET, recognized, char_spans, duration)
    assert len(windows) == 10
    assert all(windows[i][1] <= windows[i + 1][0] + 1e-9 for i in range(len(windows) - 1))


def test_dropped_mora_interpolates_without_crashing():
    recognized = "ちょっとまってださい"  # "く" dropped entirely
    char_spans = _uniform_spans(recognized)
    duration = len(recognized) * 0.1
    windows = mora_time_windows(TARGET, recognized, char_spans, duration)
    assert len(windows) == 10
    assert all(start <= end for start, end in windows)
    assert all(windows[i][1] <= windows[i + 1][0] + 1e-9 for i in range(len(windows) - 1))
    assert windows[0][0] >= 0.0
    assert windows[-1][1] <= duration + 1e-9


def test_fully_garbled_recognition_still_returns_one_window_per_mora():
    recognized = "んんんんんんんんんん"
    char_spans = _uniform_spans(recognized)
    windows = mora_time_windows(TARGET, recognized, char_spans, 1.0)
    assert len(windows) == 10
    assert all(0.0 <= start <= end <= 1.0 for start, end in windows)


def test_empty_target_returns_empty_list():
    assert mora_time_windows("", "abc", [], 1.0) == []
