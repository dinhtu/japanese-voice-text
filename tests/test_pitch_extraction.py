"""Pitch-per-mora extraction tests (no ASR model involved).

Uses synthetic tones rather than real speech, since these only need to
check the mechanics (Hz -> semitone, voiced/silence gating, bucketing into
`num_morae` slices) work at all -- not that they sound natural.
"""

import numpy as np

from app.services.pitch_extraction import (
    _pause_aware_bucket_edges,
    _snap_to_pause,
    extract_pitch_for_windows,
    extract_pitch_per_mora,
)

SAMPLE_RATE = 16_000


def _tone(freq_hz: float, seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(sample_rate * seconds), dtype=np.float32)


def test_silence_returns_all_unvoiced_placeholders():
    points = extract_pitch_per_mora(_silence(1.0), SAMPLE_RATE, num_morae=5)
    assert len(points) == 5
    assert all(not p["voiced"] and p["semitone"] is None for p in points)


def test_empty_audio_returns_placeholders_of_the_requested_length():
    points = extract_pitch_per_mora(np.array([], dtype=np.float32), SAMPLE_RATE, num_morae=3)
    assert points == [{"semitone": None, "voiced": False}] * 3


def test_zero_morae_returns_empty_list():
    assert extract_pitch_per_mora(_tone(220, 0.5), SAMPLE_RATE, num_morae=0) == []


def test_two_tones_bucket_into_two_morae_in_order():
    """220Hz then 440Hz (one octave = +12 semitones), no silence padding --
    with 2 target morae, the split should land near the midpoint and each
    half should read close to its own tone's semitone value."""
    audio = np.concatenate([_tone(220, 0.5), _tone(440, 0.5)])
    points = extract_pitch_per_mora(audio, SAMPLE_RATE, num_morae=2)

    assert len(points) == 2
    assert points[0]["voiced"] and points[1]["voiced"]
    # 220 -> 440 Hz is +12 semitones; allow slack for NCF/median-filter noise.
    assert (points[1]["semitone"] - points[0]["semitone"]) > 8


def test_leading_and_trailing_silence_is_trimmed_before_bucketing():
    """A short tone buried in silence should still land as voiced once
    trimmed to its own span, rather than being diluted across mostly-silent
    buckets."""
    audio = np.concatenate([_silence(0.5), _tone(300, 0.4), _silence(0.5)])
    points = extract_pitch_per_mora(audio, SAMPLE_RATE, num_morae=3)
    assert len(points) == 3
    assert any(p["voiced"] for p in points)


def test_bucket_boundary_snaps_toward_a_real_pause_between_tones():
    """A short internal silence (like a sokuon) sitting well off the
    equal-time midpoint should pull the interior boundary toward itself,
    instead of leaving the naive 50/50 split to slice partway into the
    second tone and blend it into the first mora's average.

    tone1 is short (0.2s) and tone2 is long (0.6s), so the naive midpoint
    (0.45s) lands deep inside tone2 -- 0.15s past where tone2 actually
    starts (0.3s). Snapping toward the 0.2-0.3s pause keeps that 0.15s of
    tone2 out of bucket 0, so the measured gap between buckets should be
    close to the tones' true ~8.8-semitone difference (12*log2(500/300)),
    not diluted by a blended bucket 0.
    """
    audio = np.concatenate([_tone(300, 0.2), _silence(0.1), _tone(500, 0.6)])
    points = extract_pitch_per_mora(audio, SAMPLE_RATE, num_morae=2)
    assert len(points) == 2
    assert points[0]["voiced"] and points[1]["voiced"]
    assert (points[1]["semitone"] - points[0]["semitone"]) > 6


def test_snap_to_pause_finds_a_real_dip_within_its_search_window():
    energy = np.ones(50)
    energy[18:22] = 0.01  # a clear dip away from the naive midpoint (25)
    snapped = _snap_to_pause(energy, naive_idx=25, lo=0, hi=50, radius=10)
    assert 18 <= snapped <= 22


def test_snap_to_pause_ignores_noise_with_no_real_dip():
    rng = np.random.RandomState(0)
    energy = np.ones(50) + rng.normal(0, 0.01, 50)
    snapped = _snap_to_pause(energy, naive_idx=25, lo=0, hi=50, radius=10)
    assert snapped == 25


def test_pause_aware_bucket_edges_stay_monotonic_with_several_pauses():
    energy = np.ones(90)
    energy[25:30] = 0.01
    energy[60:65] = 0.01
    edges = _pause_aware_bucket_edges(energy, start=0, end=90, num_morae=3)
    assert list(edges) == sorted(edges)
    assert len(set(int(e) for e in edges)) == len(edges)
    assert edges[0] == 0 and edges[-1] == 90


def test_pause_aware_bucket_edges_never_collide_on_one_shared_pause():
    """Two interior boundaries both drawn toward the same short pause must
    still end up strictly increasing, not stacked on the same frame."""
    energy = np.ones(90)
    energy[44:46] = 0.01
    edges = _pause_aware_bucket_edges(energy, start=0, end=90, num_morae=3)
    assert list(edges) == sorted(edges)
    assert len(set(int(e) for e in edges)) == len(edges)


def test_extract_pitch_for_windows_reads_the_right_tone_per_window():
    """Two back-to-back tones with REAL windows handed in (as
    app.services.mora_timing would produce from ASR timing) should each
    read close to their own tone, even though the tones aren't equal
    length -- unlike extract_pitch_per_mora, there's no bucketing/snapping
    heuristic here, just "average the pitch inside this exact window"."""
    tone1 = _tone(300, 0.2)
    tone2 = _tone(500, 0.6)
    audio = np.concatenate([tone1, tone2])
    windows = [(0.0, 0.2), (0.2, 0.8)]
    points = extract_pitch_for_windows(audio, SAMPLE_RATE, windows)

    assert len(points) == 2
    assert points[0]["voiced"] and points[1]["voiced"]
    # 300 -> 500 Hz is 12*log2(500/300) ~= 8.8 semitones.
    assert (points[1]["semitone"] - points[0]["semitone"]) > 6


def test_extract_pitch_for_windows_marks_silent_window_unvoiced():
    audio = np.concatenate([_tone(300, 0.3), _silence(0.3)])
    windows = [(0.0, 0.3), (0.3, 0.6)]
    points = extract_pitch_for_windows(audio, SAMPLE_RATE, windows)
    assert points[0]["voiced"]
    assert not points[1]["voiced"] and points[1]["semitone"] is None


def test_extract_pitch_for_windows_empty_windows_list():
    assert extract_pitch_for_windows(_tone(220, 0.3), SAMPLE_RATE, []) == []


def test_extract_pitch_for_windows_no_audio_returns_placeholders():
    windows = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)]
    points = extract_pitch_for_windows(np.array([], dtype=np.float32), SAMPLE_RATE, windows)
    assert points == [{"semitone": None, "voiced": False}] * 3
