"""Pitch-per-mora extraction tests (no ASR model involved).

Uses synthetic tones rather than real speech, since these only need to
check the mechanics (Hz -> semitone, voiced/silence gating, bucketing into
`num_morae` slices) work at all -- not that they sound natural.
"""

import numpy as np

from app.services.pitch_extraction import extract_pitch_per_mora

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
