"""Energy VAD fallback -- does not require silero-vad."""

import numpy as np

from app.services.vad_fluency import _energy_vad, _merge_segments, analyze_vad


def _tone(seconds: float, sr: int = 16_000) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_energy_vad_finds_speech_island():
    sr = 16_000
    silence = np.zeros(sr, dtype=np.float32)
    samples = np.concatenate([silence, _tone(1.0, sr), silence])
    result = _energy_vad(samples, sr)
    assert result.method == "energy"
    assert result.speech_duration > 0.5
    assert result.pause_count == 0
    assert result.speech_ratio > 0.2


def test_merge_segments_fuses_japanese_devoiced_gaps():
    merged = _merge_segments([(0.0, 0.4), (0.55, 0.9), (1.4, 1.8)])
    assert merged == [(0.0, 0.9), (1.4, 1.8)]


def test_energy_vad_leading_silence_is_not_a_pause():
    sr = 16_000
    samples = np.concatenate([np.zeros(2 * sr, dtype=np.float32), _tone(1.0, sr)])
    result = _energy_vad(samples, sr)
    assert result.pause_count == 0
    assert 0.8 <= result.speech_duration <= 1.2


def test_energy_vad_counts_only_long_internal_pauses():
    sr = 16_000
    short_gap = np.concatenate([_tone(0.5, sr), np.zeros(int(0.15 * sr)), _tone(0.5, sr)])
    long_gap = np.concatenate([_tone(0.5, sr), np.zeros(int(0.8 * sr)), _tone(0.5, sr)])
    assert _energy_vad(short_gap, sr).pause_count == 0
    assert _energy_vad(long_gap, sr).pause_count == 1


def test_energy_vad_empty_is_zero():
    result = _energy_vad(np.zeros(0, dtype=np.float32), 16_000)
    assert result.speech_duration == 0.0
    assert result.speech_ratio == 0.0


def test_analyze_vad_never_raises_on_silence():
    result = analyze_vad(np.zeros(1600, dtype=np.float32), 16_000)
    assert result.speech_duration >= 0
    assert result.method in {"silero", "energy"}
