"""Energy VAD fallback -- does not require silero-vad."""

import numpy as np

from app.services.vad_fluency import _energy_vad, analyze_vad


def test_energy_vad_finds_speech_island():
    sr = 16_000
    silence = np.zeros(sr, dtype=np.float32)
    tone = 0.4 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    samples = np.concatenate([silence, tone, silence])
    result = _energy_vad(samples, sr)
    assert result.method == "energy"
    assert result.speech_duration > 0.5
    assert result.pause_count == 0
    assert result.speech_ratio > 0.2


def test_energy_vad_empty_is_zero():
    result = _energy_vad(np.zeros(0, dtype=np.float32), 16_000)
    assert result.speech_duration == 0.0
    assert result.speech_ratio == 0.0


def test_analyze_vad_never_raises_on_silence():
    result = analyze_vad(np.zeros(1600, dtype=np.float32), 16_000)
    assert result.speech_duration >= 0
    assert result.method in {"silero", "energy"}
