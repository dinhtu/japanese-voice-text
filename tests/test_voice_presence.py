import numpy as np

from app.services.vad_fluency import VadResult
from app.services.voice_presence import check_voice, voiced_seconds

SR = 16000
rng = np.random.default_rng(0)


def _voice(seconds: float, f0: float = 120.0) -> np.ndarray:
    """A vowel-like buzz: harmonics of f0 with a slow loudness swell."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return (0.3 * x * np.hanning(t.size)).astype(np.float32)


def _vad(speech_s: float, duration: float, method: str = "silero") -> VadResult:
    return VadResult(speech_s, 0, speech_s / duration if duration else 0.0, method)


def test_voiced_seconds_finds_a_voice_not_noise():
    assert voiced_seconds(_voice(1.0), SR) > 0.5
    assert voiced_seconds((0.2 * rng.standard_normal(SR)).astype(np.float32), SR) < 0.02
    assert voiced_seconds(np.zeros(SR, dtype=np.float32), SR) == 0.0


def test_silero_hearing_nothing_is_silent():
    noise = (0.2 * rng.standard_normal(2 * SR)).astype(np.float32)
    assert check_voice(noise, SR, _vad(0.0, 2.0)).verdict == "silent"


def test_whisper_is_speech_without_a_voice():
    # Silero hears speech, but nothing in it is periodic.
    hiss = (0.2 * rng.standard_normal(2 * SR)).astype(np.float32)
    presence = check_voice(hiss, SR, _vad(1.5, 2.0))
    assert presence.verdict == "whisper"
    assert not presence.voiced


def test_reading_aloud_is_a_voice():
    clip = np.concatenate([np.zeros(SR // 2, dtype=np.float32), _voice(1.0)])
    assert check_voice(clip, SR, _vad(1.0, 1.5)).voiced


def test_energy_vad_needs_some_voice():
    noise = (0.2 * rng.standard_normal(2 * SR)).astype(np.float32)
    assert check_voice(noise, SR, _vad(2.0, 2.0, method="energy")).verdict == "silent"
    assert check_voice(_voice(1.0), SR, _vad(1.0, 1.0, method="energy")).voiced
