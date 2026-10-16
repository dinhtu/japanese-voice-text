"""Is a voice reading at all? Silence, noise and whispering are not scored.

wav2vec2 always writes something: rustle, room tone or a whisper come back
as words, and a whisper of the target can even score 100. Two cheap
signals decide before the ASR result is trusted:

- Silero VAD speech (vad_fluency): 0 s on silence, room tone, white noise,
  rustle and mains hum; a whisper still counts as speech.
- Voiced (periodic) time: frames whose normalised autocorrelation peaks
  above VOICED_NCCF at a 60-400 Hz pitch lag -- the vocal folds vibrate.
  A whisper has none. Measured on TTS clips and LPC-whispered copies of
  them: read aloud 0.14 .. 1.0 of the speech time, whispered 0.02 .. 0.08.
  pYIN was tried first and lost a plainly read "cap" entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.vad_fluency import VadResult

VOICED_NCCF = 0.6
# Voiced share of the speech time below which the reading is a whisper.
MIN_VOICED_SHARE = 0.1
# Without Silero the energy VAD finds "speech" in any noise: then at least
# this much voiced time is required instead.
MIN_VOICED_S = 0.05

_FRAME_S = 0.04
_HOP_S = 0.01


@dataclass(frozen=True)
class VoicePresence:
    verdict: str  # "voice" | "silent" | "whisper"
    speech_s: float
    voiced_s: float

    @property
    def voiced(self) -> bool:
        return self.verdict == "voice"


def voiced_seconds(samples: np.ndarray, sample_rate: int) -> float:
    """Seconds of loud frames with a clear pitch period (60-400 Hz)."""
    frame = int(sample_rate * _FRAME_S)
    hop = int(sample_rate * _HOP_S)
    x = np.asarray(samples, dtype=np.float64)
    if frame <= 0 or hop <= 0 or x.size < frame:
        return 0.0
    count = 1 + (x.size - frame) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop][:count]
    frames = frames - frames.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    loud = rms > max(0.15 * float(rms.max()), 1e-4)
    frames = frames[loud]
    if not len(frames):
        return 0.0
    lo, hi = int(sample_rate / 400), min(int(sample_rate / 60), frame - 1)
    spectrum = np.fft.rfft(frames, n=2 * frame, axis=1)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :frame]
    power = frames * frames
    # Energy of the two overlapping parts at lag k: x[k:] and x[:frame-k].
    tail = np.cumsum(power[:, ::-1], axis=1)[:, ::-1]
    head = np.cumsum(power, axis=1)
    lags = np.arange(lo, hi)
    nccf = ac[:, lags] / (np.sqrt(tail[:, lags] * head[:, frame - 1 - lags]) + 1e-12)
    return float(np.sum(nccf.max(axis=1) > VOICED_NCCF) * _HOP_S)


def check_voice(samples: np.ndarray, sample_rate: int, vad: VadResult) -> VoicePresence:
    duration = len(samples) / sample_rate if sample_rate else 0.0
    speech = vad.speech_ratio * duration
    if vad.method == "silero" and speech <= 0:
        return VoicePresence("silent", 0.0, 0.0)
    voiced = voiced_seconds(samples, sample_rate)
    if vad.method != "silero" and voiced < MIN_VOICED_S:
        return VoicePresence("silent", speech, voiced)
    if voiced < MIN_VOICED_SHARE * speech:
        return VoicePresence("whisper", speech, voiced)
    return VoicePresence("voice", speech, voiced)
