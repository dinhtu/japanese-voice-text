"""Speech / pause segmentation for the fluency head.

Prefers Silero VAD (tiny ONNX, CPU, free) when `silero-vad` is installed.
Falls back to an energy gate so /evaluate never depends on a download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_silero_model = None
_silero_failed = False


@dataclass(frozen=True)
class VadResult:
    speech_duration: float
    pause_count: int
    speech_ratio: float
    method: str  # "silero" | "energy"


def _try_silero(samples: np.ndarray, sample_rate: int) -> VadResult | None:
    global _silero_model, _silero_failed
    if _silero_failed:
        return None
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except Exception:  # noqa: BLE001
        _silero_failed = True
        logger.info("silero-vad not installed; fluency uses energy VAD")
        return None

    if _silero_model is None:
        try:
            _silero_model = load_silero_vad()
        except Exception:  # noqa: BLE001
            _silero_failed = True
            logger.warning("Silero VAD failed to load; fluency uses energy VAD", exc_info=True)
            return None

    wav = torch.from_numpy(np.asarray(samples, dtype=np.float32))
    try:
        stamps = get_speech_timestamps(wav, _silero_model, sampling_rate=sample_rate)
    except Exception:  # noqa: BLE001
        logger.warning("Silero VAD inference failed; fluency uses energy VAD", exc_info=True)
        return None

    if not stamps:
        return VadResult(0.0, 0, 0.0, "silero")

    speech = 0.0
    for seg in stamps:
        start = seg["start"] / sample_rate if isinstance(seg, dict) else seg.start / sample_rate
        end = seg["end"] / sample_rate if isinstance(seg, dict) else seg.end / sample_rate
        speech += max(0.0, end - start)
    duration = len(samples) / sample_rate if sample_rate else 0.0
    ratio = min(1.0, speech / duration) if duration > 0 else 0.0
    return VadResult(speech, max(0, len(stamps) - 1), ratio, "silero")


def _energy_vad(samples: np.ndarray, sample_rate: int) -> VadResult:
    """RMS gate -- same idea as pitch_extraction's voiced mask, no extra model."""
    if samples.size == 0 or sample_rate <= 0:
        return VadResult(0.0, 0, 0.0, "energy")
    frame = max(1, int(sample_rate * 0.02))
    n = (len(samples) // frame) * frame
    if n < frame:
        return VadResult(0.0, 0, 0.0, "energy")
    frames = samples[:n].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    thresh = max(float(np.median(rms)) * 1.8, float(np.max(rms)) * 0.08)
    voiced = rms > thresh
    if not voiced.any():
        return VadResult(0.0, 0, 0.0, "energy")

    speech_frames = int(voiced.sum())
    speech = speech_frames * frame / sample_rate
    duration = len(samples) / sample_rate
    # Rising edges = new speech islands after a pause.
    padded = np.concatenate([[False], voiced])
    pause_count = int(np.sum((~padded[:-1]) & padded[1:])) - 1
    pause_count = max(0, pause_count)
    ratio = min(1.0, speech / duration) if duration > 0 else 0.0
    return VadResult(speech, pause_count, ratio, "energy")


def analyze_vad(samples: np.ndarray, sample_rate: int) -> VadResult:
    """Speech span + in-utterance pause count. Never raises."""
    try:
        result = _try_silero(samples, sample_rate)
        if result is not None:
            return result
    except Exception:  # noqa: BLE001
        logger.warning("Silero VAD path crashed; fluency uses energy VAD", exc_info=True)
    return _energy_vad(np.asarray(samples, dtype=np.float32), sample_rate)
