"""Speech / pause segmentation for the fluency head.

Prefers Silero VAD (tiny ONNX, CPU, free) when `silero-vad` is installed.
Falls back to an energy gate so /evaluate never depends on a download.

Japanese voiceless mora (っ, /s/, /k/) look like silence to a default
100ms VAD, so segments closer than MERGE_GAP_S are fused before we count
pauses. Leading / trailing file silence is ignored for pause_count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_silero_model = None
_silero_failed = False

# Gaps shorter than this are unvoiced consonants, not hesitation.
MERGE_GAP_S = 0.30


@dataclass(frozen=True)
class VadResult:
    speech_duration: float  # first speech → last speech (pace denominator)
    pause_count: int
    speech_ratio: float  # voiced seconds / file length (diagnostic only)
    method: str  # "silero" | "energy"


def _merge_segments(
    segments: list[tuple[float, float]], gap_s: float = MERGE_GAP_S
) -> list[tuple[float, float]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_s:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _from_segments(
    segments: list[tuple[float, float]], file_duration: float, method: str
) -> VadResult:
    merged = _merge_segments(segments)
    if not merged:
        return VadResult(0.0, 0, 0.0, method)
    voiced = sum(max(0.0, end - start) for start, end in merged)
    span = merged[-1][1] - merged[0][0]
    ratio = min(1.0, voiced / file_duration) if file_duration > 0 else 0.0
    return VadResult(max(0.0, span), max(0, len(merged) - 1), ratio, method)


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
        stamps = get_speech_timestamps(
            wav,
            _silero_model,
            sampling_rate=sample_rate,
            min_silence_duration_ms=int(MERGE_GAP_S * 1000),
            min_speech_duration_ms=80,
            speech_pad_ms=80,
        )
    except TypeError:
        # Older silero-vad without the extra kwargs.
        try:
            stamps = get_speech_timestamps(wav, _silero_model, sampling_rate=sample_rate)
        except Exception:  # noqa: BLE001
            logger.warning("Silero VAD inference failed; fluency uses energy VAD", exc_info=True)
            return None
    except Exception:  # noqa: BLE001
        logger.warning("Silero VAD inference failed; fluency uses energy VAD", exc_info=True)
        return None

    duration = len(samples) / sample_rate if sample_rate else 0.0
    segments: list[tuple[float, float]] = []
    for seg in stamps or []:
        start = seg["start"] / sample_rate if isinstance(seg, dict) else seg.start / sample_rate
        end = seg["end"] / sample_rate if isinstance(seg, dict) else seg.end / sample_rate
        if end > start:
            segments.append((start, end))
    return _from_segments(segments, duration, "silero")


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
    duration = len(samples) / sample_rate
    if not voiced.any():
        return VadResult(0.0, 0, 0.0, "energy")

    segments: list[tuple[float, float]] = []
    in_run = False
    run_start = 0
    for i, flag in enumerate(voiced):
        if flag and not in_run:
            in_run = True
            run_start = i
        elif not flag and in_run:
            segments.append((run_start * frame / sample_rate, i * frame / sample_rate))
            in_run = False
    if in_run:
        segments.append((run_start * frame / sample_rate, len(voiced) * frame / sample_rate))
    return _from_segments(segments, duration, "energy")


def analyze_vad(samples: np.ndarray, sample_rate: int) -> VadResult:
    """Speech span + in-utterance pause count. Never raises."""
    try:
        result = _try_silero(samples, sample_rate)
        if result is not None:
            return result
    except Exception:  # noqa: BLE001
        logger.warning("Silero VAD path crashed; fluency uses energy VAD", exc_info=True)
    return _energy_vad(np.asarray(samples, dtype=np.float32), sample_rate)
