"""Learner's pitch (F0) per mora, for comparing against the reference pattern.

Extracts a per-frame fundamental-frequency estimate from a recording with
torchaudio's built-in pitch detector (normalized cross-correlation + median
smoothing -- no extra dependency beyond what the ASR pipeline already
needs), gates out frames that are not actually voiced speech with a
short-time energy threshold (the detector itself has no voiced/unvoiced
output and guesses a "pitch" even in silence), converts Hz to semitones
relative to the *speaker's own* median pitch (so a low male voice and a
high female voice read as the same shape), trims off leading/trailing
silence, and buckets what is left into exactly `num_morae` equal-time
slices -- one per mora of the target text, in order -- so the frontend can
plot the learner's curve on the *same x positions* as the reference
chart's mora columns.

## Limitation

Bucketing by equal time slices across the trimmed voiced span is not a
real forced alignment: there is no model tying a specific moment of audio
to a specific mora. A dropped or inserted mora (or very uneven pacing)
shifts every bucket after it. Good enough to compare overall shape at a
glance; not a phoneme-accurate score. See app/services/pitch_accent.py
for the (text -> expected pattern) half of the comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

FRAME_TIME = 0.01  # seconds between analysis frames
FREQ_LOW = 75.0  # Hz -- below the lowest realistic adult voice
FREQ_HIGH = 500.0  # Hz -- above the highest realistic adult voice fundamental


def _frame_energy(samples: np.ndarray, sample_rate: int, n_frames: int) -> np.ndarray:
    """Short-time RMS energy, one value per pitch-detector frame.

    detect_pitch_frequency has no voiced/unvoiced output of its own, so this
    is what tells actual speech apart from silence/noise between words.
    """
    hop = max(1, round(sample_rate * FRAME_TIME))
    win = hop * 2
    energy = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        window = samples[start : start + win]
        if window.size:
            energy[i] = float(np.sqrt(np.mean(np.square(window))))
    return energy


def extract_pitch_per_mora(
    samples: np.ndarray,
    sample_rate: int,
    num_morae: int,
) -> list[dict]:
    """Return exactly `num_morae` pitch samples, one per target mora.

    Each point is {"semitone": float | None, "voiced": bool}, in the same
    order as the target text's morae (app.services.pitch_accent's output).
    `semitone` is relative to this recording's own median voiced pitch;
    None where that mora's time slice had no voiced speech.

    `samples` must be mono float audio at `sample_rate`. Always returns a
    list of exactly `num_morae` items (all unvoiced placeholders if there
    is no audio or no voiced speech at all), so the frontend can zip it
    positionally against the reference pattern without a length check.
    """
    empty = [{"semitone": None, "voiced": False} for _ in range(max(num_morae, 0))]
    if num_morae <= 0 or samples.size == 0:
        return empty

    waveform = torch.from_numpy(np.ascontiguousarray(samples)).float().unsqueeze(0)
    pitch_hz = (
        torchaudio.functional.detect_pitch_frequency(
            waveform,
            sample_rate,
            frame_time=FRAME_TIME,
            freq_low=FREQ_LOW,
            freq_high=FREQ_HIGH,
        )
        .squeeze(0)
        .numpy()
    )
    n_frames = pitch_hz.shape[0]
    if n_frames == 0:
        return empty

    energy = _frame_energy(samples, sample_rate, n_frames)
    threshold = max(0.25 * float(energy.max()), 1e-4) if energy.size else 1e-4
    voiced = (energy > threshold) & (pitch_hz > 0)
    if not np.any(voiced):
        return empty

    reference_hz = float(np.median(pitch_hz[voiced]))
    semitone = 12.0 * np.log2(np.clip(pitch_hz, 1e-6, None) / reference_hz)

    # Trim to the voiced span so leading/trailing silence (very common in a
    # mic recording) does not compress where the actual speech lands.
    voiced_indices = np.flatnonzero(voiced)
    start, end = int(voiced_indices[0]), int(voiced_indices[-1]) + 1

    bucket_edges = np.linspace(start, end, num_morae + 1).round().astype(int)
    points: list[dict] = []
    for i in range(num_morae):
        lo, hi = bucket_edges[i], max(bucket_edges[i] + 1, bucket_edges[i + 1])
        window_voiced = voiced[lo:hi]
        if np.any(window_voiced):
            value = float(np.mean(semitone[lo:hi][window_voiced]))
            points.append({"semitone": round(value, 2), "voiced": True})
        else:
            points.append({"semitone": None, "voiced": False})
    return points
