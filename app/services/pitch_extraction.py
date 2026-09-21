"""Learner's pitch (F0) per mora, for comparing against the reference pattern.

Extracts a per-frame fundamental-frequency estimate from a recording with
torchaudio's built-in pitch detector (normalized cross-correlation + median
smoothing -- no extra dependency beyond what the ASR pipeline already
needs), gates out frames that are not actually voiced speech with a
short-time energy threshold (the detector itself has no voiced/unvoiced
output and guesses a "pitch" even in silence), converts Hz to semitones
relative to the *speaker's own* median pitch (so a low male voice and a
high female voice read as the same shape), trims off leading/trailing
silence, and buckets what is left into exactly `num_morae` slices -- one
per mora of the target text, in order -- so the frontend can plot the
learner's curve on the *same x positions* as the reference chart's mora
columns.

## Bucket boundaries: equal time, nudged toward real pauses

A pure equal-time split assumes every mora takes the same amount of time,
which is wrong exactly where it matters most: a sokuon (small tsu, "cl")
is a brief silence, a chouon (long vowel, "-") is a held tone, and
real speech is never perfectly evenly paced. So each *interior* boundary
starts at its equal-time position and then searches a small window around
it for a genuine dip in short-time energy (a pause between syllables) and
snaps to that instead -- only when the dip is clearly quieter than its
surroundings, so a smoothly-voiced stretch with no real gap is left at its
equal-time split rather than snapping to whatever's marginally quietest.

## Limitation

This is still not a real forced alignment: there is no model tying a
specific moment of audio to a specific mora, just a heuristic that prefers
actual silences over blind equal division. A dropped or inserted mora (or
very uneven pacing without any pause between syllables) can still shift
buckets. Good enough to compare overall shape at a glance; not a
phoneme-accurate score. See app/services/pitch_accent.py for the
(text -> expected pattern) half of the comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

FRAME_TIME = 0.01  # seconds between analysis frames
FREQ_LOW = 75.0  # Hz -- below the lowest realistic adult voice
FREQ_HIGH = 500.0  # Hz -- above the highest realistic adult voice fundamental

# How far (as a fraction of a mora's average equal-time width) an interior
# bucket boundary is allowed to drift while looking for a real pause.
_SNAP_SEARCH_FRACTION = 0.4
# A boundary only snaps to a dip when it is at least this much quieter than
# the local mean energy around it -- otherwise it's just noise, not a pause.
_SNAP_DIP_RATIO = 0.7


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


def _snap_to_pause(energy: np.ndarray, naive_idx: int, lo: int, hi: int, radius: int) -> int:
    """Nudge one interior bucket boundary toward a nearby energy dip.

    Searches `energy[naive_idx - radius : naive_idx + radius]` (clipped to
    `[lo, hi]`) for its quietest frame and snaps there, but only if that
    frame is clearly quieter than the window's average -- a real pause,
    not just the naturally lowest point of an otherwise steady sound.
    Falls back to `naive_idx` untouched when no such dip exists nearby.
    """
    window_lo = max(lo, naive_idx - radius)
    window_hi = min(hi, naive_idx + radius) + 1
    if window_hi - window_lo < 2:
        return naive_idx
    window = energy[window_lo:window_hi]
    local_mean = float(np.mean(window))
    if local_mean <= 0:
        return naive_idx
    min_offset = int(np.argmin(window))
    min_val = float(window[min_offset])
    if min_val > _SNAP_DIP_RATIO * local_mean:
        return naive_idx
    return window_lo + min_offset


def _pause_aware_bucket_edges(energy: np.ndarray, start: int, end: int, num_morae: int) -> np.ndarray:
    """Equal-time bucket edges, each interior one nudged toward a real pause."""
    edges = np.linspace(start, end, num_morae + 1).round().astype(int)
    if num_morae > 1:
        avg_width = (end - start) / num_morae
        radius = max(1, round(avg_width * _SNAP_SEARCH_FRACTION))
        for i in range(1, num_morae):
            edges[i] = _snap_to_pause(energy, int(edges[i]), start, end, radius)
        # Snapping can push neighboring edges out of order (e.g. two
        # boundaries pulled toward the same short pause) -- restore a
        # strictly increasing sequence by clamping forward then backward,
        # each pass never moving an edge further than a snap already did.
        for i in range(1, num_morae):
            edges[i] = max(edges[i], edges[i - 1] + 1)
        for i in range(num_morae - 1, 0, -1):
            edges[i] = min(edges[i], edges[i + 1] - 1)
    edges[0], edges[-1] = start, end
    return edges


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

    bucket_edges = _pause_aware_bucket_edges(energy, start, end, num_morae)
    points: list[dict] = []
    for i in range(num_morae):
        lo, hi = int(bucket_edges[i]), max(int(bucket_edges[i]) + 1, int(bucket_edges[i + 1]))
        window_voiced = voiced[lo:hi]
        if np.any(window_voiced):
            value = float(np.mean(semitone[lo:hi][window_voiced]))
            points.append({"semitone": round(value, 2), "voiced": True})
        else:
            points.append({"semitone": None, "voiced": False})
    return points
