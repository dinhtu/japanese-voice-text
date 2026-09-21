"""Learner's pitch (F0) per mora, for comparing against the reference pattern.

Extracts a per-frame fundamental-frequency estimate from a recording with
torchaudio's built-in pitch detector (normalized cross-correlation + median
smoothing -- no extra dependency beyond what the ASR pipeline already
needs), gates out frames that are not actually voiced speech with a
short-time energy threshold (the detector itself has no voiced/unvoiced
output and guesses a "pitch" even in silence), and converts Hz to
semitones relative to the *speaker's own* median pitch (so a low male
voice and a high female voice read as the same shape). Two ways to then
split that into one point per mora:

  extract_pitch_for_windows -- given REAL per-mora time windows (from
      app.services.mora_timing, which derives them from the ASR model's
      own CTC decode timing), just averages the semitone curve inside
      each one. This is the accurate path and is used whenever the ASR
      recognition needed to build those windows succeeded.

  extract_pitch_per_mora -- the ASR-free fallback: trims to the voiced
      span and buckets it into `num_morae` equal-time slices, each
      interior boundary nudged toward a nearby silence (see its own
      docstring). Used when ASR-based alignment isn't available (the
      recognizer failed, or produced something the alignment couldn't use)
      so there is always *some* pitch curve to show.

## Limitation

Neither path is a phoneme-accurate forced alignment. The windowed path is
only as good as the ASR's own recognition and CTC spike timing (see
app.services.mora_timing's docstring for specifics); the bucketed
fallback has no model behind it at all, just a heuristic preference for
real pauses over blind equal division. Good enough to compare overall
shape at a glance. See app/services/pitch_accent.py for the
(text -> expected pattern) half of the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class _PitchAnalysis:
    """Shared per-frame analysis both extraction paths aggregate over."""

    semitone: np.ndarray  # pitch in semitones relative to this clip's own median voiced pitch
    voiced: np.ndarray  # bool mask -- has real speech energy AND a detected pitch
    energy: np.ndarray  # short-time RMS energy (kept for extract_pitch_per_mora's boundary snapping)
    frame_time: float  # seconds per frame (== FRAME_TIME, kept alongside for clarity)


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


def _analyze(samples: np.ndarray, sample_rate: int) -> _PitchAnalysis | None:
    """Run the pitch detector once and gate/convert it, shared by both
    extraction paths below. Returns None if there's nothing to analyze
    (no audio, no frames, or no voiced speech detected at all)."""
    if samples.size == 0:
        return None

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
        return None

    energy = _frame_energy(samples, sample_rate, n_frames)
    threshold = max(0.25 * float(energy.max()), 1e-4) if energy.size else 1e-4
    voiced = (energy > threshold) & (pitch_hz > 0)
    if not np.any(voiced):
        return None

    reference_hz = float(np.median(pitch_hz[voiced]))
    semitone = 12.0 * np.log2(np.clip(pitch_hz, 1e-6, None) / reference_hz)
    return _PitchAnalysis(semitone=semitone, voiced=voiced, energy=energy, frame_time=FRAME_TIME)


def _aggregate(analysis: _PitchAnalysis, lo: int, hi: int) -> dict:
    """Mean semitone over the voiced frames of analysis.semitone[lo:hi]."""
    lo = max(0, lo)
    hi = max(lo + 1, min(hi, analysis.voiced.shape[0]))
    window_voiced = analysis.voiced[lo:hi]
    if np.any(window_voiced):
        value = float(np.mean(analysis.semitone[lo:hi][window_voiced]))
        return {"semitone": round(value, 2), "voiced": True}
    return {"semitone": None, "voiced": False}


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


def extract_pitch_for_windows(
    samples: np.ndarray,
    sample_rate: int,
    windows: list[tuple[float, float]],
) -> list[dict]:
    """Return one pitch sample per `windows` entry, using REAL per-mora time
    windows (app.services.mora_timing.mora_time_windows) instead of a
    guessed equal-time split.

    Each point is {"semitone": float | None, "voiced": bool}. `windows` is
    a list of (start_sec, end_sec) pairs, one per target mora, in order --
    always returns a list the same length as `windows`, so the frontend can
    zip it positionally against the reference pattern without a length
    check.
    """
    empty = [{"semitone": None, "voiced": False} for _ in windows]
    if not windows:
        return empty

    analysis = _analyze(samples, sample_rate)
    if analysis is None:
        return empty

    points: list[dict] = []
    for start_sec, end_sec in windows:
        lo = int(round(start_sec / analysis.frame_time))
        hi = int(round(end_sec / analysis.frame_time))
        points.append(_aggregate(analysis, lo, hi))
    return points


def extract_pitch_per_mora(
    samples: np.ndarray,
    sample_rate: int,
    num_morae: int,
) -> list[dict]:
    """Return exactly `num_morae` pitch samples, one per target mora, using
    equal-time buckets nudged toward real pauses (see module docstring --
    this is the ASR-free fallback; prefer extract_pitch_for_windows when
    real per-mora timing from the ASR model is available).

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
    if num_morae <= 0:
        return empty

    analysis = _analyze(samples, sample_rate)
    if analysis is None:
        return empty

    # Trim to the voiced span so leading/trailing silence (very common in a
    # mic recording) does not compress where the actual speech lands.
    voiced_indices = np.flatnonzero(analysis.voiced)
    start, end = int(voiced_indices[0]), int(voiced_indices[-1]) + 1

    bucket_edges = _pause_aware_bucket_edges(analysis.energy, start, end, num_morae)
    return [
        _aggregate(analysis, int(bucket_edges[i]), int(bucket_edges[i + 1]))
        for i in range(num_morae)
    ]
