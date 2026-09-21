"""Learner's pitch (F0) per mora, for comparing against the reference pattern.

Extracts a per-frame fundamental-frequency estimate from a recording with
torchaudio's built-in pitch detector (normalized cross-correlation + median
smoothing -- no extra dependency beyond what the ASR pipeline already
needs), gates out frames that are not actually voiced speech with a
short-time energy threshold (the detector itself has no voiced/unvoiced
output and guesses a "pitch" even in silence). Two ways to then split that
into one point per mora:

  extract_pitch_for_windows -- given REAL per-mora time windows (from
      app.services.mora_timing, which derives them from the ASR model's
      own CTC decode timing), averages the pitch curve inside each one and
      converts it to semitones. This is the accurate path and is used
      whenever the ASR recognition needed to build those windows
      succeeded.

  extract_pitch_per_mora -- the ASR-free fallback: trims to the voiced
      span and buckets it into `num_morae` equal-time slices, each
      interior boundary nudged toward a nearby silence (see its own
      docstring). Used when ASR-based alignment isn't available.

## Semitones are relative to a reference pitch -- and which reference
   matters a lot

Hz alone doesn't compare across speakers (a low male voice and a high
female voice need different baselines), so pitch is expressed as
semitones above/below a reference: `12 * log2(f0 / reference_hz)`. The
obvious choice is the *whole clip's* own median voiced pitch -- simple,
and what `extract_pitch_per_mora` still does.

But the reference H/L pattern from app.services.pitch_accent is not a
"top half of my range vs bottom half" signal -- it's a *per accent-phrase*
pattern: each phrase (a word or short run of words with its own pitch
contour) restarts its own High/Low shape, and Japanese speech has natural
*declination* -- pitch drifts gradually downward across a sentence even
with no accent change at all. A later phrase's High can sit at a lower
raw Hz than an earlier phrase's Low purely from declination, which is
completely normal speech and not a pronunciation error. Comparing every
mora in the sentence against one whole-clip median mixes that normal
downward drift into the comparison and can make an accurately-pronounced
recording look like it doesn't match the reference shape at all.

`extract_pitch_for_windows` fixes this when it's given each mora's accent
*phrase index* and expected H/L *label* (`app.services.pitch_accent.
MoraPitch.phrase` / `.pitch`, the same fields `/pitch-accent` already
returns): for each phrase it takes the geometric mean of that phrase's own
H-labeled morae's median Hz and its own L-labeled morae's median Hz, and
uses that as the phrase's reference pitch. In semitone terms that puts the
reference exactly halfway between the phrase's own High and Low targets,
so a High mora reads as clearly positive and a Low mora as clearly
negative *within that phrase* regardless of how far declination has
dragged the phrase's absolute pitch down -- a plain per-phrase median
would instead skew toward whichever label (H or L) happens to cover more
frames in that phrase. A phrase with only one label present, or no voiced
audio at all, falls back to that phrase's own overall median, then to the
whole clip's median, so it still gets *some* answer.

## Limitation

Neither extraction path is a phoneme-accurate forced alignment, and
per-phrase normalization is still a median-based heuristic, not a real
declination model -- an unusually short or heavily-devoiced phrase can
give a noisy local reference. Good enough to compare overall shape at a
glance. See app/services/pitch_accent.py for the (text -> expected
pattern) half of the comparison, and app/services/mora_timing.py for
where the real per-mora time windows come from.
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
    """Shared per-frame analysis both extraction paths aggregate over.

    Kept as raw Hz (not semitones) because the two extraction paths
    disagree on what "the reference pitch" is: one whole-clip median, or
    one median per accent phrase -- see module docstring.
    """

    pitch_hz: np.ndarray
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
    """Run the pitch detector once and gate it, shared by both extraction
    paths below. Returns None if there's nothing to analyze (no audio, no
    frames, or no voiced speech detected at all)."""
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

    return _PitchAnalysis(pitch_hz=pitch_hz, voiced=voiced, energy=energy, frame_time=FRAME_TIME)


def _semitone(pitch_hz: np.ndarray, reference_hz: float) -> np.ndarray:
    return 12.0 * np.log2(np.clip(pitch_hz, 1e-6, None) / reference_hz)


def _aggregate(analysis: _PitchAnalysis, lo: int, hi: int, reference_hz: float) -> dict:
    """Mean semitone (relative to `reference_hz`) over the voiced frames of
    analysis.pitch_hz[lo:hi]."""
    lo = max(0, lo)
    hi = max(lo + 1, min(hi, analysis.voiced.shape[0]))
    window_voiced = analysis.voiced[lo:hi]
    if np.any(window_voiced):
        semitone = _semitone(analysis.pitch_hz[lo:hi][window_voiced], reference_hz)
        return {"semitone": round(float(np.mean(semitone)), 2), "voiced": True}
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


def _phrase_reference_hz(
    analysis: _PitchAnalysis,
    frame_windows: list[tuple[int, int]],
    phrases: list[int],
    pitch_labels: list[str],
    global_reference_hz: float,
) -> list[float]:
    """One reference pitch (Hz) per point: the geometric mean of its own
    accent phrase's H-labeled and L-labeled median Hz (the semitone
    midpoint between the phrase's own High and Low targets -- see module
    docstring for why that's the right center, not a plain phrase median).

    Falls back to that phrase's own overall median when only one label is
    present in it, and to `global_reference_hz` when the phrase has no
    voiced audio at all.
    """

    def _voiced_hz(frame_indices: list[int]) -> np.ndarray:
        if not frame_indices:
            return np.array([])
        idx = np.array(frame_indices, dtype=int)
        return analysis.pitch_hz[idx][analysis.voiced[idx]]

    by_phrase_all: dict[int, list[int]] = {}
    by_phrase_h: dict[int, list[int]] = {}
    by_phrase_l: dict[int, list[int]] = {}
    for (lo, hi), phrase_id, label in zip(frame_windows, phrases, pitch_labels):
        lo = max(0, lo)
        hi = min(analysis.voiced.shape[0], hi)
        if hi <= lo:
            continue
        frames = list(range(lo, hi))
        by_phrase_all.setdefault(phrase_id, []).extend(frames)
        (by_phrase_h if label == "H" else by_phrase_l).setdefault(phrase_id, []).extend(frames)

    phrase_ref: dict[int, float] = {}
    for phrase_id in set(phrases):
        h_hz = _voiced_hz(by_phrase_h.get(phrase_id, []))
        l_hz = _voiced_hz(by_phrase_l.get(phrase_id, []))
        if h_hz.size and l_hz.size:
            phrase_ref[phrase_id] = float(np.sqrt(float(np.median(h_hz)) * float(np.median(l_hz))))
            continue
        all_hz = _voiced_hz(by_phrase_all.get(phrase_id, []))
        if all_hz.size:
            phrase_ref[phrase_id] = float(np.median(all_hz))

    return [phrase_ref.get(p, global_reference_hz) for p in phrases]


def extract_pitch_for_windows(
    samples: np.ndarray,
    sample_rate: int,
    windows: list[tuple[float, float]],
    phrases: list[int] | None = None,
    pitch_labels: list[str] | None = None,
) -> list[dict]:
    """Return one pitch sample per `windows` entry, using REAL per-mora time
    windows (app.services.mora_timing.mora_time_windows) instead of a
    guessed equal-time split.

    `phrases` and `pitch_labels`, when both given, must be the same length
    as `windows`: each mora's 0-based accent-phrase index and expected
    "H"/"L" label (app.services.pitch_accent.MoraPitch.phrase / .pitch).
    Each point is then expressed in semitones relative to *its own
    phrase's* H/L-midpoint reference rather than the whole clip's median --
    see the module docstring for why that matters. Without both, every
    point uses the whole clip's median, same as extract_pitch_per_mora.

    Each point is {"semitone": float | None, "voiced": bool}. Always
    returns a list the same length as `windows`, so the frontend can zip
    it positionally against the reference pattern without a length check.
    """
    empty = [{"semitone": None, "voiced": False} for _ in windows]
    if not windows:
        return empty

    analysis = _analyze(samples, sample_rate)
    if analysis is None:
        return empty

    global_reference_hz = float(np.median(analysis.pitch_hz[analysis.voiced]))

    frame_windows = [
        (int(round(start_sec / analysis.frame_time)), int(round(end_sec / analysis.frame_time)))
        for start_sec, end_sec in windows
    ]

    if (
        phrases is not None
        and pitch_labels is not None
        and len(phrases) == len(windows)
        and len(pitch_labels) == len(windows)
    ):
        reference_per_point = _phrase_reference_hz(
            analysis, frame_windows, phrases, pitch_labels, global_reference_hz
        )
    else:
        reference_per_point = [global_reference_hz] * len(windows)

    return [
        _aggregate(analysis, lo, hi, ref_hz)
        for (lo, hi), ref_hz in zip(frame_windows, reference_per_point)
    ]


def extract_pitch_per_mora(
    samples: np.ndarray,
    sample_rate: int,
    num_morae: int,
) -> list[dict]:
    """Return exactly `num_morae` pitch samples, one per target mora, using
    equal-time buckets nudged toward real pauses (see module docstring --
    this is the ASR-free fallback; prefer extract_pitch_for_windows when
    real per-mora timing from the ASR model is available). Semitones here
    are always relative to the whole clip's own median -- no phrase
    information is available for this fallback path.

    Each point is {"semitone": float | None, "voiced": bool}, in the same
    order as the target text's morae (app.services.pitch_accent's output).
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

    reference_hz = float(np.median(analysis.pitch_hz[analysis.voiced]))

    # Trim to the voiced span so leading/trailing silence (very common in a
    # mic recording) does not compress where the actual speech lands.
    voiced_indices = np.flatnonzero(analysis.voiced)
    start, end = int(voiced_indices[0]), int(voiced_indices[-1]) + 1

    bucket_edges = _pause_aware_bucket_edges(analysis.energy, start, end, num_morae)
    return [
        _aggregate(analysis, int(bucket_edges[i]), int(bucket_edges[i + 1]), reference_hz)
        for i in range(num_morae)
    ]
