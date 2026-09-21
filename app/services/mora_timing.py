"""Real per-mora audio time windows for the learner's recording, derived
from the ASR model's own frame-level CTC decode instead of guessed from
equal time division (see app/services/pitch_extraction.py's
`extract_pitch_per_mora` for that simpler, ASR-free fallback).

## How it works

`KanaRecognizer.transcribe(..., with_timing=True)` (src/asr/inference.py)
returns, for each character of its recognized kana output, the audio time
span where that character's CTC decode "turned on". That is real timing,
but it is keyed to the RECOGNIZED string, which rarely has the exact same
characters as the TARGET string once you account for mispronunciations,
dropped morae, or extra sounds the model heard that aren't there.

This module bridges the two with the same Levenshtein alignment already
used for pronunciation scoring (src/asr/metrics.edit_distance's DP table,
generalized here to also report *matches*, not just edits -- scoring only
ever needed the edits, but a correctly-recognized mora is exactly the case
with the most trustworthy timing, and for a reasonably-paced recording
it's the majority case). Each target mora inherits its time window from
whichever recognized character(s) aligned to it; a target mora the ASR
did not align to anything at all (a genuine drop, or a recognition miss)
gets a linearly-interpolated placeholder window between its nearest
aligned neighbors, so every mora still gets *some* window to look for
pitch in.

## Limitation

CTC "spike" timing is where the model became *confident* about a symbol --
usually inside the right mora, but not a guaranteed true phonetic onset --
and a substitution borrows the wrong character's timing (the window used
for a mora the model misheard is wherever the character it *did* hear
spiked, which can drift a little from where the target mora was actually
said). Morae with no alignment at all fall back to interpolated
placeholders with no particular claim to accuracy, same as before. Still a
real, audio-grounded signal for anything the ASR *did* hear correctly,
which pure equal-time division never had.
"""

from __future__ import annotations

import numpy as np

from app.services.mora_diff import split_mora_spans


def _full_alignment(ref: list[str], hyp: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein backtrace including matches, not just edits.

    Same DP table as src.asr.metrics.edit_distance / edit_ops, but the
    traceback also emits a pair for every *matching* character.

    Returns (ref_idx, hyp_idx) pairs in ref order:
      (i, j)    - ref[i] matched, or was substituted for, hyp[j]
      (i, None) - ref[i] was deleted (no corresponding hyp character)
      (None, j) - hyp[j] was inserted (no corresponding ref character)
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return list(reversed(pairs))


def _interpolate_gaps(
    anchors: list[tuple[float, float] | None], duration: float
) -> list[tuple[float, float]]:
    """Fill unaligned morae with linearly-spaced placeholder windows between
    their nearest ASR-anchored neighbors (or the clip's own boundaries)."""
    n = len(anchors)
    known: list[tuple[int, float, float]] = [(-1, 0.0, 0.0)]
    known += [(i, a[0], a[1]) for i, a in enumerate(anchors) if a is not None]
    known.append((n, duration, duration))

    windows: list[tuple[float, float] | None] = list(anchors)
    for k in range(len(known) - 1):
        i1, _s1, e1 = known[k]
        i2, s2, _e2 = known[k + 1]
        gap_indices = list(range(i1 + 1, i2))
        if not gap_indices:
            continue
        lo, hi = e1, max(s2, e1)
        edges = np.linspace(lo, hi, len(gap_indices) + 1)
        for offset, idx in enumerate(gap_indices):
            windows[idx] = (float(edges[offset]), float(edges[offset + 1]))
    return [w if w is not None else (0.0, 0.0) for w in windows]


def _clean_windows(
    windows: list[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """Clamp to [0, duration] and force non-decreasing order.

    The alignment above is a heuristic, not a guarantee -- this is a last
    defensive pass so an unusual case (e.g. overlapping anchors from a
    tangled substitution pattern) can never hand the pitch extractor a
    window that runs backwards or outside the clip.
    """
    cleaned: list[tuple[float, float]] = []
    prev_end = 0.0
    for start, end in windows:
        start = min(max(start, prev_end), duration)
        end = min(max(end, start), duration)
        cleaned.append((start, end))
        prev_end = end
    return cleaned


def mora_time_windows(
    target_hiragana: str,
    recognized_kana: str,
    char_spans: list[tuple[float, float]],
    duration: float,
) -> list[tuple[float, float]]:
    """Real (or best-effort interpolated) time window per target mora.

    `recognized_kana` and `char_spans` must be the exact `kana` /
    `char_spans` pair returned together by one
    `KanaRecognizer.transcribe(..., with_timing=True)` call -- they are
    aligned to each other by list position. `target_hiragana` is the same
    normalized reading `/pitch-accent` splits into morae (see
    app.services.pitch_accent and app.services.normalization.to_hiragana).

    Returns exactly one window per target mora, in order, always covering
    [0, duration] with no gaps -- suitable to hand straight to
    app.services.pitch_extraction.extract_pitch_for_windows. Returns an
    empty list if `target_hiragana` has no morae.
    """
    spans = split_mora_spans(target_hiragana)
    if not spans:
        return []

    pairs = _full_alignment(list(target_hiragana), list(recognized_kana))
    ref_to_hyp = {r: h for r, h in pairs if r is not None and h is not None}

    anchors: list[tuple[float, float] | None] = []
    for _text, start_char, end_char in spans:
        hyp_indices = [
            ref_to_hyp[c]
            for c in range(start_char, end_char)
            if c in ref_to_hyp and 0 <= ref_to_hyp[c] < len(char_spans)
        ]
        if hyp_indices:
            starts = [char_spans[h][0] for h in hyp_indices]
            ends = [char_spans[h][1] for h in hyp_indices]
            anchors.append((min(starts), max(ends)))
        else:
            anchors.append(None)

    windows = _interpolate_gaps(anchors, duration)
    return _clean_windows(windows, duration)
