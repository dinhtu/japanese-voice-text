"""Word / syllable time windows for the English target, from wav2vec2 CTC.

The target letters (not what ASR decoded) are force-aligned onto the
wav2vec2-base-960h emissions with the same Viterbi the Japanese page uses
(src/asr/force_align.py). wav2vec2 letters are spiky -- each fires on one
or two frames near the start of its sound -- so a syllable window runs
from its first letter's spike to the next syllable's first spike; the last
syllable of a word runs to the next word's first spike (capped), which
covers the vowel and coda where the F0 actually is.

When alignment is impossible (no emissions, clip shorter than the
target) the speech span is split in proportion to letter counts and the
result is flagged `aligned=False` so pause-based checks are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.english_intonation import PitchTrack, SentenceAnalysis

# Longest stretch after a word's last letter spike still counted as that
# word's final syllable (a long vowel + coda; anything after is a pause).
_TAIL_S = 0.35


@dataclass(frozen=True)
class SyllableTiming:
    windows: list[list[tuple[float, float]]]  # per word, per syllable
    word_spans: list[tuple[float, float]] | None  # first..last letter spike
    aligned: bool
    # Words whose letters the forced path could only place on frames where
    # the model hears something else: the learner skipped them.
    missing: frozenset[int] = frozenset()
    confidence: tuple[float, ...] = ()  # mean best log-prob per word
    letter_confidence: tuple[tuple[float, ...], ...] = ()  # best log-prob per letter


# Mean (over a word's letters) of the best frame log-prob on the forced
# path. Measured on wav2vec2-base-960h: said words sit at 0 .. -2.2, said
# but misheard ones ("song" heard as SUM) around -4.3, skipped words at
# -6.3 .. -13.5.
MISSING_LOGPROB = -5.0
# At or above this the word counts as said even when greedy decoding
# spelled it by ear ("KAT", "HEMORICH"): below the said-but-misheard band
# on purpose, so near misses pass; skipped words stay under MISSING_LOGPROB.
HEARD_LOGPROB = -4.5
# ...and most of its letters must really be there: a good mean can hide a
# missing tail ("weekend" read as "week" averages about -4.3).
HEARD_LETTER_LOGPROB = -6.0
HEARD_LETTER_SHARE = 0.75
# More than this share of missing words: the timing itself is unreliable.
MAX_MISSING_SHARE = 0.5


def heard_words(timing: SyllableTiming | None) -> frozenset[int]:
    """Target word indices the forced path found clearly in the audio."""
    if timing is None:
        return frozenset()
    out = set()
    for i, (mean, letters) in enumerate(zip(timing.confidence, timing.letter_confidence)):
        found = sum(1 for v in letters if v >= HEARD_LETTER_LOGPROB)
        if letters and mean >= HEARD_LOGPROB and found >= HEARD_LETTER_SHARE * len(letters):
            out.add(i)
    return frozenset(out)


def _syllable_letter_counts(analysis: SentenceAnalysis) -> list[list[int]]:
    """Spoken letters per syllable; always sums to the word's spoken letters
    (display chunks of "café" or "U.S." may not map one-to-one)."""
    out = []
    for w in analysis.words:
        counts = [len(s.text.replace("'", "").replace(".", "")) for s in w.pron.syllables]
        total = len(w.spoken) or sum(counts)
        if sum(counts) != total:
            n = len(counts)
            edges = [round(total * k / n) for k in range(n + 1)]
            counts = [edges[k + 1] - edges[k] for k in range(n)]
        out.append(counts)
    return out


def _letter_targets(analysis: SentenceAnalysis, vocab: list[str]):
    """CTC target ids and, per word, the index of each of its letters."""
    stoi = {tok: i for i, tok in enumerate(vocab)}
    sep = stoi.get("|")
    ids: list[int] = []
    letter_pos: list[list[int]] = []
    for w_i, word in enumerate(analysis.words):
        if w_i and sep is not None:
            ids.append(sep)
        positions = []
        for ch in (word.spoken or word.text).upper():
            if not ch.isalpha():
                continue
            tok = stoi.get(ch)
            if tok is None:
                return None
            positions.append(len(ids))
            ids.append(tok)
        letter_pos.append(positions)
    return ids, letter_pos


def align_syllables(
    analysis: SentenceAnalysis,
    log_probs: np.ndarray | None,
    vocab: list[str] | None,
    duration: float,
) -> SyllableTiming | None:
    """Forced-align the target letters; None when it cannot be done."""
    if log_probs is None or not vocab or duration <= 0 or not analysis.words:
        return None
    built = _letter_targets(analysis, vocab)
    if built is None:
        return None
    ids, letter_pos = built
    blank = vocab.index("<pad>") if "<pad>" in vocab else 0

    from src.asr.force_align import _fill_missing, _viterbi_token_frames

    raw_spans = _viterbi_token_frames(log_probs, ids, blank=blank)
    if raw_spans is None:
        return None
    confidence = []
    letter_confidence = []
    for pos in letter_pos:
        best = [
            float(np.max(log_probs[raw_spans[i][0]:raw_spans[i][1], ids[i]]))
            if raw_spans[i] is not None else -30.0
            for i in pos
        ]
        confidence.append(round(float(np.mean(best)), 2) if best else -30.0)
        letter_confidence.append(tuple(round(b, 2) for b in best))
    missing = frozenset(w for w, c in enumerate(confidence) if c < MISSING_LOGPROB)
    spans = _fill_missing(raw_spans, log_probs.shape[0])
    frame_time = duration / log_probs.shape[0]
    starts = [s * frame_time for s, _e in spans]
    ends = [e * frame_time for _s, e in spans]

    counts = _syllable_letter_counts(analysis)
    word_spans = [
        (starts[pos[0]], ends[pos[-1]]) if pos else (0.0, 0.0) for pos in letter_pos
    ]
    windows: list[list[tuple[float, float]]] = []
    for w_i, pos in enumerate(letter_pos):
        next_start = word_spans[w_i + 1][0] if w_i + 1 < len(word_spans) else duration
        tail_end = min(next_start, word_spans[w_i][1] + _TAIL_S)
        firsts = []
        offset = 0
        for n in counts[w_i]:
            firsts.append(starts[pos[min(offset, len(pos) - 1)]])
            offset += n
        word_windows = []
        for k, start in enumerate(firsts):
            end = firsts[k + 1] if k + 1 < len(firsts) else tail_end
            word_windows.append((start, max(end, start + frame_time)))
        windows.append(word_windows)
    aligned = len(missing) <= MAX_MISSING_SHARE * len(letter_pos)
    return SyllableTiming(
        windows, word_spans if aligned else None, aligned, missing, tuple(confidence),
        tuple(letter_confidence),
    )


def proportional_syllables(analysis: SentenceAnalysis, track: PitchTrack) -> SyllableTiming:
    """Fallback: split the speech span by letter counts (no pause checks)."""
    counts = _syllable_letter_counts(analysis)
    speech = np.flatnonzero(track.speech)
    if speech.size:
        start, end = speech[0] * track.frame_time, (speech[-1] + 1) * track.frame_time
    else:
        start, end = 0.0, track.speech.shape[0] * track.frame_time
    total = sum(max(1, n) for word in counts for n in word) or 1
    step = (end - start) / total
    windows, t = [], start
    for word in counts:
        word_windows = []
        for n in word:
            word_windows.append((t, t + max(1, n) * step))
            t += max(1, n) * step
        windows.append(word_windows)
    return SyllableTiming(windows, None, False)
