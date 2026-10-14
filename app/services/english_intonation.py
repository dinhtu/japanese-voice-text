"""English sentence intonation: reference contour, learner measures, checks.

English is a stress language, not a pitch-accent one, so instead of a
per-word H/L step this follows the usual teaching model (and ToBI):

* every content word carries a pitch accent on its dictionary-stressed
  syllable (sentence stress, the "●" row);
* the LAST accent of each intonation phrase is the nucleus ("◉") -- the
  biggest pitch movement;
* after the nucleus the voice falls to the bottom of the range for a
  statement / wh-question, or rises for a yes/no question; a phrase that
  is not the last one ends part-way down (continuation);
* phrases are split at punctuation (, . ; : ! ? —).

Reference values are in semitones relative to the speaker's mean F0 --
the same scale as the learner curve from pitch_extraction -- so both lines
share one y axis. Learner-side checks, all from measured F0/energy on
syllable windows (CTC forced alignment of the target letters):

* peaks       -- each accented syllable is >= 1.5 st above the weak
                 syllables around it (or >= 0.5 st and >= 3 dB louder)
* word stress -- in each multi-syllable word, the most prominent syllable
                 (pitch + loudness + length) is the dictionary one
* breaks      -- a pause >= 150 ms (or a shorter one with a pitch drop /
                 reset) at each phrase boundary
* final       -- statement falls >= 2 st from the nucleus to the end
                 (question rises >= 2 st)

This module is numpy-only so it can be tested without torch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from app.services.english_lexicon import WordPronunciation, pronounce, word_ipa

# Weak / function words: no sentence stress in neutral read-aloud speech.
FUNCTION_WORDS = frozenset({
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "or",
    "but", "is", "are", "was", "were", "be", "been", "being", "am", "do",
    "does", "did", "have", "has", "had", "i", "you", "he", "she", "we",
    "they", "it", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "its", "with", "as", "from", "by", "that", "than",
    "this", "so", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "if", "into", "onto", "i'm", "it's", "you're",
    "we're", "they're", "he's", "she's", "i've", "i'll", "i'd", "let's",
    "there", "some",
})
_WH = frozenset({"what", "where", "when", "who", "whom", "whose", "why", "which", "how"})

_TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
_BREAK = re.compile(r"[,.;:!?—–]|\s-\s")

PEAK_MIN_ST = 1.5
PEAK_SOFT_ST = 0.5
PEAK_LOUDER_DB = 3.0
FINAL_MIN_ST = 2.0
BREAK_PAUSE_S = 0.15
BREAK_SHORT_PAUSE_S = 0.06
BREAK_PITCH_ST = 1.5
STRESS_TIE = 0.75
CREAK_MIN_S = 0.06


@dataclass(frozen=True)
class WordInfo:
    index: int
    text: str  # as written in the target
    phrase: int
    content: bool
    sentence_stress: bool
    nucleus: bool
    pron: WordPronunciation

    @property
    def ipa(self) -> str | None:
        return word_ipa(self.pron, mark_stress=self.content or len(self.pron.syllables) > 1)


@dataclass(frozen=True)
class RefPoint:
    word: int
    syllable: int
    role: str  # "weak" | "secondary" | "stress" | "nucleus"
    semitone: float
    end: float  # value at the end of the syllable (differs on a final fall/rise)
    lo: float
    hi: float


@dataclass(frozen=True)
class SentenceAnalysis:
    text: str
    words: tuple[WordInfo, ...]
    phrase_ends: tuple[str, ...]  # per phrase: "continue" | "fall" | "rise"
    reference: tuple[RefPoint, ...]

    @property
    def final_tone(self) -> str:
        return self.phrase_ends[-1] if self.phrase_ends else "fall"


def _final_tone(text: str, words: list[str]) -> str:
    stripped = text.strip()
    if stripped.endswith("?"):
        return "fall" if words and words[0].lower() in _WH else "rise"
    return "fall"


def analyze_sentence(text: str) -> SentenceAnalysis:
    """Words, phrases, stress roles and the reference contour for `text`."""
    tokens = list(_TOKEN.finditer(text))
    phrase = 0
    raw: list[tuple[str, int]] = []
    prev_end = 0
    for m in tokens:
        if raw and _BREAK.search(text[prev_end:m.start()]):
            phrase += 1
        raw.append((m.group(), phrase))
        prev_end = m.end()
    n_phrases = phrase + 1 if raw else 0

    words: list[WordInfo] = []
    for p in range(n_phrases):
        members = [(i, w) for i, (w, ph) in enumerate(raw) if ph == p]
        content = [w.lower().replace("’", "'") not in FUNCTION_WORDS for _i, w in members]
        nucleus_at = max((k for k, c in enumerate(content) if c), default=len(members) - 1)
        for k, (i, w) in enumerate(members):
            words.append(WordInfo(
                index=i,
                text=w,
                phrase=p,
                content=content[k],
                sentence_stress=content[k] or k == nucleus_at,
                nucleus=k == nucleus_at,
                pron=pronounce(w),
            ))

    final = _final_tone(text, [w for w, _p in raw])
    phrase_ends = tuple(
        final if p == n_phrases - 1 else "continue" for p in range(n_phrases)
    )
    analysis = SentenceAnalysis(text, tuple(words), phrase_ends, ())
    return SentenceAnalysis(text, tuple(words), phrase_ends, tuple(_reference(analysis)))


def _role(word: WordInfo, s: int) -> str:
    syl = word.pron.syllables[s]
    if s == word.pron.stress_index and word.sentence_stress:
        return "nucleus" if word.nucleus else "stress"
    if syl.stress == 2 and word.content:
        return "secondary"
    return "weak"


def _reference(analysis: SentenceAnalysis) -> list[RefPoint]:
    out: list[RefPoint] = []
    for p, ending in enumerate(analysis.phrase_ends):
        sylls = [
            (w, s, _role(w, s))
            for w in analysis.words if w.phrase == p
            for s in range(len(w.pron.syllables))
        ]
        n = len(sylls)
        nuc = next((k for k, (_w, _s, r) in enumerate(sylls) if r == "nucleus"), n - 1)
        values: list[float] = []
        for k, (_w, _s, role) in enumerate(sylls):
            base = 1.0 - 2.0 * (k / (n - 1) if n > 1 else 0.0)  # gentle declination
            values.append(base + {"weak": -1.0, "secondary": 0.5, "stress": 3.5, "nucleus": 4.5}[role])
        ends = list(values)
        if ending == "rise":
            values[nuc] = ends[nuc] = values[nuc] - 4.5  # low nucleus, then up
        # After the nucleus: glide to the phrase-final level.
        target = {"fall": -5.0, "rise": 5.0, "continue": -1.5}[ending]
        tail = list(range(nuc + 1, n))
        if tail:
            start = values[nuc]
            for j, k in enumerate(tail, start=1):
                values[k] = ends[k] = start + (target - start) * j / len(tail)
        else:
            ends[nuc] = target  # nucleus is the last syllable: move inside it
        for k, (w, s, role) in enumerate(sylls):
            band = 2.5 if role == "weak" else 2.0
            mid = (values[k] + ends[k]) / 2
            out.append(RefPoint(
                w.index, s, role, round(values[k], 2), round(ends[k], 2),
                round(min(values[k], ends[k], mid) - band, 2),
                round(max(values[k], ends[k], mid) + band, 2),
            ))
    return out


# ------------------------------------------------------------- learner side


@dataclass(frozen=True)
class PitchTrack:
    """Per-frame F0 (semitones vs the clip's mean F0, NaN if unvoiced),
    loudness in dB, and a speech mask (energy above the noise floor)."""

    semitone: np.ndarray
    energy_db: np.ndarray
    speech: np.ndarray
    frame_time: float


@dataclass(frozen=True)
class SyllableMeasure:
    word: int
    syllable: int
    start: float
    end: float
    semitone: float | None  # median over voiced frames
    start_semitone: float | None  # first third
    end_semitone: float | None  # last third
    peak: float | None  # 90th percentile
    energy_db: float | None
    speech_time: float = 0.0  # seconds of speech (not silence) in the window
    voiced_time: float = 0.0  # seconds with a measured F0

    @property
    def creaky(self) -> bool:
        """Clearly spoken but almost no F0: creak / vocal fry, which in
        English marks the bottom of a statement-final fall."""
        return self.speech_time >= CREAK_MIN_S and self.voiced_time < 0.3 * self.speech_time

    @property
    def voiced(self) -> bool:
        return self.semitone is not None


def _median(values: np.ndarray) -> float | None:
    values = values[~np.isnan(values)]
    return round(float(np.median(values)), 2) if values.size else None


def measure_syllables(
    track: PitchTrack,
    windows: list[list[tuple[float, float]]],
) -> list[SyllableMeasure]:
    """One measure per (word, syllable) window, in sentence order."""
    out = []
    n_frames = track.semitone.shape[0]
    for w, word_windows in enumerate(windows):
        for s, (start, end) in enumerate(word_windows):
            lo = min(n_frames, max(0, int(round(start / track.frame_time))))
            hi = min(n_frames, max(lo + 1, int(round(end / track.frame_time))))
            seg = track.semitone[lo:hi]
            third = max(1, (hi - lo) // 3)
            voiced = seg[~np.isnan(seg)]
            loud = track.energy_db[lo:hi]
            out.append(SyllableMeasure(
                word=w,
                syllable=s,
                start=round(start, 3),
                end=round(end, 3),
                semitone=_median(seg),
                start_semitone=_median(seg[:third]),
                end_semitone=_median(seg[-third:]),
                peak=round(float(np.percentile(voiced, 90)), 2) if voiced.size else None,
                energy_db=round(float(np.percentile(loud, 75)), 2) if loud.size else None,
                speech_time=round(float(np.sum(track.speech[lo:hi])) * track.frame_time, 3),
                voiced_time=round(voiced.size * track.frame_time, 3),
            ))
    return out


def longest_pause(track: PitchTrack, start: float, end: float) -> float:
    """Longest run of non-speech frames between two times, in seconds."""
    lo = max(0, int(round(start / track.frame_time)))
    hi = min(track.speech.shape[0], int(round(end / track.frame_time)))
    best = run = 0
    for frame in range(lo, hi):
        run = 0 if track.speech[frame] else run + 1
        best = max(best, run)
    return round(best * track.frame_time, 3)


@dataclass(frozen=True)
class PeakCheck:
    word: int
    syllable: int
    ok: bool | None
    delta: float | None  # st above the weak syllables around it


@dataclass(frozen=True)
class BreakCheck:
    after_word: int
    ok: bool | None
    pause: float | None
    drop: float | None  # st, phrase-end fall + next-phrase reset


@dataclass(frozen=True)
class FinalCheck:
    expected: str  # "fall" | "rise"
    ok: bool | None
    delta: float | None  # st, end minus nucleus (negative = fell)
    creaky: bool = False  # statement ended in creak (F0 below tracking)


@dataclass(frozen=True)
class WordStressCheck:
    word: int
    expected: int
    heard: int | None
    ok: bool | None
    accepted: tuple[int, ...] = ()  # every dictionary stress for this word


@dataclass(frozen=True)
class IntonationResult:
    analysis: SentenceAnalysis
    syllables: list[SyllableMeasure]
    peaks: list[PeakCheck]
    breaks: list[BreakCheck]
    final: FinalCheck | None
    word_stress: list[WordStressCheck]
    aligned: bool
    score: float | None = field(default=None)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _check_peaks(analysis, by_key, ref_by_key) -> list[PeakCheck]:
    out = []
    for p, ending in enumerate(analysis.phrase_ends):
        keys = [(r.word, r.syllable) for r in analysis.reference if _phrase_of(analysis, r.word) == p]
        for k, key in enumerate(keys):
            role = ref_by_key[key].role
            if role not in ("stress", "nucleus"):
                continue
            if role == "nucleus" and ending == "rise":
                continue  # a low-rise nucleus is judged by the final check
            m = by_key[key]
            here = m.peak if m.peak is not None else m.semitone
            if here is None:
                out.append(PeakCheck(*key, None, None))
                continue
            near = [
                by_key[keys[j]] for j in range(max(0, k - 2), min(len(keys), k + 3))
                if j != k and ref_by_key[keys[j]].role in ("weak", "secondary")
                and by_key[keys[j]].voiced
            ]
            if not near:
                near = [by_key[x] for x in keys if x != key and by_key[x].voiced]
            if not near:
                out.append(PeakCheck(*key, None, None))
                continue
            delta = here - float(np.median([n.semitone for n in near]))
            louder = None
            if m.energy_db is not None:
                levels = [n.energy_db for n in near if n.energy_db is not None]
                louder = m.energy_db - float(np.median(levels)) if levels else None
            ok = delta >= PEAK_MIN_ST or (
                delta >= PEAK_SOFT_ST and louder is not None and louder >= PEAK_LOUDER_DB
            )
            out.append(PeakCheck(*key, ok, round(delta, 2)))
    return out


def _phrase_of(analysis: SentenceAnalysis, word: int) -> int:
    return analysis.words[word].phrase


def _check_final(analysis, measures, ref_by_key) -> FinalCheck | None:
    if not analysis.phrase_ends:
        return None
    last = len(analysis.phrase_ends) - 1
    expected = analysis.final_tone
    seq = [m for m in measures if _phrase_of(analysis, m.word) == last]
    nuc = next(
        (k for k, m in enumerate(seq) if ref_by_key[(m.word, m.syllable)].role == "nucleus"),
        max(0, len(seq) - 1),
    )
    tail = [m for m in seq[nuc:] if m.voiced]
    creaky = expected == "fall" and bool(seq) and len(seq) - 1 >= nuc and seq[-1].creaky
    if not tail:
        return FinalCheck(expected, True if creaky else None, None, creaky)
    end = tail[-1].end_semitone if tail[-1].end_semitone is not None else tail[-1].semitone
    # Also measured against the phrase's middle: when the nucleus itself is
    # said low ("the RECORD is old" with no accent on "old"), the voice has
    # already come down before it and still ends at the bottom.
    level = float(np.median([m.semitone for m in seq if m.voiced]))
    if expected == "rise":
        bottom = min(min(v for v in (m.semitone, m.start_semitone) if v is not None) for m in tail)
        delta = max(end - bottom, end - level)
        return FinalCheck(expected, delta >= FINAL_MIN_ST, round(delta, 2))
    top = max(max(v for v in (m.peak, m.semitone, m.start_semitone) if v is not None) for m in tail)
    delta = min(end - top, end - level)
    return FinalCheck(expected, delta <= -FINAL_MIN_ST or creaky, round(delta, 2), creaky)


def _check_breaks(analysis, measures, pauses, aligned) -> list[BreakCheck]:
    out = []
    for p in range(len(analysis.phrase_ends) - 1):
        before = [m for m in measures if _phrase_of(analysis, m.word) == p and m.voiced]
        after = [m for m in measures if _phrase_of(analysis, m.word) == p + 1 and m.voiced]
        last_word = max(w.index for w in analysis.words if w.phrase == p)
        pause = pauses[last_word] if pauses and last_word < len(pauses) else None
        if not aligned or pause is None:
            out.append(BreakCheck(last_word, None, pause, None))
            continue
        drop = None
        if before and after:
            end = before[-1].end_semitone if before[-1].end_semitone is not None else before[-1].semitone
            top = max(m.peak or m.semitone for m in before)
            nxt = after[0].semitone
            # Fall into the boundary, or a jump back up after it.
            drop = max(top - end, nxt - end)
        ok = pause >= BREAK_PAUSE_S or (
            pause >= BREAK_SHORT_PAUSE_S and drop is not None and drop >= BREAK_PITCH_ST
        )
        out.append(BreakCheck(last_word, ok, pause, round(drop, 2) if drop is not None else None))
    return out


def _check_word_stress(analysis, measures) -> list[WordStressCheck]:
    out = []
    for w in analysis.words:
        if len(w.pron.syllables) < 2:
            continue
        sylls = [m for m in measures if m.word == w.index]
        expected = w.pron.stress_index
        accepted = w.pron.accepted_stress or (expected,)
        if len(sylls) != len(w.pron.syllables) or not any(m.voiced for m in sylls):
            out.append(WordStressCheck(w.index, expected, None, None, accepted))
            continue
        pitch = np.array([
            (m.peak if m.peak is not None else m.semitone) if m.voiced else np.nan for m in sylls
        ])
        floor = np.nanmin(pitch) - 1.0
        pitch = np.where(np.isnan(pitch), floor, pitch)
        loud = np.array([m.energy_db if m.energy_db is not None else np.nan for m in sylls])
        loud = np.where(np.isnan(loud), np.nanmin(loud) if np.any(~np.isnan(loud)) else 0.0, loud)
        # Speech frames only: the last syllable's window runs on into any
        # pause, which must not read as a long (stressed) vowel.
        length = np.log2(np.array([max(m.speech_time, 0.02) for m in sylls]))
        prom = (pitch - pitch.mean()) + 0.3 * (loud - loud.mean()) + 1.0 * (length - length.mean())
        heard = int(np.argmax(prom))
        ok = any(
            heard == idx or prom[idx] >= prom[heard] - STRESS_TIE
            for idx in accepted if idx < len(prom)
        )
        if heard in accepted:
            expected = heard  # show the reading the learner actually used
        out.append(WordStressCheck(w.index, expected, heard, bool(ok), accepted))
    return out


def _score(peaks, breaks, final, word_stress) -> float | None:
    parts: list[tuple[float, float]] = []
    peak_credit = [
        1.0 if c.ok else _clamp01((c.delta or 0.0) / PEAK_MIN_ST)
        for c in peaks if c.ok is not None
    ]
    if peak_credit:
        parts.append((float(np.mean(peak_credit)), 0.40))
    stress = [c.ok for c in word_stress if c.ok is not None]
    if stress:
        parts.append((sum(stress) / len(stress), 0.20))
    if final is not None and final.ok is not None:
        if final.ok:
            parts.append((1.0, 0.25))
        else:
            moved = -final.delta if final.expected == "fall" else final.delta
            parts.append((_clamp01((moved or 0.0) / FINAL_MIN_ST), 0.25))
    brk = [
        1.0 if c.ok else 0.6 * _clamp01((c.pause or 0.0) / BREAK_PAUSE_S)
        for c in breaks if c.ok is not None
    ]
    if brk:
        parts.append((float(np.mean(brk)), 0.15))
    if not parts:
        return None
    total = sum(w for _v, w in parts)
    return round(100.0 * sum(v * w for v, w in parts) / total, 2)


def evaluate_intonation(
    analysis: SentenceAnalysis,
    track: PitchTrack,
    windows: list[list[tuple[float, float]]],
    word_spans: list[tuple[float, float]] | None,
    aligned: bool,
) -> IntonationResult:
    """Measure the learner on syllable `windows` (one list per target word)
    and run the checks. `word_spans` (start, end) per word give the gaps
    searched for pauses; pass None when timing is only a rough split."""
    measures = measure_syllables(track, windows)
    by_key = {(m.word, m.syllable): m for m in measures}
    ref_by_key = {(r.word, r.syllable): r for r in analysis.reference}
    pauses = None
    if word_spans:
        pauses = [
            longest_pause(track, word_spans[i][1], word_spans[i + 1][0])
            for i in range(len(word_spans) - 1)
        ]
    peaks = _check_peaks(analysis, by_key, ref_by_key)
    breaks = _check_breaks(analysis, measures, pauses, aligned)
    final = _check_final(analysis, measures, ref_by_key)
    word_stress = _check_word_stress(analysis, measures)
    return IntonationResult(
        analysis, measures, peaks, breaks, final, word_stress, aligned,
        _score(peaks, breaks, final, word_stress),
    )
