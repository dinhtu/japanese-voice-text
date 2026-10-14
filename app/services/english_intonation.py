"""English sentence intonation: reference contour, learner measures, checks.

English is a stress language, not a pitch-accent one, so instead of a
per-word H/L step this follows the usual teaching model (and ToBI):

* every content word carries a pitch accent on its dictionary-stressed
  syllable (sentence stress, the "●" row);
* the LAST accent of each intonation phrase is the nucleus ("◉") -- the
  biggest pitch movement;
* each phrase ends by the punctuation that closes it: a statement, wh- or
  alternative ("tea or coffee?") question falls to the bottom of the range,
  a yes/no question rises, a tag question ("..., isn't it?") may do either,
  and a phrase closed by , ; : or a dash ends part-way down (continuation);
* phrases are split at punctuation that is followed by a space or the end
  -- not inside "3.50" or "U.S." -- after numbers / titles are written out
  as spoken (english_text.speech_tokens).

Reference values are in semitones relative to the speaker's mean F0 --
the same scale as the learner curve from pitch_extraction -- so both lines
share one y axis. Learner-side checks, all from measured F0/energy on
syllable windows (CTC forced alignment of the target letters):

* peaks       -- each accented syllable is >= 1.5 st above the weak
                 syllables around it (or >= 0.5 st and >= 3 dB louder)
* word stress -- in each multi-syllable word, the most prominent syllable
                 (pitch + loudness + voiced length) is a dictionary one
* breaks      -- a pause >= 150 ms (or a shorter one with a pitch drop /
                 reset) at each phrase boundary
* ends        -- every sentence-final phrase falls (or rises) >= 2 st
                 from its nucleus / its middle level to its last voiced frames

Words the alignment says were not said (`missing`) fail their checks.
This module is numpy-only so it can be tested without torch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from app.services.english_lexicon import WordPronunciation, pronounce, word_ipa
from app.services.english_text import SpeechToken, speech_tokens

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
PRONOUNS = frozenset({
    "i", "you", "he", "she", "we", "they", "it", "me", "him", "her", "us",
    "them", "one", "this", "that",
})
_WH = frozenset({"what", "where", "when", "who", "whom", "whose", "why", "which", "how"})
_DISCOURSE = frozenset({"so", "well", "and", "but", "oh", "ok", "okay", "now", "then", "hey", "and"})
_TAG_AUX = frozenset({
    "is", "isn't", "are", "aren't", "was", "wasn't", "were", "weren't", "do",
    "don't", "does", "doesn't", "did", "didn't", "can", "can't", "could",
    "couldn't", "will", "won't", "would", "wouldn't", "have", "haven't",
    "has", "hasn't", "had", "hadn't", "should", "shouldn't", "am", "aren't",
})

# Punctuation that ends a phrase: followed by a space, a closing quote /
# bracket, or the next word -- never inside a number or "U.S.".
_BREAK = re.compile(r"[,.;:!?—–…](?=[\s\"'”’)\]]|$)|\s[-–—]\s")

PEAK_MIN_ST = 1.5
PEAK_SOFT_ST = 0.5
PEAK_LOUDER_DB = 3.0
FINAL_MIN_ST = 2.0
BREAK_PAUSE_S = 0.15
BREAK_SHORT_PAUSE_S = 0.06
BREAK_PITCH_ST = 1.5
STRESS_TIE = 0.75
CREAK_MIN_S = 0.06
CREAK_DESCENT_ST = 1.0


@dataclass(frozen=True)
class WordInfo:
    index: int
    text: str  # as displayed (numbers / titles already written out)
    phrase: int
    content: bool
    sentence_stress: bool
    nucleus: bool
    pron: WordPronunciation
    spoken: str = ""  # lowercase a-z letters said for this word

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
    # Per phrase: "continue" | "fall" | "rise" | "either" (tag question).
    phrase_ends: tuple[str, ...]
    reference: tuple[RefPoint, ...]

    @property
    def final_tone(self) -> str:
        return self.phrase_ends[-1] if self.phrase_ends else "fall"


def _bare(token: SpeechToken) -> str:
    """"what's" -> "what": the word without its clitic."""
    return token.spoken.split("'")[0]


def _question_tone(phrases: list[list[SpeechToken]]) -> str:
    """Tone of a question made of `phrases` (its last one ends in "?")."""
    last = phrases[-1]
    if len(phrases) > 1 and len(last) <= 3 and last[0].spoken in _TAG_AUX and last[-1].spoken in PRONOUNS:
        return "either"  # tag question: rising (asking) or falling (checking)
    if any(t.spoken == "or" for t in last[1:]):
        return "fall"  # alternative question: "tea or coffee?"
    for phrase in phrases:
        head = [t for t in phrase[:2]]
        if head and _bare(head[0]) in _WH:
            return "fall"
        if len(head) == 2 and head[0].spoken in _DISCOURSE and _bare(head[1]) in _WH:
            return "fall"
    return "rise"


def _phrases(expanded: str, tokens: list[SpeechToken]) -> tuple[list[list[SpeechToken]], list[str]]:
    """Tokens grouped into phrases, plus the punctuation closing each."""
    phrases: list[list[SpeechToken]] = []
    closers: list[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            phrases.append([tok])
            continue
        gap = expanded[tokens[i - 1].end:tok.start]
        if _BREAK.search(gap):
            closers.append(gap)
            phrases.append([tok])
        else:
            phrases[-1].append(tok)
    if tokens:
        closers.append(expanded[tokens[-1].end:])
    return phrases, closers


def _phrase_ends(phrases: list[list[SpeechToken]], closers: list[str]) -> list[str]:
    ends: list[str] = []
    sentence_start = 0
    for p, closer in enumerate(closers):
        last = p == len(closers) - 1
        if "?" in closer:
            tone = _question_tone(phrases[sentence_start:p + 1])
        elif re.search(r"[.!…]", closer) or last:
            tone = "fall"
        else:
            ends.append("continue")
            continue
        ends.append(tone)
        sentence_start = p + 1
    return ends


def analyze_sentence(text: str) -> SentenceAnalysis:
    """Words, phrases, stress roles and the reference contour for `text`."""
    expanded, tokens = speech_tokens(text)
    phrases, closers = _phrases(expanded, tokens)
    words: list[WordInfo] = []
    index = 0
    for p, members in enumerate(phrases):
        content = [t.spoken not in FUNCTION_WORDS for t in members]
        nucleus_at = max((k for k, c in enumerate(content) if c), default=None)
        if nucleus_at is None:
            # No content word ("I can do it."): never a pronoun.
            nucleus_at = max(
                (k for k, t in enumerate(members) if t.spoken not in PRONOUNS),
                default=len(members) - 1,
            )
        for k, tok in enumerate(members):
            words.append(WordInfo(
                index=index,
                text=tok.text,
                phrase=p,
                content=content[k],
                sentence_stress=content[k] or k == nucleus_at,
                nucleus=k == nucleus_at,
                pron=pronounce(tok.text, tok.spoken, tok.initialism),
                spoken=tok.spoken.replace("'", ""),
            ))
            index += 1
    phrase_ends = tuple(_phrase_ends(phrases, closers))
    analysis = SentenceAnalysis(expanded, tuple(words), phrase_ends, ())
    return SentenceAnalysis(expanded, tuple(words), phrase_ends, tuple(_reference(analysis)))


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
        # After the nucleus: glide to the phrase-final level. A tag question
        # is drawn with the (more common) checking fall.
        target = {"fall": -5.0, "either": -5.0, "rise": 5.0, "continue": -1.5}[ending]
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
    start_semitone: float | None  # first third of the voiced frames
    end_semitone: float | None  # last third of the voiced frames
    peak: float | None  # 90th percentile
    energy_db: float | None
    speech_time: float = 0.0  # seconds of speech (not silence) in the window
    voiced_time: float = 0.0  # seconds with a measured F0
    missing: bool = False  # the alignment says this word was not said

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
    missing: set[int] | frozenset[int] = frozenset(),
) -> list[SyllableMeasure]:
    """One measure per (word, syllable) window, in sentence order."""
    out = []
    n_frames = track.semitone.shape[0]
    for w, word_windows in enumerate(windows):
        for s, (start, end) in enumerate(word_windows):
            if w in missing:
                out.append(SyllableMeasure(
                    w, s, round(start, 3), round(end, 3), None, None, None, None, None,
                    missing=True,
                ))
                continue
            lo = min(n_frames, max(0, int(round(start / track.frame_time))))
            hi = min(n_frames, max(lo + 1, int(round(end / track.frame_time))))
            seg = track.semitone[lo:hi]
            # Thirds of the VOICED frames: the window may run on into a
            # pause, which must not shift "start" / "end".
            voiced = seg[~np.isnan(seg)]
            third = max(1, voiced.size // 3)
            loud = track.energy_db[lo:hi]
            out.append(SyllableMeasure(
                word=w,
                syllable=s,
                start=round(start, 3),
                end=round(end, 3),
                semitone=_median(voiced),
                start_semitone=_median(voiced[:third]),
                end_semitone=_median(voiced[-third:]),
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
    expected: str  # "fall" | "rise" | "either"
    ok: bool | None
    delta: float | None  # st, end minus nucleus / phrase level (negative = fell)
    creaky: bool = False  # statement ended in creak (F0 below tracking)
    phrase: int = 0  # which phrase this sentence end closes


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
    ends: list[FinalCheck]  # one per sentence end, in order
    word_stress: list[WordStressCheck]
    aligned: bool
    score: float | None = field(default=None)
    missing: frozenset[int] = frozenset()

    @property
    def final(self) -> FinalCheck | None:
        return self.ends[-1] if self.ends else None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _phrase_of(analysis: SentenceAnalysis, word: int) -> int:
    return analysis.words[word].phrase


def _check_peaks(analysis, by_key, ref_by_key, missing) -> list[PeakCheck]:
    out = []
    all_keys = [(r.word, r.syllable) for r in analysis.reference]
    weak_anywhere = [
        by_key[k] for k in all_keys
        if ref_by_key[k].role in ("weak", "secondary") and by_key[k].voiced
    ]
    for p, ending in enumerate(analysis.phrase_ends):
        keys = [k for k in all_keys if _phrase_of(analysis, k[0]) == p]
        for k, key in enumerate(keys):
            role = ref_by_key[key].role
            if role not in ("stress", "nucleus"):
                continue
            if role == "nucleus" and ending == "rise":
                continue  # a low-rise nucleus is judged by the end check
            if key[0] in missing:
                out.append(PeakCheck(*key, False, None))
                continue
            m = by_key[key]
            here = m.peak if m.peak is not None else m.semitone
            if here is None:
                out.append(PeakCheck(*key, None, None))
                continue
            # Compare with WEAK syllables only: nearby, else anywhere in
            # the phrase, else anywhere in the sentence -- never with the
            # other accents, which are supposed to be high too.
            weak = lambda x: ref_by_key[x].role in ("weak", "secondary") and by_key[x].voiced  # noqa: E731
            near = [by_key[keys[j]] for j in range(max(0, k - 2), min(len(keys), k + 3)) if j != k and weak(keys[j])]
            near = near or [by_key[x] for x in keys if weak(x)] or weak_anywhere
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


def _end_value(m: SyllableMeasure) -> float:
    return m.end_semitone if m.end_semitone is not None else m.semitone


def _start_value(m: SyllableMeasure) -> float:
    return m.start_semitone if m.start_semitone is not None else m.semitone


def _check_end(analysis, measures, ref_by_key, phrase: int, expected: str, missing) -> FinalCheck:
    seq = [m for m in measures if _phrase_of(analysis, m.word) == phrase]
    last_word = max(w.index for w in analysis.words if w.phrase == phrase)
    if last_word in missing:
        return FinalCheck(expected, False, None, False, phrase)
    nuc = next(
        (k for k, m in enumerate(seq) if ref_by_key[(m.word, m.syllable)].role == "nucleus"),
        max(0, len(seq) - 1),
    )
    tail = [m for m in seq[nuc:] if m.voiced]
    voiced_seq = [m for m in seq if m.voiced]
    if not tail or not voiced_seq:
        return FinalCheck(expected, None, None, False, phrase)
    end = _end_value(tail[-1])
    # Like with like: end of the last voiced frames against the START /
    # median of the syllables from the nucleus on (not their 90th
    # percentile), and against the phrase's middle level for a nucleus
    # that was itself said low ("the RECORD is old").
    level = float(np.median([m.semitone for m in voiced_seq]))
    top = max(max(_start_value(m), m.semitone) for m in tail)
    bottom = min(min(_start_value(m), m.semitone) for m in tail)
    fell = min(end - top, end - level)
    rose = max(end - bottom, end - level)
    if expected == "rise":
        return FinalCheck(expected, rose >= FINAL_MIN_ST, round(rose, 2), False, phrase)
    # Creak at the very end counts as reaching the bottom only when the
    # voiced part was already on its way down.
    creaky = (
        seq[-1].creaky and seq[-1] is not tail[-1]
        and (end <= top - CREAK_DESCENT_ST or end <= level - CREAK_DESCENT_ST)
        and end - bottom < 1.0  # no rise from the nucleus on
    )
    fall_ok = fell <= -FINAL_MIN_ST or creaky
    if expected == "either":
        rise_ok = rose >= FINAL_MIN_ST
        delta = fell if fall_ok or abs(fell) >= abs(rose) else rose
        return FinalCheck(expected, fall_ok or rise_ok, round(delta, 2), creaky, phrase)
    return FinalCheck(expected, fall_ok, round(fell, 2), creaky, phrase)


def _check_ends(analysis, measures, ref_by_key, missing) -> list[FinalCheck]:
    return [
        _check_end(analysis, measures, ref_by_key, p, ending, missing)
        for p, ending in enumerate(analysis.phrase_ends)
        if ending != "continue"
    ]


def _check_breaks(analysis, measures, pauses, aligned, missing) -> list[BreakCheck]:
    out = []
    for p in range(len(analysis.phrase_ends) - 1):
        before = [m for m in measures if _phrase_of(analysis, m.word) == p and m.voiced]
        after = [m for m in measures if _phrase_of(analysis, m.word) == p + 1 and m.voiced]
        last_word = max(w.index for w in analysis.words if w.phrase == p)
        pause = pauses[last_word] if pauses and last_word < len(pauses) else None
        if not aligned or pause is None or last_word in missing or last_word + 1 in missing:
            out.append(BreakCheck(last_word, None, pause, None))
            continue
        drop = None
        if before and after:
            end = _end_value(before[-1])
            top = max(m.peak if m.peak is not None else m.semitone for m in before)
            # Fall into the boundary, or a jump back up after it.
            drop = max(top - end, _start_value(after[0]) - end)
        ok = pause >= BREAK_PAUSE_S or (
            pause >= BREAK_SHORT_PAUSE_S and drop is not None and drop >= BREAK_PITCH_ST
        )
        out.append(BreakCheck(last_word, ok, pause, round(drop, 2) if drop is not None else None))
    return out


def _check_word_stress(analysis, measures, missing) -> list[WordStressCheck]:
    out = []
    for w in analysis.words:
        if len(w.pron.syllables) < 2:
            continue
        sylls = [m for m in measures if m.word == w.index]
        expected = w.pron.stress_index
        accepted = w.pron.accepted_stress or (expected,)
        if (
            w.index in missing
            or len(sylls) != len(w.pron.syllables)
            or not any(m.voiced for m in sylls)
        ):
            out.append(WordStressCheck(w.index, expected, None, None, accepted))
            continue
        pitch = np.array([
            (m.peak if m.peak is not None else m.semitone) if m.voiced else np.nan for m in sylls
        ])
        floor = np.nanmin(pitch) - 1.0
        pitch = np.where(np.isnan(pitch), floor, pitch)
        loud = np.array([m.energy_db if m.energy_db is not None else np.nan for m in sylls])
        loud = np.where(np.isnan(loud), np.nanmin(loud) if np.any(~np.isnan(loud)) else 0.0, loud)
        # Voiced time, lightly weighted: a final "-n't" / "-er" window runs
        # on to the next word and must not read as a long stressed vowel.
        length = np.log2(np.array([max(m.voiced_time, 0.02) for m in sylls]))
        prom = (pitch - pitch.mean()) + 0.3 * (loud - loud.mean()) + 0.5 * (length - length.mean())
        heard = int(np.argmax(prom))
        ok = any(
            heard == idx or prom[idx] >= prom[heard] - STRESS_TIE
            for idx in accepted if idx < len(prom)
        )
        if heard in accepted:
            expected = heard  # show the reading the learner actually used
        out.append(WordStressCheck(w.index, expected, heard, bool(ok), accepted))
    return out


def _end_credit(check: FinalCheck) -> float:
    if check.ok:
        return 1.0
    if check.delta is None:
        return 0.0
    moved = check.delta if check.expected == "rise" else -check.delta
    if check.expected == "either":
        moved = abs(check.delta)
    return _clamp01(moved / FINAL_MIN_ST)


def _score(peaks, breaks, ends, word_stress) -> float | None:
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
    end_credit = [_end_credit(c) for c in ends if c.ok is not None]
    if end_credit:
        parts.append((float(np.mean(end_credit)), 0.25))
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
    missing: set[int] | frozenset[int] = frozenset(),
) -> IntonationResult:
    """Measure the learner on syllable `windows` (one list per target word)
    and run the checks. `word_spans` (start, end) per word give the gaps
    searched for pauses; pass None when timing is only a rough split.
    `missing` are words the alignment found no trace of."""
    missing = frozenset(missing)
    measures = measure_syllables(track, windows, missing)
    by_key = {(m.word, m.syllable): m for m in measures}
    ref_by_key = {(r.word, r.syllable): r for r in analysis.reference}
    pauses = None
    if word_spans:
        pauses = [
            longest_pause(track, word_spans[i][1], word_spans[i + 1][0])
            for i in range(len(word_spans) - 1)
        ]
    peaks = _check_peaks(analysis, by_key, ref_by_key, missing)
    breaks = _check_breaks(analysis, measures, pauses, aligned, missing)
    ends = _check_ends(analysis, measures, ref_by_key, missing)
    word_stress = _check_word_stress(analysis, measures, missing)
    return IntonationResult(
        analysis, measures, peaks, breaks, ends, word_stress, aligned,
        _score(peaks, breaks, ends, word_stress), missing,
    )
