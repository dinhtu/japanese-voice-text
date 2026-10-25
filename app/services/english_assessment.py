"""Two extra English reports, returned next to the legacy fields (which stay
exactly as they were, so old clients and the page keep working):

`reading_assessment` (scoring_version en-reading-v1) -- six criteria, each
0-100, from signals the pipeline already measures:

    content_accuracy    25%  word WER on the transcript's word tokens
                             (english_scoring.align_word_tokens):
                             100 * max(0, 1 - (S + D + I) / N)
    phoneme_accuracy    30%  forced-alignment letter posteriors of the target
                             (a GOP proxy on wav2vec2 letters, not phonemes:
                             status "estimated", calibrated=False)
    word_stress         15%  dictionary stress heard on multi-syllable words
    intonation          10%  sentence-stress peaks + sentence-final fall/rise
    fluency             10%  words CORRECT per minute + unexpected pauses +
                             extra / repeated words
    rhythm_and_linking  10%  pauses at phrase breaks, none inside phrases,
                             function words shorter than stressed syllables

    overall = sum(w_i * s_i) / sum(w_i) over the criteria that could be
    measured; never a missing criterion counted as 0. Null when content or
    phoneme accuracy is missing. After a failed content gate the misread
    words are the verdict: overall = content accuracy alone.

Content gate: voice + timing alone used to earn fluency / rhythm /
intonation even when the learner read other words. Here every
prosody-type criterion (word stress, intonation, fluency, rhythm) is
"gated" -- not scored, not in the overall -- unless at least
COVERAGE_GATE of the target words were read right; when it passes,
stress / intonation / rhythm only look at words that were read right and
fluency counts only correct words. Repeated words ("to to") and fillers
are fluency issues, not WER insertions.

`gopt_assessment` -- the English GOPT (SpeechOcean762) heads on their own:
`average` is the plain mean of its four aspect heads; `adjusted_average`
applies the same gate (fluency / prosodic dropped when it fails), scales
every head by min(word coverage, content accuracy), leaves out collapsed
heads (< GOPT_COLLAPSED, out-of-domain features) and never exceeds
`average`. GOPT is handed the target phones, so on its own it cannot tell
that other words were said; per-word heads of unread words are 0.

Pure numpy + dicts (validated by app/schemas/english_assessment.py), so it
is testable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from app.services.english_alignment import (
    HEARD_LETTER_LOGPROB,
    SyllableTiming,
)
from app.services.english_intonation import (
    PEAK_MIN_ST,
    BREAK_PAUSE_S,
    IntonationResult,
    SentenceAnalysis,
    _end_credit,
)
from app.services.english_scoring import _FILLERS, SNAP_GATE, ExtraPiece, WordMatch

SCORING_VERSION = "en-reading-v1"
WEIGHTS: dict[str, float] = {
    "content_accuracy": 0.25,
    "phoneme_accuracy": 0.30,
    "word_stress": 0.15,
    "intonation": 0.10,
    "fluency": 0.10,
    "rhythm_and_linking": 0.10,
}
REQUIRED = ("content_accuracy", "phoneme_accuracy")
GATED = ("word_stress", "intonation", "fluency", "rhythm_and_linking")
# Statuses whose score counts in the overall.
COUNTED = ("scored", "estimated")

COVERAGE_GATE = 0.5

# Letter posterior on the forced path -> 0..1 articulation credit. Clear
# letters spike at -0.5 .. 0 (english_alignment HEARD_STRONG_LOGPROB);
# below HEARD_LETTER_LOGPROB the letter is not really there.
GOP_CLEAR = -1.0
GOP_ABSENT = HEARD_LETTER_LOGPROB
# A word read right whose letters score under this is flagged unclear.
WEAK_ARTICULATION = 50.0

# Read-aloud pace in words correct per minute: full credit inside the
# band, falling to PACE_FLOOR_SCORE at WCPM_FLOOR and to PACE_FAST_SCORE
# at WCPM_FAST. Very short targets carry more relative silence.
WCPM_LO = 80.0
WCPM_LO_SHORT = 40.0
WCPM_HI = 220.0
WCPM_FLOOR = 20.0
WCPM_FAST = 320.0
PACE_FLOOR_SCORE = 20.0
PACE_FAST_SCORE = 60.0
# Pace advice only below this pace score.
PACE_ADVICE = 85.0
# A VAD blip makes the speech span tiny; above this the file length is used.
WPM_IMPLAUSIBLE = 400.0
PAUSE_PENALTY = 8.0
PAUSE_PENALTY_MAX = 32.0
EXTRA_PENALTY = 6.0
EXTRA_PENALTY_MAX = 24.0
# Extra tokens shorter than EDGE_EXTRA before the first / after the last
# word are breath, clicks or noise the ASR spelled out; inside the
# sentence a lone letter other than "a" / "i" is a CTC scrap.
EDGE_EXTRA = 5
SINGLE_LETTER_WORDS = frozenset({"a", "i"})
# Hesitation sounds: a fluency issue, not a content (WER) one.
FILLERS = _FILLERS

# A pause this long between two words of one phrase breaks the linking.
LINK_BREAK_S = 0.25
# Function-word syllable length / stressed content syllable length.
REDUCTION_GOOD = 0.7
REDUCTION_BAD = 1.1

GOPT_COLLAPSED = 8.0
GOPT_HEADS = ("accuracy", "completeness", "fluency", "prosodic")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y0
    return y0 + (min(max(x, min(x0, x1)), max(x0, x1)) - x0) / (x1 - x0) * (y1 - y0)


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(float(x), nd)


def _criterion(status: str, score: float | None = None, metric: str | None = None,
               reason: str | None = None, **detail: Any) -> dict:
    return {
        "score": _r(max(0.0, min(100.0, score))) if score is not None and status in COUNTED else None,
        "status": status,
        "metric": metric,
        "reason": reason,
        "detail": detail or None,
    }


# --- content ---------------------------------------------------------------

def inserted_words(extras: Sequence[ExtraPiece]) -> list[ExtraPiece]:
    """Extra tokens that are new words (not repeats, fillers, breath or
    scraps): the I of WER."""
    return [
        e for e in extras
        if not e.repeat and e.letters not in FILLERS and (
            len(e.letters) >= EDGE_EXTRA if e.edge
            else len(e.letters) >= 2 or e.letters in SINGLE_LETTER_WORDS
        )
    ]


def repetitions(extras: Sequence[ExtraPiece]) -> list[ExtraPiece]:
    """A target word said again ("to to", "nice nice"): a fluency issue."""
    return [e for e in extras if e.repeat and e.letters not in FILLERS]


def fillers(extras: Sequence[ExtraPiece]) -> list[ExtraPiece]:
    return [e for e in extras if e.letters in FILLERS]


def _cmu_phones(word: str) -> list[tuple[str, ...]] | None:
    from app.services.english_lexicon import _cmudict

    entries = _cmudict().get(word) or _cmudict().get(word.replace("'", ""))
    if not entries:
        return None
    return [tuple(p.rstrip("012") for p in e) for e in entries]


def _other_real_word(target: str, heard: str) -> bool:
    """`heard` is a different dictionary word, not `target` spelled by ear:
    "class" for "glass", "waiter" for "water" (but not "meat" for "meet",
    "kat" for "cat")."""
    if not heard or heard == target.replace("'", ""):
        return False
    heard_phones = _cmu_phones(heard)
    target_phones = _cmu_phones(target)
    if heard_phones is None or target_phones is None:
        return False
    return not set(heard_phones) & set(target_phones)


@dataclass(frozen=True)
class WordVerdict:
    index: int
    status: str  # "correct" | "mispronounced" | "substituted" | "omitted"
    confidence: float | None  # how sure the error call is, not a score


def word_verdicts(matches: Sequence[WordMatch], timing: SyllableTiming | None,
                  confirmed: frozenset[int] = frozenset()) -> list[WordVerdict]:
    """Per word: read right, roughly right, replaced, or left out. An
    omission needs the transcript to agree (nothing heard for the word);
    the forced path alone always places every word somewhere. A word the
    ASR spelled far off but the forced path clearly finds in the audio
    (`confirmed`, english_alignment.heard_words) counts as roughly right."""
    missing = timing.missing if timing is not None else frozenset()
    out = []
    for i, m in enumerate(matches):
        if not m.ok and m.heard and i in confirmed:
            out.append(WordVerdict(i, "mispronounced", 0.4))
        elif m.ok and i not in confirmed and _other_real_word(m.target, m.heard):
            # Close in sound, but the transcript holds another real word.
            out.append(WordVerdict(i, "substituted", 0.6))
        elif m.ok:
            if m.similarity < SNAP_GATE:
                out.append(WordVerdict(i, "mispronounced", _r(_clamp01(0.4 + (SNAP_GATE - m.similarity)))))
            else:
                out.append(WordVerdict(i, "correct", None))
        elif not m.heard:
            out.append(WordVerdict(i, "omitted", 0.9 if i in missing else 0.7))
        else:
            conf = _clamp01(0.5 + (1.0 - m.similarity) * 0.5)
            out.append(WordVerdict(i, "substituted", _r(min(0.95, conf))))
    return out


def content_accuracy(matches: Sequence[WordMatch], verdicts: Sequence[WordVerdict],
                     extras: Sequence[ExtraPiece]) -> dict:
    n = len(matches)
    subs = sum(1 for v in verdicts if v.status == "substituted")
    dels = sum(1 for v in verdicts if v.status == "omitted")
    ins = len(inserted_words(extras))
    wer = (subs + dels + ins) / n if n else 1.0
    return _criterion(
        "scored", 100.0 * max(0.0, 1.0 - wer), "normalized_wer",
        wer=_r(wer, 4), substitutions=subs, deletions=dels, insertions=ins, reference_words=n,
    )


# --- phoneme proxy ---------------------------------------------------------

def word_articulation(timing: SyllableTiming | None) -> list[float] | None:
    """0-100 per target word from its letters' best forced-path log-prob."""
    if timing is None or not timing.letter_confidence:
        return None
    out = []
    for letters in timing.letter_confidence:
        if not letters:
            out.append(0.0)
            continue
        credit = [_clamp01((lp - GOP_ABSENT) / (GOP_CLEAR - GOP_ABSENT)) for lp in letters]
        out.append(round(100.0 * float(np.mean(credit)), 2))
    return out


def phoneme_accuracy(articulation: list[float] | None, timing: SyllableTiming | None,
                     words: Sequence[str]) -> dict:
    if timing is not None and not timing.aligned:
        return _criterion("insufficient_evidence", reason="alignment_unreliable")
    if articulation is None or timing is None or len(articulation) != len(words):
        return _criterion("insufficient_evidence", reason="no_forced_alignment")
    sizes = [max(1, len(w.replace("'", ""))) for w in words]
    score = sum(a * s for a, s in zip(articulation, sizes)) / sum(sizes)
    return _criterion(
        "estimated", score, "ctc_forced_alignment_letter_posterior",
        calibrated=False, unit="letter",
    )


# --- prosody ---------------------------------------------------------------

def _gate(content: dict, verdicts: Sequence[WordVerdict]) -> dict:
    """On the words read right only: extra words lower content accuracy
    but do not make the target's own words unmeasurable."""
    n = len(verdicts) or 1
    read = sum(1 for v in verdicts if v.status in ("correct", "mispronounced"))
    coverage = read / n
    return {
        "passed": bool(coverage >= COVERAGE_GATE),
        "content_score": _r(content.get("score") or 0.0),
        "word_coverage": _r(coverage, 4),
        "min_word_coverage": COVERAGE_GATE,
    }


def word_stress(intonation: IntonationResult | None, read: set[int]) -> dict:
    if intonation is None:
        return _criterion("insufficient_evidence", reason="no_pitch_track")
    checks = [c for c in intonation.word_stress if c.word in read and c.ok is not None]
    if not checks:
        return _criterion("not_assessed", reason="no_multisyllable_word_measured")
    right = sum(1 for c in checks if c.ok)
    return _criterion(
        "scored", 100.0 * right / len(checks), "stressed_syllable_prominence",
        checked_words=len(checks), correct=right,
    )


def _phrase_last_word(analysis: SentenceAnalysis, phrase: int) -> int | None:
    words = [w.index for w in analysis.words if w.phrase == phrase]
    return max(words) if words else None


def intonation_score(intonation: IntonationResult | None, read: set[int]) -> dict:
    if intonation is None:
        return _criterion("insufficient_evidence", reason="no_pitch_track")
    peaks = [
        1.0 if c.ok else _clamp01((c.delta or 0.0) / PEAK_MIN_ST)
        for c in intonation.peaks if c.ok is not None and c.word in read
    ]
    ends = [
        _end_credit(c) for c in intonation.ends
        if c.ok is not None and _phrase_last_word(intonation.analysis, c.phrase) in read
    ]
    parts = []
    if peaks:
        parts.append((float(np.mean(peaks)), 0.6))
    if ends:
        parts.append((float(np.mean(ends)), 0.4))
    if not parts:
        return _criterion("insufficient_evidence", reason="no_voiced_stress_or_ending")
    score = 100.0 * sum(v * w for v, w in parts) / sum(w for _v, w in parts)
    return _criterion(
        "scored", score, "f0_peaks_and_final_tone",
        peaks_checked=len(peaks), endings_checked=len(ends),
        final_tone=intonation.analysis.final_tone,
    )


def speech_seconds(duration: float, vad) -> float:
    if vad is not None and vad.speech_duration > 0:
        return float(vad.speech_duration)
    return float(duration)


def fluency(n_words: int, n_read: int, duration: float, vad, analysis: SentenceAnalysis | None,
            extras: Sequence[ExtraPiece]) -> tuple[dict, dict]:
    """(criterion, audio facts). Pace is words CORRECT per minute, so a
    misread sentence is not fluent just because it was said at speed."""
    seconds = speech_seconds(duration, vad)
    if seconds <= 0 or n_words <= 0:
        return _criterion("insufficient_evidence", reason="no_speech"), {}
    if n_words / seconds * 60 > WPM_IMPLAUSIBLE and duration > seconds:
        seconds = float(duration)
    wpm = n_words / seconds * 60
    wcpm = n_read / seconds * 60
    lo = WCPM_LO_SHORT if n_words <= 3 else WCPM_LO
    if lo <= wcpm <= WCPM_HI:
        pace = 100.0
    elif wcpm < lo:
        pace = _lerp(wcpm, WCPM_FLOOR, lo, PACE_FLOOR_SCORE, 100.0) if wcpm > WCPM_FLOOR else (
            PACE_FLOOR_SCORE * wcpm / WCPM_FLOOR
        )
    else:
        pace = _lerp(wcpm, WCPM_HI, WCPM_FAST, 100.0, PACE_FAST_SCORE)
    expected = max(0, len(analysis.phrase_ends) - 1) if analysis is not None else 0
    pauses = vad.pause_count if vad is not None else None
    unexpected = max(0, pauses - expected) if pauses is not None else 0
    inner = [e for e in inserted_words(extras) if not e.edge] + repetitions(extras)
    hesitations = fillers(extras)
    penalty = min(PAUSE_PENALTY_MAX, PAUSE_PENALTY * unexpected) + min(
        EXTRA_PENALTY_MAX, EXTRA_PENALTY * (len(inner) + len(hesitations))
    )
    criterion = _criterion(
        "scored", pace - penalty, "wcpm_pauses_repetitions",
        words_per_minute=_r(wpm, 1), correct_words_per_minute=_r(wcpm, 1),
        pace_score=_r(pace), unexpected_pauses=unexpected,
        repetitions_or_extra_words=len(inner), fillers=len(hesitations),
    )
    facts = {
        "speech_seconds": _r(seconds),
        "words_per_minute": _r(wpm, 1),
        "correct_words_per_minute": _r(wcpm, 1),
        "pause_count": pauses,
        "unexpected_pause_count": unexpected if pauses is not None else None,
        "vad_method": vad.method if vad is not None else None,
    }
    return criterion, facts


def rhythm_and_linking(intonation: IntonationResult | None, read: set[int]) -> dict:
    if intonation is None:
        return _criterion("insufficient_evidence", reason="no_pitch_track")
    if not intonation.aligned or not intonation.pauses:
        return _criterion("insufficient_evidence", reason="no_word_timing")
    analysis = intonation.analysis
    words = analysis.words
    parts: list[tuple[float, float]] = []
    breaks = [
        1.0 if c.ok else 0.6 * _clamp01((c.pause or 0.0) / BREAK_PAUSE_S)
        for c in intonation.breaks
        if c.ok is not None and c.after_word in read and c.after_word + 1 in read
    ]
    if breaks:
        parts.append((float(np.mean(breaks)), 0.3))
    junctions = [
        intonation.pauses[i] for i in range(min(len(words) - 1, len(intonation.pauses)))
        if words[i].phrase == words[i + 1].phrase and i in read and i + 1 in read
    ]
    broken = [i for i in range(min(len(words) - 1, len(intonation.pauses)))
              if words[i].phrase == words[i + 1].phrase and i in read and i + 1 in read
              and intonation.pauses[i] >= LINK_BREAK_S]
    if junctions:
        parts.append((1.0 - len(broken) / len(junctions), 0.4))
    weak = [m.voiced_time for m in intonation.syllables
            if m.word in read and not m.missing and m.voiced_time > 0 and not words[m.word].content]
    strong = [m.voiced_time for m in intonation.syllables
              if m.word in read and not m.missing and m.voiced_time > 0 and words[m.word].content
              and m.syllable == words[m.word].pron.stress_index]
    ratio = None
    if weak and strong:
        ratio = float(np.mean(weak)) / max(float(np.mean(strong)), 1e-3)
        parts.append((_lerp(ratio, REDUCTION_GOOD, REDUCTION_BAD, 1.0, 0.3), 0.3))
    if not parts:
        return _criterion("not_assessed", reason="too_few_words_read")
    score = 100.0 * sum(v * w for v, w in parts) / sum(w for _v, w in parts)
    return _criterion(
        "scored", score, "phrase_breaks_linking_reduction",
        phrase_breaks_checked=len(breaks), junctions_checked=len(junctions),
        broken_links=[[i, i + 1] for i in broken], reduction_ratio=_r(ratio, 3),
    )


def _overall(criteria: dict[str, dict], gate_passed: bool) -> tuple[float | None, float]:
    """A failed content gate is itself the verdict: the overall is content
    accuracy alone (articulation of words that were not said is moot)."""
    used = {k: c for k, c in criteria.items() if c["status"] in COUNTED and c["score"] is not None}
    if not gate_passed:
        used = {k: c for k, c in used.items() if k == "content_accuracy"}
        required: tuple[str, ...] = ("content_accuracy",)
    else:
        required = REQUIRED
    if any(k not in used for k in required):
        return None, 0.0
    total = sum(WEIGHTS[k] for k in used)
    return _r(sum(WEIGHTS[k] * c["score"] for k, c in used.items()) / total), _r(total, 4)


# --- messages --------------------------------------------------------------

_MESSAGES = {
    "vi": {
        "omitted_words": "Bạn đã bỏ sót: {words}. Hãy đọc đủ từng từ, kể cả mạo từ và giới từ.",
        "substituted_words": "Các từ đọc chưa đúng: {words}. Nghe lại mẫu và đọc chậm từng từ này.",
        "extra_words": "Có từ thừa không có trong câu: {words}. Hãy đọc đúng theo văn bản, không chen thêm từ.",
        "repeated_words": "Có từ bị đọc lặp lại: {words}. Hãy đọc liền mạch, không lặp hay sửa giữa chừng.",
        "unclear_words": "Phát âm chưa rõ: {words}. Hãy mở miệng rõ hơn và đọc đủ âm cuối.",
        "wrong_stress": "Đặt trọng âm sai ở: {words}. Nhấn mạnh (cao và dài hơn) vào âm tiết viết hoa.",
        "final_tone_fall": "Cuối câu nên xuống giọng rõ ràng.",
        "final_tone_rise": "Câu hỏi Yes/No nên lên giọng ở cuối câu.",
        "weak_sentence_stress": "Các từ quan trọng chưa được nhấn rõ: {words}. Hãy nâng giọng ở âm tiết nhấn của các từ này.",
        "too_slow": "Tốc độ đọc còn chậm. Luyện đọc theo cụm từ thay vì từng từ một.",
        "too_fast": "Bạn đọc hơi nhanh. Hãy chậm lại để các từ rõ ràng hơn.",
        "too_many_pauses": "Có nhiều chỗ ngập ngừng giữa câu. Hãy chỉ ngắt hơi ở dấu câu.",
        "broken_linking": "Hãy nối liền các từ trong cùng một cụm: {words}.",
        "no_reduction": "Đọc lướt nhẹ các từ chức năng (a, the, to, of...) để tạo nhịp tự nhiên của tiếng Anh.",
        "content_gate": "Hãy đọc đúng các từ trong câu trước. Trọng âm, ngữ điệu, nhịp và độ trôi chảy chỉ được chấm khi bạn đọc đúng ít nhất một nửa câu.",
        "no_voice": "Không nghe thấy giọng đọc. Hãy đọc to, rõ và gần micro hơn.",
    },
    "en": {
        "omitted_words": "You skipped: {words}. Read every word, including articles and prepositions.",
        "substituted_words": "These words were not read correctly: {words}. Listen to the model and say them slowly.",
        "extra_words": "Extra words not in the text: {words}. Read exactly what is written.",
        "repeated_words": "Repeated words: {words}. Read smoothly without repeating or restarting.",
        "unclear_words": "Not clearly pronounced: {words}. Open your mouth more and finish the final sounds.",
        "wrong_stress": "Wrong word stress on: {words}. Make the capitalised syllable louder, longer and higher.",
        "final_tone_fall": "Let your voice fall clearly at the end of the sentence.",
        "final_tone_rise": "A yes/no question should rise at the end.",
        "weak_sentence_stress": "Key words were not stressed: {words}. Raise your pitch on their stressed syllables.",
        "too_slow": "Your reading is slow. Practise reading in phrases rather than word by word.",
        "too_fast": "You read a little fast. Slow down so each word is clear.",
        "too_many_pauses": "There are several hesitations. Pause only at punctuation.",
        "broken_linking": "Link the words inside each phrase: {words}.",
        "no_reduction": "Say function words (a, the, to, of...) lightly to get a natural English rhythm.",
        "content_gate": "Read the right words first. Stress, intonation, rhythm and fluency are only scored once at least half the sentence is read correctly.",
        "no_voice": "No voice was heard. Read aloud, clearly and closer to the microphone.",
    },
}


def _msg(lang: str, code: str, **params: Any) -> str:
    table = _MESSAGES.get(lang) or _MESSAGES["en"]
    return table[code].format(**params)


def _quote(items: Sequence[str], limit: int = 5) -> str:
    shown = [f"'{w}'" for w in items[:limit]]
    return ", ".join(shown) + (" ..." if len(items) > limit else "")


def _stress_label(syllables: Sequence[str], index: int | None) -> str:
    return "-".join(s.upper() if k == index else s.lower() for k, s in enumerate(syllables))


# --- report ----------------------------------------------------------------

def _word_texts(analysis: SentenceAnalysis | None, matches: Sequence[WordMatch]) -> list[str]:
    if analysis is not None and len(analysis.words) == len(matches):
        return [w.text for w in analysis.words]
    return [m.target for m in matches]


def assess_reading(
    *,
    target_text: str,
    recognized_text: str,
    matches: Sequence[WordMatch],
    extras: Sequence[ExtraPiece],
    analysis: SentenceAnalysis | None,
    timing: SyllableTiming | None,
    intonation: IntonationResult | None,
    vad,
    duration: float,
    confirmed: frozenset[int] = frozenset(),
    legacy_score: float | None = None,
    lang: str = "vi",
) -> dict:
    """The `reading_assessment` dict (see module docstring)."""
    if intonation is not None and len(intonation.analysis.words) != len(matches):
        intonation = None  # tokenizers drifted apart: no per-word prosody
    if analysis is not None and len(analysis.words) != len(matches):
        analysis = None
    if timing is not None and len(timing.letter_confidence) != len(matches):
        timing = None
    if timing is None or not timing.aligned:
        confirmed = frozenset()  # an unreliable path confirms nothing
    texts = _word_texts(analysis, matches)
    spoken = [m.target for m in matches]
    verdicts = word_verdicts(matches, timing, confirmed)
    read = {v.index for v in verdicts if v.status in ("correct", "mispronounced")}
    # Letter posteriors on an unreliable path say nothing about articulation.
    articulation = word_articulation(timing) if timing is not None and timing.aligned else None

    criteria: dict[str, dict] = {}
    criteria["content_accuracy"] = content_accuracy(matches, verdicts, extras)
    criteria["phoneme_accuracy"] = phoneme_accuracy(articulation, timing, spoken)
    gate = _gate(criteria["content_accuracy"], verdicts)
    flu, audio = fluency(len(matches), len(read), duration, vad, analysis, extras)
    if gate["passed"]:
        criteria["word_stress"] = word_stress(intonation, read)
        criteria["intonation"] = intonation_score(intonation, read)
        criteria["fluency"] = flu
        criteria["rhythm_and_linking"] = rhythm_and_linking(intonation, read)
    else:
        for key in GATED:
            criteria[key] = _criterion("gated", reason="content_mismatch")
    for key, c in criteria.items():
        c["weight"] = WEIGHTS[key]
    criteria = {k: criteria[k] for k in WEIGHTS}

    overall, weights_used = _overall(criteria, gate["passed"])
    if not gate["passed"]:
        status = "content_mismatch"
    elif overall is None:
        status = "insufficient_evidence"
    elif all(c["status"] in COUNTED for c in criteria.values()):
        status = "complete"
    else:
        status = "partial_assessment"

    stress = {c.word: c for c in intonation.word_stress} if intonation is not None else {}
    word_results = []
    for v, m, text in zip(verdicts, matches, texts):
        issues: list[str] = []
        confidence = v.confidence
        if v.status == "omitted":
            issues.append("possible_omission")
        elif v.status == "substituted":
            issues.append("substitution")
        elif v.status == "mispronounced":
            issues.append("mispronunciation")
        art = articulation[v.index] if articulation is not None else None
        if v.index in read and art is not None and art < WEAK_ARTICULATION:
            issues.append("unclear_articulation")
            confidence = max(confidence or 0.0, _r(0.5 + (WEAK_ARTICULATION - art) / 200.0))
        check = stress.get(v.index) if v.index in read and gate["passed"] else None
        if check is not None and check.ok is False:
            issues.append("wrong_stress")
            confidence = max(confidence or 0.0, 0.6)
        start = end = None
        # Timestamps only for words the transcript confirms: the forced
        # path places a skipped word somewhere anyway.
        if (timing is not None and timing.aligned and v.index in read
                and v.index not in timing.missing and v.index < len(timing.windows)):
            start, end = _r(timing.windows[v.index][0][0], 3), _r(timing.windows[v.index][-1][1], 3)
        info = analysis.words[v.index] if analysis is not None else None
        word_results.append({
            "index": v.index,
            "word": text,
            "recognized": m.heard or None,
            "status": v.status,
            "issue": issues[0] if issues else None,
            "issues": issues,
            "confidence": _r(confidence) if issues else None,
            "similarity": _r(m.similarity, 4),
            "phoneme_score": art,
            "start": start,
            "end": end,
            "syllables": [s.text for s in info.pron.syllables] if info is not None else None,
            "expected_stress": (check.expected if check else info.pron.stress_index) if info is not None else None,
            "heard_stress": check.heard if check else None,
            "stress_ok": check.ok if check else None,
        })

    improvements = _improvements(criteria, gate, word_results, extras, intonation, read, lang)
    if analysis is None:
        audio_extra = {}
    else:
        audio_extra = {"phrase_count": len(analysis.phrase_ends)}
    return {
        "language": "en-US",
        "reference_text": target_text,
        "recognized_text": recognized_text,
        "status": status,
        "overall_score": overall,
        "weights_used": weights_used,
        "content_gate": gate,
        "criteria": criteria,
        "word_results": word_results,
        "inserted": [
            {"after_word": e.after, "text": e.letters} for e in inserted_words(extras)
        ],
        "repeated": [
            {"after_word": e.after, "text": e.letters} for e in repetitions(extras)
        ],
        "improvements": improvements,
        "audio_analysis": {
            "duration_seconds": _r(duration),
            **audio,
            **audio_extra,
        },
        "legacy_score": _r(legacy_score),
        "scoring_version": SCORING_VERSION,
    }


def _improvements(criteria, gate, words, extras, intonation, read, lang) -> list[dict]:
    out: list[dict] = []

    def add(priority, category, code, words_=(), evidence=None, **params):
        out.append({
            "priority": priority,
            "category": category,
            "code": code,
            "message": _msg(lang, code, **params),
            "words": list(words_),
            "evidence": evidence,
        })

    omitted = [w["word"] for w in words if w["status"] == "omitted"]
    if omitted:
        add("high", "content_accuracy", "omitted_words", omitted, "reference_vs_transcript",
            words=_quote(omitted))
    subs = [w for w in words if w["status"] == "substituted"]
    if subs:
        labels = [f"{w['word']} → {w['recognized']}" for w in subs]
        add("high", "content_accuracy", "substituted_words", [w["word"] for w in subs],
            "reference_vs_transcript", words=_quote(labels))
    if not gate["passed"]:
        add("high", "content_accuracy", "content_gate", evidence="content_gate")
    extra = inserted_words(extras)
    if extra:
        add("medium", "content_accuracy", "extra_words", [e.letters for e in extra],
            "unmatched_transcript", words=_quote([e.letters for e in extra]))
    repeated = repetitions(extras)
    if repeated:
        add("medium", "fluency", "repeated_words", [e.letters for e in repeated],
            "unmatched_transcript", words=_quote([e.letters for e in repeated]))
    unclear = [w["word"] for w in words if "unclear_articulation" in w["issues"]]
    if unclear:
        add("medium", "phoneme_accuracy", "unclear_words", unclear, "forced_alignment_posterior",
            words=_quote(unclear))
    stressed = [w for w in words if "wrong_stress" in w["issues"] and w["syllables"]]
    if stressed:
        add("medium", "word_stress", "wrong_stress", [w["word"] for w in stressed],
            "syllable_prominence",
            words=_quote([_stress_label(w["syllables"], w["expected_stress"]) for w in stressed]))
    if gate["passed"] and intonation is not None and criteria["intonation"]["status"] == "scored":
        weak_peaks = [
            intonation.analysis.words[c.word].text for c in intonation.peaks
            if c.ok is False and c.word in read
        ]
        if weak_peaks:
            add("low", "intonation", "weak_sentence_stress", weak_peaks, "f0_peaks",
                words=_quote(weak_peaks))
        for c in intonation.ends:
            if c.ok is False and c.expected in ("fall", "rise"):
                add("low", "intonation", f"final_tone_{c.expected}", evidence="f0_final_tone")
                break
    flu = criteria["fluency"]
    if flu["status"] == "scored" and flu["detail"]:
        wcpm = flu["detail"]["correct_words_per_minute"] or 0.0
        # Only when the pace really cost points, not at the band's edge.
        if (flu["detail"]["pace_score"] or 0.0) < PACE_ADVICE:
            if wcpm < WCPM_LO and len(words) > 3:
                add("low", "fluency", "too_slow", evidence="wcpm")
            elif wcpm > WCPM_HI:
                add("low", "fluency", "too_fast", evidence="wcpm")
        if flu["detail"]["unexpected_pauses"] >= 2:
            add("medium", "fluency", "too_many_pauses", evidence="vad_pauses")
    rhy = criteria["rhythm_and_linking"]
    if rhy["status"] == "scored" and rhy["detail"]:
        pairs = [f"{words[a]['word']} {words[b]['word']}" for a, b in rhy["detail"]["broken_links"]]
        if pairs:
            add("low", "rhythm_and_linking", "broken_linking", pairs, "word_junction_pauses",
                words=_quote(pairs))
        ratio = rhy["detail"]["reduction_ratio"]
        if ratio is not None and ratio >= 1.0:
            add("low", "rhythm_and_linking", "no_reduction", evidence="syllable_durations")
    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda item: order[item["priority"]])
    return out


def no_voice_reading(target_text: str, words: Sequence[str], duration: float, verdict: str,
                     legacy_score: float | None = 0, lang: str = "vi") -> dict:
    """Silence / whisper: every word omitted, nothing else measurable."""
    criteria = {k: _criterion("not_assessed", reason=f"no_voice:{verdict}") for k in WEIGHTS}
    criteria["content_accuracy"] = _criterion(
        "scored", 0.0, "normalized_wer", wer=1.0, substitutions=0, deletions=len(words),
        insertions=0, reference_words=len(words),
    )
    for key, c in criteria.items():
        c["weight"] = WEIGHTS[key]
    return {
        "language": "en-US",
        "reference_text": target_text,
        "recognized_text": "",
        "status": "no_voice",
        "overall_score": 0.0,
        "weights_used": WEIGHTS["content_accuracy"],
        "content_gate": {
            "passed": False, "content_score": 0.0, "word_coverage": 0.0,
            "min_word_coverage": COVERAGE_GATE,
        },
        "criteria": criteria,
        "word_results": [
            {"index": i, "word": w, "recognized": None, "status": "omitted",
             "issue": "possible_omission", "issues": ["possible_omission"], "confidence": 0.9,
             "similarity": 0.0, "phoneme_score": None, "start": None, "end": None,
             "syllables": None, "expected_stress": None, "heard_stress": None, "stress_ok": None}
            for i, w in enumerate(words)
        ],
        "inserted": [],
        "repeated": [],
        "improvements": [{
            "priority": "high", "category": "content_accuracy", "code": "no_voice",
            "message": _msg(lang, "no_voice"), "words": [], "evidence": f"voice_presence:{verdict}",
        }],
        "audio_analysis": {"duration_seconds": _r(duration)},
        "legacy_score": _r(legacy_score),
        "scoring_version": SCORING_VERSION,
    }


# --- coaching facts --------------------------------------------------------

_CRITERION_VI = {
    "content_accuracy": "Đọc đúng nội dung",
    "phoneme_accuracy": "Phát âm từng âm (ước lượng)",
    "word_stress": "Trọng âm từ",
    "intonation": "Ngữ điệu câu",
    "fluency": "Độ trôi chảy",
    "rhythm_and_linking": "Nhịp điệu, nối âm",
}
_STATUS_VI = {
    "gated": "chưa chấm vì đọc sai hoặc thiếu quá nhiều từ",
    "insufficient_evidence": "chưa đủ dữ liệu để chấm",
    "not_assessed": "không áp dụng cho câu này",
}


def coach_facts(reading: dict | None) -> list[str]:
    """Extra lines for the /coach facts block: only what the analyzers
    found, so the LLM explains and prioritises but never scores or guesses
    (criteria that were not scored are named as such)."""
    if not reading:
        return []
    lines = [
        "Báo cáo chấm chi tiết (en-reading-v1). CHỈ nhận xét các lỗi được liệt kê dưới đây; "
        "tiêu chí \"chưa chấm\" thì không được khen cũng không được chê.",
    ]
    overall = reading.get("overall_score")
    lines.append(
        f"Điểm tổng theo báo cáo này: {overall}/100" if overall is not None
        else "Điểm tổng theo báo cáo này: chưa đủ dữ liệu."
    )
    for key, label in _CRITERION_VI.items():
        c = reading["criteria"].get(key) or {}
        if c.get("score") is not None:
            lines.append(f"- {label}: {c['score']}/100")
        else:
            lines.append(f"- {label}: {_STATUS_VI.get(c.get('status'), 'chưa chấm')}")
    if not reading["content_gate"]["passed"]:
        lines.append(
            "Người học đọc sai hoặc thiếu quá nhiều từ so với câu mẫu: KHÔNG khen ngữ điệu, "
            "nhịp hay độ trôi chảy; ưu tiên hướng dẫn đọc đúng từng từ."
        )
    words = reading.get("word_results") or []
    omitted = [w["word"] for w in words if w["status"] == "omitted"]
    if omitted:
        lines.append("Từ bị bỏ sót: " + _quote(omitted, 8))
    subs = [f"'{w['word']}' (nghe thành '{w['recognized']}')" for w in words if w["status"] == "substituted"]
    if subs:
        lines.append("Từ đọc sai: " + ", ".join(subs[:8]))
    unclear = [w["word"] for w in words if "unclear_articulation" in w["issues"]]
    if unclear:
        lines.append("Từ phát âm chưa rõ (theo bộ căn chỉnh âm thanh): " + _quote(unclear, 8))
    stress = [
        f"'{w['word']}' (đúng: {_stress_label(w['syllables'], w['expected_stress'])})"
        for w in words if "wrong_stress" in w["issues"] and w["syllables"]
    ]
    if stress:
        lines.append("Sai trọng âm: " + ", ".join(stress[:8]))
    if reading.get("inserted"):
        lines.append("Từ thừa không có trong câu: " + _quote([e["text"] for e in reading["inserted"]], 5))
    if reading.get("repeated"):
        lines.append("Từ bị đọc lặp: " + _quote([e["text"] for e in reading["repeated"]], 5))
    for item in reading.get("improvements") or []:
        if item["category"] in ("intonation", "rhythm_and_linking") or item["code"] in (
            "too_slow", "too_fast", "too_many_pauses",
        ):
            lines.append(f"Phát hiện ({item['category']}): {_msg('vi', item['code'], words=_quote(item['words']))}")
    return lines


# --- GOPT ------------------------------------------------------------------

def _gopt_word(w, texts: dict, statuses: dict) -> dict:
    """GOPT scores the target phones whether or not they were said, so a
    word the transcript says was not read gets 0 (raw heads kept apart)."""
    read_ok = statuses.get(w.index) in ("correct", "mispronounced")

    def adjust(value):
        return value if value is None or read_ok else 0.0

    return {
        "index": w.index,
        "word": texts.get(w.index, ""),
        "read_ok": read_ok,
        "phones": list(w.phones),
        "accuracy": adjust(w.accuracy),
        "stress": adjust(w.stress),
        "total": adjust(w.total),
        "raw": {"accuracy": w.accuracy, "stress": w.stress, "total": w.total},
        "phone_scores": list(w.phone_scores),
    }


def assess_gopt(detail, reading: dict, legacy_score: float | None = None) -> dict:
    """The `gopt_assessment` dict. `detail` is english_gopt.GoptDetail or
    None (model off / failed); `reading` is assess_reading's dict, whose
    gate and per-word verdicts GOPT on its own cannot see."""
    gate = reading["content_gate"]
    if reading["status"] == "no_voice":
        return {
            "status": "no_voice", "model": "gopt-librispeech", "heads": None, "head_reliable": None,
            "average": 0.0, "adjusted_heads": None, "adjusted_average": 0.0, "content_gate": gate,
            "feature_source": None, "phone_count": 0, "truncated": False, "words": [],
            "legacy_score": _r(legacy_score),
        }
    if detail is None:
        return {
            "status": "unavailable", "model": "gopt-librispeech", "heads": None,
            "head_reliable": None, "average": None, "adjusted_heads": None,
            "adjusted_average": None, "content_gate": gate, "feature_source": None,
            "phone_count": 0, "truncated": False, "words": [], "legacy_score": _r(legacy_score),
        }
    s = detail.scores
    heads = {
        "accuracy": s.pronunciation,
        "completeness": s.completeness,
        "fluency": s.fluency,
        "prosodic": s.intonation,
        "total": s.overall,
    }
    reliable = {k: v >= GOPT_COLLAPSED for k, v in heads.items()}
    average = _r(float(np.mean([heads[k] for k in GOPT_HEADS])))
    # Share of the sentence actually read: skipped / replaced words and,
    # through content accuracy, inserted ones.
    factor = min(gate["word_coverage"] or 0.0, (gate["content_score"] or 0.0) / 100.0)
    adjusted: dict[str, float | None] = {}
    for k in GOPT_HEADS:
        if not reliable[k] or (k in ("fluency", "prosodic") and not gate["passed"]):
            adjusted[k] = None
        else:
            adjusted[k] = _r(heads[k] * factor)
    kept = [v for v in adjusted.values() if v is not None]
    adjusted_average = float(np.mean(kept)) if kept else None
    if adjusted_average is not None and not gate["passed"]:
        # Dropping the (often lowest) prosodic head must not lift a misread.
        adjusted_average = min(adjusted_average, average)
    status = "scored" if gate["passed"] else "content_mismatch"
    if not kept:
        status = "insufficient_evidence"
    statuses = {w["index"]: w["status"] for w in reading["word_results"]}
    texts = {w["index"]: w["word"] for w in reading["word_results"]}
    return {
        "status": status,
        "model": "gopt-librispeech",
        "heads": heads,
        "head_reliable": reliable,
        "average": average,
        "adjusted_heads": adjusted,
        "adjusted_average": _r(adjusted_average),
        "content_gate": gate,
        "feature_source": detail.feature_source,
        "phone_count": detail.phone_count,
        "truncated": detail.truncated,
        "words": [_gopt_word(w, texts, statuses) for w in detail.words],
        "legacy_score": _r(legacy_score),
    }
