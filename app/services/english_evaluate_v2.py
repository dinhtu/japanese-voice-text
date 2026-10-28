"""English /evaluate-v2 (scoring_version en-reading-v2): Ollama-first.

    text + audio
      -> audio check        load 16 kHz mono, level / clipping, VAD, voice present?
      -> ASR (Ollama)       Qwen3-ASR transcript, language, top-20 token alternatives
      -> content            word WER of that transcript against the target
      -> timing (Python)    wav2vec2 CTC forced alignment of the target -- the one
                            job Ollama cannot do (no Ollama model emits timestamps);
                            EN_V2_ALIGNER=none turns it off
      -> analysis           pronunciation: per target word, Qwen3-ASR's margin
                            between the target and its best rival word at the
                            token where it starts (qwen3_asr.word_evidence);
                            word stress, sentence stress / final tone, phrase
                            breaks / linking from F0 + energy on the aligned
                            syllables; fluency from words-correct-per-minute,
                            VAD pauses, repeats and fillers
      -> score (Python)     S = 0.25 C + 0.30 P + 0.15 W + 0.15 I + 0.15 F
      -> feedback (Ollama)  OLLAMA_MODEL (qwen3:8b) explains the computed results
      -> JSON

Rules: a criterion without enough evidence is null with a status, never
0; the overall needs content and pronunciation (after a failed content
gate -- fewer than half the words read right -- the misread words are the
verdict and the overall is content alone, while stress / intonation /
fluency are not scored); pronunciation keeps the score of every word read
with evidence even when other words were misread. Word-level only: Ollama
has no phoneme model, so `phonemes` stays empty, and the pronunciation
margins are not calibrated on learner speech yet.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from app.services import qwen3_asr
from app.services.english_assessment import (
    COUNTED,
    PACE_ADVICE,
    WCPM_HI,
    WCPM_LO,
    _criterion,
    _gate,
    _msg,
    _phrase_last_word,
    _quote,
    _r,
    _stress_label,
    content_accuracy,
    fillers,
    fluency,
    inserted_words,
    intonation_score,
    repetitions,
    rhythm_and_linking,
    stress_checks,
    word_stress,
    word_verdicts,
)
from app.services.english_ollama_assessment import ASR_LANGUAGE, rival_evidence, speech_token_words
from app.services.english_scoring import align_word_tokens, transcript_tokens

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.services.english_alignment import SyllableTiming
    from app.services.english_intonation import IntonationResult, SentenceAnalysis

logger = logging.getLogger(__name__)

SCORING_VERSION = "en-reading-v2"
WEIGHTS: dict[str, float] = {
    "content_accuracy": 0.25,
    "pronunciation": 0.30,
    "word_stress": 0.15,
    "intonation": 0.15,
    "fluency": 0.15,
}
REQUIRED = ("content_accuracy", "pronunciation")
GATED = ("word_stress", "intonation", "fluency")
# A word read right whose pronunciation evidence is under this is flagged
# ("grass" for glass scores ~66 on TTS; clean words 100). Uncalibrated.
MISPRONOUNCED = 70.0
LONG_PAUSE_S = 0.5
# Inside the 15 % intonation criterion: sentence intonation vs rhythm / linking.
INTONATION_SHARE = 0.6
CLIPPING_LEVEL = 0.99
_OK = ("correct", "mispronounced")


# --- acoustics (sync, inside gpu_session) ----------------------------------

@dataclass
class Acoustics:
    target_text: str
    samples: np.ndarray
    sample_rate: int
    duration: float
    vad: Any
    presence: Any
    analysis: "SentenceAnalysis"
    timing: "SyllableTiming | None"
    intonation: "IntonationResult | None"
    aligner: str | None
    quality: dict = field(default_factory=dict)


def _quality(samples: np.ndarray, sample_rate: int, vad) -> dict:
    x = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    return {
        "duration_seconds": _r(len(x) / sample_rate if sample_rate else 0.0),
        "rms_dbfs": _r(20 * np.log10(rms)) if rms > 0 else None,
        "peak": _r(float(np.max(np.abs(x))) if x.size else 0.0, 3),
        "clipping_ratio": _r(float(np.mean(np.abs(x) >= CLIPPING_LEVEL)) if x.size else 0.0, 4),
        "speech_seconds": _r(vad.speech_duration) if vad is not None else None,
        "vad_method": vad.method if vad is not None else None,
    }


_ACOUSTICS_LOCK = threading.Lock()


def measure_acoustics(target_text: str, audio_path: str | Path, asr=None, aligner: str = "wav2vec2") -> Acoustics:
    """Audio check, VAD, forced-alignment timing and pitch measures.
    `asr` is the wav2vec2 EnglishASRService, used for its CTC emissions
    only (never its transcript). One request at a time: Silero VAD and the
    lazily loaded wav2vec2 are shared, stateful models."""
    with _ACOUSTICS_LOCK:
        return _measure_acoustics(target_text, audio_path, asr, aligner)


def _measure_acoustics(target_text: str, audio_path: str | Path, asr, aligner: str) -> Acoustics:
    from app.services.english_intonation import analyze_sentence
    from app.services.english_use_cases import _align, _intonation
    from app.services.vad_fluency import analyze_vad
    from app.services.voice_presence import check_voice
    from src.asr.inference import load_audio

    samples, sample_rate = load_audio(audio_path)
    vad = analyze_vad(samples, sample_rate)
    presence = check_voice(samples, sample_rate, vad)
    analysis = analyze_sentence(target_text)
    timing = None
    used = None
    if presence.voiced and aligner == "wav2vec2" and asr is not None:
        try:
            recognition = asr.recognize(audio_path)
            analysis, timing = _align(target_text, recognition)
            used = "wav2vec2_ctc_forced_alignment" if timing is not None else None
        except Exception:  # noqa: BLE001
            logger.warning("v2 forced alignment failed; no word timing", exc_info=True)
    intonation = None
    if presence.voiced:
        try:
            intonation = _intonation((analysis, timing), samples, sample_rate)
        except Exception:  # noqa: BLE001
            logger.warning("v2 pitch analysis failed", exc_info=True)
    return Acoustics(
        target_text=target_text, samples=samples, sample_rate=sample_rate,
        duration=len(samples) / sample_rate if sample_rate else 0.0, vad=vad, presence=presence,
        analysis=analysis, timing=timing, intonation=intonation, aligner=used,
        quality=_quality(samples, sample_rate, vad),
    )


# --- pronunciation evidence ------------------------------------------------

def pronunciation_evidence(heard: dict, matches, extras, target_text: str | None = None
                           ) -> tuple[list[dict | None], str | None]:
    """Evidence per target word the transcript has, from its own token
    alternatives (english_ollama_assessment.rival_evidence); (evidence or
    None per word, why it is missing)."""
    if not heard.get("stream"):
        return [None] * len(matches), "no_token_alternatives"  # old Ollama: no logprobs
    evidence = rival_evidence(heard["stream"], heard["transcript"], matches, extras, target_text)
    if evidence is None:
        return [None] * len(matches), "transcript_not_located"
    return evidence, None


def pronunciation_criterion(evidence: Sequence[dict | None], verdicts, matches, why: str | None) -> dict:
    """Mean evidence over the words that were read (the transcript has
    them), letter-weighted; misread / skipped words are content errors."""
    sized = [
        (ev["score"], max(1, len(m.target.replace("'", ""))))
        for ev, v, m in zip(evidence, verdicts, matches)
        if ev and ev.get("score") is not None and v.status in _OK
    ]
    if not sized:
        return _criterion("insufficient_evidence", reason=why or "no_word_read_with_evidence")
    score = sum(s * n for s, n in sized) / sum(n for _s, n in sized)
    return _criterion(
        "estimated", score, "qwen3_asr_rival_margin", calibrated=False, unit="word",
        words_scored=len(sized), margin_full_nats=qwen3_asr.MARGIN_FULL,
    )


# --- other criteria --------------------------------------------------------

def intonation_criterion(intonation, read: set[int], aligned: bool) -> dict:
    """15 %: sentence intonation (stress peaks, final tone) and rhythm
    (phrase breaks, linking, function-word reduction), each when measured.
    Needs word timing: on guessed syllable positions the peak checks would
    be scoring the wrong syllables."""
    if not aligned:
        return _criterion("insufficient_evidence", reason="no_word_timing")
    tone = intonation_score(intonation, read)
    rhythm = rhythm_and_linking(intonation, read)
    parts = [(c, w) for c, w in ((tone, INTONATION_SHARE), (rhythm, 1 - INTONATION_SHARE))
             if c["status"] in COUNTED and c["score"] is not None]
    if not parts:
        return _criterion("insufficient_evidence", reason=tone.get("reason") or rhythm.get("reason"),
                          intonation=tone, rhythm=rhythm)
    score = sum(c["score"] * w for c, w in parts) / sum(w for _c, w in parts)
    return _criterion("scored", score, "f0_peaks_final_tone_breaks_linking",
                      intonation=tone, rhythm=rhythm)


def stress_criterion(intonation, read: set[int], aligned: bool) -> dict:
    if not aligned:
        return _criterion("insufficient_evidence", reason="no_word_timing")
    return word_stress(intonation, read)


def _overall(criteria: dict[str, dict], gate_passed: bool) -> tuple[float | None, float]:
    used = {k: c for k, c in criteria.items() if c["status"] in COUNTED and c["score"] is not None}
    required: tuple[str, ...] = REQUIRED
    if not gate_passed:
        used = {k: c for k, c in used.items() if k == "content_accuracy"}
        required = ("content_accuracy",)
    if any(k not in used for k in required):
        return None, 0.0
    total = sum(WEIGHTS[k] for k in used)
    return _r(sum(WEIGHTS[k] * c["score"] for k, c in used.items()) / total), _r(total, 4)


# --- report ----------------------------------------------------------------

def _empty_criteria(status: str, reason: str) -> dict[str, dict]:
    out = {}
    for key, weight in WEIGHTS.items():
        c = _criterion(status, reason=reason)
        c["weight"] = weight
        out[key] = c
    return out


def _base(acoustics: Acoustics | None, target_text: str, status: str, **extra: Any) -> dict:
    return {
        "language": "en-US",
        "reference_text": target_text,
        "recognized_text": None,
        "detected_language": None,
        "status": status,
        "overall_score": None,
        "weights_used": 0.0,
        "content_gate": None,
        "criteria": _empty_criteria("not_assessed", status),
        "word_results": [],
        "errors": [],
        "improvements": [],
        "audio_analysis": dict(acoustics.quality) if acoustics else {},
        "asr": None,
        "alignment": {"method": acoustics.aligner if acoustics else None,
                      "aligned": bool(acoustics and acoustics.timing and acoustics.timing.aligned)},
        "feedback": None,
        "scoring_version": SCORING_VERSION,
        **extra,
    }


def _words_and_errors(matches, verdicts, evidence, acoustics: Acoustics, read: set[int],
                      gate_passed: bool, extras) -> tuple[list[dict], list[dict]]:
    timing = acoustics.timing
    analysis = acoustics.analysis
    intonation = acoustics.intonation
    stress = stress_checks(intonation) if timing is not None and timing.aligned else {}
    same = analysis is not None and len(analysis.words) == len(matches)
    words, errors = [], []
    for k, (m, v) in enumerate(zip(matches, verdicts)):
        info = analysis.words[k] if same else None
        text = info.text if info else m.target
        start = end = None
        if timing is not None and timing.aligned and k in read and k not in timing.missing and k < len(timing.windows):
            start, end = _r(timing.windows[k][0][0], 3), _r(timing.windows[k][-1][1], 3)
        ev = evidence[k] if k < len(evidence) else None
        check = stress.get(k) if gate_passed and k in read else None
        issues = []
        if v.status == "omitted":
            issues.append("omission")
            errors.append({"type": "omission", "index": k, "word": text, "heard": None,
                           "start_seconds": None, "end_seconds": None, "confidence": v.confidence})
        elif v.status == "substituted":
            issues.append("substitution")
            errors.append({"type": "substitution", "index": k, "word": text, "heard": m.heard,
                           "start_seconds": start, "end_seconds": end, "confidence": v.confidence})
        if k not in read:
            ev = None  # a word not said has no pronunciation evidence
        if v.status == "mispronounced":
            # The ASR spelled it as no real word ("wata"): heard, but off.
            issues.append("mispronunciation")
            errors.append({"type": "mispronunciation", "index": k, "word": text, "heard": m.heard,
                           "start_seconds": start, "end_seconds": end, "confidence": v.confidence})
        elif ev and ev.get("score") is not None and ev["score"] < MISPRONOUNCED:
            issues.append("mispronunciation")
            errors.append({"type": "mispronunciation", "index": k, "word": text, "heard": ev.get("rival"),
                           "start_seconds": start, "end_seconds": end,
                           "confidence": _r(1 - ev["score"] / 100.0)})
        if check is not None and check.ok is False:
            issues.append("word_stress")
            errors.append({"type": "word_stress", "index": k, "word": text,
                           "heard": None, "start_seconds": start, "end_seconds": end, "confidence": 0.6,
                           "expected": _stress_label([s.text for s in info.pron.syllables], check.expected) if info else None})
        words.append({
            "index": k,
            "text": text,
            "status": v.status,
            "heard": m.heard or None,
            "start_seconds": start,
            "end_seconds": end,
            "pronunciation": ev,
            "syllables": [s.text for s in info.pron.syllables] if info else None,
            "expected_stress": (check.expected if check else info.pron.stress_index) if info else None,
            "heard_stress": check.heard if check else None,
            "stress_ok": check.ok if check else None,
            "phonemes": [],
            "issues": issues,
        })
    for e in inserted_words(extras):
        errors.append({"type": "insertion", "index": e.after, "word": e.letters, "heard": e.letters,
                       "start_seconds": None, "end_seconds": None, "confidence": None})
    for e in repetitions(extras):
        errors.append({"type": "repetition", "index": e.after, "word": e.letters, "heard": e.letters,
                       "start_seconds": None, "end_seconds": None, "confidence": None})
    for e in fillers(extras):
        errors.append({"type": "filler", "index": e.after, "word": e.letters, "heard": e.letters,
                       "start_seconds": None, "end_seconds": None, "confidence": None})
    return words, errors


def _improvements(criteria: dict, gate: dict, words: list[dict], errors: list[dict], intonation,
                  read: set[int], lang: str, wrong_language: str | None = None) -> list[dict]:
    out: list[dict] = []

    def add(priority, category, code, items=(), evidence=None, **params):
        out.append({"priority": priority, "category": category, "code": code,
                    "message": _msg(lang, code, **params), "words": list(items), "evidence": evidence})

    by = lambda kind: [e for e in errors if e["type"] == kind]  # noqa: E731
    if wrong_language:
        # Not English at all: per-word omissions would only be noise.
        add("high", "content_accuracy", "wrong_language", evidence="asr_language", language=wrong_language)
        return out
    if by("omission"):
        names = [e["word"] for e in by("omission")]
        add("high", "content_accuracy", "omitted_words", names, "reference_vs_transcript", words=_quote(names))
    if by("substitution"):
        subs = by("substitution")
        add("high", "content_accuracy", "substituted_words", [e["word"] for e in subs], "reference_vs_transcript",
            words=_quote([f"{e['word']} → {e['heard']}" for e in subs]))
    if not gate["passed"]:
        add("high", "content_accuracy", "content_gate", evidence="content_gate")
    if by("mispronunciation"):
        mis = by("mispronunciation")
        labels = [f"{e['word']} (~{e['heard']})" if e["heard"] else e["word"] for e in mis]
        add("high", "pronunciation", "mispronounced_words", [e["word"] for e in mis],
            "qwen3_asr_rival_margin", words=_quote(labels))
    if by("insertion"):
        add("medium", "content_accuracy", "extra_words", [e["word"] for e in by("insertion")],
            "unmatched_transcript", words=_quote([e["word"] for e in by("insertion")]))
    if by("repetition"):
        add("medium", "fluency", "repeated_words", [e["word"] for e in by("repetition")],
            "unmatched_transcript", words=_quote([e["word"] for e in by("repetition")]))
    if by("word_stress"):
        ws = by("word_stress")
        add("medium", "word_stress", "wrong_stress", [e["word"] for e in ws], "syllable_prominence",
            words=_quote([e.get("expected") or e["word"] for e in ws]))
    tone = (criteria["intonation"].get("detail") or {}).get("intonation") or {}
    if gate["passed"] and intonation is not None and tone.get("status") == "scored":
        weak = [intonation.analysis.words[c.word].text for c in intonation.peaks if c.ok is False and c.word in read]
        if weak:
            add("low", "intonation", "weak_sentence_stress", weak, "f0_peaks", words=_quote(weak))
        for c in intonation.ends:
            if (c.ok is False and c.expected in ("fall", "rise")
                    and _phrase_last_word(intonation.analysis, c.phrase) in read):
                add("low", "intonation", f"final_tone_{c.expected}", evidence="f0_final_tone")
                break
    flu = criteria["fluency"]
    if flu["status"] == "scored" and flu.get("detail"):
        detail = flu["detail"]
        wcpm = detail.get("correct_words_per_minute") or 0.0
        if (detail.get("pace_score") or 100) < PACE_ADVICE:
            if wcpm < WCPM_LO and len(words) > 3:
                add("low", "fluency", "too_slow", evidence="wcpm")
            elif wcpm > WCPM_HI:
                add("low", "fluency", "too_fast", evidence="wcpm")
        if (detail.get("unexpected_pauses") or 0) >= 2:
            add("medium", "fluency", "too_many_pauses", evidence="vad_pauses")
    rhythm = (criteria["intonation"].get("detail") or {}).get("rhythm") or {}
    if rhythm.get("status") == "scored" and rhythm.get("detail"):
        pairs = [f"{words[a]['text']} {words[b]['text']}" for a, b in rhythm["detail"].get("broken_links", [])
                 if a < len(words) and b < len(words)]
        if pairs:
            add("low", "intonation", "broken_linking", pairs, "word_junction_pauses", words=_quote(pairs))
        if (rhythm["detail"].get("reduction_ratio") or 0) >= 1.0:
            add("low", "intonation", "no_reduction", evidence="syllable_durations")
    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda item: order[item["priority"]])
    return out


def feedback_lines(report: dict) -> list[str]:
    """The whole facts block for the Ollama feedback: computed results and
    the computed advice only, so the comment explains them and cannot
    contradict them."""
    names = {"content_accuracy": "Đọc đúng nội dung", "pronunciation": "Phát âm (ước lượng từ Qwen3-ASR)",
             "word_stress": "Trọng âm từ", "intonation": "Nhịp điệu, ngữ điệu", "fluency": "Độ trôi chảy"}
    lines = [
        "Đây là câu tiếng Anh; đơn vị là từ. Báo cáo chấm (en-reading-v2) bên dưới đã được tính sẵn: "
        "CHỈ giải thích các kết quả và lời khuyên này, không thêm lỗi khác, không mâu thuẫn với chúng; "
        "tiêu chí chưa chấm thì không khen cũng không chê.",
        f"Điểm tổng: {report['overall_score']}" if report["overall_score"] is not None else "Điểm tổng: chưa đủ dữ liệu.",
    ]
    for key, label in names.items():
        c = report["criteria"][key]
        lines.append(f"- {label}: {c['score']}/100" if c["score"] is not None
                     else f"- {label}: chưa chấm ({c.get('reason') or c['status']})")
    kinds = {"omission": "Bỏ sót", "substitution": "Đọc sai thành từ khác",
             "mispronunciation": "Phát âm chưa chuẩn (dễ nghe thành từ khác)",
             "word_stress": "Sai trọng âm", "insertion": "Từ thừa", "repetition": "Lặp từ", "filler": "Từ đệm"}
    errors = report["errors"]
    for kind, label in kinds.items():
        items = [e for e in errors if e["type"] == kind]
        if items:
            shown = [f"'{e['word']}'" + (f" (nghe như '{e['heard']}')" if e.get("heard") and kind in ("substitution", "mispronunciation") else "")
                     + (f" (đúng: {e['expected']})" if e.get("expected") else "")
                     for e in items[:8]]
            lines.append(f"{label}: " + ", ".join(shown))
    if report["status"] == "language_mismatch":
        lines.append(f"Bản ghi là tiếng {report.get('detected_language')}, không phải câu tiếng Anh mẫu: "
                     "chưa chấm được phát âm, trọng âm, ngữ điệu hay độ trôi chảy.")
    elif not any(e["type"] in ("omission", "substitution", "mispronunciation") for e in errors) and report["word_results"]:
        lines.append("Không phát hiện từ nào đọc sai hay phát âm lệch.")
    flu = (report["criteria"]["fluency"].get("detail") or {})
    if flu.get("correct_words_per_minute") is not None:
        lines.append(f"Tốc độ: {flu['correct_words_per_minute']} từ đúng/phút; "
                     f"ngắt hơi không đúng chỗ: {flu.get('unexpected_pauses', 0)} lần.")
    advice = report.get("improvements") or []
    if advice:
        lines.append("Lời khuyên đã tính (ưu tiên từ trên xuống):")
        lines.extend(f"* {item['message']}" for item in advice[:6])
    else:
        lines.append("Không có lời khuyên sửa lỗi nào: có thể khen ngắn gọn.")
    return lines


async def _feedback(report: dict, settings: "Settings", lang: str) -> dict:
    from app.services.coaching import (
        SUPPORTED_LANGUAGES,
        CoachingUnavailableError,
        PronunciationFacts,
        generate_comment,
    )
    from app.services.scoring import _feedback as level_of

    if lang not in SUPPORTED_LANGUAGES:
        lang = "vi"
    score = report["overall_score"]
    if score is None:
        score = report["criteria"]["content_accuracy"]["score"] or 0
    facts = PronunciationFacts(
        text=report["reference_text"], score=int(round(score)), level=level_of(int(round(score)))[0],
        total_morae=len(report["word_results"]),
        extra_lines=feedback_lines(report),
        facts_only=True,
    )
    t0 = time.perf_counter()
    try:
        comment = await generate_comment(facts, settings, lang=lang, target_lang="en")
    except CoachingUnavailableError as e:
        return {"status": "unavailable", "model": settings.ollama_model, "assessment": None,
                "suggestion": None, "lang": lang, "error": str(e), "latency_ms": None}
    except Exception as e:  # noqa: BLE001
        logger.warning("v2 feedback failed", exc_info=True)
        return {"status": "unavailable", "model": settings.ollama_model, "assessment": None,
                "suggestion": None, "lang": lang, "error": str(e) or type(e).__name__, "latency_ms": None}
    return {"status": "generated", "model": settings.ollama_model, "assessment": comment.assessment,
            "suggestion": comment.suggestion or None, "lang": lang, "error": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


async def evaluate_v2(acoustics: Acoustics, settings: "Settings", lang: str = "vi",
                      with_feedback: bool = True) -> dict:
    """The /evaluate-v2 report. Never raises for Ollama problems: they are
    statuses."""
    target_text = acoustics.target_text
    msg_lang = lang if lang in ("vi", "en") else "en"
    if not acoustics.presence.voiced:
        report = _base(acoustics, target_text, "no_voice")
        report["improvements"] = [{"priority": "high", "category": "content_accuracy", "code": "no_voice",
                                   "message": _msg(msg_lang, "no_voice"), "words": [],
                                   "evidence": f"voice_presence:{acoustics.presence.verdict}"}]
        report["overall_score"] = 0.0
        return report
    try:
        heard = await qwen3_asr.transcribe(acoustics.samples, acoustics.sample_rate, settings, ASR_LANGUAGE)
    except qwen3_asr.AsrError as e:
        return _base(acoustics, target_text, "asr_unavailable",
                     asr={"model": settings.en_ollama_asr_model, "status": e.status, "error": str(e)})

    transcript = heard["transcript"]
    # Numbers / titles written out and contractions read as the target has
    # them -- the same tokens rival_evidence locates in the stream.
    tokens = speech_token_words(transcript, target_text)[0]
    matches, extras = align_word_tokens(target_text, " ".join(tokens))
    verdicts = word_verdicts(matches, acoustics.timing)
    read = {v.index for v in verdicts if v.status in _OK}
    criteria: dict[str, dict] = {"content_accuracy": content_accuracy(matches, verdicts, extras)}
    gate = _gate(criteria["content_accuracy"], verdicts)

    evidence, why = pronunciation_evidence(heard, matches, extras, target_text)
    criteria["pronunciation"] = pronunciation_criterion(evidence, verdicts, matches, why)
    aligned = bool(acoustics.timing and acoustics.timing.aligned)
    intonation = acoustics.intonation
    if intonation is not None and len(intonation.analysis.words) != len(matches):
        intonation = None
    flu, audio = fluency(len(matches), len(read), acoustics.duration, acoustics.vad,
                         acoustics.analysis, extras)
    # Pitch / stress checks only where the word was read AND the aligner
    # found it: a word it lost has "failed" checks that measured nothing.
    measured = read - set(acoustics.timing.missing) if aligned else read
    if gate["passed"]:
        criteria["word_stress"] = stress_criterion(intonation, measured, aligned)
        criteria["intonation"] = intonation_criterion(intonation, measured, aligned)
        criteria["fluency"] = flu
    else:
        for key in GATED:
            criteria[key] = _criterion("gated", reason="content_mismatch")
    for key, c in criteria.items():
        c["weight"] = WEIGHTS[key]
    criteria = {k: criteria[k] for k in WEIGHTS}

    overall, weights_used = _overall(criteria, gate["passed"])
    language = heard["language"]
    # Another language altogether (no English word came through).
    wrong_language = language if (
        language and language.lower() != "english" and not transcript_tokens(transcript)
    ) else None
    if not transcript:
        status = "no_speech"
    elif wrong_language:
        status = "language_mismatch"
    elif not gate["passed"]:
        status = "content_mismatch"
    elif overall is None:
        status = "insufficient_evidence"
    elif all(c["status"] in COUNTED for c in criteria.values()):
        status = "complete"
    else:
        status = "partial_assessment"

    words, errors = _words_and_errors(matches, verdicts, evidence, acoustics, read, gate["passed"], extras)
    if wrong_language:
        errors = []  # not the English text at all: per-word omissions are noise
    long_pauses = None
    if intonation is not None and aligned and intonation.pauses:
        long_pauses = sum(1 for p in intonation.pauses if p >= LONG_PAUSE_S)
    report = _base(
        acoustics, target_text, status,
        recognized_text=transcript,
        detected_language=language,
        language_mismatch=bool(language) and language.lower() != "english",
        overall_score=overall,
        weights_used=weights_used,
        content_gate=gate,
        criteria=criteria,
        word_results=words,
        errors=errors,
        audio_analysis={**acoustics.quality, **audio, "long_pause_count": long_pauses,
                        "phrase_count": len(acoustics.analysis.phrase_ends) if acoustics.analysis else None},
        asr={"model": settings.en_ollama_asr_model, "status": "ok", "error": None,
             "calls": heard["calls"], "latency_ms": heard["latency_ms"],
             "token_alternatives": bool(heard.get("stream"))},
    )
    report["improvements"] = _improvements(criteria, gate, words, errors, intonation, measured, msg_lang,
                                           wrong_language)
    if with_feedback and getattr(settings, "en_v2_feedback", True):
        report["feedback"] = await _feedback(report, settings, lang)
    return report
