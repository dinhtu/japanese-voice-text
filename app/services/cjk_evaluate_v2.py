"""Chinese / Korean /evaluate-v2: the transcript comes from Qwen3-ASR (Ollama).

    text + audio
      -> audio side (local, inside gpu_session)   voice check, VAD, the CTC
                       model's forced alignment to the target (syllable
                       windows), F0 -- chinese_use_cases / korean_use_cases
      -> ASR (Ollama)  Qwen3-ASR (EN_OLLAMA_ASR_MODEL, the English v2 model)
                       held to Chinese / Korean; never given the target
      -> score         as v1, on that transcript: toneless pinyin syllables
                       (zh) / Hangul CER (ko); tones, rhythm and F0 on the
                       CTC windows

A silent or whispered take is v1's no-voice result and Ollama is not
called. Qwen3-ASR down is AsrUnavailableError (the route answers 503): there
is no transcript to score. inference_ms is Qwen3-ASR's latency plus the CTC
model's time; v2_assessment says what ran.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from fastapi.concurrency import run_in_threadpool

from app.services import qwen3_asr

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.services.use_cases import EvaluationResult

ENGINE = "qwen3-asr"
# Language names as Qwen3-ASR writes them ("language Chinese<asr_text>...").
ASR_LANGUAGES = {"zh": "Chinese", "ko": "Korean"}
SCORING_VERSIONS = {"zh": "zh-reading-v2", "ko": "ko-reading-v2"}

T = TypeVar("T")
_MEASURE_LOCK = threading.Lock()


class AsrUnavailableError(Exception):
    """Qwen3-ASR could not transcribe the take: nothing to score."""


def measure_locked(measure: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """The audio side, one take at a time: Silero VAD and the lazily loaded
    CTC models are shared, stateful models (run off the event loop)."""
    with _MEASURE_LOCK:
        return measure(*args, **kwargs)


def _alignment(take: Any, ctc_model: str, fallback: str | None) -> dict:
    recognition = take.recognition
    windows = bool(recognition and recognition.windows)
    return {
        "model": ctc_model,
        "method": "ctc_forced_alignment" if windows else (None if take.no_voice else fallback),
        "used_for_timing": windows,
        "transcript": recognition.text if recognition else None,
        "inference_ms": round(recognition.inference_time * 1000, 1) if recognition else None,
    }


async def evaluate_v2(
    take: Any,
    lang_code: str,
    score: Callable[[Any, str, float], "EvaluationResult"],
    settings: "Settings",
    ctc_model: str,
    timing_fallback: str | None = None,
) -> tuple["EvaluationResult", dict]:
    """(the v1-shaped result, v2_assessment) for a measured take.
    `take` is a ChineseTake / KoreanTake (samples, sample_rate, recognition,
    no_voice); `score(take, transcript, inference_ms)` is the language's
    v1 scoring; `timing_fallback` names the timing used without CTC windows
    (Korean splits the speech evenly; Chinese then has none)."""
    language = ASR_LANGUAGES[lang_code]
    asr = {
        "engine": ENGINE, "model": settings.en_ollama_asr_model, "language": language,
        "status": "skipped", "transcript": None, "detected_language": None,
        "calls": 0, "latency_ms": None,
    }
    assessment = {
        "scoring_version": SCORING_VERSIONS[lang_code],
        "status": "no_voice",
        "asr": asr,
        "alignment": _alignment(take, ctc_model, timing_fallback),
    }
    if take.no_voice is not None:
        return take.no_voice, assessment

    try:
        heard = await qwen3_asr.transcribe(take.samples, take.sample_rate, settings, language)
    except qwen3_asr.AsrError as e:
        raise AsrUnavailableError(str(e)) from e
    transcript = heard["transcript"]
    asr.update(status="ok", transcript=transcript, detected_language=heard["language"],
               calls=heard["calls"], latency_ms=heard["latency_ms"])
    assessment["status"] = "scored" if transcript else "no_speech"
    ctc_ms = assessment["alignment"]["inference_ms"] or 0.0
    # Off the event loop: Korean scoring runs the F0 extraction (torchaudio).
    result = await run_in_threadpool(score, take, transcript, round(heard["latency_ms"] + ctc_ms, 1))
    return result, assessment
