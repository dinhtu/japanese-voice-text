"""Korean pronunciation API, matching the existing language response shape."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.routes import _validate_upload
from app.core.config import Settings, get_settings
from app.core.vram import gpu_session
from app.schemas.coaching import CoachComment, CoachResponse, TextGuideResponse
from app.schemas.korean import KoreanEvaluateResponse, KoreanEvaluateV2Response
from app.schemas.pitch_accent import MoraPitchItem, PitchAccentResponse
from app.schemas.pronunciation import MoraStatusItem
from app.services.cjk_evaluate_v2 import AsrUnavailableError, evaluate_v2, measure_locked
from app.services.coaching import (
    SUPPORTED_LANGUAGES,
    CoachingUnavailableError,
    build_facts,
    generate_comment,
    generate_text_reading_guide,
)
from app.services.english_text import WordPitch
from app.services.korean_asr import KoreanASRService, get_korean_asr_service
from app.services.korean_text import (
    korean_pitch_pattern,
    korean_pronunciation,
    normalize_korean,
)
from app.services.korean_use_cases import EvaluateKoreanUseCase, score_korean
from app.services.use_cases import EmptyTargetError, EvaluationResult
from app.services.voice_presence import NO_VOICE_FACT

router = APIRouter()
logger = logging.getLogger(__name__)


def _check_text(text: str) -> str:
    text = text.strip()
    if not normalize_korean(text):
        raise HTTPException(400, "Field 'text' must contain Hangul syllables.")
    return text


def _coach_lang(lang: str | None) -> str:
    """The /coach language rule: jp -> ja; unsupported -> 400."""
    lang = (lang or "vi").strip().lower()
    lang = "ja" if lang == "jp" else lang
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported lang '{lang}'.")
    return lang


async def _save_upload(audio: UploadFile, settings: Settings, prefix: str) -> str:
    suffix = _validate_upload(audio)
    content = await audio.read()
    if not content:
        raise HTTPException(400, "Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(400, "Audio exceeds the upload limit.")
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    with os.fdopen(fd, "wb") as file:
        file.write(content)
    return path


@router.get(
    "/reading",
    summary="Lấy cách đọc chuẩn theo quy tắc phát âm tiếng Hàn",
)
def korean_reading(text: str = Query(..., min_length=1, max_length=200)) -> dict[str, str]:
    text = _check_text(text)
    reading = korean_pronunciation(text)
    return {"reading": reading}


@router.get(
    "/pitch-accent",
    response_model=PitchAccentResponse,
    summary="Lấy cao độ mẫu Accentual Phrase (H/L) của câu tiếng Hàn",
)
def korean_pitch_accent(text: str = Query(..., min_length=1, max_length=200)) -> PitchAccentResponse:
    text = _check_text(text)
    pattern = korean_pitch_pattern(text)
    if not pattern:
        raise HTTPException(400, "No Hangul characters found.")
    return PitchAccentResponse(
        success=True,
        text=text,
        reading=" ".join(item["mora"] for item in pattern),
        pattern=[
            MoraPitchItem(mora=item["mora"], pitch=item["pitch"], phrase=item["phrase"])
            for item in pattern
        ],
    )


def _score(asr: KoreanASRService, text: str, path: str) -> EvaluationResult:
    with gpu_session():
        return EvaluateKoreanUseCase(asr).execute(text, path)


def _response(result: EvaluationResult, cls: type[KoreanEvaluateResponse] = KoreanEvaluateResponse,
              **extra) -> KoreanEvaluateResponse:
    """The /evaluate shape: the recognized text's AP pattern and a
    mora_status item per Hangul syllable (`extra`: the v2 fields)."""
    rec_moras = [
        MoraPitchItem(mora=item["mora"], pitch=item["pitch"], phrase=item["phrase"])
        # recognized_hiragana: the transcript with its numbers spelled out.
        for item in (korean_pitch_pattern(result.recognized_hiragana) if result.recognized_text else [])
    ]
    data = cls.from_result(result, recognized_moras=rec_moras).model_dump()
    bad = {e.position for e in result.score.errors if e.type in ("sub", "del")}
    return cls.model_validate({
        **data,
        "mora_status": [MoraStatusItem(mora=c, ok=i not in bad) for i, c in enumerate(result.target_hiragana)],
        **extra,
    })


async def _coach_comment(result: EvaluationResult, settings: Settings, lang: str):
    """The /coach comment for a result already scored; raises CoachingUnavailableError."""
    units = [WordPitch(mora=c, pitch="L") for c in result.target_hiragana]
    facts = build_facts(
        result.target_text,
        result.score.score,
        result.score.level,
        units,
        result.score.errors,
        [],
        None,
    )
    if result.aspects.method.startswith("no-voice"):
        # Nothing heard (v2 only): the no-voice line alone, not every syllable as missing.
        facts.extra_lines = [NO_VOICE_FACT]
        facts.facts_only = True
    return await generate_comment(facts, settings, lang=lang, target_lang="ko")


async def _add_coach(response: KoreanEvaluateResponse, result: EvaluationResult,
                     settings: Settings, lang: str) -> None:
    """`coach` from the result just scored, or `coach_error` when Ollama fails."""
    try:
        comment = await _coach_comment(result, settings, lang)
        response.coach = CoachComment(assessment=comment.assessment, suggestion=comment.suggestion, lang=lang)
    except CoachingUnavailableError as exc:
        response.coach_error = str(exc)


@router.post("/evaluate", response_model=KoreanEvaluateResponse)
async def evaluate_korean(
    text: str = Form(...),
    audio: UploadFile = File(...),
    lang: str = Form("vi", description="Coach comment language: vi | en | ja | ko | tw"),
    coach: bool = Form(False, description="Also return the /coach comment (same request, no re-scoring)"),
    asr: KoreanASRService = Depends(get_korean_asr_service),
    settings: Settings = Depends(get_settings),
) -> KoreanEvaluateResponse:
    text = _check_text(text)
    coach_lang = _coach_lang(lang) if coach else None
    path = await _save_upload(audio, settings, "ko_pronunciation_")
    try:
        result = _score(asr, text, path)
        response = _response(result)
        if coach_lang is not None:
            await _add_coach(response, result, settings, coach_lang)
        return response
    except (EmptyTargetError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(422, f"Could not read the audio: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Korean evaluation failed")
        raise HTTPException(500, f"Inference failed: {exc}") from exc
    finally:
        Path(path).unlink(missing_ok=True)


@router.post(
    "/evaluate-v2",
    response_model=KoreanEvaluateV2Response,
    summary="Chấm đọc tiếng Hàn v2: transcript từ Qwen3-ASR (Ollama) + nhận xét AI",
)
async def evaluate_korean_v2(
    text: str = Form(...),
    audio: UploadFile = File(...),
    lang: str = Form("vi", description="Coach comment language: vi | en | ja | ko | tw"),
    feedback: bool = Form(True, description="Also return the AI comment (`coach`)"),
    asr: KoreanASRService = Depends(get_korean_asr_service),
    settings: Settings = Depends(get_settings),
) -> KoreanEvaluateV2Response:
    """The /evaluate fields scored on Qwen3-ASR's transcript (a CTC model,
    when configured, only times the syllables), plus `coach` and
    `v2_assessment`: a client switches endpoints without other changes."""
    text = _check_text(text)
    lang = _coach_lang(lang)
    path = await _save_upload(audio, settings, "ko_v2_")
    try:
        with gpu_session():
            take = await run_in_threadpool(measure_locked, EvaluateKoreanUseCase(asr).measure, text, path)
        # After gpu_session: the local model is offloaded before Ollama loads Qwen3-ASR.
        result, assessment = await evaluate_v2(
            take, "ko", score_korean, settings, settings.ko_asr_model, timing_fallback="even_split")
        response = _response(result, KoreanEvaluateV2Response, v2_assessment=assessment)
        if feedback:
            await _add_coach(response, result, settings, lang)
        return response
    except AsrUnavailableError as exc:
        # No transcript, no score: an error, as English /evaluate-v2.
        raise HTTPException(503, str(exc) or "Qwen3-ASR unavailable") from exc
    except (EmptyTargetError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(422, f"Could not read the audio: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Korean evaluate-v2 failed")
        raise HTTPException(500, f"Inference failed: {exc}") from exc
    finally:
        Path(path).unlink(missing_ok=True)


@router.post("/coach", response_model=CoachResponse)
async def coach_korean(
    text: str = Form(...),
    audio: UploadFile = File(...),
    lang: str = Form("vi"),
    asr: KoreanASRService = Depends(get_korean_asr_service),
    settings: Settings = Depends(get_settings),
) -> CoachResponse:
    text = _check_text(text)
    lang = _coach_lang(lang)
    path = await _save_upload(audio, settings, "ko_coach_")
    try:
        comment = await _coach_comment(_score(asr, text, path), settings, lang)
        return CoachResponse(
            assessment=comment.assessment,
            suggestion=comment.suggestion,
            lang=lang,
        )
    except CoachingUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except (EmptyTargetError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(422, f"Could not read the audio: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Korean coaching failed")
        raise HTTPException(500, f"Coaching failed: {exc}") from exc
    finally:
        Path(path).unlink(missing_ok=True)


@router.get(
    "/text-guide",
    response_model=TextGuideResponse,
    summary="Tạo hướng dẫn phát âm từng cụm từ cho văn bản tiếng Hàn bằng Ollama AI",
)
async def get_korean_text_guide(
    text: str = Query(..., description="Văn bản/câu tiếng Hàn cần hướng dẫn phát âm (ví dụ: 안녕하세요.)"),
    lang: str = Query("vi", description="Ngôn ngữ câu hướng dẫn (jp, en, ko, tw, vi)"),
    settings: Settings = Depends(get_settings),
) -> TextGuideResponse:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    try:
        result = await generate_text_reading_guide(text, settings, lang=lang, target_lang="ko")
        return TextGuideResponse(text=text, lang=lang, meaning=result.meaning, guide=result.guide)
    except CoachingUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Korean text reading guide generation failed")
        raise HTTPException(status_code=500, detail=f"Text guide failed: {e}") from e


@router.post(
    "/text-guide",
    response_model=TextGuideResponse,
    summary="Tạo hướng dẫn phát âm từng cụm từ cho văn bản tiếng Hàn bằng Ollama AI (POST)",
)
async def post_korean_text_guide(
    text: str = Form(..., description="Văn bản/câu tiếng Hàn cần hướng dẫn phát âm"),
    lang: str = Form("vi", description="Ngôn ngữ câu hướng dẫn (jp, en, ko, tw, vi)"),
    settings: Settings = Depends(get_settings),
) -> TextGuideResponse:
    return await get_korean_text_guide(text=text, lang=lang, settings=settings)
