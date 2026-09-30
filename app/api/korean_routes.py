"""Korean pronunciation API, matching the existing language response shape."""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.routes import _validate_upload
from app.core.config import Settings, get_settings
from app.core.vram import gpu_session
from app.schemas.coaching import CoachResponse, TextGuideResponse
from app.schemas.pronunciation import EvaluateResponse, MoraStatusItem
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
from app.services.korean_use_cases import EvaluateKoreanUseCase
from app.services.use_cases import EmptyTargetError

router = APIRouter()
logger = logging.getLogger(__name__)


def _check_text(text: str) -> str:
    text = text.strip()
    if not normalize_korean(text):
        raise HTTPException(400, "Field 'text' must contain Hangul syllables.")
    return text


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
    summary="Lấy cao độ mẫu Accentual Phrase (H/L) của câu tiếng Hàn",
)
def korean_pitch_accent(text: str = Query(..., min_length=1, max_length=200)) -> dict:
    text = _check_text(text)
    pattern = korean_pitch_pattern(text)
    if not pattern:
        raise HTTPException(400, "No Hangul characters found.")
    return {
        "success": True,
        "text": text,
        "reading": " ".join(item["mora"] for item in pattern),
        "pattern": pattern,
    }


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_korean(
    text: str = Form(...),
    audio: UploadFile = File(...),
    asr: KoreanASRService = Depends(get_korean_asr_service),
    settings: Settings = Depends(get_settings),
) -> EvaluateResponse:
    text = _check_text(text)
    path = await _save_upload(audio, settings, "ko_pronunciation_")
    try:
        with gpu_session():
            result = EvaluateKoreanUseCase(asr).execute(text, path)
        response = EvaluateResponse.from_result(result)
        target_chars = list(normalize_korean(text))
        bad = {e.position for e in result.score.errors if e.type in ("sub", "del")}
        rec_pitch = korean_pitch_pattern(result.recognized_text) if result.recognized_text else []
        return response.model_copy(
            update={
                "mora_status": [
                    MoraStatusItem(mora=c, ok=i not in bad) for i, c in enumerate(target_chars)
                ],
                "recognized_pitch_pattern": rec_pitch,
            }
        )
    except (EmptyTargetError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(422, f"Could not read the audio: {exc}") from exc
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
    lang = lang.strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported lang '{lang}'.")
    path = await _save_upload(audio, settings, "ko_coach_")
    try:
        with gpu_session():
            result = EvaluateKoreanUseCase(asr).execute(text, path)
        units = [WordPitch(mora=c, pitch="L") for c in normalize_korean(text)]
        facts = build_facts(
            text,
            result.score.score,
            result.score.level,
            units,
            result.score.errors,
            [],
            None,
        )
        comment = await generate_comment(facts, settings, lang=lang, target_lang="ko")
        return CoachResponse(
            assessment=comment.assessment,
            suggestion=comment.suggestion,
            lang=lang,
        )
    except CoachingUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
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
        guide = await generate_text_reading_guide(text, settings, lang=lang, target_lang="ko")
        return TextGuideResponse(text=text, lang=lang, guide=guide)
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
