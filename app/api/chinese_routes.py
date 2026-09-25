"""Chinese pronunciation API; same response shape as the English endpoint."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from app.api.routes import _validate_upload
from app.core.config import Settings, get_settings
from app.core.vram import gpu_session
from app.schemas.pronunciation import EvaluateResponse, MoraStatusItem
from app.schemas.coaching import CoachResponse
from app.services.coaching import (
    SUPPORTED_LANGUAGES, CoachingUnavailableError, build_facts, generate_comment,
)
from app.services.chinese_asr import ChineseASRService, get_chinese_asr_service
from app.services.chinese_text import normalize_chinese
from app.services.chinese_use_cases import EvaluateChineseUseCase
from app.services.use_cases import EmptyTargetError

router = APIRouter()


@router.get("/reading")
def chinese_reading(text: str = Query(..., min_length=1, max_length=200)) -> dict[str, str]:
    from pypinyin import Style, lazy_pinyin

    target = normalize_chinese(text)
    if not target:
        raise HTTPException(400, "No Chinese characters found.")
    return {"reading": " ".join(lazy_pinyin(target, style=Style.TONE))}

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_chinese(text: str = Form(...), audio: UploadFile = File(...),
                           asr: ChineseASRService = Depends(get_chinese_asr_service),
                           settings: Settings = Depends(get_settings)) -> EvaluateResponse:
    text = text.strip()
    if not normalize_chinese(text):
        raise HTTPException(400, "Field 'text' must contain Chinese characters.")
    suffix = _validate_upload(audio)
    content = await audio.read()
    if not content:
        raise HTTPException(400, "Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(400, "Audio exceeds the upload limit.")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="zh_pronunciation_")
    try:
        with os.fdopen(fd, "wb") as f: f.write(content)
        with gpu_session():
            result = EvaluateChineseUseCase(asr).execute(text, tmp_path)
        response = EvaluateResponse.from_result(result)
        bad = {e.position for e in result.score.errors if e.type in ("sub", "del")}
        return response.model_copy(update={
            "mora_status": [MoraStatusItem(mora=c, ok=i not in bad) for i, c in enumerate(result.target_hiragana)]
        })
    except (EmptyTargetError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/coach", response_model=CoachResponse)
async def coach_chinese(
    text: str = Form(...), audio: UploadFile = File(...), lang: str = Form("vi"),
    asr: ChineseASRService = Depends(get_chinese_asr_service),
    settings: Settings = Depends(get_settings),
) -> CoachResponse:
    text = text.strip()
    if not normalize_chinese(text):
        raise HTTPException(400, "Field 'text' must contain Chinese characters.")
    lang = lang.strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported lang '{lang}'.")
    suffix = _validate_upload(audio)
    content = await audio.read()
    if not content:
        raise HTTPException(400, "Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(400, "Audio exceeds the upload limit.")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="zh_coach_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        with gpu_session():
            result = EvaluateChineseUseCase(asr).execute(text, tmp_path)
        from app.services.english_text import WordPitch
        units = [WordPitch(mora=s, pitch="L") for s in result.target_reading.split()]
        facts = build_facts(
            text=text, score=result.score.score, level=result.score.level,
            moras=units, errors=result.score.errors, duration_issues=[], pitch_points=None,
        )
        comment = await generate_comment(facts, settings, lang=lang, target_lang="zh")
        return CoachResponse(assessment=comment.assessment, suggestion=comment.suggestion, lang=lang)
    except CoachingUnavailableError as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Coaching failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)
