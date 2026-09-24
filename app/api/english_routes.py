"""English pronunciation page API — same shapes as /api/pronunciation/*."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.routes import RIFF_MAGIC, _validate_upload
from app.core.config import Settings, get_settings
from app.core.vram import gpu_session
from app.schemas.coaching import CoachResponse
from app.schemas.pitch_accent import MoraPitchItem, PitchAccentResponse
from app.schemas.pronunciation import EvaluateResponse, MoraStatusItem
from app.services.coaching import (
    SUPPORTED_LANGUAGES,
    CoachingUnavailableError,
    build_facts,
    generate_comment,
)
from app.services.english_asr import EnglishASRService, get_english_asr_service
from app.services.english_text import english_words, word_pitch_pattern
from app.services.english_use_cases import EvaluateEnglishUseCase
from app.services.use_cases import EmptyTargetError

router = APIRouter()
logger = logging.getLogger(__name__)


def get_en_use_case(
    asr: EnglishASRService = Depends(get_english_asr_service),
) -> EvaluateEnglishUseCase:
    return EvaluateEnglishUseCase(asr)


def _word_status(target: str, recognized: str) -> list[MoraStatusItem]:
    tw, rw = english_words(target), english_words(recognized)
    items = []
    for i, w in enumerate(tw):
        ok = i < len(rw) and rw[i] == w
        items.append(MoraStatusItem(mora=w, ok=ok))
    return items


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_english(
    text: str = Form(...),
    audio: UploadFile = File(...),
    use_case: EvaluateEnglishUseCase = Depends(get_en_use_case),
    settings: Settings = Depends(get_settings),
) -> EvaluateResponse:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")
    _validate_upload(audio)
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(status_code=400, detail="Audio exceeds the upload limit.")
    if not content.startswith(RIFF_MAGIC):
        raise HTTPException(status_code=400, detail="File is not a valid WAV (RIFF) file.")

    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="en_pronunciation_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        with gpu_session():
            result = use_case.execute(text.strip(), tmp_path)
            recognized_words = word_pitch_pattern(result.recognized_hiragana)
            response = EvaluateResponse.from_result(result, recognized_words)
            return response.model_copy(
                update={"mora_status": _word_status(result.target_text, result.recognized_hiragana)}
            )
    except EmptyTargetError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=f"Could not read the audio: {e}") from e
    except Exception as e:  # noqa: BLE001
        logger.exception("English evaluation failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/pitch-accent", response_model=PitchAccentResponse)
def english_pitch_accent(text: str = Query(..., min_length=1)) -> PitchAccentResponse:
    text = text.strip()
    words = word_pitch_pattern(text)
    if not words:
        raise HTTPException(status_code=400, detail="No pronounceable English words.")
    return PitchAccentResponse(
        text=text,
        reading=" ".join(w.mora for w in words),
        pattern=[MoraPitchItem(mora=w.mora, pitch=w.pitch, phrase=w.phrase) for w in words],
    )


@router.post("/coach", response_model=CoachResponse)
async def coach_english(
    text: str = Form(...),
    audio: UploadFile = File(...),
    lang: str = Form("vi"),
    asr: EnglishASRService = Depends(get_english_asr_service),
    settings: Settings = Depends(get_settings),
) -> CoachResponse:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")
    lang = (lang or "vi").strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported lang '{lang}'.")
    _validate_upload(audio)
    content = await audio.read()
    if not content or not content.startswith(RIFF_MAGIC):
        raise HTTPException(status_code=400, detail="Need a non-empty WAV file.")

    words = word_pitch_pattern(text)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="en_coach_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        with gpu_session():
            result = EvaluateEnglishUseCase(asr).execute(text, tmp_path)
        pitch_points = [
            {"semitone": p.get("semitone"), "voiced": p.get("voiced")}
            for p in (result.measured_pitch or [])
        ]
        facts = build_facts(
            text=text,
            score=result.score.score,
            level=result.score.level,
            moras=words,
            errors=result.score.errors,
            duration_issues=[],
            pitch_points=pitch_points or None,
        )
        comment = await generate_comment(
            facts, settings, lang=lang, target_lang="en"
        )
        return CoachResponse(
            assessment=comment.assessment, suggestion=comment.suggestion, lang=lang
        )
    except CoachingUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except EmptyTargetError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("English coaching failed")
        raise HTTPException(status_code=500, detail=f"Coaching failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)
