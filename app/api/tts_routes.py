"""Reference audio for the "Nghe mẫu" button, proxied to the Qwen3-TTS
service so its API key never reaches the browser."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Form, HTTPException
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.services.tts_client import (
    SUPPORTED_TTS_LANGS,
    TtsFailedError,
    TtsUnavailableError,
    generate_speech,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class TtsResponse(BaseModel):
    success: bool = True
    text: str
    lang: str
    audio_url: str
    duration_seconds: float | None = None


@router.post("", response_model=TtsResponse, summary="Tạo audio đọc mẫu cho câu cần luyện (Qwen3-TTS)")
async def create_tts(
    text: str = Form(..., max_length=500, description="Câu cần đọc mẫu"),
    lang: str = Form("ja", description="Ngôn ngữ của câu: ja | en | zh | ko"),
    settings: Settings = Depends(get_settings),
) -> TtsResponse:
    text = " ".join(text.split())
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")
    lang = (lang or "ja").strip().lower()
    if lang not in SUPPORTED_TTS_LANGS:
        raise HTTPException(status_code=400, detail=f"Unsupported lang '{lang}'.")

    try:
        audio = await generate_speech(text, lang, settings)
    except TtsUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except TtsFailedError as e:
        logger.warning("TTS generation failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e
    return TtsResponse(text=text, lang=lang, audio_url=audio.audio_url, duration_seconds=audio.duration_seconds)
