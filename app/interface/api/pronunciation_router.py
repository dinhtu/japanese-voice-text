"""Pronunciation evaluation endpoints."""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.application.pronunciation.asr_service import ASRService, get_asr_service
from app.application.pronunciation.use_cases import (
    EmptyTargetError,
    EvaluatePronunciationUseCase,
)
from app.core.config import Settings, get_settings
from app.interface.schemas.pronunciation import EvaluateResponse

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".wav"}
# curl and some browsers send a generic binary type for .wav uploads.
ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
    "application/octet-stream", "binary/octet-stream", "",
}
RIFF_MAGIC = b"RIFF"


def get_use_case(
    asr_service: ASRService = Depends(get_asr_service),
) -> EvaluatePronunciationUseCase:
    return EvaluatePronunciationUseCase(asr_service)


def _validate_upload(audio: UploadFile) -> None:
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only .wav files are accepted.")
    # The declared type is only advisory — the RIFF magic check below is what
    # actually proves the payload is a WAV.
    ctype = (audio.content_type or "").lower().split(";")[0].strip()
    if ctype not in ALLOWED_CONTENT_TYPES and not ctype.startswith("audio/"):
        raise HTTPException(
            status_code=400, detail=f"Unsupported content type: {audio.content_type}"
        )


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate Japanese pronunciation against a target text",
)
async def evaluate_pronunciation(
    text: str = Form(..., description="Japanese target text (kanji or kana)"),
    audio: UploadFile = File(..., description="WAV recording of the user reading it"),
    use_case: EvaluatePronunciationUseCase = Depends(get_use_case),
    settings: Settings = Depends(get_settings),
) -> EvaluateResponse:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    _validate_upload(audio)

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Audio exceeds the {settings.max_audio_bytes // (1024 * 1024)}MB limit.",
        )
    if not content.startswith(RIFF_MAGIC):
        raise HTTPException(status_code=400, detail="File is not a valid WAV (RIFF) file.")

    # Temporary file, removed as soon as inference finishes.
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="pronunciation_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        result = use_case.execute(text.strip(), tmp_path)
        return EvaluateResponse.from_result(result)

    except EmptyTargetError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        logger.warning("Audio decode failed: %s", e)
        raise HTTPException(status_code=422, detail=f"Could not read the audio: {e}") from e
    except FileNotFoundError as e:
        logger.error("Model unavailable: %s", e)
        raise HTTPException(status_code=500, detail="ASR model is not available.") from e
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Pronunciation evaluation failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)
