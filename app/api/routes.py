"""Pronunciation evaluation endpoints."""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.services.asr_service import ASRService, get_asr_service
from src.asr.inference import load_audio
from app.services.coaching import (
    SUPPORTED_LANGUAGES,
    CoachingUnavailableError,
    build_facts,
    generate_comment,
)
from app.services.mora_timing import resolve_mora_windows
from app.services.normalization import to_hiragana
from app.services.pitch_accent import pitch_accent_pattern
from app.services.pitch_extraction import extract_pitch_for_windows, extract_pitch_per_mora
from app.services.prosody_issues import detect_duration_issues
from app.services.scoring import score_pronunciation
from app.services.use_cases import (
    EmptyTargetError,
    EvaluatePronunciationUseCase,
)
from app.core.config import Settings, get_settings
from app.core.vram import gpu_session
from app.schemas.coaching import CoachResponse
from app.schemas.pronunciation import EvaluateResponse
from app.schemas.pitch_accent import PitchAccentResponse
from app.schemas.pitch_contour import PitchContourResponse

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {
    ".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".aac",
    ".webm", ".weba", ".wma", ".aiff", ".aif", ".mp4",
}
# curl and some browsers send a generic binary type or audio/video mime types.
ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
    "audio/mpeg", "audio/mp3", "audio/x-mp3", "audio/x-mpeg",
    "audio/ogg", "audio/vorbis", "audio/opus", "audio/flac", "audio/x-flac",
    "audio/mp4", "audio/aac", "audio/x-m4a", "audio/m4a", "audio/x-aac",
    "audio/webm", "audio/weba", "audio/x-ms-wma", "audio/aiff", "audio/x-aiff",
    "video/webm", "video/mp4",
    "application/octet-stream", "binary/octet-stream", "",
}


def get_use_case(
    asr_service: ASRService = Depends(get_asr_service),
) -> EvaluatePronunciationUseCase:
    return EvaluatePronunciationUseCase(asr_service)


def _validate_upload(audio: UploadFile) -> str:
    """Validate uploaded audio and return its file extension (defaulting to .wav)."""
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {suffix}. Supported formats: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )
    ctype = (audio.content_type or "").lower().split(";")[0].strip()
    if ctype and not (ctype.startswith("audio/") or ctype.startswith("video/") or ctype in ALLOWED_CONTENT_TYPES):
        raise HTTPException(
            status_code=400, detail=f"Unsupported content type: {audio.content_type}"
        )
    return suffix if suffix in ALLOWED_SUFFIXES else ".wav"


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate Japanese pronunciation against a target text",
)
async def evaluate_pronunciation(
    text: str = Form(..., description="Japanese target text (kanji or kana)"),
    audio: UploadFile = File(..., description="Audio recording of the user reading it"),
    use_case: EvaluatePronunciationUseCase = Depends(get_use_case),
    settings: Settings = Depends(get_settings),
) -> EvaluateResponse:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    suffix = _validate_upload(audio)

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Audio exceeds the {settings.max_audio_bytes // (1024 * 1024)}MB limit.",
        )

    # Temporary file, removed as soon as inference finishes.
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="pronunciation_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        with gpu_session():
            result = use_case.execute(text.strip(), tmp_path)

            # Dictionary H/L of the recognized kana (API/coach). The practice
            # page plots measured_pitch (WAV F0) from EvaluationResult.
            try:
                recognized_moras = pitch_accent_pattern(result.recognized_hiragana)
            except ValueError:
                recognized_moras = []
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Recognized-text pitch pattern failed; "
                    "recognized_pitch_pattern will be empty",
                    exc_info=True,
                )
                recognized_moras = []

            return EvaluateResponse.from_result(result, recognized_moras)

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


@router.get(
    "/pitch-accent",
    response_model=PitchAccentResponse,
    summary="Reference pitch-accent pattern (cao do mau) for a Japanese text",
)
def get_pitch_accent(
    text: str = Query(..., min_length=1, description="Japanese target text (kanji or kana)"),
) -> PitchAccentResponse:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    try:
        moras = pitch_accent_pattern(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Pitch accent extraction failed")
        raise HTTPException(
            status_code=500, detail=f"Pitch accent extraction failed: {e}"
        ) from e

    return PitchAccentResponse.from_result(text, moras)


@router.post(
    "/pitch-contour",
    response_model=PitchContourResponse,
    summary=(
        "Learner's pitch (F0) per mora from a recording, aligned to "
        "/pitch-accent's reference pattern for the same text using the "
        "ASR model's own recognition timing where possible"
    ),
)
async def get_pitch_contour(
    text: str = Form(
        ..., description="Same target text sent to /pitch-accent, so the points line up"
    ),
    audio: UploadFile = File(..., description="WAV recording to analyze"),
    asr_service: ASRService = Depends(get_asr_service),
    settings: Settings = Depends(get_settings),
) -> PitchContourResponse:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    suffix = _validate_upload(audio)

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Audio exceeds the {settings.max_audio_bytes // (1024 * 1024)}MB limit.",
        )

    try:
        moras = pitch_accent_pattern(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    num_morae = len(moras)

    target_hiragana = to_hiragana(text)

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="pitch_contour_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        with gpu_session():
            samples, sample_rate = load_audio(tmp_path)
            duration = round(len(samples) / sample_rate, 2)

            # Force-align the *target* kana (jp-pitch-accent-analyzer style),
            # then fall back to decode+Levenshtein, then equal-time buckets.
            windows = None
            if target_hiragana:
                try:
                    recognition = asr_service.recognize(
                        tmp_path, with_timing=True, align_to=target_hiragana
                    )
                    windows = resolve_mora_windows(
                        target_hiragana,
                        recognition.kana,
                        recognition.char_spans,
                        duration,
                        aligned_char_spans=recognition.aligned_char_spans,
                    ) or None
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "ASR-based pitch alignment failed; falling back to "
                        "equal-time buckets",
                        exc_info=True,
                    )
                    windows = None

            if windows is not None and len(windows) == num_morae:
                points = extract_pitch_for_windows(samples, sample_rate, windows)
            else:
                points = extract_pitch_per_mora(samples, sample_rate, num_morae)
    except RuntimeError as e:
        logger.warning("Audio decode failed: %s", e)
        raise HTTPException(status_code=422, detail=f"Could not read the audio: {e}") from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Pitch contour extraction failed")
        raise HTTPException(
            status_code=500, detail=f"Pitch extraction failed: {e}"
        ) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return PitchContourResponse(duration=duration, points=points)


@router.post(
    "/coach",
    response_model=CoachResponse,
    summary=(
        "Natural-language coaching comment for a recording, in the "
        "requested `lang` (vi | en), written by a locally-run Ollama "
        "model strictly from this app's own measured facts (score, "
        "per-mora errors, sokuon/chouon timing, pitch-accent direction) "
        "-- never given the raw audio, never asked to judge anything "
        "itself"
    ),
)
async def coach_pronunciation(
    text: str = Form(
        ..., description="Same target text sent to /evaluate, so the facts line up"
    ),
    audio: UploadFile = File(..., description="Audio recording to analyze"),
    lang: str = Form(
        "vi",
        description=f"Comment language. One of: {', '.join(SUPPORTED_LANGUAGES)}",
    ),
    asr_service: ASRService = Depends(get_asr_service),
    settings: Settings = Depends(get_settings),
) -> CoachResponse:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    lang = (lang or "vi").strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported lang '{lang}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}",
        )

    suffix = _validate_upload(audio)

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Audio exceeds the {settings.max_audio_bytes // (1024 * 1024)}MB limit.",
        )

    try:
        moras = pitch_accent_pattern(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    target_hiragana = to_hiragana(text)
    if not target_hiragana:
        raise HTTPException(
            status_code=400,
            detail="Target text contains no pronounceable Japanese content.",
        )

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="coach_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        with gpu_session():
            recognition = asr_service.recognize(
                tmp_path, with_timing=True, align_to=target_hiragana
            )
            recognized_hiragana = to_hiragana(recognition.kana)
            score = score_pronunciation(target_hiragana, recognized_hiragana)

        # Sokuon/chouon duration and pitch-direction facts both need real
        # per-mora timing (app.services.mora_timing) -- optional, same
        # graceful-degrade rule as /pitch-contour: if ASR timing isn't
        # available or doesn't line up, the comment is still generated,
        # just without those two fact categories.
        duration_issues: list = []
        pitch_points = None
        if recognition.char_spans is not None or recognition.aligned_char_spans:
            try:
                samples, sample_rate = load_audio(tmp_path)
                duration = len(samples) / sample_rate
                windows = resolve_mora_windows(
                    target_hiragana,
                    recognition.kana,
                    recognition.char_spans,
                    duration,
                    aligned_char_spans=recognition.aligned_char_spans,
                )
                if len(windows) == len(moras):
                    duration_issues = detect_duration_issues(moras, windows)
                    pitch_points = extract_pitch_for_windows(samples, sample_rate, windows)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Timing-dependent coaching facts (duration/pitch) failed; "
                    "comment will be generated without them",
                    exc_info=True,
                )

        facts = build_facts(
            text=text,
            score=score.score,
            level=score.level,
            moras=moras,
            errors=score.errors,
            duration_issues=duration_issues,
            pitch_points=pitch_points,
        )

        comment = await generate_comment(facts, settings, lang=lang)
        return CoachResponse(
            assessment=comment.assessment, suggestion=comment.suggestion, lang=lang
        )

    except CoachingUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except RuntimeError as e:
        logger.warning("Audio decode failed: %s", e)
        raise HTTPException(status_code=422, detail=f"Could not read the audio: {e}") from e
    except FileNotFoundError as e:
        logger.error("Model unavailable: %s", e)
        raise HTTPException(status_code=500, detail="ASR model is not available.") from e
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Coaching comment generation failed")
        raise HTTPException(status_code=500, detail=f"Coaching failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)
