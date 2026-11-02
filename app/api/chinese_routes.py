"""Chinese pronunciation API: the /evaluate shape plus per-syllable tone checks."""
from __future__ import annotations
import logging
import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from app.api.routes import _validate_upload
from app.core.config import Settings, get_settings
from app.core.vram import gpu_session
from app.schemas.chinese import ChineseEvaluateResponse
from app.schemas.coaching import CoachComment, CoachResponse, TextGuideResponse
from app.schemas.pronunciation import EvaluateResponse, MoraStatusItem
from app.services.coaching import (
    SUPPORTED_LANGUAGES, CoachingUnavailableError, build_facts, generate_comment,
    generate_text_reading_guide,
)
from app.services.chinese_asr import ChineseASRService, get_chinese_asr_service
from app.services.chinese_text import chinese_pitch_pattern, normalize_chinese
from app.services.chinese_use_cases import EvaluateChineseUseCase
from app.services.use_cases import EmptyTargetError, EvaluationResult

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/reading")
def chinese_reading(text: str = Query(..., min_length=1, max_length=200)) -> dict[str, str]:
    from pypinyin import Style, lazy_pinyin

    target = normalize_chinese(text)
    if not target:
        raise HTTPException(400, "No Chinese characters found.")
    return {"reading": " ".join(lazy_pinyin(target, style=Style.TONE))}


@router.get("/pitch-accent")
def chinese_pitch_accent(text: str = Query(..., min_length=1, max_length=200)) -> dict:
    pattern = chinese_pitch_pattern(text)
    if not pattern:
        raise HTTPException(400, "No Chinese characters found.")
    return {
        "success": True,
        "text": text.strip(),
        "reading": " ".join(item["mora"] for item in pattern),
        "pattern": pattern,
    }

def _coach_lang(lang: str | None) -> str:
    """The /coach language rule: jp -> ja; unsupported -> 400."""
    lang = (lang or "vi").strip().lower()
    lang = "ja" if lang == "jp" else lang
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported lang '{lang}'.")
    return lang


def _target(text: str) -> str:
    text = text.strip()
    if not normalize_chinese(text):
        raise HTTPException(400, "Field 'text' must contain Chinese characters.")
    return text


async def _save_upload(audio: UploadFile, settings: Settings, prefix: str) -> str:
    """The upload in a temp file (the caller deletes it); 400 if unusable."""
    suffix = _validate_upload(audio)
    content = await audio.read()
    if not content:
        raise HTTPException(400, "Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(400, "Audio exceeds the upload limit.")
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return tmp_path


def _score(asr: ChineseASRService, text: str, path: str) -> EvaluationResult:
    with gpu_session():
        return EvaluateChineseUseCase(asr).execute(text, path)


def _tone_facts(result: EvaluationResult) -> list[str]:
    """The tone checks, for the coach (the score itself ignores tones)."""
    if result.aspects.method.startswith("no-voice"):
        return ["Không nghe thấy giọng đọc trong bản ghi (im lặng, quá nhỏ hoặc chỉ thì thầm): "
                "chưa chấm được gì. Hãy nhắc người học đọc to, rõ, gần micro rồi thử lại."]
    judged = [t for t in result.tones or [] if t["ok"] is not None]
    if not judged:
        return ["Thanh điệu: không đo được (thiếu cao độ hoặc không dóng được thời gian từng âm tiết)."]
    bad = [t for t in judged if not t["ok"]]
    line = f"Thanh điệu (đo từ đường F0 từng âm tiết): đúng {len(judged) - len(bad)}/{len(judged)} âm tiết"
    if bad:
        line += "; lệch: " + ", ".join(
            f"{t['hanzi']} {t['pinyin']} (cần thanh {t['surface_tone']}"
            + (f", nghe giống thanh {t['heard_tone']})" if t["heard_tone"] not in (None, t["surface_tone"]) else ")")
            for t in bad[:8]
        )
    return [line + "."]


async def _coach_comment(result: EvaluationResult, settings: Settings, lang: str):
    """The /coach comment for a result already scored; raises CoachingUnavailableError."""
    from app.services.english_text import WordPitch

    units = [WordPitch(mora=s, pitch="L") for s in (result.target_reading or "").split()]
    facts = build_facts(
        text=result.target_text, score=result.score.score, level=result.score.level,
        moras=units, errors=result.score.errors, duration_issues=[], pitch_points=None,
    )
    facts.extra_lines = _tone_facts(result)
    return await generate_comment(facts, settings, lang=lang, target_lang="zh")


@router.post("/evaluate", response_model=ChineseEvaluateResponse)
async def evaluate_chinese(
    text: str = Form(...), audio: UploadFile = File(...),
    lang: str = Form("vi", description="Coach comment language: vi | en | ja | ko | tw"),
    coach: bool = Form(False, description="Also return the /coach comment (same request, no re-scoring)"),
    asr: ChineseASRService = Depends(get_chinese_asr_service),
    settings: Settings = Depends(get_settings),
) -> ChineseEvaluateResponse:
    text = _target(text)
    coach_lang = _coach_lang(lang) if coach else None
    tmp_path = await _save_upload(audio, settings, "zh_pronunciation_")
    try:
        result = _score(asr, text, tmp_path)
        bad = {e.position for e in result.score.errors if e.type in ("sub", "del")}
        response = ChineseEvaluateResponse.model_validate({
            **EvaluateResponse.from_result(result).model_dump(),
            "mora_status": [MoraStatusItem(mora=c, ok=i not in bad) for i, c in enumerate(result.target_hiragana)],
            "tones": result.tones or [],
        })
        if coach_lang is not None:
            try:
                comment = await _coach_comment(result, settings, coach_lang)
                response.coach = CoachComment(
                    assessment=comment.assessment, suggestion=comment.suggestion, lang=coach_lang)
            except CoachingUnavailableError as e:
                response.coach_error = str(e)
        return response
    except (EmptyTargetError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.exception("Chinese evaluate failed")
        raise HTTPException(500, f"Inference failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/coach", response_model=CoachResponse)
async def coach_chinese(
    text: str = Form(...), audio: UploadFile = File(...), lang: str = Form("vi"),
    asr: ChineseASRService = Depends(get_chinese_asr_service),
    settings: Settings = Depends(get_settings),
) -> CoachResponse:
    text = _target(text)
    lang = _coach_lang(lang)
    tmp_path = await _save_upload(audio, settings, "zh_coach_")
    try:
        comment = await _coach_comment(_score(asr, text, tmp_path), settings, lang)
        return CoachResponse(assessment=comment.assessment, suggestion=comment.suggestion, lang=lang)
    except CoachingUnavailableError as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.exception("Chinese coach failed")
        raise HTTPException(500, f"Coaching failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get(
    "/text-guide",
    response_model=TextGuideResponse,
    summary="Tạo hướng dẫn phát âm từng cụm từ cho văn bản tiếng Trung bằng Ollama AI",
)
async def get_chinese_text_guide(
    text: str = Query(..., description="Văn bản/câu tiếng Trung cần hướng dẫn phát âm (ví dụ: 你好，我是学生。)"),
    lang: str = Query("vi", description="Ngôn ngữ câu hướng dẫn (jp, en, ko, tw, vi)"),
    settings: Settings = Depends(get_settings),
) -> TextGuideResponse:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    try:
        result = await generate_text_reading_guide(text, settings, lang=lang, target_lang="zh")
        return TextGuideResponse(text=text, lang=lang, meaning=result.meaning, guide=result.guide)
    except CoachingUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Chinese text reading guide generation failed")
        raise HTTPException(status_code=500, detail=f"Text guide failed: {e}") from e


@router.post(
    "/text-guide",
    response_model=TextGuideResponse,
    summary="Tạo hướng dẫn phát âm từng cụm từ cho văn bản tiếng Trung bằng Ollama AI (POST)",
)
async def post_chinese_text_guide(
    text: str = Form(..., description="Văn bản/câu tiếng Trung cần hướng dẫn phát âm"),
    lang: str = Form("vi", description="Ngôn ngữ câu hướng dẫn (jp, en, ko, tw, vi)"),
    settings: Settings = Depends(get_settings),
) -> TextGuideResponse:
    return await get_chinese_text_guide(text=text, lang=lang, settings=settings)
