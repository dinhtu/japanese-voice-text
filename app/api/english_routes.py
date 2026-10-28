"""English pronunciation page API — same shapes as /api/pronunciation/*."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.routes import _validate_upload
from app.core.config import Settings, get_settings
from app.core.vram import gpu_session
from app.schemas.coaching import CoachResponse, TextGuideResponse
from app.schemas.english_assessment import (
    CoachComment,
    EnglishEvaluateResponse,
    GoptAssessment,
    OllamaAssessment,
    ReadingAssessment,
)
from app.schemas.english_intonation import build_intonation
from app.schemas.english_v2 import EnglishEvaluateV2Response
from app.schemas.pitch_accent import MoraPitchItem, PitchAccentResponse
from app.schemas.pronunciation import MoraStatusItem
from app.services.coaching import (
    SUPPORTED_LANGUAGES,
    CoachingUnavailableError,
    build_facts,
    generate_comment,
    generate_text_reading_guide,
)
from app.services.english_asr import EnglishASRService, get_english_asr_service
from app.services.english_assessment import coach_facts
from app.services.english_ollama_assessment import assess_with_ollama
from app.services.english_intonation import (
    IntonationResult,
    SentenceAnalysis,
    analyze_sentence,
)
from app.services.english_scoring import match_words
from app.services.english_text import word_pitch_pattern
from app.services.english_use_cases import EvaluateEnglishUseCase
from app.services.use_cases import EmptyTargetError

router = APIRouter()
logger = logging.getLogger(__name__)


def get_en_use_case(
    asr: EnglishASRService = Depends(get_english_asr_service),
) -> EvaluateEnglishUseCase:
    return EvaluateEnglishUseCase(asr)


def _word_status(
    target: str,
    recognized: str,
    analysis: SentenceAnalysis | None = None,
    intonation: IntonationResult | None = None,
) -> list[MoraStatusItem]:
    """Same letter alignment + gate as the score, so "x/y từ đúng" agrees with
    it. Each word also carries its syllables, IPA and dictionary stress; with
    `intonation`, whether the learner stressed the right syllable -- only for
    words that were actually said (a misread word has no stress verdict)."""
    matches = match_words(target, recognized)
    if intonation is not None:
        analysis = intonation.analysis
    words = analysis.words if analysis is not None else ()
    # Both sides come from english_text.speech_tokens, so they line up.
    if len(words) != len(matches):
        return [MoraStatusItem(mora=w.target, ok=w.ok) for w in matches]
    stress = {c.word: c for c in intonation.word_stress} if intonation is not None else {}
    items = []
    for match, word in zip(matches, words):
        check = stress.get(word.index) if match.ok else None
        items.append(MoraStatusItem(
            mora=word.text,
            ok=match.ok,
            syllables=[s.text for s in word.pron.syllables],
            ipa=word.ipa,
            # A heteronym shows the dictionary reading the learner used.
            stress_index=check.expected if check else word.pron.stress_index,
            heard_stress_index=check.heard if check else None,
            stress_ok=check.ok if check else None,
        ))
    return items


def _assessment_lang(lang: str | None) -> str:
    """Language of reading_assessment's improvement messages (vi / en)."""
    lang = (lang or "vi").strip().lower()
    return lang if lang in {"vi", "en"} else "en"


def _jsonable(value):
    """json.dumps default: numpy scalars and pydantic models."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.item()


def _coach_lang(lang: str | None) -> str:
    """The /coach language rule: jp -> ja; unsupported -> 400."""
    lang = (lang or "vi").strip().lower()
    lang = "ja" if lang == "jp" else lang
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported lang '{lang}'.")
    return lang


async def _coach_comment(text: str, result, settings: Settings, lang: str, words=None):
    """The /coach comment for an English result already computed (the same
    facts /coach always used); raises CoachingUnavailableError."""
    words = words if words is not None else word_pitch_pattern(text)
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
    try:
        facts.extra_lines = coach_facts(result.reading_assessment)
    except Exception:  # noqa: BLE001 -- extra facts must not cost the comment
        logger.warning("English coach facts failed; using the basic facts", exc_info=True)
    return await generate_comment(facts, settings, lang=lang, target_lang="en")


def _assessment_fields(result) -> dict:
    """reading_assessment / gopt_assessment, validated; a report that does
    not fit its schema is dropped (logged) instead of failing /evaluate."""
    out = {}
    for key, model in (("reading_assessment", ReadingAssessment), ("gopt_assessment", GoptAssessment)):
        data = getattr(result, key, None)
        if data is None:
            continue
        try:
            # Round-trip through JSON so a stray numpy scalar in a free-form
            # detail dict fails here, not later in response serialization.
            plain = json.loads(json.dumps(data, default=lambda v: v.item()))
            out[key] = model.model_validate(plain)
        except Exception:  # noqa: BLE001
            logger.warning("English %s did not validate; omitted", key, exc_info=True)
    return out


async def _ollama_field(result, settings: Settings) -> OllamaAssessment | None:
    """ollama_assessment (Qwen3-ASR via Ollama), or None if even its error
    report cannot be built: it is an optional extra and never fails /evaluate."""
    try:
        data = await assess_with_ollama(
            result.assessment_inputs, result.reading_assessment, result.gopt_assessment, settings
        )
        plain = json.loads(json.dumps(data, default=lambda v: v.item()))
        return OllamaAssessment.model_validate(plain)
    except Exception:  # noqa: BLE001
        logger.warning("English ollama_assessment failed; omitted", exc_info=True)
        return None


@router.post("/evaluate", response_model=EnglishEvaluateResponse)
async def evaluate_english(
    text: str = Form(...),
    audio: UploadFile = File(...),
    lang: str = Form("vi", description="Messages / coach comment language: vi | en | ja"),
    coach: bool = Form(False, description="Also return the /coach comment (same request, no re-scoring)"),
    use_case: EvaluateEnglishUseCase = Depends(get_en_use_case),
    settings: Settings = Depends(get_settings),
) -> EnglishEvaluateResponse:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")
    coach_lang = _coach_lang(lang) if coach else None
    suffix = _validate_upload(audio)
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(status_code=400, detail="Audio exceeds the upload limit.")

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="en_pronunciation_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        with gpu_session():
            result = use_case.execute(text.strip(), tmp_path, lang=_assessment_lang(lang))
            recognized_words = word_pitch_pattern(result.recognized_hiragana)
            response = EnglishEvaluateResponse.from_result(result, recognized_words)
            intonation = result.intonation
            analysis = intonation.analysis if intonation is not None else analyze_sentence(result.target_text)
            update = {
                "mora_status": _word_status(
                    result.target_text, result.recognized_hiragana, analysis, intonation
                ),
            }
            if analysis.words:
                update["intonation"] = build_intonation(analysis, intonation)
            update.update(_assessment_fields(result))
            response = response.model_copy(update=update)
        # After gpu_session: wav2vec2 is offloaded, so Qwen3-ASR has the VRAM.
        ollama = await _ollama_field(result, settings)
        if ollama is not None:
            response = response.model_copy(update={"ollama_assessment": ollama})
        if coach_lang is not None:
            try:
                comment = await _coach_comment(result.target_text, result, settings, coach_lang)
                response = response.model_copy(update={"coach": CoachComment(
                    assessment=comment.assessment, suggestion=comment.suggestion, lang=coach_lang)})
            except CoachingUnavailableError as e:
                response = response.model_copy(update={"coach_error": str(e)})
        return response
    except EmptyTargetError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=f"Could not read the audio: {e}") from e
    except Exception as e:  # noqa: BLE001
        logger.exception("English evaluation failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post(
    "/evaluate-v2",
    response_model=EnglishEvaluateV2Response,
    summary="Chấm đọc tiếng Anh v2: Qwen3-ASR (Ollama) + căn thời gian + nhận xét qwen3 (Ollama)",
)
async def evaluate_english_v2(
    text: str = Form(...),
    audio: UploadFile = File(...),
    lang: str = Form("vi", description="Feedback / improvements language: vi | en | ja"),
    feedback: bool = Form(True, description="Also return the AI comment (`coach`)"),
    asr: EnglishASRService = Depends(get_english_asr_service),
    settings: Settings = Depends(get_settings),
) -> EnglishEvaluateV2Response:
    """en-reading-v2, returned in the /evaluate shape (same fields, values
    from v2) plus `coach` (the comment /coach would give) and
    `v2_assessment` (the full v2 report): a client switches between the two
    endpoints without other changes."""
    from app.services.english_evaluate_v2 import evaluate_v2, legacy_fields, measure_acoustics
    from app.services.english_text import expand_for_speech, normalize_english

    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")
    if not normalize_english(expand_for_speech(text)):
        raise HTTPException(status_code=400, detail="Target text contains no pronounceable English content.")
    lang = _coach_lang(lang)
    suffix = _validate_upload(audio)
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(status_code=400, detail="Audio exceeds the upload limit.")

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="en_v2_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        with gpu_session():
            # wav2vec2 + pitch are CPU/GPU-bound: off the event loop.
            acoustics = await run_in_threadpool(
                measure_acoustics, text, tmp_path,
                asr if settings.en_v2_aligner == "wav2vec2" else None, settings.en_v2_aligner,
            )
        # After gpu_session: wav2vec2 is offloaded before the Ollama models load.
        report = await evaluate_v2(acoustics, settings, lang=lang, with_feedback=feedback)
        if report["status"] == "asr_unavailable":
            # No transcript, no score: an error, as /coach does when Ollama is down.
            raise HTTPException(status_code=503, detail=(report.get("asr") or {}).get("error") or "Qwen3-ASR unavailable")
        data = legacy_fields(report, acoustics)
        analysis = acoustics.analysis
        if analysis is not None and analysis.words:
            data["intonation"] = build_intonation(analysis, acoustics.intonation)
        data["v2_assessment"] = report
        return EnglishEvaluateV2Response.model_validate(
            json.loads(json.dumps(data, default=_jsonable))
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=f"Could not read the audio: {e}") from e
    except Exception as e:  # noqa: BLE001
        logger.exception("English evaluate-v2 failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/pitch-accent", response_model=PitchAccentResponse)
def english_pitch_accent(text: str = Query(..., min_length=1)) -> PitchAccentResponse:
    text = text.strip()
    analysis = analyze_sentence(text)
    if not analysis.words:
        raise HTTPException(status_code=400, detail="No pronounceable English words.")
    # `pattern` / `reading` are exactly what older clients got (one H/L item
    # per word_pitch_pattern word, same split as `measured_pitch`);
    # `intonation` carries syllables, word / sentence stress and the
    # reference contour.
    words = word_pitch_pattern(text)
    pattern = [MoraPitchItem(mora=w.mora, pitch=w.pitch, phrase=w.phrase) for w in words]
    return PitchAccentResponse(
        text=text,
        reading=" ".join(w.mora for w in words),
        pattern=pattern,
        patterns=[pattern],
        intonation=build_intonation(analysis),
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
    if lang == "jp":
        lang = "ja"
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported lang '{lang}'.")
    suffix = _validate_upload(audio)
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(status_code=400, detail="Audio exceeds the upload limit.")

    words = word_pitch_pattern(text)
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="en_coach_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        with gpu_session():
            result = EvaluateEnglishUseCase(asr).execute(text, tmp_path)
        comment = await _coach_comment(text, result, settings, lang, words)
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


@router.get(
    "/text-guide",
    response_model=TextGuideResponse,
    summary="Tạo hướng dẫn phát âm từng cụm từ cho văn bản tiếng Anh bằng Ollama AI",
)
async def get_english_text_guide(
    text: str = Query(..., description="Văn bản/câu tiếng Anh cần hướng dẫn phát âm (ví dụ: How are you today?)"),
    lang: str = Query("vi", description="Ngôn ngữ câu hướng dẫn (jp, en, ko, tw, vi)"),
    settings: Settings = Depends(get_settings),
) -> TextGuideResponse:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")

    try:
        result = await generate_text_reading_guide(text, settings, lang=lang, target_lang="en")
        return TextGuideResponse(text=text, lang=lang, meaning=result.meaning, guide=result.guide)
    except CoachingUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("English text reading guide generation failed")
        raise HTTPException(status_code=500, detail=f"Text guide failed: {e}") from e


@router.post(
    "/text-guide",
    response_model=TextGuideResponse,
    summary="Tạo hướng dẫn phát âm từng cụm từ cho văn bản tiếng Anh bằng Ollama AI (POST)",
)
async def post_english_text_guide(
    text: str = Form(..., description="Văn bản/câu tiếng Anh cần hướng dẫn phát âm"),
    lang: str = Form("vi", description="Ngôn ngữ câu hướng dẫn (jp, en, ko, tw, vi)"),
    settings: Settings = Depends(get_settings),
) -> TextGuideResponse:
    return await get_english_text_guide(text=text, lang=lang, settings=settings)
