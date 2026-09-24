"""Pronunciation evaluation use case.

    WAV -> ASR -> recognized kana
                      |
    target text -> normalization <- normalization
                      |
                  CER / score
                      |
                  aspect scores (fluency / rhythm / intonation / overall)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from app.services.aspect_scoring import AspectScores, score_aspects
from app.services.asr_service import ASRService
from app.services.coaching import pitch_direction_match
from app.services.mora_diff import split_mora_spans
from app.services.mora_timing import resolve_mora_windows
from app.services.normalization import to_hiragana
from app.services.scoring import ScoreResult, score_pronunciation

logger = logging.getLogger(__name__)


class EmptyTargetError(ValueError):
    """The target text has no pronounceable content."""


@dataclass
class EvaluationResult:
    target_text: str
    target_hiragana: str
    recognized_text: str
    recognized_hiragana: str
    score: ScoreResult
    aspects: AspectScores
    audio_duration: float
    inference_ms: float
    measured_pitch: list[dict] | None = None


def _speech_span(windows: list[tuple[float, float]]) -> float | None:
    if not windows:
        return None
    start, end = windows[0][0], windows[-1][1]
    span = end - start
    return span if span > 0 else None


def _measure_aspects(
    target_text: str,
    target_hiragana: str,
    score: ScoreResult,
    recognition,
    audio_path: str | Path,
) -> tuple[AspectScores, list[dict]]:
    """Best-effort extras around the CER score. Never raises to the caller."""
    n_morae = len(split_mora_spans(target_hiragana))
    windows = None
    moras = None
    duration_issues: list = []
    pitch_matched = pitch_total = None
    speech_duration = None
    measured_pitch: list[dict] = []

    try:
        from app.services.pitch_accent import pitch_accent_pattern

        moras = pitch_accent_pattern(target_text)
        n_morae = len(moras) or n_morae
    except Exception:  # noqa: BLE001
        logger.warning("Aspect pitch-accent pattern failed; mora count falls back", exc_info=True)
        moras = None

    if target_hiragana:
        try:
            windows = resolve_mora_windows(
                target_hiragana,
                recognition.kana,
                recognition.char_spans,
                recognition.duration,
                aligned_char_spans=recognition.aligned_char_spans,
            ) or None
            speech_duration = _speech_span(windows) if windows else None
        except Exception:  # noqa: BLE001
            logger.warning("Aspect mora windows failed", exc_info=True)
            windows = None

    samples = None
    sample_rate = None
    try:
        from src.asr.inference import load_audio

        samples, sample_rate = load_audio(audio_path)
    except Exception:  # noqa: BLE001
        logger.warning("Aspect audio reload failed", exc_info=True)

    if windows is not None and moras is not None and len(windows) == len(moras):
        try:
            from app.services.prosody_issues import detect_duration_issues

            duration_issues = detect_duration_issues(moras, windows)
        except Exception:  # noqa: BLE001
            logger.warning("Aspect duration issues failed", exc_info=True)

    if moras and samples is not None and sample_rate:
        try:
            from app.services.pitch_extraction import (
                extract_pitch_for_windows,
                extract_pitch_per_mora,
            )

            if windows is not None and len(windows) == len(moras):
                points = extract_pitch_for_windows(samples, sample_rate, windows)
            else:
                points = extract_pitch_per_mora(samples, sample_rate, len(moras))
            pitch_matched, pitch_total = pitch_direction_match(moras, points)
            measured_pitch = [
                {
                    "mora": mora.mora,
                    "semitone": point.get("semitone"),
                    "voiced": bool(point.get("voiced")),
                    "expected": mora.pitch,
                }
                for mora, point in zip(moras, points)
            ]
        except Exception:  # noqa: BLE001
            logger.warning(
                "Measured F0 extraction failed; learner pitch chart will be empty",
                exc_info=True,
            )

    gop_score = None
    try:
        from app.services.gop_scoring import score_gop
        from src.asr.phoneme_converter import JapanesePhonemeConverter

        phones = JapanesePhonemeConverter().text_to_phonemes(target_text).split()
        gop_score = score_gop(getattr(recognition, "phoneme_probs", None), phones)
    except Exception:  # noqa: BLE001
        logger.warning("GOP scoring failed; pronunciation stays CER-only", exc_info=True)

    pause_count = 0
    speech_ratio = None
    vad_method = None
    if samples is not None and sample_rate:
        try:
            from app.services.vad_fluency import analyze_vad

            vad = analyze_vad(samples, sample_rate)
            vad_method = vad.method
            pause_count = vad.pause_count
            speech_ratio = vad.speech_ratio
            if vad.speech_duration > 0:
                speech_duration = vad.speech_duration
        except Exception:  # noqa: BLE001
            logger.warning("VAD fluency failed", exc_info=True)

    pasqa_score = None
    if moras:
        try:
            from app.core.config import get_settings
            from app.services.pasqa_intonation import score_pasqa

            settings = get_settings()
            pasqa_score = score_pasqa(
                audio_path,
                [m.mora for m in moras],
                getattr(settings, "pasqa_checkpoint", None),
                getattr(settings, "pasqa_device", "cpu"),
            )
        except Exception:  # noqa: BLE001
            logger.warning("PASQA scoring failed", exc_info=True)

    aspects = score_aspects(
        pronunciation_cer_score=score.score,
        n_morae=n_morae,
        audio_duration=recognition.duration,
        speech_duration=speech_duration,
        windows=windows,
        moras=moras,
        duration_issues=duration_issues,
        errors=score.errors,
        pitch_matched=pitch_matched,
        pitch_total=pitch_total,
        gop_score=gop_score,
        pause_count=pause_count,
        speech_ratio=speech_ratio,
        vad_method=vad_method,
        pasqa_score=pasqa_score,
    )
    return aspects, measured_pitch


class EvaluatePronunciationUseCase:
    def __init__(self, asr_service: ASRService):
        self.asr_service = asr_service

    def execute(self, target_text: str, audio_path: str | Path) -> EvaluationResult:
        """Run ASR on the audio and score it against the target text.

        Raises:
            EmptyTargetError: Target text normalizes to nothing.
            RuntimeError: Audio could not be decoded.
        """
        target_hiragana = to_hiragana(target_text)
        if not target_hiragana:
            raise EmptyTargetError(
                "Target text contains no pronounceable Japanese content."
            )

        recognition = self.asr_service.recognize(
            audio_path, with_timing=True, align_to=target_hiragana
        )
        recognized_hiragana = to_hiragana(recognition.kana)

        logger.info(
            "target=%s recognized=%s (%.0fms)",
            target_hiragana, recognized_hiragana, recognition.inference_time * 1000,
        )

        score = score_pronunciation(target_hiragana, recognized_hiragana)
        measured_pitch: list[dict] = []
        try:
            aspects, measured_pitch = _measure_aspects(
                target_text,
                target_hiragana,
                score,
                recognition,
                audio_path,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Aspect scoring failed; falling back to CER-only mix", exc_info=True)
            aspects = score_aspects(
                pronunciation_cer_score=score.score,
                n_morae=len(split_mora_spans(target_hiragana)),
                audio_duration=recognition.duration,
                errors=score.errors,
            )

        return EvaluationResult(
            target_text=target_text,
            target_hiragana=target_hiragana,
            recognized_text=recognition.kana,
            recognized_hiragana=recognized_hiragana,
            score=score,
            aspects=aspects,
            audio_duration=round(recognition.duration, 2),
            inference_ms=round(recognition.inference_time * 1000, 1),
            measured_pitch=measured_pitch,
        )
