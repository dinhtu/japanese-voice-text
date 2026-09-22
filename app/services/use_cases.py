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
from app.services.mora_timing import mora_time_windows
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
) -> AspectScores:
    """Best-effort extras around the CER score. Never raises to the caller."""
    n_morae = len(split_mora_spans(target_hiragana))
    windows = None
    moras = None
    duration_issues: list = []
    pitch_matched = pitch_total = None
    speech_duration = None

    try:
        from app.services.pitch_accent import pitch_accent_pattern

        moras = pitch_accent_pattern(target_text)
        n_morae = len(moras) or n_morae
    except Exception:  # noqa: BLE001
        logger.warning("Aspect pitch-accent pattern failed; mora count falls back", exc_info=True)
        moras = None

    if recognition.char_spans is not None and target_hiragana:
        try:
            windows = mora_time_windows(
                target_hiragana,
                recognition.kana,
                recognition.char_spans,
                recognition.duration,
            )
            speech_duration = _speech_span(windows)
        except Exception:  # noqa: BLE001
            logger.warning("Aspect mora windows failed", exc_info=True)
            windows = None

    if windows is not None and moras is not None and len(windows) == len(moras):
        try:
            from app.services.prosody_issues import detect_duration_issues
            from src.asr.inference import load_audio
            from app.services.pitch_extraction import extract_pitch_for_windows

            duration_issues = detect_duration_issues(moras, windows)
            samples, sample_rate = load_audio(audio_path)
            points = extract_pitch_for_windows(
                samples,
                sample_rate,
                windows,
                phrases=[m.phrase for m in moras],
                pitch_labels=[m.pitch for m in moras],
            )
            pitch_matched, pitch_total = pitch_direction_match(moras, points)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Aspect duration/pitch measurement failed; those heads degrade",
                exc_info=True,
            )

    return score_aspects(
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
    )


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

        recognition = self.asr_service.recognize(audio_path, with_timing=True)
        recognized_hiragana = to_hiragana(recognition.kana)

        logger.info(
            "target=%s recognized=%s (%.0fms)",
            target_hiragana, recognized_hiragana, recognition.inference_time * 1000,
        )

        score = score_pronunciation(target_hiragana, recognized_hiragana)
        try:
            aspects = _measure_aspects(
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
        )
