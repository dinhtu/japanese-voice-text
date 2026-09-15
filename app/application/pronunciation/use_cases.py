"""Pronunciation evaluation use case.

    WAV -> ASR -> recognized kana
                      |
    target text -> normalization <- normalization
                      |
                  CER / score
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from app.application.pronunciation.asr_service import ASRService
from app.application.pronunciation.normalization import to_hiragana
from app.application.pronunciation.scoring import ScoreResult, score_pronunciation

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
    audio_duration: float
    inference_ms: float


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

        recognition = self.asr_service.recognize(audio_path)
        recognized_hiragana = to_hiragana(recognition.kana)

        logger.info(
            "target=%s recognized=%s (%.0fms)",
            target_hiragana, recognized_hiragana, recognition.inference_time * 1000,
        )

        return EvaluationResult(
            target_text=target_text,
            target_hiragana=target_hiragana,
            recognized_text=recognition.kana,
            recognized_hiragana=recognized_hiragana,
            score=score_pronunciation(target_hiragana, recognized_hiragana),
            audio_duration=round(recognition.duration, 2),
            inference_ms=round(recognition.inference_time * 1000, 1),
        )
