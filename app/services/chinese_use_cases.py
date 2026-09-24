"""Mandarin read-aloud matching by toneless pinyin syllable."""
from __future__ import annotations

import logging
from pathlib import Path

from app.services.aspect_scoring import AspectScores
from app.services.chinese_asr import ChineseASRService
from app.services.chinese_text import (
    normalize_chinese, pinyin_syllables, score_chinese_fluency, score_syllables,
)
from app.services.use_cases import EmptyTargetError, EvaluationResult

logger = logging.getLogger(__name__)


class EvaluateChineseUseCase:
    def __init__(self, asr: ChineseASRService):
        self.asr = asr

    def execute(self, target_text: str, audio_path: str | Path) -> EvaluationResult:
        target = normalize_chinese(target_text)
        if not target:
            raise EmptyTargetError("Target text contains no Chinese characters.")
        target_syllables = pinyin_syllables(target)
        recognition = self.asr.recognize(audio_path, align_to=target)
        recognized = normalize_chinese(recognition.text)
        heard_syllables = pinyin_syllables(recognized)
        score = score_syllables(target_syllables, heard_syllables)

        speech_duration = recognition.duration
        pause_count = 0
        speech_ratio = None
        vad_method = None
        measured_pitch: list[dict] = []
        from src.asr.inference import load_audio

        samples, sample_rate = load_audio(audio_path)
        try:
            from app.services.vad_fluency import analyze_vad

            vad = analyze_vad(samples, sample_rate)
            speech_duration = vad.speech_duration or recognition.duration
            pause_count = vad.pause_count
            speech_ratio = vad.speech_ratio
            vad_method = vad.method
        except Exception:  # noqa: BLE001
            logger.warning("Chinese VAD unavailable", exc_info=True)

        if recognition.windows and score.score >= 60:
            try:
                from app.services.pitch_extraction import extract_pitch_for_windows

                points = extract_pitch_for_windows(samples, sample_rate, recognition.windows)
                measured_pitch = [
                    {"mora": char, "semitone": point.get("semitone"),
                     "voiced": bool(point.get("voiced")), "expected": ""}
                    for char, point in zip(target, points)
                ]
            except Exception:  # noqa: BLE001
                logger.warning("Chinese F0 unavailable", exc_info=True)

        fluency = score_chinese_fluency(len(target_syllables), speech_duration, pause_count, score.score)
        aspects = AspectScores(
            overall_score=float(score.score), pronunciation_score=float(score.score),
            fluency_score=fluency, rhythm_score=None, intonation_score=None,
            rhythm_measured=False, intonation_measured=False,
            method="pinyin-asr+vad" if vad_method else "pinyin-asr",
            vad_method=vad_method, pause_count=pause_count, speech_ratio=speech_ratio,
        )
        return EvaluationResult(
            target_text=target_text, target_hiragana=target,
            recognized_text=recognition.text, recognized_hiragana=recognized,
            score=score, aspects=aspects,
            audio_duration=round(recognition.duration, 2),
            inference_ms=round(recognition.inference_time * 1000, 1),
            measured_pitch=measured_pitch,
            target_reading=" ".join(target_syllables),
            recognized_reading=" ".join(heard_syllables),
        )
