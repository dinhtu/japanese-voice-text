"""Mandarin read-aloud matching by toneless pinyin syllable."""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from app.services.aspect_scoring import score_aspects
from app.services.chinese_asr import ChineseASRService
from app.services.chinese_text import (
    chinese_pitch_pattern, chinese_tone_matches, normalize_chinese,
    pinyin_syllables, score_syllables,
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
        pitch_contours: list[list[float | None]] = []
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

        if recognition.windows:
            try:
                from statistics import median

                from app.services.pitch_extraction import extract_mandarin_tone_contours

                pitch_contours = extract_mandarin_tone_contours(
                    samples, sample_rate, recognition.windows
                )
                points = []
                for contour in pitch_contours:
                    voiced = [float(value) for value in contour if value is not None]
                    points.append({
                        "semitone": round(median(voiced), 2) if voiced else None,
                        "voiced": bool(voiced),
                    })
                confidences = recognition.alignment_confidences or [1.0] * len(points)
                measured_pitch = [
                    {"mora": char, "semitone": point.get("semitone"),
                     "voiced": bool(point.get("voiced")), "expected": "",
                     "alignment_confidence": confidence}
                    for char, point, confidence in zip(
                        target, points, confidences
                    )
                ]
            except Exception:  # noqa: BLE001
                logger.warning("Chinese F0 unavailable", exc_info=True)

        tones = [item["surface_tone"] for item in chinese_pitch_pattern(target)]
        pitch_matched, pitch_total = chinese_tone_matches(
            pitch_contours, tones, recognition.alignment_confidences
        )
        aspects = score_aspects(
            pronunciation_cer_score=score.score,
            n_morae=len(target_syllables),
            audio_duration=recognition.duration,
            speech_duration=speech_duration,
            windows=recognition.windows,
            moras=target_syllables,
            errors=score.errors,
            pitch_matched=pitch_matched,
            pitch_total=pitch_total,
            pause_count=pause_count,
            speech_ratio=speech_ratio,
            vad_method=vad_method,
            alignment_confidences=recognition.alignment_confidences,
        )
        method = aspects.method.replace("cer", "pinyin-asr", 1)
        if aspects.rhythm_measured:
            method = method.replace("pinyin-asr", "pinyin-asr+alignment", 1)
        aspects = replace(aspects, method=method)
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
