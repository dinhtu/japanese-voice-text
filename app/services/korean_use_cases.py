"""Korean read-aloud evaluation using Hangul syllable CER, VAD and F0."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from app.services.aspect_scoring import score_aspects, score_f0_dynamics
from app.services.korean_asr import KoreanASRService
from app.services.korean_text import normalize_korean
from app.services.scoring import score_pronunciation
from app.services.use_cases import EmptyTargetError, EvaluationResult
from src.asr.inference import load_audio

logger = logging.getLogger(__name__)


class EvaluateKoreanUseCase:
    def __init__(self, asr: KoreanASRService):
        self.asr = asr

    def execute(self, target_text: str, audio_path: str | Path) -> EvaluationResult:
        target = normalize_korean(target_text)
        if not target:
            raise EmptyTargetError("Target text contains no Hangul syllables.")
        recognition = self.asr.recognize(audio_path, align_to=target)
        recognized = normalize_korean(recognition.text)
        score = score_pronunciation(target, recognized)
        samples, sample_rate = load_audio(audio_path)

        speech_duration = recognition.duration
        pause_count = 0
        speech_ratio = None
        vad_method = None
        try:
            from app.services.vad_fluency import analyze_vad
            vad = analyze_vad(samples, sample_rate)
            speech_duration = vad.speech_duration or recognition.duration
            pause_count, speech_ratio, vad_method = vad.pause_count, vad.speech_ratio, vad.method
        except Exception:  # noqa: BLE001
            logger.warning("Korean VAD unavailable", exc_info=True)

        from app.services.pitch_extraction import extract_pitch_for_windows, extract_pitch_per_mora
        windows = recognition.windows
        points = (extract_pitch_for_windows(samples, sample_rate, windows)
                  if windows else extract_pitch_per_mora(samples, sample_rate, len(target)))
        measured_pitch = [
            {"mora": char, "semitone": point.get("semitone"),
             "voiced": bool(point.get("voiced")), "expected": ""}
            for char, point in zip(target, points)
        ]
        aspects = score_aspects(
            pronunciation_cer_score=score.score,
            n_morae=len(target), audio_duration=recognition.duration,
            speech_duration=speech_duration, windows=windows, moras=list(target),
            errors=score.errors, pause_count=pause_count,
            speech_ratio=speech_ratio, vad_method=vad_method,
        )
        intonation, measured = score_f0_dynamics(points)
        aspects = replace(
            aspects, intonation_score=intonation, intonation_measured=measured,
            method=aspects.method.replace("cer", "hangul-asr", 1) + ("+f0" if measured else ""),
        )
        return EvaluationResult(
            target_text=target_text, target_hiragana=target,
            recognized_text=recognition.text, recognized_hiragana=recognized,
            score=score, aspects=aspects,
            audio_duration=round(recognition.duration, 2),
            inference_ms=round(recognition.inference_time * 1000, 1),
            measured_pitch=measured_pitch,
        )
