"""Korean read-aloud evaluation using Hangul syllable CER, VAD and F0.

Two steps, so /evaluate-v2 can score another ASR's transcript: the audio
side (KoreanTake: VAD and, from a CTC model, the forced alignment to the
target) and score_korean(), which matches a transcript on top of it."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.services.aspect_scoring import score_aspects, score_f0_dynamics
from app.services.cjk_numbers import spell_numbers_ko
from app.services.korean_asr import KoreanASRService, KoreanRecognition
from app.services.korean_text import (
    korean_pitch_pattern,
    korean_pronunciation,
    normalize_korean,
)
from app.services.scoring import score_pronunciation
from app.services.use_cases import EmptyTargetError, EvaluationResult
from app.services.voice_presence import no_voice_result
from src.asr.inference import load_audio

logger = logging.getLogger(__name__)

# KoreanASRService.recognize's limit, kept when v2 does not run it.
MAX_SECONDS = 30


@dataclass
class KoreanTake:
    """The audio side of a take: everything but the transcript to score."""

    target_text: str
    target_norm: str
    samples: Any
    sample_rate: int
    duration: float
    vad: Any = None
    # The local model's run: its transcript (v1 scores it) and, for a CTC
    # model, the windows of its forced alignment to the target. None: not
    # run (v2 with Whisper, which has no alignment; no voice) or failed.
    recognition: KoreanRecognition | None = None
    # Set when no voice was heard (v2 only): the finished result.
    no_voice: EvaluationResult | None = None


def _target_norm(target_text: str) -> str:
    target_norm = normalize_korean(target_text)
    if not target_norm:
        raise EmptyTargetError("Target text contains no Hangul syllables.")
    return target_norm


def _vad(samples, sample_rate: int):
    try:
        from app.services.vad_fluency import analyze_vad

        return analyze_vad(samples, sample_rate)
    except Exception:  # noqa: BLE001
        logger.warning("Korean VAD unavailable", exc_info=True)
        return None


def score_korean(take: KoreanTake, heard_text: str, inference_ms: float) -> EvaluationResult:
    """`heard_text` (any ASR's transcript) against the target by Hangul
    syllable CER; rhythm and F0 from the take's windows (or an even split).
    Arabic numbers in it (Qwen3-ASR writes 세 개 as "3개") are spelled out in
    Hangul first, Sino-Korean or native as the target reads them; the CTC
    model of v1 writes Hangul only."""
    target_norm, target_text = take.target_norm, take.target_text
    spoken = spell_numbers_ko(heard_text, target_text)
    recognized_norm = normalize_korean(spoken)
    score = score_pronunciation(target_norm, recognized_norm)
    samples, sample_rate = take.samples, take.sample_rate

    vad = take.vad
    speech_duration = (vad.speech_duration if vad else None) or take.duration
    pause_count = vad.pause_count if vad else 0
    speech_ratio = vad.speech_ratio if vad else None
    vad_method = vad.method if vad else None

    windows = take.recognition.windows if take.recognition else None
    if not windows and len(target_norm) > 0 and speech_duration > 0:
        step = speech_duration / len(target_norm)
        windows = [(i * step, (i + 1) * step) for i in range(len(target_norm))]

    from app.services.pitch_extraction import extract_pitch_for_windows, extract_pitch_per_mora
    points = (extract_pitch_for_windows(samples, sample_rate, windows)
              if windows else extract_pitch_per_mora(samples, sample_rate, len(target_norm)))

    pitch_refs = korean_pitch_pattern(target_text)
    expected_tones = [p["pitch"] for p in pitch_refs] if len(pitch_refs) == len(target_norm) else ["L"] * len(target_norm)

    # Compute pitch match against AP intonation contour
    voiced_pts = [float(p["semitone"]) for p in points if p.get("voiced") and p.get("semitone") is not None]
    mean_pitch = sum(voiced_pts) / len(voiced_pts) if len(voiced_pts) >= 2 else None

    pitch_matched = 0
    pitch_total = 0
    for point, exp in zip(points, expected_tones):
        if point.get("voiced") and point.get("semitone") is not None and mean_pitch is not None:
            pitch_total += 1
            if (exp == "H" and point["semitone"] >= mean_pitch) or (exp == "L" and point["semitone"] <= mean_pitch):
                pitch_matched += 1

    measured_pitch = [
        {
            "mora": char,
            "semitone": point.get("semitone"),
            "voiced": bool(point.get("voiced")),
            "expected": exp,
        }
        for char, point, exp in zip(target_norm, points, expected_tones)
    ]
    aspects = score_aspects(
        pronunciation_cer_score=score.score,
        n_morae=len(target_norm), audio_duration=take.duration,
        speech_duration=speech_duration, windows=windows, moras=list(target_norm),
        errors=score.errors, pause_count=pause_count,
        pitch_matched=pitch_matched if pitch_total > 0 else None,
        pitch_total=pitch_total if pitch_total > 0 else None,
        speech_ratio=speech_ratio, vad_method=vad_method,
    )
    if aspects.intonation_score is None:
        intonation, measured = score_f0_dynamics(points)
        aspects = replace(
            aspects, intonation_score=intonation, intonation_measured=measured,
            method=aspects.method.replace("cer", "hangul-asr", 1) + ("+f0" if measured else ""),
        )

    target_reading = korean_pronunciation(target_text)
    # The transcript as heard, numbers spelled out: the AP pattern of the
    # recognized text (route) is built from it.
    recognized_display = spoken if heard_text else (recognized_norm or "—")

    return EvaluationResult(
        target_text=target_text,
        target_hiragana=target_norm,
        recognized_text=heard_text,
        recognized_hiragana=recognized_display,
        score=score,
        aspects=aspects,
        audio_duration=round(take.duration, 2),
        inference_ms=inference_ms,
        measured_pitch=measured_pitch,
        target_reading=target_reading,
        recognized_reading=korean_pronunciation(spoken) if heard_text else None,
    )


class EvaluateKoreanUseCase:
    def __init__(self, asr: KoreanASRService):
        self.asr = asr

    def execute(self, target_text: str, audio_path: str | Path) -> EvaluationResult:
        """v1: the local model's own transcript is scored (no voice check)."""
        target_norm = _target_norm(target_text)
        recognition = self.asr.recognize(audio_path, align_to=target_norm)
        samples, sample_rate = load_audio(audio_path)
        take = KoreanTake(target_text, target_norm, samples, sample_rate, recognition.duration,
                          vad=_vad(samples, sample_rate), recognition=recognition)
        return score_korean(take, recognition.text, round(recognition.inference_time * 1000, 1))

    def measure(self, target_text: str, audio_path: str | Path) -> KoreanTake:
        """v2's audio side: VAD, the voice check (a silent take is not sent
        to any ASR) and, with a CTC model, its forced alignment to the target
        for the syllable windows; a CTC failure only costs the timing.
        Whisper is not run: it has no alignment and its transcript is unused."""
        target_norm = _target_norm(target_text)
        samples, sample_rate = load_audio(audio_path)
        duration = len(samples) / sample_rate if sample_rate else 0.0
        if duration > MAX_SECONDS:
            raise ValueError("Korean recordings must be 30 seconds or shorter.")
        take = KoreanTake(target_text, target_norm, samples, sample_rate, duration,
                          vad=_vad(samples, sample_rate))
        if take.vad is not None:
            try:
                from app.services.voice_presence import check_voice

                presence = check_voice(samples, sample_rate, take.vad)
            except Exception:  # noqa: BLE001
                logger.warning("Korean voice check unavailable", exc_info=True)
                presence = None
            if presence is not None and not presence.voiced:
                logger.info("Korean evaluate: no voice (%s)", presence.verdict)
                take.no_voice = no_voice_result(
                    target_text, target_norm, list(target_norm), duration, take.vad, presence,
                    target_reading=korean_pronunciation(target_text),
                )
                return take
        if not getattr(self.asr, "is_whisper", False):
            try:
                take.recognition = self.asr.recognize(audio_path, align_to=target_norm)
            except ValueError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("Korean CTC alignment failed; even syllable split", exc_info=True)
        return take
