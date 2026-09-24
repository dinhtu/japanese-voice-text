"""English evaluate: wav2vec2 ASR + letter CER + VAD + F0 (same 5 heads as JP)."""

from __future__ import annotations

import logging
from pathlib import Path

from dataclasses import replace

from app.services.aspect_scoring import AspectScores, score_aspects
from app.services.coaching import pitch_direction_match
from app.services.english_asr import EnglishASRService, EnglishRecognition
from app.services.english_text import english_words, normalize_english, word_pitch_pattern
from app.services.scoring import ScoreResult, score_pronunciation
from app.services.use_cases import EmptyTargetError, EvaluationResult

logger = logging.getLogger(__name__)


def _measure(
    target: str,
    score: ScoreResult,
    recognition: EnglishRecognition,
    audio_path: str | Path,
) -> tuple[AspectScores, list[dict]]:
    words = word_pitch_pattern(target)
    n = len(words) or 1
    measured_pitch: list[dict] = []
    pitch_matched = pitch_total = None
    speech_duration = None
    pause_count = 0
    speech_ratio = None
    vad_method = None
    windows = None

    samples = sample_rate = None
    try:
        from src.asr.inference import load_audio

        samples, sample_rate = load_audio(audio_path)
    except Exception:  # noqa: BLE001
        logger.warning("English audio reload failed", exc_info=True)

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
            logger.warning("English VAD failed", exc_info=True)

        try:
            from app.services.pitch_extraction import extract_pitch_per_mora

            points = extract_pitch_per_mora(samples, sample_rate, n)
            if words and len(points) == n:
                pitch_matched, pitch_total = pitch_direction_match(words, points)
                measured_pitch = [
                    {
                        "mora": w.mora,
                        "semitone": p.get("semitone"),
                        "voiced": bool(p.get("voiced")),
                        "expected": w.pitch,
                    }
                    for w, p in zip(words, points)
                ]
                duration = len(samples) / sample_rate
                step = duration / n if n else 0
                windows = [(i * step, (i + 1) * step) for i in range(n)]
        except Exception:  # noqa: BLE001
            logger.warning("English F0 failed", exc_info=True)

    # ~2 letters-as-mora per English word lands careful speech in the JP pace band.
    aspects = score_aspects(
        pronunciation_cer_score=score.score,
        n_morae=max(n * 2, 1),
        audio_duration=recognition.duration,
        speech_duration=speech_duration,
        windows=windows,
        moras=words,
        errors=score.errors,
        pitch_matched=pitch_matched,
        pitch_total=pitch_total,
        pause_count=pause_count,
        speech_ratio=speech_ratio,
        vad_method=vad_method,
    )
    method = (
        aspects.method.replace("cer", "wav2vec2", 1)
        if aspects.method.startswith("cer")
        else aspects.method
    )
    aspects = replace(aspects, method=method)

    if recognition.log_probs is not None and recognition.vocab:
        try:
            from app.services.english_gopt import get_english_gopt_service

            gopt = get_english_gopt_service().score(
                target, recognition.log_probs, recognition.vocab
            )
            if gopt is not None:
                # GOPT heads: accuracy, completeness, fluency, prosodic.
                # Completeness is NOT rhythm — keep local mora/word timing
                # for Nhịp. Ignore collapsed heads (~0 from OOD GOP features)
                # so they do not wipe CER / VAD / F0 scores.
                used = []
                pronunciation = aspects.pronunciation_score
                if gopt.pronunciation >= 8:
                    pronunciation = round(
                        0.5 * float(score.score) + 0.5 * gopt.pronunciation, 2
                    )
                    used.append("acc")
                fluency = aspects.fluency_score
                if gopt.fluency >= 8:
                    fluency = gopt.fluency
                    used.append("flu")
                elif gopt.completeness >= 8:
                    fluency = round(
                        0.7 * float(aspects.fluency_score)
                        + 0.3 * gopt.completeness,
                        2,
                    )
                    used.append("comp")
                intonation = aspects.intonation_score
                intonation_measured = aspects.intonation_measured
                if gopt.intonation >= 8:
                    intonation = gopt.intonation
                    intonation_measured = True
                    used.append("pro")
                aspects = replace(
                    aspects,
                    pronunciation_score=pronunciation,
                    fluency_score=fluency,
                    intonation_score=intonation,
                    intonation_measured=intonation_measured,
                    method="gopt+wav2vec2" if used else aspects.method,
                )
        except Exception:  # noqa: BLE001
            logger.warning("English GOPT failed; keeping wav2vec2+VAD+F0", exc_info=True)

    return aspects, measured_pitch


class EvaluateEnglishUseCase:
    def __init__(self, asr: EnglishASRService):
        self.asr = asr

    def execute(self, target_text: str, audio_path: str | Path) -> EvaluationResult:
        target_norm = normalize_english(target_text)
        if not target_norm:
            raise EmptyTargetError("Target text contains no pronounceable English content.")

        recognition = self.asr.recognize(audio_path)
        recognized_norm = normalize_english(recognition.text)
        score = score_pronunciation(target_norm.replace(" ", ""), recognized_norm.replace(" ", ""))
        try:
            aspects, measured_pitch = _measure(target_text, score, recognition, audio_path)
        except Exception:  # noqa: BLE001
            logger.warning("English aspect scoring failed", exc_info=True)
            aspects = score_aspects(
                pronunciation_cer_score=score.score,
                n_morae=max(len(english_words(target_text)) * 2, 1),
                audio_duration=recognition.duration,
                errors=score.errors,
            )
            measured_pitch = []

        return EvaluationResult(
            target_text=target_text,
            target_hiragana=target_norm,
            recognized_text=recognition.text,
            recognized_hiragana=recognized_norm,
            score=score,
            aspects=aspects,
            audio_duration=round(recognition.duration, 2),
            inference_ms=round(recognition.inference_time * 1000, 1),
            measured_pitch=measured_pitch,
        )
