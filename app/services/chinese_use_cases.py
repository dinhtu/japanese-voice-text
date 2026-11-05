"""Mandarin read-aloud matching by toneless pinyin syllable, plus a tone
check per syllable from its F0 shape.

Two steps, so /evaluate-v2 can score another ASR's transcript:
measure() is the audio side (voice check, the CTC model's transcript and
forced alignment to the target, F0 per syllable), score_chinese() matches
a transcript against the target on top of it."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from app.services.aspect_scoring import score_aspects
from app.services.cjk_numbers import spell_numbers_zh
from app.services.chinese_asr import ChineseASRService, ChineseRecognition
from app.services.chinese_text import (
    chinese_pitch_pattern, normalize_chinese, pinyin_syllables, score_syllables, tone_check,
)
from app.services.use_cases import EmptyTargetError, EvaluationResult
from app.services.voice_presence import no_voice_result

logger = logging.getLogger(__name__)


def tone_checks(
    pattern: list[dict],
    contours: list[list[float | None]],
    misread: set[int],
) -> list[dict]:
    """One check per target Hanzi: the /pitch-accent item plus the measured
    start/middle/end F0, the closest tone and `ok`. A syllable ASR heard as
    another one (`misread`) is not judged: its alignment window is a guess.
    `status` says why `ok` is null: misread | neutral | unmeasured."""
    checks = []
    for k, item in enumerate(pattern):
        values = contours[k] if k < len(contours) else [None, None, None]
        ok = heard = None
        if k in misread:
            status = "misread"
        elif item["surface_tone"] == 5:
            status = "neutral"
        else:
            ok, heard = tone_check(values, item["surface_tone"], item["phrase_final"])
            status = "unmeasured" if ok is None else "ok" if ok else "bad"
        checks.append({
            **item,
            "heard_contour": [None if v is None else round(float(v), 2) for v in values],
            "heard_tone": heard,
            "ok": ok,
            "status": status,
        })
    return checks


def _no_voice(target_text: str, target: str, syllables: list[str], pattern: list[dict],
              duration: float, vad, presence) -> EvaluationResult:
    """Silence or a whisper: nothing heard, nothing scored (as English)."""
    logger.info("Chinese evaluate: no voice (%s, speech %.2fs, voiced %.2fs)",
                presence.verdict, presence.speech_s, presence.voiced_s)
    return no_voice_result(
        target_text, target, syllables, duration, vad, presence,
        target_reading=" ".join(syllables), recognized_reading="",
        tones=tone_checks(pattern, [], set()),
    )


@dataclass
class ChineseTake:
    """The audio side of a take: everything but the transcript to score."""

    target_text: str
    target: str  # Hanzi only
    syllables: list[str]
    pattern: list[dict]
    samples: Any
    sample_rate: int
    duration: float
    vad: Any = None
    # The CTC model's run: its transcript (v1 scores it) and the windows of
    # its forced alignment to the target. None: not run (no voice) or failed.
    recognition: ChineseRecognition | None = None
    contours: list[list[float | None]] = field(default_factory=list)
    measured_pitch: list[dict] = field(default_factory=list)
    # Set when no voice was heard: the finished result, nothing to score.
    no_voice: EvaluationResult | None = None


def _f0(take: ChineseTake) -> None:
    """Start/middle/end F0 per aligned syllable, and its median for the chart."""
    if not (take.recognition and take.recognition.windows):
        return
    try:
        from statistics import median

        from app.services.pitch_extraction import extract_mandarin_tone_contours

        take.contours = extract_mandarin_tone_contours(take.samples, take.sample_rate, take.recognition.windows)
        for char, contour in zip(take.target, take.contours):
            voiced = [float(value) for value in contour if value is not None]
            take.measured_pitch.append({
                "mora": char, "semitone": round(median(voiced), 2) if voiced else None,
                "voiced": bool(voiced), "expected": "",
            })
    except Exception:  # noqa: BLE001
        logger.warning("Chinese F0 unavailable", exc_info=True)


def score_chinese(take: ChineseTake, heard_text: str, inference_ms: float) -> EvaluationResult:
    """`heard_text` (any ASR's transcript) against the target, by toneless
    pinyin syllable; tone checks and timing from the take's alignment.
    Arabic numbers in it (Qwen3-ASR writes 二十 as "20") are spelled out in
    Hanzi first; the CTC model of v1 writes Hanzi only."""
    recognized = normalize_chinese(spell_numbers_zh(heard_text, take.target_text))
    heard_syllables = pinyin_syllables(recognized)
    score = score_syllables(take.syllables, heard_syllables)
    windows = take.recognition.windows if take.recognition else None
    vad = take.vad

    misread = {e.position for e in score.errors if e.type in ("sub", "del")}
    tones = tone_checks(take.pattern, take.contours, misread)
    judged = [t["ok"] for t in tones if t["ok"] is not None]
    aspects = score_aspects(
        pronunciation_cer_score=score.score,
        n_morae=len(take.syllables),
        audio_duration=take.duration,
        speech_duration=(vad.speech_duration if vad else None) or take.duration,
        windows=windows,
        moras=take.syllables,
        errors=score.errors,
        pitch_matched=sum(judged) if judged else None,
        pitch_total=len(judged) if judged else None,
        pause_count=vad.pause_count if vad else 0,
        speech_ratio=vad.speech_ratio if vad else None,
        vad_method=vad.method if vad else None,
    )
    aspects = replace(aspects, method=aspects.method.replace("cer", "pinyin-asr", 1))
    return EvaluationResult(
        target_text=take.target_text, target_hiragana=take.target,
        recognized_text=heard_text, recognized_hiragana=recognized,
        score=score, aspects=aspects,
        audio_duration=round(take.duration, 2),
        inference_ms=inference_ms,
        measured_pitch=take.measured_pitch,
        target_reading=" ".join(take.syllables),
        recognized_reading=" ".join(heard_syllables),
        tones=tones,
    )


class EvaluateChineseUseCase:
    def __init__(self, asr: ChineseASRService):
        self.asr = asr

    def execute(self, target_text: str, audio_path: str | Path) -> EvaluationResult:
        """v1: the CTC model's own transcript is scored."""
        take = self.measure(target_text, audio_path)
        if take.no_voice is not None:
            return take.no_voice
        recognition = take.recognition
        return score_chinese(take, recognition.text, round(recognition.inference_time * 1000, 1))

    def measure(self, target_text: str, audio_path: str | Path,
                alignment_optional: bool = False) -> ChineseTake:
        """The audio side. With `alignment_optional` (v2, whose transcript
        comes from elsewhere) a CTC failure only costs the timing; a
        recording over the length limit is still refused (ValueError)."""
        from src.asr.inference import load_audio

        target = normalize_chinese(target_text)
        if not target:
            raise EmptyTargetError("Target text contains no Chinese characters.")
        syllables = pinyin_syllables(target)
        pattern = chinese_pitch_pattern(target_text)
        samples, sample_rate = load_audio(audio_path)
        duration = len(samples) / sample_rate if sample_rate else 0.0
        take = ChineseTake(target_text, target, syllables, pattern, samples, sample_rate, duration)

        try:
            from app.services.vad_fluency import analyze_vad
            from app.services.voice_presence import check_voice

            take.vad = analyze_vad(samples, sample_rate)
            presence = check_voice(samples, sample_rate, take.vad)
            if not presence.voiced:
                take.no_voice = _no_voice(target_text, target, syllables, pattern, duration, take.vad, presence)
                return take
        except Exception:  # noqa: BLE001
            logger.warning("Chinese VAD unavailable", exc_info=True)

        try:
            take.recognition = self.asr.recognize(audio_path, align_to=target)
        except ValueError:
            raise
        except Exception:
            if not alignment_optional:
                raise
            logger.warning("Chinese CTC alignment failed; no syllable timing", exc_info=True)
        if take.recognition is not None:
            take.duration = take.recognition.duration
        _f0(take)
        return take
