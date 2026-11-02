"""Mandarin read-aloud matching by toneless pinyin syllable, plus a tone
check per syllable from its F0 shape."""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from app.services.aspect_scoring import AspectScores, score_aspects
from app.services.chinese_asr import ChineseASRService
from app.services.chinese_text import (
    chinese_pitch_pattern, normalize_chinese, pinyin_syllables, score_syllables, tone_check,
)
from app.services.scoring import PronunciationError, ScoreResult
from app.services.use_cases import EmptyTargetError, EvaluationResult
from app.services.voice_presence import NO_VOICE_MESSAGES

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
    errors = [PronunciationError("del", s, "", k) for k, s in enumerate(syllables)]
    score = ScoreResult(0, 1.0, len(syllables), len(syllables), "mismatch",
                        NO_VOICE_MESSAGES[presence.verdict], errors)
    aspects = AspectScores(
        overall_score=0.0, pronunciation_score=0.0, fluency_score=0.0,
        rhythm_score=None, intonation_score=None,
        rhythm_measured=False, intonation_measured=False,
        method=f"no-voice:{presence.verdict}", vad_method=vad.method,
        pause_count=0, speech_ratio=vad.speech_ratio,
    )
    return EvaluationResult(
        target_text=target_text, target_hiragana=target,
        recognized_text="", recognized_hiragana="",
        score=score, aspects=aspects,
        audio_duration=round(duration, 2), inference_ms=0.0,
        target_reading=" ".join(syllables), recognized_reading="",
        tones=tone_checks(pattern, [], set()),
    )


class EvaluateChineseUseCase:
    def __init__(self, asr: ChineseASRService):
        self.asr = asr

    def execute(self, target_text: str, audio_path: str | Path) -> EvaluationResult:
        from src.asr.inference import load_audio

        target = normalize_chinese(target_text)
        if not target:
            raise EmptyTargetError("Target text contains no Chinese characters.")
        target_syllables = pinyin_syllables(target)
        pattern = chinese_pitch_pattern(target_text)
        samples, sample_rate = load_audio(audio_path)

        vad = None
        try:
            from app.services.vad_fluency import analyze_vad
            from app.services.voice_presence import check_voice

            vad = analyze_vad(samples, sample_rate)
            presence = check_voice(samples, sample_rate, vad)
            if not presence.voiced:
                duration = len(samples) / sample_rate if sample_rate else 0.0
                return _no_voice(target_text, target, target_syllables, pattern, duration, vad, presence)
        except Exception:  # noqa: BLE001
            logger.warning("Chinese VAD unavailable", exc_info=True)

        recognition = self.asr.recognize(audio_path, align_to=target)
        recognized = normalize_chinese(recognition.text)
        heard_syllables = pinyin_syllables(recognized)
        score = score_syllables(target_syllables, heard_syllables)

        measured_pitch: list[dict] = []
        contours: list[list[float | None]] = []
        if recognition.windows:
            try:
                from statistics import median

                from app.services.pitch_extraction import extract_mandarin_tone_contours

                contours = extract_mandarin_tone_contours(samples, sample_rate, recognition.windows)
                for char, contour in zip(target, contours):
                    voiced = [float(value) for value in contour if value is not None]
                    measured_pitch.append({
                        "mora": char, "semitone": round(median(voiced), 2) if voiced else None,
                        "voiced": bool(voiced), "expected": "",
                    })
            except Exception:  # noqa: BLE001
                logger.warning("Chinese F0 unavailable", exc_info=True)

        misread = {e.position for e in score.errors if e.type in ("sub", "del")}
        tones = tone_checks(pattern, contours, misread)
        judged = [t["ok"] for t in tones if t["ok"] is not None]
        aspects = score_aspects(
            pronunciation_cer_score=score.score,
            n_morae=len(target_syllables),
            audio_duration=recognition.duration,
            speech_duration=(vad.speech_duration if vad else None) or recognition.duration,
            windows=recognition.windows,
            moras=target_syllables,
            errors=score.errors,
            pitch_matched=sum(judged) if judged else None,
            pitch_total=len(judged) if judged else None,
            pause_count=vad.pause_count if vad else 0,
            speech_ratio=vad.speech_ratio if vad else None,
            vad_method=vad.method if vad else None,
        )
        aspects = replace(aspects, method=aspects.method.replace("cer", "pinyin-asr", 1))
        return EvaluationResult(
            target_text=target_text, target_hiragana=target,
            recognized_text=recognition.text, recognized_hiragana=recognized,
            score=score, aspects=aspects,
            audio_duration=round(recognition.duration, 2),
            inference_ms=round(recognition.inference_time * 1000, 1),
            measured_pitch=measured_pitch,
            target_reading=" ".join(target_syllables),
            recognized_reading=" ".join(heard_syllables),
            tones=tones,
        )
