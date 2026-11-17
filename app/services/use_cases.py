"""Pronunciation evaluation use case.

    WAV -> voice check (VAD + voiced time; silence / whisper stop here)
        -> ASR -> recognized kana
                      |
    target text -> normalization <- normalization
                      |
                  per-mora score (ja_scoring)
                      |
                  aspect scores (fluency / rhythm / intonation / overall)
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from app.services.aspect_scoring import AspectScores, score_aspects
from app.services.asr_service import ASRService
from app.services.coaching import pitch_direction_match
from app.services.ja_scoring import score_japanese
from app.services.mora_diff import split_mora_spans
from app.services.mora_timing import resolve_mora_windows
from app.services.normalization import to_hiragana
from app.services.scoring import PronunciationError, ScoreResult

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
    target_reading: str | None = None
    recognized_reading: str | None = None
    # English only: app.services.english_intonation.IntonationResult
    intonation: object | None = None
    # English only: app.services.english_assessment dicts (additive reports)
    reading_assessment: dict | None = None
    gopt_assessment: dict | None = None
    # English only: english_assessment.AssessmentInputs (not serialized)
    assessment_inputs: object | None = None
    # Mandarin only: one tone check per target Hanzi (see chinese_use_cases)
    tones: list[dict] | None = None


def _speech_span(windows: list[tuple[float, float]]) -> float | None:
    if not windows:
        return None
    start, end = windows[0][0], windows[-1][1]
    span = end - start
    return span if span > 0 else None


@dataclass
class VoiceCheck:
    """The audio and VAD read once before the ASR, reused by the aspects."""

    samples: object = None
    sample_rate: int | None = None
    vad: object = None
    presence: object = None  # voice_presence.VoicePresence, None if unknown

    @property
    def no_voice(self) -> bool:
        return self.presence is not None and not self.presence.voiced


def check_japanese_voice(audio_path: str | Path) -> VoiceCheck:
    """Load the take, run the VAD and the voice check.

    Raises:
        RuntimeError: Audio could not be decoded (the ASR would fail too).
    A VAD / voice-check failure only means the take is scored ungated.
    """
    from src.asr.inference import load_audio

    check = VoiceCheck()
    check.samples, check.sample_rate = load_audio(audio_path)
    try:
        from app.services import vad_fluency, voice_presence

        check.vad = vad_fluency.analyze_vad(check.samples, check.sample_rate)
        check.presence = voice_presence.check_voice(check.samples, check.sample_rate, check.vad)
    except Exception:  # noqa: BLE001
        logger.warning("Japanese voice check unavailable", exc_info=True)
    return check


def _no_voice(target_text: str, target_hiragana: str, check: VoiceCheck) -> EvaluationResult:
    """Silence or a whisper: no ASR, score 0, every mora missing."""
    from app.services.voice_presence import no_voice_result

    presence = check.presence
    logger.info("Japanese evaluate: no voice (%s, speech %.2fs, voiced %.2fs)",
                presence.verdict, presence.speech_s, presence.voiced_s)
    spans = split_mora_spans(target_hiragana)
    duration = len(check.samples) / check.sample_rate if check.sample_rate else 0.0
    result = no_voice_result(
        target_text, target_hiragana, [text for text, _s, _e in spans],
        duration, check.vad, presence,
    )
    # Positions are character offsets in target_hiragana, as for a scored take.
    result.score = replace(
        result.score,
        errors=[PronunciationError("del", text, "", start) for text, start, _e in spans],
        mora_ok=[False] * len(spans),
    )
    return result


def _measure_aspects(
    target_text: str,
    target_hiragana: str,
    score: ScoreResult,
    recognition,
    audio_path: str | Path,
    check: VoiceCheck | None = None,
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

    samples = check.samples if check else None
    sample_rate = check.sample_rate if check else None
    if samples is None:
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

            vad = check.vad if check and check.vad is not None else analyze_vad(samples, sample_rate)
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

        check = check_japanese_voice(audio_path)
        if check.no_voice:
            return _no_voice(target_text, target_hiragana, check)

        recognition = self.asr_service.recognize(
            audio_path, with_timing=True, align_to=target_hiragana
        )
        recognized_hiragana = to_hiragana(recognition.kana)

        logger.info(
            "target=%s recognized=%s (%.0fms)",
            target_hiragana, recognized_hiragana, recognition.inference_time * 1000,
        )

        score = score_japanese(target_hiragana, recognized_hiragana)
        measured_pitch: list[dict] = []
        try:
            aspects, measured_pitch = _measure_aspects(
                target_text,
                target_hiragana,
                score,
                recognition,
                audio_path,
                check,
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
