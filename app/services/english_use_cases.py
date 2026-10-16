"""English evaluate: wav2vec2 ASR + word-gated score + VAD + stress/intonation checks."""

from __future__ import annotations

import logging
from pathlib import Path

from dataclasses import replace

import numpy as np

from app.services.aspect_scoring import AspectScores, score_aspects
from app.services.english_alignment import (
    SyllableTiming,
    align_syllables,
    heard_words,
    proportional_syllables,
)
from app.services.english_asr import EnglishASRService, EnglishRecognition
from app.services.english_intonation import (
    IntonationResult,
    PitchTrack,
    SentenceAnalysis,
    analyze_sentence,
    evaluate_intonation,
)
from app.services.english_scoring import (
    SNAP_GATE,
    match_words,
    score_english,
    snap_transcript,
)
from app.services.english_text import (
    english_words,
    expand_for_speech,
    normalize_english,
    word_pitch_pattern,
)
from app.services.scoring import ScoreResult
from app.services.use_cases import EmptyTargetError, EvaluationResult
from app.services.vad_fluency import VadResult
from app.services.voice_presence import VoicePresence, check_voice

logger = logging.getLogger(__name__)


def _align(target: str, recognition: EnglishRecognition) -> tuple[SentenceAnalysis, SyllableTiming | None]:
    """Target words and their forced-aligned timing (None when impossible)."""
    analysis = analyze_sentence(target)
    timing = None
    if analysis.words:
        try:
            timing = align_syllables(
                analysis, recognition.log_probs, recognition.vocab, recognition.duration
            )
        except Exception:  # noqa: BLE001
            logger.warning("English forced alignment failed", exc_info=True)
    return analysis, timing


def _intonation(
    aligned: tuple[SentenceAnalysis, SyllableTiming | None],
    samples,
    sample_rate: int,
) -> IntonationResult | None:
    """Sentence stress / phrase / final-tone checks on forced-aligned syllables."""
    from app.services.pitch_extraction import frame_features

    analysis, timing = aligned
    features = frame_features(samples, sample_rate)
    if not analysis.words or features is None:
        return None
    track = PitchTrack(*features)
    timing = timing or proportional_syllables(analysis, track)
    return evaluate_intonation(
        analysis, track, timing.windows, timing.word_spans, timing.aligned, timing.missing
    )


def _legacy_owner(target: str, analysis) -> list[int | None]:
    """For each legacy word (english_words: apostrophes split, digits
    dropped) the analysis word it is part of, matched letter by letter."""
    spoken = [w.spoken for w in analysis.words]
    owners: list[int | None] = []
    a = c = 0
    for word in english_words(target):
        found = None
        while a < len(spoken):
            if spoken[a].startswith(word, c):
                found = a
                c += len(word)
                if c >= len(spoken[a]):
                    a, c = a + 1, 0
                break
            a, c = a + 1, 0  # a number word / title the legacy split lacks
        owners.append(found)
    return owners


def _word_pitch(target: str, intonation: IntonationResult | None) -> list[dict]:
    """Legacy `measured_pitch`: exactly one point per word_pitch_pattern()
    word (same split and order as before), measured at the stressed
    syllable of the word it belongs to; unvoiced when nothing was measured."""
    legacy = word_pitch_pattern(target)
    if intonation is None:
        return [
            {"mora": w.mora, "semitone": None, "voiced": False, "expected": w.pitch}
            for w in legacy
        ]
    by_key = {(m.word, m.syllable): m for m in intonation.syllables}
    words = intonation.analysis.words
    out = []
    for w, owner in zip(legacy, _legacy_owner(target, intonation.analysis)):
        m = by_key.get((owner, words[owner].pron.stress_index)) if owner is not None else None
        out.append({
            "mora": w.mora,
            "semitone": m.semitone if m else None,
            "voiced": bool(m and m.voiced),
            "expected": w.pitch,
        })
    return out


def _load(audio_path: str | Path) -> tuple[np.ndarray | None, int | None]:
    try:
        from src.asr.inference import load_audio

        return load_audio(audio_path)
    except Exception:  # noqa: BLE001
        logger.warning("English audio load failed", exc_info=True)
        return None, None


def _vad(samples: np.ndarray | None, sample_rate: int | None) -> VadResult | None:
    if samples is None or not sample_rate:
        return None
    try:
        from app.services.vad_fluency import analyze_vad

        return analyze_vad(samples, sample_rate)
    except Exception:  # noqa: BLE001
        logger.warning("English VAD failed", exc_info=True)
        return None


_NO_VOICE_MESSAGES = {
    "silent": "No voice was heard. Read the sentence aloud, close to the microphone.",
    "whisper": "Only whispering was heard. Read the sentence aloud in your normal voice.",
}


def _no_voice(
    target_text: str,
    target_norm: str,
    samples: np.ndarray,
    sample_rate: int,
    vad: VadResult,
    presence: VoicePresence,
) -> EvaluationResult:
    """Silence, noise or a whisper: nothing heard, nothing scored."""
    logger.info(
        "English evaluate: no voice (%s, speech %.2fs, voiced %.2fs)",
        presence.verdict, presence.speech_s, presence.voiced_s,
    )
    letters = len(target_norm.replace(" ", ""))
    score = ScoreResult(
        score=0, cer=1.0, distance=letters, target_length=letters,
        level="mismatch", message=_NO_VOICE_MESSAGES[presence.verdict],
    )
    aspects = AspectScores(
        overall_score=0.0, pronunciation_score=0.0, fluency_score=0.0,
        rhythm_score=None, intonation_score=None,
        rhythm_measured=False, intonation_measured=False,
        method=f"no-voice:{presence.verdict}", vad_method=vad.method,
        pause_count=0, speech_ratio=vad.speech_ratio,
    )
    return EvaluationResult(
        target_text=target_text,
        target_hiragana=target_norm,
        recognized_text="",
        recognized_hiragana="",
        score=score,
        aspects=aspects,
        audio_duration=round(len(samples) / sample_rate, 2),
        inference_ms=0.0,
        measured_pitch=_word_pitch(target_text, None),
    )


def _measure(
    target: str,
    score: ScoreResult,
    recognition: EnglishRecognition,
    aligned: tuple[SentenceAnalysis, SyllableTiming | None],
    samples: np.ndarray | None,
    sample_rate: int | None,
    vad: VadResult | None,
) -> tuple[AspectScores, list[dict], IntonationResult | None]:
    words = word_pitch_pattern(target)
    n = len(words) or 1
    measured_pitch = _word_pitch(target, None)
    intonation: IntonationResult | None = None
    speech_duration = None
    pause_count = 0
    speech_ratio = None
    vad_method = None
    windows = None

    if vad is not None:
        vad_method = vad.method
        pause_count = vad.pause_count
        speech_ratio = vad.speech_ratio
        if vad.speech_duration > 0:
            speech_duration = vad.speech_duration

    if samples is not None and sample_rate:
        # Rhythm keeps its old equal word windows; English rhythm is not
        # mora-even, so real word timing would only add noise there.
        duration = len(samples) / sample_rate
        step = duration / n if n else 0
        windows = [(i * step, (i + 1) * step) for i in range(n)]

        try:
            intonation = _intonation(aligned, samples, sample_rate)
        except Exception:  # noqa: BLE001
            logger.warning("English intonation failed", exc_info=True)
        measured_pitch = _word_pitch(target, intonation)

    # ~2 letters-as-mora per English word lands careful speech in the JP pace band.
    aspects = score_aspects(
        pronunciation_cer_score=score.score,
        n_morae=max(n * 2, 1),
        audio_duration=recognition.duration,
        speech_duration=speech_duration,
        windows=windows,
        moras=words,
        errors=score.errors,
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
    if intonation is not None and intonation.score is not None:
        aspects = replace(
            aspects,
            intonation_score=intonation.score,
            intonation_measured=True,
            method=f"{aspects.method}+stress",
        )

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
                # The stress / phrase / final-tone checks are what the chart
                # shows, so they own Ngữ điệu; GOPT prosodic only fills in
                # when they could not run.
                intonation_score = aspects.intonation_score
                intonation_measured = aspects.intonation_measured
                if not intonation_measured and gopt.intonation >= 8:
                    intonation_score = gopt.intonation
                    intonation_measured = True
                    used.append("pro")
                aspects = replace(
                    aspects,
                    pronunciation_score=pronunciation,
                    fluency_score=fluency,
                    intonation_score=intonation_score,
                    intonation_measured=intonation_measured,
                    method=(
                        "gopt+" + aspects.method.split("+", 1)[-1] if used else aspects.method
                    ),
                )
        except Exception:  # noqa: BLE001
            logger.warning("English GOPT failed; keeping wav2vec2+VAD+F0", exc_info=True)

    return aspects, measured_pitch, intonation


class EvaluateEnglishUseCase:
    def __init__(self, asr: EnglishASRService):
        self.asr = asr

    def execute(self, target_text: str, audio_path: str | Path) -> EvaluationResult:
        # Numbers / titles written out first: "Room 101" has three more words.
        target_norm = normalize_english(expand_for_speech(target_text))
        if not target_norm:
            raise EmptyTargetError("Target text contains no pronounceable English content.")

        samples, sample_rate = _load(audio_path)
        vad = _vad(samples, sample_rate)
        if vad is not None:
            presence = check_voice(samples, sample_rate, vad)
            if not presence.voiced:
                return _no_voice(target_text, target_norm, samples, sample_rate, vad, presence)

        recognition = self.asr.recognize(audio_path)
        aligned = _align(target_text, recognition)
        # Greedy CTC spells by ear ("KAT"); a target word the audio itself
        # carries, or one that sounds like it, is written as the target
        # spells it -- both in "AI nghe được" and before scoring.
        heard = heard_words(aligned[1]) | {
            i for i, m in enumerate(match_words(target_text, recognition.text))
            if m.similarity >= SNAP_GATE
        }
        heard_text = snap_transcript(target_text, recognition.text, heard)
        if heard_text != recognition.text:
            logger.info("English ASR %r read as %r", recognition.text, heard_text)
        recognized_norm = normalize_english(heard_text)
        score, _words = score_english(target_text, heard_text)
        intonation = None
        try:
            aspects, measured_pitch, intonation = _measure(
                target_text, score, recognition, aligned, samples, sample_rate, vad
            )
        except Exception:  # noqa: BLE001
            logger.warning("English aspect scoring failed", exc_info=True)
            aspects = score_aspects(
                pronunciation_cer_score=score.score,
                n_morae=max(len(english_words(target_text)) * 2, 1),
                audio_duration=recognition.duration,
                errors=score.errors,
            )
            measured_pitch = _word_pitch(target_text, None)

        return EvaluationResult(
            target_text=target_text,
            target_hiragana=target_norm,
            recognized_text=heard_text,
            recognized_hiragana=recognized_norm,
            score=score,
            aspects=aspects,
            audio_duration=round(recognition.duration, 2),
            inference_ms=round(recognition.inference_time * 1000, 1),
            measured_pitch=measured_pitch,
            intonation=intonation,
        )
