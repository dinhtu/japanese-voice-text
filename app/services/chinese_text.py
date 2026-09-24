"""Mandarin text normalization and toneless syllable matching."""
from __future__ import annotations

from app.services.scoring import PronunciationError, ScoreResult, _feedback
from src.asr.metrics import edit_distance, edit_ops


def normalize_chinese(text: str) -> str:
    return "".join(c for c in text if "\u4e00" <= c <= "\u9fff")


def pinyin_syllables(text: str) -> list[str]:
    from pypinyin import Style, lazy_pinyin

    return lazy_pinyin(normalize_chinese(text), style=Style.NORMAL)


def score_syllables(target: list[str], heard: list[str]) -> ScoreResult:
    """Pinyin without lexical tones: homophonous Hanzi count as the same sound."""
    if not target:
        raise ValueError("Target contains no Mandarin syllables.")
    distance = edit_distance(target, heard)
    error_rate = distance / len(target)
    score = round(max(0.0, 1.0 - error_rate) * 100)
    level, _ = _feedback(score)
    message = (
        "ASR syllables match the target closely; tones are not evaluated."
        if score >= 75 else "Some recognized syllables differ; tones are not evaluated."
    )
    errors = [
        PronunciationError(
            type=op,
            target=target[i] if op != "ins" else "",
            recognized=heard[j] if op != "del" else "",
            position=i,
        )
        for op, i, j in edit_ops(target, heard)
    ]
    return ScoreResult(score, round(error_rate, 4), distance, len(target), level, message, errors)


def score_chinese_fluency(syllables: int, duration: float, pauses: int, accuracy: int) -> float:
    """Broad Mandarin reading band in syllables/s, penalizing hesitations."""
    if duration <= 0 or accuracy <= 0:
        return 0.0
    rate = syllables / duration
    if rate < 2.0:
        pace = max(20.0, 50.0 * rate)
    elif rate > 6.0:
        pace = max(20.0, 100.0 - 20.0 * (rate - 6.0))
    else:
        pace = 100.0
    return round(max(0.0, pace - max(0, pauses - 1) * 5) * (0.75 + 0.25 * accuracy / 100), 2)
