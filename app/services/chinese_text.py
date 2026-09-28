"""Mandarin text normalization and toneless syllable matching."""
from __future__ import annotations

from app.services.scoring import PronunciationError, ScoreResult, _feedback
from src.asr.metrics import edit_distance, edit_ops


def normalize_chinese(text: str) -> str:
    return "".join(c for c in text if "\u4e00" <= c <= "\u9fff")


def pinyin_syllables(text: str) -> list[str]:
    from pypinyin import Style, lazy_pinyin

    return lazy_pinyin(normalize_chinese(text), style=Style.NORMAL)


_TONE_CONTOURS = {
    1: [5, 5],
    2: [3, 5],
    3: [2, 1, 4],
    4: [5, 1],
    5: [3, 3],
}


def chinese_pitch_pattern(text: str) -> list[dict]:
    """Return pinyin and the standard Chao contour for each Hanzi syllable."""
    import re
    from pypinyin import Style, lazy_pinyin

    target = normalize_chinese(text)
    if not target:
        return []
    numbered = lazy_pinyin(target, style=Style.TONE3, neutral_tone_with_five=True)
    tones = []
    syllables = []
    for value in numbered:
        match = re.search(r"([1-5])$", value)
        tone = int(match.group(1)) if match else 5
        tones.append(tone)
        syllables.append(re.sub(r"[1-5]$", "", value))

    # Basic third-tone sandhi: in a run of third tones, all but the last rise.
    surface = tones.copy()
    for index in range(len(surface) - 1):
        if tones[index] == tones[index + 1] == 3:
            surface[index] = 2

    return [
        {
            "mora": syllable,
            "tone": tone,
            "surface_tone": spoken,
            "contour": _TONE_CONTOURS[spoken],
        }
        for syllable, tone, spoken in zip(syllables, tones, surface)
    ]


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
