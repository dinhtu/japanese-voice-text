"""Mandarin text normalization, toneless syllable matching and tone checks."""
from __future__ import annotations

import re

from app.services.scoring import PronunciationError, ScoreResult, _feedback
from src.asr.metrics import edit_distance, edit_ops


def _is_hanzi(char: str) -> bool:
    return "一" <= char <= "鿿"


def normalize_chinese(text: str) -> str:
    return "".join(c for c in text if _is_hanzi(c))


def pinyin_syllables(text: str) -> list[str]:
    from pypinyin import Style, lazy_pinyin

    return lazy_pinyin(normalize_chinese(text), style=Style.NORMAL)


# Chao tone letters (1 = low, 5 = high) of each tone as drawn on the chart.
_TONE_CONTOURS = {
    1: [5, 5],
    2: [3, 5],
    3: [2, 1, 4],
    4: [5, 1],
    5: [3, 3],
}
# A third tone with more speech after it in the phrase only dips (21).
_HALF_THIRD = [2, 1, 1]
# 一 keeps its first tone in numbers: after these, or before a digit (十一, 一二三).
_NUMBER_BEFORE = set("零〇一二三四五六七八九十百千万亿两第")
_DIGITS = set("零〇一二三四五六七八九十两")


def _phrase_finals(text: str) -> list[bool]:
    """One flag per Hanzi of `text`: True when punctuation or the end follows it."""
    chars = [c for c in text if not c.isspace()]
    return [
        not (k + 1 < len(chars) and _is_hanzi(chars[k + 1]))
        for k, c in enumerate(chars) if _is_hanzi(c)
    ]


def _surface_tones(chars: str, tones: list[int], finals: list[bool]) -> list[int]:
    """Tone sandhi inside a phrase: 3+3 -> 2+3, 不 before a 4th tone -> 2,
    一 -> 2 before a 4th tone and -> 4 before 1/2/3, except in numbers
    (第一, 十一, a counted 一) and in V一V, where it is neutral. pypinyin's
    phrase dictionary already applies some of these (不是 bú, 一个 yí)."""
    surface = tones.copy()
    for k in range(len(tones) - 1):
        if finals[k]:
            continue
        after = tones[k + 1]
        if tones[k] == 3 and after == 3:
            surface[k] = 2
        elif chars[k] == "不" and tones[k] == 4 and after == 4:
            surface[k] = 2
        elif chars[k] == "一" and tones[k] == 1:
            before = chars[k - 1] if k else ""
            if before and before == chars[k + 1]:
                surface[k] = 5
            elif before not in _NUMBER_BEFORE and chars[k + 1] not in _DIGITS and after != 5:
                surface[k] = 2 if after == 4 else 4
    return surface


def chinese_pitch_pattern(text: str) -> list[dict]:
    """Pinyin, dictionary tone, spoken (sandhi) tone and Chao contour per Hanzi."""
    from pypinyin import Style, lazy_pinyin
    from pypinyin.contrib.tone_convert import to_tone

    target = normalize_chinese(text)
    if not target:
        return []
    numbered = lazy_pinyin(target, style=Style.TONE3, neutral_tone_with_five=True)
    tones = []
    syllables = []
    for value in numbered:
        match = re.search(r"([1-5])$", value)
        tones.append(int(match.group(1)) if match else 5)
        syllables.append(re.sub(r"[1-5]$", "", value))
    finals = _phrase_finals(text)
    surface = _surface_tones(target, tones, finals)
    return [
        {
            "hanzi": char,
            "mora": syllable,
            "pinyin": to_tone(f"{syllable}{spoken}"),
            "tone": tone,
            "surface_tone": spoken,
            "phrase_final": final,
            "contour": _HALF_THIRD if spoken == 3 and not final else _TONE_CONTOURS[spoken],
        }
        for char, syllable, tone, spoken, final in zip(target, syllables, tones, surface, finals)
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
        "ASR syllables match the target closely; tones are checked separately from F0."
        if score >= 75 else "Some recognized syllables differ; tones are checked separately from F0."
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


# A syllable's F0 shape (start / middle / end, semitones) fits its tone
# when the error below is at most TONE_TOLERANCE semitones.
TONE_TOLERANCE = 1.5


def _tone_error(start: float, middle: float, end: float, tone: int) -> float:
    if tone == 1:
        return max(start, middle, end) - min(start, middle, end)
    if tone == 2:
        return max(0.0, 1.5 - (end - start)) + max(0.0, start - middle)
    if tone == 3:
        return max(0.0, 0.8 - (start - middle)) + max(0.0, 0.8 - (end - middle))
    # fourth tone
    return max(0.0, 1.5 - (start - end)) + max(0.0, middle - start)


def _half_third_error(start: float, middle: float, end: float) -> float:
    """21: falls low and stays there."""
    return max(0.0, 0.8 - (start - min(middle, end))) + max(0.0, end - start)


def tone_check(values: list[float | None], tone: int, final: bool = True) -> tuple[bool | None, int | None]:
    """(fits, closest tone 1-4) for one syllable's start/middle/end F0.
    (None, None) when the shape was not measured, or for the neutral tone,
    whose pitch follows the syllable before it. Mid-phrase a third tone may
    be the half third (21) or the full dip-and-rise."""
    if tone == 5 or len(values) != 3 or any(value is None for value in values):
        return None, None
    start, middle, end = (float(value) for value in values)
    error = _tone_error(start, middle, end, tone)
    if tone == 3 and not final:
        error = min(error, _half_third_error(start, middle, end))
    heard = min((1, 2, 3, 4), key=lambda t: _tone_error(start, middle, end, t))
    return error <= TONE_TOLERANCE, heard

