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
# 一 keeps yī in numbers and where it ends a word: after these (十一, 第一,
# 星期一, 唯一, 统一, 万一) or before these (一二三, 一月, 一日, 一号).
_YI_KEEPS_AFTER = set("零〇一二三四五六七八九十百千万亿两第期唯统之单同")
_YI_KEEPS_BEFORE = set("零〇一二三四五六七八九十两月日号")
# 儿 after these is the -r of erhua (一点儿, 这儿), not a syllable of its own.
_ERHUA_AFTER = set("点会块玩哪这那边空事")


def _phrase_finals(text: str) -> list[bool]:
    """One flag per Hanzi of `text`: True when punctuation or the end follows it."""
    chars = [c for c in text if not c.isspace()]
    return [
        not (k + 1 < len(chars) and _is_hanzi(chars[k + 1]))
        for k, c in enumerate(chars) if _is_hanzi(c)
    ]


def _yi_tone(chars: str, k: int, tones: list[int], final: bool) -> int:
    before = chars[k - 1] if k else ""
    after = "" if final else chars[k + 1]
    if not after or before in _YI_KEEPS_AFTER or after in _YI_KEEPS_BEFORE:
        return 1
    if before == after and (k < 2 or chars[k - 2] != "一"):  # 看一看, not 一天一天
        return 5
    if tones[k + 1] == 5:
        return tones[k]
    return 2 if tones[k + 1] == 4 else 4


def _surface_tones(chars: str, tones: list[int], finals: list[bool]) -> list[int]:
    """Tone sandhi inside a phrase: 3+3 -> 2+3; 不 -> 2 before a 4th tone and
    neutral in V不V; 一 -> 2 before a 4th tone and -> 4 before 1/2/3, except
    in numbers and word-final 一 (yī) and in V一V (neutral); erhua 儿 is
    neutral. pypinyin's phrase dictionary already applies some of these
    (不是 bú, 一个 yí)."""
    surface = tones.copy()
    for k, char in enumerate(chars):
        before = chars[k - 1] if k else ""
        after = "" if finals[k] else chars[k + 1]
        if char == "一":
            surface[k] = _yi_tone(chars, k, tones, finals[k])
        elif char == "不" and before and before == after:
            surface[k] = 5
        elif char == "不" and after and tones[k + 1] == 4:
            surface[k] = 2
        elif char == "儿" and before in _ERHUA_AFTER:
            surface[k] = 5
        elif after and tones[k] == 3 and tones[k + 1] == 3:
            surface[k] = 2
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


# A syllable's F0 shape (start / middle / end, semitones from the clip mean)
# fits its tone when the error below is under TONE_TOLERANCE semitones and
# no other tone fits better by more than TONE_MARGIN.
TONE_TOLERANCE = 1.5
TONE_MARGIN = 0.5


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
    """21: starts mid or low, falls and stays low (a high fall is tone 4)."""
    low = min(middle, end)
    return (max(0.0, 0.8 - (start - low)) + max(0.0, end - start)
            + max(0.0, low + 0.5) + max(0.0, start - 1.0))


def tone_check(values: list[float | None], tone: int, final: bool = True) -> tuple[bool | None, int | None]:
    """(fits, closest tone 1-4) for one syllable's start/middle/end F0.
    (None, None) when the shape was not measured, or for the neutral tone,
    whose pitch follows the syllable before it. Mid-phrase a third tone may
    be the half third (21) or the full dip-and-rise."""
    if tone == 5 or len(values) != 3 or any(value is None for value in values):
        return None, None
    start, middle, end = (float(value) for value in values)
    errors = {t: _tone_error(start, middle, end, t) for t in (1, 2, 3, 4)}
    if tone == 3 and not final:
        errors[3] = min(errors[3], _half_third_error(start, middle, end))
    closest = min(errors, key=lambda t: (errors[t], t != tone))
    ok = errors[tone] < TONE_TOLERANCE and errors[tone] <= errors[closest] + TONE_MARGIN
    return ok, tone if ok else closest
