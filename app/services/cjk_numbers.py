"""Spell the Arabic numbers in a Chinese / Korean ASR transcript out in the
target's script, so a correctly read number is not scored as deleted.

Qwen3-ASR (/evaluate-v2) applies inverse text normalisation: 我今年二十岁
read correctly can come back as "我今年20岁", 사과 세 개 as "사과 3개". The
scorers keep only Hanzi / Hangul, so the digits would vanish and the target's
number syllables would count as deleted (v1's CTC model writes characters
only and never hits this).

A digit string has several readings (二/两, 二十/二零, Sino 삼 / native 세 /
셋...). Each number gets a few candidates and the one closest to the target
is kept, left to right; the transcript only says which digits were heard, so
the reading the learner meant gets the benefit of the doubt. The target's
own digits are not scored (the normalisers drop them), so a target that has
digits is left alone: spelling the transcript's would add syllables.
"""
from __future__ import annotations

import re
from typing import Callable

# Fullwidth digits and the signs inside numbers, as ASCII (not the
# fullwidth comma: in Chinese it separates clauses, not thousands).
_FULLWIDTH = str.maketrans("０１２３４５６７８９．：％", "0123456789.:%")
_NUMBER = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2})(?!\d)"
    r"|(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?(?P<pct>%)?"
)


def has_digits(text: str) -> bool:
    return any(c.isdigit() for c in (text or "").translate(_FULLWIDTH) if c.isascii())


def _spell(
    text: str,
    target_text: str,
    candidates: Callable[[re.Match], list[str]],
    distance: Callable[[str], int],
) -> str:
    text = (text or "").translate(_FULLWIDTH)
    matches = list(_NUMBER.finditer(text))
    if not matches or has_digits(target_text):
        return text
    options = [candidates(m) or [m.group(0)] for m in matches]
    chosen = [opts[0] for opts in options]

    def build() -> str:
        out, last = [], 0
        for match, piece in zip(matches, chosen):
            out += [text[last:match.start()], piece]
            last = match.end()
        return "".join(out) + text[last:]

    for i, opts in enumerate(options):
        best, best_d = chosen[i], None
        for opt in opts:
            chosen[i] = opt
            d = distance(build())
            if best_d is None or d < best_d:
                best, best_d = opt, d
        chosen[i] = best
    return build()


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


def _integer(match: re.Match) -> tuple[str, int]:
    digits = match.group("int").replace(",", "")
    return digits, int(digits)


# --- Mandarin ---------------------------------------------------------------

_ZH_DIGITS = "零一二三四五六七八九"
_ZH_GROUPS = ["", "万", "亿", "万亿"]


def _zh_section(n: int) -> str:
    """0 < n < 10000: 一千零五, 一百一十."""
    out, zero = "", False
    for unit, d in zip(("千", "百", "十", ""), (n // 1000, n // 100 % 10, n // 10 % 10, n % 10)):
        if d == 0:
            zero = bool(out)
        else:
            out += ("零" if zero else "") + _ZH_DIGITS[d] + unit
            zero = False
    return out


def zh_cardinal(n: int) -> str:
    """二十, 一百一十, 一万零五, 十五."""
    if n == 0:
        return "零"
    groups = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    if len(groups) > len(_ZH_GROUPS):
        return ""
    out, zero = "", False
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            zero = bool(out)
            continue
        if out and (zero or g < 1000):
            out += "零"
        out += _zh_section(g) + _ZH_GROUPS[i]
        zero = False
    return out[1:] if out.startswith("一十") else out


def zh_digits(digits: str) -> str:
    return "".join(_ZH_DIGITS[int(d)] for d in digits)


def _zh_liang(s: str) -> str:
    """两 for a leading 2 counted with a measure word or 百/千/万/亿."""
    return "两" if s == "二" else re.sub(r"^二(?=[百千万亿])", "两", s)


def _zh_candidates(match: re.Match) -> list[str]:
    if match.group("h"):
        hour, minute = int(match.group("h")), int(match.group("m"))
        hours = _dedupe([zh_cardinal(hour), _zh_liang(zh_cardinal(hour))])
        mins = ("零" if minute < 10 else "") + zh_cardinal(minute) if minute else ""
        tails = _dedupe([mins, mins + "分" if mins else "", "半" if minute == 30 else "", "整" if not minute else ""])
        return [h + "点" + t for h in hours for t in tails or [""]]
    digits, n = _integer(match)
    whole = _dedupe([zh_cardinal(n), _zh_liang(zh_cardinal(n))])
    if len(digits) >= 2:
        whole.append(zh_digits(digits))  # years (二零二四年), codes
    if match.group("frac"):
        whole = [w + "点" + zh_digits(match.group("frac")) for w in whole]
    if match.group("pct"):
        whole = ["百分之" + w for w in whole]
    return _dedupe(whole)


def spell_numbers_zh(text: str, target_text: str) -> str:
    """`text` with each Arabic number written in Hanzi, read the way closest
    to `target_text` (by toneless pinyin syllable)."""
    if not has_digits(text):
        return text or ""
    from app.services.chinese_text import pinyin_syllables
    from src.asr.metrics import edit_distance

    target = pinyin_syllables(target_text)
    return _spell(text, target_text, _zh_candidates,
                  lambda s: edit_distance(target, pinyin_syllables(s)))


# --- Korean -----------------------------------------------------------------

_KO_SINO = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
_KO_GROUPS = ["", "만", "억", "조"]
_KO_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
_KO_ONES = ["", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉"]
# Before a counter (세 개, 두 시, 스무 살).
_KO_ONES_ATTR = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]


def _ko_section(n: int) -> str:
    out = ""
    for unit, d in zip("천백십", (n // 1000, n // 100 % 10, n // 10 % 10)):
        if d:
            out += ("" if d == 1 else _KO_SINO[d]) + unit
    return out + _KO_SINO[n % 10]


def ko_sino(n: int) -> str:
    """Sino-Korean: 삼, 이십, 백일, 만 오천."""
    if n == 0:
        return "영"
    groups = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    if len(groups) > len(_KO_GROUPS):
        return ""
    out = ""
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g:
            out += ("" if g == 1 and i == 1 else _ko_section(g)) + _KO_GROUPS[i]
    return out


def ko_native(n: int, attributive: bool) -> str:
    """Native Korean below 100 (셋 / 세, 스물 / 스무); hundreds and up stay
    Sino (백한 살). '' when n is 0."""
    if n <= 0:
        return ""
    head, n = ko_sino(n - n % 100) if n >= 100 else "", n % 100
    if not n:
        return head
    tens, ones = _KO_TENS[n // 10], (_KO_ONES_ATTR if attributive else _KO_ONES)[n % 10]
    if attributive and n == 20:
        tens = "스무"
    return head + tens + ones


def ko_digits(digits: str, zero: str = "공") -> str:
    return "".join(zero if d == "0" else _KO_SINO[int(d)] for d in digits)


def _ko_candidates(match: re.Match) -> list[str]:
    if match.group("h"):
        hour, minute = int(match.group("h")), int(match.group("m"))
        hours = _dedupe([ko_native(hour, True), ko_sino(hour)])
        tails = _dedupe([ko_sino(minute) + "분" if minute else "", "반" if minute == 30 else ""])
        return [h + "시" + t for h in hours for t in tails or [""]]
    digits, n = _integer(match)
    if match.group("frac"):
        whole = [ko_sino(n) + "점" + ko_digits(match.group("frac"), "영")]
    else:
        whole = [ko_sino(n), ko_native(n, True), ko_native(n, False)]
        if len(digits) >= 2:
            whole += [ko_digits(digits), ko_digits(digits, "영")]  # phone numbers, codes
    if match.group("pct"):
        whole = [w + tail for w in whole for tail in ("퍼센트", "프로")]
    return _dedupe(whole)


def spell_numbers_ko(text: str, target_text: str) -> str:
    """`text` with each Arabic number written in Hangul (Sino-Korean or
    native, whichever is closer to `target_text`)."""
    if not has_digits(text):
        return text or ""
    from app.services.korean_text import normalize_korean
    from src.asr.metrics import edit_distance

    target = list(normalize_korean(target_text))
    return _spell(text, target_text, _ko_candidates,
                  lambda s: edit_distance(target, list(normalize_korean(s))))
