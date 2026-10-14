"""Normalize English target / ASR text for scoring and the pitch chart."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_KEEP = re.compile(r"[^a-z\s]")
_SPACE = re.compile(r"\s+")

# Weak / function words sit L; content words sit H — a coarse English
# stress sketch so the same H/L chart as Japanese still has a "mẫu".
_FUNCTION = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "or",
    "but", "is", "are", "was", "were", "be", "am", "do", "does", "did",
    "have", "has", "had", "i", "you", "he", "she", "we", "they", "it",
    "my", "your", "his", "her", "our", "their", "with", "as", "from",
    "that", "this", "so", "too", "very",
}


@dataclass(frozen=True)
class WordPitch:
    mora: str  # word — same field name as Japanese so the chart/API match
    pitch: str  # "H" | "L"
    phrase: int = 0


def normalize_english(text: str) -> str:
    """Lowercase letters and single spaces. Empty if nothing pronounceable."""
    if not text or not text.strip():
        return ""
    folded = _KEEP.sub(" ", text.lower())
    return _SPACE.sub(" ", folded).strip()


def english_words(text: str) -> list[str]:
    return [w for w in normalize_english(text).split(" ") if w]


def word_pitch_pattern(text: str) -> list[WordPitch]:
    words = english_words(text)
    return [
        WordPitch(mora=w, pitch="L" if w in _FUNCTION else "H")
        for w in words
    ]


# ------------------------------------------------- spoken-word tokenization
#
# The score and the intonation chart both need the words as they are SAID:
# "3.50" is "three point five zero", "Mr." is "Mister", "U.S." is one word.
# speech_tokens() is the one tokenizer both use, so their word lists always
# line up one-to-one.

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORDINAL = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}

# Titles are always read out in full. "St." depends on what follows.
_TITLES = {
    "mr": "Mister", "mrs": "Missus", "ms": "Miz", "dr": "Doctor",
    "prof": "Professor", "jr": "Junior", "sr": "Senior", "vs": "versus",
    "etc": "et cetera", "approx": "approximately",
}
_LATIN = {"e.g.": "for example", "i.e.": "that is"}


def number_words(n: int) -> str:
    """0 <= n < 10**12 in words ("one hundred twenty three")."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {number_words(rest)}" if rest else "")
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= size:
            head, rest = divmod(n, size)
            return f"{number_words(head)} {name}" + (f" {number_words(rest)}" if rest else "")
    return str(n)


def _ordinal_words(n: int) -> str:
    words = number_words(n).split()
    last = words[-1]
    if last in _ORDINAL:
        words[-1] = _ORDINAL[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _digits(text: str) -> str:
    return " ".join(_ONES[int(d)] for d in text)


def _expand_number(m: re.Match) -> str:
    money, whole, frac_sep, frac, ordinal, percent = m.group(
        "money", "whole", "sep", "frac", "ord", "pct"
    )
    integer = int(whole.replace(",", ""))
    if ordinal:
        return _ordinal_words(integer)
    if frac_sep == ":":  # clock time
        minutes = int(frac)
        if minutes == 0:
            spoken = f"{number_words(integer)} o'clock"
        elif minutes < 10:
            spoken = f"{number_words(integer)} oh {number_words(minutes)}"
        else:
            spoken = f"{number_words(integer)} {number_words(minutes)}"
    elif money and frac and len(frac) == 2:
        spoken = f"{number_words(integer)} dollars"
        if int(frac):
            spoken += f" {number_words(int(frac))}"
        return spoken
    elif frac:
        spoken = f"{number_words(integer)} point {_digits(frac)}"
    else:
        spoken = number_words(integer)
    if money:
        spoken += " dollar" if integer == 1 and not frac else " dollars"
    if percent:
        spoken += " percent"
    return spoken


_NUMBER = re.compile(
    r"(?P<money>\$)?(?P<whole>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:(?P<sep>[.:])(?P<frac>\d+))?"
    r"(?:(?P<ord>st|nd|rd|th)\b)?(?P<pct>%)?"
)
_TITLE = re.compile(r"\b([A-Za-z]+)\.(?=\s|$)")
_SAINT = re.compile(r"\bSt\.(?=\s+[A-Z])")
_BARE_TITLE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof)\b(?=\s+[A-Z])")


def expand_for_speech(text: str) -> str:
    """Numbers, titles and Latin abbreviations written out as spoken."""
    for short, long in _LATIN.items():
        text = re.sub(re.escape(short), long, text, flags=re.IGNORECASE)
    text = _SAINT.sub("Saint", text)
    text = re.sub(r"\bSt\.", "Street", text)

    def title(m: re.Match) -> str:
        long = _TITLES.get(m.group(1).lower())
        if not long:
            return m.group(0)
        # "..., etc." at the very end still closes the sentence.
        return long + "." if m.end() == len(m.string) else long

    text = _TITLE.sub(title, text)
    # British style, no period: "Dr Brown", "Mr Smith".
    text = _BARE_TITLE.sub(lambda m: _TITLES[m.group(1).lower()], text)
    return _NUMBER.sub(_expand_number, text)


@dataclass(frozen=True)
class SpeechToken:
    text: str  # as displayed, e.g. "Anna", "don't", "U.S.", "café"
    spoken: str  # lowercase a-z (+ apostrophes) used for lookup / letters
    start: int  # offsets in the expanded text, for punctuation between words
    end: int
    initialism: bool = False  # "U.S.", "a.m.": read letter by letter


# Dotted initialisms first ("U.S.", "a.m."), then words with any letters
# (accented too) and inner apostrophes.
_SPEECH_TOKEN = re.compile(
    r"(?P<init>(?:[^\W\d_]\.){2,})|(?P<word>[^\W\d_]+(?:['’][^\W\d_]+)*)"
)


def _fold(text: str) -> str:
    """"Café" -> "cafe": strip accents, keep only a-z and apostrophes."""
    decomposed = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    return "".join(ch for ch in decomposed if "a" <= ch <= "z" or ch == "'")


def speech_tokens(text: str) -> tuple[str, list[SpeechToken]]:
    """(expanded text, tokens). Tokens with no a-z letters are dropped."""
    expanded = expand_for_speech(text or "")
    tokens = []
    for m in _SPEECH_TOKEN.finditer(expanded):
        initialism = m.group("init") is not None
        spoken = _fold(m.group().replace(".", "")).strip("'")
        if not spoken.replace("'", ""):
            continue
        tokens.append(SpeechToken(m.group(), spoken, m.start(), m.end(), initialism))
    return expanded, tokens
