"""Normalize English target / ASR text for scoring and the pitch chart."""

from __future__ import annotations

import re
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
