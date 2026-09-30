"""Korean text normalization for ASR comparison."""

import re
import unicodedata

_HANGUL = re.compile(r"[가-힣]")


def normalize_korean(text: str) -> str:
    """Compose Hangul jamo and keep modern Hangul syllables only."""
    return "".join(_HANGUL.findall(unicodedata.normalize("NFC", text or "")))


def korean_syllables(text: str) -> list[str]:
    return list(normalize_korean(text))
