"""Japanese text to phoneme sequence conversion using pyopenjtalk."""

import re

import pyopenjtalk

from src.asr.phoneme_vocab import PHONEMES

_VALID_PHONEMES = set(PHONEMES)
_DROP_TOKENS = {"pau", "sil"}
_CLEAN_RE = re.compile(r"[・]+")


class JapanesePhonemeConverter:
    """Convert Japanese text to project-compatible phoneme tokens."""

    def text_to_phonemes(self, text: str) -> str:
        text = _CLEAN_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        phoneme_str = pyopenjtalk.g2p(text, kana=False)
        if not phoneme_str:
            return ""

        out: list[str] = []
        for token in phoneme_str.split():
            if token in _DROP_TOKENS:
                continue
            if token in _VALID_PHONEMES:
                out.append(token)

        return " ".join(out)
