"""Japanese text to kana (hiragana) conversion using pyopenjtalk."""

import re
import unicodedata

import pyopenjtalk

_ALPHA_TO_KATA = {
    "A": "エー", "B": "ビー", "C": "シー", "D": "ディー", "E": "イー", "F": "エフ",
    "G": "ジー", "H": "エイチ", "I": "アイ", "J": "ジェー", "K": "ケー", "L": "エル",
    "M": "エム", "N": "エヌ", "O": "オー", "P": "ピー", "Q": "キュー", "R": "アール",
    "S": "エス", "T": "ティー", "U": "ユー", "V": "ブイ", "W": "ダブリュー", "X": "エックス",
    "Y": "ワイ", "Z": "ゼット",
}
_DROP_CHARS = str.maketrans("", "", "、。？！,.!?「」『』（）()［］[]{}・…:;\"'`")
_CLEAN_RE = re.compile(r"[・]+")


class JapaneseKanaConverter:
    """Convert Japanese text to space-separated hiragana characters."""

    def text_to_kana(self, text: str) -> str:
        text = _CLEAN_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        katakana = pyopenjtalk.g2p(text, kana=True)
        if not katakana:
            return ""

        # NFKC: full-width latin (e.g. "Ａ") -> ASCII ("A")
        katakana = unicodedata.normalize("NFKC", katakana)
        katakana = "".join(_ALPHA_TO_KATA.get(ch.upper(), ch) for ch in katakana)
        katakana = katakana.translate(_DROP_CHARS)
        katakana = "".join(ch for ch in katakana if self._is_kana(ch))

        if not katakana:
            return ""

        hiragana = self._kata_to_hira(katakana)
        # Deliberately drop <sp> to avoid brittle word-boundary supervision.
        return " ".join(list(hiragana))

    def _kata_to_hira(self, text: str) -> str:
        """Convert katakana to hiragana."""
        result = []
        for ch in text:
            cp = ord(ch)
            if 0x30A1 <= cp <= 0x30F6:
                result.append(chr(cp - 0x60))
            elif ch == "ー":
                result.append("ー")
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def _is_kana(ch: str) -> bool:
        cp = ord(ch)
        return (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF) or ch == "ー"
