"""Japanese text to kana (hiragana) conversion using pykakasi."""

import logging
import re
import unicodedata

import pykakasi

logger = logging.getLogger(__name__)

_kakasi = pykakasi.kakasi()

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

    @property
    def engine(self) -> str:
        return "pykakasi"

    def text_to_kana(self, text: str) -> str:
        text = _CLEAN_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        # pykakasi converts Japanese text into natural compound hiragana readings
        res = _kakasi.convert(text)
        hira = "".join(item["hira"] for item in res)
        if not hira:
            return ""

        # NFKC normalization
        hira = unicodedata.normalize("NFKC", hira)
        hira = "".join(_ALPHA_TO_KATA.get(ch.upper(), ch) for ch in hira)
        hira = hira.translate(_DROP_CHARS)
        hira = "".join(ch for ch in self._kata_to_hira(hira) if self._is_kana(ch))

        if not hira:
            return ""

        # Deliberately drop <sp> to avoid brittle word-boundary supervision.
        return " ".join(list(hira))

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
