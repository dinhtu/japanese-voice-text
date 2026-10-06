"""Japanese text to kana (hiragana) conversion using pyopenjtalk."""

import logging
import re
import unicodedata

try:
    import pyopenjtalk
except Exception:  # noqa: BLE001
    pyopenjtalk = None

try:
    import pykakasi

    _kakasi = pykakasi.kakasi()
except Exception:  # noqa: BLE001
    _kakasi = None

logger = logging.getLogger(__name__)

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
        return "pyopenjtalk" if pyopenjtalk is not None else "pykakasi"

    def text_to_kana(self, text: str) -> str:
        text = _CLEAN_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        katakana = ""
        # 1. Primary engine: pyopenjtalk (accurate context-aware G2P)
        if pyopenjtalk is not None:
            try:
                katakana = pyopenjtalk.g2p(text, kana=True)
            except Exception:  # noqa: BLE001
                logger.debug("pyopenjtalk failed, falling back to pykakasi", exc_info=True)
                katakana = ""

        # 2. Fallback engine: pykakasi
        if not katakana and _kakasi is not None:
            # Preprocessing rule to prevent pykakasi from converting '今日は' into 'こんにちは'
            preprocessed = re.sub(r"今日(?=は)", "今日 ", text)
            res = _kakasi.convert(preprocessed)
            katakana = "".join(item["kana"] for item in res)

        if not katakana:
            return ""

        # NFKC normalization
        katakana = unicodedata.normalize("NFKC", katakana)
        katakana = "".join(_ALPHA_TO_KATA.get(ch.upper(), ch) for ch in katakana)
        katakana = katakana.translate(_DROP_CHARS)
        hiragana = "".join(ch for ch in self._kata_to_hira(katakana) if self._is_kana(ch))

        if not hiragana:
            return ""

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
