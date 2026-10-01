"""Japanese text to kana (hiragana) conversion using fugashi (NINJAL UniDic) & pyopenjtalk."""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

try:
    import fugashi

    _fugashi_tagger = fugashi.Tagger()
except Exception:  # noqa: BLE001
    _fugashi_tagger = None

try:
    import pyopenjtalk
except Exception:  # noqa: BLE001
    pyopenjtalk = None

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

    def _g2p_unidic(self, text: str) -> str:
        if _fugashi_tagger is None:
            return ""
        try:
            nodes = list(_fugashi_tagger(text))
            words: list[str] = []
            for w in nodes:
                pron = getattr(w.feature, "pron", None)
                kana = getattr(w.feature, "kana", None)
                if pron and pron != "*":
                    words.append(pron)
                elif kana and kana != "*":
                    words.append(kana)
                else:
                    words.append(w.surface)
            return "".join(words)
        except Exception:  # noqa: BLE001
            logger.debug("Fugashi UniDic G2P failed; falling back", exc_info=True)
            return ""

    def text_to_kana(self, text: str) -> str:
        text = _CLEAN_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        katakana = self._g2p_unidic(text)
        if not katakana and pyopenjtalk is not None:
            try:
                katakana = pyopenjtalk.g2p(text, kana=True)
            except Exception:  # noqa: BLE001
                katakana = ""
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
