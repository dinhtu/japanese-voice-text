"""Japanese text to kana (hiragana) conversion using Ollama with pykakasi fallback."""

import json
import logging
import os
import re
import unicodedata
import urllib.request

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
    """Convert Japanese text to space-separated hiragana characters.

    Primary engine: Locally hosted Ollama model (e.g. qwen3:8b / qwen2.5:14b).
    Fallback engine: pykakasi (with regex preprocessing).
    """

    @property
    def engine(self) -> str:
        return "ollama+pykakasi"

    def _ollama_to_kana(self, text: str) -> str:
        """Call local Ollama to convert Japanese text to accurate Hiragana."""
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        timeout = float(os.getenv("OLLAMA_TIMEOUT_S", "5"))

        prompt = (
            "Convert the following Japanese sentence into pure Hiragana reading according to natural pronunciation.\n"
            "Output ONLY the Hiragana text without spaces, Romaji, punctuation, or explanations.\n\n"
            f"Input: {text}\n"
            "Hiragana:"
        )
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw = data.get("response", "").strip()
            raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
            raw = re.sub(r"[\r\n\t ]+", "", raw)
            return raw

    def text_to_kana(self, text: str) -> str:
        text = _CLEAN_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        hira = ""
        # 1. Try local Ollama first for 100% context-aware accurate G2P
        try:
            hira = self._ollama_to_kana(text)
        except Exception:  # noqa: BLE001
            logger.debug("Ollama G2P unavailable or failed; falling back to pykakasi", exc_info=True)
            hira = ""

        # 2. Fallback to pykakasi if Ollama is offline or timed out
        if not hira:
            # Preprocessing: separate '今日' and 'は' to prevent pykakasi from mapping
            # the noun '今日' (kyou) + particle 'は' (wa) into the greeting 'こんにちは'.
            preprocessed = re.sub(r"今日(?=は)", "今日 ", text)
            res = _kakasi.convert(preprocessed)
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
