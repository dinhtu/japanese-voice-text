"""Japanese text to kana (hiragana) conversion using Ollama with pykakasi fallback."""

import json
import logging
import os
import re
import unicodedata
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

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

# Per-request override of KANA_USE_OLLAMA (e.g. /pitch-accent?flag_use_ollama=1).
# None = follow the env var. A ContextVar keeps concurrent requests separate.
_ollama_override: ContextVar[bool | None] = ContextVar("kana_use_ollama_override", default=None)


@contextmanager
def use_ollama_g2p(enabled: bool | None = True) -> Iterator[None]:
    """Force Ollama kana conversion on (True) / off (False) inside the block;
    None leaves the KANA_USE_OLLAMA env setting in charge."""
    token = _ollama_override.set(enabled)
    try:
        yield
    finally:
        _ollama_override.reset(token)


class JapaneseKanaConverter:
    """Convert Japanese text to space-separated hiragana characters.

    Primary engine: Locally hosted Ollama model (e.g. qwen3:8b / qwen2.5:14b).
    Fallback engine: pykakasi (with regex preprocessing).
    """

    def _is_ollama_enabled(self) -> bool:
        override = _ollama_override.get()
        if override is not None:
            return override
        flag = os.getenv("KANA_USE_OLLAMA") or os.getenv("USE_OLLAMA_G2P") or "0"
        return flag.strip().lower() in ("1", "true", "yes", "on")

    @property
    def engine(self) -> str:
        return "ollama+pykakasi" if self._is_ollama_enabled() else "pykakasi"

    def _ollama_to_kana(self, text: str) -> str:
        """Call local Ollama to convert Japanese text to accurate Hiragana."""
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        timeout = float(os.getenv("OLLAMA_TIMEOUT_S", "5"))

        prompt = (
            "You are an expert native Japanese phonetic converter (Furigana / G2P engine).\n"
            "Convert the Japanese sentence into natural, spoken-accurate pure Hiragana (実際の口語・発音通りのひらがな).\n\n"
            "Key pronunciation rules:\n"
            "1. '何' reading:\n"
            "   - Read as 'なん' before counters, numbers, and /t, d, n, s, z/ sounds (e.g., 何時 -> なんじ, 何人 -> なんにん, 何年 -> なんねん, 何ですか -> なんですか, 何で -> なんで, 何個 -> なんこ, 何曜日 -> なんようび).\n"
            "   - Read as 'なに' before particles を, が, も, から, まで, or standalone (e.g., 何を食べますか -> なにをたべますか, 何が好きですか -> なにがすきですか, 何も -> なにも).\n"
            "2. '今日' reading:\n"
            "   - In sentences, read as 'きょう' (e.g., 今日は教室で -> きょうはきょうしつで), NEVER as 'こんにちは'.\n"
            "3. Compound words & Counters:\n"
            "   - 'お母さん' -> 'おかあさん', '時計' -> 'とけい', '一日' -> 'ついたち' (1st of month) or 'いちにち' (1 full day).\n\n"
            "Examples:\n"
            "- Input: 今日は教室で日本語を勉強します\n  Hiragana: きょうはきょうしつでにほんごをべんきょうします\n"
            "- Input: 何時に何を食べますか？何人で行きますか？\n  Hiragana: なんじになにをたべますかなんにんでいきますか\n"
            "- Input: お母さんは時計を買いました。\n  Hiragana: おかあさんはとけいをかいました\n\n"
            "Format: Output ONLY the Hiragana text without spaces, Romaji, punctuation, markdown, or explanations.\n\n"
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
        # 1. Try local Ollama if enabled via env (KANA_USE_OLLAMA=1) or per request
        #    (use_ollama_g2p / flag_use_ollama=1 on /pitch-accent)
        if self._is_ollama_enabled():
            try:
                hira = self._ollama_to_kana(text)
            except Exception:  # noqa: BLE001
                logger.debug("Ollama G2P unavailable or failed; falling back to pykakasi", exc_info=True)
                hira = ""

        # 2. Fallback to pykakasi (or default if KANA_USE_OLLAMA=0)
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
