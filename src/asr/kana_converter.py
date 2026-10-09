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

# Engine behind the latest text_to_kana() result in this context ("ollama" /
# "pykakasi"), so the API can report where a reading came from.
_last_engine: ContextVar[str | None] = ContextVar("kana_last_engine", default=None)

_unidic_tagger = None


def last_g2p_engine() -> str | None:
    """Engine that produced the latest text_to_kana() result in this context."""
    return _last_engine.get()


def _to_hira(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch for ch in text)


def _unidic_draft(text: str) -> str:
    """Dictionary reading per word, e.g. "出身[しゅっしん] は[は]", used as the
    Ollama draft. Empty string if fugashi/UniDic is unavailable."""
    global _unidic_tagger
    try:
        if _unidic_tagger is None:
            import fugashi

            _unidic_tagger = fugashi.Tagger()
        parts = []
        for w in _unidic_tagger(text):
            kana = getattr(w.feature, "kana", None)
            if kana and kana != "*" and _to_hira(kana) != w.surface:
                parts.append(f"{w.surface}[{_to_hira(kana)}]")
            else:
                parts.append(w.surface)
        return " ".join(parts)
    except Exception:  # noqa: BLE001
        logger.debug("UniDic draft unavailable", exc_info=True)
        return ""


_OLLAMA_PROMPT = """You are an expert Japanese reading engine (furigana / G2P).
Convert the Japanese text in <text> into pure hiragana exactly as a native speaker reads it aloud.

<draft> is a dictionary reading: each word followed by its reading in [brackets].
- The draft is usually right. KEEP it, including small っ (促音), ん and long vowels.
  Example: 出身 is しゅっしん (never しゅつしん, never しんしゅつ).
- Change a draft reading ONLY when the dictionary picked the wrong reading for this context:
  - 何: なん before counters and t/d/n sounds (何時 なんじ, 何人 なんにん, 何ですか なんですか, 何で なんで);
    なに before を/が/も or standalone (何を なにを, 何が なにが, 何も なにも).
  - 今日 in a sentence is きょう, never こんにち / こんにちは.
  - お母さん おかあさん, お父さん おとうさん, 日本 にほん, 上手 じょうず, 大人 おとな,
    一日 ついたち (date) / いちにち (whole day).
- Keep particles as written: は stays は, を stays を, へ stays へ.
- Never reorder, drop or add sounds. Ignore punctuation.

Examples:
<text>出身</text>
<draft>出身[しゅっしん]</draft>
Hiragana: しゅっしん

<text>学校に行きます。</text>
<draft>学校[がっこう] に 行き[いき] ます 。</draft>
Hiragana: がっこうにいきます

<text>今日は教室で日本語を勉強します</text>
<draft>今日[きょう] は 教室[きょうしつ] で 日本[にっぽん] 語[ご] を 勉強[べんきょう] し ます</draft>
Hiragana: きょうはきょうしつでにほんごをべんきょうします

<text>何時に何を食べますか？</text>
<draft>何[なん] 時[じ] に 何[なん] を 食べ[たべ] ます か ？</draft>
Hiragana: なんじになにをたべますか

<text>お母さんは時計を買いました。</text>
<draft>お 母[はは] さん は 時計[とけい] を 買い[かい] まし た 。</draft>
Hiragana: おかあさんはとけいをかいました

Output ONLY the hiragana: no spaces, romaji, punctuation, markdown or explanation.

<text>{text}</text>
<draft>{draft}</draft>
Hiragana:"""


def build_ollama_prompt(text: str) -> str:
    draft = _unidic_draft(text) or text
    return _OLLAMA_PROMPT.replace("{text}", text).replace("{draft}", draft)


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

        payload = json.dumps({
            "model": model,
            "prompt": build_ollama_prompt(text),
            "stream": False,
            # qwen3 otherwise "thinks" first; its kana would leak into the reading.
            "think": False,
            # Greedy + fixed seed: same text -> same reading on every call.
            "options": {
                "temperature": 0.0,
                "top_k": 1,
                "seed": 42,
                "num_predict": max(64, len(text) * 4),
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw = data.get("response", "").strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
            raw = re.sub(r"[\r\n\t ]+", "", raw)
            return raw

    def text_to_kana(self, text: str) -> str:
        _last_engine.set(None)
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
            if not any(self._is_kana(ch) for ch in hira):
                hira = ""  # romaji/explanation only -> no usable reading
            if hira:
                _last_engine.set("ollama")

        # 2. Fallback to pykakasi (or default if KANA_USE_OLLAMA=0)
        if not hira:
            _last_engine.set("pykakasi")
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
