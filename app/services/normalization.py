"""Japanese text normalization for pronunciation comparison.

Both the admin's target text and the ASR output are pushed through the same
reading-based normalizer so that only real pronunciation differences survive:

    今日も一日頑張りましょう  ->  きょーもいちにちがんばりましょー
    げんきょうも...ましょう    ->  げんきょーも...ましょー
    げんきょーも...ましょー    ->  げんきょーも...ましょー  (idempotent)

Reuses the project's existing JapaneseKanaConverter (pyopenjtalk), which already
handles kanji -> reading, katakana -> hiragana, NFKC width folding, long vowels
(おう -> おー) and punctuation stripping.
"""

import re

from src.asr.kana_converter import JapaneseKanaConverter

# Characters the ASR vocabulary can emit; used by the fallback path.
_KANA_ONLY_RE = re.compile(r"[^ぁ-ゖァ-ヺーｱ-ﾝ]")

_converter = JapaneseKanaConverter()


def _kata_to_hira(text: str) -> str:
    return "".join(
        chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch for ch in text
    )


def _strip_to_kana(text: str) -> str:
    """Fallback: keep only kana characters, fold katakana to hiragana."""
    return _kata_to_hira(_KANA_ONLY_RE.sub("", text))


def to_hiragana(text: str) -> str:
    """Normalize any Japanese text to a bare hiragana reading string.

    Returns an empty string for input with no pronounceable content.
    """
    if not text or not text.strip():
        return ""

    try:
        kana = _converter.text_to_kana(text).replace(" ", "")
    except Exception:  # noqa: BLE001 — pyopenjtalk can fail on odd input
        kana = ""

    # Garbled/partial ASR output can defeat the g2p parser; keep the raw kana.
    return kana or _strip_to_kana(text)
