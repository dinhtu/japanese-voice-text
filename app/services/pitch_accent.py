"""Reference pitch-accent ("cao do mau") pattern for Japanese text using UniDic (NINJAL)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import fugashi

logger = logging.getLogger(__name__)

_fugashi_tagger = fugashi.Tagger()

_SMALL_YOON = set("ゃゅょぁぃぅぇぉャュョァィゥェォ")


@dataclass
class MoraPitch:
    """One mora of the sentence and the pitch it should be read at."""

    mora: str
    pitch: str  # "H" | "L"
    phrase: int  # 0-based accent phrase index


def _split_kana_morae(kana: str) -> list[str]:
    """Split a kana reading into individual morae (kya stays one mora)."""
    morae: list[str] = []
    for ch in kana:
        if ch in _SMALL_YOON and morae:
            morae[-1] += ch
        else:
            morae.append(ch)
    return morae


def _kata_to_hira(text: str) -> str:
    return "".join(
        chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch for ch in text
    )


def _is_kana(ch: str) -> bool:
    cp = ord(ch)
    return (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF) or ch == "ー"


def pitch_accent_pattern(text: str) -> list[MoraPitch]:
    """Generate pitch accent pattern directly from UniDic aType features."""
    if not text or not text.strip():
        raise ValueError("Text is empty; nothing to analyze.")

    from app.services.normalization import to_hiragana

    full_kana = to_hiragana(text)
    if not full_kana:
        raise ValueError("No pronounceable Japanese content found.")

    full_morae = _split_kana_morae(full_kana)
    if not full_morae:
        raise ValueError("No pronounceable Japanese content found.")

    nodes = list(_fugashi_tagger(text))
    hl_list: list[tuple[str, int]] = []
    downstep_occurred = False
    in_high = False
    phrase_idx = 0

    for w in nodes:
        pos = getattr(w.feature, "pos1", "")
        pron = getattr(w.feature, "pron", "") or getattr(w.feature, "kana", "") or w.surface
        token_morae = _split_kana_morae("".join(ch for ch in _kata_to_hira(pron) if _is_kana(ch)))
        if not token_morae:
            continue

        atype_str = getattr(w.feature, "aType", "*")
        atype = int(atype_str) if atype_str and atype_str.isdigit() else None

        if pos in ("助詞", "助動詞") and atype is None:
            for _ in token_morae:
                hl_list.append(("L" if downstep_occurred or not in_high else "H", phrase_idx))
        else:
            downstep_occurred = False
            if atype == 1:
                in_high = False
                downstep_occurred = True
                for idx, _ in enumerate(token_morae):
                    hl_list.append(("H" if idx == 0 else "L", phrase_idx))
            elif atype == 0:
                in_high = True
                for idx, _ in enumerate(token_morae):
                    hl_list.append(("L" if idx == 0 and len(token_morae) > 1 else "H", phrase_idx))
            elif atype and atype >= 2:
                for idx, _ in enumerate(token_morae):
                    m_idx = idx + 1
                    if m_idx == 1:
                        hl_list.append(("L", phrase_idx))
                    elif m_idx <= atype:
                        hl_list.append(("H", phrase_idx))
                    else:
                        downstep_occurred = True
                        in_high = False
                        hl_list.append(("L", phrase_idx))
            else:
                for _ in token_morae:
                    hl_list.append(("H" if in_high else "L", phrase_idx))

    res: list[MoraPitch] = []
    for i, mora in enumerate(full_morae):
        pitch, phrase = hl_list[i] if i < len(hl_list) else ("L", 0)
        res.append(MoraPitch(mora=mora, pitch=pitch, phrase=phrase))
    return res
