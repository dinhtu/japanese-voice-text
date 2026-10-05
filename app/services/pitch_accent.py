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


def _hira_to_kata(text: str) -> str:
    return "".join(
        chr(ord(ch) + 0x60) if 0x3041 <= ord(ch) <= 0x3096 else ch for ch in text
    )


def _kata_to_hira(text: str) -> str:
    return "".join(
        chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch for ch in text
    )


def _is_kana(ch: str) -> bool:
    cp = ord(ch)
    return (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF) or ch == "ー"


def pitch_accent_patterns(text: str) -> list[list[MoraPitch]]:
    """Generate all valid pitch accent pattern variants directly from UniDic aType features."""
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
    phrase_idx = 0

    # Collect variants per node: list of list of (pitch, phrase_idx)
    node_variants: list[list[list[tuple[str, int]]]] = []

    for w in nodes:
        pos = getattr(w.feature, "pos1", "")
        pron = getattr(w.feature, "pron", "") or getattr(w.feature, "kana", "") or w.surface
        token_morae = _split_kana_morae("".join(ch for ch in _kata_to_hira(pron) if _is_kana(ch)))
        if not token_morae:
            continue

        atype_str = getattr(w.feature, "aType", "*")
        candidate_atypes: list[int | None] = []
        if atype_str and atype_str != "*":
            candidate_atypes = [int(x.strip()) for x in atype_str.split(",") if x.strip().isdigit()]

        # Fallback: if hiragana-written loanword (e.g. ぷろじぇくと), try looking up Katakana in UniDic
        if not candidate_atypes and w.surface:
            kata_surface = _hira_to_kata(w.surface)
            if kata_surface != w.surface:
                kata_nodes = list(_fugashi_tagger(kata_surface))
                if kata_nodes:
                    k_atype_str = getattr(kata_nodes[0].feature, "aType", "*")
                    if k_atype_str and k_atype_str != "*":
                        candidate_atypes = [
                            int(x.strip()) for x in k_atype_str.split(",") if x.strip().isdigit()
                        ]

        if not candidate_atypes:
            candidate_atypes = [None]

        w_options: list[list[tuple[str, int]]] = []
        for atype in candidate_atypes:
            hl: list[tuple[str, int]] = []
            if pos in ("助詞", "助動詞") and atype is None:
                for _ in token_morae:
                    hl.append(("L", phrase_idx))
            elif atype == 1:
                for idx, _ in enumerate(token_morae):
                    hl.append(("H" if idx == 0 else "L", phrase_idx))
            elif atype == 0:
                for idx, _ in enumerate(token_morae):
                    hl.append(("L" if idx == 0 and len(token_morae) > 1 else "H", phrase_idx))
            elif atype and atype >= 2:
                for idx, _ in enumerate(token_morae):
                    m_idx = idx + 1
                    if m_idx == 1:
                        hl.append(("L", phrase_idx))
                    elif m_idx <= atype:
                        hl.append(("H", phrase_idx))
                    else:
                        hl.append(("L", phrase_idx))
            else:
                # Standard Japanese default for unknown content words: Heiban (L H H H...)
                for idx, _ in enumerate(token_morae):
                    hl.append(("L" if idx == 0 and len(token_morae) > 1 else "H", phrase_idx))
            w_options.append(hl)

        node_variants.append(w_options)

    # Cartesion product of node options (limit up to 5 unique combinations)
    combos: list[list[tuple[str, int]]] = [[]]
    for w_opts in node_variants:
        next_combos: list[list[tuple[str, int]]] = []
        for c in combos:
            for opt in w_opts:
                next_combos.append(c + opt)
        combos = next_combos[:5]

    all_patterns: list[list[MoraPitch]] = []
    seen: set[tuple[str, ...]] = set()

    for hl_list in combos:
        mora_pitches: list[MoraPitch] = []
        for i, mora in enumerate(full_morae):
            pitch, p_idx = hl_list[i] if i < len(hl_list) else ("L", 0)
            mora_pitches.append(MoraPitch(mora=mora, pitch=pitch, phrase=p_idx))
        key = tuple(m.pitch for m in mora_pitches)
        if key not in seen:
            seen.add(key)
            all_patterns.append(mora_pitches)

    return all_patterns or [[MoraPitch(mora=m, pitch="L", phrase=0) for m in full_morae]]


def pitch_accent_pattern(text: str) -> list[MoraPitch]:
    """Generate primary pitch accent pattern directly from UniDic aType features."""
    patterns = pitch_accent_patterns(text)
    return patterns[0]
