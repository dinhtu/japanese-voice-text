"""Reference pitch-accent ("cao do mau") pattern for Japanese text.

Same theoretical H/L as jp-pitch-accent-analyzer
(https://github.com/deeplearningcafe/jp-pitch-accent-analyzer):

  1. `pyopenjtalk.extract_fullcontext(..., run_marine=True)` when the
     Marine estimator is installed -- corpus-trained accent, better on
     compounds than the raw NAIST dictionary.
  2. One H/L per mora from the HTS `/A:p1+p2+p3/` fields, using that
     repo's rule (not a 2-rail guess):

        p1 > 0              -> Low   (already past the nucleus)
        p1 < 0 and p2 == 1  -> Low   (first mora, still before nucleus)
        otherwise           -> High  (at the nucleus, or rising toward it)

     which is heiban LHHHH, atamadaka HLLL, naka/odaka LHH..HLL.

The frontend draws this as an OJAD-style step (horizontal per mora,
vertical only when H/L changes, gap at accent-phrase boundaries).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pyopenjtalk

logger = logging.getLogger(__name__)

_SMALL_YOON = set("ゃゅょぁぃぅぇぉャュョァィゥェォ")

_P3_RE = re.compile(r"-(.*?)\+")
_A_RE = re.compile(r"/A:([0-9\-]+)\+([0-9]+)\+([0-9]+)")

_marine_unavailable = False


@dataclass
class MoraPitch:
    """One mora of the sentence and the pitch it should be read at."""

    mora: str
    pitch: str  # "H" | "L"
    phrase: int  # 0-based accent phrase index


def _phone(label: str) -> str:
    match = _P3_RE.search(label)
    return match.group(1) if match else ""


def _extract_fullcontext(text: str) -> list[str]:
    """Prefer Marine (same as jp-pitch-accent-analyzer); fall back to dictionary."""
    global _marine_unavailable
    if not _marine_unavailable:
        try:
            return pyopenjtalk.extract_fullcontext(text, run_marine=True)
        except TypeError:
            return pyopenjtalk.extract_fullcontext(text)
        except Exception as exc:  # noqa: BLE001 — marine missing / model load
            _marine_unavailable = True
            logger.info(
                "Marine accent unavailable (%s); using OpenJTalk dictionary", exc
            )
    return pyopenjtalk.extract_fullcontext(text)


def _hl_from_a(p1: int, p2: int) -> bool:
    """True = High. Same three-way rule as jp-pitch-accent-analyzer."""
    if p1 > 0:
        return False
    if p1 < 0 and p2 == 1:
        return False
    return True


def _mora_hl_from_labels(labels: list[str]) -> list[tuple[bool, int]]:
    """(is_high, phrase_index) per mora, jp-pitch-accent-analyzer rules."""
    raw: list[tuple[int, int, int]] = []  # (p1, p2, phrase)
    phrase_idx = 0
    last_p2: int | None = None

    for label in labels:
        ph = _phone(label)
        if ph == "sil":
            continue
        if ph == "pau":
            if last_p2 is not None:
                phrase_idx += 1
                last_p2 = None
            continue

        a_match = _A_RE.search(label)
        if not a_match:
            continue
        p1 = int(a_match.group(1))
        p2 = int(a_match.group(2))

        if p2 == last_p2:
            continue
        if last_p2 is not None and p2 < last_p2:
            phrase_idx += 1

        raw.append((p1, p2, phrase_idx))
        last_p2 = p2

    # Some OpenJTalk builds mark heiban as all-positive p1 (no nucleus).
    # The repo rule would then paint every mora Low; force LHHHH instead.
    by_phrase: dict[int, list[int]] = {}
    for i, (p1, _p2, phrase) in enumerate(raw):
        by_phrase.setdefault(phrase, []).append(i)
    heiban_phrases = {
        phrase
        for phrase, idxs in by_phrase.items()
        if idxs and all(raw[i][0] > 0 for i in idxs)
    }

    out: list[tuple[bool, int]] = []
    for p1, p2, phrase in raw:
        if phrase in heiban_phrases:
            is_high = p2 != 1
        else:
            is_high = _hl_from_a(p1, p2)
        out.append((is_high, phrase))
    return out


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
    """Return the expected per-mora High/Low pitch for `text`.

    Raises:
        ValueError: `text` has no pronounceable Japanese content.
    """
    if not text or not text.strip():
        raise ValueError("Text is empty; nothing to analyze.")

    labels = _extract_fullcontext(text)
    hl = _mora_hl_from_labels(labels)
    if not hl:
        raise ValueError("No pronounceable Japanese content found.")

    katakana = pyopenjtalk.g2p(text, kana=True) or ""
    kana = "".join(ch for ch in _kata_to_hira(katakana) if _is_kana(ch))
    kana_morae = _split_kana_morae(kana)

    if len(kana_morae) != len(hl):
        logger.warning(
            "pitch/kana mora count mismatch (%d vs %d) for %r",
            len(hl),
            len(kana_morae),
            text,
        )

    result: list[MoraPitch] = []
    n = max(len(hl), len(kana_morae))
    for i in range(n):
        is_high, phrase = hl[i] if i < len(hl) else (False, hl[-1][1] if hl else 0)
        mora = kana_morae[i] if i < len(kana_morae) else "?"
        result.append(MoraPitch(mora=mora, pitch="H" if is_high else "L", phrase=phrase))
    return result
