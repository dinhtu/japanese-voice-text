"""Reference pitch-accent ("cao do mau") pattern for Japanese text.

Given a sentence, returns the expected High/Low pitch for each mora - the
same rise-and-fall shape a native speaker produces, and the same signal
Japanese TTS engines are driven by. Used to show a learner the intonation
they should aim for *before* they record themselves reading it.

## Where the signal comes from

`pyopenjtalk.extract_fullcontext()` returns one HTS full-context label per
*phoneme*. Three numeric fields recur on every label and are all we need:

    a1  distance from this mora to the accent nucleus of its accent
        phrase - exactly 0 at the nucleus mora, never 0 in a phrase with
        no nucleus (heiban / flat accent).
    a2  position of this mora within its accent phrase, counting from 1.
    a3  position of this mora counting backward from the end of its
        accent phrase (1 at the phrase's last mora).

All phones belonging to one mora (e.g. the "ch" and "o" of "cho" in
"chotto") carry the *same* a1/a2/a3, so grouping consecutive phones by
those three numbers recovers the mora and accent-phrase boundaries
without needing a separate word segmenter. From there, standard Japanese
pitch-accent rules give a mechanical High/Low pattern per accent phrase:

    nucleus at mora 1  (atamadaka):  H L L L ...
    no nucleus         (heiban):     L H H H ...
    nucleus at mora k  (naka/odaka): L H ... H(k) L L ...

This is the same a1/a2/a3 signal open-source Japanese TTS front-ends
encode as prosody symbols for a neural model (see ESPnet's
`pyopenjtalk_g2p_prosody`); here it is turned into an explicit H/L value
per mora instead, for display, rather than fed to a synthesizer.

## Limitation

This is OpenJTalk's *dictionary* accent for each word, phrase-joined by
its own built-in rules - not a model fit to a labelled accent corpus. It
is usually right, but compound nouns and unusual proper nouns can shift
accent in ways only a native speaker (or a corpus-trained accent
estimator, e.g. the "marine" package) would catch.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pyopenjtalk

logger = logging.getLogger(__name__)

# Phones that always end a mora: vowels, the moraic nasal, the geminate
# ("small tsu"). Everything else (consonants) starts a mora and waits for
# one of these to close it.
_MORA_FINAL = set("aiueoAIUEO") | {"N", "cl"}

# Small kana that fuse with the *previous* character into one mora
# (kya, fa, ...) rather than counting as a mora of their own.
_SMALL_YOON = set("ゃゅょぁぃぅぇぉャュョァィゥェォ")

_P3_RE = re.compile(r"-(.*?)\+")
_A1_RE = re.compile(r"/A:([0-9\-]+)\+")
_A2_RE = re.compile(r"\+(\d+)\+")
_A3_RE = re.compile(r"\+(\d+)/")


def _feature(regex: "re.Pattern[str]", label: str) -> int | None:
    match = regex.search(label)
    return int(match.group(1)) if match else None


def _phone(label: str) -> str:
    match = _P3_RE.search(label)
    return match.group(1) if match else ""


@dataclass
class MoraPitch:
    """One mora of the sentence and the pitch it should be read at."""

    mora: str
    pitch: str  # "H" | "L"
    phrase: int  # 0-based accent phrase index


def _group_accent_phrases(labels: list[str]) -> list[list[tuple[int, int, int]]]:
    """Group full-context labels into accent phrases of (a1, a2, a3) morae."""
    phrases: list[list[tuple[int, int, int]]] = [[]]
    phrase_ended = False

    for label in labels:
        p3 = _phone(label)
        if p3 == "sil":
            continue
        if p3 == "pau":
            if phrases[-1]:
                phrases.append([])
                phrase_ended = False
            continue
        if p3 not in _MORA_FINAL:
            continue  # leading consonant of a mora - wait for what closes it

        a1 = _feature(_A1_RE, label)
        a2 = _feature(_A2_RE, label)
        a3 = _feature(_A3_RE, label)
        if a1 is None or a2 is None or a3 is None:
            continue

        if phrase_ended:
            phrases.append([])
            phrase_ended = False
        phrases[-1].append((a1, a2, a3))
        if a3 == 1:
            phrase_ended = True

    return [phrase for phrase in phrases if phrase]


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

    labels = pyopenjtalk.extract_fullcontext(text)
    phrases = _group_accent_phrases(labels)
    if not phrases:
        raise ValueError("No pronounceable Japanese content found.")

    katakana = pyopenjtalk.g2p(text, kana=True) or ""
    kana = "".join(ch for ch in _kata_to_hira(katakana) if _is_kana(ch))
    kana_morae = _split_kana_morae(kana)

    result: list[MoraPitch] = []
    for phrase_idx, phrase in enumerate(phrases):
        nucleus = next((a2 for a1, a2, _a3 in phrase if a1 == 0), None)
        for a1, a2, _a3 in phrase:
            if nucleus is None:
                pitch = "L" if a2 == 1 else "H"  # heiban: low, then high forever
            elif nucleus == 1:
                pitch = "H" if a2 == 1 else "L"  # atamadaka: high, then low forever
            else:
                pitch = "L" if (a2 == 1 or a2 > nucleus) else "H"  # naka/odaka
            result.append(MoraPitch(mora="", pitch=pitch, phrase=phrase_idx))

    # Both mora lists come from OpenJTalk's analysis of the same text, so
    # counts normally match; fall back to a placeholder instead of failing
    # the whole request over a rare digraph/loanword mismatch.
    if len(kana_morae) != len(result):
        logger.warning(
            "pitch/kana mora count mismatch (%d vs %d) for %r",
            len(result), len(kana_morae), text,
        )
    for i, mp in enumerate(result):
        mp.mora = kana_morae[i] if i < len(kana_morae) else "?"

    return result
