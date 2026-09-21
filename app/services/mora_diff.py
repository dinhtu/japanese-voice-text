"""Per-mora pronunciation status ("Tung am tiet" grid).

Regroups the character-level edit-distance diff already computed for
`/evaluate` (see `app/services/scoring.py`) onto the target's morae, so the
UI can highlight which syllable(s) of the sentence were mispronounced.

## Limitation

This is a real signal, but it is binary: "this mora's character(s) came out
of the edit-distance alignment as a match" vs. "as a substitution/deletion".
It is not a continuously varying acoustic confidence score (the 96 / 54 / 78
style numbers some pronunciation apps show). Producing a genuine per-mora
number would need the ASR model's own softmax confidence exposed and
aligned per token, which `KanaRecognizer.transcribe()` does not currently
return - that would be a real change to the decode pipeline, not something
this function fabricates. Until that exists, mora-level feedback here is
"matches the target" (green) vs. "doesn't" (red), built from data the app
already computes for the existing error list, nothing invented.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.scoring import PronunciationError

# Small kana that fuse with the *previous* character into one mora
# (kya, sha, ...) rather than counting as a mora of their own - same rule
# used in app/services/pitch_accent.py, kept local here so this module has
# no dependency on pyopenjtalk.
_SMALL_YOON = set("ゃゅょぁぃぅぇぉャュョァィゥェォ")


@dataclass
class MoraStatus:
    """One mora of the target sentence and whether it came out correct."""

    mora: str
    ok: bool  # True = matched the target ("dung"), False = sub/del ("sai")


def _split_mora_spans(kana: str) -> list[tuple[str, int, int]]:
    """Split kana text into (mora_text, start_index, end_index) spans.

    `end_index` is exclusive, so `kana[start:end] == mora_text`. Mirrors
    `pitch_accent._split_kana_morae`'s small-yoon-fusion rule, but also
    tracks each mora's character span in the original string.
    """
    spans: list[tuple[str, int, int]] = []
    for i, ch in enumerate(kana):
        if ch in _SMALL_YOON and spans:
            text, start, _end = spans[-1]
            spans[-1] = (text + ch, start, i + 1)
        else:
            spans.append((ch, i, i + 1))
    return spans


def split_mora_spans(kana: str) -> list[tuple[str, int, int]]:
    """Public entry point for `_split_mora_spans`, for other modules that
    need target-mora character boundaries (e.g. app.services.mora_timing's
    ASR-based pitch alignment)."""
    return _split_mora_spans(kana)


def mora_status(
    target_hiragana: str, errors: list[PronunciationError]
) -> list[MoraStatus]:
    """Map character-level edit-distance errors onto the target's morae.

    A mora is marked wrong when a "sub" or "del" operation lands inside its
    character span (both are indexed by target character position - see
    `scoring.collect_errors`). "ins" operations are extra sounds the ASR
    heard that aren't in the target text at all, so they don't land on any
    target mora and aren't represented in this per-mora view; they still
    appear in the full `errors` list.
    """
    spans = _split_mora_spans(target_hiragana)
    bad_positions = {e.position for e in errors if e.type in ("sub", "del")}
    return [
        MoraStatus(
            mora=text,
            ok=not any(start <= pos < end for pos in bad_positions),
        )
        for text, start, end in spans
    ]
