"""Pitch-accent pattern extraction tests (no ASR model involved)."""

import pytest

from app.services.pitch_accent import pitch_accent_pattern


def test_heiban_word_is_low_then_high():
    """konnichiwa (heiban, no accent nucleus): L H H H H.

    Matches the documented example for ESPnet's pyopenjtalk_g2p_prosody,
    which this module's a1/a2/a3 parsing is based on:
    ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']
    -- the '[' between mora 1 and 2 is the only transition, i.e. low then
    high for the rest of the phrase.
    """
    moras = pitch_accent_pattern("こんにちは")
    assert [m.mora for m in moras] == ["こ", "ん", "に", "ち", "は"]
    assert [m.pitch for m in moras] == ["L", "H", "H", "H", "H"]


def test_practice_sentence_mora_count_and_reading():
    """chotto matte kudasai splits into 10 morae, matching the on-screen chart."""
    moras = pitch_accent_pattern("ちょっと待ってください")
    assert [m.mora for m in moras] == [
        "ちょ", "っ", "と", "ま", "っ", "て", "く", "だ", "さ", "い",
    ]
    # Every mora gets a pitch, and only H/L values are ever produced.
    assert {m.pitch for m in moras} <= {"H", "L"}
    # ちょっと is atamadaka (HLL) in the standard dictionary / Marine.
    assert [m.pitch for m in moras[:3]] == ["H", "L", "L"]


def test_empty_text_raises():
    with pytest.raises(ValueError):
        pitch_accent_pattern("")
    with pytest.raises(ValueError):
        pitch_accent_pattern("   ")
