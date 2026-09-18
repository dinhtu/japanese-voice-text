"""Japanese normalization tests."""

import pytest

from app.services.normalization import to_hiragana


@pytest.mark.parametrize(
    "text, expected",
    [
        ("げんきょうもいちにちがんばりましょう", "げんきょーもいちにちがんばりましょー"),
        ("今日も一日頑張りましょう", "きょーもいちにちがんばりましょー"),
        # Already-normalized ASR output must pass through unchanged.
        ("げんきょーもいちにちがんばりましょー", "げんきょーもいちにちがんばりましょー"),
        ("コンピューター", "こんぴゅーたー"),
    ],
)
def test_readings(text, expected):
    assert to_hiragana(text) == expected


def test_punctuation_and_whitespace_are_dropped():
    assert to_hiragana("こんにちは、 げんき？") == to_hiragana("こんにちはげんき")


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_input_returns_empty(text):
    assert to_hiragana(text) == ""
