"""CMU-style English phones for GOPT (stress stripped, 39 classes + pad).

Ids are 0-38 so GOPT's one_hot(phn+1, 40) stays in range. No extra package.
"""

from __future__ import annotations

import re

from app.services.english_text import english_words

# SpeechOcean762 / Kaldi pure phones minus ZH (folded into SH).
PHONES: tuple[str, ...] = (
    "SIL", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D",
    "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH",
    "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S",
    "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z",
)
PHONE_TO_ID = {p: i for i, p in enumerate(PHONES)}

# Letter the wav2vec2-960h CTC head actually emits for this phone.
PHONE_TO_LETTER: dict[str, str] = {
    "SIL": "|",
    "AA": "A", "AE": "A", "AH": "A", "AO": "A", "AW": "A", "AY": "A",
    "EH": "E", "ER": "E", "EY": "E",
    "IH": "I", "IY": "I",
    "OW": "O", "OY": "O",
    "UH": "U", "UW": "U",
    "B": "B", "CH": "C", "D": "D", "DH": "D", "F": "F", "G": "G",
    "HH": "H", "JH": "J", "K": "K", "L": "L", "M": "M", "N": "N",
    "NG": "N", "P": "P", "R": "R", "S": "S", "SH": "S", "T": "T",
    "TH": "T", "V": "V", "W": "W", "Y": "Y", "Z": "Z",
}

# Practice sentences + common function words. Values are stress-stripped CMU.
_LEXICON: dict[str, tuple[str, ...]] = {
    "a": ("AH",),
    "an": ("AH", "N"),
    "and": ("AH", "N", "D"),
    "anna": ("AE", "N", "AH"),
    "are": ("AA", "R"),
    "as": ("AE", "Z"),
    "beautiful": ("B", "Y", "UW", "T", "AH", "F", "AH", "L"),
    "call": ("K", "AO", "L"),
    "day": ("D", "EY"),
    "for": ("F", "AO", "R"),
    "go": ("G", "OW"),
    "help": ("HH", "EH", "L", "P"),
    "how": ("HH", "AW"),
    "i": ("AY",),
    "is": ("IH", "Z"),
    "it": ("IH", "T"),
    "its": ("IH", "T", "S"),
    "let": ("L", "EH", "T"),
    "lets": ("L", "EH", "T", "S"),
    "meet": ("M", "IY", "T"),
    "much": ("M", "AH", "CH"),
    "my": ("M", "AY"),
    "name": ("N", "EY", "M"),
    "nice": ("N", "AY", "S"),
    "so": ("S", "OW"),
    "teacher": ("T", "IY", "CH", "ER"),
    "thank": ("TH", "AE", "NG", "K"),
    "the": ("DH", "AH"),
    "to": ("T", "UW"),
    "today": ("T", "AH", "D", "EY"),
    "walk": ("W", "AO", "K"),
    "work": ("W", "ER", "K"),
    "you": ("Y", "UW"),
    "your": ("Y", "AO", "R"),
}

_DIGRAPHS = (
    ("tion", ("SH", "AH", "N")),
    ("ough", ("AH", "F")),
    ("eigh", ("EY",)),
    ("augh", ("AE", "F")),
    ("ch", ("CH",)),
    ("sh", ("SH",)),
    ("th", ("TH",)),
    ("ng", ("NG",)),
    ("ph", ("F",)),
    ("ck", ("K",)),
    ("qu", ("K", "W")),
    ("wh", ("W",)),
    ("kn", ("N",)),
    ("wr", ("R",)),
    ("ee", ("IY",)),
    ("ea", ("IY",)),
    ("oo", ("UW",)),
    ("ou", ("AW",)),
    ("ow", ("OW",)),
    ("ai", ("EY",)),
    ("ay", ("EY",)),
    ("oi", ("OY",)),
    ("oy", ("OY",)),
    ("oa", ("OW",)),
    ("au", ("AO",)),
    ("aw", ("AO",)),
    ("er", ("ER",)),
    ("ir", ("ER",)),
    ("ur", ("ER",)),
    ("ar", ("AA", "R")),
    ("or", ("AO", "R")),
)

_LETTER = {
    "a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F",
    "g": "G", "h": "HH", "i": "IH", "j": "JH", "k": "K", "l": "L",
    "m": "M", "n": "N", "o": "AA", "p": "P", "q": "K", "r": "R",
    "s": "S", "t": "T", "u": "AH", "v": "V", "w": "W", "x": "K",
    "y": "Y", "z": "Z",
}

_WORD_RE = re.compile(r"[^a-z]+")


def _rule_phones(word: str) -> list[str]:
    leftover = word
    out: list[str] = []
    while leftover:
        matched = False
        for spelling, phones in _DIGRAPHS:
            if leftover.startswith(spelling):
                out.extend(phones)
                leftover = leftover[len(spelling):]
                matched = True
                break
        if matched:
            continue
        phone = _LETTER.get(leftover[0])
        if phone:
            out.append(phone)
        leftover = leftover[1:]
    return out or ["AH"]


def word_phones(word: str) -> list[str]:
    key = _WORD_RE.sub("", word.lower())
    if not key:
        return []
    listed = _LEXICON.get(key)
    if listed:
        return list(listed)
    return _rule_phones(key)


def text_phones(text: str) -> list[str]:
    phones: list[str] = []
    for word in english_words(text):
        phones.extend(word_phones(word))
    return phones[:50]


def phone_ids(phones: list[str]) -> list[int]:
    return [PHONE_TO_ID.get(p, PHONE_TO_ID["AH"]) for p in phones]
