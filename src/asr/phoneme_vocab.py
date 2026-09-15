"""Phoneme vocabulary for Japanese ASR CTC training."""

# pyopenjtalk-native phone set (based on common Japanese OpenJTalk frontends).
# `pau` and `sil` are intentionally excluded from training targets.
PHONEMES = [
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
]

BLANK_TOKEN = "<blank>"
BLANK_IDX = 0


class PhonemeVocab:
    """CTC-compatible phoneme vocabulary for Japanese."""

    def __init__(self):
        self.phonemes = list(PHONEMES)
        # Index 0 = CTC blank
        self.stoi = {BLANK_TOKEN: BLANK_IDX}
        for i, ph in enumerate(PHONEMES):
            self.stoi[ph] = i + 1
        self.itos = {v: k for k, v in self.stoi.items()}
        self.size = len(self.stoi)

    def encode(self, phoneme_str: str) -> list[int]:
        """Convert space-separated phoneme string to list of token indices."""
        if not phoneme_str.strip():
            return []
        return [self.stoi[ph] for ph in phoneme_str.split() if ph in self.stoi]

    def decode(self, indices: list[int], collapse: bool = True) -> str:
        """Decode index sequence to phoneme string.

        With collapse=True, performs CTC greedy decoding:
        removes repeated tokens and blank tokens.
        """
        result = []
        prev = None
        for idx in indices:
            if collapse:
                if idx == BLANK_IDX:
                    prev = idx
                    continue
                if idx == prev:
                    continue
            token = self.itos.get(idx)
            if token and token != BLANK_TOKEN:
                result.append(token)
            prev = idx
        return " ".join(result)
