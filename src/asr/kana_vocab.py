"""Kana (hiragana) vocabulary for Japanese ASR CTC training."""

# fmt: off
KANA = [
    # Base hiragana (46)
    "あ", "い", "う", "え", "お",
    "か", "き", "く", "け", "こ",
    "さ", "し", "す", "せ", "そ",
    "た", "ち", "つ", "て", "と",
    "な", "に", "ぬ", "ね", "の",
    "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も",
    "や", "ゆ", "よ",
    "ら", "り", "る", "れ", "ろ",
    "わ", "を", "ん",
    # Dakuten (20)
    "が", "ぎ", "ぐ", "げ", "ご",
    "ざ", "じ", "ず", "ぜ", "ぞ",
    "だ", "ぢ", "づ", "で", "ど",
    "ば", "び", "ぶ", "べ", "ぼ",
    # Handakuten (5)
    "ぱ", "ぴ", "ぷ", "ぺ", "ぽ",
    # Small kana (10)
    "ぁ", "ぃ", "ぅ", "ぇ", "ぉ",
    "っ", "ゃ", "ゅ", "ょ", "ゎ",
    # Extended
    "ー",  # long vowel mark
]
# fmt: on

BLANK_TOKEN = "<blank>"
BLANK_IDX = 0


class KanaVocab:
    """CTC-compatible kana vocabulary for Japanese."""

    def __init__(self):
        self.kana = list(KANA)
        # Index 0 = CTC blank, then kana.
        self.stoi = {BLANK_TOKEN: BLANK_IDX}
        for i, k in enumerate(KANA):
            self.stoi[k] = i + 1
        self.itos = {v: k for k, v in self.stoi.items()}
        self.size = len(self.stoi)

    def encode(self, kana_str: str) -> list[int]:
        """Convert kana string (space-separated characters) to token indices.

        Input format: "こ ん に ち は"
        """
        if not kana_str.strip():
            return []
        tokens = kana_str.split()
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def decode(self, indices: list[int], collapse: bool = True) -> str:
        """Decode index sequence to kana string.

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
        return "".join(result)
