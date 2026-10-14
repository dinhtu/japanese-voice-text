"""English read-aloud score: letter CER, but a misread word loses all its letters.

Plain letter CER (spaces stripped) gives partial credit to a word the
learner got wrong -- "articulate" heard as "a taerait" scored 30. English
spelling is not phonetic, so a few shared letters do not mean the word was
said. Instead, each target word claims a stretch of the heard letters
(spaces stripped, so it does not matter where wav2vec2 -- no LM -- put
them: "a napple", "to day", "thankyou" all match):

    word errors     = letter edit distance(word, its stretch)
    word similarity = 1 - errors / len(word)
    similarity >= WORD_GATE -> the word keeps len(word) - errors letters
    similarity <  WORD_GATE -> the word keeps 0 (it was not that word)
    heard letters no word claims (extra / filler words) cost 1 each
    score = round(max(0, kept - extra) / target letters * 100)

The stretches are chosen to maximize that score. Picking per word, not
per letter, stops a dropped word from bleeding into its neighbour ("it is
a good day" read as "is a good day" must not fail "is"). When every word
passes the gate the score is exactly the old 1 - CER.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.services.english_text import normalize_english, speech_tokens
from app.services.scoring import ScoreResult, _feedback, score_pronunciation

WORD_GATE = 0.75


@dataclass(frozen=True)
class WordMatch:
    target: str  # target word as written, e.g. "don't"
    heard: str  # heard letters this word claimed ("" when nothing was)
    similarity: float  # 1 - errors / letters, before the gate

    @property
    def ok(self) -> bool:
        return self.similarity >= WORD_GATE


def _target_words(text: str) -> list[str]:
    """Words as spoken ("3.50" -> three point five zero, "Mr." -> mister),
    from the same tokenizer as the intonation chart so both line up."""
    return [t.spoken for t in speech_tokens(text)[1]]


def _letters(text: str) -> str:
    return normalize_english(text).replace(" ", "")


def _prefix_distances(word: str, heard: str) -> list[int]:
    """Edit distance from `word` to heard[:t] for every t (one DP sweep)."""
    row = list(range(len(word) + 1))
    out = [row[-1]]
    for t, ch in enumerate(heard, start=1):
        new = [t]
        for i, wc in enumerate(word, start=1):
            new.append(min(row[i] + 1, new[i - 1] + 1, row[i - 1] + (wc != ch)))
        row = new
        out.append(row[-1])
    return out


def _segment(words: list[str], hyp: str) -> tuple[list[WordMatch], float]:
    """Best stretch of `hyp` per word, and the score it earns (kept - extra)."""
    n, m = len(words), len(hyp)
    neg = float("-inf")
    best = [[neg] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0

    def relax(w: int, j: int, value: float, move: tuple) -> None:
        if value > best[w][j] + 1e-9:
            best[w][j] = value
            back[w][j] = move

    for w in range(n + 1):
        for j in range(m + 1):
            base = best[w][j]
            if base == neg:
                continue
            if j < m:  # a heard letter no word claims
                relax(w, j + 1, base - 1, ("extra",))
            if w == n:
                continue
            letters = words[w].replace("'", "")
            size = len(letters)
            # A passing stretch is at most size + size/4 letters; longer
            # ones fail and are no better than leaving letters as extra.
            limit = min(m, j + size + size // 4 + 1)
            for k, errors in enumerate(_prefix_distances(letters, hyp[j:limit]), start=j):
                similarity = max(0.0, 1.0 - errors / size)
                if similarity >= WORD_GATE:
                    gain = size - errors
                else:
                    # A failed word cannot soak up extra letters for free.
                    gain = -max(0, (k - j) - size)
                relax(w + 1, k, base + gain, ("word", j, similarity))

    matches: list[WordMatch] = []
    w, j = n, m
    while w > 0 or j > 0:
        move = back[w][j]
        if move[0] == "extra":
            j -= 1
        else:
            _kind, start, similarity = move
            matches.append(WordMatch(words[w - 1], hyp[start:j], round(similarity, 4)))
            w, j = w - 1, start
    matches.reverse()
    return matches, best[n][m]


def match_words(target_text: str, recognized_text: str) -> list[WordMatch]:
    """One WordMatch per target word; the per-word grid on the page."""
    return _segment(_target_words(target_text), _letters(recognized_text))[0]


def score_english(target_text: str, recognized_text: str) -> tuple[ScoreResult, list[WordMatch]]:
    """Word-gated score. `cer`, `distance` and `errors` stay plain letter
    CER so the CER metric and the letter diff list are unchanged; only
    `score` / `level` / `message` change.
    """
    words = _target_words(target_text)
    ref = "".join(w.replace("'", "") for w in words)
    if not ref:
        raise ValueError("Target text has no English words; cannot score against it.")
    hyp = _letters(recognized_text)
    letters = score_pronunciation(ref, hyp)
    matches, earned = _segment(words, hyp)
    score = round(max(0.0, earned) / len(ref) * 100)
    level, message = _feedback(score)
    return replace(letters, score=score, level=level, message=message), matches
