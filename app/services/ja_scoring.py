"""Japanese read-aloud score, counted per mora (the "Từng âm tiết" grid).

The ring and the grid come from one mora alignment, so they always agree:

    score = 100 * (morae marked ok) / (target morae)  - insertion penalty

Why not the character CER of `scoring.score_pronunciation`:

- The grid is per mora, but CER divides by characters and also charges
  every inserted sound. On a 4-mora word one extra kana cost 25 points
  while no cell turned red, so がっぺい could show っ✓ next to 0/100.
- A yoon mora (しゃ) is two characters, so one misread mora cost two
  errors.
- The greedy CTC model often writes an extra kana for a breath, a release
  or the gap of っ. That is model noise, not a pronunciation error, so the
  first FREE_INSERTIONS extra morae are free and the rest cost a little.

Comparison rules:

- A long vowel written ー and the same long vowel spelled with a kana are
  one sound: がっぺい = がっぺー, とうきょう = とーきょー = とおきょお.
  The G2P (pykakasi / UniDic) spells readings with kana; the ASR can write
  ー. Both sides need the same preceding vowel for the match.
- Among the alignments with the fewest edits, the one with the most
  matched morae wins (ties go to the learner).
- っ is only kept ✓ when at least one neighbouring mora is ✓: a bare gap
  is weak evidence on its own.
"""

from __future__ import annotations

from app.services.mora_diff import split_mora_spans
from app.services.scoring import PronunciationError, ScoreResult, _feedback

# Extra morae the ASR may write before any are charged.
FREE_INSERTIONS = 2
# Points per extra mora beyond FREE_INSERTIONS, and the most they can cost.
INSERTION_PENALTY = 5
MAX_INSERTION_PENALTY = 25
# A reading over OVERLONG_RATIO times the target's morae, with charged
# insertions, scores at most OVERLONG_CAP (babbling cannot collect matches).
OVERLONG_RATIO = 2.0
OVERLONG_CAP = 50

# error.type for a っ that aligned but has no ✓ neighbour.
WEAK = "weak"

_ROWS = {
    "a": "あかさたなはまやらわがざだばぱぁゃゎ",
    "i": "いきしちにひみりぎじぢびぴぃ",
    "u": "うくすつぬふむゆるぐずづぶぷゔぅゅ",
    "e": "えけせてねへめれげぜでべぺぇ",
    "o": "おこそとのほもよろをごぞどぼぽぉょ",
}
_VOWEL = {kana: vowel for vowel, row in _ROWS.items() for kana in row}
# Kana that lengthen the vowel before them (ー lengthens any vowel).
_LONG = {"a": "あ", "i": "い", "u": "う", "e": "えい", "o": "おう"}


def _vowels(morae: list[str]) -> list[str | None]:
    """Vowel each mora ends on; ー carries the previous one, ん/っ none."""
    out: list[str | None] = []
    for mora in morae:
        if mora == "ー":
            out.append(out[-1] if out else None)
        else:
            out.append(_VOWEL.get(mora[-1]))
    return out


def _lengthens(mora: str, vowel: str | None) -> bool:
    return vowel is not None and (mora == "ー" or mora in _LONG[vowel])


def _same(t: str, h: str, t_prev: str | None, h_prev: str | None) -> bool:
    if t == h:
        return True
    return t_prev is not None and t_prev == h_prev and _lengthens(t, t_prev) and _lengthens(h, h_prev)


def _align(tm: list[str], hm: list[str]) -> list[tuple[str, int, int | None]]:
    """Mora edit ops in target order: (op, target_idx, hyp_idx), op in
    match/sub/del/ins. Cost is (edits, -matches): fewest edits first, then
    the most matched morae. For ins, target_idx is the next target mora."""
    n, m = len(tm), len(hm)
    tv, hv = _vowels(tm), _vowels(hm)
    same = [
        [_same(tm[i], hm[j], tv[i - 1] if i else None, hv[j - 1] if j else None) for j in range(m)]
        for i in range(n)
    ]
    inf = (n + m + 1, 0)
    d = [[inf] * (m + 1) for _ in range(n + 1)]
    d[0][0] = (0, 0)
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = inf
            if i and j:
                e, k = d[i - 1][j - 1]
                best = min(best, (e, k - 1) if same[i - 1][j - 1] else (e + 1, k))
            if i:
                e, k = d[i - 1][j]
                best = min(best, (e + 1, k))
            if j:
                e, k = d[i][j - 1]
                best = min(best, (e + 1, k))
            d[i][j] = best

    ops: list[tuple[str, int, int | None]] = []
    i, j = n, m
    while i or j:
        if i and j:
            e, k = d[i - 1][j - 1]
            if same[i - 1][j - 1] and d[i][j] == (e, k - 1):
                ops.append(("match", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
            if not same[i - 1][j - 1] and d[i][j] == (e + 1, k):
                ops.append(("sub", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i and d[i][j] == (d[i - 1][j][0] + 1, d[i - 1][j][1]):
            ops.append(("del", i - 1, None))
            i -= 1
            continue
        ops.append(("ins", i, j - 1))
        j -= 1
    ops.reverse()
    return ops


def score_japanese(target_hiragana: str, recognized_hiragana: str) -> ScoreResult:
    """Score a normalized hiragana reading against the target, per mora.

    `errors` are mora-level (target / recognized are mora texts, position is
    the mora's first character in `target_hiragana`; an ins sits before the
    next target mora). `mora_ok` is the grid, one flag per target mora.
    `distance` counts mora edits and `cer` is distance / target morae.

    Raises:
        ValueError: If the target reading is empty.
    """
    spans = split_mora_spans(target_hiragana)
    if not spans:
        raise ValueError("Target reading is empty; cannot score against it.")
    tm = [text for text, _s, _e in spans]
    hm = [text for text, _s, _e in split_mora_spans(recognized_hiragana)]
    starts = [s for _t, s, _e in spans] + [len(target_hiragana)]

    ops = _align(tm, hm)
    ok = [False] * len(tm)
    for op, i, _j in ops:
        if op == "match":
            ok[i] = True

    weak = set()
    for i, mora in enumerate(tm):
        neighbours = [k for k in (i - 1, i + 1) if 0 <= k < len(tm)]
        if mora == "っ" and ok[i] and neighbours and not any(ok[k] for k in neighbours):
            weak.add(i)
    for i in weak:
        ok[i] = False

    errors: list[PronunciationError] = []
    n_ins = 0
    for op, i, j in ops:
        if op == "sub":
            errors.append(PronunciationError("sub", tm[i], hm[j], starts[i]))
        elif op == "del":
            errors.append(PronunciationError("del", tm[i], "", starts[i]))
        elif op == "ins":
            n_ins += 1
            errors.append(PronunciationError("ins", "", hm[j], starts[i]))
        elif i in weak:
            errors.append(PronunciationError(WEAK, tm[i], hm[j], starts[i]))

    n = len(tm)
    score = 100.0 * sum(ok) / n
    charged = max(0, n_ins - FREE_INSERTIONS)
    score -= min(MAX_INSERTION_PENALTY, charged * INSERTION_PENALTY)
    if charged and len(hm) > OVERLONG_RATIO * n:
        score = min(score, OVERLONG_CAP)
    score = round(max(0.0, score))

    distance = len(errors)
    level, message = _feedback(score)
    return ScoreResult(
        score=score,
        cer=round(distance / n, 4),
        distance=distance,
        target_length=n,
        level=level,
        message=message,
        errors=errors,
        mora_ok=ok,
    )
