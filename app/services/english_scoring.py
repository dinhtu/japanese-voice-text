"""English read-aloud score: letter CER, but a misread word loses all its letters.

Plain letter CER (spaces stripped) gives partial credit to a word the
learner got wrong -- "articulate" heard as "a taerait" scored 30. English
spelling is not phonetic, so a few shared letters do not mean the word was
said. Instead, each target word claims a stretch of the heard letters
(spaces stripped, so it does not matter where wav2vec2 -- no LM -- put
them: "a napple", "to day", "thankyou" all match):

    word errors     = letter edit distance(word, its stretch)
    word similarity = max(1 - errors / len(word), sound similarity)
    similarity >= WORD_GATE -> the word keeps len(word) * similarity letters
                               (minus 1 per letter its stretch runs over)
    similarity <  WORD_GATE -> the word keeps 0 (it was not that word)
    heard letters no word claims (extra / filler words) cost 1 each,
    EDGE_EXTRA_COST before the first / after the last word
    score = round((kept - min(extra, MAX_EXTRA_SHARE * kept)) / target letters * 100)

The stretches are chosen to maximize that score. Picking per word, not
per letter, stops a dropped word from bleeding into its neighbour ("it is
a good day" read as "is a good day" must not fail "is").

wav2vec2 writes what it hears, so a word said right often comes back
spelled by ear: "fone", "nait", "skul", "kat". A word's similarity is
therefore the better of its letter match and its sound match
(_sound_similarity: spelling folded to rough sounds, near sounds and
vowels cheap), and a passing word keeps len(word) * similarity letters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache

from app.services.english_text import normalize_english, speech_tokens
from app.services.scoring import ScoreResult, _feedback, score_pronunciation

WORD_GATE = 0.65
# A word this close is shown as the target spells it ("kat" -> "cat"); a
# passing but rougher one ("cat" for "cats") is shown as heard.
SNAP_GATE = 0.8
# A short scrap heard before the first or after the last word is mostly
# breath, a click or noise the ASR spelled out ("uh", "TO HAPPEN" around
# "cap"): cheap. A longer one is a real extra word and costs in full.
EDGE_EXTRA_COST = 0.25
EDGE_NOISE = 4
# Extra letters never take more than this share of what the words earned,
# so words read right never add up to 0.
MAX_EXTRA_SHARE = 0.7
# Extra-letter pieces shorter than these are dropped from a re-spelled
# transcript (see snap_transcript); real extra words are kept.
EDGE_SCRAP = 4
SCRAP = 3
# Hesitation sounds the ASR may spell out; never part of a target word.
_FILLERS = frozenset({"uh", "um", "er", "ah", "eh", "hm", "hmm", "mm", "erm", "uhm"})

_VOWELS = frozenset("aeiou")
# Spellings of one sound, scanned left to right (C = ch, S = sh, J = j).
_DIGRAPHS = (
    ("tch", "C"), ("sch", "sk"), ("ch", "C"), ("sh", "S"), ("th", "t"),
    ("ph", "f"), ("wh", "w"), ("ck", "k"), ("qu", "kw"), ("gh", ""),
)
_SILENT_START = (("kn", "n"), ("wr", "r"), ("gn", "n"), ("ps", "s"))
# Sounds a learner -- or wav2vec2 -- easily swaps; they cost half an error.
_NEAR = frozenset(frozenset(p) for p in (
    "kg", "td", "pb", "fv", "bv", "wv", "sz", "ts", "lr", "nm",
    "sS", "SC", "CJ", "Jg", "Jd", "Ct", "ky",
))


def _sound_key(letters: str) -> str:
    """Rough sounds of a spelling: "phone" -> "fon", "night" -> "nit",
    "school" -> "skol", "fone" -> "fon". Only for comparing two
    spellings; not a pronunciation."""
    s = letters.replace("'", "")
    for prefix, sound in _SILENT_START:
        if s.startswith(prefix) and len(s) > 2:
            s = sound + s[2:]
            break
    if s.endswith("dge"):
        s = s[:-3] + "J"
    elif s.endswith("ge") and len(s) > 3:
        s = s[:-2] + "J"
    elif s.endswith("ce") and len(s) > 3:
        s = s[:-2] + "s"  # "nice", "place": soft c, silent e
    elif len(s) > 3 and s.endswith("e") and s[-2] not in _VOWELS:
        s = s[:-1]  # silent final e
    out: list[str] = []
    i = 0
    while i < len(s):
        if s.startswith("ch", i) and i + 2 < len(s) and s[i + 2] not in _VOWELS:
            out.append("k")  # "arachnoid", "chris", "technical"
            i += 2
            continue
        for spelling, sound in _DIGRAPHS:
            if s.startswith(spelling, i):
                out.append(sound)
                i += len(spelling)
                break
        else:
            ch = s[i]
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if ch == "c":
                out.append("s" if nxt in ("e", "i", "y") else "k")
            elif ch == "q":
                out.append("k")
            elif ch == "x":
                out.append("ks")
            elif ch == "z":
                out.append("s")
            elif ch == "y":
                out.append("y" if i == 0 else "i")
            else:
                out.append(ch)
            i += 1
    key: list[str] = []
    for ch in "".join(out):
        if key and (ch == key[-1] or (ch in _VOWELS and key[-1] in _VOWELS)):
            continue  # "rr" -> r, "ai" / "oo" -> one vowel
        key.append(ch)
    return "".join(key)


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a in _VOWELS and b in _VOWELS:
        return 0.25
    return 0.5 if frozenset((a, b)) in _NEAR else 1.0


def _indel_cost(ch: str) -> float:
    return 0.5 if ch in _VOWELS or ch == "h" else 1.0


@lru_cache(maxsize=65536)
def _sound_similarity(word: str, heard: str) -> float:
    """1 - weighted sound edit distance / sounds in `word`."""
    a, b = _sound_key(word), _sound_key(heard)
    if not a:
        return 0.0
    row = [0.0]
    for ch in b:
        row.append(row[-1] + _indel_cost(ch))
    for ca in a:
        new = [row[0] + _indel_cost(ca)]
        for j, cb in enumerate(b, start=1):
            new.append(min(
                row[j] + _indel_cost(ca),
                new[j - 1] + _indel_cost(cb),
                row[j - 1] + _sub_cost(ca, cb),
            ))
        row = new
    return max(0.0, 1.0 - row[-1] / len(a))


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


def _segment(
    words: list[str], hyp: str, heard: frozenset[int] = frozenset()
) -> tuple[list[WordMatch], float, list[tuple[int | None, int, str]]]:
    """Best stretch of `hyp` per word, the score it earns (kept - extra),
    and the transcript in order as (word index or None for extra, start in
    `hyp`, letters).

    Words in `heard` were confirmed from the audio (see snap_transcript):
    they always pass and only pick the stretch that fits them best."""
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
            # Spelling by ear can run long ("box" -> "boks"); longer
            # stretches fail and are no better than leaving letters extra.
            limit = min(m, j + size + size // 2 + 1)
            for k, errors in enumerate(_prefix_distances(letters, hyp[j:limit]), start=j):
                similarity = max(0.0, 1.0 - errors / size)
                if similarity < 1.0 and k > j:
                    similarity = max(similarity, _sound_similarity(letters, hyp[j:k]))
                # No word soaks up extra words for free: letters beyond its
                # own length cost what an extra letter costs.
                over = max(0, (k - j) - size)
                if w in heard:
                    gain = size + size * similarity / 2 - over
                elif similarity >= WORD_GATE:
                    gain = size * similarity - over
                else:
                    gain = -over
                relax(w + 1, k, base + gain, ("word", j, similarity))

    matches: list[WordMatch] = []
    pieces: list[tuple[int | None, int, str]] = []
    w, j = n, m
    while w > 0 or j > 0:
        move = back[w][j]
        if move[0] == "extra":
            if pieces and pieces[-1][0] is None:
                pieces[-1] = (None, j - 1, hyp[j - 1] + pieces[-1][2])
            else:
                pieces.append((None, j - 1, hyp[j - 1]))
            j -= 1
        else:
            _kind, start, similarity = move
            matches.append(WordMatch(words[w - 1], hyp[start:j], round(similarity, 4)))
            pieces.append((w - 1, start, hyp[start:j]))
            w, j = w - 1, start
    matches.reverse()
    pieces.reverse()
    return matches, best[n][m], pieces


def match_words(target_text: str, recognized_text: str) -> list[WordMatch]:
    """One WordMatch per target word; the per-word grid on the page."""
    return _segment(_target_words(target_text), _letters(recognized_text))[0]


@dataclass(frozen=True)
class ExtraPiece:
    """Heard letters no target word claimed (an extra / repeated word)."""

    after: int  # index of the target word it follows, -1 before the first
    letters: str
    edge: bool  # before the first or after the last word
    repeat: bool = False  # says a neighbouring target word again ("to to")
    optional: bool = False  # allowed though not written ("hundred and five")


_APOSTROPHES = re.compile(r"['’]")
# "a.m.", "U.S.": one word, as the target tokenizer reads them.
_DOTTED = re.compile(r"\b(?:[A-Za-z]\.){2,}")
# Number scales after which a spoken "and" is optional (205 = "two hundred and five").
_SCALES = frozenset({"hundred", "thousand", "million", "billion"})
_CLITICS = {"s": ("is", "has"), "m": ("am",), "re": ("are",), "ve": ("have",), "ll": ("will",), "d": ("would", "had")}
_NOT_HEADS = {"can": "can", "won": "will", "ain": "is", "shan": "shall"}
_CONTRACTION = re.compile(r"^(\W*)([A-Za-z]+)['’]([A-Za-z]+)(\W*)$")


def transcript_tokens(text: str) -> list[str]:
    """Words of a transcript for word-level alignment: apostrophes dropped
    BEFORE normalize_english (which turns them into spaces), so "don't"
    stays one word "dont" -- as the target's spoken letters are -- and
    dotted initialisms kept whole ("a.m." -> "am")."""
    text = _DOTTED.sub(lambda m: m.group().replace(".", ""), text or "")
    return normalize_english(_APOSTROPHES.sub("", text)).split()


def expand_contractions(text: str, target_text: str) -> str:
    """Write out each contraction in `text` the way the target has it, when
    the target has the full form right there ("It's" -> "It is" for a target
    "it is ..."); others stay as they are ("It's" for a target "it's")."""
    words = [w.replace("'", "") for w in _target_words(target_text)]
    pairs = set(zip(words, words[1:]))
    out = []
    for word in (text or "").split():
        m = _CONTRACTION.match(word)
        if m:
            head, clitic = m.group(2), m.group(3).lower()
            low = head.lower()
            if clitic == "t" and low.endswith("n"):
                options = [(_NOT_HEADS.get(low, low[:-1]), "not")]
            else:
                options = [(low, full) for full in _CLITICS.get(clitic, ())]
            for a, b in options:
                if (a, b) in pairs:
                    first = head if a == low else a
                    word = f"{m.group(1)}{first} {b}{m.group(4)}"
                    break
        out.append(word)
    return " ".join(out)


def word_similarity(word: str, heard: str) -> float:
    """Letter match or sound match, whichever is better (as in _segment)."""
    letters, heard = word.replace("'", ""), heard.replace("'", "")
    if not letters or not heard:
        return 0.0
    similarity = max(0.0, 1.0 - _prefix_distances(letters, heard)[-1] / len(letters))
    if similarity < 1.0:
        similarity = max(similarity, _sound_similarity(letters, heard))
    return similarity


def align_word_tokens(target_text: str, recognized_text: str) -> tuple[list[WordMatch], list[ExtraPiece]]:
    """Word-level alignment for WER, on the transcript's own word breaks.

    The letter segmentation above ignores spaces, which is right for the
    score but lets a short word borrow a neighbour's letters: "I would like
    glass" gives "a" the "e" of "like", so the skipped article never shows.
    Here each target word takes whole heard tokens -- one, two or three
    joined ("to day" -> today) or half of one ("thankyou" -> thank, you) --
    or none (omitted); a token no word takes is an ExtraPiece (inserted, or
    a repeat of the word next to it)."""
    words = _target_words(target_text)
    tokens = transcript_tokens(recognized_text)
    n, m = len(words), len(tokens)
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    def pair_cost(similarity: float) -> float:
        # One edit if the word was not said; the tiny term picks the closer pairing.
        return (0.0 if similarity >= WORD_GATE else 1.0) + 0.01 * (1.0 - similarity)

    for i in range(n + 1):
        for j in range(m + 1):
            base = cost[i][j]
            if base == inf:
                continue
            steps = []
            if i < n:
                steps.append((i + 1, j, 1.0, ("del",)))
            if j < m:
                steps.append((i, j + 1, 1.0, ("ins",)))
            # A filler never stands in for a word: "like um glass" skipped "a".
            if i < n and j < m and tokens[j] not in _FILLERS:
                sim = word_similarity(words[i], tokens[j])
                steps.append((i + 1, j + 1, pair_cost(sim), ("word", tokens[j], sim)))
            # Joining ("to day", "p i n") / splitting ("thankyou") only for a
            # close match that no single token gives: otherwise an inserted
            # or repeated word ("a water", "you you") would vanish into it.
            for k in (2, 3):
                group = tokens[j:j + k]
                if i >= n or len(group) < k or set(group) & _FILLERS:
                    continue
                joined = "".join(group)
                # Spelling the word exactly ("can not", "any time", "in to")
                # is always fine.
                exact = joined == words[i].replace("'", "")
                if not exact and any(word_similarity(words[i], t) >= WORD_GATE for t in group):
                    continue
                sim = 1.0 if exact else word_similarity(words[i], joined)
                if sim >= SNAP_GATE:
                    steps.append((i + 1, j + k, pair_cost(sim) + 0.005 * k, ("word", joined, sim)))
            if i + 1 < n and j < m:
                first = words[i].replace("'", "")
                second = words[i + 1].replace("'", "")
                token = tokens[j]
                if len(token) > len(first):
                    a, b = token[:len(first)], token[len(first):]
                    sa, sb = word_similarity(first, a), word_similarity(second, b)
                    # Its first half must fit better than the whole token.
                    if sa >= SNAP_GATE and sb >= SNAP_GATE and sa > word_similarity(first, token):
                        steps.append((i + 2, j + 1, pair_cost(sa) + pair_cost(sb) + 0.005,
                                      ("split", a, sa, b, sb)))
            for ni, nj, step, move in steps:
                if base + step < cost[ni][nj] - 1e-12:
                    cost[ni][nj] = base + step
                    back[ni][nj] = (i, j, move)

    path = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj, move = back[i][j]
        path.append((pi, pj, move))
        i, j = pi, pj
    path.reverse()

    matches: list[WordMatch] = []
    extras: list[ExtraPiece] = []
    for pi, pj, move in path:
        if move[0] == "del":
            matches.append(WordMatch(words[pi], "", 0.0))
        elif move[0] == "ins":
            # `pi` words are already placed: the token follows word pi - 1.
            token = tokens[pj]
            repeat = any(
                0 <= w < n and word_similarity(words[w], token) >= SNAP_GATE for w in (pi - 1, pi)
            )
            optional = token == "and" and 0 < pi < n and words[pi - 1] in _SCALES
            extras.append(ExtraPiece(pi - 1, token, pi == 0 or pi == n, repeat and not optional, optional))
        elif move[0] == "word":
            matches.append(WordMatch(words[pi], move[1], round(move[2], 4)))
        else:
            matches.append(WordMatch(words[pi], move[1], round(move[2], 4)))
            matches.append(WordMatch(words[pi + 1], move[3], round(move[4], 4)))
    return matches, extras


def snap_transcript(target_text: str, recognized_text: str, heard: frozenset[int]) -> str:
    """The transcript with every word in `heard` (indices of the target's
    spoken words that the audio itself confirms, see english_alignment
    HEARD_LOGPROB) written as the target spells it: "SIRBERHAKNOY HEMORICH"
    -> "subarachnoid hemorrhage". Other words and extra letters stay as
    heard, in lower case, except the scraps a re-spelled word leaves
    behind ("cap pu", "w weekend o"): extra pieces shorter than
    EDGE_SCRAP letters at either end, or SCRAP letters inside."""
    heard_norm = normalize_english(recognized_text)
    if not heard:
        return heard_norm
    # Where the ASR put its word breaks, as offsets into the joined letters.
    breaks, offset = set(), 0
    for token in heard_norm.split(" "):
        breaks.add(offset)
        offset += len(token)

    def as_heard(start: int, letters: str) -> str:
        return "".join(
            (" " + ch) if k and start + k in breaks else ch for k, ch in enumerate(letters)
        )

    words = _target_words(target_text)
    pieces = _segment(words, heard_norm.replace(" ", ""), heard)[2]
    out = []
    for k, (index, start, letters) in enumerate(pieces):
        if index is not None:
            out.append(words[index] if index in heard else as_heard(start, letters))
            continue
        edge = k == 0 or k == len(pieces) - 1
        if len(letters) >= (EDGE_SCRAP if edge else SCRAP):
            out.append(as_heard(start, letters))
    return " ".join(t for t in out if t)


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
    matches, earned, pieces = _segment(words, hyp)
    kept = sum(len(m.target.replace("'", "")) * m.similarity for m in matches if m.ok)
    # Edge extras are discounted only after segmenting at full price, so
    # the segmentation never pushes a real extra word ("i DONT like it")
    # to the front to make it cheap.
    edge = sum(
        len(letters) for k, (index, _start, letters) in enumerate(pieces)
        if index is None and k in (0, len(pieces) - 1) and len(letters) <= EDGE_NOISE
    )
    penalty = max(0.0, kept - earned) - (1.0 - EDGE_EXTRA_COST) * edge
    penalty = min(max(0.0, penalty), MAX_EXTRA_SHARE * kept)
    score = round(max(0.0, kept - penalty) / len(ref) * 100)
    level, message = _feedback(score)
    return replace(letters, score=score, level=level, message=message), matches
