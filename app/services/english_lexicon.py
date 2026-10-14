"""English word pronunciations: CMUdict phones -> syllables, stress, IPA.

CMUdict (~126k words, US English) gives ARPAbet phones with a stress digit
on every vowel (1 primary, 2 secondary, 0 none). From that we build:

    syllables  -- phones split at vowels by maximal onset; a consonant
                  after a stressed short vowel closes that syllable
                  ("AN.na", "TIC.u"), as learner dictionaries show it
    ipa        -- Cambridge-style US IPA, "ˈ"/"ˌ" before stressed syllables
    spelling   -- the written word cut at the letter where each syllable's
                  first phone is spelled ("tea·cher", "beau·ti·ful"), found
                  by aligning phones to letters; used for display and to
                  split the word's audio span into syllables

Initialisms ("U.S.", "a.m.") are read letter by letter. Words CMUdict does
not know (rare names, typos) fall back to spelling rules: vowel groups are
syllables, stress goes on the first one, and no IPA is given
(`source == "rule"`) rather than guessing one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}
# Short ("checked") vowels cannot end a stressed syllable in English, so a
# following consonant is pulled into the coda: "AN.na", not "A.nna".
_LAX = {"AE", "EH", "IH", "UH", "AH"}

# Legal word-initial clusters (ARPAbet), longest first when matched.
_ONSETS = {
    ("P", "R"), ("P", "L"), ("B", "R"), ("B", "L"), ("T", "R"), ("D", "R"),
    ("K", "R"), ("K", "L"), ("G", "R"), ("G", "L"), ("F", "R"), ("F", "L"),
    ("TH", "R"), ("SH", "R"), ("S", "P"), ("S", "T"), ("S", "K"), ("S", "M"),
    ("S", "N"), ("S", "L"), ("S", "W"), ("K", "W"), ("T", "W"), ("D", "W"),
    ("G", "W"), ("P", "Y"), ("B", "Y"), ("K", "Y"), ("F", "Y"), ("M", "Y"),
    ("V", "Y"), ("HH", "Y"), ("S", "P", "R"), ("S", "T", "R"), ("S", "K", "R"),
    ("S", "P", "L"), ("S", "K", "W"), ("S", "K", "Y"), ("S", "P", "Y"),
}

_IPA = {
    "AA": "ɑː", "AE": "æ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "EH": "e",
    "EY": "eɪ", "IH": "ɪ", "IY": "iː", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ",
    "UW": "uː",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p",
    "R": "r", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}

# How each phone is commonly spelled, for aligning phones to letters.
_SPELLINGS: dict[str, tuple[str, ...]] = {
    "AA": ("a", "o", "ah", "au", "aw", "al", "ea"),
    "AE": ("a", "ai", "au"),
    "AH": ("a", "e", "i", "o", "u", "ou", "oo", "io", "ai", "y", "eo"),
    "AO": ("o", "a", "au", "aw", "ou", "oa", "al", "ough", "augh"),
    "AW": ("ou", "ow", "au"),
    "AY": ("i", "y", "ie", "igh", "ai", "ei", "uy", "eye", "ye", "is"),
    "EH": ("e", "ea", "a", "ai", "ie", "ay", "ue"),
    "ER": ("er", "ir", "ur", "or", "ar", "ear", "our", "re", "yr", "r", "ere"),
    "EY": ("a", "ai", "ay", "ei", "ey", "ea", "eigh", "e", "ae"),
    "IH": ("i", "y", "e", "ui", "ee", "ie", "u", "a"),
    "IY": ("e", "ee", "ea", "y", "i", "ie", "ei", "ey", "eo", "ae", "oe"),
    "OW": ("o", "oa", "ow", "oe", "ou", "ough", "eau", "ew"),
    "OY": ("oi", "oy"),
    "UH": ("oo", "u", "ou", "o"),
    "UW": ("u", "oo", "ew", "ue", "ou", "o", "ui", "eu", "eau", "wo", "oe"),
    "B": ("b", "bb"), "CH": ("ch", "tch", "t", "c", "ti"),
    "D": ("d", "dd", "ed"), "DH": ("th",), "F": ("f", "ff", "ph", "gh"),
    "G": ("g", "gg", "gh", "gu", "x"), "HH": ("h", "wh"),
    "JH": ("j", "g", "ge", "dg", "dge", "gi", "d"),
    "K": ("k", "c", "ck", "ch", "q", "cc", "x", "qu", "lk"),
    "L": ("l", "ll", "le"), "M": ("m", "mm", "mb", "mn"),
    "N": ("n", "nn", "kn", "gn", "pn"), "NG": ("ng", "n"), "P": ("p", "pp"),
    "R": ("r", "rr", "wr", "rh"),
    "S": ("s", "ss", "c", "ce", "sc", "x", "ps", "z"),
    "SH": ("sh", "ti", "ci", "ss", "s", "ch", "si", "sci", "c"),
    "T": ("t", "tt", "ed", "th", "pt"), "TH": ("th",), "V": ("v", "f"),
    "W": ("w", "u", "wh", "o"), "Y": ("y", "i", "j", "u", "e"),
    "Z": ("z", "s", "zz", "ss", "x", "se"), "ZH": ("s", "g", "si", "z", "ge"),
}
_SILENT = set("ehwgkbu'")

_SPELL_VOWELS = set("aeiou")
# Kept together when cutting the spelling between syllables.
_DIGRAPHS = ("tch", "ch", "sh", "th", "ph", "wh", "ck", "gh", "qu")
_SPELL_ONSETS = {
    "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "tr",
    "sc", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tw", "thr", "shr",
    "str", "scr", "spr", "spl", "squ",
}


@dataclass(frozen=True)
class Syllable:
    text: str  # spelling chunk, e.g. "tea"
    ipa: str | None  # e.g. "tiː"; None when the word is not in CMUdict
    stress: int  # 1 primary, 2 secondary, 0 none


@dataclass(frozen=True)
class WordPronunciation:
    syllables: tuple[Syllable, ...]
    source: str  # "cmudict" | "rule" | "letters"
    # Primary-stress syllables of every CMUdict reading with this many
    # syllables: "record" is RE-cord (noun) or re-CORD (verb), and without
    # part-of-speech tagging either one is accepted.
    accepted_stress: tuple[int, ...] = ()

    @property
    def stress_index(self) -> int:
        """Index of the primary-stressed syllable. With several primaries
        ("TV", "OK", "FBI") the last one: initialisms and compounds take
        end stress."""
        primaries = [i for i, syl in enumerate(self.syllables) if syl.stress == 1]
        return primaries[-1] if primaries else 0


@lru_cache(maxsize=1)
def _cmudict() -> dict[str, list[list[str]]]:
    try:
        import cmudict

        return cmudict.dict()
    except Exception:  # noqa: BLE001
        logger.warning("cmudict unavailable; English stress falls back to spelling rules", exc_info=True)
        return {}


def _split_phone(phone: str) -> tuple[str, int | None]:
    if phone[-1].isdigit():
        return phone[:-1], int(phone[-1])
    return phone, None


def _phone_ipa(base: str, stress: int | None) -> str:
    if base == "AH":
        return "ʌ" if stress else "ə"
    if base == "ER":
        return "ɝː" if stress else "ɚ"
    return _IPA.get(base, base.lower())


def syllabify(phones: list[str]) -> list[tuple[list[str], int]]:
    """Split CMU phones into (phones, stress) syllables."""
    nuclei = [i for i, p in enumerate(phones) if _split_phone(p)[0] in _VOWELS]
    if not nuclei:
        return [(list(phones), 0)]
    # Boundary before each nucleus except the first.
    starts = [0]
    for left, right in zip(nuclei, nuclei[1:]):
        cluster = [_split_phone(p)[0] for p in phones[left + 1:right]]
        onset = 0
        for size in range(min(3, len(cluster)), 0, -1):
            tail = tuple(cluster[-size:])
            if size == 1 and tail[0] != "NG" or tail in _ONSETS:
                onset = size
                break
        base, stress = _split_phone(phones[left])
        if onset and onset == len(cluster) and stress and base in _LAX:
            # Stressed short vowel needs a coda: take one consonant back.
            if _split_phone(phones[right])[1] == 0 or onset > 1:
                onset -= 1
        starts.append(right - onset)
    out = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(phones)
        chunk = phones[start:end]
        stress = next(
            (s for p in chunk for b, s in [_split_phone(p)] if b in _VOWELS and s is not None),
            0,
        )
        out.append((chunk, stress))
    return out


# --------------------------------------------------- spelling <-> syllables


def _align_phones(phones: list[str], letters: str) -> list[tuple[int, int]] | None:
    """(start, end) letter span of every phone, by a small DP: a phone eats
    one of its common spellings for free, or nothing / an odd letter at a
    cost; silent letters (final e, gh, ...) are cheap to skip."""
    bases = [_split_phone(p)[0] for p in phones]
    n, m = len(letters), len(bases)
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            here = cost[i][j]
            if here == inf:
                continue
            if i < n:  # a letter no phone uses
                skip = 0.4 if letters[i] in _SILENT else 1.0
                if here + skip < cost[i + 1][j]:
                    cost[i + 1][j], back[i + 1][j] = here + skip, ("skip",)
            if j < m:
                for spelling in _SPELLINGS.get(bases[j], ()):
                    if letters.startswith(spelling, i):
                        k = i + len(spelling)
                        # Prefer the longer spelling on ties ("ch" over "c").
                        step = here - 0.01 * len(spelling)
                        if step < cost[k][j + 1]:
                            cost[k][j + 1], back[k][j + 1] = step, ("phone", i)
                if i < n and here + 1.2 < cost[i + 1][j + 1]:  # odd spelling
                    cost[i + 1][j + 1], back[i + 1][j + 1] = here + 1.2, ("phone", i)
                if here + 1.0 < cost[i][j + 1]:  # phone with no letter (x = K S)
                    cost[i][j + 1], back[i][j + 1] = here + 1.0, ("phone", i)
    if cost[n][m] == inf:
        return None
    spans: list[tuple[int, int]] = [(0, 0)] * m
    i, j = n, m
    while j > 0 or i > 0:
        move = back[i][j]
        if move is None:
            return None
        if move[0] == "skip":
            i -= 1
        else:
            spans[j - 1] = (move[1], i)
            i, j = move[1], j - 1
    return spans


_DOUBLE = re.compile(r"([b-df-hj-np-tv-z])\1")


def _phone_cuts(sylls: list[tuple[list[str], int]], letters: str) -> list[int] | None:
    """Letter index where each syllable after the first begins."""
    phones = [p for chunk, _s in sylls for p in chunk]
    spans = _align_phones(phones, letters)
    if spans is None:
        return None
    cuts, pos = [], 0
    for chunk, _s in sylls[:-1]:
        pos += len(chunk)
        cut = spans[pos][0]
        # A doubled consonant splits across the boundary: "An·na", "hap·py",
        # whichever side the phone went to ("nec·es·sar·y").
        prev_start, prev_end = spans[pos - 1]
        next_start, next_end = spans[pos]
        if prev_end == cut and prev_end - prev_start == 2 and _DOUBLE.fullmatch(letters[prev_start:prev_end]):
            cut -= 1
        elif next_start == cut and next_end - next_start == 2 and _DOUBLE.fullmatch(letters[next_start:next_end]):
            cut += 1
        cuts.append(cut)
    if any(not (0 < c < len(letters)) for c in cuts) or any(b <= a for a, b in zip(cuts, cuts[1:])):
        return None
    return cuts


def _spelling_groups(letters: str) -> list[tuple[int, int]]:
    """(start, end) of each written vowel group that makes a syllable."""
    groups = []
    for m in re.finditer(r"[aeiou]+|(?<=[^aeiou])y(?![aeiou])", letters):
        if m.start() == 0 and m.group() == "y":
            continue
        groups.append((m.start(), m.end()))
    if len(groups) < 2:
        return groups
    # Silent final "e" ("name", "late") -- but "-le" after a consonant
    # is its own syllable ("ta·ble").
    if (
        groups[-1] == (len(letters) - 1, len(letters))
        and letters[-1] == "e"
        and not (len(letters) >= 3 and letters[-2] == "l" and letters[-3] not in _SPELL_VOWELS)
    ):
        groups.pop()
    # "-ed" / "-es" that add no syllable: "worked", "named", "names", "likes"
    # (but "wanted", "needed", "boxes", "wishes" keep theirs).
    elif groups[-1] == (len(letters) - 2, len(letters) - 1) and len(letters) >= 4:
        end, before = letters[-2:], letters[-3]
        if end == "ed" and before not in "td":
            groups.pop()
        elif end == "es" and before not in "szx" and letters[-4:-2] not in ("ch", "sh", "ge", "ce"):
            groups.pop()
    return groups


def _cut_cluster(cluster: str, prev_closed: bool) -> int:
    """How many letters of the consonant cluster stay with the left syllable."""
    if not cluster:
        return 0
    units: list[str] = []
    i = 0
    while i < len(cluster):
        for dg in _DIGRAPHS:
            if cluster.startswith(dg, i):
                units.append(dg)
                i += len(dg)
                break
        else:
            units.append(cluster[i])
            i += 1
    if len(units) == 1:
        return len(units[0]) if prev_closed else 0
    # Longest legal onset on the right; a closed syllable keeps >= 1 unit.
    for take in range(len(units), 0, -1):
        if take == len(units) and prev_closed:
            continue
        right = "".join(units[-take:])
        if take == 1 or right in _SPELL_ONSETS:
            return len(cluster) - len(right)
    return len(units[0])


def _vowel_group_cuts(letters: str, n: int, closed: list[bool]) -> list[int] | None:
    groups = _spelling_groups(letters)
    if len(groups) != n:
        return None
    cuts = []
    for k in range(n - 1):
        left_end, right_start = groups[k][1], groups[k + 1][0]
        cuts.append(left_end + _cut_cluster(letters[left_end:right_start], closed[k]))
    return cuts


def _chunks(word: str, cuts: list[int]) -> list[str]:
    """Cut `word` (apostrophes included) at letter indices of its letters."""
    out, letter, piece = [], 0, ""
    targets = list(cuts)
    for ch in word:
        if ch != "'" and targets and letter == targets[0]:
            out.append(piece)
            piece = ""
            targets.pop(0)
        piece += ch
        if ch != "'":
            letter += 1
    out.append(piece)
    return out


def _rule_pronunciation(word: str) -> WordPronunciation:
    letters = word.replace("'", "")
    n = max(1, len(_spelling_groups(letters)))
    cuts = _vowel_group_cuts(letters, n, [False] * n) if n > 1 else []
    chunks = _chunks(word, cuts or [])
    return WordPronunciation(
        tuple(Syllable(c, None, 1 if i == 0 else 0) for i, c in enumerate(chunks)),
        "rule",
        (0,),
    )


def _single_unit(word: str, sylls: list[tuple[list[str], int]], source: str) -> WordPronunciation:
    """A word whose letters cannot be split into its syllables ("W" is
    three syllables) is shown whole, carrying the primary stress."""
    ipa = ".".join("".join(_phone_ipa(*_split_phone(p)) for p in chunk) for chunk, _s in sylls)
    return WordPronunciation((Syllable(word, ipa, 1),), source, (0,))


def _from_phones(word: str, entries: list[list[str]], source: str) -> WordPronunciation:
    sylls = syllabify(entries[0])
    letters = word.replace("'", "")
    if len(sylls) > 1:
        closed = [_split_phone(chunk[-1])[0] not in _VOWELS for chunk, _s in sylls[:-1]]
        cuts = _phone_cuts(sylls, letters) or _vowel_group_cuts(letters, len(sylls), closed)
        if cuts is None:
            if len(letters) < len(sylls):
                return _single_unit(word, sylls, source)
            cuts = [round(len(letters) * (k + 1) / len(sylls)) for k in range(len(sylls) - 1)]
        chunks = _chunks(word, cuts)
        if any(not c.replace("'", "") for c in chunks):
            return _single_unit(word, sylls, source)
    else:
        chunks = [word]
    accepted: list[int] = []
    for entry in entries:
        variant = syllabify(entry)
        if len(variant) != len(sylls):
            continue
        for i, (_p, s) in enumerate(variant):
            if s == 1 and i not in accepted:
                accepted.append(i)
    return WordPronunciation(
        tuple(
            Syllable(text, "".join(_phone_ipa(*_split_phone(p)) for p in phones), stress)
            for text, (phones, stress) in zip(chunks, sylls)
        ),
        source,
        tuple(accepted) or (0,),
    )


def _letter_name(letter: str) -> list[str] | None:
    """CMUdict phones for the NAME of a letter ("u" -> Y UW1, "a" -> EY1)."""
    for entry in _cmudict().get(letter, []):
        if any(p.endswith("1") for p in entry):
            return entry
    return None


@lru_cache(maxsize=4096)
def _pronounce(key: str, initialism: bool) -> WordPronunciation:
    letters = key.replace("'", "")
    if initialism:
        names = [_letter_name(ch) for ch in letters]
        if all(names):
            # One syllable group per letter; end stress ("U.S." = you-ESS).
            sylls = []
            for i, phones in enumerate(names):
                stress = 1 if i == len(names) - 1 else 2
                sylls.append(Syllable(
                    letters[i],
                    "".join(_phone_ipa(*_split_phone(p)) for p in phones),
                    stress,
                ))
            return WordPronunciation(tuple(sylls), "letters", (len(sylls) - 1,))
    entries = _cmudict().get(key) or _cmudict().get(letters)
    if not entries:
        return _rule_pronunciation(key)
    return _from_phones(key, entries, "cmudict")


def pronounce(word: str, spoken: str | None = None, initialism: bool = False) -> WordPronunciation:
    """Pronunciation of one word. Lookup uses `spoken` (lowercase a-z,
    accents folded) when given; syllable chunks keep the written case."""
    written = word.replace("’", "'").replace(".", "") if initialism else word.replace("’", "'")
    key = spoken if spoken is not None else written.lower()
    pron = _pronounce(key, initialism)
    if len(written) != len(key):
        return pron  # e.g. ligatures folded away: show the spoken form
    chunks, pos = [], 0
    for syl in pron.syllables:
        chunks.append(written[pos:pos + len(syl.text)])
        pos += len(syl.text)
    return WordPronunciation(
        tuple(Syllable(c, s.ipa, s.stress) for c, s in zip(chunks, pron.syllables)),
        pron.source,
        pron.accepted_stress,
    )


def word_ipa(pron: WordPronunciation, mark_stress: bool = True) -> str | None:
    """"ˈæn.ə" style IPA; a monosyllable only gets "ˈ" when `mark_stress`."""
    if any(s.ipa is None for s in pron.syllables):
        return None
    if len(pron.syllables) == 1:
        syl = pron.syllables[0]
        return ("ˈ" if mark_stress and syl.stress else "") + syl.ipa
    out = ""
    for i, syl in enumerate(pron.syllables):
        mark = {1: "ˈ", 2: "ˌ"}.get(syl.stress, "." if i else "")
        out += mark + syl.ipa
    return out
