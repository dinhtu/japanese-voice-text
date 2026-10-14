"""English word pronunciations: CMUdict phones -> syllables, stress, IPA.

CMUdict (~126k words, US English) gives ARPAbet phones with a stress digit
on every vowel (1 primary, 2 secondary, 0 none). From that we build:

    syllables  -- phones split at vowels by maximal onset; a consonant
                  after a stressed short vowel closes that syllable
                  ("AN.na", "TIC.u"), as learner dictionaries show it
    ipa        -- Cambridge-style US IPA, "ˈ"/"ˌ" before stressed syllables
    spelling   -- the written word cut into the same number of chunks
                  ("tea·cher", "ar·tic·u·late"), used for display and to
                  split the word's audio span into syllables

Words CMUdict does not know (rare names, typos) fall back to spelling
rules: vowel groups are syllables, stress goes on the first one, and no
IPA is given (`source == "rule"`) rather than guessing one.
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
    source: str  # "cmudict" | "rule"
    # Primary-stress syllables of every CMUdict reading with this many
    # syllables: "record" is RE-cord (noun) or re-CORD (verb), and without
    # part-of-speech tagging either one is accepted.
    accepted_stress: tuple[int, ...] = ()

    @property
    def stress_index(self) -> int:
        """Index of the primary-stressed syllable (first one if several)."""
        for i, syl in enumerate(self.syllables):
            if syl.stress == 1:
                return i
        return 0


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


def _spelling_groups(letters: str) -> list[tuple[int, int]]:
    """(start, end) of each written vowel group that makes a syllable."""
    groups = []
    for m in re.finditer(r"[aeiou]+|(?<=[^aeiou])y(?![aeiou])", letters):
        if m.start() == 0 and m.group() == "y":
            continue
        groups.append((m.start(), m.end()))
    # Silent final "e" ("name", "late") -- but "-le" after a consonant
    # is its own syllable ("ta·ble").
    if (
        len(groups) > 1
        and groups[-1] == (len(letters) - 1, len(letters))
        and letters[-1] == "e"
        and not (len(letters) >= 3 and letters[-2] == "l" and letters[-3] not in _SPELL_VOWELS)
    ):
        groups.pop()
    # "-ed" / "-es" endings that do not add a syllable ("worked", "names").
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


def _split_spelling(word: str, n: int, closed: list[bool]) -> list[str]:
    """Cut `word` into `n` chunks; `closed[i]` says syllable i keeps a coda."""
    if n <= 1:
        return [word]
    letters = word.replace("'", "")
    groups = _spelling_groups(letters)
    if len(groups) == n:
        cuts = []
        for k in range(n - 1):
            left_end, right_start = groups[k][1], groups[k + 1][0]
            cluster = letters[left_end:right_start]
            cuts.append(left_end + _cut_cluster(cluster, closed[k]))
    else:
        # Spelling and sound disagree ("every" = 2 or 3): cut evenly.
        cuts = [round(len(letters) * (k + 1) / n) for k in range(n - 1)]
    chunks = []
    prev = 0
    for cut in cuts + [len(letters)]:
        chunks.append(letters[prev:cut])
        prev = cut
    # Put apostrophes back where they were ("don't" -> "don't").
    if "'" in word:
        out, pos = [], 0
        for chunk in chunks:
            piece = ""
            for ch in chunk:
                while pos < len(word) and word[pos] == "'":
                    piece += "'"
                    pos += 1
                piece += word[pos]
                pos += 1
            out.append(piece)
        out[-1] += word[pos:]
        chunks = out
    return chunks


def _rule_pronunciation(word: str) -> WordPronunciation:
    letters = word.replace("'", "")
    n = max(1, len(_spelling_groups(letters)))
    chunks = _split_spelling(word, n, [False] * n)
    return WordPronunciation(
        tuple(Syllable(c, None, 1 if i == 0 else 0) for i, c in enumerate(chunks)),
        "rule",
        (0,),
    )


def pronounce(word: str) -> WordPronunciation:
    """Pronunciation of one written word; syllable chunks keep its case."""
    pron = _pronounce(word.lower().replace("’", "'"))
    written = word.replace("’", "'")
    chunks, pos = [], 0
    for syl in pron.syllables:
        chunks.append(written[pos:pos + len(syl.text)])
        pos += len(syl.text)
    return WordPronunciation(
        tuple(Syllable(c, s.ipa, s.stress) for c, s in zip(chunks, pron.syllables)),
        pron.source,
        pron.accepted_stress,
    )


@lru_cache(maxsize=4096)
def _pronounce(key: str) -> WordPronunciation:
    entries = _cmudict().get(key) or _cmudict().get(key.replace("'", ""))
    if not entries:
        return _rule_pronunciation(key)
    sylls = syllabify(entries[0])
    closed = []
    for chunk, stress in sylls[:-1]:
        last = _split_phone(chunk[-1])[0]
        closed.append(last not in _VOWELS)
    chunks = _split_spelling(key, len(sylls), closed)
    accepted = []
    for entry in entries:
        variant = syllabify(entry)
        if len(variant) != len(sylls):
            continue
        idx = next((i for i, (_p, s) in enumerate(variant) if s == 1), 0)
        if idx not in accepted:
            accepted.append(idx)
    return WordPronunciation(
        tuple(
            Syllable(text, "".join(_phone_ipa(*_split_phone(p)) for p in phones), stress)
            for text, (phones, stress) in zip(chunks, sylls)
        ),
        "cmudict",
        tuple(accepted),
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
