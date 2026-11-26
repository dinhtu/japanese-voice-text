"""Korean text normalization, phonology (G2P), and pitch-accent modeling.

Provides:
- Hangul decomposition/composition
- Standard Korean phonetic realization rules (연음, 비음화, 유음화, 격음화, 경음화, ㅎ탈락)
- Pitch contour modeling (Seoul Korean Accentual Phrase H/L pattern)
- Syllable/phoneme-level error scoring
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.services.scoring import PronunciationError, ScoreResult, _feedback
from src.asr.metrics import edit_distance, edit_ops

_HANGUL = re.compile(r"[가-힣]")

CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]

JONGSEONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# High-onset consonants in Seoul Korean Accentual Phrase (AP)
_HIGH_ONSETS = {"ㅋ", "ㅌ", "ㅍ", "ㅊ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ", "ㅅ", "ㅎ"}


def is_hangul_syllable(char: str) -> bool:
    return bool(char and len(char) == 1 and 0xAC00 <= ord(char) <= 0xD7A3)


def decompose_hangul(char: str) -> tuple[str, str, str]:
    """Decompose a modern Hangul syllable into (choseong, jungseong, jongseong)."""
    if not is_hangul_syllable(char):
        return ("", "", "")
    idx = ord(char) - 0xAC00
    cho = CHOSEONG[idx // 588]
    jung = JUNGSEONG[(idx % 588) // 28]
    jong = JONGSEONG[idx % 28]
    return (cho, jung, jong)


def compose_hangul(cho: str, jung: str, jong: str = "") -> str:
    """Compose (choseong, jungseong, jongseong) into a Hangul syllable."""
    if cho in CHOSEONG and jung in JUNGSEONG and jong in JONGSEONG:
        code = 0xAC00 + CHOSEONG.index(cho) * 588 + JUNGSEONG.index(jung) * 28 + JONGSEONG.index(jong)
        return chr(code)
    return cho + jung + jong


def normalize_korean(text: str) -> str:
    """Compose Hangul jamo and keep modern Hangul syllables only."""
    return "".join(_HANGUL.findall(unicodedata.normalize("NFC", text or "")))


def korean_syllables(text: str) -> list[str]:
    return list(normalize_korean(text))


# Complex final consonants breakdown (for liaison / simplification)
_COMPLEX_JONGSEONG = {
    "ㄳ": ("ㄱ", "ㅅ"),
    "ㄵ": ("ㄴ", "ㅈ"),
    "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"),
    "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"),
    "ㄾ": ("ㄹ", "ㅌ"),
    "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"),
    "ㅄ": ("ㅂ", "ㅅ"),
}


# ---------------------------------------------------------------------------
# Standard pronunciation (표준 발음법) -> Hangul reading, IPA and Revised
# Romanization. One rule pass over the whole sentence produces the pronounced
# jamo of every syllable; the reading, the IPA and the romanization are all
# spelled from it, so /reading, target_reading, ipa and romanization agree.
# ---------------------------------------------------------------------------

_TENSE_OF = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
_LENIS_OF = {v: k for k, v in _TENSE_OF.items()}
# 음절의 끝소리 규칙: every coda is read as one of 7 (ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅇ).
_CODA_REP = {
    "": "", "ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㄳ": "ㄱ", "ㄺ": "ㄱ",
    "ㄴ": "ㄴ", "ㄵ": "ㄴ", "ㄶ": "ㄴ",
    "ㄷ": "ㄷ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ",
    "ㄹ": "ㄹ", "ㄼ": "ㄹ", "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㅀ": "ㄹ",
    "ㅁ": "ㅁ", "ㄻ": "ㅁ", "ㅂ": "ㅂ", "ㅍ": "ㅂ", "ㅄ": "ㅂ", "ㄿ": "ㅂ", "ㅇ": "ㅇ",
}
# Codas that tense a following lenis onset inside a word (제23항 [ㄱ ㄷ ㅂ];
# 제25항 the verb stems in ㄼ: 넓게 [널께]). Across words only a coda read
# [ㄱ ㄷ ㅂ] does (여덟 시 [여덜 시]).
_STOP_CODAS = frozenset("ㄱㄲㅋㄳㄺㄷㅅㅆㅈㅊㅌㅂㅍㄼㄿㅄ")
_STOP_REPS = frozenset("ㄱㄷㅂ")
_ASPIRATED_OF = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}
_NASAL_OF = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}

_IPA_ONSET = {
    "ㄱ": "k", "ㄲ": "k͈", "ㅋ": "kʰ", "ㄷ": "t", "ㄸ": "t͈", "ㅌ": "tʰ",
    "ㅂ": "p", "ㅃ": "p͈", "ㅍ": "pʰ", "ㅈ": "tɕ", "ㅉ": "t͈ɕ", "ㅊ": "tɕʰ",
    "ㅅ": "s", "ㅆ": "s͈", "ㅎ": "h", "ㅁ": "m", "ㄴ": "n", "ㄹ": "ɾ", "ㅇ": "",
}
_IPA_VOWEL = {
    "ㅏ": "a", "ㅐ": "e", "ㅑ": "ja", "ㅒ": "je", "ㅓ": "ʌ", "ㅔ": "e", "ㅕ": "jʌ",
    "ㅖ": "je", "ㅗ": "o", "ㅘ": "wa", "ㅙ": "we", "ㅚ": "we", "ㅛ": "jo", "ㅜ": "u",
    "ㅝ": "wʌ", "ㅞ": "we", "ㅟ": "wi", "ㅠ": "ju", "ㅡ": "ɯ", "ㅢ": "ɰi", "ㅣ": "i",
}
_IPA_CODA = {"": "", "ㄱ": "k", "ㄴ": "n", "ㄷ": "t", "ㄹ": "l", "ㅁ": "m", "ㅂ": "p", "ㅇ": "ŋ"}

_RR_ONSET = {
    "ㄱ": "g", "ㄲ": "kk", "ㅋ": "k", "ㄷ": "d", "ㄸ": "tt", "ㅌ": "t",
    "ㅂ": "b", "ㅃ": "pp", "ㅍ": "p", "ㅈ": "j", "ㅉ": "jj", "ㅊ": "ch",
    "ㅅ": "s", "ㅆ": "ss", "ㅎ": "h", "ㅁ": "m", "ㄴ": "n", "ㄹ": "r", "ㅇ": "",
}
_RR_VOWEL = {
    "ㅏ": "a", "ㅐ": "ae", "ㅑ": "ya", "ㅒ": "yae", "ㅓ": "eo", "ㅔ": "e", "ㅕ": "yeo",
    "ㅖ": "ye", "ㅗ": "o", "ㅘ": "wa", "ㅙ": "wae", "ㅚ": "oe", "ㅛ": "yo", "ㅜ": "u",
    "ㅝ": "wo", "ㅞ": "we", "ㅟ": "wi", "ㅠ": "yu", "ㅡ": "eu", "ㅢ": "ui", "ㅣ": "i",
}
_RR_CODA = {"": "", "ㄱ": "k", "ㄴ": "n", "ㄷ": "t", "ㄹ": "l", "ㅁ": "m", "ㅂ": "p", "ㅇ": "ng"}

# Words read as one phrase with the next one (못 해요 [모태요], 몇 해 [며태]):
# a stop coda aspirates the next word's ㅎ (제12항 붙임 2).
_CLITIC_PREV = frozenset({"것", "몇", "못", "잘못", "옛", "꼭", "푹", "쭉", "약", "첫", "헛", "갓", "덧", "맛", "곧", "옷"})
# ... and any stop coda before the numeral 한 / 하나 (제12항 붙임 2: 꽃 한 송이
# [꼬탄송이], 낮 한때 [나탄때], 밥 한 그릇 [바판그륻]).
# Not 하나 (김밥 하나 주세요 [김밥 하나] in the data).
_HAN_NUMERAL_PREFIXES = ("한때", "한번", "한잔", "한두")
_NUMERAL_WORDS = frozenset({
    "십", "삼십", "이십", "사십", "오십", "육십", "칠십", "팔십", "구십", "백", "육", "일곱", "여섯",
    "다섯", "아홉", "여덟", "열", "스물", "몇십", "수십", "수백", "삼", "칠", "팔", "일", "이", "사", "오", "구",
})
# Last syllables that are (almost always) the -(으)ㄹ adnominal: 할 때, 갈 곳.
_ADNOMINAL_L = frozenset("할될갈릴올볼낼쓸줄룰킬알꿀쉴힐탈")
# Bound nouns read tense after the -(으)ㄹ adnominal (제27항: 할 수 [할쑤], 갈 곳 [갈꼳]).
_ADN_BOUND_WORDS = frozenset({"수", "수가", "수도", "수는", "수밖에", "수를", "수만", "수조차", "수록",
                              "거", "줄", "게", "데", "듯", "뿐"})
_ADN_BOUND_PREFIXES = ("것", "곳", "거예", "거야", "거죠", "거지", "거라", "거니", "거든", "거면", "거에", "거고")
# Verb-ending syllables after a ㄹ coda that stay lenis (만들다, 살지, 놀자);
# elsewhere Sino-Korean ㄹ + ㄷ/ㅅ/ㅈ is tense (제26항: 갈등 [갈뜽], 출시 [출씨],
# 질서 [질써], 칠십 [칠씹]). 도 and 지 are decided in _l_tense().
_L_LENIS_NEXT = frozenset("지자다도되될된됩돼됐더던든듯드들세셨습")
# Sino-Korean nouns ending in ㄹ + 도 (the particle 도 stays lenis: 말도, 길도).
_L_DO_NOUNS = ("철도", "절도", "별도", "밀도")
# After 지 (誌), a noun particle: 학술지에 [학쑬찌에]; the verb -지 takes 는/도/만/요.
_JI_NOUN_PARTICLES = frozenset("에가의와로")
# Native tens before 두/둘/셋: 열두 [열두], 스물셋 [스물섿].
_NATIVE_TENS = ("열", "스물")
# ㄹ-final adjective/verb stems whose adverbial -게 stays lenis (길게, 알게 됐어요).
_L_ADVERBIAL_STEMS = ("길", "멀", "달", "잘", "알", "힘들", "낯설", "가늘", "둥글")
# What follows the bound noun 거 (것) after the -(으)ㄹ adnominal: 할거예요, 갈거지.
_GEO_ENDINGS = frozenset("예야죠지든면에라니고요")
# 거리 after the adnominal (읽을거리, 볼거리 [볼꺼리]); 줄거리 stays lenis.
_GEORI_STEMS = frozenset("볼할놀쓸")
# Verb stems in ㄴ/ㅁ that tense the ending (제24항: 신고 [신꼬], 감다 [감따]),
# read only when the word is the stem plus one of the endings. 안다 is left out
# (안다 of 알다 is [안다]) and so is the noun 참고 (參考).
_NM_STEMS = frozenset({"신", "안", "감", "남", "넘", "참", "숨", "담", "품", "삼", "심", "껴안", "더듬", "머금"})
_NM_ENDINGS = frozenset({"다", "다가", "고", "고서", "지", "지만", "지요", "지도", "게", "도록", "던", "더라"})
# Sino-Korean suffixes read tense after a sonorant coda (사건 [사껀], 가능성 [가능썽]).
_SUFFIX_TENSE = frozenset("권건")
_SUFFIX_TENSE_LATE = frozenset("성점")
# Sino-Korean ㄴ + ㄹ read [ㄴㄴ] when the ㄹ starts a suffix (제20항 다만: 생산량 [생산냥]).
_NN_SUFFIXES = frozenset("란량력료론")
# Compounds with ㄴ insertion (제29항: 밭일 [반닐], 색연필 [생년필]): the
# syllable index whose ㅇ onset becomes ㄴ.
_N_INSERT_WORDS = {
    "밭일": 1, "막일": 1, "집안일": 2, "부엌일": 2, "볼일": 1, "한여름": 1, "늦여름": 1, "첫여름": 1,
    "색연필": 1, "꽃잎": 1, "깻잎": 1, "나뭇잎": 2, "솔잎": 1, "맨입": 1, "담요": 1, "눈요기": 1,
    "식용유": 2, "휘발유": 2, "영업용": 2, "내복약": 2, "알약": 1, "물약": 1, "솜이불": 1, "홑이불": 1,
    "신여성": 1, "콩엿": 1, "서울역": 2, "뒷일": 1, "완행열차": 2, "직행열차": 2,
}
# Compounds whose second part starts with a vowel: the coda is read as its
# representative before the liaison (제15항: 맛없다 [마덥따], 겉옷 [거돋]).
_REP_LIAISON_PREFIXES = ("맛없", "멋없", "겉옷", "헛웃", "첫인", "윗옷", "낮없", "값어")
# Word-final 의 that belongs to the noun (read [이]), not the possessive particle (read [에]).
# Matched as the whole word: 사회의, 질문의, 사건의 end in a particle.
_UI_NOUNS = ("회의", "거의", "하의", "상의", "편의", "논의", "동의", "강의", "의의", "정의", "협의",
             "합의", "예의", "모의", "결의", "건의", "심의", "토의", "질의", "문의", "신의", "호의",
             "고의", "유의", "내의", "수의", "주의", "민의", "본의", "자의", "타의", "열의", "성의")
# Compound nouns ending in one of them (위원회의 is 위원회 + the particle).
_UI_COMPOUNDS = ("국무회의", "화상회의", "정상회의", "가족회의", "임원회의", "비상회의", "영상회의",
                 "학급회의", "전체회의", "정기회의", "긴급회의", "온라인강의", "화상강의", "전화문의",
                 "고객문의")
# -주의 (-ism: 민주주의, 자본주의) is a noun of 4+ syllables; these X주 + 의 are the particle.
_JU_PARTICLE = ("위주", "제주", "맥주", "소주", "번주", "음주", "지난주", "다음주", "이번주")
# Nouns whose ㅎ syllable RR keeps although it looks like 하다 (북한 Bukhan, 역할 yeokhal).
_RR_H_NOUNS = frozenset({"북한", "극한", "국한", "역할", "직할"})
# What follows 하 when it is the verb 하다 (생각하다, 축하해요): the noun 하 is
# word-final or followed by something else (막상막하, 낙하산 nakhasan).
_HA_TAIL = frozenset("다고지게기는니면세시셔셨여며자러려도던든나네냐잖겠였었십신실해했합한할함드")
# Characters read aloud that the transcriber cannot spell (besides letters and digits).
_SPOKEN_SYMBOLS = frozenset("%$€₩£¥+=&@#℃°")
# Commas kept in the IPA after their word; written as "," in the romanization.
_COMMAS = (",", "，", "、")
_RR_COMMAS = str.maketrans("，、", ",,")


@dataclass
class KoreanSyllable:
    """One Hangul syllable of the text: as written and as pronounced."""

    text: str
    reading: str
    ipa: str
    romanization: str
    # Index in normalize_korean(text) (= the /evaluate mora_status cell);
    # None for a syllable spelled from an Arabic number.
    mora_index: int | None


@dataclass
class KoreanWord:
    """One run of Hangul with the punctuation attached: the text split at
    spaces and at punctuation between Hangul ("안녕"이라고 -> "안녕" + 이라고)."""

    text: str
    reading: str
    ipa: str
    romanization: str
    start: int
    end: int
    syllables: list[KoreanSyllable]
    space_after: bool = False  # a space follows it in the text


@dataclass
class KoreanTranscription:
    reading: str
    ipa: str | None
    romanization: str | None
    words: list[KoreanWord]
    supported: bool

    def per_mora(self, n_morae: int) -> list[KoreanSyllable] | None:
        """The syllables aligned with normalize_korean(text) (n_morae long),
        or None when unsupported or the counts do not match."""
        if not self.supported:
            return None
        cells = [s for w in self.words for s in w.syllables if s.mora_index is not None]
        if len(cells) != n_morae or any(s.mora_index != i for i, s in enumerate(cells)):
            return None
        return cells


@dataclass
class _Syl:
    o: tuple[str, str, str]  # orthographic jamo
    p: list[str]  # pronounced jamo, rewritten by the rules
    word: int
    wi: int
    wl: int
    wtext: str
    mora_index: int | None
    merge: bool = False  # onset made by an ㅎ merger (좋다 [조타], 좋습니다 [조씀니다])
    rr_keep_h: bool = False  # RR writes the ㅎ of a noun (급행 geuphaeng)
    rep_liaison: bool = False  # coda read as its representative before the liaison (맛없다)
    rr_coda: str | None = None
    phrase_end: bool = False  # its word ends the text or is followed by punctuation


@dataclass
class _Word:
    hangul: str
    syls: list[_Syl]
    text: str  # as written, punctuation attached, numbers as digits
    trailing: str = ""  # non-Hangul characters right after the run (a lone ", " token too)
    pause_after: bool = False  # punctuation between this word and the next: no sandhi
    space_after: bool = False


# A token's characters for the romanization: (char, (word, syllable) or None).
_Token = list[tuple[str, tuple[int, int] | None]]


def _segment(text: str) -> tuple[list[_Word], list[_Token], bool]:
    """Split `text` into words (runs of Hangul; Arabic numbers spelled out
    first) and say whether everything pronounceable in it is Hangul."""
    from app.services.cjk_numbers import KO_NUMBER, ko_number_reading, to_ascii_digits

    text = unicodedata.normalize("NFC", text or "")
    digits = to_ascii_digits(text)  # char-for-char: positions are shared with `text`
    words: list[_Word] = []
    tokens: list[_Token] = []
    supported = True
    mora = 0
    for tok in re.finditer(r"\S+", digits):
        # (char to pronounce, the text it stands for, an original Hangul syllable?)
        chars: list[tuple[str, str, bool]] = []
        pos = tok.start()
        for num in KO_NUMBER.finditer(digits, tok.start(), tok.end()):
            chars += [(c, c, True) for c in text[pos:num.start()]]
            spelled = ko_number_reading(num, digits[num.end():].lstrip(), digits)
            if spelled and all(is_hangul_syllable(c) for c in spelled):
                chars += [(c, text[num.start():num.end()] if k == 0 else "", False)
                          for k, c in enumerate(spelled)]
            else:
                chars += [(c, c, True) for c in text[num.start():num.end()]]
            pos = num.end()
        chars += [(c, c, True) for c in text[pos:tok.end()]]

        token: _Token = []
        current: _Word | None = None
        lead = ""
        for c, src, orig in chars:
            if is_hangul_syllable(c):
                if current is None or current.trailing:
                    if current is not None:
                        current.pause_after = True  # 아/어: two words, a pause between
                    elif lead and words:
                        words[-1].pause_after = True  # an opening quote / bracket
                        words[-1].trailing += lead
                    current = _Word(hangul="", syls=[], text=lead)
                    words.append(current)
                    lead = ""
                wi = len(current.hangul)
                current.hangul += c
                current.text += src
                current.syls.append(_Syl(decompose_hangul(c), list(decompose_hangul(c)), len(words) - 1,
                                         wi, 0, "", mora if orig else None))
                mora += orig
                token.append((c, (len(words) - 1, wi)))
                continue
            if unicodedata.category(c)[0] in "LN" or c in _SPOKEN_SYMBOLS:
                supported = False
            token.append((c, None))
            if current is None:
                lead += src
            else:
                current.text += src
                current.trailing += c
        if current is not None:
            current.pause_after = bool(current.trailing)
            current.space_after = True
        elif words:
            words[-1].pause_after = True  # a token of punctuation only (—, …, a lone ",")
            words[-1].trailing += lead
        tokens.append(token)
    if words:
        words[-1].space_after = False
    for w in words:
        for s in w.syls:
            s.wl = len(w.hangul)
            s.wtext = w.hangul
            s.phrase_end = w.pause_after or w is words[-1]
    return words, tokens, supported and bool(words)


def _lexical_marks(syls: list[_Syl], words: list[_Word]) -> None:
    """ㄴ insertion (제29항): 강남역 [강남녁], 밭일 [반닐], 십육 [심뉵]; across
    words only after the -ㄹ adnominal (할 일 [할릴]). And the compounds
    read with the representative coda before a vowel (맛없다 [마덥따])."""
    for w in words:
        for compound, k in _N_INSERT_WORDS.items():
            at = w.hangul.find(compound)  # 밭일을, 한여름에
            if at >= 0 and w.syls[at + k].o[0] == "ㅇ":
                w.syls[at + k].p[0] = "ㄴ"
        for prefix in _REP_LIAISON_PREFIXES:
            at = w.hangul.find(prefix)
            if at >= 0:
                w.syls[at].rep_liaison = True
    for a, b in zip(syls, syls[1:]):
        nxt = compose_hangul(*b.o)
        if a.word == b.word:
            if nxt == "역" and b.wi >= 2 and a.o[2]:
                b.p[0] = "ㄴ"
            elif nxt == "육" and compose_hangul(*a.o) in ("십", "백", "천", "만"):
                b.p[0] = "ㄴ"
        elif nxt == "일" and b.wi == 0 and a.o[2] == "ㄹ" and a.wtext[-1] in _ADNOMINAL_L \
                and not words[a.word].pause_after:
            b.p[0] = "ㄹ"


def _cross_word(a: _Syl, b: _Syl) -> None:
    """Sandhi between two words read as one phrase: tensing after a stop
    (한국 사람 [한국 싸람]), nasalization (제18항 붙임: 못 먹어 [몬 머거]),
    the -ㄹ adnominal (제27항: 할 수 [할 쑤]), 못 해요 [모 태요] and 일 년 [일 련].
    Words keep their own syllables otherwise (no liaison across a space)."""
    f1, c2 = a.p[2], b.p[0]
    w1, w2 = a.wtext, b.wtext
    rep = _CODA_REP.get(f1, f1)
    if rep in _STOP_REPS and c2 in _TENSE_OF:
        b.p[0] = _TENSE_OF[c2]
    elif rep in _STOP_REPS and c2 == "ㅎ" and (
            w1 in _CLITIC_PREV or w1 in _NUMERAL_WORDS or w2 == "한" or w2.startswith(_HAN_NUMERAL_PREFIXES)):
        if w2 == "한" or w2.startswith(_HAN_NUMERAL_PREFIXES):
            # RR keeps the words apart before 한 as the data does (꽃 한 송이 kkot han songi,
            # 옷 한 벌 ot han beol); 못 해요 stays mo taeyo.
            a.rr_coda, b.rr_keep_h = _RR_CODA[rep], True
        a.p[2] = ""
        b.p[0] = _ASPIRATED_OF[rep]
        b.merge = True
    elif rep in _NASAL_OF and c2 in ("ㄴ", "ㅁ"):
        a.p[2] = _NASAL_OF[rep]
    elif rep == "ㄹ" and c2 == "ㄴ" and w1[-1] in "일칠팔열" and w2.startswith(("년", "냥")):
        b.p[0] = "ㄹ"  # 일 년 [일 련], 열 냥 [열 량]; elsewhere words keep their ㄴ (말 너무)
    elif f1 == "ㄹ" and c2 in _TENSE_OF and not w1.endswith("를") and (
            w2 in _ADN_BOUND_WORDS or w2.startswith(_ADN_BOUND_PREFIXES) or w1[-1] in _ADNOMINAL_L):
        b.p[0] = _TENSE_OF[c2]


def _future_l(a: _Syl, cur: str) -> bool:
    """The ㄹ coda is the -(으)ㄹ adnominal (할, 먹을, 드릴, 이길)."""
    return cur in _ADNOMINAL_L or cur.endswith("을") or (cur == "길" and a.wi >= 1)


def _l_tense(a: _Syl, b: _Syl, cur: str, nxt: str) -> bool:
    """ㄹ coda + lenis onset inside a word: tense or not.

    제26항 Sino-Korean ㄹ + ㄷ/ㅅ/ㅈ (갈등 [갈뜽], 질서 [질써], 칠십 [칠씹],
    철도 [철또]) but not verb endings (만들다, 살지, 놀자) nor the particle
    도 (말도); 제27항 the -(으)ㄹ adnominal + 지/게/거/걸 (할지 [할찌],
    할게요 [할께요], 할거예요, 갈걸 [갈껄]) but not 즐거운, 설거지, 알게 됐어요."""
    c2 = b.p[0]
    after = b.wtext[b.wi + 1:]
    stem = a.wtext[:a.wi + 1]
    if c2 in ("ㄷ", "ㅅ", "ㅈ"):
        if nxt in ("두", "둘", "셋") and stem.endswith(_NATIVE_TENS):  # 열두, 스물셋
            return False
        if nxt == "지":
            if _future_l(a, cur) and cur not in ("알", "줄"):  # 갈지, 먹을지; 알지, 줄지어
                return True
            return after[:1] in _JI_NOUN_PARTICLES or b.wtext.endswith("학술지")
        if nxt == "도":
            return (bool(after) and after[0] not in "록요") or b.wtext.endswith(_L_DO_NOUNS)
        return nxt not in _L_LENIS_NEXT
    if b.wi < 1:
        return False
    if nxt == "걸":  # 할걸, 발걸음 [발꺼름]
        return True
    if nxt == "게":  # 할게요, 갈게.; not the adverbial -게 (길게, 알게 됐어요)
        if after.startswith("요"):
            return True
        return not after and b.phrase_end and not stem.endswith(_L_ADVERBIAL_STEMS)
    if nxt == "거":  # 할거예요, 읽을거리, 별거; not 즐거운, 줄거리, 지글거리다
        if not after or after[0] in _GEO_ENDINGS:
            return _future_l(a, cur) or (cur == "별" and not after)
        if after[0] == "리":
            return cur.endswith("을") or cur in _GEORI_STEMS
    return False


def _within_word(a: _Syl, b: _Syl) -> None:
    """The rules between two syllables of one word, in the order of 표준 발음법."""
    f1, c2, v2 = a.p[2], b.p[0], b.p[1]
    nxt = compose_hangul(*b.o)
    cur = compose_hangul(*a.o)
    if not f1:
        # Tense Sino-Korean suffixes after a vowel too (여권 [여꿘], 중요성 [중요썽]).
        if c2 in _TENSE_OF and b.wi >= 1 and (nxt == "권" or (nxt == "성" and b.wi >= 2)):
            b.p[0] = _TENSE_OF[c2]
        return
    # 밟- reads [밥] before a consonant (제10항 다만: 밟다 [밥따], 밟는 [밤는]),
    # and so do 넓죽-, 넓둥글-; not before ㅎ (밟히다 [발피다]).
    if f1 == "ㄼ" and c2 not in ("ㅇ", "ㅎ") and (
            cur == "밟" or a.wtext[a.wi:].startswith(("넓죽", "넓둥"))):
        a.p[2] = f1 = "ㅂ"

    # Liaison (연음, 제13-14항) and palatalization (구개음화, 제17항).
    if c2 == "ㅇ":
        if a.rep_liaison:  # 맛없어요 [마덥써요], 겉옷 [거돋]
            a.p[2], b.p[0] = "", _CODA_REP.get(f1, f1)
        elif f1 in _COMPLEX_JONGSEONG:
            left, right = _COMPLEX_JONGSEONG[f1]
            if right == "ㅎ":  # 않아 [아나], 싫어 [시러]
                a.p[2], b.p[0] = "", left
            elif right == "ㅅ":  # 없어 [업써]
                a.p[2], b.p[0] = left, "ㅆ"
            else:  # 읽어 [일거]; 벼훑이 [벼훌치]
                a.p[2], b.p[0] = left, "ㅊ" if right == "ㅌ" and v2 == "ㅣ" else right
        elif f1 == "ㅎ":  # 좋아요 [조아요]
            a.p[2] = ""
        elif f1 == "ㄷ" and v2 == "ㅣ":  # 굳이 [구지]
            a.p[2], b.p[0] = "", "ㅈ"
        elif f1 == "ㅌ" and v2 == "ㅣ":  # 같이 [가치]
            a.p[2], b.p[0] = "", "ㅊ"
        elif f1 != "ㅇ":
            a.p[2], b.p[0] = "", f1
        return

    # ㅎ (제12항).
    if f1 in ("ㅎ", "ㄶ", "ㅀ"):
        rest = "ㄹ" if f1 == "ㅀ" else ("ㄴ" if f1 == "ㄶ" else "")
        if c2 in _ASPIRATED_OF:  # 좋다 [조타], 많고 [만코]
            a.p[2], b.p[0], b.merge = rest, _ASPIRATED_OF[c2], True
            return
        if c2 == "ㅅ":  # 좋습니다 [조씀니다]
            a.p[2], b.p[0], b.merge = rest, "ㅆ", True
            return
        if c2 in ("ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ"):  # 놓치다 [노치다]
            a.p[2] = rest
            return
        if c2 == "ㄴ" and f1 != "ㅀ":  # 놓는 [논는], 않는 [안는]
            a.p[2] = "ㄴ"
            return
    if f1 in ("ㄱ", "ㄷ", "ㅂ", "ㅈ", "ㄺ", "ㄼ", "ㄵ", "ㅅ", "ㅆ", "ㅊ", "ㅌ") and c2 == "ㅎ":
        # 축하 [추카], 못하다 [모타다], 밝히다 [발키다], 앉히다 [안치다]; 닫히다 [다치다]
        a.p[2] = {"ㄺ": "ㄹ", "ㄼ": "ㄹ", "ㄵ": "ㄴ"}.get(f1, "")
        b.p[0] = {"ㅈ": "ㅊ", "ㄵ": "ㅊ", "ㄺ": "ㅋ", "ㄼ": "ㅍ"}.get(f1) or _ASPIRATED_OF[_CODA_REP[f1]]
        b.merge = True
        if b.p[0] == "ㅌ" and v2 == "ㅣ":
            b.p[0] = "ㅊ"
        return

    # Liquidization (유음화, 제20항): 신라 [실라], 칼날 [칼랄]; 생산량 [생산냥].
    if f1 in ("ㄴ", "ㄵ") and c2 == "ㄹ":
        if nxt in _NN_SUFFIXES and b.wi >= 2:
            b.p[0] = "ㄴ"
        else:
            a.p[2], b.p[0] = "ㄹ", "ㄹ"
        return
    if f1 in ("ㄹ", "ㄼ", "ㄾ", "ㅀ") and c2 == "ㄴ":  # 넓네요 [널레요], 핥네 [할레]
        a.p[2], b.p[0] = "ㄹ", "ㄹ"
        return
    # ㄹ after another consonant is read [ㄴ] (제19항): 식량 [싱냥], 협력 [혐녁].
    if c2 == "ㄹ" and _CODA_REP.get(f1) not in ("ㄹ", "ㄴ"):
        b.p[0] = c2 = "ㄴ"

    # Nasalization (비음화, 제18항): 감사합니다 [감사함니다], 국내 [궁내].
    rep = _CODA_REP.get(f1, f1)
    if rep in _NASAL_OF and c2 in ("ㄴ", "ㅁ"):
        a.p[2] = _NASAL_OF[rep]

    # Tensification (경음화, 제23-27항).
    if c2 in _TENSE_OF:
        causative = c2 == "ㄱ" and b.o[1] in ("ㅣ", "ㅕ")  # 옮기다, 안기다 stay lenis
        if (f1 in _STOP_CODAS and not (f1 == "ㄼ" and cur == "덟")) \
                or (f1 in ("ㄵ", "ㄻ") and not causative) or f1 == "ㄾ":
            # 학교 [학꾜], 넓게 [널께], 앉다 [안따], 핥다 [할따]; the numeral 여덟시 [여덜시]
            b.p[0] = _TENSE_OF[c2]
        elif f1 == "ㄹ" and _l_tense(a, b, cur, nxt):
            b.p[0] = _TENSE_OF[c2]
        elif f1 in ("ㄴ", "ㅁ") and b.wtext[:b.wi] in _NM_STEMS and b.wtext[b.wi:] in _NM_ENDINGS \
                and b.wtext != "참고" and not (b.wtext[:b.wi] == "안" and nxt == "다"):
            b.p[0] = _TENSE_OF[c2]  # 신고 [신꼬], 감다 [감따], 껴안다 [껴안따]
        elif f1 in ("ㄴ", "ㅁ", "ㅇ") and b.wi >= 1 and (
                nxt in _SUFFIX_TENSE or (nxt in _SUFFIX_TENSE_LATE and b.wi >= 2)):  # 인권 [인꿘]
            b.p[0] = _TENSE_OF[c2]
        elif f1 == "ㄹ" and nxt == "성" and b.wi >= 2:  # 효율성 [효율썽]
            b.p[0] = _TENSE_OF[c2]
    # 용언 ㄺ + ㄱ (제11항 다만): 읽고 [일꼬], 맑게 [말께]; 닭고기 stays [닥꼬기].
    if f1 == "ㄺ" and c2 in ("ㄱ", "ㄲ") and compose_hangul(*a.o) not in ("닭", "흙", "칡"):
        a.p[2] = "ㄹ"


def _ui_noun(word: str) -> bool:
    """`word` ends in a noun's 의 (회의, 국무회의, 민주주의), not the particle
    (사회의, 위원회의, 위주의, 맥주의)."""
    if word in _UI_NOUNS or word.endswith(_UI_COMPOUNDS):
        return True
    return len(word) >= 4 and word.endswith("주의") and not word[:-1].endswith(_JU_PARTICLE)


def _realize_vowels(syls: list[_Syl]) -> None:
    """제5항: 희 [히], 사람의 [사라메], 회의 [회이]; 져 쪄 쳐 [저 쩌 처]."""
    for s in syls:
        cho, jung, _ = s.p
        if jung == "ㅢ":
            if s.o[0] != "ㅇ":
                s.p[1] = "ㅣ"
            elif s.wi == 0:
                pass
            elif s.wi == s.wl - 1 and not _ui_noun(s.wtext):
                s.p[1] = "ㅔ"
            else:
                s.p[1] = "ㅣ"
        elif jung in ("ㅕ", "ㅖ") and cho in ("ㅈ", "ㅊ", "ㅉ"):
            s.p[1] = "ㅓ" if jung == "ㅕ" else "ㅔ"


def _romanization_marks(syls: list[_Syl]) -> None:
    """RR keeps the ㅎ of a noun after ㄱ/ㄷ/ㅂ (제3장 제1항 붙임: 급행
    geuphaeng, 집현전 Jiphyeonjeon, 북한 Bukhan, 낙하산 nakhasan, 축하 chukha);
    verbs and adjectives in 하-/히- are written aspirated (착해요 chakaeyo,
    축하해요 chukahaeyo, 막히다 makida)."""
    for a, b in zip(syls, syls[1:]):
        if a.word != b.word or not b.merge or b.o[0] != "ㅎ" or a.o[2] not in ("ㄱ", "ㄷ", "ㅂ", "ㄺ", "ㄼ"):
            continue
        syl = compose_hangul(*b.o)
        after = b.wtext[b.wi + 1:b.wi + 2]
        if syl in ("해", "했", "히", "힌", "힐", "힙", "혀", "혔", "함"):
            verbal = True
        elif syl == "하":
            verbal = after in _HA_TAIL if after else False
        elif syl == "합":
            verbal = after in ("니", "시")
        elif syl in ("한", "할"):
            verbal = compose_hangul(*a.o) + syl not in _RR_H_NOUNS
        else:  # 학, 행, 형, 현, 화, 회...: never 하다
            verbal = False
        if not verbal:
            b.rr_keep_h = True
            a.rr_coda = {"ㄱ": "k", "ㄷ": "t", "ㅂ": "p", "ㄺ": "lk", "ㄼ": "lp"}[a.o[2]]


def _ipa(s: _Syl, prev: _Syl | None) -> str:
    cho, jung, jong = s.p
    onset = "l" if cho == "ㄹ" and prev is not None and prev.p[2] == "ㄹ" else _IPA_ONSET[cho]
    return onset + _IPA_VOWEL[jung] + _IPA_CODA[jong]


def _rr(s: _Syl, prev: _Syl | None) -> str:
    cho, ocho = s.p[0], s.o[0]
    # RR does not write tensification (학교 hakgyo, 할 수 hal su) ...
    if cho in _LENIS_OF and ocho == _LENIS_OF[cho] and not s.merge:
        cho = ocho
    # ... nor the ㅆ of ㄳ/ㄽ/ㅄ + vowel (없어 eopseo).
    if cho == "ㅆ" and ocho == "ㅇ" and prev is not None and prev.o[2] in ("ㄳ", "ㄽ", "ㅄ"):
        cho = "ㅅ"
    if s.rr_keep_h:
        onset = "h"
    elif cho == "ㄹ" and prev is not None and prev.p[2] == "ㄹ":
        onset = "l"  # 신라 silla
    else:
        onset = _RR_ONSET[cho]
    coda = s.rr_coda if s.rr_coda is not None else _RR_CODA[s.p[2]]
    return onset + _RR_VOWEL[s.o[1]] + coda  # vowels as written (사람의 saramui)


def transcribe_korean(text: str) -> KoreanTranscription:
    """`text` read by 표준 발음법: the Hangul reading, the IPA ([han.ku.kʌ])
    and the Revised Romanization (hangugeo), per sentence, word and syllable.

    Arabic numbers are spelled out first (3개 -> 세개). `ipa` and
    `romanization` are None when the text has other things to read (Latin
    letters, digits it cannot spell, symbols such as %): the reading then
    covers the Hangul only."""
    words, tokens, supported = _segment(text)
    syls = [s for w in words for s in w.syls]
    _lexical_marks(syls, words)
    for a, b in zip(syls, syls[1:]):
        if a.word == b.word:
            _within_word(a, b)
        elif not words[a.word].pause_after:
            _cross_word(a, b)
    for s in syls:
        s.p[2] = _CODA_REP.get(s.p[2], s.p[2])
    _realize_vowels(syls)
    _romanization_marks(syls)

    out_words: list[KoreanWord] = []
    rr_of: dict[tuple[int, int], str] = {}
    mora = 0
    for wi, w in enumerate(words):
        cells = []
        for k, s in enumerate(w.syls):
            # The syllable before, across a space too (일 년 [il ljʌn] il lyeon).
            prev = w.syls[k - 1] if k else (
                words[wi - 1].syls[-1] if wi and not words[wi - 1].pause_after else None)
            rr = _rr(s, prev)
            rr_of[(wi, k)] = rr
            cells.append(KoreanSyllable(text=compose_hangul(*s.o), reading=compose_hangul(*s.p),
                                        ipa=_ipa(s, prev), romanization=rr, mora_index=s.mora_index))
        start = mora
        mora += sum(c.mora_index is not None for c in cells)
        out_words.append(KoreanWord(
            text=w.text, reading="".join(c.reading for c in cells),
            ipa=".".join(c.ipa for c in cells), romanization="".join(c.romanization for c in cells),
            start=start, end=mora, syllables=cells, space_after=w.space_after,
        ))

    reading = " ".join(w.reading for w in out_words)
    if not supported:
        return KoreanTranscription(reading=reading, ipa=None, romanization=None, words=out_words, supported=False)
    ipa = "[" + " ".join(w.ipa + ("," if any(c in src.trailing for c in _COMMAS) else "")
                         for w, src in zip(out_words, words)) + "]"
    romanization = " ".join("".join(rr_of[ref] if ref else c for c, ref in tok) for tok in tokens)
    romanization = romanization.translate(_RR_COMMAS)  # 좋아요，정말요 -> Joayo,jeongmaryo
    if re.search(r"[.?!]\W*$", text.strip()):  # a sentence: capitalised like the exercise data
        romanization = re.sub(r"(^[^a-z]*|[.?!]\s+[^a-z\s]*)([a-z])",
                              lambda m: m.group(1) + m.group(2).upper(), romanization)
    return KoreanTranscription(reading=reading, ipa=ipa, romanization=romanization, words=out_words,
                               supported=True)


def korean_pronunciation(text: str) -> str:
    """Korean text in its standard pronunciation, written in Hangul, one
    group per word (numbers spelled out, other non-Hangul dropped):
    한국어 -> 한구거, 감사합니다 -> 감사함니다, 신라 -> 실라, 좋다 -> 조타,
    학교 -> 학꾜, 같이 -> 가치, 못 먹어요 -> 몬 머거요, 3개 -> 세개.
    See transcribe_korean() for the IPA and the romanization."""
    if not text:
        return ""
    return transcribe_korean(text).reading


def korean_pitch_pattern(text: str) -> list[dict]:
    """Generate Seoul Korean Accentual Phrase (AP) intonation model for each syllable.

    K-ToBI Accentual Phrase pattern:
    - High onset (격음/된소리/ㅅ/ㅎ): starts H (pattern: H H L H or H L)
    - Low onset (other consonants): starts L (pattern: L H L H or L H)
    - Declarative sentence ends with a falling boundary tone (L).
    """
    raw_words = re.findall(r"[가-힣]+", unicodedata.normalize("NFC", text))
    if not raw_words:
        return []

    is_question = "?" in text
    pattern = []

    for phrase_idx, word in enumerate(raw_words):
        sylls = list(word)
        m = len(sylls)
        if m == 0:
            continue

        cho, _, _ = decompose_hangul(sylls[0])
        is_high_onset = cho in _HIGH_ONSETS

        # Base AP contour for m syllables
        if m == 1:
            tones = ["H"] if is_high_onset else ["L"]
        elif m == 2:
            tones = ["H", "H"] if is_high_onset else ["L", "H"]
        elif m == 3:
            tones = ["H", "L", "H"] if is_high_onset else ["L", "H", "H"]
        else:
            first = "H" if is_high_onset else "L"
            mid = ["H"] * (m - 3)
            tones = [first, "H"] + mid + ["L", "H"]
            tones = tones[:m]

        # Sentence-final boundary tone
        if phrase_idx == len(raw_words) - 1 and not is_question and len(tones) > 0:
            tones[-1] = "L"

        for s, t in zip(sylls, tones):
            pattern.append({
                "mora": s,
                "pitch": t,
                "phrase": phrase_idx,
            })

    return pattern


def score_korean_syllables(target_text: str, recognized_text: str) -> ScoreResult:
    """Score Korean read-aloud considering both orthographic and phonetic realizations."""
    norm_target = normalize_korean(target_text)
    norm_rec = normalize_korean(recognized_text)

    if not norm_target:
        level, message = _feedback(0)
        return ScoreResult(
            score=0,
            cer=1.0,
            distance=len(norm_rec),
            target_length=0,
            level=level,
            message=message,
            errors=[],
        )

    # Also compute phonetic readings
    ph_target = "".join(normalize_korean(korean_pronunciation(target_text)))
    ph_rec = "".join(normalize_korean(korean_pronunciation(recognized_text)))

    # Direct match (orthographic or phonetic)
    if norm_target == norm_rec or ph_target == ph_rec or ph_target == norm_rec or norm_target == ph_rec:
        level, message = _feedback(100)
        return ScoreResult(
            score=100,
            cer=0.0,
            distance=0,
            target_length=len(norm_target),
            level=level,
            message=message,
            errors=[],
        )

    # Compute edit distance on the best matching representation
    d_orth = edit_distance(list(norm_target), list(norm_rec))
    d_phon = edit_distance(list(ph_target), list(ph_rec))

    if d_phon <= d_orth:
        base_t, base_r = ph_target, ph_rec
        best_d = d_phon
    else:
        base_t, base_r = norm_target, norm_rec
        best_d = d_orth

    # Map operations back to target syllables
    raw_ops = edit_ops(list(base_t), list(base_r))
    errors: list[PronunciationError] = []
    for tag, src, dst in raw_ops:
        if tag == "insert":
            errors.append(PronunciationError(type="ins", target="", recognized=base_r[dst], position=src))
        elif tag == "delete":
            errors.append(PronunciationError(type="del", target=norm_target[min(src, len(norm_target)-1)], recognized="", position=src))
        elif tag == "replace":
            # Check if this character is a valid phonetic variant (e.g. 합 vs 함)
            t_char = norm_target[min(src, len(norm_target)-1)]
            r_char = norm_rec[min(dst, len(norm_rec)-1)] if dst < len(norm_rec) else base_r[dst]
            if t_char != r_char:
                errors.append(PronunciationError(type="sub", target=t_char, recognized=r_char, position=src))

    actual_dist = len(errors)
    n = len(norm_target)
    cer = actual_dist / n if n > 0 else 0.0
    score = max(0, round((1.0 - cer) * 100))
    level, message = _feedback(score)

    return ScoreResult(
        score=score,
        cer=round(cer, 4),
        distance=actual_dist,
        target_length=n,
        level=level,
        message=message,
        errors=errors,
    )
