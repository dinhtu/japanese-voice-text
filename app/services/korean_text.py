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


def korean_pronunciation(text: str) -> str:
    """Convert Korean orthographic text into standard phonetic reading (Hangul).

    Applies standard Korean pronunciation rules:
    - Liaison (연음): e.g. 한국어 -> 한구거, 음악 -> 으막, 있어요 -> 이써요
    - Nasalization (비음화): e.g. 감사합니다 -> 감사함니다, 있는 -> 인는, 국내 -> 궁내
    - Liquidization (유음화): e.g. 신라 -> 실라, 연락 -> 열락
    - Aspiration & H-merger (격음화/ㅎ탈락): e.g. 좋아요 -> 조아요, 축하 -> 추카, 좋다 -> 조타
    - Tensification (경음화): e.g. 학교 -> 학꾜, 식당 -> 식땅, 반갑습니다 -> 반갑씀니다
    - Palatalization (구개음화): e.g. 같이 -> 가치, 굳이 -> 구지
    """
    if not text:
        return ""

    # Tokenize into words to respect word boundaries
    words = re.findall(r"[가-힣]+", unicodedata.normalize("NFC", text))
    phonetic_words = []

    for word in words:
        sylls = [list(decompose_hangul(c)) for c in word]
        n = len(sylls)
        if n == 0:
            continue

        # Step 1: Pairwise phonological rules between adjacent syllables
        for i in range(n - 1):
            c1, v1, f1 = sylls[i]
            c2, v2, f2 = sylls[i + 1]

            # 1.1 Liaison (연음) when next syllable begins with vowel (ㅇ)
            if f1 and c2 == "ㅇ":
                if f1 in _COMPLEX_JONGSEONG:
                    left, right = _COMPLEX_JONGSEONG[f1]
                    if right == "ㅎ":
                        # ㄶ, ㅀ before vowel -> ㅎ drops, leaving left consonant
                        sylls[i][2] = ""
                        sylls[i + 1][0] = left
                    elif right == "ㅅ":
                        sylls[i][2] = left
                        sylls[i + 1][0] = "ㅆ"
                    else:
                        sylls[i][2] = left
                        sylls[i + 1][0] = right
                elif f1 == "ㅎ":
                    # ㅎ before vowel drops (좋아요 -> 조아요)
                    sylls[i][2] = ""
                elif f1 == "ㅅ":
                    sylls[i][2] = ""
                    sylls[i + 1][0] = "ㅅ"
                elif f1 == "ㅆ":
                    sylls[i][2] = ""
                    sylls[i + 1][0] = "ㅆ"
                elif f1 == "ㄷ" and v2 == "ㅣ":
                    # Palatalization: 굳이 -> 구지
                    sylls[i][2] = ""
                    sylls[i + 1][0] = "ㅈ"
                elif f1 == "ㅌ" and v2 == "ㅣ":
                    # Palatalization: 같이 -> 가치
                    sylls[i][2] = ""
                    sylls[i + 1][0] = "ㅊ"
                elif f1 != "ㅇ":
                    sylls[i][2] = ""
                    sylls[i + 1][0] = f1
                continue

            # 1.2 Aspiration with ㅎ (격음화)
            if f1 in ("ㅎ", "ㄶ", "ㅀ") and c2 in ("ㄱ", "ㄷ", "ㅂ", "ㅈ"):
                merges = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}
                sylls[i][2] = "ㄹ" if f1 == "ㅀ" else ("ㄴ" if f1 == "ㄶ" else "")
                sylls[i + 1][0] = merges[c2]
                continue
            if f1 in ("ㄱ", "ㄷ", "ㅂ", "ㅈ", "ㄺ", "ㄼ") and c2 == "ㅎ":
                merges = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ", "ㄺ": "ㅋ", "ㄼ": "ㅍ"}
                sylls[i][2] = "ㄹ" if f1 in ("ㄺ", "ㄼ") else ""
                sylls[i + 1][0] = merges[f1]
                continue

            # 1.3 Liquidization (유음화: ㄴ+ㄹ -> ㄹㄹ, ㄹ+ㄴ -> ㄹㄹ)
            if (f1 in ("ㄴ", "ㄵ") and c2 == "ㄹ") or (f1 in ("ㄹ", "ㄾ", "ㅀ") and c2 == "ㄴ"):
                sylls[i][2] = "ㄹ"
                sylls[i + 1][0] = "ㄹ"
                continue

            # 1.4 Nasalization (비음화)
            # ㅂ, ㅍ, ㄿ, ㅄ + ㄴ/ㅁ -> [ㅁ]
            if f1 in ("ㅂ", "ㅍ", "ㄿ", "ㅄ") and c2 in ("ㄴ", "ㅁ"):
                sylls[i][2] = "ㅁ"
            # ㄷ, ㅅ, ㅆ, ㅈ, ㅊ, ㅌ, ㅎ + ㄴ/ㅁ -> [ㄴ]
            elif f1 in ("ㄷ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅌ", "ㅎ") and c2 in ("ㄴ", "ㅁ"):
                sylls[i][2] = "ㄴ"
            # ㄱ, ㄲ, ㅋ, ㄳ, ㄺ + ㄴ/ㅁ -> [ㅇ]
            elif f1 in ("ㄱ", "ㄲ", "ㅋ", "ㄳ", "ㄺ") and c2 in ("ㄴ", "ㅁ"):
                sylls[i][2] = "ㅇ"

            # 1.5 Tensification (경음화) after stop coda [ㄱ, ㄷ, ㅂ]
            rep_stop = f1 in ("ㄱ", "ㄲ", "ㅋ", "ㄳ", "ㄺ", "ㄷ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅌ", "ㅂ", "ㅍ", "ㄼ", "ㄿ", "ㅄ")
            if rep_stop and c2 in ("ㄱ", "ㄷ", "ㅂ", "ㅅ", "ㅈ"):
                tensed = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
                sylls[i + 1][0] = tensed[c2]

        # Step 2: Final coda simplification (음절의 끝소리 규칙)
        for i in range(n):
            cho, jung, jong = sylls[i]
            if not jong:
                continue
            if jong in ("ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅌ", "ㅎ"):
                sylls[i][2] = "ㄷ"
            elif jong in ("ㅋ", "ㄲ", "ㄳ"):
                sylls[i][2] = "ㄱ"
            elif jong in ("ㅍ", "ㅄ"):
                sylls[i][2] = "ㅂ"
            elif jong == "ㄺ":
                sylls[i][2] = "ㄱ"
            elif jong == "ㄻ":
                sylls[i][2] = "ㅁ"
            elif jong in ("ㄼ", "ㄽ", "ㄾ", "ㅀ"):
                sylls[i][2] = "ㄹ"
            elif jong == "ㄵ":
                sylls[i][2] = "ㄴ"

        word_out = "".join(compose_hangul(*s) for s in sylls)
        phonetic_words.append(word_out)

    return " ".join(phonetic_words)


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
        return ScoreResult(score=0, cer=1.0, distance=len(norm_rec), length=0, errors=[], feedback=_feedback(0))

    # Also compute phonetic readings
    ph_target = "".join(normalize_korean(korean_pronunciation(target_text)))
    ph_rec = "".join(normalize_korean(korean_pronunciation(recognized_text)))

    # Direct match (orthographic or phonetic)
    if norm_target == norm_rec or ph_target == ph_rec or ph_target == norm_rec or norm_target == ph_rec:
        return ScoreResult(score=100, cer=0.0, distance=0, length=len(norm_target), errors=[], feedback=_feedback(100))

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
            errors.append(PronunciationError(type="ins", target=None, recognized=base_r[dst], position=src))
        elif tag == "delete":
            errors.append(PronunciationError(type="del", target=norm_target[min(src, len(norm_target)-1)], recognized=None, position=src))
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

    return ScoreResult(
        score=score,
        cer=round(cer, 4),
        distance=actual_dist,
        length=n,
        errors=errors,
        feedback=_feedback(score),
    )
