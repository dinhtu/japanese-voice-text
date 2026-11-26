"""The Korean responses: /evaluate and /evaluate-v2 (EvaluateResponse plus
the optional coach comment and, for v2, what v2 did), /reading,
/pitch-accent and /transcribe.

Every Korean transcription field (ipa, romanization, words) lives in the
Korean subclasses here, so the ja / en / zh payloads keep their shape."""

from typing import Annotated

from pydantic import BaseModel, Field

from app.schemas.cjk_v2 import CjkV2Assessment
from app.schemas.coaching import CoachComment
from app.schemas.pitch_accent import MoraPitchItem, PitchAccentResponse
from app.schemas.pronunciation import EvaluateResponse, MoraStatusItem

_IPA_NULL = " Null when the text has something the transcriber cannot read (Latin letters...)."


class KoreanMoraStatusItem(MoraStatusItem):
    """One Hangul syllable of the target, correct or not, with how it is pronounced."""

    ipa: str | None = Field(
        default=None,
        description='IPA of this syllable after the sound changes (표준 발음법), e.g. 합니다 -> "ham", "ni", "ta".'
        + _IPA_NULL,
    )
    romanization: str | None = Field(
        default=None,
        description='Revised Romanization of this syllable, lowercase, e.g. 한국어 -> "han", "gu", "geo".'
        + _IPA_NULL,
    )


    mora: str = Field(description="One written Hangul syllable of the target (digits have no cell)")


class KoreanMoraPitchItem(MoraPitchItem):
    mora: str = Field(description="One written Hangul syllable (digits have no cell)")
    ipa: str | None = Field(default=None, description='IPA of this syllable, e.g. "kam".' + _IPA_NULL)
    romanization: str | None = Field(default=None, description='Revised Romanization, e.g. "gam".' + _IPA_NULL)


class KoreanWordSyllableItem(BaseModel):
    """One pronounced syllable of a word, including those spelled from a number."""

    text: str = Field(description="The Hangul syllable: as written, or spelled from a digit (3개 -> 세, 개)")
    reading: str = Field(description="Its standard pronunciation in Hangul, e.g. 국 -> 구")
    ipa: str = Field(description='IPA, no dots or brackets, e.g. "ku"')
    romanization: str = Field(description='Revised Romanization, lowercase, e.g. "gu"')
    mora_index: int | None = Field(
        description="Its cell in mora_status / pattern; null for a syllable spelled from a number "
        "(it has no cell)",
    )


class KoreanWordItem(BaseModel):
    """One run of Hangul of the text with the punctuation attached: the text
    split at spaces and at punctuation between Hangul ("안녕"이라고 -> "안녕" and
    이라고; 가.나 -> 가. and 나). Show a word with its own `ipa` / `romanization`
    (or its `syllables`), never by joining mora_status cells: a number's
    syllables have no cell (3개: one cell 개, the word reads segae)."""

    text: str = Field(description="The word as written, with its punctuation (numbers as digits)")
    reading: str = Field(description="Its standard pronunciation in Hangul, e.g. 한구거")
    ipa: str = Field(description='IPA, syllables joined by ".", no brackets, e.g. "han.ku.kʌ"')
    romanization: str = Field(description='Revised Romanization, lowercase, no punctuation, e.g. "hangugeo"')
    start: int = Field(description="Index of its first syllable in mora_status / pattern")
    end: int = Field(description="Exclusive end index in mora_status / pattern (start == end: no Hangul, e.g. 50%)")
    space_after: bool = Field(
        default=False,
        description="A space follows it in the text (false inside 가.나 or \"안녕\"이라고, and for the last word)",
    )
    syllables: list[KoreanWordSyllableItem] = Field(
        default_factory=list,
        description="Every pronounced syllable of the word in order (their ipa joined by '.' and their "
        "romanization joined give the word's), with its mora_status index",
    )


class KoreanEvaluateResponse(EvaluateResponse):
    mora_status: list[KoreanMoraStatusItem] = Field(
        default_factory=list,
        description="One item per Hangul syllable of the target, flagged correct/incorrect, "
        "with its IPA and romanization",
    )
    target_ipa: str | None = Field(
        default=None,
        description='The target in IPA after the sound changes, e.g. "[kam.sa.ham.ni.ta]" '
        "(the format of the exercise `kana`)." + _IPA_NULL,
    )
    target_romanization: str | None = Field(
        default=None,
        description='The target in Revised Romanization, e.g. "Gamsahamnida." (the exercise `romaji`).'
        + _IPA_NULL,
    )
    words: list[KoreanWordItem] | None = Field(
        default=None,
        description="The target word by word (start/end index mora_status)." + _IPA_NULL,
    )
    coach: CoachComment | None = Field(
        default=None, description="AI comment (same fields as POST /coach); only with coach=true",
    )
    coach_error: str | None = Field(default=None, description="Why `coach` is null when it was asked for")


class KoreanEvaluateV2Response(KoreanEvaluateResponse):
    """POST /evaluate-v2: the /evaluate fields (values from the Qwen3-ASR
    transcript) plus what v2 did; `coach` comes with feedback=true."""

    v2_assessment: CjkV2Assessment | None = Field(
        default=None, description="ASR engine / model, raw transcript, alignment used for timing",
    )


class KoreanReadingResponse(BaseModel):
    reading: str = Field(description="Standard pronunciation in Hangul, one group per word, e.g. 감사함니다")
    ipa: str | None = Field(default=None, description='e.g. "[kam.sa.ham.ni.ta]".' + _IPA_NULL)
    romanization: str | None = Field(default=None, description='e.g. "gamsahamnida".' + _IPA_NULL)
    words: list[KoreanWordItem] | None = Field(default=None, description="Word by word." + _IPA_NULL)


class KoreanPitchAccentResponse(PitchAccentResponse):
    reading: str = Field(
        description="The written Hangul syllables, space-joined (digits dropped; see words / ipa "
        "for the pronunciation)",
    )
    pattern: list[KoreanMoraPitchItem] = Field(
        description="Accentual Phrase H/L per Hangul syllable, with its IPA and romanization",
    )
    ipa: str | None = Field(default=None, description='e.g. "[an.njʌŋ.ha.se.jo]".' + _IPA_NULL)
    romanization: str | None = Field(default=None, description='e.g. "annyeonghaseyo".' + _IPA_NULL)
    words: list[KoreanWordItem] | None = Field(default=None, description="Word by word." + _IPA_NULL)


TRANSCRIBE_MAX_ITEMS = 200
TRANSCRIBE_MAX_CHARS = 500


class KoreanTranscribeRequest(BaseModel):
    items: dict[str, Annotated[str, Field(max_length=TRANSCRIBE_MAX_CHARS)]] = Field(
        max_length=TRANSCRIBE_MAX_ITEMS,
        description=f'id -> Korean text, at most {TRANSCRIBE_MAX_ITEMS} items of at most '
        f'{TRANSCRIBE_MAX_CHARS} characters, e.g. {{"12": "감사합니다"}}',
    )


class KoreanTranscriptionItem(BaseModel):
    kana: str = Field(description='IPA as stored in the exercise `kana`, e.g. "[kam.sa.ham.ni.ta]"')
    romaji: str = Field(description='Revised Romanization as stored in `romaji`, e.g. "Gamsahamnida."')
    reading: str = Field(description="Standard pronunciation in Hangul, e.g. 감사함니다")


class KoreanTranscribeResponse(BaseModel):
    results: dict[str, KoreanTranscriptionItem] = Field(default_factory=dict)
    unsupported: list[str] = Field(
        default_factory=list,
        description="ids not transcribed (empty, no Hangul, or Latin letters / symbols "
        "the transcriber cannot read); absent from `results`",
    )
