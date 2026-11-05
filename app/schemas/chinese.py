"""The Mandarin /evaluate and /evaluate-v2 responses: EvaluateResponse plus
a tone check per syllable and the optional coach comment (and, for v2, what
v2 did)."""

from pydantic import BaseModel, Field

from app.schemas.cjk_v2 import CjkV2Assessment
from app.schemas.coaching import CoachComment
from app.schemas.pronunciation import EvaluateResponse


class ToneCheckItem(BaseModel):
    """One target Hanzi: the tone to say and how the take's F0 fit it."""

    hanzi: str
    mora: str = Field(description='Pinyin without tone, e.g. "hao"')
    pinyin: str = Field(description='Pinyin of the spoken tone, e.g. "hǎo"')
    tone: int = Field(description="Dictionary tone 1-5 (5 = neutral)")
    surface_tone: int = Field(description="Tone after sandhi (3+3, 不, 一): the one checked")
    phrase_final: bool = Field(description="Last syllable before punctuation or the end")
    contour: list[int] = Field(description="Reference Chao contour (1-5); [2, 1, 1] = half third")
    heard_contour: list[float | None] = Field(
        description="Start / middle / end F0 of the syllable, semitones from the clip mean",
    )
    heard_tone: int | None = Field(default=None, description="Tone 1-4 the measured shape is closest to")
    ok: bool | None = Field(default=None, description="F0 shape fits surface_tone; null when not judged")
    status: str = Field(description='"ok" | "bad" | "misread" | "neutral" | "unmeasured"')


class ChineseEvaluateResponse(EvaluateResponse):
    tones: list[ToneCheckItem] = Field(
        default_factory=list,
        description="One tone check per target Hanzi; intonation_score is ok / judged",
    )
    coach: CoachComment | None = Field(
        default=None, description="AI comment (same fields as POST /coach); only with coach=true",
    )
    coach_error: str | None = Field(default=None, description="Why `coach` is null when it was asked for")


class ChineseEvaluateV2Response(ChineseEvaluateResponse):
    """POST /evaluate-v2: the /evaluate fields (values from the Qwen3-ASR
    transcript) plus what v2 did; `coach` comes with feedback=true."""

    v2_assessment: CjkV2Assessment | None = Field(
        default=None, description="ASR engine / model, raw transcript, alignment used for timing",
    )
