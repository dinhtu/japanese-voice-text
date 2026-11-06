"""The Korean /evaluate and /evaluate-v2 responses: EvaluateResponse plus
the optional coach comment (and, for v2, what v2 did)."""

from pydantic import Field

from app.schemas.cjk_v2 import CjkV2Assessment
from app.schemas.coaching import CoachComment
from app.schemas.pronunciation import EvaluateResponse


class KoreanEvaluateResponse(EvaluateResponse):
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
