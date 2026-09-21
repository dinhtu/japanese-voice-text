"""Response schemas for the pronunciation API."""

from pydantic import BaseModel, Field

from app.services.mora_diff import mora_status
from app.services.use_cases import EvaluationResult


class PronunciationErrorItem(BaseModel):
    """One character-level mismatch, for future phoneme-level feedback."""

    type: str = Field(description='"sub" | "del" | "ins"')
    target: str
    recognized: str
    position: int


class MoraStatusItem(BaseModel):
    """One mora of the target sentence, correct or not (see mora_diff.py)."""

    mora: str
    ok: bool = Field(description="True = matched the target, False = mispronounced")


class Feedback(BaseModel):
    level: str = Field(description="excellent | good | fair | poor | mismatch")
    message: str


class EvaluateResponse(BaseModel):
    success: bool = True
    target_text: str = Field(description="Target text exactly as submitted")
    target_hiragana: str = Field(description="Target normalized to a hiragana reading")
    recognized_text: str = Field(description="Raw kana from the ASR model")
    recognized_hiragana: str = Field(description="ASR output normalized the same way")
    score: int = Field(ge=0, le=100)
    cer: float = Field(description="Character error rate against the target reading")
    distance: int = Field(description="Levenshtein distance in characters")
    audio_duration: float = Field(description="Audio length in seconds")
    inference_ms: float
    feedback: Feedback
    errors: list[PronunciationErrorItem] = Field(default_factory=list)
    mora_status: list[MoraStatusItem] = Field(
        default_factory=list,
        description="Target sentence broken into morae, each flagged correct/incorrect",
    )

    @classmethod
    def from_result(cls, result: EvaluationResult) -> "EvaluateResponse":
        score = result.score
        return cls(
            target_text=result.target_text,
            target_hiragana=result.target_hiragana,
            recognized_text=result.recognized_text,
            recognized_hiragana=result.recognized_hiragana,
            score=score.score,
            cer=score.cer,
            distance=score.distance,
            audio_duration=result.audio_duration,
            inference_ms=result.inference_ms,
            feedback=Feedback(level=score.level, message=score.message),
            errors=[PronunciationErrorItem(**vars(e)) for e in score.errors],
            mora_status=[
                MoraStatusItem(mora=m.mora, ok=m.ok)
                for m in mora_status(result.target_hiragana, score.errors)
            ],
        )
