"""Response schemas for the pronunciation API."""

from pydantic import BaseModel, Field

from app.schemas.pitch_accent import MoraPitchItem
from app.services.mora_diff import mora_status
from app.services.pitch_accent import MoraPitch
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
    score: int = Field(ge=0, le=100, description="CER read-aloud match, 0-100")
    overall_score: float = Field(
        ge=0, le=100,
        description="Weighted mix of the four aspect scores (GOPT-style)",
    )
    pronunciation_score: float = Field(
        ge=0, le=100, description="Same as score: kana CER match",
    )
    fluency_score: float = Field(
        ge=0, le=100, description="Mora/sec vs a careful-reading pace band",
    )
    rhythm_score: float = Field(
        ge=0, le=100,
        description="Mora-duration evenness plus sokuon/chouon and beat edits",
    )
    intonation_score: float | None = Field(
        default=None,
        description="F0 direction vs the H/L pattern. Null when no voiced pitch.",
    )
    rhythm_measured: bool = Field(
        default=False, description="True when per-mora CTC windows were used",
    )
    intonation_measured: bool = Field(
        default=False, description="True when voiced F0 points were available",
    )
    aspect_method: str = Field(
        default="local-aspect",
        description="local-aspect = deterministic Japanese signals, not MIT GOPT",
    )
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
    recognized_pitch_pattern: list[MoraPitchItem] = Field(
        default_factory=list,
        description="Same H/L-per-mora pitch pattern shape as /pitch-accent's "
        "`pattern`, but computed from the ASR-recognized text instead of the "
        "target text -- i.e. the dictionary accent pattern for whatever the "
        "model actually heard, not a measurement of the recording's audio. "
        "This lets the frontend draw the learner's pitch with the exact same "
        "chart as the reference pattern instead of a different-shaped, "
        "audio-measured curve. Empty when the recognized text had no "
        "pronounceable content.",
    )

    @classmethod
    def from_result(
        cls,
        result: EvaluationResult,
        recognized_moras: list[MoraPitch] | None = None,
    ) -> "EvaluateResponse":
        score = result.score
        aspects = result.aspects
        return cls(
            target_text=result.target_text,
            target_hiragana=result.target_hiragana,
            recognized_text=result.recognized_text,
            recognized_hiragana=result.recognized_hiragana,
            score=score.score,
            overall_score=aspects.overall_score,
            pronunciation_score=aspects.pronunciation_score,
            fluency_score=aspects.fluency_score,
            rhythm_score=aspects.rhythm_score,
            intonation_score=aspects.intonation_score,
            rhythm_measured=aspects.rhythm_measured,
            intonation_measured=aspects.intonation_measured,
            aspect_method=aspects.method,
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
            recognized_pitch_pattern=[
                MoraPitchItem(mora=m.mora, pitch=m.pitch, phrase=m.phrase)
                for m in (recognized_moras or [])
            ],
        )
