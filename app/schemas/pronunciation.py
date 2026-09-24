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


class MeasuredPitchItem(BaseModel):
    """F0 measured from the recording, one point per target mora."""

    mora: str
    semitone: float | None = None
    voiced: bool = False
    expected: str = Field(description='"H" | "L" from the reference accent pattern')


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
        description="Same as score: CER read-aloud match, 0-100",
    )
    pronunciation_score: float = Field(
        ge=0, le=100, description="Kana CER mixed with phoneme GOP when available",
    )
    fluency_score: float = Field(
        ge=0, le=100, description="Mora/sec plus Silero/energy VAD pauses",
    )
    rhythm_score: float = Field(
        ge=0, le=100,
        description="Mora-duration evenness plus sokuon/chouon and beat edits",
    )
    intonation_score: float | None = Field(
        default=None,
        description="F0 vs H/L, blended with PASQA MOS when configured. "
        "Null when neither signal is available.",
    )
    rhythm_measured: bool = Field(
        default=False, description="True when per-mora CTC windows were used",
    )
    intonation_measured: bool = Field(
        default=False, description="True when voiced F0 points were available",
    )
    aspect_method: str = Field(
        default="local-aspect",
        description="Which free heads ran, e.g. gop+vad+pasqa or cer+f0",
    )
    vad_method: str | None = Field(
        default=None,
        description='"silero" if Silero VAD ran, "energy" if the RMS fallback ran, '
        "null if fluency had no VAD at all",
    )
    pause_count: int = Field(
        default=0, description="In-utterance pauses from VAD (speech islands minus 1)"
    )
    speech_ratio: float | None = Field(
        default=None, description="Voiced/speech seconds divided by file length, 0-1"
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
        description="Dictionary H/L of the ASR-recognized text (same shape as "
        "/pitch-accent). Kept for the coach/API; the practice chart draws "
        "`measured_pitch` instead.",
    )
    measured_pitch: list[MeasuredPitchItem] = Field(
        default_factory=list,
        description="F0 measured from the WAV, one point per target mora "
        "(same order as /pitch-accent). This is the learner curve the "
        "practice page should draw -- not recognized_pitch_pattern.",
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
            overall_score=float(score.score),
            pronunciation_score=aspects.pronunciation_score,
            fluency_score=aspects.fluency_score,
            rhythm_score=aspects.rhythm_score,
            intonation_score=aspects.intonation_score,
            rhythm_measured=aspects.rhythm_measured,
            intonation_measured=aspects.intonation_measured,
            aspect_method=aspects.method,
            vad_method=aspects.vad_method,
            pause_count=aspects.pause_count,
            speech_ratio=aspects.speech_ratio,
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
            measured_pitch=[
                MeasuredPitchItem(
                    mora=str(p.get("mora", "")),
                    semitone=p.get("semitone"),
                    voiced=bool(p.get("voiced")),
                    expected=str(p.get("expected", "L")),
                )
                for p in (result.measured_pitch or [])
            ],
        )
