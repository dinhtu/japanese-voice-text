"""POST /api/pronunciation-en/evaluate-v2 response (en-reading-v2), built by
app/services/english_evaluate_v2.py. Null scores mean "not enough
evidence", never 0."""

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.english_assessment import (
    ContentGate,
    CriterionResult,
    EnglishEvaluateResponse,
    Improvement,
)


class PronunciationEvidence(BaseModel):
    score: float | None = Field(default=None, description="0-100 from the rival margin")
    p_target: float | None = Field(default=None, description="Qwen3-ASR probability of the target word")
    rival: str | None = Field(default=None, description="Most likely other word / token at that spot")
    p_rival: float | None = None
    margin: float | None = Field(default=None, description="log P(target) - log P(rival), nats")


class WordResultV2(BaseModel):
    index: int
    text: str
    status: str = Field(description='"correct" | "mispronounced" | "substituted" | "omitted"')
    heard: str | None = None
    start_seconds: float | None = Field(default=None, description="From forced alignment; null if not read / no timing")
    end_seconds: float | None = None
    pronunciation: PronunciationEvidence | None = None
    syllables: list[str] | None = None
    expected_stress: int | None = None
    heard_stress: int | None = None
    stress_ok: bool | None = None
    phonemes: list[Any] = Field(default_factory=list, description="Empty: no phoneme model in Ollama")
    issues: list[str] = Field(default_factory=list)


class ErrorItem(BaseModel):
    type: str = Field(
        description='"omission" | "substitution" | "mispronunciation" | "word_stress" | '
        '"insertion" | "repetition" | "filler"',
    )
    index: int = Field(description="Target word index (for insertions: the word before, -1 = start)")
    word: str
    heard: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = Field(default=None, description="How sure the error call is, 0-1")
    expected: str | None = Field(default=None, description="Word stress: the right stress, e.g. TEA-cher")


class Feedback(BaseModel):
    status: str = Field(description='"generated" | "unavailable"')
    model: str | None = None
    assessment: str | None = None
    suggestion: str | None = None
    lang: str | None = None
    error: str | None = None
    latency_ms: float | None = None


class EvaluateV2Response(BaseModel):
    language: str = "en-US"
    reference_text: str
    recognized_text: str | None = Field(default=None, description="Qwen3-ASR transcript")
    detected_language: str | None = None
    language_mismatch: bool = False
    status: str = Field(
        description='"complete" | "partial_assessment" | "content_mismatch" | '
        '"language_mismatch" | "insufficient_evidence" | "no_speech" | "no_voice" | "asr_unavailable"',
    )
    overall_score: float | None = Field(
        default=None, description="0.25 C + 0.30 P + 0.15 W + 0.15 I + 0.15 F over the measured criteria",
    )
    weights_used: float = 0.0
    content_gate: ContentGate | None = None
    criteria: dict[str, CriterionResult] = Field(
        description="content_accuracy, pronunciation, word_stress, intonation, fluency",
    )
    word_results: list[WordResultV2] = Field(default_factory=list)
    errors: list[ErrorItem] = Field(default_factory=list)
    improvements: list[Improvement] = Field(default_factory=list)
    audio_analysis: dict[str, Any] = Field(default_factory=dict)
    asr: dict[str, Any] | None = Field(default=None, description="Qwen3-ASR model, calls, latency")
    alignment: dict[str, Any] | None = Field(default=None, description="Word-timing method")
    feedback: Feedback | None = Field(default=None, description="OLLAMA_MODEL explanation of the scores")
    scoring_version: str


class EnglishEvaluateV2Response(EnglishEvaluateResponse):
    """POST /evaluate-v2: the /evaluate fields (same names and types, values
    from en-reading-v2) plus `coach` and the full v2 report, so a client can
    switch between /evaluate and /evaluate-v2 without other changes."""

    v2_assessment: EvaluateV2Response | None = Field(
        default=None, description="Full en-reading-v2 report (criteria, word results, errors)",
    )
