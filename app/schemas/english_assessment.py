"""`reading_assessment` / `gopt_assessment` on the English /evaluate response
(built by app/services/english_assessment.py). Additive: the legacy fields
of EvaluateResponse are unchanged."""

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.pronunciation import EvaluateResponse


class CriterionResult(BaseModel):
    score: float | None = Field(
        default=None, description="0-100; null unless status is scored / estimated",
    )
    weight: float
    status: str = Field(
        description='"scored" | "estimated" (proxy metric, counted, not calibrated) | '
        '"gated" (words misread, not scored) | "insufficient_evidence" | "not_assessed"',
    )
    metric: str | None = None
    reason: str | None = None
    detail: dict[str, Any] | None = None


class ContentGate(BaseModel):
    passed: bool = Field(
        description="word_coverage >= min_word_coverage; False: stress / intonation / "
        "fluency / rhythm are gated",
    )
    content_score: float | None
    word_coverage: float | None = Field(description="Share of target words read right, 0-1")
    min_word_coverage: float


class WordResult(BaseModel):
    index: int
    word: str
    recognized: str | None = Field(default=None, description="Heard letters this word claimed")
    status: str = Field(description='"correct" | "mispronounced" | "substituted" | "omitted"')
    issue: str | None = None
    issues: list[str] = Field(default_factory=list)
    confidence: float | None = Field(
        default=None,
        description="How sure the error call is (0-1), not a pronunciation score",
    )
    similarity: float | None = None
    phoneme_score: float | None = Field(
        default=None, description="0-100 forced-alignment letter posterior (estimated)",
    )
    start: float | None = Field(default=None, description="Seconds; only for words read right")
    end: float | None = None
    syllables: list[str] | None = None
    expected_stress: int | None = None
    heard_stress: int | None = None
    stress_ok: bool | None = None


class InsertedWord(BaseModel):
    after_word: int = Field(description="Index of the target word before it, -1 at the start")
    text: str


class Improvement(BaseModel):
    priority: str = Field(description='"high" | "medium" | "low"')
    category: str
    code: str
    message: str
    words: list[str] = Field(default_factory=list)
    evidence: str | None = None


class ReadingAssessment(BaseModel):
    language: str = "en-US"
    reference_text: str
    recognized_text: str
    status: str = Field(
        description='"complete" | "partial_assessment" | "content_mismatch" | '
        '"insufficient_evidence" | "no_voice"',
    )
    overall_score: float | None = Field(
        default=None,
        description="Weighted mean of the measured criteria; null when content or "
        "phoneme accuracy is missing",
    )
    weights_used: float = Field(description="Sum of the weights behind overall_score")
    content_gate: ContentGate
    criteria: dict[str, CriterionResult]
    word_results: list[WordResult] = Field(default_factory=list)
    inserted: list[InsertedWord] = Field(
        default_factory=list, description="Extra words not in the text (WER insertions)",
    )
    repeated: list[InsertedWord] = Field(
        default_factory=list, description="Target words said again (fluency, not WER)",
    )
    improvements: list[Improvement] = Field(default_factory=list)
    audio_analysis: dict[str, Any] = Field(default_factory=dict)
    legacy_score: float | None = Field(
        default=None, description="The legacy `score` of this response, for comparison",
    )
    scoring_version: str


class GoptWordResult(BaseModel):
    index: int
    word: str
    read_ok: bool = Field(description="False: the transcript says this word was not read")
    phones: list[str] = Field(default_factory=list)
    accuracy: float | None = Field(default=None, description="0 when read_ok is False")
    stress: float | None = None
    total: float | None = None
    raw: dict[str, float | None] | None = Field(
        default=None, description="GOPT word heads before the read_ok adjustment",
    )
    phone_scores: list[float] = Field(default_factory=list)


class GoptAssessment(BaseModel):
    status: str = Field(
        description='"scored" | "content_mismatch" | "insufficient_evidence" | '
        '"unavailable" | "no_voice"',
    )
    model: str
    heads: dict[str, float] | None = Field(
        default=None, description="GOPT utterance heads, 0-100 (accuracy, completeness, "
        "fluency, prosodic, total)",
    )
    head_reliable: dict[str, bool] | None = Field(
        default=None, description="False: head collapsed (< 8) on out-of-domain features",
    )
    average: float | None = Field(
        default=None, description="Plain mean of accuracy, completeness, fluency, prosodic",
    )
    adjusted_heads: dict[str, float | None] | None = Field(
        default=None,
        description="Heads x min(word coverage, content accuracy); fluency / prosodic "
        "null when the content gate fails; collapsed heads null",
    )
    adjusted_average: float | None = Field(
        default=None, description="Mean of the non-null adjusted heads, at most `average`",
    )
    content_gate: ContentGate
    feature_source: str | None = Field(
        default=None, description='"forced_alignment" | "proportional" phone windows',
    )
    phone_count: int = 0
    truncated: bool = Field(default=False, description="GOPT sees 50 phones at most")
    words: list[GoptWordResult] = Field(default_factory=list)
    legacy_score: float | None = None


class AsrAgreement(BaseModel):
    same_verdict_share: float = Field(description="Share of target words both ASRs judged alike")
    both_read: list[str] = Field(default_factory=list)
    both_wrong: list[str] = Field(default_factory=list, description="Strong evidence of a real error")
    wrong_only_by_wav2vec2: list[str] = Field(default_factory=list)
    wrong_only_by_qwen3_asr: list[str] = Field(default_factory=list)


class OllamaAssessment(BaseModel):
    status: str = Field(
        description='reading statuses ("complete" | "partial_assessment" | '
        '"content_mismatch" | "insufficient_evidence") or "no_speech" | "disabled" | '
        '"unavailable" (Ollama down / model not pulled) | "unsupported" (server cannot '
        'take audio) | "invalid_output" | "no_voice" | "skipped"',
    )
    model: str | None = Field(default=None, description="EN_OLLAMA_ASR_MODEL")
    language: str | None = Field(default=None, description="Language Qwen3-ASR detected")
    language_mismatch: bool = False
    transcript: str | None = Field(default=None, description="Qwen3-ASR transcript as returned")
    recognized_text: str | None = Field(default=None, description="Transcript as scored")
    overall_score: float | None = None
    weights_used: float | None = None
    content_gate: ContentGate | None = None
    criteria: dict[str, CriterionResult] | None = None
    word_results: list[WordResult] = Field(default_factory=list)
    inserted: list[InsertedWord] = Field(default_factory=list)
    repeated: list[InsertedWord] = Field(default_factory=list)
    improvements: list[Improvement] = Field(default_factory=list)
    confidence_source: str | None = Field(
        default=None,
        description='Phoneme accuracy from "qwen3_asr_logprobs" or "forced_alignment"',
    )
    agreement: AsrAgreement | None = Field(
        default=None, description="wav2vec2 vs Qwen3-ASR verdict per target word",
    )
    comparison: dict[str, float | None] | None = Field(
        default=None,
        description="Qwen3-ASR vs reading_assessment vs GOPT vs legacy, side by side",
    )
    latency_ms: float | None = None
    error: str | None = None
    scoring_version: str


class CoachComment(BaseModel):
    """The /coach comment ({assessment, suggestion, lang}), returned inside
    the evaluate response so a client makes one call."""

    assessment: str
    suggestion: str = ""
    lang: str


class EnglishEvaluateResponse(EvaluateResponse):
    """EvaluateResponse plus the English-only reports (null if they could
    not be built; the legacy fields never depend on them). /evaluate-v2
    returns this same shape too."""

    coach: CoachComment | None = Field(
        default=None,
        description="AI comment (same fields as POST /coach). /evaluate: only with "
        "coach=true; /evaluate-v2: always, unless feedback=false or Ollama failed",
    )
    coach_error: str | None = Field(default=None, description="Why `coach` is null when it was asked for")

    reading_assessment: ReadingAssessment | None = Field(
        default=None,
        description="en-reading-v1: six weighted criteria with a content gate, "
        "per-word results and improvements",
    )
    gopt_assessment: GoptAssessment | None = Field(
        default=None,
        description="English GOPT heads, their average, and the content-gated average",
    )
    ollama_assessment: OllamaAssessment | None = Field(
        default=None,
        description="The audio transcribed by Qwen3-ASR (Ollama, EN_OLLAMA_ASR_MODEL) "
        "and scored like reading_assessment, for comparison",
    )
