"""v2_assessment of the Chinese / Korean /evaluate-v2 (see
app/services/cjk_evaluate_v2.py): what produced the transcript and the timing."""

from pydantic import BaseModel, Field


class V2AsrInfo(BaseModel):
    engine: str = Field(default="qwen3-asr", description="ASR that wrote the scored transcript")
    model: str = Field(description="Ollama model (EN_OLLAMA_ASR_MODEL)")
    language: str = Field(description='Language Qwen3-ASR was held to: "Chinese" | "Korean"')
    status: str = Field(description='"ok" | "skipped" (no voice: Ollama not called)')
    transcript: str | None = Field(default=None, description="Raw Qwen3-ASR transcript (the one scored)")
    detected_language: str | None = Field(default=None, description="Language of the reply, after forcing")
    calls: int = Field(default=0, description="Ollama /api/chat calls")
    latency_ms: float | None = None


class V2AlignmentInfo(BaseModel):
    model: str = Field(description="Local CTC model used for the syllable windows")
    method: str | None = Field(
        default=None,
        description='"ctc_forced_alignment" | "even_split" (Korean without CTC windows) | null (no timing)',
    )
    used_for_timing: bool = Field(description="The CTC forced alignment gave the per-syllable windows")
    transcript: str | None = Field(default=None, description="The CTC model's own transcript: not scored")
    inference_ms: float | None = None


class CjkV2Assessment(BaseModel):
    scoring_version: str = Field(description='"zh-reading-v2" | "ko-reading-v2"')
    status: str = Field(description='"scored" | "no_speech" (Qwen3-ASR heard no words) | "no_voice"')
    asr: V2AsrInfo
    alignment: V2AlignmentInfo
