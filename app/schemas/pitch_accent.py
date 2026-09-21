"""Response schema for the pitch-accent (reference intonation) endpoint."""

from pydantic import BaseModel, Field

from app.services.pitch_accent import MoraPitch


class MoraPitchItem(BaseModel):
    mora: str = Field(description="One mora, in hiragana (e.g. 'cho')")
    pitch: str = Field(description='"H" (high) or "L" (low)')
    phrase: int = Field(description="0-based accent phrase index")


class PitchAccentResponse(BaseModel):
    success: bool = True
    text: str = Field(description="Text exactly as submitted")
    reading: str = Field(description="Hiragana reading, mora-joined")
    pattern: list[MoraPitchItem]

    @classmethod
    def from_result(cls, text: str, moras: list[MoraPitch]) -> "PitchAccentResponse":
        return cls(
            text=text,
            reading="".join(m.mora for m in moras),
            pattern=[
                MoraPitchItem(mora=m.mora, pitch=m.pitch, phrase=m.phrase) for m in moras
            ],
        )
