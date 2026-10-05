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
    pattern: list[MoraPitchItem] = Field(description="Primary reference pitch-accent pattern")
    patterns: list[list[MoraPitchItem]] = Field(
        default_factory=list,
        description="All accepted pitch-accent patterns (primary first, then alternatives)",
    )

    @classmethod
    def from_result(
        cls,
        text: str,
        moras: list[MoraPitch],
        all_patterns: list[list[MoraPitch]] | None = None,
    ) -> "PitchAccentResponse":
        primary = [
            MoraPitchItem(mora=m.mora, pitch=m.pitch, phrase=m.phrase) for m in moras
        ]
        if all_patterns:
            pats = [
                [MoraPitchItem(mora=m.mora, pitch=m.pitch, phrase=m.phrase) for m in p]
                for p in all_patterns
            ]
        else:
            pats = [primary]
        return cls(
            text=text,
            reading="".join(m.mora for m in moras),
            pattern=primary,
            patterns=pats,
        )
