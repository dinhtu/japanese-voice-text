"""Response schema for the learner pitch-contour endpoint."""

from pydantic import BaseModel, Field


class PitchContourPoint(BaseModel):
    semitone: float | None = Field(
        default=None,
        description=(
            "Pitch in semitones relative to this recording's own median "
            "voiced pitch; null where this mora's time slice had no voiced "
            "speech."
        ),
    )
    voiced: bool


class PitchContourResponse(BaseModel):
    success: bool = True
    duration: float = Field(description="Audio length in seconds")
    points: list[PitchContourPoint] = Field(
        description="One point per mora of the submitted text, same order as /pitch-accent"
    )
