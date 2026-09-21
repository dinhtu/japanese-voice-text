"""Response schema for the /coach endpoint (see app/services/coaching.py)."""

from pydantic import BaseModel, Field


class CoachResponse(BaseModel):
    success: bool = True
    comment: str = Field(
        description="Natural-language Vietnamese coaching comment, generated "
        "by a locally-run Ollama model strictly from this app's own "
        "measured facts -- see app/services/coaching.py"
    )
