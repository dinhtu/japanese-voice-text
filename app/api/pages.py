"""Server-rendered pages (the practice UI lives in the same app as the API)."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.constants.practice_texts import DEFAULT_PRACTICE_TEXT, PRACTICE_TEXTS
from app.core.config import TEMPLATES_DIR

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def practice_page(request: Request) -> HTMLResponse:
    """Read-aloud practice screen."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {"texts": PRACTICE_TEXTS, "default_text": DEFAULT_PRACTICE_TEXT},
    )
