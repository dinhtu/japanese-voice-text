"""Server-rendered pages (the practice UI lives in the same app as the API)."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.constants.english_texts import (
    DEFAULT_PRACTICE_TEXT as EN_DEFAULT,
    PRACTICE_TEXTS as EN_TEXTS,
)
from app.constants.practice_texts import DEFAULT_PRACTICE_TEXT, PRACTICE_TEXTS
from app.constants.chinese_texts import (
    DEFAULT_PRACTICE_TEXT as ZH_DEFAULT,
    PRACTICE_TEXTS as ZH_TEXTS,
)
from app.core.config import TEMPLATES_DIR, get_settings

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
settings = get_settings()

# Starlette's url_for() returns an absolute URL built from the request host, so
# behind a reverse proxy it leaks the internal address (http://127.0.0.1:8005).
# asset_url() honours PUBLIC_BASE_URL and falls back to a relative path.
templates.env.globals["asset_url"] = settings.asset_url


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def practice_page(request: Request) -> HTMLResponse:
    """Read-aloud practice screen."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "texts": PRACTICE_TEXTS,
            "default_text": DEFAULT_PRACTICE_TEXT,
            "api_base_url": settings.api_base_url,
        },
    )


@router.get("/en", response_class=HTMLResponse, include_in_schema=False)
def english_practice_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "en.html",
        {
            "texts": EN_TEXTS,
            "default_text": EN_DEFAULT,
            "api_base_url": settings.api_base_url,
        },
    )


@router.get("/zh", response_class=HTMLResponse, include_in_schema=False)
def chinese_practice_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "zh.html",
        {
            "texts": ZH_TEXTS,
            "default_text": ZH_DEFAULT,
            "api_base_url": settings.api_base_url,
        },
    )
