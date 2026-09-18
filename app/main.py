"""FastAPI app for Japanese pronunciation evaluation (MVP).

Serves both the JSON API and the practice web page from one process.

Run:
    python run.py
    # or, with auto-reload: uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app.api import pages, routes
from app.core.config import STATIC_DIR, get_settings
from app.services.asr_service import ASRService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.eager_load:
        try:
            ASRService(settings).load()
        except Exception as e:  # noqa: BLE001 — keep serving /health for diagnosis
            logger.error("ASR model failed to load at startup: %s", e)
    yield
    logger.info("Shutting down pronunciation API")


app = FastAPI(
    title="Japanese Pronunciation Evaluation API",
    description=(
        "MVP: transcribes a WAV recording with the hiragana ASR model and scores it "
        "against a target Japanese text using normalized-reading edit distance. "
        "The practice page is served at /."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Behind nginx/Caddy: rebuild scheme and host from X-Forwarded-*, otherwise the
# app thinks it is on http://127.0.0.1:<port> and hands that out in redirects.
app.add_middleware(
    ProxyHeadersMiddleware,
    trusted_hosts=settings.forwarded_allow_ips,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(routes.router, prefix="/api/pronunciation", tags=["pronunciation"])
app.include_router(pages.router, tags=["web"])


@app.get("/health", tags=["health"])
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": ASRService(settings).is_loaded,
        "checkpoint": str(settings.checkpoint),
    }
