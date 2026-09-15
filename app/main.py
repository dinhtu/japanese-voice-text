"""FastAPI app for Japanese pronunciation evaluation (MVP).

Run:
    uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.application.pronunciation.asr_service import ASRService
from app.core.config import get_settings
from app.interface.api import pronunciation_router

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
        "against a target Japanese text using normalized-reading edit distance."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    pronunciation_router.router,
    prefix="/api/pronunciation",
    tags=["pronunciation"],
)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": ASRService(settings).is_loaded,
        "checkpoint": str(settings.checkpoint),
    }
