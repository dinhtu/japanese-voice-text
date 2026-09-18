"""Runtime settings for the pronunciation API (environment-driven)."""

import os
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT_DIR / "static"
TEMPLATES_DIR = ROOT_DIR / "templates"

DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "checkpoints" / "best-medium-ep5-inference.pt"
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


class Settings:
    """Config read once from the environment.

    Environment variables:
        ASR_CHECKPOINT     Path to the .pt checkpoint.
        ASR_PRETRAINED     Base encoder id (default: taken from the checkpoint).
        ASR_INTER_CTC_LAYER
        ASR_DEVICE         cuda | mps | cpu (default: auto-detect).
        ASR_FP16           1 to enable FP16 inference.
        ASR_EAGER_LOAD     1 to load the model at startup instead of first request.
        MAX_AUDIO_MB       Upload size limit (default 25).
        CORS_ORIGINS       Comma-separated origins.
    """

    def __init__(self) -> None:
        self.checkpoint = Path(os.getenv("ASR_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
        self.pretrained = os.getenv("ASR_PRETRAINED") or None
        layer = os.getenv("ASR_INTER_CTC_LAYER")
        self.inter_ctc_layer = int(layer) if layer else None
        self.device = os.getenv("ASR_DEVICE") or None
        self.fp16 = os.getenv("ASR_FP16", "0") == "1"
        self.eager_load = os.getenv("ASR_EAGER_LOAD", "1") == "1"
        self.max_audio_bytes = int(float(os.getenv("MAX_AUDIO_MB", "25")) * 1024 * 1024)

        origins = os.getenv("CORS_ORIGINS", "").strip()
        self.cors_origins = (
            [o.strip() for o in origins.split(",") if o.strip()]
            if origins
            else list(DEFAULT_CORS_ORIGINS)
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
