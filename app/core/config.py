"""Runtime settings for the pronunciation API (environment-driven)."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT_DIR / "static"
TEMPLATES_DIR = ROOT_DIR / "templates"

# Real environment variables win over the file (systemd / Docker / CI).
load_dotenv(ROOT_DIR / ".env", override=False)

DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "checkpoints" / "best-medium-ep5-inference.pt"
DEFAULT_PASQA_CHECKPOINT = ROOT_DIR / "models" / "pasqa" / "checkpoint-100000steps.pkl"
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


class Settings:
    """Config read once from the environment.

    Environment variables:
        APP_HOST / APP_PORT   Bind address for `python run.py`.
        PUBLIC_BASE_URL       Public origin the browser sees (e.g. behind nginx).
                              Empty = emit same-origin relative URLs, which is
                              what you want for a normal reverse proxy.
        FORWARDED_ALLOW_IPS   Proxy IPs whose X-Forwarded-* headers are trusted.
        ASR_CHECKPOINT     Path to the .pt checkpoint.
        ASR_PRETRAINED     Base encoder id (default: taken from the checkpoint).
        ASR_INTER_CTC_LAYER
        ASR_DEVICE         cuda | mps | cpu (default: auto-detect).
        ASR_FP16           1 to enable FP16 inference.
        ASR_EAGER_LOAD     1 to load the model at startup instead of first request.
        MAX_AUDIO_MB       Upload size limit (default 25).
        CORS_ORIGINS       Comma-separated origins.
        OLLAMA_HOST        Local Ollama server URL (default http://localhost:11434).
        OLLAMA_MODEL       Model tag to use for /coach, e.g. "qwen3:8b" (must
                           already be pulled: `ollama pull qwen3:8b`).
        OLLAMA_TIMEOUT_S   Request timeout in seconds (default 30).
        OLLAMA_TEMPERATURE Sampling temperature for /coach's generated text
                           (default 0.4 -- fairly grounded, not too random).
        PASQA_CHECKPOINT   PASQA .pkl. Empty = use models/pasqa/checkpoint-100000steps.pkl
                           when that file exists, else skip PASQA.
        PASQA_DEVICE       cpu (default) or cuda -- keep cpu on a 12GB box.
        EN_ASR_MODEL       HuggingFace id for /en (default facebook/wav2vec2-base-960h).
        EN_ASR_DEVICE      cuda | mps | cpu (default: same auto-detect as ASR).
        GOPT_CHECKPOINT    Official MIT GOPT .pth for /en. Empty = models/gopt/best_audio_model.pth
                           (auto-downloaded on first English evaluate, ~121KB).
        GOPT_DEVICE        cpu (default) — tiny transformer, keep off the 3060.
        GOPT_ENABLED       1 to run GOPT on /en (default), 0 to skip.
    """

    def __init__(self) -> None:
        self.app_host = os.getenv("APP_HOST", "0.0.0.0")
        self.app_port = int(os.getenv("APP_PORT", "8000"))
        # No trailing slash, so f"{base}/static/..." is always well formed.
        self.public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
        self.forwarded_allow_ips = os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1").strip()

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
        # The public domain is always allowed to call its own API.
        if self.public_base_url and self.public_base_url not in self.cors_origins:
            self.cors_origins.append(self.public_base_url)

        # /coach: turns app.services.prosody_issues / mora_diff / pitch
        # findings into a natural-language Vietnamese coaching comment via
        # a locally-run Ollama model. Never sent to a third party -- the
        # request stays on this machine (or wherever OLLAMA_HOST points).
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b").strip()
        self.ollama_timeout_s = float(os.getenv("OLLAMA_TIMEOUT_S", "30"))
        self.ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.4"))

        # PASQA: empty env uses the downloaded default if present.
        pasqa = os.getenv("PASQA_CHECKPOINT", "").strip()
        if pasqa:
            self.pasqa_checkpoint = Path(pasqa)
        elif DEFAULT_PASQA_CHECKPOINT.is_file():
            self.pasqa_checkpoint = DEFAULT_PASQA_CHECKPOINT
        else:
            self.pasqa_checkpoint = None
        self.pasqa_device = os.getenv("PASQA_DEVICE", "cpu").strip() or "cpu"
        self.en_asr_model = os.getenv(
            "EN_ASR_MODEL", "facebook/wav2vec2-base-960h"
        ).strip() or "facebook/wav2vec2-base-960h"
        self.en_asr_device = os.getenv("EN_ASR_DEVICE") or None
        self.zh_asr_model = os.getenv(
            "ZH_ASR_MODEL", "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        ).strip() or "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        self.zh_asr_device = os.getenv("ZH_ASR_DEVICE") or None

        self.gopt_checkpoint = Path(
            os.getenv("GOPT_CHECKPOINT", "").strip()
            or (ROOT_DIR / "models" / "gopt" / "best_audio_model.pth")
        )
        self.gopt_device = os.getenv("GOPT_DEVICE", "cpu").strip() or "cpu"
        self.gopt_enabled = os.getenv("GOPT_ENABLED", "1") == "1"

    def asset_url(self, path: str) -> str:
        """URL for a file in /static.

        Relative by default: the page is same-origin with the API, so the
        browser resolves it against whatever domain it loaded the page from.
        Set PUBLIC_BASE_URL only when assets must be absolute (CDN, embedding
        the page on another host).
        """
        return f"{self.public_base_url}/static/{path.lstrip('/')}"

    @property
    def api_base_url(self) -> str:
        """Origin the page calls for /api/... ("" = same origin)."""
        return self.public_base_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
