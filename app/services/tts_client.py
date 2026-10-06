"""Client for the ComfyUIVoice TTS service used by the "Nghe mẫu" button.

POST {TTS_API_URL}/tts with JSON {text, lang, voice, format?, speed?}.
Voice names come from GET {TTS_API_URL}/voices (asuka, emma, vivian, sohee...).
The 200 response is JSON whose schema the service does not publish, so the
audio location is read from the usual field names (url / audio_url / file...);
a raw audio body is also accepted and passed on as a data: URL.
"""

from __future__ import annotations

import base64
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit, urlunsplit

if TYPE_CHECKING:
    from app.core.config import Settings

SUPPORTED_TTS_LANGS = ("ja", "en", "zh", "ko")

# Same sentence + voice -> reuse the generated file for a while instead of
# re-running the model on every click.
_CACHE_TTL_S = 30 * 60
_CACHE_MAX = 200
_cache: "OrderedDict[tuple[str, str, str], tuple[float, TtsAudio]]" = OrderedDict()

# Where the audio URL may live in the JSON answer, most specific first.
_URL_KEYS = ("audio_url", "url", "file_url", "download_url", "src", "path")
_FILE_KEYS = ("file", "filename", "file_name", "name")


class TtsUnavailableError(RuntimeError):
    """Not configured, or the service could not be reached (HTTP 503)."""


class TtsFailedError(RuntimeError):
    """The service answered but did not produce audio (HTTP 502)."""


@dataclass(frozen=True)
class TtsAudio:
    audio_url: str
    duration_seconds: float | None = None


def clear_cache() -> None:
    _cache.clear()


# Suffixes people paste along with the host (the Swagger link, the endpoint
# itself...). Stripped so `{base}/tts` hits the real route.
_URL_SUFFIXES = ("/api/v1/generate", "/api/v1", "/tts", "/openapi.json", "/redoc", "/docs")


def normalize_base_url(url: str) -> str:
    """'https://host/docs#/default/tts_tts_post' -> 'https://host'."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    changed = True
    while changed:
        changed = False
        for suffix in _URL_SUFFIXES:
            if path.endswith(suffix):
                path = path[: -len(suffix)].rstrip("/")
                changed = True
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


def _audio_location(job: Any, base: str, audio_path: str = "audio") -> str:
    """Absolute audio URL from the /tts JSON answer ("" if none found)."""
    if isinstance(job, str):
        job = {"url": job}
    if not isinstance(job, dict):
        return ""
    for nested in ("data", "result", "audio"):
        if isinstance(job.get(nested), dict):
            found = _audio_location(job[nested], base, audio_path)
            if found:
                return found
    for key in _URL_KEYS:
        value = job.get(key)
        if isinstance(value, str) and value.strip():
            return urljoin(f"{base}/", value.strip())
    for key in _FILE_KEYS:
        value = job.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            # A bare file name ("3f2c…e1.mp3"): served under TTS_AUDIO_PATH.
            prefix = f"{audio_path}/" if audio_path else ""
            return urljoin(f"{base}/", name if "/" in name else f"{prefix}{name}")
    return ""


def _duration(job: Any) -> float | None:
    if not isinstance(job, dict):
        return None
    for key in ("duration_seconds", "duration", "seconds"):
        value = job.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


async def generate_speech(text: str, lang: str, settings: "Settings") -> TtsAudio:
    if lang not in SUPPORTED_TTS_LANGS:
        raise ValueError(f"Unsupported lang '{lang}'. Supported: {', '.join(SUPPORTED_TTS_LANGS)}")
    voice = settings.tts_voices.get(lang, "")
    if not voice:
        raise TtsUnavailableError(f"Chưa cấu hình giọng đọc TTS_VOICE_{lang.upper()} trong .env.")

    key = (text, lang, voice)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL_S:
        _cache.move_to_end(key)
        return hit[1]

    try:
        import httpx
    except ImportError as e:  # pragma: no cover - httpx is in requirements.txt
        raise TtsUnavailableError("Thiếu package 'httpx'. Cài bằng: pip install httpx") from e

    payload: dict[str, Any] = {"text": text, "lang": lang, "voice": voice}
    if settings.tts_format:
        payload["format"] = settings.tts_format
    if settings.tts_speed is not None:
        payload["speed"] = settings.tts_speed
    headers = {"accept": "application/json"}
    if settings.tts_api_key:
        headers["x-api-key"] = settings.tts_api_key

    base = normalize_base_url(settings.tts_api_url)
    endpoint = f"{base}/tts"
    try:
        async with httpx.AsyncClient(timeout=settings.tts_timeout_s) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
    except httpx.TimeoutException as e:
        raise TtsUnavailableError("Dịch vụ tạo audio phản hồi quá lâu. Hãy thử lại.") from e
    except httpx.HTTPError as e:
        raise TtsUnavailableError(f"Không kết nối được dịch vụ tạo audio ({e}).") from e

    if response.status_code in (401, 403):
        raise TtsUnavailableError("Dịch vụ tạo audio từ chối truy cập (kiểm tra TTS_API_KEY).")
    if response.status_code == 404:
        raise TtsFailedError(
            f"Không tìm thấy API tạo audio tại {endpoint} (HTTP 404). "
            "Kiểm tra TTS_API_URL trong .env: chỉ ghi gốc domain, "
            "ví dụ https://text-to-audio.commude-vietnam.work."
        )
    if response.status_code >= 400:
        raise TtsFailedError(
            f"Dịch vụ tạo audio báo lỗi HTTP {response.status_code} ({endpoint}): {response.text[:300]}"
        )

    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type.startswith("audio/"):
        # Audio returned directly instead of a link: hand it to the page inline.
        encoded = base64.b64encode(response.content).decode("ascii")
        audio = TtsAudio(audio_url=f"data:{content_type};base64,{encoded}")
    else:
        try:
            job = response.json()
        except ValueError as e:
            raise TtsFailedError("Dịch vụ tạo audio trả về dữ liệu không hợp lệ.") from e
        if isinstance(job, dict) and job.get("error"):
            raise TtsFailedError(f"Không tạo được audio: {job.get('error')}")
        audio_url = _audio_location(job, base, getattr(settings, "tts_audio_path", "audio"))
        if not audio_url:
            keys = ", ".join(sorted(job)) if isinstance(job, dict) else type(job).__name__
            raise TtsFailedError(f"Dịch vụ tạo audio không trả về đường dẫn file (các field nhận được: {keys}).")
        audio = TtsAudio(audio_url=audio_url, duration_seconds=_duration(job))

    _cache[key] = (time.monotonic(), audio)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return audio
