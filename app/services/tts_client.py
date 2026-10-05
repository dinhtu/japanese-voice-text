"""Client for the external Qwen3-TTS service used by the "Nghe mẫu" button.

POST {TTS_API_URL}/api/v1/generate (form fields text / language / speaker /
wait, header `api-key`) waits for the job and returns a JobResponse whose
`audio_url` points at /media/audio/<file> on the same service.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urljoin

if TYPE_CHECKING:
    from app.core.config import Settings

SUPPORTED_TTS_LANGS = ("ja", "en", "zh", "ko")

# Same sentence + language -> reuse the generated file for a while instead of
# re-running the 1.7B model on every click.
_CACHE_TTL_S = 30 * 60
_CACHE_MAX = 200
_cache: "OrderedDict[tuple[str, str, str], tuple[float, TtsAudio]]" = OrderedDict()


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


async def generate_speech(text: str, lang: str, settings: "Settings") -> TtsAudio:
    if lang not in SUPPORTED_TTS_LANGS:
        raise ValueError(f"Unsupported lang '{lang}'. Supported: {', '.join(SUPPORTED_TTS_LANGS)}")
    if not settings.tts_api_key:
        raise TtsUnavailableError("Chưa cấu hình TTS_API_KEY trong .env nên chưa tạo được audio mẫu.")

    language = settings.tts_languages.get(lang, "")
    speaker = settings.tts_speakers.get(lang, "")
    key = (text, language, speaker)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL_S:
        _cache.move_to_end(key)
        return hit[1]

    try:
        import httpx
    except ImportError as e:  # pragma: no cover - httpx is in requirements.txt
        raise TtsUnavailableError("Thiếu package 'httpx'. Cài bằng: pip install httpx") from e

    form = {"text": text, "wait": "true"}
    if language:
        form["language"] = language
    if speaker:
        form["speaker"] = speaker

    base = settings.tts_api_url
    try:
        async with httpx.AsyncClient(timeout=settings.tts_timeout_s) as client:
            response = await client.post(
                f"{base}/api/v1/generate",
                data=form,
                headers={"api-key": settings.tts_api_key, "accept": "application/json"},
            )
    except httpx.TimeoutException as e:
        raise TtsUnavailableError("Dịch vụ tạo audio phản hồi quá lâu. Hãy thử lại.") from e
    except httpx.HTTPError as e:
        raise TtsUnavailableError(f"Không kết nối được dịch vụ tạo audio ({e}).") from e

    if response.status_code in (401, 403):
        raise TtsUnavailableError("TTS_API_KEY không hợp lệ (dịch vụ tạo audio từ chối).")
    if response.status_code >= 400:
        raise TtsFailedError(f"Dịch vụ tạo audio báo lỗi HTTP {response.status_code}: {response.text[:300]}")

    try:
        job = response.json()
    except ValueError as e:
        raise TtsFailedError("Dịch vụ tạo audio trả về dữ liệu không hợp lệ.") from e

    audio_url = (job.get("audio_url") or "").strip()
    if job.get("error") or not audio_url:
        raise TtsFailedError(
            f"Không tạo được audio (status: {job.get('status')}, error: {job.get('error') or 'không có audio_url'})."
        )

    audio = TtsAudio(
        audio_url=urljoin(f"{base}/", audio_url),
        duration_seconds=job.get("duration_seconds"),
    )
    _cache[key] = (time.monotonic(), audio)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return audio
