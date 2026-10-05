"""/api/tts: proxy to the Qwen3-TTS service (no network — httpx is stubbed)."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services import tts_client


class _Settings:
    tts_api_url = "https://tts.example"
    tts_api_key = "secret"
    tts_timeout_s = 5
    tts_languages = {"ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean"}
    tts_speakers = {"ja": "Ono_Anna", "en": "", "zh": "", "ko": ""}


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _stub_httpx(monkeypatch, response, calls):
    import httpx

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, data=None, headers=None):
            calls.append({"url": url, "data": data, "headers": headers})
            return response

    monkeypatch.setattr(httpx, "AsyncClient", Client)


@pytest.fixture(autouse=True)
def _clear_cache():
    tts_client.clear_cache()
    yield
    tts_client.clear_cache()


def test_generate_speech_sends_key_language_and_speaker(monkeypatch):
    calls = []
    _stub_httpx(monkeypatch, _Response(200, {"status": "done", "audio_url": "/media/audio/a.wav", "duration_seconds": 1.5}), calls)
    audio = asyncio.run(tts_client.generate_speech("はしで", "ja", _Settings()))
    assert audio.audio_url == "https://tts.example/media/audio/a.wav"
    assert audio.duration_seconds == 1.5
    assert calls[0]["url"] == "https://tts.example/api/v1/generate"
    assert calls[0]["headers"]["api-key"] == "secret"
    assert calls[0]["data"] == {"text": "はしで", "wait": "true", "language": "Japanese", "speaker": "Ono_Anna"}
    # second call for the same sentence is served from the cache
    asyncio.run(tts_client.generate_speech("はしで", "ja", _Settings()))
    assert len(calls) == 1


def test_generate_speech_requires_api_key():
    settings = _Settings()
    settings.tts_api_key = ""
    with pytest.raises(tts_client.TtsUnavailableError):
        asyncio.run(tts_client.generate_speech("hello", "en", settings))


def test_generate_speech_reports_job_error(monkeypatch):
    _stub_httpx(monkeypatch, _Response(200, {"status": "failed", "audio_url": None, "error": "boom"}), [])
    with pytest.raises(tts_client.TtsFailedError):
        asyncio.run(tts_client.generate_speech("hello", "en", _Settings()))


def test_tts_route_validates_and_returns_audio_url(monkeypatch):
    calls = []
    _stub_httpx(monkeypatch, _Response(200, {"status": "done", "audio_url": "https://cdn.example/x.wav"}), calls)
    app.dependency_overrides[get_settings] = lambda: _Settings()
    try:
        client = TestClient(app)
        assert client.post("/api/tts", data={"text": "  ", "lang": "en"}).status_code == 400
        assert client.post("/api/tts", data={"text": "hi", "lang": "fr"}).status_code == 400
        ok = client.post("/api/tts", data={"text": "How are you?", "lang": "en"})
        assert ok.status_code == 200
        assert ok.json()["audio_url"] == "https://cdn.example/x.wav"
        assert "secret" not in ok.text
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_tts_route_503_without_key():
    settings = _Settings()
    settings.tts_api_key = ""
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        response = TestClient(app).post("/api/tts", data={"text": "hello", "lang": "en"})
        assert response.status_code == 503
    finally:
        app.dependency_overrides.pop(get_settings, None)
