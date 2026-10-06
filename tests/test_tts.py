"""/api/tts: proxy to the ComfyUIVoice service (no network — httpx is stubbed)."""

import asyncio

import pytest

from app.services import tts_client


class _Settings:
    tts_api_url = "https://tts.example"
    tts_api_key = ""
    tts_timeout_s = 5
    tts_voices = {"ja": "asuka", "en": "emma", "zh": "vivian", "ko": "sohee"}
    tts_format = "mp3"
    tts_speed = None
    tts_audio_path = "audio"


class _Response:
    def __init__(self, status_code, payload=None, content=b"", content_type="application/json"):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = str(payload)
        self.headers = {"content-type": content_type}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
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

        async def post(self, url, json=None, headers=None):
            calls.append({"url": url, "json": json, "headers": headers})
            return response

    monkeypatch.setattr(httpx, "AsyncClient", Client)


@pytest.fixture(autouse=True)
def _clear_cache():
    tts_client.clear_cache()
    yield
    tts_client.clear_cache()


def _run(text="はしで", lang="ja", settings=None):
    return asyncio.run(tts_client.generate_speech(text, lang, settings or _Settings()))


def test_posts_json_with_lang_voice_and_format(monkeypatch):
    calls = []
    _stub_httpx(monkeypatch, _Response(200, {"url": "/audio/a.mp3", "duration": 1.5}), calls)
    audio = _run()
    assert audio.audio_url == "https://tts.example/audio/a.mp3"
    assert audio.duration_seconds == 1.5
    assert calls[0]["url"] == "https://tts.example/tts"
    assert calls[0]["json"] == {"text": "はしで", "lang": "ja", "voice": "asuka", "format": "mp3"}
    assert "x-api-key" not in calls[0]["headers"]
    _run()  # same sentence again -> cache
    assert len(calls) == 1


def test_sends_speed_and_optional_key(monkeypatch):
    calls = []
    _stub_httpx(monkeypatch, _Response(200, {"audio_url": "https://cdn.example/x.mp3"}), calls)
    settings = _Settings()
    settings.tts_speed = 0.9
    settings.tts_api_key = "k"
    assert _run("hello", "en", settings).audio_url == "https://cdn.example/x.mp3"
    assert calls[0]["json"]["speed"] == 0.9
    assert calls[0]["json"]["voice"] == "emma"
    assert calls[0]["headers"]["x-api-key"] == "k"


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"file": "3f2c.mp3"}, "https://tts.example/audio/3f2c.mp3"),
        ({"data": {"url": "/out/b.mp3"}}, "https://tts.example/out/b.mp3"),
        ("https://cdn.example/c.mp3", "https://cdn.example/c.mp3"),
    ],
)
def test_reads_audio_location_variants(monkeypatch, payload, expected):
    _stub_httpx(monkeypatch, _Response(200, payload), [])
    assert _run().audio_url == expected


def test_audio_body_becomes_data_url(monkeypatch):
    _stub_httpx(monkeypatch, _Response(200, None, content=b"ID3abc", content_type="audio/mpeg"), [])
    assert _run().audio_url.startswith("data:audio/mpeg;base64,")


def test_missing_location_lists_fields(monkeypatch):
    _stub_httpx(monkeypatch, _Response(200, {"ok": True, "job": "1"}), [])
    with pytest.raises(tts_client.TtsFailedError, match="job, ok"):
        _run()


def test_missing_voice_is_unavailable():
    settings = _Settings()
    settings.tts_voices = {"ja": ""}
    with pytest.raises(tts_client.TtsUnavailableError):
        _run(settings=settings)


def test_404_names_the_url(monkeypatch):
    _stub_httpx(monkeypatch, _Response(404, {"detail": "Not Found"}), [])
    settings = _Settings()
    settings.tts_api_url = "https://tts.example/docs#/default/tts_tts_post"
    with pytest.raises(tts_client.TtsFailedError, match="https://tts.example/tts"):
        _run(settings=settings)


@pytest.mark.parametrize(
    "configured",
    [
        "https://tts.example",
        "https://tts.example/",
        "https://tts.example/docs",
        "https://tts.example/docs#/default/tts_tts_post",
        "https://tts.example/tts",
        "https://tts.example/api/v1/generate",
        "https://tts.example/openapi.json",
    ],
)
def test_normalize_base_url_strips_pasted_suffixes(configured):
    assert tts_client.normalize_base_url(configured) == "https://tts.example"


def test_tts_route_validates_and_returns_audio_url(monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import app

    _stub_httpx(monkeypatch, _Response(200, {"url": "https://cdn.example/x.mp3"}), [])
    app.dependency_overrides[get_settings] = lambda: _Settings()
    try:
        client = TestClient(app)
        assert client.post("/api/tts", data={"text": "  ", "lang": "en"}).status_code == 400
        assert client.post("/api/tts", data={"text": "hi", "lang": "fr"}).status_code == 400
        ok = client.post("/api/tts", data={"text": "How are you?", "lang": "en"})
        assert ok.status_code == 200
        assert ok.json()["audio_url"] == "https://cdn.example/x.mp3"
    finally:
        app.dependency_overrides.pop(get_settings, None)
