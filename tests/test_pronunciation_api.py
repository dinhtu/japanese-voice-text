"""API-layer tests with a stubbed ASR service (no model loading)."""

import io
import struct

import pytest
from fastapi.testclient import TestClient

from app.application.pronunciation.asr_service import get_asr_service
from app.constants.practice_texts import DEFAULT_PRACTICE_TEXT, PRACTICE_TEXTS
from app.main import app
from src.asr.inference import RecognitionResult

TARGET = "げんきょうもいちにちがんばりましょう"


class StubASRService:
    """Returns a fixed transcription instead of running the model."""

    def __init__(self, kana: str = "げんきょーもいちにちがんばりましょー"):
        self.kana = kana

    is_loaded = True

    def recognize(self, audio_path):
        return RecognitionResult(kana=self.kana, duration=6.34, inference_time=1.1)


def make_wav(seconds: float = 0.1, sample_rate: int = 16_000) -> bytes:
    """Minimal silent 16-bit mono WAV."""
    n = int(seconds * sample_rate)
    data = b"\x00\x00" * n
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", len(data),
    )
    return header + data


@pytest.fixture
def client():
    app.dependency_overrides[get_asr_service] = lambda: StubASRService()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def post(client, text=TARGET, filename="test.wav", content=None, content_type="audio/wav"):
    content = make_wav() if content is None else content
    return client.post(
        "/api/pronunciation/evaluate",
        data={"text": text},
        files={"audio": (filename, io.BytesIO(content), content_type)},
    )


def test_evaluate_returns_full_result(client):
    response = post(client)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["target_text"] == TARGET
    assert body["target_hiragana"] == "げんきょーもいちにちがんばりましょー"
    assert body["recognized_text"] == "げんきょーもいちにちがんばりましょー"
    assert body["score"] == 100
    assert body["cer"] == 0.0
    assert body["feedback"]["level"] == "excellent"


def test_empty_text_is_rejected(client):
    assert post(client, text="   ").status_code == 400


def test_non_japanese_target_is_rejected(client):
    assert post(client, text="!!!").status_code == 400


def test_generic_binary_content_type_is_accepted(client):
    """curl sends application/octet-stream for -F audio=@file.wav."""
    response = post(client, content_type="application/octet-stream")
    assert response.status_code == 200


def test_non_wav_extension_is_rejected(client):
    response = post(client, filename="test.mp3", content_type="audio/mpeg")
    assert response.status_code == 400
    assert "wav" in response.json()["detail"].lower()


def test_non_riff_content_is_rejected(client):
    response = post(client, content=b"this is not audio at all")
    assert response.status_code == 400


def test_empty_file_is_rejected(client):
    assert post(client, content=b"").status_code == 400


def test_missing_audio_field_is_unprocessable(client):
    response = client.post("/api/pronunciation/evaluate", data={"text": TARGET})
    assert response.status_code == 422


def test_health_endpoint(client):
    assert client.get("/health").json()["status"] == "ok"


def test_practice_page_renders_every_target(client):
    """The page and the API are served by the same app."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    body = response.text
    assert DEFAULT_PRACTICE_TEXT.text in body
    for item in PRACTICE_TEXTS:
        assert item.text in body


def test_static_assets_are_served(client):
    for path, content_type in [("/static/app.css", "text/css"), ("/static/app.js", "javascript")]:
        response = client.get(path)
        assert response.status_code == 200, path
        assert content_type in response.headers["content-type"]
