"""API-layer tests with a stubbed ASR service (no model loading)."""

import io
import struct

import pytest
from fastapi.testclient import TestClient

from app.services.asr_service import get_asr_service
from app.constants.practice_texts import DEFAULT_PRACTICE_TEXT, PRACTICE_TEXTS
from app.main import app
from src.asr.inference import RecognitionResult

TARGET = "げんきょうもいちにちがんばりましょう"


class StubASRService:
    """Returns a fixed transcription instead of running the model."""

    def __init__(self, kana: str = "げんきょーもいちにちがんばりましょー"):
        self.kana = kana

    is_loaded = True

    def recognize(self, audio_path, with_timing=False, align_to=None):
        return RecognitionResult(kana=self.kana, duration=6.34, inference_time=1.1)


def _wav_header(n_bytes: int, sample_rate: int = 16_000) -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + n_bytes, b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", n_bytes,
    )


def make_wav(seconds: float = 0.1, sample_rate: int = 16_000) -> bytes:
    """Minimal silent 16-bit mono WAV."""
    n = int(seconds * sample_rate)
    data = b"\x00\x00" * n
    return _wav_header(len(data), sample_rate) + data


def make_tone_wav(freq_hz: float = 220.0, seconds: float = 1.0, sample_rate: int = 16_000) -> bytes:
    """16-bit mono sine — enough voiced energy for F0 extraction."""
    import math

    n = int(seconds * sample_rate)
    frames = bytearray()
    for i in range(n):
        sample = int(16000 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        frames += struct.pack("<h", sample)
    return _wav_header(len(frames), sample_rate) + bytes(frames)


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
    assert body["pronunciation_score"] == 100.0
    assert body["cer"] == 0.0
    assert body["feedback"]["level"] == "excellent"
    assert 0 <= body["overall_score"] <= 100
    assert 0 <= body["fluency_score"] <= 100
    assert 0 <= body["rhythm_score"] <= 100
    assert body["intonation_score"] is None or 0 <= body["intonation_score"] <= 100
    assert body["aspect_method"].startswith("cer")
    assert body["vad_method"] in (None, "silero", "energy")
    assert isinstance(body["pause_count"], int)
    assert body["speech_ratio"] is None or 0 <= body["speech_ratio"] <= 1
    assert body["rhythm_measured"] is False
    assert body["intonation_measured"] is False
    assert isinstance(body["measured_pitch"], list)
    for item in body["measured_pitch"]:
        assert {"mora", "semitone", "voiced", "expected"} <= set(item)
        assert item["expected"] in ("H", "L")


def test_evaluate_measured_pitch_is_f0_from_audio(client):
    """Learner curve must come from WAV F0, not dictionary H/L of ASR text."""
    response = post(client, content=make_tone_wav())
    assert response.status_code == 200
    pitch = response.json()["measured_pitch"]
    if not pitch:
        pytest.skip("pitch-accent / F0 path unavailable in this environment")
    for item in pitch:
        assert {"mora", "semitone", "voiced", "expected"} <= set(item)
        assert item["expected"] in ("H", "L")
        assert isinstance(item["mora"], str) and item["mora"]
    voiced = [p for p in pitch if p["voiced"] and p["semitone"] is not None]
    assert voiced, "a 220Hz tone should yield at least one voiced F0 point"


def test_evaluate_returns_recognized_pitch_pattern(client):
    """Same H/L-per-mora shape as /pitch-accent's `pattern`, but computed
    from recognized_hiragana (see app/api/routes.py's /evaluate handler
    and app/schemas/pronunciation.py's recognized_pitch_pattern field)."""
    response = post(client)
    assert response.status_code == 200
    body = response.json()
    pattern = body["recognized_pitch_pattern"]
    assert isinstance(pattern, list)
    assert len(pattern) > 0
    for item in pattern:
        assert set(item) == {"mora", "pitch", "phrase"}
        assert item["pitch"] in ("H", "L")
        assert isinstance(item["phrase"], int)


def test_evaluate_recognized_pitch_pattern_is_empty_when_recognition_is_empty():
    """Graceful degrade: an ASR result with no pronounceable content must
    not fail /evaluate -- it just means an empty pitch pattern (same rule
    as the other pitch-related facts in this app, see /coach)."""
    app.dependency_overrides[get_asr_service] = lambda: StubASRService(kana="")
    try:
        with TestClient(app) as c:
            response = post(c)
            assert response.status_code == 200
            body = response.json()
            assert body["recognized_pitch_pattern"] == []
            assert isinstance(body["measured_pitch"], list)
    finally:
        app.dependency_overrides.clear()


def test_empty_text_is_rejected(client):
    assert post(client, text="   ").status_code == 400


def test_non_japanese_target_is_rejected(client):
    assert post(client, text="!!!").status_code == 400


def test_generic_binary_content_type_is_accepted(client):
    """curl sends application/octet-stream for -F audio=@file.wav."""
    response = post(client, content_type="application/octet-stream")
    assert response.status_code == 200


def test_mp3_and_other_audio_extensions_are_accepted(client):
    """MP3, FLAC, OGG, M4A, etc. are accepted by the API."""
    for ext, ctype in [("mp3", "audio/mpeg"), ("flac", "audio/flac"), ("m4a", "audio/mp4")]:
        response = post(client, filename=f"test.{ext}", content_type=ctype)
        assert response.status_code == 200


def test_unsupported_extension_is_rejected(client):
    response = post(client, filename="test.xyz", content_type="application/xyz")
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_invalid_audio_content_is_handled(client):
    response = post(client, content=b"this is not audio at all")
    assert response.status_code in (400, 422)


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


def test_category_guide_optional_param_handled(client):
    # Without running Ollama, calling /category-guide returns 503 Service Unavailable (whether category_name is passed or not)
    response_with_param = client.get("/api/pronunciation/category-guide", params={"category_name": "促音「っ」", "lang": "vi"})
    assert response_with_param.status_code in (503, 500)

    response_no_param = client.get("/api/pronunciation/category-guide", params={"lang": "vi"})
    assert response_no_param.status_code in (503, 500)


def test_text_guide_empty_text_rejected(client):
    response = client.get("/api/pronunciation/text-guide", params={"text": "  ", "lang": "vi"})
    assert response.status_code == 400


def test_text_guide_returns_503_when_ollama_unavailable(client):
    response = client.get("/api/pronunciation/text-guide", params={"text": "高校で英語を勉強します。", "lang": "vi"})
    assert response.status_code in (503, 500)




def test_english_text_guide_empty_text_rejected(client):
    response = client.get("/api/pronunciation-en/text-guide", params={"text": "  ", "lang": "vi"})
    assert response.status_code == 400


def test_english_text_guide_returns_503_when_ollama_unavailable(client):
    response = client.get("/api/pronunciation-en/text-guide", params={"text": "How are you today?", "lang": "vi"})
    assert response.status_code in (503, 500)


def test_english_text_guide_uses_english_prompt():
    from app.services.coaching import ENGLISH_TEXT_READING_GUIDE_PROMPTS, TEXT_READING_GUIDE_PROMPTS

    for key in ("vi", "en", "jp", "ko", "tw"):
        assert ENGLISH_TEXT_READING_GUIDE_PROMPTS[key] != TEXT_READING_GUIDE_PROMPTS[key]


def test_chinese_text_guide_empty_text_rejected(client):
    response = client.get("/api/pronunciation-zh/text-guide", params={"text": "  ", "lang": "vi"})
    assert response.status_code == 400


def test_chinese_text_guide_returns_503_when_ollama_unavailable(client):
    response = client.get("/api/pronunciation-zh/text-guide", params={"text": "你好，我是学生。", "lang": "vi"})
    assert response.status_code in (503, 500)


def test_chinese_text_guide_uses_chinese_prompt():
    from app.services.coaching import CHINESE_TEXT_READING_GUIDE_PROMPTS, TEXT_READING_GUIDE_PROMPTS

    for key in ("vi", "en", "jp", "ko", "tw"):
        assert CHINESE_TEXT_READING_GUIDE_PROMPTS[key] != TEXT_READING_GUIDE_PROMPTS[key]


def test_pitch_accent_flag_use_ollama_overrides_env(client, monkeypatch):
    from src.asr import kana_converter

    monkeypatch.setenv("KANA_USE_OLLAMA", "0")
    seen = []

    def fake_ollama(self, text):
        seen.append(text)
        return "きょうは"

    monkeypatch.setattr(kana_converter.JapaneseKanaConverter, "_ollama_to_kana", fake_ollama)

    assert client.get("/api/pronunciation/pitch-accent", params={"text": "今日は"}).status_code == 200
    assert seen == []  # env says 0 and no flag -> pykakasi only

    response = client.get("/api/pronunciation/pitch-accent", params={"text": "今日は", "flag_use_ollama": 1})
    assert response.status_code == 200
    assert seen  # flag=1 -> Ollama used despite KANA_USE_OLLAMA=0

    seen.clear()
    assert client.get("/api/pronunciation/pitch-accent", params={"text": "今日は"}).status_code == 200
    assert seen == []  # override does not leak into the next request

    assert client.get("/api/pronunciation/pitch-accent", params={"text": "今日は", "flag_use_ollama": 2}).status_code == 422


def test_pitch_accent_reports_engine(client, monkeypatch):
    from src.asr import kana_converter

    monkeypatch.setenv("KANA_USE_OLLAMA", "0")
    monkeypatch.setattr(
        kana_converter.JapaneseKanaConverter, "_ollama_to_kana", lambda self, text: "しゅっしん"
    )
    body = client.get("/api/pronunciation/pitch-accent", params={"text": "出身"}).json()
    assert body["engine"] == "pykakasi"
    assert body["reading"] == "しゅっしん"

    body = client.get(
        "/api/pronunciation/pitch-accent", params={"text": "出身", "flag_use_ollama": 1}
    ).json()
    assert body["engine"] == "ollama"

    def broken_ollama(self, text):
        raise TimeoutError("ollama down")

    monkeypatch.setattr(kana_converter.JapaneseKanaConverter, "_ollama_to_kana", broken_ollama)
    body = client.get(
        "/api/pronunciation/pitch-accent", params={"text": "出身", "flag_use_ollama": 1}
    ).json()
    assert body["engine"] == "pykakasi"  # Ollama failed -> fallback is reported, not hidden
    assert body["reading"] == "しゅっしん"


def test_ollama_prompt_carries_unidic_draft():
    from src.asr.kana_converter import build_ollama_prompt

    prompt = build_ollama_prompt("出身。")
    assert "<text>出身。</text>" in prompt
    assert "<draft>出身[しゅっしん] 。</draft>" in prompt


def test_ollama_request_is_deterministic(monkeypatch):
    import json

    from src.asr import kana_converter

    sent = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"response": "<think>しんしゅつ?</think>\nしゅっしん"}).encode()

    def fake_urlopen(req, timeout):
        sent.update(json.loads(req.data))
        return FakeResp()

    monkeypatch.setattr(kana_converter.urllib.request, "urlopen", fake_urlopen)
    assert kana_converter.JapaneseKanaConverter()._ollama_to_kana("出身") == "しゅっしん"
    assert sent["think"] is False
    assert sent["options"]["temperature"] == 0.0
    assert sent["options"]["top_k"] == 1
    assert "seed" in sent["options"]


def test_use_ollama_g2p_context_restores_env_setting(monkeypatch):
    from src.asr.kana_converter import JapaneseKanaConverter, use_ollama_g2p

    monkeypatch.setenv("KANA_USE_OLLAMA", "0")
    converter = JapaneseKanaConverter()
    assert not converter._is_ollama_enabled()
    with use_ollama_g2p(True):
        assert converter._is_ollama_enabled()
    assert not converter._is_ollama_enabled()
    monkeypatch.setenv("KANA_USE_OLLAMA", "1")
    with use_ollama_g2p(None):
        assert converter._is_ollama_enabled()
