import io
import unicodedata
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.aspect_scoring import score_f0_dynamics
from app.services.korean_asr import KoreanRecognition, get_korean_asr_service
from app.services.korean_text import (
    decompose_hangul,
    compose_hangul,
    korean_pitch_pattern,
    korean_pronunciation,
    korean_syllables,
    normalize_korean,
    score_korean_syllables,
)
from app.services.scoring import score_pronunciation
from app.services.coaching import KOREAN_TEXT_READING_GUIDE_PROMPTS, TEXT_READING_GUIDE_PROMPTS


class StubKoreanASRService:
    def __init__(self, text: str = "안녕하세요"):
        self.text = text

    is_loaded = True

    def recognize(self, audio_path, align_to=""):
        return KoreanRecognition(text=self.text, duration=2.5, inference_time=0.3, windows=None)


def make_wav(seconds: float = 0.1, sample_rate: int = 16_000) -> bytes:
    import struct
    n = int(seconds * sample_rate)
    data = b"\x00\x00" * n
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", len(data),
    )
    return header + data


def test_decompose_and_compose_hangul():
    cho, jung, jong = decompose_hangul("한")
    assert (cho, jung, jong) == ("ㅎ", "ㅏ", "ㄴ")
    assert compose_hangul(cho, jung, jong) == "한"


def test_normalize_korean_composes_jamo_and_removes_non_hangul():
    decomposed = unicodedata.normalize("NFD", "안녕")
    assert normalize_korean(f" {decomposed}, hello! ") == "안녕"
    assert korean_syllables("한국어") == ["한", "국", "어"]


def test_korean_pronunciation_rules():
    # Nasalization: ㅂ + ㄴ -> [ㅁ]
    assert korean_pronunciation("감사합니다") == "감사함니다"
    # Liaison: 국 + 어 -> [구거]
    assert korean_pronunciation("한국어") == "한구거"
    # H-deletion: 좋아 -> [조아]
    assert korean_pronunciation("좋아요") == "조아요"
    # Tensification: 학 + 교 -> [학꾜]
    assert korean_pronunciation("학교") == "학꾜"


def test_score_korean_syllables_tolerates_phonetic_realization():
    # If target is orthographic 감사합니다 and recognized is phonetic 감사함니다, score should be 100
    result = score_korean_syllables("감사합니다", "감사함니다")
    assert result.score == 100
    assert result.distance == 0


def test_korean_pitch_pattern_accentual_phrase():
    pattern = korean_pitch_pattern("안녕하세요")
    assert len(pattern) == 5
    for item in pattern:
        assert set(item) == {"mora", "pitch", "phrase"}
        assert item["pitch"] in ("H", "L")


def test_reference_free_f0_score_requires_two_voiced_points():
    assert score_f0_dynamics([{"semitone": 0.0, "voiced": True}]) == (None, False)
    score, measured = score_f0_dynamics([
        {"semitone": -2.0, "voiced": True},
        {"semitone": 2.0, "voiced": True},
    ])
    assert measured is True
    assert score == 100.0


def test_korean_evaluate_endpoint():
    app.dependency_overrides[get_korean_asr_service] = lambda: StubKoreanASRService(text="감사함니다")
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/pronunciation-ko/evaluate",
                data={"text": "감사합니다"},
                files={"audio": ("sample.wav", io.BytesIO(make_wav()), "audio/wav")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["score"] == 100
            assert data["target_text"] == "감사합니다"
            assert data["target_hiragana"] == "감사함니다"
            assert len(data["mora_status"]) == 5
            assert all(m["ok"] is True for m in data["mora_status"])
    finally:
        app.dependency_overrides.clear()


def test_korean_pitch_accent_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/pronunciation-ko/pitch-accent", params={"text": "안녕하세요"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["pattern"]) == 5


def test_korean_reading_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/pronunciation-ko/reading", params={"text": "감사합니다"})
        assert resp.status_code == 200
        assert resp.json()["reading"] == "감사함니다"


def test_korean_evaluate_rejects_non_korean():
    with TestClient(app) as client:
        resp = client.post(
            "/api/pronunciation-ko/evaluate",
            data={"text": "hello world"},
            files={"audio": ("sample.wav", io.BytesIO(make_wav()), "audio/wav")},
        )
        assert resp.status_code == 400


def test_korean_text_guide_empty_rejected():
    with TestClient(app) as client:
        resp = client.get("/api/pronunciation-ko/text-guide", params={"text": " ", "lang": "vi"})
        assert resp.status_code == 400


def test_korean_text_guide_returns_503_when_ollama_unavailable():
    with TestClient(app) as client:
        resp = client.get("/api/pronunciation-ko/text-guide", params={"text": "안녕하세요.", "lang": "vi"})
        assert resp.status_code in (503, 500)


def test_korean_text_guide_uses_korean_prompt():
    for key in ("vi", "en", "jp", "ko", "tw"):
        assert KOREAN_TEXT_READING_GUIDE_PROMPTS[key] != TEXT_READING_GUIDE_PROMPTS[key]
