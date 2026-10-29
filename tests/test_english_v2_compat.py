"""/evaluate-v2 answers in the /evaluate shape (+ coach), and /evaluate can
return the coach comment in the same request."""

import asyncio
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import english_routes
from app.core.config import get_settings
from app.schemas.english_assessment import EnglishEvaluateResponse
from app.schemas.english_v2 import EnglishEvaluateV2Response
from app.services import coaching
from app.services import english_evaluate_v2 as v2
from app.services import qwen3_asr
from app.services.aspect_scoring import AspectScores
from app.services.english_asr import get_english_asr_service
from app.services.scoring import ScoreResult
from app.services.use_cases import EvaluationResult
from tests.test_english_evaluate_v2 import TARGET, _acoustics, _reply, _settings


def _report(monkeypatch, reply, acoustics=None, feedback=True):
    async def fake_chat(_settings, wav, **_kw):
        return reply

    async def fake_comment(facts, settings, lang="vi", target_lang="ja"):
        return coaching.CoachingComment("Đọc rõ.", "Chú ý 'glass'.")

    monkeypatch.setattr(qwen3_asr, "chat", fake_chat)
    monkeypatch.setattr(coaching, "generate_comment", fake_comment)
    acoustics = acoustics or _acoustics()
    report = asyncio.run(v2.evaluate_v2(acoustics, _settings(), with_feedback=feedback))
    return report, acoustics


def test_legacy_fields_have_every_evaluate_field(monkeypatch):
    report, acoustics = _report(monkeypatch, _reply("I would like a glass of water.", {4: {"grass": 0.0013}}))
    data = v2.legacy_fields(report, acoustics)
    data["v2_assessment"] = report
    out = EnglishEvaluateV2Response.model_validate(json.loads(json.dumps(data)))
    v1_fields = set(EnglishEvaluateResponse.model_fields)
    assert v1_fields <= set(out.model_dump())
    assert out.score == round(report["overall_score"]) and out.overall_score == report["overall_score"]
    assert out.pronunciation_score == report["criteria"]["pronunciation"]["score"]
    assert out.fluency_score == report["criteria"]["fluency"]["score"]
    assert out.recognized_text == "I would like a glass of water." and out.cer == 0
    glass = [m for m in out.mora_status if m.mora == "glass"][0]
    assert glass.ok is False  # flagged as mispronounced ("~grass")
    assert out.coach.assessment == "Đọc rõ." and out.coach.suggestion == "Chú ý 'glass'."
    assert out.feedback.level in ("excellent", "good", "fair", "poor", "mismatch")
    assert len(out.measured_pitch) == len(out.mora_status)


def test_top_level_criterion_scores_rebuild_the_overall(monkeypatch):
    report, acoustics = _report(monkeypatch, _reply("I would like a glass of water.", {4: {"grass": 0.0013}}))
    data = v2.legacy_fields(report, acoustics)
    criteria = report["criteria"]
    assert data["content_score"] == criteria["content_accuracy"]["score"]
    assert data["word_stress_score"] == criteria["word_stress"]["score"]
    assert data["intonation_criterion_score"] == criteria["intonation"]["score"]
    five = {"content_accuracy": data["content_score"], "pronunciation": data["pronunciation_score"],
            "word_stress": data["word_stress_score"], "intonation": data["intonation_criterion_score"],
            "fluency": data["fluency_score"]}
    used = {k: s for k, s in five.items() if s is not None}
    rebuilt = sum(v2.WEIGHTS[k] * s for k, s in used.items()) / sum(v2.WEIGHTS[k] for k in used)
    assert abs(rebuilt - report["overall_score"]) < 0.01


def test_top_level_criterion_scores_are_null_when_not_scored(monkeypatch):
    report, acoustics = _report(monkeypatch, _reply("The cat is sleeping on the sofa today."), feedback=False)
    data = v2.legacy_fields(report, acoustics)
    assert data["content_score"] == report["criteria"]["content_accuracy"]["score"] is not None
    assert data["word_stress_score"] is None and data["intonation_criterion_score"] is None  # gated
    # A failed gate: the overall is content alone; fluency_score keeps the /evaluate 0.
    assert report["status"] == "content_mismatch"
    assert data["overall_score"] == data["content_score"] and data["fluency_score"] == 0.0
    report, acoustics = _report(monkeypatch, _reply("x"), acoustics=_acoustics(voiced=False), feedback=False)
    data = v2.legacy_fields(report, acoustics)
    assert data["content_score"] is None and data["word_stress_score"] is None
    assert data["intonation_criterion_score"] is None


def test_legacy_fields_for_misreading_and_no_voice(monkeypatch):
    report, acoustics = _report(monkeypatch, _reply("The cat is sleeping on the sofa today."), feedback=False)
    data = v2.legacy_fields(report, acoustics)
    assert data["score"] == round(report["overall_score"]) and data["fluency_score"] == 0.0
    assert data["coach"] is None and data["coach_error"] is None
    assert data["errors"] and data["cer"] > 0
    report, acoustics = _report(monkeypatch, _reply("x"), acoustics=_acoustics(voiced=False))
    data = v2.legacy_fields(report, acoustics)
    assert data["score"] == 0 and data["feedback"]["level"] == "mismatch"


def _client(monkeypatch, reply=None, error=None):
    async def fake_chat(_settings, wav, **_kw):
        if error is not None:
            raise error
        return reply

    async def fake_comment(facts, settings, lang="vi", target_lang="ja"):
        return coaching.CoachingComment("Nhận xét.", "Gợi ý.")

    monkeypatch.setattr(qwen3_asr, "chat", fake_chat)
    monkeypatch.setattr(coaching, "generate_comment", fake_comment)
    monkeypatch.setattr(v2, "measure_acoustics", lambda *a, **k: _acoustics())
    app = FastAPI()
    app.include_router(english_routes.router, prefix="/en")
    app.dependency_overrides[get_english_asr_service] = lambda: None
    app.dependency_overrides[get_settings] = lambda: _settings(max_audio_bytes=10_000_000, en_v2_aligner="none")
    return TestClient(app)


def _post(client, url, **data):
    return client.post(url, data={"text": TARGET, **data},
                       files={"audio": ("a.wav", qwen3_asr.wav_bytes([0.0] * 1600), "audio/wav")})


def test_evaluate_v2_route_returns_the_evaluate_shape_with_coach(monkeypatch):
    client = _client(monkeypatch, _reply("I would like glass of water."))
    r = _post(client, "/en/evaluate-v2")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(EnglishEvaluateResponse.model_fields) <= set(body)
    assert body["coach"] == {"assessment": "Nhận xét.", "suggestion": "Gợi ý.", "lang": "vi"}
    assert body["v2_assessment"]["scoring_version"] == "en-reading-v2"
    assert body["content_score"] == body["v2_assessment"]["criteria"]["content_accuracy"]["score"]
    assert "word_stress_score" in body and "intonation_criterion_score" in body
    assert [m["mora"] for m in body["mora_status"]][3] == "a" and body["mora_status"][3]["ok"] is False
    # feedback=false: no comment, no error.
    body = _post(client, "/en/evaluate-v2", feedback="false").json()
    assert body["coach"] is None and body["coach_error"] is None


def test_evaluate_v2_route_503_when_ollama_asr_is_down(monkeypatch):
    client = _client(monkeypatch, error=qwen3_asr.AsrError("unavailable", "Không kết nối được Ollama"))
    r = _post(client, "/en/evaluate-v2")
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]


def test_evaluate_v2_route_rejects_unknown_lang(monkeypatch):
    client = _client(monkeypatch, _reply("I would like a glass of water."))
    assert _post(client, "/en/evaluate-v2", lang="xx").status_code == 400


class _FakeUseCase:
    def execute(self, text, path, lang="vi"):
        score = ScoreResult(score=90, cer=0.1, distance=1, target_length=10, level="excellent", message="ok")
        aspects = AspectScores(overall_score=90.0, pronunciation_score=90.0, fluency_score=80.0,
                               rhythm_score=None, intonation_score=None, rhythm_measured=False,
                               intonation_measured=False)
        return EvaluationResult(text, "i would like a glass of water", "i would like a glass of water",
                                "i would like a glass of water", score, aspects, 2.0, 10.0, measured_pitch=[])


def test_evaluate_coach_flag_returns_the_comment_without_rescoring(monkeypatch):
    calls = []

    async def fake_comment(facts, settings, lang="vi", target_lang="ja"):
        calls.append((facts.score, lang, target_lang))
        return coaching.CoachingComment("Tốt.", "")

    async def no_ollama(result, settings):
        return None

    monkeypatch.setattr(english_routes, "generate_comment", fake_comment)
    monkeypatch.setattr(english_routes, "_ollama_field", no_ollama)
    app = FastAPI()
    app.include_router(english_routes.router, prefix="/en")
    app.dependency_overrides[english_routes.get_en_use_case] = lambda: _FakeUseCase()
    client = TestClient(app)
    body = _post(client, "/en/evaluate", coach="true", lang="jp").json()
    assert body["coach"] == {"assessment": "Tốt.", "suggestion": "", "lang": "ja"}
    assert calls == [(90, "ja", "en")]
    # Without the flag /evaluate is as before: no comment, no Ollama chat call.
    body = _post(client, "/en/evaluate").json()
    assert body["coach"] is None and len(calls) == 1


def test_silent_take_still_gets_the_coach_comment(monkeypatch):
    seen = {}

    async def fake_comment(facts, settings, lang="vi", target_lang="ja"):
        seen["lines"] = "\n".join(facts.extra_lines)
        return coaching.CoachingComment("Không nghe thấy giọng.", "Hãy đọc to hơn.")

    report, acoustics = _report(monkeypatch, _reply("x"), acoustics=_acoustics(voiced=False))
    monkeypatch.setattr(coaching, "generate_comment", fake_comment)
    report = asyncio.run(v2.evaluate_v2(acoustics, _settings(), with_feedback=True))
    assert report["status"] == "no_voice" and report["feedback"]["status"] == "generated"
    assert "Không nghe thấy giọng đọc" in seen["lines"] and "Không phát hiện từ nào" not in seen["lines"]
    data = v2.legacy_fields(report, acoustics)
    assert data["score"] == 0 and data["coach"]["assessment"] == "Không nghe thấy giọng."
    assert data["coach_error"] is None


def test_evaluate_v2_route_silent_take_has_coach(monkeypatch):
    client = _client(monkeypatch, _reply("x"))
    monkeypatch.setattr(v2, "measure_acoustics", lambda *a, **k: _acoustics(voiced=False))
    body = _post(client, "/en/evaluate-v2").json()
    assert body["v2_assessment"]["status"] == "no_voice" and body["score"] == 0
    assert body["coach"] == {"assessment": "Nhận xét.", "suggestion": "Gợi ý.", "lang": "vi"}
