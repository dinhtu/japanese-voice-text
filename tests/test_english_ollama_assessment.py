import asyncio
import json
from types import SimpleNamespace

import pytest

from app.schemas.english_assessment import OllamaAssessment
from app.services import english_ollama_assessment as oa
from app.services.english_assessment import no_voice_reading
from tests.test_english_assessment import _assess, _intonation


def _settings(enabled: bool = True):
    return SimpleNamespace(
        ollama_host="http://localhost:11434", ollama_model="qwen3:8b",
        en_ollama_assessment=enabled, en_ollama_assessment_timeout_s=5.0,
    )


def _reply(**scores) -> str:
    return json.dumps({
        "criteria": {k: {"score": scores.get(k, 80), "reason": f"{k} ok"} for k in oa.WEIGHTS},
        "overall_score": scores.get("overall", 82),
        "summary": "Good reading.",
    })


def _run(reading, monkeypatch, reply=None, error=None, intonation=None, settings=None, lang="vi"):
    seen = {}

    async def fake_chat(_settings, messages):
        seen["messages"] = messages
        if error is not None:
            raise error
        return reply

    monkeypatch.setattr(oa, "_chat", fake_chat)
    gopt = {"average": 35.0, "adjusted_average": 30.0}
    out = asyncio.run(oa.assess_with_ollama(reading, gopt, intonation, settings or _settings(), lang))
    OllamaAssessment.model_validate(out)
    return out, seen


def test_facts_hold_measurements_not_the_other_scores():
    target = "my teacher is beautiful"
    intonation = _intonation(target, {1: True, 3: True}, {1: True, 3: False}, True, pauses=[0.0, 0.4, 0.0])
    reading = _assess(target, "my teacher is banana", intonation=intonation)
    facts = oa.llm_facts(reading, intonation)
    text = json.dumps(facts)
    assert "overall" not in text and '"score"' not in text and "gopt" not in text.lower()
    assert facts["words_read_correctly"] == "3 of 4"
    assert facts["words"][3]["status"] == "substituted" and facts["words"][3]["heard_as"] == "banana"
    # Pitch checks only on words that were read right.
    assert [p["word"] for p in facts["intonation"]["key_word_pitch"]] == ["teacher"]


def test_scored_reply_is_clamped_and_reweighted(monkeypatch):
    reading = _assess("nice to meet you", "nice to meet you")
    out, seen = _run(reading, monkeypatch, _reply(content_accuracy=120, word_stress="n/a", overall=90))
    assert out["status"] == "scored"
    c = out["criteria"]
    assert c["content_accuracy"]["score"] == 100  # clamped
    assert c["word_stress"]["score"] is None  # not a number
    assert out["overall_score"] == 90
    # Weighted with reading_assessment's weights over the non-null scores.
    expected = (0.25 * 100 + 0.30 * 80 + 0.10 * 80 * 3) / (0.25 + 0.30 + 0.30)
    assert out["weighted_overall"] == pytest.approx(expected, abs=0.01)
    cmp = out["comparison"]
    assert cmp["reading_overall"] == reading["overall_score"] and cmp["gopt_average"] == 35.0
    assert cmp["delta_vs_reading"] == pytest.approx(out["weighted_overall"] - reading["overall_score"], abs=0.01)
    assert "Vietnamese" in seen["messages"][0]["content"]


def test_english_reasons_when_lang_en(monkeypatch):
    _out, seen = _run(_assess("nice to meet you", "nice to meet you"), monkeypatch, _reply(), lang="en")
    assert "Write them in English" in seen["messages"][0]["content"]


def test_ollama_down_is_a_status_not_an_error(monkeypatch):
    out, _ = _run(_assess("nice to meet you", "nice to meet you"), monkeypatch,
                  error=ConnectionError("connection refused"))
    assert out["status"] == "unavailable" and "refused" in out["error"]
    assert out["criteria"] is None


def test_invalid_reply(monkeypatch):
    out, _ = _run(_assess("nice to meet you", "nice to meet you"), monkeypatch, reply="Great job!")
    assert out["status"] == "invalid_output"


def test_disabled_and_no_voice_skip_the_call(monkeypatch):
    reading = _assess("nice to meet you", "nice to meet you")
    out, seen = _run(reading, monkeypatch, _reply(), settings=_settings(enabled=False))
    assert out["status"] == "disabled" and "messages" not in seen
    silent = no_voice_reading("Nice to meet you", ["Nice", "to", "meet", "you"], 1.0, "silent")
    out, seen = _run(silent, monkeypatch, _reply())
    assert out["status"] == "no_voice" and "messages" not in seen
