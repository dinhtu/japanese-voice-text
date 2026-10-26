import asyncio
import base64
import io
import json
import math
import wave
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from app.schemas.english_assessment import OllamaAssessment
from app.services import english_ollama_assessment as oa
from app.services import qwen3_asr
from app.services.english_assessment import AssessmentInputs, no_voice_reading
from app.services.english_intonation import analyze_sentence
from app.services.english_scoring import align_word_tokens
from tests.test_english_assessment import _assess, _timing, _vad

TARGET = "I would like a glass of water."


def _settings(enabled: bool = True):
    return SimpleNamespace(
        ollama_host="http://ollama.test", ollama_model="qwen3:8b",
        en_ollama_asr_model="frozenlab/qwen3-asr",
        en_ollama_assessment=enabled, en_ollama_assessment_timeout_s=5.0,
    )


def _inputs(target: str = TARGET, seconds: float = 2.0) -> AssessmentInputs:
    return AssessmentInputs(
        target_text=target, analysis=analyze_sentence(target), timing=_timing(target),
        intonation=None, vad=_vad(seconds), duration=seconds,
        samples=np.zeros(int(16000 * seconds), dtype=np.float32), sample_rate=16000,
        legacy_score=90,
    )


def _logprobs(content: str, low: dict[str, float] | None = None) -> list[dict]:
    """Tokens of `content` the way a BPE tokenizer might cut it: the tag in
    pieces, a leading space on each word, punctuation on its own."""
    head, _, text = content.partition("<asr_text>")
    toks = [head[:8], head[8:], "<", "asr", "_text", ">"]
    for k, word in enumerate(text.split(" ")):
        core = word.rstrip(".,?!")
        toks.append((" " if k else "") + core)
        if word != core:
            toks.append(word[len(core):])
    low = low or {}
    return [{"token": t, "logprob": low.get(t.strip(), -0.05)} for t in toks if t]


def _run(monkeypatch, content=None, logprobs=True, error=None, reading=None, inputs="auto", settings=None):
    seen = []

    async def fake_chat(_settings, wav, **_kw):
        seen.append(wav)
        if error is not None:
            raise error
        data = {"message": {"role": "assistant", "content": content}}
        if logprobs:
            data["logprobs"] = _logprobs(content) if logprobs is True else logprobs
        return data

    monkeypatch.setattr(qwen3_asr, "chat", fake_chat)
    out = asyncio.run(oa.assess_with_ollama(
        _inputs() if inputs == "auto" else inputs, reading, {"average": 35.0, "adjusted_average": 30.0},
        settings or _settings(),
    ))
    OllamaAssessment.model_validate(json.loads(json.dumps(out)))
    return out, seen


# --- audio / reply ---------------------------------------------------------

def test_wav_bytes_is_16k_mono_pcm16_with_riff_header():
    data = oa.wav_bytes(np.sin(np.linspace(0, 100, 16000)).astype(np.float32))
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 16000)


def test_audio_windows_cut_at_30s_and_fold_short_tail():
    sr = 16000
    assert [len(w) / sr for w in oa.audio_windows(np.zeros(70 * sr), sr)] == [30, 30, 10]
    assert [len(w) / sr for w in oa.audio_windows(np.zeros(int(60.5 * sr)), sr)] == [30, 30.5]
    assert len(oa.audio_windows(np.zeros(5 * sr), sr)) == 1


def test_parse_reply():
    assert oa.parse_reply("language English<asr_text>Hello there.") == ("English", "Hello there.")
    assert oa.parse_reply("language None<asr_text>") == (None, "")
    with pytest.raises(ValueError):
        oa.parse_reply("I can only see the image you have provided")


def test_word_confidence_from_split_tokens():
    content = "language English<asr_text>I would like a glass of water."
    lp = _logprobs(content, {"glass": math.log(0.4)})
    conf = oa.raw_word_confidences(lp, "I would like a glass of water.")
    assert len(conf) == 7 and conf[4] == pytest.approx(0.4, abs=1e-3)
    assert conf[0] == pytest.approx(math.exp(-0.05), abs=1e-3)
    # Tokens that spell something else are not trusted.
    assert oa.raw_word_confidences(lp, "I would like a cup of water.") is None
    assert oa.raw_word_confidences(None, "x") is None


def test_numbers_expand_and_keep_their_confidence():
    tokens, conf = oa.speech_tokens_with_confidence("I have 2 cats", [0.9, 0.8, 0.5, 0.7])
    assert tokens == ["i", "have", "two", "cats"] and conf == [0.9, 0.8, 0.5, 0.7]


@pytest.mark.parametrize("target, heard", [
    ("see you today", "see you to day"),
    ("thank you", "thankyou"),
    ("nice to meet you", "nice to to meet you"),
    ("I would like a glass", "i would like glass"),
])
def test_target_confidence_replays_the_alignment(target, heard):
    tokens = heard.split()
    conf = [round(0.5 + 0.1 * k, 2) for k in range(len(tokens))]
    matches, extras = align_word_tokens(target, heard)
    out = oa.target_confidence(matches, extras, tokens, conf)
    assert out is not None and len(out) == len(matches)
    assert all((c is None) == (not m.heard) for c, m in zip(out, matches))


# --- report ----------------------------------------------------------------

def test_qwen_transcript_is_scored_with_its_confidence(monkeypatch):
    reading = _assess(TARGET, "i would like a glass of water")
    out, seen = _run(monkeypatch, "language English<asr_text>I would like glass of water.", reading=reading)
    assert out["status"] in ("complete", "partial_assessment")
    assert out["language"] == "English" and out["transcript"] == "I would like glass of water."
    assert [w["status"] for w in out["word_results"]][3] == "omitted"
    assert out["criteria"]["content_accuracy"]["score"] == pytest.approx(100 * (1 - 1 / 7), abs=0.01)
    assert out["confidence_source"] == "qwen3_asr_logprobs"
    assert out["criteria"]["phoneme_accuracy"]["metric"] == "asr_token_confidence"
    # wav2vec2 heard "a", Qwen3-ASR did not.
    assert out["agreement"]["wrong_only_by_qwen3_asr"] == ["a"]
    assert out["comparison"]["reading_overall"] == reading["overall_score"]
    assert len(seen) == 1 and seen[0][:4] == b"RIFF"


def test_without_logprobs_phoneme_falls_back_to_forced_alignment(monkeypatch):
    out, _ = _run(monkeypatch, "language English<asr_text>I would like a glass of water.", logprobs=False)
    assert out["confidence_source"] == "forced_alignment"
    assert out["criteria"]["phoneme_accuracy"]["metric"] == "ctc_forced_alignment_letter_posterior"


def test_misreading_gates_prosody_like_reading(monkeypatch):
    out, _ = _run(monkeypatch, "language English<asr_text>The cat is sleeping on the sofa.")
    assert out["status"] == "content_mismatch"
    assert out["criteria"]["fluency"]["status"] == "gated"


def test_no_speech_and_other_language(monkeypatch):
    out, _ = _run(monkeypatch, "language None<asr_text>", logprobs=False)
    assert out["status"] == "no_speech" and out["overall_score"] == 0
    out, _ = _run(monkeypatch, "language Vietnamese<asr_text>Tôi muốn một cốc nước.", logprobs=False)
    assert out["language_mismatch"] is True


def test_reply_without_tag_is_invalid_output(monkeypatch):
    out, _ = _run(monkeypatch, "I can only see the image you have provided", logprobs=False)
    assert out["status"] == "invalid_output" and "qwen3a" in out["error"]


def test_ollama_errors_disabled_and_skips(monkeypatch):
    out, _ = _run(monkeypatch, error=oa.AsrError("unavailable", "connection refused"))
    assert out["status"] == "unavailable" and out["criteria"] is None
    out, seen = _run(monkeypatch, "x", settings=_settings(enabled=False))
    assert out["status"] == "disabled" and not seen
    silent = no_voice_reading("Nice to meet you", ["Nice", "to", "meet", "you"], 1.0, "silent")
    out, seen = _run(monkeypatch, "x", reading=silent)
    assert out["status"] == "no_voice" and not seen
    out, seen = _run(monkeypatch, "x", inputs=None)
    assert out["status"] == "skipped" and not seen


# --- HTTP ------------------------------------------------------------------

def _with_transport(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_request_carries_audio_and_never_the_target(monkeypatch):
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "language English<asr_text>Hi."}})

    _with_transport(monkeypatch, handler)
    data = asyncio.run(qwen3_asr.chat(_settings(), oa.wav_bytes(np.zeros(1600))))
    system, msg = sent["messages"]
    assert system == {"role": "system", "content": qwen3_asr.VERBATIM_PROMPT}
    assert sent["model"] == "frozenlab/qwen3-asr" and msg["content"] == ""
    assert base64.b64decode(msg["images"][0])[:4] == b"RIFF"
    assert sent["options"]["temperature"] == 0 and sent["logprobs"] is True
    assert "glass" not in json.dumps(sent)  # no target-sentence context
    assert data["message"]["content"].endswith("Hi.")


@pytest.mark.parametrize("response, status, text", [
    (httpx.Response(404, json={"error": "model not found"}), "unavailable", "ollama pull frozenlab/qwen3-asr"),
    (httpx.Response(400, json={"error": "frozenlab/qwen3-asr does not support vision"}), "unsupported", "does not support"),
    (httpx.Response(500, text="boom"), "unavailable", "HTTP 500"),
])
def test_http_errors_map_to_statuses(monkeypatch, response, status, text):
    _with_transport(monkeypatch, lambda request: response)
    with pytest.raises(oa.AsrError) as e:
        asyncio.run(qwen3_asr.chat(_settings(), b"RIFF"))
    assert e.value.status == status and text in str(e.value)


def test_connection_error_maps_to_unavailable(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    _with_transport(monkeypatch, handler)
    with pytest.raises(oa.AsrError) as e:
        asyncio.run(qwen3_asr.chat(_settings(), b"RIFF"))
    assert e.value.status == "unavailable" and "ollama.test" in str(e.value)
