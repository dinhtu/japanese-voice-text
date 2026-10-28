import asyncio
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from app.schemas.english_v2 import EvaluateV2Response
from app.services import english_evaluate_v2 as v2
from app.services import qwen3_asr
from app.services.english_intonation import analyze_sentence
from app.services.vad_fluency import VadResult
from tests.test_english_assessment import _intonation, _timing

TARGET = "I would like a glass of water."


# --- Ollama replies --------------------------------------------------------

def _tok(token: str, p: float = 1.0, alts: dict[str, float] | None = None) -> dict:
    alts = alts or {}
    top = {token: p, **alts}
    return {
        "token": token, "logprob": math.log(p),
        "top_logprobs": [{"token": t, "logprob": math.log(max(q, 1e-12))}
                         for t, q in sorted(top.items(), key=lambda kv: -kv[1])],
    }


def _reply(transcript: str, alts: dict[int, dict[str, float]] | None = None, logprobs: bool = True) -> dict:
    """An Ollama /api/chat response for `transcript`: one token per word
    (with a leading space after the first), punctuation on its own, and
    `alts` = rival tokens for the k-th word."""
    alts = alts or {}
    toks = [_tok("language"), _tok(" English"), _tok("<asr_text>")]
    for k, word in enumerate(transcript.split(" ") if transcript else []):
        core = word.rstrip(".,?!")
        given = alts.get(k, {})
        p = max(1e-6, 1.0 - sum(given.values()))
        toks.append(_tok((" " if k else "") + core, p, {(" " if k else "") + a: q for a, q in given.items()}))
        if word != core:
            toks.append(_tok(word[len(core):]))
    data = {"message": {"role": "assistant", "content": f"language English<asr_text>{transcript}"}}
    if logprobs:
        data["logprobs"] = toks
    return data


def _settings(**over):
    base = dict(ollama_host="http://ollama.test", ollama_model="qwen3:8b",
                en_ollama_asr_model="frozenlab/qwen3-asr", en_ollama_assessment_timeout_s=5.0,
                en_v2_feedback=True)
    base.update(over)
    return SimpleNamespace(**base)


def _acoustics(target: str = TARGET, voiced: bool = True, timing="auto", intonation=None, seconds=2.4):
    analysis = analyze_sentence(target)
    return v2.Acoustics(
        target_text=target, samples=np.zeros(int(16000 * seconds), dtype=np.float32), sample_rate=16000,
        duration=seconds, vad=VadResult(seconds - 0.4, 0, 0.8, "energy"),
        presence=SimpleNamespace(voiced=voiced, verdict="voice" if voiced else "silent"),
        analysis=analysis, timing=_timing(target) if timing == "auto" else timing,
        intonation=intonation, aligner="wav2vec2_ctc_forced_alignment" if timing else None,
        quality={"duration_seconds": seconds},
    )


def _run(monkeypatch, reply, acoustics=None, feedback=False, error=None, **settings):
    calls = []

    async def fake_chat(_settings, wav, **_kw):
        calls.append(wav[:4])
        if error is not None:
            raise error
        return reply

    monkeypatch.setattr(qwen3_asr, "chat", fake_chat)
    report = asyncio.run(v2.evaluate_v2(acoustics or _acoustics(), _settings(**settings), with_feedback=feedback))
    EvaluateV2Response.model_validate(json.loads(json.dumps(report)))
    return report, calls


# --- reports ---------------------------------------------------------------

def test_clean_reading_scores_every_criterion_with_document_weights(monkeypatch):
    target = "my teacher is beautiful"
    intonation = _intonation(target, {1: True, 3: True}, {1: True, 3: True}, True, pauses=[0.0, 0.0, 0.0])
    report, calls = _run(monkeypatch, _reply("My teacher is beautiful."),
                         acoustics=_acoustics(target, intonation=intonation, seconds=1.8))
    c = report["criteria"]
    assert {k: c[k]["weight"] for k in c} == {
        "content_accuracy": 0.25, "pronunciation": 0.30, "word_stress": 0.15, "intonation": 0.15, "fluency": 0.15}
    assert c["content_accuracy"]["score"] == 100 and c["pronunciation"]["score"] == 100
    assert c["pronunciation"]["status"] == "estimated" and c["pronunciation"]["metric"] == "qwen3_asr_rival_margin"
    assert c["word_stress"]["score"] == 100 and c["intonation"]["score"] == 100
    expected = sum(c[k]["score"] * c[k]["weight"] for k in c)
    assert report["status"] == "complete" and report["overall_score"] == pytest.approx(expected, abs=0.01)
    assert calls == [b"RIFF"]  # one transcription call, no extra scoring calls
    assert all(w["phonemes"] == [] for w in report["word_results"])


def test_repaired_misreading_is_caught_from_token_alternatives(monkeypatch):
    # Qwen3-ASR writes "glass" but keeps "grass" as a 0.13 % alternative.
    report, _ = _run(monkeypatch, _reply("I would like a glass of water.", {4: {"grass": 0.0013}}))
    glass = report["word_results"][4]
    assert glass["status"] == "correct" and glass["pronunciation"]["rival"] == "grass"
    assert glass["pronunciation"]["score"] < v2.MISPRONOUNCED
    assert ("mispronunciation", "glass", "grass") in [(e["type"], e["word"], e["heard"]) for e in report["errors"]]
    assert report["improvements"][0]["code"] == "mispronounced_words"
    assert report["criteria"]["pronunciation"]["score"] < 100


def test_skipped_word_does_not_count_against_the_next(monkeypatch):
    # Where "a" was skipped the model still half-expects it before "glass".
    report, _ = _run(monkeypatch, _reply("I would like glass of water.", {3: {"a": 0.003}}))
    assert report["word_results"][3]["status"] == "omitted"
    glass = report["word_results"][4]
    assert glass["pronunciation"]["score"] == 100 and "mispronunciation" not in glass["issues"]
    assert [e["type"] for e in report["errors"]] == ["omission"]


def test_skipped_and_misread_words_carry_no_pronunciation_score(monkeypatch):
    report, calls = _run(monkeypatch, _reply("I would like a cup of."))
    by = {w["text"]: w for w in report["word_results"]}
    assert by["water"]["status"] == "omitted" and by["water"]["pronunciation"] is None
    assert by["glass"]["status"] == "substituted" and by["glass"]["pronunciation"] is None
    assert len(calls) == 1 and report["asr"]["calls"] == 1


def test_hyphenated_and_split_words_are_scored_on_their_own_tokens(monkeypatch):
    # "twenty-one": "one" sits on the "one" token, not on "twenty".
    reply = {"message": {"content": "language English<asr_text>I am twenty-one."},
             "logprobs": [_tok("language"), _tok(" English"), _tok("<asr_text>"), _tok("I"), _tok(" am"),
                          _tok(" twenty"), _tok("-"), _tok("one"), _tok(".")]}
    report, _ = _run(monkeypatch, reply, acoustics=_acoustics("I am twenty one."))
    assert [w["pronunciation"]["score"] for w in report["word_results"]] == [100, 100, 100, 100]
    assert not report["errors"]
    # "cannot" for "can not": the joined token is the target, not a rival.
    report, _ = _run(monkeypatch, _reply("I cannot swim.", {1: {"can": 0.0001}}), acoustics=_acoustics("I can not swim."))
    words = report["word_results"]
    # (Floor = the list's own lowest entry here: a real list has 20.)
    assert words[1]["pronunciation"]["score"] > 90 and words[2]["pronunciation"] is None
    assert not [e for e in report["errors"] if e["type"] == "mispronunciation"]


def test_contraction_read_right_is_correct(monkeypatch):
    reply = {"message": {"content": "language English<asr_text>I don't like it."},
             "logprobs": [_tok("language"), _tok(" English"), _tok("<asr_text>"), _tok("I"), _tok(" don"),
                          _tok("'t"), _tok(" like"), _tok(" it"), _tok(".")]}
    report, _ = _run(monkeypatch, reply, acoustics=_acoustics("I don't like it."))
    assert [w["status"] for w in report["word_results"]] == ["correct"] * 4
    assert report["criteria"]["content_accuracy"]["score"] == 100
    assert report["word_results"][1]["pronunciation"]["score"] == 100


def test_shorter_real_word_is_a_rival_but_the_next_word_is_not(monkeypatch):
    # Said "a apple", repaired to "an": "a" is the error, not a fragment of "an".
    report, _ = _run(monkeypatch, _reply("I want an apple.", {2: {"a": 0.07}}), acoustics=_acoustics("I want an apple."))
    an = report["word_results"][2]
    assert an["pronunciation"]["rival"] == "a" and an["pronunciation"]["score"] < v2.MISPRONOUNCED
    # The model wondering whether "and" was skipped ("five" next) is no rival.
    report, _ = _run(monkeypatch, _reply("bread and five.", {1: {"five": 0.0013}}), acoustics=_acoustics("bread and five."))
    assert report["word_results"][1]["pronunciation"]["score"] == 100


def test_other_language_is_reported_as_such(monkeypatch):
    reply = {"message": {"content": "language Japanese<asr_text>私は水を一杯飲みたいです。"}}
    report, _ = _run(monkeypatch, reply)
    assert report["status"] == "language_mismatch" and report["language_mismatch"] is True
    assert [i["code"] for i in report["improvements"]] == ["wrong_language"]
    assert "Japanese" in report["improvements"][0]["message"]


def test_spelling_off_word_is_flagged(monkeypatch):
    report, _ = _run(monkeypatch, _reply("I would like a glass of wata."))
    water = report["word_results"][6]
    assert water["status"] == "mispronounced" and "mispronunciation" in water["issues"]
    assert ("mispronunciation", "water", "wata") in [(e["type"], e["word"], e["heard"]) for e in report["errors"]]


def test_reading_other_words_gates_prosody_and_scores_content_only(monkeypatch):
    report, _ = _run(monkeypatch, _reply("The cat is sleeping on the sofa today."))
    c = report["criteria"]
    assert report["status"] == "content_mismatch"
    assert all(c[k]["status"] == "gated" and c[k]["score"] is None for k in ("word_stress", "intonation", "fluency"))
    assert report["overall_score"] == c["content_accuracy"]["score"] and report["weights_used"] == 0.25


def test_repeats_and_fillers_are_fluency_errors(monkeypatch):
    report, _ = _run(monkeypatch, _reply("I would would like uh a glass of water."))
    kinds = sorted(e["type"] for e in report["errors"])
    assert kinds == ["filler", "repetition"]
    assert report["criteria"]["content_accuracy"]["score"] == 100


def test_without_token_alternatives_pronunciation_is_missing_not_zero(monkeypatch):
    report, _ = _run(monkeypatch, _reply("I would like a glass of water.", logprobs=False))
    p = report["criteria"]["pronunciation"]
    assert p["status"] == "insufficient_evidence" and p["score"] is None and p["reason"] == "no_token_alternatives"
    assert report["overall_score"] is None and report["status"] == "insufficient_evidence"


def test_no_word_timing_leaves_word_stress_and_intonation_unassessed(monkeypatch):
    report, _ = _run(monkeypatch, _reply("I would like a glass of water."), acoustics=_acoustics(timing=None))
    assert report["criteria"]["word_stress"]["reason"] == "no_word_timing"
    assert report["criteria"]["intonation"]["reason"] == "no_word_timing"
    assert report["alignment"] == {"method": None, "aligned": False}
    assert all(w["start_seconds"] is None for w in report["word_results"])


def test_no_voice_and_asr_down(monkeypatch):
    report, calls = _run(monkeypatch, _reply("x"), acoustics=_acoustics(voiced=False))
    assert report["status"] == "no_voice" and report["overall_score"] == 0 and calls == []
    report, _ = _run(monkeypatch, None, error=qwen3_asr.AsrError("unavailable", "connection refused"))
    assert report["status"] == "asr_unavailable" and report["overall_score"] is None
    assert report["asr"]["status"] == "unavailable"
    assert all(c["score"] is None for c in report["criteria"].values())


def test_feedback_from_ollama_chat_model(monkeypatch):
    from app.services import coaching

    seen = {}

    async def fake_comment(facts, settings, lang="vi", target_lang="ja"):
        seen.update(facts=facts, lang=lang, target_lang=target_lang)
        return coaching.CoachingComment("Đọc rõ ràng.", "Chú ý 'glass'.")

    monkeypatch.setattr(coaching, "generate_comment", fake_comment)
    report, _ = _run(monkeypatch, _reply("I would like a glass of water.", {4: {"grass": 0.0013}}), feedback=True)
    fb = report["feedback"]
    assert fb["status"] == "generated" and fb["assessment"] == "Đọc rõ ràng." and fb["model"] == "qwen3:8b"
    assert seen["target_lang"] == "en" and seen["facts"].facts_only
    lines = "\n".join(seen["facts"].extra_lines)
    assert "grass" in lines and "Lời khuyên đã tính" in lines
    from app.services.coaching import _format_facts

    block = _format_facts(seen["facts"], target_lang="en")
    assert "Không có từ nào đọc sai" not in block and "dữ liệu cao độ" not in block

    async def down(*_a, **_k):
        raise coaching.CoachingUnavailableError("Model 'qwen3:8b' chưa được tải")

    monkeypatch.setattr(coaching, "generate_comment", down)
    report, _ = _run(monkeypatch, _reply("I would like a glass of water."), feedback=True)
    assert report["feedback"]["status"] == "unavailable" and "chưa được tải" in report["feedback"]["error"]
    report, _ = _run(monkeypatch, _reply("I would like a glass of water."), feedback=True, en_v2_feedback=False)
    assert report["feedback"] is None


# --- Qwen3-ASR pieces ------------------------------------------------------

def test_word_evidence_margin():
    clean = qwen3_asr.word_evidence(_tok(" glass", 1 - 1e-9, {" glasses": 1e-9})["top_logprobs"], "glass")
    assert clean["score"] == 100
    repaired = qwen3_asr.word_evidence(_tok(" glass", 0.9986, {" grass": 0.0013})["top_logprobs"], "glass")
    assert repaired["rival"] == "grass" and 60 < repaired["score"] < 70
    missing = qwen3_asr.word_evidence(_tok(" cup", 0.99, {" mug": 0.01})["top_logprobs"], "glass")
    assert missing["score"] == 0 and missing["p_target"] is None
    # Fragments: "gl" is the target's own start, "gr" a rival's.
    frag = qwen3_asr.word_evidence(_tok(" gl", 0.6, {" gr": 0.4})["top_logprobs"], "glass")
    assert frag["rival"] == "gr" and frag["score"] < 10
    homophone = qwen3_asr.word_evidence(_tok(" meat", 0.9, {" mean": 0.1})["top_logprobs"], "meet",
                                        same_word=lambda a, b: {a, b} == {"meat", "meet"})
    assert homophone["rival"] == "mean" and homophone["p_target"] == pytest.approx(0.9, abs=1e-3)
    ignored = qwen3_asr.word_evidence(_tok(" glass", 0.997, {" a": 0.003})["top_logprobs"], "glass", ignore=["a"])
    assert ignored["score"] == 100 and ignored["rival"] is None


def test_stream_positions_and_tag_split():
    data = {"logprobs": [_tok("language"), _tok(" English<"), _tok("asr_"), _tok("text>I"),
                         _tok(" would"), _tok(" like"), _tok(" water"), _tok(".")]}
    stream = qwen3_asr.stream_after_tag(data)
    assert [t["token"] for t in stream] == ["I", " would", " like", " water", "."]
    starts, token_at = qwen3_asr.stream_positions(stream, "I would like water.")
    assert starts == [0, 2, 8, 13] and [token_at(c) for c in starts] == [0, 1, 2, 3]
    assert qwen3_asr.stream_positions(stream, "I would like wine.") is None


def test_word_evidence_guards():
    # Only skipped words / punctuation listed: no evidence, not 100 or a crash.
    assert qwen3_asr.word_evidence([{"token": " a", "logprob": -0.05}, {"token": ".", "logprob": -3.0}],
                                   "of", ignore=["a"])["score"] is None
    assert qwen3_asr.word_evidence([{"token": " a", "logprob": -0.01}, {"token": "a", "logprob": -6.0}],
                                   "glass", ignore=["a"])["score"] is None
    # A dropped ending is a rival when "walk" is a real word.
    ev = qwen3_asr.word_evidence(_tok(" walked", 0.986, {" walk": 0.014})["top_logprobs"], "walked",
                                 is_word=lambda w: w in {"walk", "a"})
    assert ev["rival"] == "walk" and ev["score"] < v2.MISPRONOUNCED


def test_base_url_accepts_ollama_host_forms():
    assert qwen3_asr.base_url("127.0.0.1:11500") == "http://127.0.0.1:11500"
    assert qwen3_asr.base_url("0.0.0.0") == "http://127.0.0.1:11434"
    assert qwen3_asr.base_url("http://localhost:11434/") == "http://localhost:11434"
    assert qwen3_asr.base_url("https://ollama.example.com") == "https://ollama.example.com:443"


def test_silence_is_trimmed_and_windows_cut_in_pauses():
    sr = 16000
    speech = np.sin(np.linspace(0, 400 * np.pi, sr)).astype(np.float32) * 0.5
    clip = np.concatenate([np.zeros(29 * sr, dtype=np.float32), speech, np.zeros(10 * sr, dtype=np.float32)])
    trimmed = qwen3_asr.trim_silence(clip, sr)
    assert 1.0 <= len(trimmed) / sr <= 1.7
    # 26 s of speech, a pause at 27-28 s, more speech: cut in the pause, not at 30 s.
    loud = np.ones(26 * sr, dtype=np.float32) * 0.3
    long = np.concatenate([loud, np.zeros(2 * sr, dtype=np.float32), loud])
    first = qwen3_asr.audio_windows(long, sr)[0]
    assert 26 <= len(first) / sr <= 28


def test_multi_window_streams_are_joined(monkeypatch):
    replies = iter([_reply("I would like"), _reply("a glass of water.")])

    async def fake_chat(_settings, wav, **_kw):
        return next(replies)

    monkeypatch.setattr(qwen3_asr, "chat", fake_chat)
    monkeypatch.setattr(qwen3_asr, "trim_silence", lambda s, sr: s)
    heard = asyncio.run(qwen3_asr.transcribe(np.ones(16000 * 50, dtype=np.float32) * 0.1, 16000, _settings()))
    assert heard["transcript"] == "I would like a glass of water." and heard["calls"] == 2
    assert qwen3_asr.stream_positions(heard["stream"], heard["transcript"]) is not None


def test_contraction_for_a_full_form_target(monkeypatch):
    reply = {"message": {"content": "language English<asr_text>It's a nice day."},
             "logprobs": [_tok("language"), _tok(" English"), _tok("<asr_text>"), _tok("It"), _tok("'s"),
                          _tok(" a"), _tok(" nice"), _tok(" day"), _tok(".")]}
    report, _ = _run(monkeypatch, reply, acoustics=_acoustics("It is a nice day."))
    assert [w["status"] for w in report["word_results"]] == ["correct"] * 5
    assert not [e for e in report["errors"] if e["type"] in ("omission", "substitution")]


def test_compounds_spellings_titles_and_number_and(monkeypatch):
    # "any" is how "any time" starts: no rival for "anytime".
    # (A real top-20 list runs down to tiny rivals; "zebra" stands for them.)
    report, _ = _run(monkeypatch, _reply("You can call me anytime.", {4: {"any": 0.3, "zebra": 1e-9}}),
                     acoustics=_acoustics("You can call me anytime."))
    assert report["word_results"][4]["pronunciation"]["score"] == 100 and not report["errors"]
    # An accepted spelling is the word itself.
    report, _ = _run(monkeypatch, _reply("I love the flavor.", {3: {"flavour": 0.2, "zebra": 1e-9}}),
                     acoustics=_acoustics("I love the flavour."))
    assert report["word_results"][3]["status"] == "correct" and not report["errors"]
    # Titles / numbers in the transcript are expanded like the target.
    report, _ = _run(monkeypatch, _reply("Mr. Smith is here."), acoustics=_acoustics("Mister Smith is here."))
    assert report["criteria"]["content_accuracy"]["score"] == 100
    report, _ = _run(monkeypatch, _reply("Room two hundred and five."), acoustics=_acoustics("Room 205."))
    assert report["criteria"]["content_accuracy"]["score"] == 100 and not report["errors"]


def test_word_lost_by_the_aligner_gives_no_tone_advice(monkeypatch):
    target = "my teacher is beautiful"
    intonation = _intonation(target, {1: True, 3: True}, {1: True, 3: False}, False, pauses=[0.0, 0.0, 0.0])
    report, _ = _run(monkeypatch, _reply("My teacher is beautiful."),
                     acoustics=_acoustics(target, timing=_timing(target, missing={3}), intonation=intonation))
    codes = [i["code"] for i in report["improvements"]]
    assert "final_tone_fall" not in codes and "weak_sentence_stress" not in codes


def test_language_mismatch_lists_no_omissions(monkeypatch):
    report, _ = _run(monkeypatch, {"message": {"content": "language Japanese<asr_text>水をください。"}})
    assert report["status"] == "language_mismatch" and report["errors"] == []
    lines = "\n".join(v2.feedback_lines(report))
    assert "Bỏ sót" not in lines and "Japanese" in lines


def test_timeout_is_reported_as_such(monkeypatch):
    import httpx

    def handler(request):
        raise httpx.ReadTimeout("read timed out", request=request)

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(qwen3_asr.AsrError) as e:
        asyncio.run(qwen3_asr.chat(_settings(), b"RIFF"))
    assert "không trả lời trong" in str(e.value) and "ReadTimeout" in str(e.value)


def test_window_heard_as_another_language_is_decoded_again_as_english(monkeypatch):
    seen = []

    async def fake_chat(_settings, wav, **kw):
        seen.append(kw.get("language"))
        if kw.get("language"):
            return _reply("I would like a glass of water.")
        return {"message": {"content": "language Japanese<asr_text>アイウッドライク。"}}

    monkeypatch.setattr(qwen3_asr, "chat", fake_chat)
    report = asyncio.run(v2.evaluate_v2(_acoustics(), _settings(), with_feedback=False))
    assert seen == [None, "English"]
    assert report["recognized_text"] == "I would like a glass of water."
    assert report["status"] != "language_mismatch" and report["language_mismatch"] is False
    assert report["asr"]["calls"] == 2


def test_no_speech_is_not_forced_to_english(monkeypatch):
    seen = []

    async def fake_chat(_settings, wav, **kw):
        seen.append(kw.get("language"))
        return {"message": {"content": "language None<asr_text>"}}

    monkeypatch.setattr(qwen3_asr, "chat", fake_chat)
    monkeypatch.setattr(qwen3_asr, "trim_silence", lambda s, sr: s)
    heard = asyncio.run(qwen3_asr.transcribe(np.ones(16000, dtype=np.float32) * 0.1, 16000, _settings(), "English"))
    assert seen == [None] and heard["transcript"] == "" and heard["language"] is None


def test_without_a_target_language_the_detected_one_is_kept(monkeypatch):
    seen = []

    async def fake_chat(_settings, wav, **kw):
        seen.append(kw.get("language"))
        return {"message": {"content": "language Japanese<asr_text>水をください。"}}

    monkeypatch.setattr(qwen3_asr, "chat", fake_chat)
    monkeypatch.setattr(qwen3_asr, "trim_silence", lambda s, sr: s)
    heard = asyncio.run(qwen3_asr.transcribe(np.ones(16000, dtype=np.float32) * 0.1, 16000, _settings()))
    assert seen == [None] and heard["language"] == "Japanese" and heard["calls"] == 1


def test_forced_language_is_prefilled_and_put_back_in_the_reply(monkeypatch):
    import httpx

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "I would like water."},
            "logprobs": [_tok("I"), _tok(" would"), _tok(" like"), _tok(" water"), _tok(".")],
        })

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    data = asyncio.run(qwen3_asr.chat(_settings(), b"RIFF", language="English"))
    assert sent[0]["messages"][-1] == {"role": "assistant", "content": "language English<asr_text>"}
    assert qwen3_asr.parse_reply(data["message"]["content"]) == ("English", "I would like water.")
    stream = qwen3_asr.stream_after_tag(data)
    assert [t["token"] for t in stream] == ["I", " would", " like", " water", "."]
    assert qwen3_asr.raw_word_confidences(data["logprobs"], "I would like water.") == [1.0, 1.0, 1.0, 1.0]
