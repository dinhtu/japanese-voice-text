"""zh / ko /evaluate-v2 (Qwen3-ASR transcript, mocked at the Ollama call)
and the Korean /evaluate?coach=true, with audio, VAD, CTC and F0 stubbed."""
import sys
import types

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import chinese_routes, korean_routes
from app.services import coaching, korean_use_cases, qwen3_asr, vad_fluency, voice_presence
from app.services.chinese_asr import ChineseRecognition, get_chinese_asr_service
from app.services.korean_asr import KoreanRecognition, get_korean_asr_service
from app.services.vad_fluency import VadResult
from app.services.voice_presence import VoicePresence

WAV = ("a.wav", b"RIFF0000WAVE", "audio/wav")


class _FakeZhASR:
    """The CTC model: its own transcript differs from Qwen3-ASR's on purpose."""

    def __init__(self, text="你跑吗", windows=((0.0, 0.3), (0.3, 0.6), (0.6, 0.9))):
        self.text, self.windows, self.calls = text, list(windows) if windows else None, 0

    def recognize(self, path, align_to=""):
        self.calls += 1
        return ChineseRecognition(self.text, 1.0, 0.01, self.windows)


class _FakeKoASR:
    def __init__(self, text="감사합니다", windows=None, whisper=False):
        self.text, self.windows, self.is_whisper, self.calls = text, windows, whisper, 0

    def recognize(self, path, align_to=""):
        self.calls += 1
        return KoreanRecognition(self.text, 1.0, 0.02, self.windows)


def _stub_audio(monkeypatch, verdict="voice", contours=None):
    samples = (np.zeros(16_000, dtype=np.float32), 16_000)
    inference = types.ModuleType("src.asr.inference")
    inference.load_audio = lambda path: samples
    monkeypatch.setitem(sys.modules, "src.asr.inference", inference)
    monkeypatch.setattr(korean_use_cases, "load_audio", lambda path: samples)
    monkeypatch.setattr(vad_fluency, "analyze_vad", lambda s, sr: VadResult(0.9, 0, 0.9, "silero"))
    monkeypatch.setattr(voice_presence, "check_voice", lambda s, sr, vad: VoicePresence(verdict, 0.9, 0.5))
    pitch = types.ModuleType("app.services.pitch_extraction")
    pitch.extract_mandarin_tone_contours = lambda s, sr, windows: contours or [[None] * 3 for _ in windows]
    pitch.extract_pitch_for_windows = lambda s, sr, windows: [{"semitone": None, "voiced": False} for _ in windows]
    pitch.extract_pitch_per_mora = lambda s, sr, n: [{"semitone": None, "voiced": False} for _ in range(n)]
    monkeypatch.setitem(sys.modules, "app.services.pitch_extraction", pitch)


def _stub_qwen(monkeypatch, transcript="", language="Chinese", error=None):
    """Qwen3-ASR replies `transcript`; records each call's forced language."""
    calls = []

    async def fake_chat(_settings, wav, **kw):
        calls.append(kw.get("language"))
        if error is not None:
            raise error
        return {"message": {"content": f"language {language}<asr_text>{transcript}"}}

    monkeypatch.setattr(qwen3_asr, "chat", fake_chat)
    return calls


def _stub_coach(monkeypatch, error=None):
    calls = []

    async def fake_comment(facts, settings, lang="vi", target_lang="ja"):
        calls.append((facts, lang, target_lang))
        if error is not None:
            raise error
        return coaching.CoachingComment("Tốt.", "Luyện thêm.")

    monkeypatch.setattr(chinese_routes, "generate_comment", fake_comment)
    monkeypatch.setattr(korean_routes, "generate_comment", fake_comment)
    return calls


def _client(zh_asr=None, ko_asr=None):
    app = FastAPI()
    app.include_router(chinese_routes.router, prefix="/zh")
    app.include_router(korean_routes.router, prefix="/ko")
    app.dependency_overrides[get_chinese_asr_service] = lambda: zh_asr or _FakeZhASR()
    app.dependency_overrides[get_korean_asr_service] = lambda: ko_asr or _FakeKoASR()
    return TestClient(app)


def _post(client, url, text, **data):
    return client.post(url, data={"text": text, **data}, files={"audio": WAV})


# --- Chinese v2 --------------------------------------------------------------

def test_zh_v2_scores_the_qwen3_transcript_with_ctc_timing(monkeypatch):
    _stub_audio(monkeypatch, contours=[[-1.0, 0.0, 2.0], [1.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    asked = _stub_qwen(monkeypatch, "你好吗？")
    coach = _stub_coach(monkeypatch)
    zh = _FakeZhASR()
    r = _post(_client(zh_asr=zh), "/zh/evaluate-v2", "你好吗？", lang="jp")
    assert r.status_code == 200, r.text
    body = r.json()
    # Scored on Qwen3-ASR's "你好吗", not the CTC model's "你跑吗".
    assert body["score"] == 100 and body["recognized_text"] == "你好吗？"
    assert [m["ok"] for m in body["mora_status"]] == [True, True, True]
    assert [t["status"] for t in body["tones"]] == ["ok", "ok", "neutral"]
    assert body["coach"] == {"assessment": "Tốt.", "suggestion": "Luyện thêm.", "lang": "ja"}
    assert body["coach_error"] is None and coach[0][2] == "zh"
    v2 = body["v2_assessment"]
    assert v2["scoring_version"] == "zh-reading-v2" and v2["status"] == "scored"
    assert v2["asr"]["engine"] == "qwen3-asr" and v2["asr"]["language"] == "Chinese"
    assert v2["asr"]["transcript"] == "你好吗？" and v2["asr"]["calls"] == 1
    assert v2["alignment"]["method"] == "ctc_forced_alignment" and v2["alignment"]["used_for_timing"] is True
    assert v2["alignment"]["transcript"] == "你跑吗"
    assert asked == [None] and zh.calls == 1
    # Every v1 field is there.
    v1 = _post(_client(), "/zh/evaluate", "你好吗？").json()
    assert set(v1) <= set(body)


def test_zh_v2_without_feedback_does_not_call_the_coach(monkeypatch):
    _stub_audio(monkeypatch)
    _stub_qwen(monkeypatch, "你好吗")
    coach = _stub_coach(monkeypatch)
    body = _post(_client(), "/zh/evaluate-v2", "你好吗", feedback="false").json()
    assert body["coach"] is None and body["coach_error"] is None and coach == []


def test_zh_v2_coach_failure_is_coach_error(monkeypatch):
    _stub_audio(monkeypatch)
    _stub_qwen(monkeypatch, "你好吗")
    _stub_coach(monkeypatch, error=coaching.CoachingUnavailableError("Ollama down"))
    body = _post(_client(), "/zh/evaluate-v2", "你好吗").json()
    assert body["coach"] is None and body["coach_error"] == "Ollama down" and body["score"] == 100


def test_zh_v2_other_language_is_decoded_again_as_chinese(monkeypatch):
    _stub_audio(monkeypatch)
    asked = _stub_qwen(monkeypatch, "你好吗", language="Japanese")
    body = _post(_client(), "/zh/evaluate-v2", "你好吗", feedback="false").json()
    assert asked == [None, "Chinese"] and body["v2_assessment"]["asr"]["calls"] == 2


def test_zh_v2_qwen3_asr_down_is_503(monkeypatch):
    _stub_audio(monkeypatch)
    _stub_qwen(monkeypatch, error=qwen3_asr.AsrError("unavailable", "Không kết nối được Ollama"))
    coach = _stub_coach(monkeypatch)
    r = _post(_client(), "/zh/evaluate-v2", "你好吗")
    assert r.status_code == 503 and "Ollama" in r.json()["detail"] and coach == []


def test_zh_v2_silent_take_is_not_sent_to_qwen3(monkeypatch):
    _stub_audio(monkeypatch, verdict="silent")
    asked = _stub_qwen(monkeypatch, "你好吗")
    zh = _FakeZhASR()
    body = _post(_client(zh_asr=zh), "/zh/evaluate-v2", "你好吗", feedback="false").json()
    assert asked == [] and zh.calls == 0
    assert body["score"] == 0 and body["aspect_method"] == "no-voice:silent"
    assert body["v2_assessment"]["status"] == "no_voice" and body["v2_assessment"]["asr"]["status"] == "skipped"


def test_zh_v2_ctc_failure_only_costs_the_timing(monkeypatch):
    _stub_audio(monkeypatch)
    _stub_qwen(monkeypatch, "你好吗")

    class Broken(_FakeZhASR):
        def recognize(self, path, align_to=""):
            raise OSError("model missing")

    body = _post(_client(zh_asr=Broken()), "/zh/evaluate-v2", "你好吗", feedback="false").json()
    assert body["score"] == 100
    assert [t["status"] for t in body["tones"]] == ["unmeasured", "unmeasured", "neutral"]
    assert body["v2_assessment"]["alignment"]["used_for_timing"] is False
    assert body["v2_assessment"]["alignment"]["method"] is None


def test_zh_v2_rejects_unknown_lang(monkeypatch):
    _stub_audio(monkeypatch)
    assert _post(_client(), "/zh/evaluate-v2", "你好吗", lang="xx").status_code == 400


# --- Korean v2 -------------------------------------------------------------

def test_ko_v2_scores_the_qwen3_transcript(monkeypatch):
    _stub_audio(monkeypatch)
    asked = _stub_qwen(monkeypatch, "감사합니다.", language="Korean")
    coach = _stub_coach(monkeypatch)
    windows = [(k * 0.2, (k + 1) * 0.2) for k in range(5)]
    ko = _FakeKoASR(text="감사함미다", windows=windows)
    r = _post(_client(ko_asr=ko), "/ko/evaluate-v2", "감사합니다", lang="en")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["score"] == 100 and body["recognized_text"] == "감사합니다."
    assert len(body["mora_status"]) == 5 and all(m["ok"] for m in body["mora_status"])
    assert body["recognized_pitch_pattern"] and body["target_reading"] == "감사함니다"
    assert body["coach"]["lang"] == "en" and coach[0][2] == "ko"
    v2 = body["v2_assessment"]
    assert v2["scoring_version"] == "ko-reading-v2" and v2["asr"]["language"] == "Korean"
    assert v2["alignment"] == {"model": v2["alignment"]["model"], "method": "ctc_forced_alignment",
                               "used_for_timing": True, "transcript": "감사함미다", "inference_ms": 20.0}
    assert asked == [None] and ko.calls == 1
    v1 = _post(_client(), "/ko/evaluate", "감사합니다").json()
    assert set(v1) <= set(body)


def test_ko_v2_with_whisper_does_not_run_it(monkeypatch):
    _stub_audio(monkeypatch)
    _stub_qwen(monkeypatch, "감사합니다", language="Korean")
    ko = _FakeKoASR(whisper=True)
    body = _post(_client(ko_asr=ko), "/ko/evaluate-v2", "감사합니다", feedback="false").json()
    assert ko.calls == 0 and body["score"] == 100 and body["coach"] is None
    assert body["v2_assessment"]["alignment"]["method"] == "even_split"
    assert body["v2_assessment"]["alignment"]["used_for_timing"] is False


def test_ko_v2_misread_syllable_is_reported(monkeypatch):
    _stub_audio(monkeypatch)
    _stub_qwen(monkeypatch, "감사합니까", language="Korean")
    body = _post(_client(), "/ko/evaluate-v2", "감사합니다", feedback="false").json()
    assert body["score"] < 100 and [m["ok"] for m in body["mora_status"]] == [True, True, True, True, False]


def test_ko_v2_qwen3_asr_down_is_503(monkeypatch):
    _stub_audio(monkeypatch)
    _stub_qwen(monkeypatch, error=qwen3_asr.AsrError("unavailable", "Model chưa được tải"))
    r = _post(_client(), "/ko/evaluate-v2", "감사합니다")
    assert r.status_code == 503 and r.json()["detail"] == "Model chưa được tải"


def test_ko_v2_silent_take_is_the_no_voice_result(monkeypatch):
    _stub_audio(monkeypatch, verdict="whisper")
    asked = _stub_qwen(monkeypatch, "감사합니다", language="Korean")
    coach = _stub_coach(monkeypatch)
    ko = _FakeKoASR()
    body = _post(_client(ko_asr=ko), "/ko/evaluate-v2", "감사합니다").json()
    assert asked == [] and ko.calls == 0
    assert body["score"] == 0 and body["aspect_method"] == "no-voice:whisper"
    assert body["feedback"]["message"] == voice_presence.NO_VOICE_MESSAGES["whisper"]
    assert body["v2_assessment"]["status"] == "no_voice"
    # The coach is told nothing was heard, not that every syllable is missing.
    facts = coach[0][0]
    assert facts.facts_only and facts.extra_lines == [voice_presence.NO_VOICE_FACT]


# --- Korean v1 /evaluate?coach= and /coach -----------------------------------

def test_ko_evaluate_returns_coach_in_one_request(monkeypatch):
    _stub_audio(monkeypatch)
    coach = _stub_coach(monkeypatch)
    ko = _FakeKoASR()
    body = _post(_client(ko_asr=ko), "/ko/evaluate", "감사합니다", coach="true", lang="jp").json()
    assert body["coach"] == {"assessment": "Tốt.", "suggestion": "Luyện thêm.", "lang": "ja"}
    assert body["score"] == 100 and ko.calls == 1
    facts, lang, target = coach[0]
    assert (lang, target) == ("ja", "ko") and facts.text == "감사합니다"


def test_ko_evaluate_without_coach_is_unchanged(monkeypatch):
    _stub_audio(monkeypatch)
    coach = _stub_coach(monkeypatch)
    body = _post(_client(), "/ko/evaluate", "감사합니다").json()
    assert body["coach"] is None and body["coach_error"] is None and coach == []
    assert "v2_assessment" not in body and body["score"] == 100


def test_ko_evaluate_reports_coach_error_instead_of_failing(monkeypatch):
    _stub_audio(monkeypatch)
    _stub_coach(monkeypatch, error=coaching.CoachingUnavailableError("Ollama down"))
    body = _post(_client(), "/ko/evaluate", "감사합니다", coach="true").json()
    assert body["coach"] is None and body["coach_error"] == "Ollama down"


@pytest.mark.parametrize("url", ["/ko/evaluate", "/ko/coach"])
def test_ko_rejects_unknown_coach_lang(monkeypatch, url):
    _stub_audio(monkeypatch)
    assert _post(_client(), url, "감사합니다", coach="true", lang="xx").status_code == 400


def test_ko_coach_endpoint_is_unchanged(monkeypatch):
    _stub_audio(monkeypatch)
    coach = _stub_coach(monkeypatch)
    r = _post(_client(ko_asr=_FakeKoASR(text="감사함미다")), "/ko/coach", "감사합니다", lang="vi")
    assert r.status_code == 200
    assert r.json() == {"success": True, "assessment": "Tốt.", "suggestion": "Luyện thêm.", "lang": "vi"}
    facts = coach[0][0]
    assert not facts.facts_only and facts.extra_lines == []
    assert facts.wrong_morae == ["합→함", "니→미"]


def test_ko_coach_ollama_down_is_503(monkeypatch):
    _stub_audio(monkeypatch)
    _stub_coach(monkeypatch, error=coaching.CoachingUnavailableError("Ollama down"))
    assert _post(_client(), "/ko/coach", "감사합니다").status_code == 503


# --- Arabic numbers in the Qwen3-ASR transcript ------------------------------

@pytest.mark.parametrize("target,heard", [
    ("我今年二十岁", "我今年20岁"),
    ("我有三个苹果", "我有3个苹果"),
    ("我有两个苹果", "我有2个苹果"),
    ("二零二四年", "2024年"),
    ("现在三点半", "现在3:30"),
])
def test_zh_v2_digits_read_correctly_score_100(monkeypatch, target, heard):
    _stub_audio(monkeypatch)
    _stub_qwen(monkeypatch, heard)
    n = len(target)
    zh = _FakeZhASR(windows=[(k * 0.1, (k + 1) * 0.1) for k in range(n)])
    body = _post(_client(zh_asr=zh), "/zh/evaluate-v2", target, feedback="false").json()
    assert body["score"] == 100, body
    assert body["recognized_text"] == heard and body["recognized_hiragana"] == target
    assert all(m["ok"] for m in body["mora_status"])
    assert "misread" not in {t["status"] for t in body["tones"]}


def test_zh_v2_wrong_number_is_still_an_error(monkeypatch):
    _stub_audio(monkeypatch)
    _stub_qwen(monkeypatch, "我有4个苹果")
    body = _post(_client(zh_asr=_FakeZhASR(windows=None)), "/zh/evaluate-v2", "我有三个苹果",
                 feedback="false").json()
    assert body["score"] < 100 and [m["ok"] for m in body["mora_status"]][2] is False


@pytest.mark.parametrize("target,heard", [
    ("사과 세 개", "사과 3개"),
    ("세 시 반이에요", "3:30이에요"),
    ("삼월 이십일", "3월 20일"),
    ("스무 살", "20살"),
])
def test_ko_v2_digits_read_correctly_score_100(monkeypatch, target, heard):
    _stub_audio(monkeypatch)
    _stub_qwen(monkeypatch, heard, language="Korean")
    body = _post(_client(), "/ko/evaluate-v2", target, feedback="false").json()
    assert body["score"] == 100, body
    assert body["recognized_text"] == heard
    assert all(m["ok"] for m in body["mora_status"])
    assert "".join(m["mora"] for m in body["recognized_pitch_pattern"]) == "".join(target.split())


def test_numbers_in_the_target_are_left_alone():
    from app.services.cjk_numbers import spell_numbers_ko, spell_numbers_zh

    assert spell_numbers_zh("我有3个", "我有3个") == "我有3个"
    assert spell_numbers_ko("3개", "3개") == "3개"


@pytest.mark.parametrize("n,zh,ko", [
    (10, "十", "십"), (15, "十五", "십오"), (105, "一百零五", "백오"), (110, "一百一十", "백십"),
    (1005, "一千零五", "천오"), (10010, "一万零一十", "만십"), (2024, "二千零二十四", "이천이십사"),
])
def test_cardinals(n, zh, ko):
    from app.services.cjk_numbers import ko_sino, zh_cardinal

    assert zh_cardinal(n) == zh and ko_sino(n) == ko
