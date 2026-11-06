import sys
import types

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import chinese_routes
from app.services import coaching, vad_fluency, voice_presence
from app.services.chinese_asr import ChineseRecognition, get_chinese_asr_service
from app.services.chinese_text import (
    chinese_pitch_pattern,
    normalize_chinese,
    pinyin_syllables,
    score_syllables,
    tone_check,
)
from app.services.chinese_use_cases import EvaluateChineseUseCase
from app.services.vad_fluency import VadResult
from app.services.voice_presence import VoicePresence


def test_tone_check_expected_shapes():
    shapes = [[0.0, 0.1, 0.0], [-1.0, 0.0, 2.0], [1.0, -1.0, 1.0], [2.0, 0.0, -2.0]]
    assert [tone_check(v, t) for v, t in zip(shapes, [1, 2, 3, 4])] == [
        (True, 1), (True, 2), (True, 3), (True, 4)
    ]


def test_tone_check_names_the_tone_that_was_said():
    # A 4th-tone fall where a 1st (level) tone was expected.
    assert tone_check([2.0, 0.0, -2.0], 1) == (False, 4)


def test_tone_check_skips_unmeasured_and_neutral():
    assert tone_check([None, 0.0, 1.0], 2) == (None, None)
    assert tone_check([0.0, 0.0, 0.0], 5) == (None, None)


def test_half_third_is_accepted_mid_phrase_only():
    half = [0.5, -1.5, -2.0]  # 21: falls low and stays there
    assert tone_check(half, 3, final=False)[0] is True
    assert tone_check(half, 3, final=True)[0] is False


def test_mid_phrase_third_tone_still_rejects_level_and_high_fall():
    assert tone_check([2.0, 2.0, 2.0], 3, final=False) == (False, 1)
    assert tone_check([2.0, 0.0, -2.0], 3, final=False) == (False, 4)


def test_a_flat_syllable_is_not_a_rise_or_fall():
    assert tone_check([0.0, 0.0, 0.0], 2) == (False, 1)
    assert tone_check([0.0, 0.0, 0.0], 4) == (False, 1)


def test_chinese_pitch_pattern_and_third_tone_sandhi():
    pattern = chinese_pitch_pattern("你好")
    assert [(p["mora"], p["tone"], p["surface_tone"]) for p in pattern] == [
        ("ni", 3, 2), ("hao", 3, 3)
    ]
    assert pattern[0]["contour"] == [3, 5]
    assert [p["pinyin"] for p in pattern] == ["ní", "hǎo"]


def test_no_third_tone_sandhi_across_punctuation():
    pattern = chinese_pitch_pattern("好，好！")
    assert [p["surface_tone"] for p in pattern] == [3, 3]
    assert [p["phrase_final"] for p in pattern] == [True, True]


def test_half_third_contour_mid_phrase():
    pattern = chinese_pitch_pattern("老鼠爱大米")
    assert pattern[1]["surface_tone"] == 3 and pattern[1]["contour"] == [2, 1, 1]
    assert pattern[-1]["contour"] == [2, 1, 4]


def test_yi_and_bu_sandhi():
    def spoken(text):
        return [p["surface_tone"] for p in chinese_pitch_pattern(text)]

    assert spoken("一天") == [4, 1]
    assert spoken("一样") == [2, 4]
    assert spoken("不是") == [2, 4]
    assert spoken("不好") == [4, 3]
    assert spoken("看一看") == [4, 5, 4]
    # Numbers keep yī.
    assert spoken("第一") == [4, 1]
    assert spoken("十一") == [2, 1]
    assert spoken("一二三") == [1, 4, 1]
    assert spoken("十一个") == [2, 1, 4]
    assert spoken("一月一日") == [1, 4, 1, 4]
    # Word-final 一 keeps yī; 一天一天 is not V一V.
    assert spoken("星期一我去")[2] == 1
    assert spoken("唯一方法")[1] == 1
    assert spoken("一天一天") == [4, 1, 4, 1]
    # V不V and erhua are neutral, so they are not judged.
    assert spoken("去不去") == [4, 5, 4]
    assert spoken("一点儿")[2] == 5


def test_normalize_chinese_keeps_hanzi_only():
    assert normalize_chinese("你好， world！") == "你好"


def test_homophone_is_not_penalized_as_wrong_pronunciation():
    expected = pinyin_syllables("是")
    heard = pinyin_syllables("事")
    assert expected == heard == ["shi"]
    assert score_syllables(expected, heard).score == 100


def test_syllable_deletion_has_target_position():
    result = score_syllables(["ni", "hao", "ma"], ["ni", "ma"])
    assert result.distance == 1
    assert [(e.type, e.position, e.target) for e in result.errors] == [("del", 1, "hao")]
    assert result.target_length == 3


# --- the use case, with the audio, VAD, ASR and F0 stubbed ------------------

class _FakeASR:
    def __init__(self, text="你好吗", windows=((0.0, 0.3), (0.3, 0.6), (0.6, 0.9))):
        self.text, self.windows, self.calls = text, list(windows), 0

    def recognize(self, path, align_to=""):
        self.calls += 1
        return ChineseRecognition(self.text, 1.0, 0.01, self.windows)


def _stub_audio(monkeypatch, verdict="voice", contours=None):
    inference = types.ModuleType("src.asr.inference")
    inference.load_audio = lambda path: (np.zeros(16_000, dtype=np.float32), 16_000)
    monkeypatch.setitem(sys.modules, "src.asr.inference", inference)
    monkeypatch.setattr(vad_fluency, "analyze_vad", lambda s, sr: VadResult(0.9, 0, 0.9, "silero"))
    monkeypatch.setattr(voice_presence, "check_voice", lambda s, sr, vad: VoicePresence(verdict, 0.9, 0.5))
    pitch = types.ModuleType("app.services.pitch_extraction")
    pitch.extract_mandarin_tone_contours = lambda s, sr, windows: contours or [[None] * 3 for _ in windows]
    monkeypatch.setitem(sys.modules, "app.services.pitch_extraction", pitch)


def test_silent_take_is_not_scored(monkeypatch):
    _stub_audio(monkeypatch, verdict="silent")
    asr = _FakeASR()
    result = EvaluateChineseUseCase(asr).execute("你好吗", "x.wav")
    assert asr.calls == 0
    assert result.score.score == 0 and result.score.level == "mismatch"
    assert result.aspects.method == "no-voice:silent"
    assert [e.type for e in result.score.errors] == ["del"] * 3
    assert [t["status"] for t in result.tones] == ["unmeasured", "unmeasured", "neutral"]


def test_tone_checks_skip_misread_syllables_and_feed_intonation(monkeypatch):
    # 你(ní, rise) ok; 好 heard as another syllable: not judged; 吗 neutral.
    _stub_audio(monkeypatch, contours=[[-1.0, 0.0, 2.0], [2.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
    result = EvaluateChineseUseCase(_FakeASR("你跑吗")).execute("你好吗", "x.wav")
    assert [t["status"] for t in result.tones] == ["ok", "misread", "neutral"]
    assert result.aspects.intonation_measured is True


def test_wrong_tone_is_reported_with_the_heard_tone(monkeypatch):
    _stub_audio(monkeypatch, contours=[[2.0, 0.0, -2.0], [1.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    result = EvaluateChineseUseCase(_FakeASR()).execute("你好吗", "x.wav")
    first = result.tones[0]
    assert (first["surface_tone"], first["ok"], first["heard_tone"], first["status"]) == (2, False, 4, "bad")


# --- /evaluate with coach=true ----------------------------------------------

def _client(monkeypatch, comment=None, error=None):
    calls = []

    async def fake_comment(facts, settings, lang="vi", target_lang="ja"):
        calls.append((facts, lang, target_lang))
        if error is not None:
            raise error
        return comment or coaching.CoachingComment("Tốt.", "Chú ý 好.")

    _stub_audio(monkeypatch, contours=[[2.0, 0.0, -2.0], [1.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    monkeypatch.setattr(chinese_routes, "generate_comment", fake_comment)
    app = FastAPI()
    app.include_router(chinese_routes.router, prefix="/zh")
    app.dependency_overrides[get_chinese_asr_service] = lambda: _FakeASR()
    return TestClient(app), calls


def _post(client, url, **data):
    return client.post(url, data={"text": "你好吗？", **data},
                       files={"audio": ("a.wav", b"RIFF0000WAVE", "audio/wav")})


def test_evaluate_returns_tones_and_coach_in_one_request(monkeypatch):
    client, calls = _client(monkeypatch)
    r = _post(client, "/zh/evaluate", coach="true", lang="jp")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["coach"] == {"assessment": "Tốt.", "suggestion": "Chú ý 好.", "lang": "ja"}
    assert [t["status"] for t in body["tones"]] == ["bad", "ok", "neutral"]
    facts, lang, target = calls[0]
    assert (lang, target) == ("ja", "zh")
    assert any("Thanh điệu" in line and "你 ní" in line for line in facts.extra_lines)


def test_evaluate_without_coach_does_not_call_ollama(monkeypatch):
    client, calls = _client(monkeypatch)
    body = _post(client, "/zh/evaluate").json()
    assert body["coach"] is None and body["coach_error"] is None and calls == []


def test_evaluate_reports_coach_error_instead_of_failing(monkeypatch):
    client, _ = _client(monkeypatch, error=coaching.CoachingUnavailableError("Ollama down"))
    body = _post(client, "/zh/evaluate", coach="true").json()
    assert body["coach"] is None and body["coach_error"] == "Ollama down"
    assert body["score"] == 100


def test_evaluate_rejects_unknown_coach_lang(monkeypatch):
    client, _ = _client(monkeypatch)
    assert _post(client, "/zh/evaluate", coach="true", lang="xx").status_code == 400
