import io
import re
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
    transcribe_korean,
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


@pytest.mark.parametrize("text, ipa, rr", [
    ("한국어", "[han.ku.kʌ]", "hangugeo"),
    ("감사합니다", "[kam.sa.ham.ni.ta]", "gamsahamnida"),
    ("읽고", "[il.k͈o]", "ilgo"),
    ("식량", "[siŋ.njaŋ]", "singnyang"),
    ("할 수 있어요.", "[hal s͈u i.s͈ʌ.jo]", "Hal su isseoyo."),
    ("학교", "[hak.k͈jo]", "hakgyo"),
    ("같이", "[ka.tɕʰi]", "gachi"),
    ("신라", "[sil.la]", "silla"),
    ("좋다", "[tɕo.tʰa]", "jota"),
    ("좋아요", "[tɕo.a.jo]", "joayo"),
    ("한국어를 공부해요.", "[han.ku.kʌ.ɾɯl koŋ.pu.he.jo]", "Hangugeoreul gongbuhaeyo."),
    ("같이 가요.", "[ka.tɕʰi ka.jo]", "Gachi gayo."),
    ("3개 주세요.", "[se.ke tɕu.se.jo]", "Segae juseyo."),
    ("놓치다", "[no.tɕʰi.ta]", "nochida"),  # 제12항: ㅎ + ㅊ -> [ㅊ]
    ("밭일", "[pan.nil]", "bannil"),  # ㄴ insertion
    ("급행", "[kɯ.pʰeŋ]", "geuphaeng"),  # RR keeps the h of a noun
    ("못 먹었어요.", "[mon mʌ.kʌ.s͈ʌ.jo]", "Mon meogeosseoyo."),
    ("다른 사람의 의견", "[ta.ɾɯn sa.ɾa.me ɰi.kjʌn]", "dareun saramui uigyeon"),
    ("네, 알겠습니다.", "[ne, al.ket.s͈ɯm.ni.ta]", "Ne, algetseumnida."),
    ("뚫리다", "[t͈ul.li.ta]", "ttullida"),
    ("이십육", "[i.sim.njuk]", "isimnyuk"),
    # -ㄹ게/-ㄹ거 tense only after the -(으)ㄹ adnominal (제27항).
    ("할게요.", "[hal.k͈e.jo]", "Halgeyo."),
    ("할거예요.", "[hal.k͈ʌ.je.jo]", "Halgeoyeyo."),
    ("읽을거리", "[il.kɯl.k͈ʌ.ɾi]", "ilgeulgeori"),
    ("즐거운 시간", "[tɕɯl.kʌ.un si.kan]", "jeulgeoun sigan"),
    ("설거지", "[sʌl.kʌ.tɕi]", "seolgeoji"),
    ("줄거리", "[tɕul.kʌ.ɾi]", "julgeori"),
    ("알게 됐어요.", "[al.ke twe.s͈ʌ.jo]", "Alge dwaesseoyo."),
    ("길게", "[kil.ke]", "gilge"),
    # ㄼ: ㄹ + ㄴ liquidization, and 밟- read [밥] before a consonant (제10항 다만).
    ("방이 넓네요.", "[paŋ.i nʌl.le.jo]", "Bangi neolleyo."),
    ("짧네요", "[t͈ɕal.le.jo]", "jjalleyo"),
    ("밟다", "[pap.t͈a]", "bapda"),
    ("밟는", "[pam.nɯn]", "bamneun"),
    ("잔디를 밟지 마세요.", "[tɕan.ti.ɾɯl pap.t͈ɕi ma.se.jo]", "Jandireul bapji maseyo."),
    # ㄾ stems tense the ending (제25항); ㄾ + 이 palatalizes (제17항).
    ("핥다", "[hal.t͈a]", "halda"),
    ("수박 겉핥기에", "[su.pak k͈ʌ.tʰal.k͈i.e]", "subak geotalgie"),
    ("벼훑이", "[pjʌ.hul.tɕʰi]", "byeohulchi"),
    # A stop coda + the numeral 한 aspirates across the space (제12항 붙임 2).
    ("꽃 한 송이", "[k͈o tʰan soŋ.i]", "kkot han songi"),
    ("밥 한 그릇", "[pa pʰan kɯ.ɾɯt]", "bap han geureut"),
    ("낮 한때", "[na tʰan.t͈e]", "nat hanttae"),
    ("옷 한 벌", "[o tʰan pʌl]", "ot han beol"),
    ("못 해요.", "[mo tʰe.jo]", "Mo taeyo."),
    # 일 년 [일련].
    ("일 년", "[il ljʌn]", "il lyeon"),
    ("말 너무", "[mal nʌ.mu]", "mal neomu"),
    # ㄵ + ㅎ (제12항 붙임 1).
    ("앉히다", "[an.tɕʰi.ta]", "anchida"),
    # Word-final 의: the particle [에] after X주 / 회, the noun [이] in -주의 and 회의 compounds.
    ("위주의", "[wi.tɕu.e]", "wijuui"),
    ("제주의 바다", "[tɕe.tɕu.e pa.ta]", "jejuui bada"),
    ("민주주의", "[min.tɕu.tɕu.i]", "minjujuui"),
    ("국무회의", "[kuŋ.mu.hwe.i]", "gungmuhoeui"),
    ("위원회의", "[wi.wʌn.hwe.e]", "wiwonhoeui"),
    # ㄹ + ㄷ/ㅅ/ㅈ: Sino-Korean tense, verb endings and the particle 도 lenis (제26-27항).
    ("철도", "[tɕʰʌl.t͈o]", "cheoldo"),
    ("질서", "[tɕil.s͈ʌ]", "jilseo"),
    ("70", "[tɕʰil.s͈ip]", "chilsip"),
    ("올지 모르겠어요.", "[ol.t͈ɕi mo.ɾɯ.ke.s͈ʌ.jo]", "Olji moreugesseoyo."),
    ("학술지에", "[hak.s͈ul.t͈ɕi.e]", "haksuljie"),
    ("말도", "[mal.to]", "maldo"),
    ("만들다", "[man.tɯl.ta]", "mandeulda"),
    ("알지만", "[al.tɕi.man]", "aljiman"),
    ("12시", "[jʌl.tu.si]", "yeoldusi"),
    # ㄴ/ㅁ verb stems tense the ending (제24항); 안다 (알다) and nouns do not.
    ("남다", "[nam.t͈a]", "namda"),
    ("신고", "[sin.k͈o]", "singo"),
    ("껴안다", "[k͈jʌ.an.t͈a]", "kkyeoanda"),
    ("잘 안다", "[tɕal an.ta]", "jal anda"),
    ("감기", "[kam.ki]", "gamgi"),
    ("신고를", "[sin.ko.ɾɯl]", "singoreul"),
    # 여덟 reads [여덜]: no tensing after it.
    ("여덟 시", "[jʌ.tʌl si]", "yeodeol si"),
    ("8시", "[jʌ.tʌl.si]", "yeodeolsi"),
    # RR keeps the ㅎ of a noun (제3장 제1항 붙임), not of 하다 / 히다.
    ("집현전", "[tɕi.pʰjʌn.tɕʌn]", "jiphyeonjeon"),
    ("북한", "[pu.kʰan]", "bukhan"),
    ("낙하산", "[na.kʰa.san]", "nakhasan"),
    ("역할", "[jʌ.kʰal]", "yeokhal"),
    ("축하해요.", "[tɕʰu.kʰa.he.jo]", "Chukahaeyo."),
    ("생각해요", "[seŋ.ka.kʰe.jo]", "saenggakaeyo"),
    ("익힌", "[i.kʰin]", "ikin"),
    # Phone numbers digit by digit.
    ("010-1234-5678", "[koŋ.il.koŋ i.ɾi.sam.sa o.juk.tɕʰil.pʰal]", "gongilgong-irisamsa-oyukchilpal"),
    # Commas: a lone one, fullwidth and ideographic, kept after the word.
    ("네 , 좋아요.", "[ne, tɕo.a.jo]", "Ne , joayo."),
    ("좋아요，정말요.", "[tɕo.a.jo, tɕʌŋ.ma.ɾjo]", "Joayo,jeongmaryo."),
])
def test_transcribe_korean_ipa_and_romanization(text, ipa, rr):
    tr = transcribe_korean(text)
    assert tr.supported
    assert (tr.ipa, tr.romanization) == (ipa, rr)


@pytest.mark.parametrize("text", ["1/2", "3~4명", "2020-2021년"])
def test_transcribe_korean_unspellable_number_pairs_are_null(text):
    """A fraction, a range or another pair of numbers is not guessed."""
    tr = transcribe_korean(text)
    assert not tr.supported and tr.ipa is None and tr.romanization is None


@pytest.mark.parametrize("text", ["KTX를 타요.", "TV를 봐요", "ㅋㅋ 좋아요", "50% 할인", "OK"])
def test_transcribe_korean_unsupported_text_is_null(text):
    tr = transcribe_korean(text)
    if text == "50% 할인":  # the % of a number is spelled (오십퍼센트)
        assert tr.supported and tr.ipa == "[o.sip.pʰʌ.sen.tʰɯ ha.ɾin]"
        return
    assert not tr.supported
    assert tr.ipa is None and tr.romanization is None
    assert tr.per_mora(len(normalize_korean(text))) is None


def test_transcribe_korean_syllables_line_up_with_the_reading():
    """Per syllable: one per normalize_korean syllable; the syllable values
    joined give the word's; reading, IPA and RR come from the same jamo."""
    for text in ("감사합니다", "한국어를 공부해요.", "할 수 있어요.", "좋습니다", "꽃잎이 떨어졌어요.",
                 "읽고 싶은 책이 있어요?", "3개 주세요.", "20살이에요.", "\"안녕,\" 하고 말했어요.",
                 "꽃 한 송이 주세요.", "못 해요.", "일 년 동안", "밟는 아이", "010-1234-5678이에요.",
                 "네 , 좋아요."):
        tr = transcribe_korean(text)
        cells = tr.per_mora(len(normalize_korean(text)))
        assert cells is not None, text
        assert "".join(c.text for c in cells) == normalize_korean(text)
        for w in tr.words:
            assert w.ipa == ".".join(s.ipa for s in w.syllables)
            assert w.romanization == "".join(s.romanization for s in w.syllables)
            assert w.reading == "".join(s.reading for s in w.syllables)
        assert tr.reading == korean_pronunciation(text)
        assert re.sub(r"[^a-zʰ͈ɕʌɯɾŋɰ.]", "", tr.ipa.replace(" ", ".")) == ".".join(w.ipa for w in tr.words)
        assert re.sub(r"[^a-z]", "", tr.romanization.lower()) == "".join(w.romanization for w in tr.words)


def test_korean_pronunciation_numbers_and_phrases():
    assert korean_pronunciation("3개 주세요.") == "세개 주세요"
    assert korean_pronunciation("3층") == "삼층"
    assert korean_pronunciation("못 먹어요") == "몬 머거요"
    assert korean_pronunciation("신라") == "실라"
    assert korean_pronunciation("같이") == "가치"


def test_korean_reading_hint_for_text_guide():
    from app.services.coaching import korean_reading_hint

    assert korean_reading_hint("한국어를 공부해요.") == "한국어를(한구거를 / hangugeoreul) 공부해요.(공부해요 / gongbuhaeyo)"
    assert korean_reading_hint("KTX를 타요.") == ""


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


def _evaluate(text: str, heard: str) -> dict:
    """POST /evaluate with the ASR stubbed to hear `heard`. The WAV is 1 s:
    torchaudio's pitch detector needs more frames than its 30-frame median
    window (a 0.1 s file fails with "maximum size for tensor ..." -> 422)."""
    app.dependency_overrides[get_korean_asr_service] = lambda: StubKoreanASRService(text=heard)
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/pronunciation-ko/evaluate",
                data={"text": text},
                files={"audio": ("sample.wav", io.BytesIO(make_wav(1.0)), "audio/wav")},
            )
            assert resp.status_code == 200, resp.text
            return resp.json()
    finally:
        app.dependency_overrides.clear()


def test_korean_evaluate_endpoint():
    data = _evaluate("감사합니다", "감사함니다")
    assert data["success"] is True
    # The heard 함 is scored against the written 합 by syllable CER
    # (score_pronunciation on normalize_korean), so the phonetic spelling costs one syllable.
    assert data["score"] == 80
    assert data["target_text"] == "감사합니다"
    assert data["target_hiragana"] == "감사합니다"
    assert data["target_reading"] == "감사함니다"
    assert len(data["mora_status"]) == 5
    assert [m["ok"] for m in data["mora_status"]] == [True, True, False, True, True]


def test_korean_evaluate_mora_status_carries_transcription():
    data = _evaluate("한국어를 공부해요.", "한국어를 공부해요")
    cells = data["mora_status"]
    assert [m["mora"] for m in cells] == list("한국어를공부해요")
    assert [m["ipa"] for m in cells] == ["han", "ku", "kʌ", "ɾɯl", "koŋ", "pu", "he", "jo"]
    assert [m["romanization"] for m in cells] == ["han", "gu", "geo", "reul", "gong", "bu", "hae", "yo"]
    assert all(m["syllables"] is None and m["stress_index"] is None for m in cells)
    assert data["target_ipa"] == "[han.ku.kʌ.ɾɯl koŋ.pu.he.jo]"
    assert data["target_romanization"] == "Hangugeoreul gongbuhaeyo."
    words = [{k: v for k, v in w.items() if k != "syllables"} for w in data["words"]]
    assert words == [
        {"text": "한국어를", "reading": "한구거를", "ipa": "han.ku.kʌ.ɾɯl", "romanization": "hangugeoreul",
         "start": 0, "end": 4, "space_after": True},
        {"text": "공부해요.", "reading": "공부해요", "ipa": "koŋ.pu.he.jo", "romanization": "gongbuhaeyo",
         "start": 4, "end": 8, "space_after": False},
    ]
    assert data["words"][0]["syllables"][1] == {"text": "국", "reading": "구", "ipa": "ku",
                                                "romanization": "gu", "mora_index": 1}


def test_korean_evaluate_numbers_keep_alignment():
    # normalize_korean drops the digit: mora_status is 개 주 세 요, the
    # transcription reads 3 as 세 but only the written syllables get a cell.
    data = _evaluate("3개 주세요.", "세 개 주세요")
    assert [m["mora"] for m in data["mora_status"]] == ["개", "주", "세", "요"]
    assert [m["romanization"] for m in data["mora_status"]] == ["gae", "ju", "se", "yo"]
    assert data["target_ipa"] == "[se.ke tɕu.se.jo]"
    assert data["words"][0]["text"] == "3개"
    assert (data["words"][0]["start"], data["words"][0]["end"]) == (0, 1)
    # The word's syllables include the spelled 세 (no cell) next to 개 (cell 0).
    assert [(s["text"], s["romanization"], s["mora_index"]) for s in data["words"][0]["syllables"]] == [
        ("세", "se", None), ("개", "gae", 0)]
    assert data["words"][0]["romanization"] == "segae"


def test_korean_evaluate_latin_text_has_null_transcription():
    data = _evaluate("KTX를 타요.", "를 타요")
    assert len(data["mora_status"]) == 3
    assert all(m["ipa"] is None and m["romanization"] is None for m in data["mora_status"])
    assert data["target_ipa"] is None
    assert data["target_romanization"] is None
    assert data["words"] is None


def test_korean_pitch_accent_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/pronunciation-ko/pitch-accent", params={"text": "안녕하세요"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["pattern"]) == 5
        assert [p["romanization"] for p in data["pattern"]] == ["an", "nyeong", "ha", "se", "yo"]
        assert [p["ipa"] for p in data["pattern"]] == ["an", "njʌŋ", "ha", "se", "jo"]
        assert data["ipa"] == "[an.njʌŋ.ha.se.jo]"
        assert data["romanization"] == "annyeonghaseyo"
        assert data["words"][0]["start"] == 0 and data["words"][0]["end"] == 5
        assert data["intonation"] is None


def test_korean_reading_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/pronunciation-ko/reading", params={"text": "감사합니다"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["reading"] == "감사함니다"
        assert data["ipa"] == "[kam.sa.ham.ni.ta]"
        assert data["romanization"] == "gamsahamnida"
        assert len(data["words"]) == 1
        word = data["words"][0]
        assert {k: v for k, v in word.items() if k != "syllables"} == {
            "text": "감사합니다", "reading": "감사함니다", "ipa": "kam.sa.ham.ni.ta",
            "romanization": "gamsahamnida", "start": 0, "end": 5, "space_after": False}
        assert [s["ipa"] for s in word["syllables"]] == ["kam", "sa", "ham", "ni", "ta"]
        assert [s["mora_index"] for s in word["syllables"]] == [0, 1, 2, 3, 4]

        quoted = client.get("/api/pronunciation-ko/reading", params={"text": '"안녕"이라고 했어요.'}).json()
        assert [(w["text"], w["space_after"]) for w in quoted["words"]] == [
            ('"안녕"', False), ("이라고", True), ("했어요.", False)]

        latin = client.get("/api/pronunciation-ko/reading", params={"text": "KTX를 타요."}).json()
        assert latin["ipa"] is None and latin["romanization"] is None and latin["words"] is None


def test_korean_transcribe_endpoint():
    with TestClient(app) as client:
        resp = client.post("/api/pronunciation-ko/transcribe", json={"items": {
            "1": "감사합니다", "2": "할 수 있어요.", "3": "KTX를 타요.", "4": "", "5": "3개 주세요.", "6": "OK",
        }})
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"]["1"] == {"kana": "[kam.sa.ham.ni.ta]", "romaji": "gamsahamnida",
                                        "reading": "감사함니다"}
        assert data["results"]["2"] == {"kana": "[hal s͈u i.s͈ʌ.jo]", "romaji": "Hal su isseoyo.",
                                        "reading": "할 쑤 이써요"}
        assert data["results"]["5"]["kana"] == "[se.ke tɕu.se.jo]"
        assert sorted(data["unsupported"]) == ["3", "4", "6"]
        assert not set(data["unsupported"]) & set(data["results"])


def test_korean_transcribe_endpoint_limits():
    with TestClient(app) as client:
        too_many = {str(i): "가" for i in range(201)}
        assert client.post("/api/pronunciation-ko/transcribe", json={"items": too_many}).status_code == 422
        too_long = {"1": "가" * 501}
        assert client.post("/api/pronunciation-ko/transcribe", json={"items": too_long}).status_code == 422
        ok = {str(i): "가" for i in range(200)}
        assert client.post("/api/pronunciation-ko/transcribe", json={"items": ok}).status_code == 200


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
