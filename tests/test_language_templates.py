from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def test_language_pages_use_their_own_endpoints():
    templates = Path(__file__).resolve().parents[1] / "templates"
    env = Environment(loader=FileSystemLoader(templates), undefined=StrictUndefined)
    context = {
        "api_base_url": "",
        "asset_url": lambda path: f"/static/{path}",
        "default_text": {"id": "sample", "text": "hello", "reading": "", "meaning": ""},
        "texts": [],
    }
    english = env.get_template("en.html").render(context)
    chinese = env.get_template("zh.html").render(context)
    korean = env.get_template("ko.html").render(context)

    assert "data-evaluate-url=\"/api/pronunciation-en/evaluate\"" in english
    assert "data-text-guide-url=\"/api/pronunciation-en/text-guide\"" in english
    assert "id=\"custom-input\"" in english
    assert "id=\"search-suggest\"" in english
    assert "id=\"guide-btn\"" not in english
    assert "pronunciation-zh" not in english
    assert "data-reading-url" not in english
    assert "data-evaluate-url=\"/api/pronunciation-zh/evaluate\"" in chinese
    assert "data-reading-url=\"/api/pronunciation-zh/reading\"" in chinese
    assert "data-pitch-url=\"/api/pronunciation-zh/pitch-accent\"" in chinese
    assert "data-tts-lang=\"zh-CN\"" in chinese
    assert "data-evaluate-url=\"/api/pronunciation-ko/evaluate\"" in korean
    assert "data-reading-url=\"/api/pronunciation-ko/reading\"" in korean
    assert "data-pitch-url=\"/api/pronunciation-ko/pitch-accent\"" in korean
    assert "data-text-guide-url=\"/api/pronunciation-ko/text-guide\"" in korean
    assert "data-coach-url=\"/api/pronunciation-ko/coach\"" in korean
    assert "data-tts-lang=\"ko-KR\"" in korean
    assert "data-target-lang=\"ko\"" in korean
    assert "Pitch Accent" not in korean
    assert "data-text-guide-url=\"/api/pronunciation-zh/text-guide\"" in chinese
    assert "id=\"custom-input\"" in chinese
    assert "id=\"guide-btn\"" not in chinese


def test_en_zh_ko_pages_offer_the_v1_v2_switch():
    """en, zh and ko pick /evaluate or /evaluate-v2 ("Cách chấm"); ja has no v2."""
    templates = Path(__file__).resolve().parents[1] / "templates"
    env = Environment(loader=FileSystemLoader(templates), undefined=StrictUndefined)
    context = {
        "api_base_url": "",
        "asset_url": lambda path: f"/static/{path}",
        "default_text": {"id": "sample", "text": "hello", "reading": "", "meaning": ""},
        "texts": [],
    }
    for page, lang in (("en.html", "en"), ("zh.html", "zh"), ("ko.html", "ko")):
        html = env.get_template(page).render(context)
        assert f"data-evaluate-v2-url=\"/api/pronunciation-{lang}/evaluate-v2\"" in html, page
        assert "<fieldset class=\"api-mode\" id=\"api-mode\">" in html, page
        assert "name=\"api-mode\" value=\"v1\" checked" in html, page
        assert "name=\"api-mode\" value=\"v2\"" in html, page
        assert "id=\"result-note-v1\"" in html, page
        assert "id=\"result-note-v2\" hidden" in html, page
        assert "id=\"result-v2\" hidden" in html, page
        assert "id=\"result-v2-body\"" in html, page
        assert "id=\"coach-lang\"" in html, page

    japanese = (templates / "index.html").read_text(encoding="utf-8")
    assert "data-evaluate-v2-url" not in japanese
    assert "api-mode" not in japanese
