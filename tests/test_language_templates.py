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
    assert "id=\"guide-btn\"" in english
    assert "pronunciation-zh" not in english
    assert "data-reading-url" not in english
    assert "data-evaluate-url=\"/api/pronunciation-zh/evaluate\"" in chinese
    assert "data-reading-url=\"/api/pronunciation-zh/reading\"" in chinese
    assert "data-pitch-url=\"/api/pronunciation-zh/pitch-accent\"" in chinese
    assert "data-tts-lang=\"zh-CN\"" in chinese
    assert "data-evaluate-url=\"/api/pronunciation-ko/evaluate\"" in korean
    assert "data-text-guide-url=\"/api/pronunciation-ko/text-guide\"" in korean
    assert "data-coach-url=\"/api/pronunciation-ko/coach\"" in korean
    assert "data-tts-lang=\"ko-KR\"" in korean
    assert "data-target-lang=\"ko\"" in korean
    assert "Pitch Accent" not in korean
    assert "data-text-guide-url=\"/api/pronunciation-zh/text-guide\"" in chinese
    assert "id=\"guide-btn\"" in chinese
