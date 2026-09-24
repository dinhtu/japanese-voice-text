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

    assert "data-evaluate-url=\"/api/pronunciation-en/evaluate\"" in english
    assert "pronunciation-zh" not in english
    assert "data-reading-url" not in english
    assert "data-evaluate-url=\"/api/pronunciation-zh/evaluate\"" in chinese
    assert "data-reading-url=\"/api/pronunciation-zh/reading\"" in chinese
    assert "data-tts-lang=\"zh-CN\"" in chinese
