from app.services.chinese_text import (
    normalize_chinese,
    pinyin_syllables,
    score_chinese_fluency,
    score_syllables,
)


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


def test_chinese_fluency_uses_syllable_rate_and_pauses():
    fluent = score_chinese_fluency(12, 3.0, 0, 100)
    slow = score_chinese_fluency(12, 12.0, 0, 100)
    paused = score_chinese_fluency(12, 3.0, 3, 100)
    assert fluent > slow
    assert fluent > paused
    assert score_chinese_fluency(12, 3.0, 0, 0) == 0
