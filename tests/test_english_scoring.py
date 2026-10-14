import pytest

from app.services.english_scoring import WORD_GATE, match_words, score_english
from app.services.scoring import score_pronunciation


def _score(target: str, heard: str) -> int:
    return score_english(target, heard)[0].score


def _ok(target: str, heard: str) -> list[tuple[str, bool]]:
    return [(w.target, w.ok) for w in match_words(target, heard)]


def test_wrong_single_word_scores_zero():
    # Plain letter CER used to give this 30.
    result, words = score_english("articulate", "a taerait")
    assert result.score == 0
    assert result.level == "mismatch"
    assert [w.ok for w in words] == [False]
    # CER metric stays letter-level for the UI.
    assert result.cer == pytest.approx(0.7)


def test_asr_misspelling_keeps_partial_credit():
    assert _score("articulate", "articulat") == 90
    assert _ok("articulate", "articulat") == [("articulate", True)]


def test_all_words_pass_equals_old_letter_cer():
    for target, heard in [
        ("Nice to meet you", "nice to meat you"),
        ("Thank you very much", "thank you vary much"),
        ("Nice to meet you", "uh nice to meet you"),
    ]:
        old = score_pronunciation(
            target.lower().replace(" ", ""), heard.replace(" ", "")
        ).score
        assert _score(target, heard) == old


def test_asr_word_boundaries_do_not_matter():
    assert _score("How are you today", "how are you to day") == 100
    assert _score("thank you very much", "thankyou very much") == 100
    assert _score("an apple", "a napple") == 100
    assert _score("understand", "under st and") == 100
    assert all(ok for _w, ok in _ok("an apple", "a napple"))


def test_failed_word_loses_all_its_letters():
    # "thank" (5 letters) heard as "sank": 2 errors -> 0.6 < gate.
    # Old CER only took 2 letters off; now the whole word goes.
    assert _score("Thank you very much", "sank you very much") == 69
    assert _ok("Thank you very much", "sank you very much")[0] == ("thank", False)


def test_extra_words_still_cost_points():
    # A free skip used to make these 100.
    assert _score("I like it", "i dont like it") < 60
    assert _score("hello", "hello how are you i am fine thank you") == 0


def test_contraction_is_one_word():
    words = match_words("Let's go", "lets go")
    assert [(w.target, w.ok) for w in words] == [("let's", True), ("go", True)]
    assert _score("Don't worry", "don worry") == 89


def test_stopping_early_keeps_the_words_that_were_read():
    assert _ok("Thank you very much", "thank you") == [
        ("thank", True), ("you", True), ("very", False), ("much", False),
    ]
    assert _score("Thank you very much", "thank you") == 50


def test_dropped_word_does_not_fail_its_neighbours():
    assert _ok("It is a good day", "is a good day") == [
        ("it", False), ("is", True), ("a", True), ("good", True), ("day", True),
    ]
    assert _ok("go to", "go") == [("go", True), ("to", False)]
    assert _ok("Please sit down here", "sit down here")[1:] == [
        ("sit", True), ("down", True), ("here", True),
    ]


def test_failed_word_cannot_soak_up_extra_words():
    assert _score("I like it", "i dont like it") == 43


def test_nothing_heard_scores_zero():
    assert _score("nice to meet you", "") == 0
    assert not any(ok for _w, ok in _ok("nice to meet you", ""))


def test_gate_value():
    assert WORD_GATE == 0.75


def test_empty_target_rejected():
    with pytest.raises(ValueError):
        score_english("123 !!", "hello")
