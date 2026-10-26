import pytest

from app.services.english_scoring import WORD_GATE, match_words, score_english, snap_transcript
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
    assert _score("articulate", "articulat") == 100  # sounds the same
    assert _ok("articulate", "articulat") == [("articulate", True)]


def test_all_words_pass_never_scores_below_old_letter_cer():
    for target, heard in [
        ("Nice to meet you", "nice to meat you"),
        ("Thank you very much", "thank you vary much"),
        ("Nice to meet you", "uh nice to meet you"),
    ]:
        old = score_pronunciation(
            target.lower().replace(" ", ""), heard.replace(" ", "")
        ).score
        assert _score(target, heard) >= old
    assert 90 <= _score("Nice to meet you", "uh nice to meet you") < 100  # filler costs a little


@pytest.mark.parametrize("target, heard", [
    ("cat", "kat"), ("cat", "cad"), ("dog", "dok"), ("phone", "fone"),
    ("night", "nait"), ("school", "skul"), ("knife", "naif"),
    ("photograph", "fotograf"), ("hemorrhage", "hemoredge"),
    ("technical", "teknikal"), ("I like it", "i laik it"),
    ("Nice to meet you", "nais tu mit yu"), ("Thank you very much", "senkyu veri mat"),
    ("subarachnoid hemorrhage", "SIRBERHAKNOY HEMORICH"),
])
def test_word_spelled_by_ear_passes(target, heard):
    assert all(ok for _w, ok in _ok(target, heard))
    assert _score(target, heard) >= 75


def test_different_word_still_fails():
    assert _ok("cat", "dog") == [("cat", False)]
    assert _score("cat", "dog") == 0


def test_asr_word_boundaries_do_not_matter():
    assert _score("How are you today", "how are you to day") == 100
    assert _score("thank you very much", "thankyou very much") == 100
    assert _score("an apple", "a napple") == 100
    assert _score("understand", "under st and") == 100
    assert all(ok for _w, ok in _ok("an apple", "a napple"))


def test_failed_word_loses_all_its_letters():
    # "very" heard as "happy": nothing alike -> the whole word goes.
    # Old CER only took 3 letters off.
    assert _score("Thank you very much", "thank you happy much") == 69
    assert _ok("Thank you very much", "thank you happy much")[2] == ("very", False)


def test_extra_words_still_cost_points():
    # A free skip used to make these 100.
    assert _score("I like it", "i dont like it") < 60
    # A whole extra sentence costs in full, down to MAX_EXTRA_SHARE.
    assert _score("hello", "hello how are you i am fine thank you") == 30


def test_noise_scraps_around_the_reading_cost_little():
    # The ASR spelled breath / a click before or after the word.
    assert _score("cap", "the cap") == 75
    assert _score("cap", "cap the") == 75
    assert _score("Nice to meet you", "nice to meet you and") == 88
    # A word judged right never ends at 0 points ("1/1 từ đúng", 0 điểm).
    assert _ok("cap", "to happen") == [("cap", True)]
    assert _score("cap", "to happen") > 0


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
    assert WORD_GATE == 0.65


def test_snap_writes_confirmed_words_as_the_target():
    target = "subarachnoid hemorrhage"
    heard = "SIRBERHAKNOY HEMORICH"
    assert snap_transcript(target, heard, frozenset({0, 1})) == target
    assert snap_transcript(target, heard, frozenset({1})) == "sirberhaknoy hemorrhage"
    assert snap_transcript(target, heard, frozenset()) == heard.lower()


def test_snap_keeps_extra_words_and_fills_dropped_ones():
    assert snap_transcript("I like it", "i dont laik it", frozenset({0, 1, 2})) == "i dont like it"
    assert snap_transcript("I like it", "", frozenset({0, 1, 2})) == "i like it"
    # Extra words keep the ASR's own word breaks.
    assert snap_transcript("hello", "HELLO HOW ARE YOU", frozenset({0})) == "hello how are you"


def test_snap_drops_scraps_left_around_respelled_words():
    assert snap_transcript("Nice to meet you", "uh nice to meat you", frozenset({0, 1, 2, 3})) == (
        "nice to meet you"
    )
    assert snap_transcript("cap", "GEMPU", frozenset({0})) == "cap"
    assert snap_transcript("weekend", "WUI GWENDO", frozenset({0})) == "weekend"
    # A snapped transcript scores the extra word like before.
    assert _score("I like it", "i dont like it") == 43


def test_empty_target_rejected():
    with pytest.raises(ValueError):
        score_english("!! ?", "hello")


def test_numbers_and_titles_are_scored_as_spoken():
    assert _score("I have 2 cats.", "i have two cats") == 100
    assert _score("I have 2 cats.", "i have twenty cats") < 100
    assert _score("Mr. Smith is here.", "mister smith is here") == 100
    assert _score("It costs $3.50.", "it costs three dollars fifty") == 100


def test_heard_words_from_forced_alignment():
    pytest.importorskip("torch")
    import numpy as np

    from app.services.english_alignment import align_syllables, heard_words
    from app.services.english_intonation import analyze_sentence

    vocab = ["<pad>", "|", "A", "C", "D", "G", "K", "O", "T"]
    # Greedy hears "KAT" and then silence; "C" is the runner-up under K.
    path = [0, 6, 0, 2, 0, 8] + [0] * 14
    log_probs = np.full((len(path), len(vocab)), -10.0)
    for t, tok in enumerate(path):
        log_probs[t, tok] = -0.01
    log_probs[1, 3] = -2.0
    timing = align_syllables(analyze_sentence("cat dog"), log_probs, vocab, duration=0.4)
    assert heard_words(timing) == frozenset({0})  # "cat" said, "dog" never was
    assert heard_words(None) == frozenset()


def test_word_missing_its_tail_is_not_confirmed():
    pytest.importorskip("torch")
    import numpy as np

    from app.services.english_alignment import align_syllables, heard_words
    from app.services.english_intonation import analyze_sentence

    vocab = ["<pad>", "|", "D", "E", "K", "N", "W"]
    # "weekend" read as "week": W E E K, then silence where E N D should be.
    path = [6, 0, 3, 0, 3, 0, 4] + [0] * 13
    log_probs = np.full((len(path), len(vocab)), -10.0)
    for t, tok in enumerate(path):
        log_probs[t, tok] = -0.01
    timing = align_syllables(analyze_sentence("weekend"), log_probs, vocab, duration=0.4)
    assert timing.confidence[0] > -4.5  # the mean alone would let it through
    assert heard_words(timing) == frozenset()
    assert _ok("weekend", "week") == [("weekend", False)]
    assert _score("weekend", "week") == 0


def test_transcript_tokens_keep_contractions_and_initialisms():
    from app.services.english_scoring import transcript_tokens

    # (Digits are written out before this: expand_for_speech.)
    assert transcript_tokens("I don't know, it's nine a.m. in the U.S.") == [
        "i", "dont", "know", "its", "nine", "am", "in", "the", "us"]


def test_expand_contractions_follows_the_target():
    from app.services.english_scoring import expand_contractions

    assert expand_contractions("It's fine, I can't.", "It is fine, I can not.") == "It is fine, I can not."
    assert expand_contractions("It's fine.", "It's fine.") == "It's fine."
    assert expand_contractions("I won't go.", "I will not go.") == "I will not go."


def test_exact_spelling_over_two_tokens_joins():
    from app.services.english_scoring import align_word_tokens

    for target, heard in [("I cannot swim", "i can not swim"), ("call me anytime", "call me any time"),
                          ("walk into the room", "walk in to the room")]:
        matches, extras = align_word_tokens(target, heard)
        assert all(m.ok for m in matches) and not extras, (target, heard)
