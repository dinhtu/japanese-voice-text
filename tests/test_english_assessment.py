import numpy as np
import pytest

from app.schemas.english_assessment import GoptAssessment, ReadingAssessment
from app.services.english_alignment import SyllableTiming
from app.services.english_assessment import (
    COVERAGE_GATE,
    GATED,
    WEIGHTS,
    assess_gopt,
    assess_reading,
    coach_facts,
    no_voice_reading,
)
from app.services.english_gopt import GoptDetail, GoptScores, GoptWord, build_word_gopt_features
from app.services.english_intonation import (
    FinalCheck,
    IntonationResult,
    PeakCheck,
    WordStressCheck,
    analyze_sentence,
)
from app.services.english_scoring import align_word_tokens
from app.services.vad_fluency import VadResult


def _timing(target: str, lp: float = -0.2, aligned: bool = True, missing=frozenset()) -> SyllableTiming:
    analysis = analyze_sentence(target)
    windows, letters, t = [], [], 0.0
    for w in analysis.words:
        word_windows = []
        for _s in w.pron.syllables:
            word_windows.append((t, t + 0.2))
            t += 0.2
        windows.append(word_windows)
        letters.append(tuple(lp for _ch in w.spoken.replace("'", "")))
    return SyllableTiming(
        windows, None, aligned, frozenset(missing),
        tuple(lp for _ in analysis.words), tuple(letters),
    )


def _vad(seconds: float = 2.0, pauses: int = 0) -> VadResult:
    return VadResult(seconds, pauses, 0.8, "energy")


def _assess(target: str, heard: str, *, timing="auto", intonation=None, vad="auto", duration=2.0):
    matches, extras = align_word_tokens(target, heard)
    return assess_reading(
        target_text=target,
        recognized_text=heard,
        matches=matches,
        extras=extras,
        analysis=analyze_sentence(target),
        timing=_timing(target) if timing == "auto" else timing,
        intonation=intonation,
        vad=_vad(duration) if vad == "auto" else vad,
        duration=duration,
        legacy_score=90,
    )


def test_perfect_reading_scores_content_and_phoneme():
    report = _assess("Nice to meet you", "nice to meet you")
    c = report["criteria"]
    assert c["content_accuracy"]["score"] == 100
    assert c["content_accuracy"]["status"] == "scored"
    assert c["phoneme_accuracy"]["status"] == "estimated"
    assert c["phoneme_accuracy"]["score"] == 100
    assert report["content_gate"]["passed"]
    # No pitch track was given: those criteria are missing, not zero.
    assert c["word_stress"]["status"] == "insufficient_evidence"
    assert c["word_stress"]["score"] is None
    assert c["fluency"]["status"] == "scored"
    assert report["status"] == "partial_assessment"
    assert report["overall_score"] is not None
    assert report["weights_used"] == pytest.approx(0.65)
    assert [w["status"] for w in report["word_results"]] == ["correct"] * 4
    ReadingAssessment.model_validate(report)


def test_omitted_article_is_wer_deletion_with_message():
    target = "I would like a glass of water."
    report = _assess(target, "i would like glass of water")
    content = report["criteria"]["content_accuracy"]
    assert content["detail"]["deletions"] == 1
    assert content["score"] == pytest.approx(100 * (1 - 1 / 7), abs=0.01)
    omitted = [w for w in report["word_results"] if w["status"] == "omitted"]
    assert [w["word"] for w in omitted] == ["a"]
    assert omitted[0]["issue"] == "possible_omission"
    assert omitted[0]["start"] is None  # no timestamp for a skipped word
    first = report["improvements"][0]
    assert first["priority"] == "high" and first["code"] == "omitted_words"
    assert "'a'" in first["message"]


def test_substituted_word():
    report = _assess("I would like a glass of water", "i would like a cup of water")
    subs = [w for w in report["word_results"] if w["status"] == "substituted"]
    assert [w["word"] for w in subs] == ["glass"]
    assert subs[0]["recognized"] == "cup"
    assert 0 < subs[0]["confidence"] <= 0.95
    assert report["criteria"]["content_accuracy"]["detail"]["substitutions"] == 1


def test_inserted_words_count_in_wer_and_fluency():
    report = _assess("nice to meet you", "nice to really meet you")
    content = report["criteria"]["content_accuracy"]
    assert content["detail"]["insertions"] == 1
    assert report["inserted"] == [{"after_word": 1, "text": "really"}]
    assert report["criteria"]["fluency"]["detail"]["repetitions_or_extra_words"] == 1


def test_voice_with_wrong_words_gates_prosody():
    """The old bars gave rhythm / fluency / intonation to any voiced clip."""
    report = _assess("I would like a glass of water", "banana apple orange tree")
    assert not report["content_gate"]["passed"]
    for key in GATED:
        assert report["criteria"][key]["status"] == "gated"
        assert report["criteria"][key]["score"] is None
    assert report["status"] == "content_mismatch"
    assert report["content_gate"]["word_coverage"] < COVERAGE_GATE
    # The misread words are the verdict: overall is content accuracy alone,
    # even though phoneme accuracy was estimated.
    assert report["criteria"]["phoneme_accuracy"]["status"] == "estimated"
    assert report["weights_used"] == WEIGHTS["content_accuracy"]
    assert report["overall_score"] == report["criteria"]["content_accuracy"]["score"]
    assert any(i["code"] == "content_gate" for i in report["improvements"])
    ReadingAssessment.model_validate(report)


def test_failed_gate_scores_from_content_alone():
    target = "I would like a glass of water"
    report = _assess(target, "banana apple orange tree", timing=_timing(target, aligned=False))
    assert report["criteria"]["phoneme_accuracy"]["status"] == "insufficient_evidence"
    assert report["status"] == "content_mismatch"
    assert report["overall_score"] == report["criteria"]["content_accuracy"]["score"]
    assert report["weights_used"] == WEIGHTS["content_accuracy"]
    # Letters on an unreliable path are not read as "unclear" words.
    assert all(w["phoneme_score"] is None for w in report["word_results"])


def test_audio_confirmed_word_counts_as_read():
    target = "articulate"
    matches, extras = align_word_tokens(target, "taerait")
    assert not matches[0].ok and matches[0].heard

    def statuses(confirmed):
        return [w["status"] for w in assess_reading(
            target_text=target, recognized_text="taerait", matches=matches,
            extras=extras, analysis=analyze_sentence(target), timing=_timing(target),
            intonation=None, vad=_vad(), duration=2.0, confirmed=confirmed,
        )["word_results"]]

    assert statuses(frozenset()) == ["substituted"]
    assert statuses(frozenset({0})) == ["mispronounced"]


def test_overall_is_null_without_phoneme_evidence():
    report = _assess("nice to meet you", "nice to meet you", timing=None)
    assert report["criteria"]["phoneme_accuracy"]["status"] == "insufficient_evidence"
    assert report["overall_score"] is None
    assert report["status"] == "insufficient_evidence"


def test_unreliable_alignment_is_insufficient_evidence():
    report = _assess("nice to meet you", "nice to meet you", timing=_timing("nice to meet you", aligned=False))
    assert report["criteria"]["phoneme_accuracy"]["status"] == "insufficient_evidence"


def test_low_letter_posterior_lowers_phoneme_and_flags_words():
    target = "nice to meet you"
    report = _assess(target, "nice to meet you", timing=_timing(target, lp=-4.5))
    assert report["criteria"]["phoneme_accuracy"]["score"] < 50
    assert all("unclear_articulation" in w["issues"] for w in report["word_results"])
    assert any(i["code"] == "unclear_words" for i in report["improvements"])


def test_fluency_counts_correct_words_only():
    target = "the quick brown fox jumps over the lazy dog"
    good = _assess(target, "the quick brown fox jumps over the lazy dog", duration=6.0)
    half = _assess(target, "the quick brown fox jumps over xx yy zz", duration=6.0)
    g = good["criteria"]["fluency"]["detail"]
    h = half["criteria"]["fluency"]["detail"]
    assert g["words_per_minute"] == h["words_per_minute"]
    assert h["correct_words_per_minute"] < g["correct_words_per_minute"]


def test_unexpected_pauses_cost_fluency():
    target = "the quick brown fox jumps over the lazy dog"
    calm = _assess(target, target, vad=_vad(5.0, 0), duration=5.0)
    halting = _assess(target, target, vad=_vad(5.0, 3), duration=5.0)
    assert halting["criteria"]["fluency"]["score"] < calm["criteria"]["fluency"]["score"]
    assert halting["criteria"]["fluency"]["detail"]["unexpected_pauses"] == 3


def _intonation(target: str, stress_ok_by_word: dict[int, bool], peaks_ok: dict[int, bool],
                end_ok: bool, pauses=None) -> IntonationResult:
    analysis = analyze_sentence(target)
    stress = [
        WordStressCheck(i, 0, 0 if ok else 1, ok, (0,)) for i, ok in stress_ok_by_word.items()
    ]
    peaks = [PeakCheck(i, 0, ok, 2.0 if ok else 0.0) for i, ok in peaks_ok.items()]
    ends = [FinalCheck(analysis.final_tone, end_ok, -3.0 if end_ok else 0.0, phrase=len(analysis.phrase_ends) - 1)]
    return IntonationResult(
        analysis, [], peaks, [], ends, stress, True, None, frozenset(),
        tuple(pauses) if pauses is not None else tuple(0.0 for _ in analysis.words[1:]),
    )


def test_word_stress_only_on_words_read_right():
    target = "my teacher is beautiful"
    intonation = _intonation(target, {1: False, 3: True}, {1: True, 3: True}, True)
    report = _assess(target, "my teacher is beautiful", intonation=intonation)
    ws = report["criteria"]["word_stress"]
    assert ws["status"] == "scored" and ws["score"] == 50
    teacher = report["word_results"][1]
    assert "wrong_stress" in teacher["issues"] and teacher["stress_ok"] is False
    stress_msg = next(i for i in report["improvements"] if i["code"] == "wrong_stress")
    assert "TEA-cher" in stress_msg["message"]

    # "beautiful" misread: its (good) stress check no longer counts.
    misread = _assess(target, "my teacher is banana", intonation=intonation)
    ws = misread["criteria"]["word_stress"]
    assert ws["detail"]["checked_words"] == 1 and ws["score"] == 0


def test_intonation_and_rhythm_scored_with_timing():
    target = "my teacher is beautiful"
    intonation = _intonation(target, {1: True, 3: True}, {1: True, 3: True}, True, pauses=[0.0, 0.4, 0.0])
    report = _assess(target, target, intonation=intonation)
    assert report["criteria"]["intonation"]["score"] == 100
    rhythm = report["criteria"]["rhythm_and_linking"]
    assert rhythm["status"] == "scored"
    assert rhythm["detail"]["broken_links"] == [[1, 2]]
    assert any(i["code"] == "broken_linking" for i in report["improvements"])
    assert report["status"] == "complete"


def test_no_voice_report():
    report = no_voice_reading("Nice to meet you", ["Nice", "to", "meet", "you"], 1.5, "silent")
    assert report["status"] == "no_voice" and report["overall_score"] == 0
    assert all(w["status"] == "omitted" for w in report["word_results"])
    ReadingAssessment.model_validate(report)
    gopt = assess_gopt(None, report, 0)
    assert gopt["status"] == "no_voice"
    GoptAssessment.model_validate(gopt)


def _detail(acc=80.0, comp=90.0, flu=70.0, pro=60.0, tot=75.0, n=4) -> GoptDetail:
    return GoptDetail(
        GoptScores(overall=tot, pronunciation=acc, completeness=comp, fluency=flu, intonation=pro),
        tuple(GoptWord(i, ("AH",), 70.0, 80.0, 75.0, (66.0,)) for i in range(n)),
        n, False, "forced_alignment",
    )


def test_gopt_average_and_adjusted_average():
    reading = _assess("nice to meet you", "nice to meet you")
    gopt = assess_gopt(_detail(), reading, 90)
    assert gopt["average"] == pytest.approx((80 + 90 + 70 + 60) / 4)
    assert gopt["adjusted_average"] == pytest.approx(gopt["average"])  # coverage 1
    assert gopt["status"] == "scored"
    assert all(w["read_ok"] for w in gopt["words"])
    GoptAssessment.model_validate(gopt)


def test_gopt_gated_when_words_are_wrong():
    reading = _assess("I would like a glass of water", "banana apple orange tree")
    gopt = assess_gopt(_detail(n=7), reading, 10)
    assert gopt["status"] == "content_mismatch"
    assert gopt["adjusted_heads"]["fluency"] is None
    assert gopt["adjusted_heads"]["prosodic"] is None
    # Raw heads stay visible for comparison.
    assert gopt["heads"]["fluency"] == 70
    assert gopt["adjusted_average"] < gopt["average"]


def test_gopt_collapsed_head_left_out_of_adjusted():
    reading = _assess("nice to meet you", "nice to meet you")
    gopt = assess_gopt(_detail(flu=2.0), reading)
    assert gopt["head_reliable"]["fluency"] is False
    assert gopt["adjusted_heads"]["fluency"] is None
    assert gopt["adjusted_average"] == pytest.approx((80 + 90 + 60) / 3, abs=0.01)


def test_gopt_unavailable():
    reading = _assess("nice to meet you", "nice to meet you")
    gopt = assess_gopt(None, reading)
    assert gopt["status"] == "unavailable" and gopt["average"] is None
    GoptAssessment.model_validate(gopt)


def test_coach_facts_only_report_findings():
    gated = coach_facts(_assess("I would like a glass of water", "banana apple orange tree"))
    text = "\n".join(gated)
    assert "KHÔNG khen" in text
    assert "chưa chấm vì đọc sai" in text
    omitted = "\n".join(coach_facts(_assess("I would like a glass of water.", "i would like glass of water")))
    assert "Từ bị bỏ sót: 'a'" in omitted
    assert coach_facts(None) == []


def _aligned(target: str, heard: str) -> list[tuple[str, str]]:
    return [(m.target, m.heard) for m in align_word_tokens(target, heard)[0]]


def test_align_word_tokens_finds_skipped_short_word():
    # The letter segmentation hands "a" the "e" of "like"; tokens do not.
    assert _aligned("I would like a glass", "i would like glass") == [
        ("i", "i"), ("would", "would"), ("like", "like"), ("a", ""), ("glass", "glass"),
    ]


def test_align_word_tokens_joins_and_splits_tokens():
    assert _aligned("see you today", "see you to day")[-1] == ("today", "today")
    assert _aligned("thank you", "thankyou") == [("thank", "thank"), ("you", "you")]
    assert _aligned("nice to meet you", "nice to meat you")[2] == ("meet", "meat")


def test_align_word_tokens_extra_tokens():
    matches, extras = align_word_tokens("nice to meet you", "uh nice to meet you")
    assert all(m.ok for m in matches)
    assert extras[0].after == -1 and extras[0].edge and extras[0].letters == "uh"
    _m, extras = align_word_tokens("nice to meet you", "nice to to meet you")
    # Either "to" may be the repeat.
    assert [(e.letters, e.edge) for e in extras] == [("to", False)] and extras[0].after in (0, 1)


def test_fillers_cost_fluency_not_content():
    report = _assess("nice to meet you", "nice uh to meet you")
    assert report["criteria"]["content_accuracy"]["score"] == 100
    assert report["criteria"]["fluency"]["detail"]["fillers"] == 1


def test_gopt_features_use_word_frames_and_truncate():
    vocab = ["<pad>", "|", "A", "B", "N", "S", "T", "I", "D", "E", "U", "W", "R", "M", "O", "Y"]
    log_probs = np.full((100, len(vocab)), -10.0)
    words = [["N", "AY", "S"], ["T", "UW"]]
    feat, phn, slots = build_word_gopt_features(log_probs, words, vocab, [(0, 30), (60, 100)])
    assert slots == [[0, 1, 2], [3, 4]]
    assert feat.shape == (1, 50, 84) and list(phn[0, :5] >= 0) == [True] * 5
    assert phn[0, 5] == -1

    many = [["AH"] * 20, ["AH"] * 20, ["AH"] * 20]
    _feat, phn, slots = build_word_gopt_features(log_probs, many, vocab)
    assert [len(s) for s in slots] == [20, 20, 0]
    assert int((phn[0] >= 0).sum()) == 40


@pytest.mark.parametrize("target, heard, extra", [
    ("I want water", "i want a water", "a"),
    ("I study English", "i study the english", "the"),
    ("Please listen carefully", "please listen to carefully", "to"),
])
def test_inserted_word_is_not_joined_into_its_neighbour(target, heard, extra):
    report = _assess(target, heard)
    assert [e["text"] for e in report["inserted"]] == [extra]
    assert report["criteria"]["content_accuracy"]["detail"]["insertions"] == 1


@pytest.mark.parametrize("target, heard, repeat", [
    ("She is a teacher", "she is a a teacher", "a"),
    ("Nice to meet you", "nice to meet you you", "you"),
    ("Nice to meet you", "nice nice to meet you", "nice"),
    ("I want to go home", "i want to to go home", "to"),
])
def test_repeated_word_is_fluency_not_wer(target, heard, repeat):
    report = _assess(target, heard)
    assert [e["text"] for e in report["repeated"]] == [repeat]
    assert report["criteria"]["content_accuracy"]["score"] == 100
    assert report["criteria"]["fluency"]["detail"]["repetitions_or_extra_words"] == 1
    assert any(i["code"] == "repeated_words" for i in report["improvements"])


def test_filler_never_stands_in_for_a_skipped_word():
    report = _assess("I would like a glass of water", "i would like um glass of water")
    assert report["word_results"][3]["status"] == "omitted"
    assert report["criteria"]["fluency"]["detail"]["fillers"] == 1


def test_spelled_out_letters_join_into_one_word():
    assert _aligned("My PIN is safe", "my p i n is safe")[1] == ("pin", "pin")


def test_other_real_word_is_a_substitution():
    report = _assess("I would like a glass of water", "i would like a class of waiter")
    subs = [w["word"] for w in report["word_results"] if w["status"] == "substituted"]
    assert subs == ["glass", "water"]
    # A homophone or a by-ear spelling is still the right word.
    homophone = _assess("Nice to meet you", "nice to meat you")
    assert [w["status"] for w in homophone["word_results"]] == ["correct"] * 4


def test_repeating_a_short_prompt_does_not_fail_the_gate():
    report = _assess("Thanks", "thanks thanks")
    assert report["content_gate"]["passed"]
    assert report["criteria"]["content_accuracy"]["score"] == 100


def test_unaligned_path_confirms_no_word():
    target = "articulate"
    matches, extras = align_word_tokens(target, "taerait")
    report = assess_reading(
        target_text=target, recognized_text="taerait", matches=matches, extras=extras,
        analysis=analyze_sentence(target), timing=_timing(target, aligned=False),
        intonation=None, vad=_vad(), duration=2.0, confirmed=frozenset({0}),
    )
    assert report["word_results"][0]["status"] == "substituted"


def test_gopt_adjusted_never_above_average_and_unread_words_zero():
    reading = _assess("I would like a glass of water", "i would banana apple orange tree")
    assert not reading["content_gate"]["passed"]
    gopt = assess_gopt(_detail(acc=90.0, comp=90.0, flu=10.0, pro=10.0, n=7), reading)
    assert gopt["adjusted_average"] <= gopt["average"]
    # Extra words lower the factor through content accuracy.
    extra = _assess("I would like a glass of water", "i would like a glass of water i would like")
    assert assess_gopt(_detail(n=7), extra)["adjusted_heads"]["accuracy"] < 80.0
    skipped = _assess("I would like a glass of water.", "i would like glass of water")
    words = assess_gopt(_detail(n=7), skipped)["words"]
    assert words[3]["read_ok"] is False and words[3]["accuracy"] == 0.0
    assert words[3]["raw"]["accuracy"] == 70.0


def test_initialism_phones_are_letter_names():
    from app.services.english_gopt import word_cmu_phones

    assert word_cmu_phones("us", initialism=True) == ["Y", "UW", "EH", "S"]
    assert word_cmu_phones("us") == ["AH", "S"]
