import numpy as np
import pytest

from app.services.english_alignment import proportional_syllables
from app.services.english_intonation import (
    PitchTrack,
    analyze_sentence,
    evaluate_intonation,
)
from app.services.english_lexicon import pronounce, word_ipa

FRAME = 0.01
SENTENCE = "My name is Anna. I work as a teacher."


def _syllables(word: str) -> list[str]:
    return [s.text for s in pronounce(word).syllables]


# ------------------------------------------------------------- dictionary


@pytest.mark.parametrize("word, sylls, ipa, stress", [
    ("Anna", ["An", "na"], "ˈæn.ə", 0),
    ("teacher", ["tea", "cher"], "ˈtiː.tʃɚ", 0),
    ("articulate", ["ar", "tic", "u", "late"], "ɑːrˈtɪk.jəˌleɪt", 1),
    ("photograph", ["pho", "to", "graph"], "ˈfoʊ.təˌɡræf", 0),
    ("understand", ["un", "der", "stand"], "ˌʌn.dɚˈstænd", 2),
    ("table", ["ta", "ble"], "ˈteɪ.bəl", 0),
    ("today", ["to", "day"], "təˈdeɪ", 1),
])
def test_cmudict_syllables_ipa_and_stress(word, sylls, ipa, stress):
    pron = pronounce(word)
    assert pron.source == "cmudict"
    assert [s.text for s in pron.syllables] == sylls
    assert word_ipa(pron) == ipa
    assert pron.stress_index == stress


def test_case_and_apostrophe_are_kept_in_syllables():
    assert _syllables("Anna") == ["An", "na"]
    assert _syllables("don't") == ["don't"]


def test_heteronym_accepts_both_stresses():
    assert set(pronounce("record").accepted_stress) == {0, 1}


def test_unknown_word_falls_back_to_spelling_rules():
    pron = pronounce("Xylophonia")
    assert pron.source == "rule"
    assert word_ipa(pron) is None
    assert pron.stress_index == 0


# ------------------------------------------------------- sentence analysis


def test_mockup_sentence_roles():
    a = analyze_sentence(SENTENCE)
    marks = [
        ("◉" if w.nucleus else "●" if w.sentence_stress else "·") + w.text
        for w in a.words
    ]
    assert marks == ["·My", "●name", "·is", "◉Anna", "·I", "●work", "·as", "·a", "◉teacher"]
    assert a.phrase_ends == ("continue", "fall")
    # Function words carry no stress mark; content monosyllables do.
    assert [w.ipa for w in a.words[:2]] == ["maɪ", "ˈneɪm"]


def test_reference_peaks_on_accents_and_falls_at_the_end():
    a = analyze_sentence(SENTENCE)
    ref = {(r.word, r.syllable): r for r in a.reference}
    assert ref[(3, 0)].role == "nucleus" and ref[(3, 1)].role == "weak"
    assert ref[(3, 0)].semitone > ref[(2, 0)].semitone + 3
    assert ref[(8, 1)].semitone == -5.0  # "cher" at the bottom


@pytest.mark.parametrize("text, end", [
    ("Do you like coffee?", "rise"),
    ("Where is the station?", "fall"),
    ("I like coffee.", "fall"),
])
def test_final_tone_from_punctuation(text, end):
    assert analyze_sentence(text).final_tone == end


# ------------------------------------------------- checks on synthetic F0


def _windows(analysis, gap_after: int | None = None, gap: float = 0.0):
    """0.2 s per syllable, optional silence after word `gap_after`."""
    t, windows, spans = 0.1, [], []
    for w in analysis.words:
        word_windows = []
        for _ in w.pron.syllables:
            word_windows.append((t, t + 0.2))
            t += 0.2
        windows.append(word_windows)
        spans.append((word_windows[0][0], word_windows[-1][1]))
        if gap_after is not None and w.index == gap_after:
            t += gap
    return windows, spans, t + 0.1


def _track(windows, total, value_of, silent=()):
    n = int(round(total / FRAME)) + 1
    st = np.full(n, np.nan)
    speech = np.zeros(n, dtype=bool)
    energy = np.full(n, -60.0)
    for w, word_windows in enumerate(windows):
        for s, (a, b) in enumerate(word_windows):
            lo, hi = int(round(a / FRAME)), int(round(b / FRAME))
            speech[lo:hi] = True
            energy[lo:hi] = -15.0
            if (w, s) not in silent:
                st[lo:hi] = value_of(w, s, np.linspace(0, 1, hi - lo))
    return PitchTrack(st, energy, speech, FRAME)


def _follow_reference(analysis):
    ref = {(r.word, r.syllable): r for r in analysis.reference}
    return lambda w, s, x: ref[(w, s)].semitone + (ref[(w, s)].end - ref[(w, s)].semitone) * x


def test_reading_that_follows_the_reference_passes_everything():
    a = analyze_sentence(SENTENCE)
    windows, spans, total = _windows(a, gap_after=3, gap=0.3)
    result = evaluate_intonation(a, _track(windows, total, _follow_reference(a)), windows, spans, True)
    assert all(c.ok for c in result.peaks) and len(result.peaks) == 4
    assert [c.ok for c in result.breaks] == [True]
    assert result.final.ok
    assert all(c.ok for c in result.word_stress)
    assert result.score == 100.0


def test_monotone_reading_fails_peaks_and_final_fall():
    a = analyze_sentence(SENTENCE)
    windows, spans, total = _windows(a)
    result = evaluate_intonation(a, _track(windows, total, lambda w, s, x: 0 * x), windows, spans, True)
    assert not any(c.ok for c in result.peaks)
    assert result.final.ok is False
    assert result.breaks[0].ok is False  # no pause, no pitch reset
    assert result.score < 30


def test_stress_on_the_wrong_syllable_is_caught():
    a = analyze_sentence(SENTENCE)
    windows, spans, total = _windows(a, gap_after=3, gap=0.3)
    follow = _follow_reference(a)
    # "tea-CHER": high on the second syllable, low on the first.
    wrong = lambda w, s, x: (4.0 + 0 * x if s == 1 else -3.0 + 0 * x) if w == 8 else follow(w, s, x)
    result = evaluate_intonation(a, _track(windows, total, wrong), windows, spans, True)
    teacher = next(c for c in result.word_stress if c.word == 8)
    assert (teacher.expected, teacher.heard, teacher.ok) == (0, 1, False)


def test_short_pause_with_pitch_reset_counts_as_a_break():
    a = analyze_sentence(SENTENCE)
    windows, spans, total = _windows(a, gap_after=3, gap=0.08)
    result = evaluate_intonation(a, _track(windows, total, _follow_reference(a)), windows, spans, True)
    assert result.breaks[0].ok and result.breaks[0].pause == pytest.approx(0.08, abs=0.02)


def test_breaks_are_not_judged_without_real_timing():
    a = analyze_sentence(SENTENCE)
    windows, spans, total = _windows(a, gap_after=3, gap=0.3)
    result = evaluate_intonation(a, _track(windows, total, _follow_reference(a)), windows, None, False)
    assert [c.ok for c in result.breaks] == [None]


def test_creaky_statement_end_counts_as_a_fall():
    a = analyze_sentence("I work as a teacher.")
    windows, spans, total = _windows(a)
    flat = lambda w, s, x: 1.0 + 0 * x  # never falls while voiced
    track = _track(windows, total, flat, silent={(4, 1)})  # "cher" creaks
    result = evaluate_intonation(a, track, windows, spans, True)
    assert result.final.creaky and result.final.ok


def test_rising_question_checked_by_final_rise():
    a = analyze_sentence("Do you like coffee?")
    windows, spans, total = _windows(a)
    up = lambda w, s, x: (-2.0 + 6.0 * x) if w == 3 and s == 1 else (2.5 if w == 2 else -1.0) + 0 * x
    result = evaluate_intonation(a, _track(windows, total, up), windows, spans, True)
    assert result.final.expected == "rise" and result.final.ok


def test_proportional_fallback_covers_every_syllable():
    a = analyze_sentence(SENTENCE)
    windows, _spans, total = _windows(a)
    track = _track(windows, total, lambda w, s, x: 0 * x)
    timing = proportional_syllables(a, track)
    assert not timing.aligned and timing.word_spans is None
    assert [len(w) for w in timing.windows] == [len(w.pron.syllables) for w in a.words]


def test_forced_alignment_on_fake_emissions():
    pytest.importorskip("torch")
    from app.services.english_alignment import align_syllables

    a = analyze_sentence("Anna")
    vocab = ["<pad>", "|", "A", "N"]
    # 20 frames: blank, A, blank, N, blank, N, A, blanks ...
    path = [0, 2, 2, 0, 3, 0, 0, 3, 2, 2] + [0] * 10
    log_probs = np.full((len(path), len(vocab)), -10.0)
    for t, tok in enumerate(path):
        log_probs[t, tok] = -0.01
    timing = align_syllables(a, log_probs, vocab, duration=0.4)
    (an, na), = timing.windows
    assert timing.aligned
    assert an[0] == pytest.approx(0.02) and na[0] == pytest.approx(0.14)
