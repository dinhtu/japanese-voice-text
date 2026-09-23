"""GOP from fake phoneme posteriors -- no ASR checkpoint."""

import numpy as np

from app.services.gop_scoring import mix_pronunciation, score_gop
from src.asr.phoneme_vocab import PhonemeVocab


def test_gop_high_when_canonical_phone_dominates_its_window():
    vocab = PhonemeVocab()
    phones = ["k", "a"]
    t, v = 8, vocab.size
    probs = np.full((t, v), 0.01, dtype=np.float64)
    k_idx, a_idx = vocab.stoi["k"], vocab.stoi["a"]
    probs[:4, k_idx] = 0.7
    probs[4:, a_idx] = 0.7
    probs = probs / probs.sum(axis=1, keepdims=True)
    score = score_gop(probs, phones, vocab)
    assert score is not None
    assert score >= 90


def test_gop_low_when_wrong_phone_dominates():
    vocab = PhonemeVocab()
    t, v = 8, vocab.size
    probs = np.full((t, v), 0.01, dtype=np.float64)
    probs[:, vocab.stoi["o"]] = 0.8
    probs = probs / probs.sum(axis=1, keepdims=True)
    score = score_gop(probs, ["k", "a"], vocab)
    assert score is not None
    assert score < 40


def test_gop_none_without_inputs():
    assert score_gop(None, ["a"]) is None
    assert score_gop(np.zeros((4, 10)), []) is None


def test_mix_pronunciation_weights_gop():
    assert mix_pronunciation(100, None) == 100.0
    mixed = mix_pronunciation(100, 0)
    assert mixed == 45.0
