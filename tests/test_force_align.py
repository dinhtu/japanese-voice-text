"""CTC Viterbi forced-align (no ASR model)."""

import numpy as np

from src.asr.force_align import _fill_missing, _viterbi_token_frames
from src.asr.kana_vocab import BLANK_IDX


def _emissions(path: list[int], n_classes: int = 8) -> np.ndarray:
    """Near-one-hot log-probs along a known CTC path."""
    t_count = len(path)
    log_probs = np.full((t_count, n_classes), -20.0, dtype=np.float64)
    for t, tok in enumerate(path):
        log_probs[t, tok] = 0.0
    return log_probs


def test_viterbi_recovers_per_token_frames():
    # blank, 1, 1, blank, 2, blank
    path = [BLANK_IDX, 1, 1, BLANK_IDX, 2, BLANK_IDX]
    spans = _viterbi_token_frames(_emissions(path), [1, 2])
    assert spans == [(1, 3), (4, 5)]


def test_viterbi_keeps_repeated_identical_tokens_apart():
    # い い -- two of the same kana must not collapse
    path = [BLANK_IDX, 3, 3, BLANK_IDX, 3, 3, BLANK_IDX]
    spans = _viterbi_token_frames(_emissions(path), [3, 3])
    assert spans is not None
    assert spans[0] is not None and spans[1] is not None
    assert spans[0][1] <= spans[1][0]


def test_fill_missing_interpolates_a_gap():
    filled = _fill_missing([(0, 2), None, (6, 8)], t_count=8)
    assert filled[0] == (0, 2)
    assert filled[2] == (6, 8)
    assert filled[1][0] >= 2 and filled[1][1] <= 6
