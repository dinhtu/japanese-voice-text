"""Goodness-of-Pronunciation from the Dual CTC phoneme head.

Uses the InterCTC softmax this model already produces (no extra VRAM).
Canonical phones come from pyopenjtalk (same set as training). Each
target phone is scored as the mean posterior inside the time window of
the aligned recognized phone -- a phone the mouth missed gets a low
posterior even when kana CER still looks fine.

Returns None when there are no phones or no frame posteriors.
"""

from __future__ import annotations

import math

import numpy as np

from app.services.mora_timing import _full_alignment
from src.asr.inference import _ctc_collapse_with_onsets
from src.asr.phoneme_vocab import BLANK_IDX, PhonemeVocab

# CTC posteriors are spread across frames; a clean phone often sits
# around 0.25--0.55, not 0.95. Stretch that band toward 0--100.
_GOP_P_LO = 0.04
_GOP_P_HI = 0.55


def _posterior_to_score(p: float) -> float:
    if p <= _GOP_P_LO:
        return 0.0
    if p >= _GOP_P_HI:
        return 100.0
    return 100.0 * (p - _GOP_P_LO) / (_GOP_P_HI - _GOP_P_LO)


def _hyp_spans(
    pred_ids: list[int], num_frames: int
) -> list[tuple[int, int, int]]:
    """(token_id, start_frame, end_frame) after CTC collapse."""
    tokens, onsets = _ctc_collapse_with_onsets(pred_ids, blank_idx=BLANK_IDX)
    spans: list[tuple[int, int, int]] = []
    for i, (tok, start) in enumerate(zip(tokens, onsets)):
        end = onsets[i + 1] if i + 1 < len(onsets) else num_frames
        spans.append((tok, start, max(end, start + 1)))
    return spans


def score_gop(
    phoneme_probs: np.ndarray | None,
    target_phones: list[str],
    vocab: PhonemeVocab | None = None,
) -> float | None:
    """Utterance GOP 0--100 from (T, V) phoneme softmax vs canonical phones."""
    if phoneme_probs is None or phoneme_probs.size == 0 or not target_phones:
        return None

    vocab = vocab or PhonemeVocab()
    probs = np.asarray(phoneme_probs, dtype=np.float64)
    if probs.ndim != 2:
        return None
    num_frames = probs.shape[0]
    pred_ids = probs.argmax(axis=1).tolist()
    hyp = _hyp_spans(pred_ids, num_frames)
    if not hyp:
        # No spikes: score each target phone on an equal-time slice.
        return _equal_time_gop(probs, target_phones, vocab)

    hyp_phones = [vocab.itos.get(tok, "") for tok, _s, _e in hyp]
    pairs = _full_alignment(target_phones, hyp_phones)

    scores: list[float] = []
    for ref_i, hyp_j in pairs:
        if ref_i is None:
            continue
        phone = target_phones[ref_i]
        idx = vocab.stoi.get(phone)
        if idx is None:
            continue
        if hyp_j is None:
            scores.append(0.0)
            continue
        _tok, lo, hi = hyp[hyp_j]
        window = probs[lo:hi, idx]
        if window.size == 0:
            scores.append(0.0)
            continue
        scores.append(_posterior_to_score(float(window.mean())))

    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def _equal_time_gop(
    probs: np.ndarray, target_phones: list[str], vocab: PhonemeVocab
) -> float | None:
    n = len(target_phones)
    t = probs.shape[0]
    if n == 0 or t == 0:
        return None
    scores: list[float] = []
    for i, phone in enumerate(target_phones):
        idx = vocab.stoi.get(phone)
        if idx is None:
            continue
        lo = math.floor(i * t / n)
        hi = max(lo + 1, math.floor((i + 1) * t / n))
        scores.append(_posterior_to_score(float(probs[lo:hi, idx].mean())))
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def mix_pronunciation(cer_score: float, gop_score: float | None) -> float:
    """Blend kana CER with phone GOP. GOP missing → CER only."""
    if gop_score is None:
        return round(max(0.0, min(100.0, cer_score)), 2)
    return round(0.45 * cer_score + 0.55 * gop_score, 2)
