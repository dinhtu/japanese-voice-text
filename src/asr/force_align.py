"""CTC forced alignment of a target kana string onto DualCTC emissions.

Same idea as jp-pitch-accent-analyzer's Wav2Vec2 path
(`torchaudio.functional.forced_align` of the *target* transcript), but
run on this project's kana head so we do not load a second aligner.

A custom Viterbi is used instead of `merge_tokens` so consecutive identical
kana (いい, おー) stay two windows instead of collapsing into one.
"""

from __future__ import annotations

import numpy as np
import torch

from src.asr.kana_vocab import BLANK_IDX, KanaVocab

_NEG = -1e30


def _viterbi_token_frames(
    log_probs: np.ndarray, targets: list[int], blank: int = BLANK_IDX
) -> list[tuple[int, int] | None] | None:
    """Best CTC path; one (start, end) frame span per target token.

    `log_probs` is (T, C). Returns None when the path is degenerate
    (no frames, target longer than T, or the DP dies).
    """
    if log_probs.ndim != 2:
        return None
    t_count, n_classes = log_probs.shape
    length = len(targets)
    if length == 0 or t_count < length:
        return None
    if any(tok < 0 or tok >= n_classes or tok == blank for tok in targets):
        return None

    # Standard CTC states: even = blank, odd = the i-th target token.
    n_states = 2 * length + 1

    def emit_token(state: int) -> int:
        return blank if state % 2 == 0 else targets[state // 2]

    dp = np.full((t_count, n_states), _NEG, dtype=np.float64)
    back = np.full((t_count, n_states), -1, dtype=np.int16)

    dp[0, 0] = float(log_probs[0, blank])
    dp[0, 1] = float(log_probs[0, targets[0]])

    for t in range(1, t_count):
        prev = dp[t - 1]
        for state in range(n_states):
            token = emit_token(state)
            best = prev[state]
            src = state
            if state > 0 and prev[state - 1] > best:
                best = prev[state - 1]
                src = state - 1
            # Skip a blank only when this token differs from the previous
            # one; otherwise two identical kana would share one state.
            if state >= 2 and state % 2 == 1:
                prev_token = targets[state // 2 - 1]
                if token != prev_token and prev[state - 2] > best:
                    best = prev[state - 2]
                    src = state - 2
            dp[t, state] = best + float(log_probs[t, token])
            back[t, state] = src

    end_state = n_states - 1
    if n_states >= 2 and dp[t_count - 1, n_states - 2] > dp[t_count - 1, end_state]:
        end_state = n_states - 2
    if dp[t_count - 1, end_state] <= _NEG / 2:
        return None

    states = [0] * t_count
    state = end_state
    for t in range(t_count - 1, -1, -1):
        states[t] = state
        if t > 0:
            state = int(back[t, state])

    spans: list[tuple[int, int] | None] = []
    for i in range(length):
        token_state = 2 * i + 1
        frames = [t for t, s in enumerate(states) if s == token_state]
        if frames:
            spans.append((frames[0], frames[-1] + 1))
        else:
            spans.append(None)
    if sum(1 for s in spans if s is not None) == 0:
        return None
    return spans


def _fill_missing(
    spans: list[tuple[int, int] | None], t_count: int
) -> list[tuple[int, int]]:
    """Linearly fill tokens the Viterbi path never visited."""
    known = [(-1, 0, 0)]
    known += [(i, s[0], s[1]) for i, s in enumerate(spans) if s is not None]
    known.append((len(spans), t_count, t_count))

    filled: list[tuple[int, int] | None] = list(spans)
    for k in range(len(known) - 1):
        i1, _s1, e1 = known[k]
        i2, s2, _e2 = known[k + 1]
        gap = list(range(i1 + 1, i2))
        if not gap:
            continue
        lo, hi = e1, max(s2, e1)
        edges = np.linspace(lo, hi, len(gap) + 1)
        for offset, idx in enumerate(gap):
            filled[idx] = (int(edges[offset]), max(int(edges[offset + 1]), int(edges[offset]) + 1))
    return [s if s is not None else (0, 1) for s in filled]


def align_kana(
    kana_logits: torch.Tensor,
    vocab: KanaVocab,
    target_hiragana: str,
    duration: float,
) -> list[tuple[float, float]] | None:
    """One (start_sec, end_sec) window per character of `target_hiragana`.

    `kana_logits` is (1, T, V) or (T, V) raw DualCTC kana logits.
    Returns None when the target cannot be encoded or alignment fails,
    so callers can fall back to decode-then-Levenshtein timing.
    """
    if not target_hiragana or duration <= 0:
        return None
    ids = [vocab.stoi[ch] for ch in target_hiragana if ch in vocab.stoi]
    if len(ids) != len(target_hiragana):
        return None

    logits = kana_logits.detach().float()
    if logits.ndim == 3:
        logits = logits.squeeze(0)
    if logits.ndim != 2 or logits.shape[0] == 0:
        return None

    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
    frame_spans = _viterbi_token_frames(log_probs, ids)
    if frame_spans is None:
        return None
    frame_spans = _fill_missing(frame_spans, log_probs.shape[0])
    frame_time = duration / log_probs.shape[0]
    windows: list[tuple[float, float]] = []
    prev_end = 0.0
    for start_f, end_f in frame_spans:
        start = min(max(start_f * frame_time, prev_end), duration)
        end = min(max(end_f * frame_time, start), duration)
        windows.append((start, end))
        prev_end = end
    return windows
