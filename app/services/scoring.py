"""Pronunciation scoring: character-level edit distance between two readings.

This is a *text similarity* score over normalized hiragana, not an acoustic
phoneme-level pronunciation score. See README for the limitation.
"""

from dataclasses import dataclass, field

from src.asr.metrics import edit_distance, edit_ops

# score threshold (inclusive lower bound) -> (level, message)
_FEEDBACK_LEVELS: list[tuple[int, str, str]] = [
    (90, "excellent", "Pronunciation matches the target very closely."),
    (75, "good", "Pronunciation is close to the target."),
    (60, "fair", "Understandable, but several sounds differ from the target."),
    (40, "poor", "Many sounds differ from the target. Try reading more slowly."),
    (0, "mismatch", "The speech does not match the target text."),
]


@dataclass
class PronunciationError:
    """One character-level mismatch between target and recognized reading."""

    type: str  # "sub" | "del" | "ins"
    target: str
    recognized: str
    position: int


@dataclass
class ScoreResult:
    score: int
    cer: float
    distance: int
    target_length: int
    level: str
    message: str
    errors: list[PronunciationError] = field(default_factory=list)


def _feedback(score: int) -> tuple[str, str]:
    for threshold, level, message in _FEEDBACK_LEVELS:
        if score >= threshold:
            return level, message
    return _FEEDBACK_LEVELS[-1][1], _FEEDBACK_LEVELS[-1][2]


def collect_errors(target: str, recognized: str) -> list[PronunciationError]:
    """Character-level diff between the two readings, in target order."""
    ref, hyp = list(target), list(recognized)
    errors = []
    for op, ref_idx, hyp_idx in edit_ops(ref, hyp):
        if op == "sub":
            errors.append(PronunciationError("sub", ref[ref_idx], hyp[hyp_idx], ref_idx))
        elif op == "del":
            errors.append(PronunciationError("del", ref[ref_idx], "", ref_idx))
        else:  # ins
            errors.append(PronunciationError("ins", "", hyp[hyp_idx], ref_idx))
    return errors


def score_pronunciation(
    target_hiragana: str,
    recognized_hiragana: str,
    with_errors: bool = True,
) -> ScoreResult:
    """Compare two normalized hiragana strings and produce a 0-100 score.

        CER        = edit_distance / len(target)
        similarity = max(0, 1 - CER)
        score      = round(similarity * 100)

    Raises:
        ValueError: If the target reading is empty (nothing to compare against).
    """
    ref, hyp = list(target_hiragana), list(recognized_hiragana)
    if not ref:
        raise ValueError("Target reading is empty; cannot score against it.")

    distance = edit_distance(ref, hyp)
    cer = distance / len(ref)
    score = round(max(0.0, 1.0 - cer) * 100)
    level, message = _feedback(score)

    return ScoreResult(
        score=score,
        cer=round(cer, 4),
        distance=distance,
        target_length=len(ref),
        level=level,
        message=message,
        errors=collect_errors(target_hiragana, recognized_hiragana) if with_errors else [],
    )
