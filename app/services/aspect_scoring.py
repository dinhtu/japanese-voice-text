"""GOPT-style utterance scores from this app's own Japanese signals.

The MIT GOPT checkpoint (SpeechOcean762 + Librispeech GOP) is English-only
and would invent Japanese scores, so it is not loaded. This module predicts
the same five utterance aspects GOPT does -- accuracy, fluency, rhythm,
intonation, overall -- from measurements the repo already makes:

* pronunciation  -- CER score (app.services.scoring)
* fluency        -- mora/sec vs a careful-reading band
* rhythm         -- mora-duration evenness + sokuon/chouon shortfalls
* intonation     -- F0 direction vs the OpenJTalk H/L pattern
* overall        -- weighted mix of whichever aspects were actually measured

No extra neural net, no extra VRAM. Numbers are deterministic and testable.
Unmeasured intonation (no voiced F0) stays None rather than a guessed value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from app.services.pitch_accent import MoraPitch
    from app.services.prosody_issues import DurationIssue
    from app.services.scoring import PronunciationError

# Careful Japanese read-aloud sits around 4--7 mora/s. Outside that band
# the pace score falls; the plateau is wide so a slow learner is not
# punished for being careful.
_FLUENCY_LO = 3.8
_FLUENCY_HI = 7.2
_FLUENCY_TOO_SLOW = 1.2
_FLUENCY_TOO_FAST = 12.0

# Ordinary-mora duration CV: mora-timed speech is fairly even. L2 reading
# is allowed more jitter than native before the evenness score drops.
_RHYTHM_CV_GOOD = 0.35
_RHYTHM_CV_BAD = 0.85

# GOPT/Azure-style mix: accuracy heaviest, the three prosodic heads share
# the rest. Missing heads are dropped and the rest renormalized.
_WEIGHT_PRONUNCIATION = 0.40
_WEIGHT_FLUENCY = 0.20
_WEIGHT_RHYTHM = 0.20
_WEIGHT_INTONATION = 0.20


@dataclass(frozen=True)
class AspectScores:
    overall_score: float
    pronunciation_score: float
    fluency_score: float
    rhythm_score: float
    intonation_score: float | None
    rhythm_measured: bool
    intonation_measured: bool
    method: str = "local-aspect"


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _round_score(value: float) -> float:
    return round(_clamp(value), 2)


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def score_fluency(
    duration_s: float, n_morae: int, pronunciation: float
) -> float:
    """Pace of the recording vs a careful Japanese reading band.

    Completeness is folded in: a short clip that happens to sit at a
    native mora/sec because half the sentence was skipped is not fluent.
    """
    if n_morae <= 0 or duration_s <= 0:
        return 0.0

    rate = n_morae / duration_s
    if _FLUENCY_LO <= rate <= _FLUENCY_HI:
        pace = 100.0
    elif rate < _FLUENCY_LO:
        pace = _lerp(rate, _FLUENCY_TOO_SLOW, _FLUENCY_LO, 25.0, 100.0)
    else:
        pace = _lerp(rate, _FLUENCY_HI, _FLUENCY_TOO_FAST, 100.0, 20.0)

    completeness = 0.4 + 0.6 * (pronunciation / 100.0)
    return _round_score(pace * completeness)


def score_rhythm(
    windows: Sequence[tuple[float, float]] | None,
    moras: Sequence["MoraPitch"] | None,
    duration_issues: Sequence["DurationIssue"],
    errors: Sequence["PronunciationError"],
) -> tuple[float, bool]:
    """Evenness of mora timing, plus sokuon/chouon and insertion/deletion.

    When per-mora windows are missing, still score from edit ops -- those
    are rhythm breaks (dropped / extra beats) even without duration data.
    `measured` is True only when real windows were used.
    """
    n_ins = sum(1 for e in errors if e.type == "ins")
    n_del = sum(1 for e in errors if e.type == "del")
    edit_penalty = min(28.0, 6.0 * n_del + 5.0 * n_ins)

    issue_penalty = 0.0
    for issue in duration_issues:
        if issue.expected_seconds <= 0:
            continue
        ratio = min(issue.measured_seconds / issue.expected_seconds, 1.0)
        issue_penalty += min(18.0, (1.0 - ratio) * 22.0)

    measured = bool(
        windows
        and moras
        and len(windows) == len(moras)
        and len(windows) >= 2
    )
    if not measured:
        return _round_score(100.0 - edit_penalty), False

    ordinary: list[float] = []
    for mora, (start, end) in zip(moras, windows):
        dur = end - start
        if mora.mora in {"っ", "ー"} or dur <= 0:
            continue
        ordinary.append(dur)

    if len(ordinary) < 2:
        evenness = 100.0
    else:
        mean = sum(ordinary) / len(ordinary)
        if mean <= 0:
            evenness = 100.0
        else:
            var = sum((d - mean) ** 2 for d in ordinary) / len(ordinary)
            cv = (var ** 0.5) / mean
            if cv <= _RHYTHM_CV_GOOD:
                evenness = 100.0
            elif cv >= _RHYTHM_CV_BAD:
                evenness = 32.0
            else:
                evenness = _lerp(cv, _RHYTHM_CV_GOOD, _RHYTHM_CV_BAD, 100.0, 32.0)

    return _round_score(evenness - issue_penalty - edit_penalty), True


def score_intonation(
    pitch_matched: int | None, pitch_total: int | None
) -> tuple[float | None, bool]:
    """Share of voiced morae whose F0 sat on the expected H/L side.

    Floor of 20 when something was voiced but every mora went the wrong
    way -- they did produce pitch, it just did not match. None when there
    is no voiced F0 at all (do not invent a number).
    """
    if not pitch_total:
        return None, False
    ratio = (pitch_matched or 0) / pitch_total
    return _round_score(20.0 + 80.0 * ratio), True


def combine_overall(
    pronunciation: float,
    fluency: float,
    rhythm: float,
    intonation: float | None,
) -> float:
    parts: list[tuple[float, float]] = [
        (pronunciation, _WEIGHT_PRONUNCIATION),
        (fluency, _WEIGHT_FLUENCY),
        (rhythm, _WEIGHT_RHYTHM),
    ]
    if intonation is not None:
        parts.append((intonation, _WEIGHT_INTONATION))
    total_w = sum(w for _s, w in parts)
    if total_w <= 0:
        return _round_score(pronunciation)
    return _round_score(sum(s * w for s, w in parts) / total_w)


def score_aspects(
    *,
    pronunciation_cer_score: int,
    n_morae: int,
    audio_duration: float,
    speech_duration: float | None = None,
    windows: Sequence[tuple[float, float]] | None = None,
    moras: Sequence["MoraPitch"] | None = None,
    duration_issues: Sequence["DurationIssue"] = (),
    errors: Sequence["PronunciationError"] = (),
    pitch_matched: int | None = None,
    pitch_total: int | None = None,
) -> AspectScores:
    """Build the five utterance scores from already-measured facts."""
    pronunciation = _round_score(float(pronunciation_cer_score))
    pace_duration = speech_duration if speech_duration and speech_duration > 0 else audio_duration
    fluency = score_fluency(pace_duration, n_morae, pronunciation)
    rhythm, rhythm_measured = score_rhythm(windows, moras, duration_issues, errors)
    intonation, intonation_measured = score_intonation(pitch_matched, pitch_total)
    overall = combine_overall(pronunciation, fluency, rhythm, intonation)
    return AspectScores(
        overall_score=overall,
        pronunciation_score=pronunciation,
        fluency_score=fluency,
        rhythm_score=rhythm,
        intonation_score=intonation,
        rhythm_measured=rhythm_measured,
        intonation_measured=intonation_measured,
    )
