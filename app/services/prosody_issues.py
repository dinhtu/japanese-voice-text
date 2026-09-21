"""Detects specific, well-known Vietnamese-learner pronunciation issues from
real per-mora timing (app.services.mora_timing) -- currently: sokuon ("っ")
and chouon ("ー") held for too short a time, one of the single most common
mistakes Vietnamese speakers make with Japanese (running "chotto" together
into something closer to "choto").

## Why duration, not just presence/absence

っ (sokuon) is not its own distinct sound -- it is a silent hold before the
next consonant releases, roughly as long as an ordinary mora. ー (chouon)
is a vowel held for a second mora's worth of time. Both are, first and
foremost, TIMING phenomena, so this measures how long that mora's own
window (from app.services.mora_timing.mora_time_windows) actually lasted
against how long an *ordinary* mora took in this same recording -- not a
fixed absolute threshold, so it adapts to the speaker's own pace instead
of penalizing someone who just talks slowly (or quickly) throughout.

## Limitation

This only flags a *clear* shortfall (see _SHORT_RATIO) -- there is no
attempt to judge gemination "quality" beyond duration, and the timing
windows themselves inherit whatever imprecision
app.services.mora_timing.mora_time_windows has (see its own docstring: a
dropped っ that the ASR also failed to recognize as anything falls back to
an *interpolated* window, which in practice tends to come out short
exactly when the learner ran the sounds together -- but it is still an
estimate, not a direct measurement of silence). A recording with no
sokuon/chouon at all, or too few ordinary morae to form a baseline,
naturally produces no issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only needed for the type hints below; deferred so this module (pure
    # duration arithmetic) never has to import pyopenjtalk just to run.
    from app.services.pitch_accent import MoraPitch

SOKUON = "っ"
CHOUON = "ー"

# A sokuon/chouon read at or above this fraction of the recording's own
# median "ordinary" mora duration is left alone -- only a clear shortfall
# is worth coaching on.
_SHORT_RATIO = 0.55


@dataclass
class DurationIssue:
    """One sokuon/chouon mora read clearly shorter than expected."""

    kind: str  # "short_sokuon" | "short_chouon"
    mora_index: int  # 0-based position in the target's mora sequence
    mora: str
    prev_mora: str | None
    next_mora: str | None
    measured_seconds: float
    expected_seconds: float  # the recording's own median ordinary-mora duration


def _ordinary_mora_durations(
    moras: list[MoraPitch], windows: list[tuple[float, float]]
) -> list[float]:
    return [
        end - start
        for mora, (start, end) in zip(moras, windows)
        if mora.mora not in (SOKUON, CHOUON) and (end - start) > 0
    ]


def detect_duration_issues(
    moras: list[MoraPitch], windows: list[tuple[float, float]]
) -> list[DurationIssue]:
    """Flag sokuon/chouon morae read clearly shorter than this recording's
    own median ordinary-mora duration.

    `moras` and `windows` must be the same length and positionally aligned
    (app.services.pitch_accent.pitch_accent_pattern's output, and
    app.services.mora_timing.mora_time_windows's output, for the same
    target text). Returns [] if there is nothing to measure against (e.g.
    too few ordinary morae to form a baseline) or nothing worth flagging.
    """
    if len(moras) != len(windows) or not moras:
        return []

    ordinary = _ordinary_mora_durations(moras, windows)
    if not ordinary:
        return []
    ordinary.sort()
    baseline = ordinary[len(ordinary) // 2]  # median: robust to a stray long/short outlier
    if baseline <= 0:
        return []

    issues: list[DurationIssue] = []
    for i, (mora, (start, end)) in enumerate(zip(moras, windows)):
        if mora.mora not in (SOKUON, CHOUON):
            continue
        measured = end - start
        if measured >= _SHORT_RATIO * baseline:
            continue
        issues.append(
            DurationIssue(
                kind="short_sokuon" if mora.mora == SOKUON else "short_chouon",
                mora_index=i,
                mora=mora.mora,
                prev_mora=moras[i - 1].mora if i > 0 else None,
                next_mora=moras[i + 1].mora if i + 1 < len(moras) else None,
                measured_seconds=round(measured, 3),
                expected_seconds=round(baseline, 3),
            )
        )
    return issues
