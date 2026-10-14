"""English sentence-intonation payload (reference + learner + checks).

Returned as `intonation` by /api/pronunciation-en/pitch-accent (reference
only) and /api/pronunciation-en/evaluate (reference + learner). All pitch
values are semitones relative to the speaker's mean F0, so the reference
contour and the learner points share one axis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.english_intonation import IntonationResult, SentenceAnalysis


class SyllableItem(BaseModel):
    text: str = Field(description='Spelling chunk, e.g. "TEA" of "teacher"')
    ipa: str | None = Field(description="IPA without stress marks; null if not in CMUdict")
    stress: int = Field(description="Dictionary stress: 1 primary, 2 secondary, 0 none")


class IntonationWordItem(BaseModel):
    index: int
    text: str = Field(description="Word as written in the target")
    phrase: int = Field(description="0-based intonation phrase (split at punctuation)")
    ipa: str | None = Field(description='"ˈæn.ə" style IPA with stress marks')
    content: bool = Field(description="Content word (gets sentence stress)")
    sentence_stress: bool = Field(description="Accented in the sentence (● row)")
    nucleus: bool = Field(description="Main accent of its phrase (◉)")
    stress_index: int = Field(description="Dictionary primary-stress syllable")
    source: str = Field(description='"cmudict" or "rule" (spelling guess, no IPA)')
    syllables: list[SyllableItem]


class ReferencePointItem(BaseModel):
    word: int
    syllable: int
    role: str = Field(description='"weak" | "secondary" | "stress" | "nucleus"')
    semitone: float = Field(description="Reference pitch at the syllable")
    end: float = Field(description="Reference pitch at the syllable end (final fall/rise)")
    lo: float = Field(description="Bottom of the acceptable band")
    hi: float = Field(description="Top of the acceptable band")


class LearnerSyllableItem(BaseModel):
    word: int
    syllable: int
    start: float = Field(description="Seconds")
    end: float = Field(description="Seconds")
    semitone: float | None = Field(description="Median F0; null when unvoiced")
    start_semitone: float | None
    end_semitone: float | None
    peak: float | None
    energy_db: float | None
    speech_time: float = Field(description="Seconds of speech inside the window")
    voiced_time: float = Field(description="Seconds with a measured F0")
    missing: bool = Field(default=False, description="Word not found in the recording")


class PeakCheckItem(BaseModel):
    word: int
    syllable: int
    ok: bool | None = Field(description="null = could not measure (unvoiced)")
    delta: float | None = Field(description="Semitones above the weak syllables around it")


class BreakCheckItem(BaseModel):
    after_word: int
    ok: bool | None = Field(description="null = no reliable timing")
    pause: float | None = Field(description="Longest silence at the boundary, seconds")
    drop: float | None = Field(description="Pitch fall into / reset after the boundary, st")


class FinalCheckItem(BaseModel):
    expected: str = Field(description='"fall" | "rise" | "either" (tag question)')
    ok: bool | None
    delta: float | None = Field(description="End minus nucleus / phrase level, st (negative = fell)")
    creaky: bool = Field(
        default=False,
        description="Statement ended in creak after falling (F0 too low to track) -- counted as a fall",
    )
    phrase: int = Field(default=0, description="Phrase whose end this checks")


class WordStressItem(BaseModel):
    word: int
    expected: int = Field(description="Dictionary stressed syllable")
    heard: int | None = Field(description="Most prominent syllable in the recording")
    ok: bool | None
    accepted: list[int] = Field(
        default_factory=list,
        description='Every dictionary stress ("record": noun 0, verb 1)',
    )


class EnglishIntonation(BaseModel):
    words: list[IntonationWordItem]
    phrase_ends: list[str] = Field(
        description='Per phrase: "continue" | "fall" | "rise" | "either" (tag question)',
    )
    final_tone: str
    reference: list[ReferencePointItem] = Field(description="One point per syllable, in order")
    learner: list[LearnerSyllableItem] | None = None
    aligned: bool | None = Field(
        default=None, description="True = forced-aligned timing; False = rough split",
    )
    peaks: list[PeakCheckItem] = Field(default_factory=list)
    breaks: list[BreakCheckItem] = Field(default_factory=list)
    final: FinalCheckItem | None = Field(default=None, description="End of the last sentence")
    ends: list[FinalCheckItem] = Field(
        default_factory=list, description="Every sentence end, in order (last == final)",
    )
    missing: list[int] = Field(
        default_factory=list, description="Words the recording does not contain",
    )
    word_stress: list[WordStressItem] = Field(default_factory=list)
    score: float | None = Field(default=None, description="0-100 intonation score")


def build_intonation(
    analysis: SentenceAnalysis, result: IntonationResult | None = None
) -> EnglishIntonation:
    payload = EnglishIntonation(
        words=[
            IntonationWordItem(
                index=w.index,
                text=w.text,
                phrase=w.phrase,
                ipa=w.ipa,
                content=w.content,
                sentence_stress=w.sentence_stress,
                nucleus=w.nucleus,
                stress_index=w.pron.stress_index,
                source=w.pron.source,
                syllables=[SyllableItem(text=s.text, ipa=s.ipa, stress=s.stress) for s in w.pron.syllables],
            )
            for w in analysis.words
        ],
        phrase_ends=list(analysis.phrase_ends),
        final_tone=analysis.final_tone,
        reference=[ReferencePointItem(**vars(r)) for r in analysis.reference],
    )
    if result is None:
        return payload
    return payload.model_copy(update={
        "learner": [
            LearnerSyllableItem(**{k: v for k, v in vars(m).items()}) for m in result.syllables
        ],
        "aligned": result.aligned,
        "peaks": [PeakCheckItem(**vars(c)) for c in result.peaks],
        "breaks": [BreakCheckItem(**vars(c)) for c in result.breaks],
        "final": FinalCheckItem(**vars(result.final)) if result.final else None,
        "ends": [FinalCheckItem(**vars(c)) for c in result.ends],
        "missing": sorted(result.missing),
        "word_stress": [
            WordStressItem(**{**vars(c), "accepted": list(c.accepted)}) for c in result.word_stress
        ],
        "score": result.score,
    })
