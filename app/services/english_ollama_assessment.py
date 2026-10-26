"""`ollama_assessment` on English /evaluate: the reading scored from the
AUDIO by Qwen3-ASR (EN_OLLAMA_ASR_MODEL, default frozenlab/qwen3-asr) in
Ollama -- separate from OLLAMA_MODEL, the text model /coach uses. The
Ollama calls live in app/services/qwen3_asr.py.

1. Qwen3-ASR transcribes the learner's audio (verbatim prompt, never the
   target sentence).
2. That transcript replaces wav2vec2's: the same six criteria are scored
   by english_assessment.assess_reading on it, over the same acoustics
   (forced-alignment timing, pitch checks, VAD). Content, the per-word
   verdicts, the gate and words-correct-per-minute follow Qwen3-ASR.
   Phoneme accuracy uses, per word read, Qwen3-ASR's margin between the
   word and its best rival among the top-20 token alternatives (see
   qwen3_asr.word_evidence) when Ollama returns them, otherwise the
   forced-alignment estimate (`confidence_source` says which).
3. `agreement` puts the two ASRs' word verdicts side by side. Qwen3-ASR has
   a strong language model that "repairs" misread words, so a word it ALSO
   hears wrong is strong evidence of a real error; a word it hears right
   is weaker evidence than wav2vec2's.

Optional throughout: Ollama down, model not pulled or an old server is a
`status`, never a failed /evaluate.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from app.services.english_assessment import _cmu_phones, assess_reading
from app.services.english_scoring import (
    ExtraPiece,
    WordMatch,
    align_word_tokens,
    expand_contractions,
    transcript_tokens,
)
from app.services.english_text import expand_for_speech
from app.services.qwen3_asr import (  # noqa: F401 -- re-exported for callers / tests
    AsrError,
    audio_windows,
    letters,
    parse_reply,
    raw_word_confidences,
    stream_positions,
    transcribe,
    wav_bytes,
    word_evidence,
)

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.services.english_assessment import AssessmentInputs

logger = logging.getLogger(__name__)

SCORING_VERSION = "en-qwen3asr-v1"
_OK_STATUSES = ("correct", "mispronounced")


def _speech_words(word: str, target_text: str | None = None) -> list[str]:
    """Alignment tokens of one transcript word (see transcript_tokens), a
    contraction written out where the target has the full form."""
    if target_text:
        word = expand_contractions(word, target_text)
    return transcript_tokens(expand_for_speech(word))


def speech_token_words(transcript: str, target_text: str | None = None
                       ) -> tuple[list[str], list[int] | None, list[int | None] | None]:
    """The alignment tokens of `transcript` (numbers / titles written out,
    apostrophes dropped -- as align_word_tokens sees them) and, per token,
    the whitespace word it came from and the character offset in that word
    where its letters start ("one" in "twenty-one" -> 7; None when the
    token is not spelled there, as "one" from "21"). Both lists are None if
    expanding word by word does not give the same tokens as the whole text.
    With `target_text`, "It's" is read "it is" where the target says so."""
    tokens = _speech_words(transcript, target_text)
    owners: list[int] = []
    offsets: list[int | None] = []
    rebuilt: list[str] = []
    for k, word in enumerate(transcript.split()):
        alpha = [(j, ch) for j, ch in enumerate(word.lower()) if "a" <= ch <= "z"]
        spelled = "".join(ch for _j, ch in alpha)
        cursor = 0
        for t in _speech_words(word, target_text):
            rebuilt.append(t)
            owners.append(k)
            at = spelled.find(t, cursor)
            if at >= 0:
                offsets.append(alpha[at][0])
                cursor = at + len(t)
            else:
                offsets.append(None)
    if rebuilt != tokens:
        return tokens, None, None
    return tokens, owners, offsets


def speech_tokens_with_confidence(transcript: str, raw_conf: Sequence[float] | None,
                                  target_text: str | None = None) -> tuple[list[str], list[float] | None]:
    """The alignment tokens of `transcript` and, if known, each one's
    confidence (that of the whitespace word it came from)."""
    tokens, owners, _offsets = speech_token_words(transcript, target_text)
    if raw_conf is None or owners is None or len(transcript.split()) != len(raw_conf):
        return tokens, None
    return tokens, [raw_conf[k] for k in owners]


def target_positions(matches: Sequence[WordMatch], extras: Sequence[ExtraPiece],
                     tokens: Sequence[str]) -> list[tuple[int, str]] | None:
    """Where each target word sits in the transcript tokens, replaying
    align_word_tokens' path (extras sit after the word they follow; a word
    may take 1-3 tokens or half of one). Per word (token index, kind):
    "heard" -- the transcript has it at that token (joined tokens start
    there); "split_first" / "split_rest" -- it shares one token with its
    neighbour ("cannot" -> can + not); "gap" -- skipped, index of the next
    token said (len(tokens) = after the last). None if the replay does not fit."""
    after: dict[int, list[ExtraPiece]] = defaultdict(list)
    for e in extras:
        after[e.after].append(e)
    out: list[tuple[int, str]] = []
    p, rest = 0, None

    def take_extra(e: ExtraPiece) -> bool:
        nonlocal p
        if rest is not None or p >= len(tokens) or tokens[p] != e.letters:
            return False
        p += 1
        return True

    for e in after[-1]:
        if not take_extra(e):
            return None
    for i, m in enumerate(matches):
        need = m.heard
        if not need:
            out.append((p, "gap"))
        elif rest is not None:
            if need != rest:
                return None
            out.append((p, "split_rest"))
            p, rest = p + 1, None
        elif p < len(tokens) and tokens[p] == need:
            out.append((p, "heard"))
            p += 1
        elif p < len(tokens) and tokens[p].startswith(need):
            out.append((p, "split_first"))
            rest = tokens[p][len(need):]
        else:
            start, joined = p, ""
            while p < len(tokens) and len(joined) < len(need):
                joined += tokens[p]
                p += 1
            if joined != need:
                return None
            out.append((start, "heard"))
        for e in after[i]:
            if not take_extra(e):
                return None
    return out if p == len(tokens) and rest is None else None


def target_confidence(matches: Sequence[WordMatch], extras: Sequence[ExtraPiece],
                      tokens: Sequence[str], conf: Sequence[float] | None) -> list[float | None] | None:
    """Confidence per target word (None for a skipped one); None if unknown."""
    if conf is None:
        return None
    positions = target_positions(matches, extras, tokens)
    if positions is None:
        return None
    out: list[float | None] = []
    for m, (p, kind) in zip(matches, positions):
        if kind == "gap":
            out.append(None)
        elif kind == "heard" and p < len(tokens) and len(m.heard) > len(tokens[p]):
            # Joined tokens ("to day"): the weakest of them.
            joined, k, worst = "", p, 1.0
            while k < len(tokens) and len(joined) < len(m.heard):
                joined += tokens[k]
                worst = min(worst, conf[k])
                k += 1
            out.append(worst)
        else:
            out.append(conf[p])
    return out


_SHORT_WORDS = frozenset(
    "a i am an as at be by do go he hi if in is it me my no of oh ok on or so to up us we".split()
)


def _is_word(letters_: str) -> bool:
    """A real word, not a fragment: "walk", "an", "a" -- not "gl", "wat",
    nor CMU's odd short entries ("na", "ed")."""
    if len(letters_) <= 2:
        return letters_ in _SHORT_WORDS
    return _cmu_phones(letters_) is not None


def _homophone(token_letters: str, target: str) -> bool:
    a, b = _cmu_phones(token_letters), _cmu_phones(target)
    return bool(a and b and set(a) & set(b))


def rival_evidence(stream: Sequence[dict] | None, transcript: str, matches: Sequence[WordMatch],
                   extras: Sequence[ExtraPiece], target_text: str | None = None) -> list[dict | None] | None:
    """Pronunciation evidence (qwen3_asr.word_evidence) for every target
    word the transcript has, read at the stream token where its letters
    start. None per word that was skipped, shares a token with the word
    before it, or is not spelled in the transcript (digits); None overall
    when the stream / transcript / alignment do not line up."""
    if not stream or not transcript:
        return None
    tokens, owners, offsets = speech_token_words(transcript, target_text)
    positions = target_positions(matches, extras, tokens)
    located = stream_positions(stream, transcript)
    if positions is None or owners is None or located is None:
        return None
    word_starts, token_at = located
    target_letters = [letters(m.target) for m in matches]
    # Target words skipped at each spot: the model still half-expects them
    # there, so they are no rival for the word actually said ("like glass").
    skipped: dict[int, list[str]] = defaultdict(list)
    for (p, kind), w in zip(positions, target_letters):
        if kind == "gap":
            skipped[p].append(w)
    out: list[dict | None] = []
    for k, (m, (p, kind)) in enumerate(zip(matches, positions)):
        if kind in ("gap", "split_rest") or p >= len(tokens) or offsets[p] is None:
            out.append(None)
            continue
        idx = token_at(word_starts[owners[p]] + offsets[p])
        if idx is None:
            out.append(None)
            continue
        target = target_letters[k]
        own = []
        if kind == "split_first":
            own.append(tokens[p])  # "cannot" stands for can + not
        if len(m.heard) > len(tokens[p]):
            own.append(tokens[p])  # "in" of "in to" for "into"
        if tokens[p] == target or kind == "split_first" or m.ok:
            # The token the model wrote here spells this very word (or an
            # accepted spelling of it: "flavor" for "flavour").
            own.append(letters(str(stream[idx].get("token", ""))))
        # Skipped words, and the next target word (the model wondering
        # whether this one was skipped), are content questions, not rivals.
        ignore = [w for w in skipped.get(p, []) if w != target]
        if k + 1 < len(target_letters) and target_letters[k + 1] != target:
            ignore.append(target_letters[k + 1])
        out.append(word_evidence(stream[idx].get("top_logprobs") or [], target,
                                 same_word=_homophone, ignore=ignore, extra_own=own, is_word=_is_word))
    return out


# --- report ----------------------------------------------------------------

def agreement(wav2vec2: dict | None, qwen: dict) -> dict | None:
    """Per target word: did each ASR hear it as the target?"""
    a = (wav2vec2 or {}).get("word_results") or []
    b = qwen.get("word_results") or []
    if not a or len(a) != len(b):
        return None
    both_ok, both_wrong, only_w2v, only_qwen = [], [], [], []
    for wa, wb in zip(a, b):
        ok_a, ok_b = wa["status"] in _OK_STATUSES, wb["status"] in _OK_STATUSES
        bucket = both_ok if ok_a and ok_b else both_wrong if not ok_a and not ok_b else (
            only_qwen if ok_a else only_w2v
        )
        bucket.append(wa["word"])
    return {
        "same_verdict_share": round((len(both_ok) + len(both_wrong)) / len(a), 4),
        "both_read": both_ok,
        "both_wrong": both_wrong,
        "wrong_only_by_wav2vec2": only_w2v,
        "wrong_only_by_qwen3_asr": only_qwen,
    }


def _base(status: str, settings: "Settings | None", **extra: Any) -> dict:
    return {
        "status": status,
        "model": getattr(settings, "en_ollama_asr_model", None),
        "language": None,
        "transcript": None,
        "recognized_text": None,
        "overall_score": None,
        "weights_used": None,
        "content_gate": None,
        "criteria": None,
        "word_results": [],
        "inserted": [],
        "repeated": [],
        "improvements": [],
        "confidence_source": None,
        "agreement": None,
        "comparison": None,
        "latency_ms": None,
        "error": None,
        "scoring_version": SCORING_VERSION,
        **extra,
    }


def score_transcript(inputs: "AssessmentInputs", transcript: str,
                     raw_conf: Sequence[float] | None,
                     stream: Sequence[dict] | None = None) -> tuple[dict, str]:
    """assess_reading on Qwen3-ASR's transcript; (report, confidence_source).
    Phoneme accuracy: the rival margin per word read (from `stream`'s
    top-20 alternatives), else the chosen tokens' probability, else the
    forced-alignment estimate."""
    tokens, conf = speech_tokens_with_confidence(transcript, raw_conf, inputs.target_text)
    recognized = " ".join(tokens)
    matches, extras = align_word_tokens(inputs.target_text, recognized)
    source, metric = "forced_alignment", "asr_token_confidence"
    evidence = rival_evidence(stream, transcript, matches, extras, inputs.target_text)
    if evidence is not None and any(e and e.get("score") is not None for e in evidence):
        word_conf = [e["score"] / 100.0 if e and e.get("score") is not None else None for e in evidence]
        source = metric = "qwen3_asr_rival_margin"
    else:
        word_conf = target_confidence(matches, extras, tokens, conf)
        if word_conf is not None:
            source = "qwen3_asr_logprobs"
    report = assess_reading(
        target_text=inputs.target_text,
        recognized_text=recognized,
        matches=matches,
        extras=extras,
        analysis=inputs.analysis,
        timing=inputs.timing,
        intonation=inputs.intonation,
        vad=inputs.vad,
        duration=inputs.duration,
        legacy_score=inputs.legacy_score,
        lang=inputs.lang,
        word_confidence=word_conf,
        confidence_metric=metric,
    )
    return report, source


async def assess_with_ollama(
    inputs: "AssessmentInputs | None",
    reading: dict | None,
    gopt: dict | None,
    settings: "Settings",
) -> dict:
    """The `ollama_assessment` dict. Never raises."""
    if not getattr(settings, "en_ollama_assessment", True):
        return _base("disabled", settings)
    if reading and reading.get("status") == "no_voice":
        return _base("no_voice", settings)
    if inputs is None or inputs.samples is None or not inputs.sample_rate:
        return _base("skipped", settings, error="no audio / acoustics to score")
    t0 = time.perf_counter()
    try:
        heard = await transcribe(np.asarray(inputs.samples), int(inputs.sample_rate), settings)
    except AsrError as e:
        logger.warning("Qwen3-ASR assessment: %s", e)
        return _base(e.status, settings, error=str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("Qwen3-ASR assessment failed", exc_info=True)
        return _base("unavailable", settings, error=str(e) or type(e).__name__)
    latency = round((time.perf_counter() - t0) * 1000, 1)
    try:
        report, source = score_transcript(inputs, heard["transcript"], heard["word_confidence"], heard.get("stream"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Qwen3-ASR transcript scoring failed", exc_info=True)
        return _base("invalid_output", settings, latency_ms=latency, error=f"scoring: {e}")

    status = "no_speech" if not heard["transcript"] else report["status"]
    overall = report["overall_score"]
    reading_overall = (reading or {}).get("overall_score")
    return _base(
        status,
        settings,
        language=heard["language"],
        language_mismatch=bool(heard["language"]) and heard["language"].lower() != "english",
        transcript=heard["transcript"],
        recognized_text=report["recognized_text"],
        overall_score=overall,
        weights_used=report["weights_used"],
        content_gate=report["content_gate"],
        criteria=report["criteria"],
        word_results=report["word_results"],
        inserted=report["inserted"],
        repeated=report["repeated"],
        improvements=report["improvements"],
        confidence_source=source,
        agreement=agreement(reading, report),
        latency_ms=latency,
        comparison={
            "qwen3_asr_overall": overall,
            "reading_overall": reading_overall,
            "gopt_average": (gopt or {}).get("average"),
            "gopt_adjusted_average": (gopt or {}).get("adjusted_average"),
            "legacy_score": (reading or {}).get("legacy_score"),
            "delta_vs_reading": (
                round(overall - reading_overall, 2)
                if overall is not None and reading_overall is not None else None
            ),
        },
    )
