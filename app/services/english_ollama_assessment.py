"""`ollama_assessment`: the local LLM's own scores for one English reading,
returned next to reading_assessment / gopt_assessment for comparison.

A text LLM cannot hear the audio, so it scores the MEASURED facts: the
transcript against the target, each word's status and letter clarity,
pace / pauses / repeats, and the pitch checks on words that were read
right. It is deliberately NOT shown reading_assessment's or GOPT's scores,
so its numbers are an independent opinion, not a copy.

Same six criteria and weights as reading_assessment. `overall_score` is
the model's own; `weighted_overall` re-weights the model's criterion
scores with english_assessment.WEIGHTS so the two reports compare like for
like. temperature 0 + a fixed seed keep repeated calls close, but an LLM
score is still an opinion: never feed it back into the other scores.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from app.services.english_assessment import WEIGHTS

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.services.english_intonation import IntonationResult

logger = logging.getLogger(__name__)

SCORING_VERSION = "en-ollama-v1"
SEED = 42
NUM_PREDICT = 700

_CRITERION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": ["number", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}
# Passed to Ollama's `format` so the reply is this JSON object.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "object",
            "properties": {key: _CRITERION_SCHEMA for key in WEIGHTS},
            "required": list(WEIGHTS),
        },
        "overall_score": {"type": ["number", "null"]},
        "summary": {"type": "string"},
    },
    "required": ["criteria", "overall_score", "summary"],
}

_REASON_LANG = {"vi": "Vietnamese", "en": "English"}

SYSTEM_PROMPT = """\
You are an experienced English pronunciation examiner. A learner read one \
English sentence aloud. You cannot hear the recording; you get a JSON block \
of facts MEASURED from it by speech-recognition and acoustic analysis. \
Score the reading from those facts only.

Score each criterion from 0 to 100 (100 = native-like for read-aloud):
- content_accuracy: were the right words read (skipped, replaced, extra words)?
- phoneme_accuracy: how clearly the words were articulated ("clarity" per word, \
0-100, from forced alignment; low = unclear or a different sound).
- word_stress: stress on the right syllable of multi-syllable words.
- intonation: key words raised in pitch; the sentence end falls / rises as expected.
- fluency: pace in words CORRECT per minute (about 80-220 is normal for reading \
aloud), hesitation pauses, repeated words, fillers.
- rhythm_and_linking: pauses at phrase breaks, words inside a phrase linked \
without pauses, function words shorter than stressed syllables \
(reduction_ratio below about 0.7 is natural English, above 1.0 is choppy).

RULES:
- Use only the facts given. If a criterion has no data (null or listed in \
"not_measured"), give it "score": null; never guess.
- Pitch, stress and timing facts cover only words that were read correctly. \
If few words were read correctly, prosody says little about the target \
sentence: judge it on that basis.
- "overall_score": your overall judgement, 0-100, weighting content and \
phoneme accuracy most.
- "reason": one short sentence per criterion; "summary": 1-2 sentences. \
Write them in {reason_lang}.
- Reply with ONLY the JSON object."""


def _clamp_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(max(0.0, min(100.0, float(value))), 2)


def llm_facts(reading: dict, intonation: "IntonationResult | None" = None) -> dict:
    """The facts block: measurements only, none of the computed scores."""
    words = reading["word_results"]
    read = {w["index"] for w in words if w["status"] in ("correct", "mispronounced")}

    def stress_label(w: dict) -> str | None:
        syllables, index = w.get("syllables"), w.get("expected_stress")
        if not syllables or len(syllables) < 2 or index is None:
            return None
        return "-".join(s.upper() if k == index else s.lower() for k, s in enumerate(syllables))

    content = reading["criteria"]["content_accuracy"].get("detail") or {}
    flu = reading["criteria"]["fluency"].get("detail") or {}
    rhythm = reading["criteria"]["rhythm_and_linking"].get("detail") or {}
    audio = reading.get("audio_analysis") or {}
    facts: dict[str, Any] = {
        "target_text": reading["reference_text"],
        "recognized_text": reading["recognized_text"],
        "words_read_correctly": f"{len(read)} of {len(words)}",
        "word_errors": {
            "substituted": content.get("substitutions"),
            "omitted": content.get("deletions"),
            "extra_words": [e["text"] for e in reading.get("inserted") or []],
            "repeated_words": [e["text"] for e in reading.get("repeated") or []],
            "fillers": flu.get("fillers"),
        },
        "words": [
            {
                "word": w["word"],
                "status": w["status"],
                "heard_as": w["recognized"],
                "clarity": w["phoneme_score"],
                "dictionary_stress": stress_label(w),
                "stress_correct": w["stress_ok"],
            }
            for w in words
        ],
        "timing": {
            "speech_seconds": audio.get("speech_seconds"),
            "words_per_minute": audio.get("words_per_minute"),
            "correct_words_per_minute": audio.get("correct_words_per_minute"),
            "pauses": audio.get("pause_count"),
            "pauses_not_at_punctuation": audio.get("unexpected_pause_count"),
        },
        "rhythm": {
            "pauses_inside_phrases_between": [
                f"{words[a]['word']} | {words[b]['word']}" for a, b in rhythm.get("broken_links", [])
            ] if rhythm else None,
            "reduction_ratio": rhythm.get("reduction_ratio"),
        },
    }
    if intonation is not None and len(intonation.analysis.words) == len(words):
        names = [w["word"] for w in words]
        facts["intonation"] = {
            "expected_sentence_end": intonation.analysis.final_tone,
            "key_word_pitch": [
                {"word": names[c.word], "raised": c.ok, "semitones_above_neighbours": c.delta}
                for c in intonation.peaks if c.ok is not None and c.word in read
            ],
            "sentence_ends": [
                {"expected": c.expected, "as_expected": c.ok, "pitch_change_semitones": c.delta}
                for c in intonation.ends if c.ok is not None
            ],
            "phrase_breaks": [
                {"after": names[c.after_word], "paused": c.ok, "pause_seconds": c.pause}
                for c in intonation.breaks
                if c.ok is not None and c.after_word in read and c.after_word + 1 in read
            ],
        }
    else:
        facts["intonation"] = None
    facts["not_measured"] = [
        key for key, c in reading["criteria"].items()
        if c["status"] in ("insufficient_evidence", "not_assessed")
    ]
    return facts


async def _chat(settings: "Settings", messages: list[dict]) -> str:
    """One constrained-JSON chat call; raises on any Ollama problem."""
    from ollama import AsyncClient

    client = AsyncClient(host=settings.ollama_host, timeout=settings.en_ollama_assessment_timeout_s)
    response = await client.chat(
        model=settings.ollama_model,
        messages=messages,
        think=False,
        stream=False,
        format=JSON_SCHEMA,
        options={"temperature": 0, "seed": SEED, "num_predict": NUM_PREDICT},
    )
    return ((response.get("message", {}) or {}).get("content", "") or "").strip()


def _base(status: str, settings: "Settings | None", **extra: Any) -> dict:
    return {
        "status": status,
        "model": getattr(settings, "ollama_model", None),
        "criteria": None,
        "overall_score": None,
        "weighted_overall": None,
        "summary": None,
        "comparison": None,
        "latency_ms": None,
        "error": None,
        "scoring_version": SCORING_VERSION,
        **extra,
    }


def parse_scores(content: str) -> tuple[dict, float | None, str]:
    """(criteria, overall, summary) from the model reply; ValueError if unusable."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"not JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("criteria"), dict):
        raise ValueError("no 'criteria' object")
    criteria = {}
    for key, weight in WEIGHTS.items():
        item = data["criteria"].get(key)
        item = item if isinstance(item, dict) else {}
        criteria[key] = {
            "score": _clamp_score(item.get("score")),
            "weight": weight,
            "reason": str(item.get("reason") or "").strip() or None,
        }
    return criteria, _clamp_score(data.get("overall_score")), str(data.get("summary") or "").strip()


def _weighted(criteria: dict) -> float | None:
    used = {k: c["score"] for k, c in criteria.items() if c["score"] is not None}
    if not used:
        return None
    return round(sum(WEIGHTS[k] * s for k, s in used.items()) / sum(WEIGHTS[k] for k in used), 2)


async def assess_with_ollama(
    reading: dict | None,
    gopt: dict | None,
    intonation: "IntonationResult | None",
    settings: "Settings",
    lang: str = "vi",
) -> dict:
    """The `ollama_assessment` dict. Never raises: a missing / down Ollama
    is a status, not a failed /evaluate."""
    if not getattr(settings, "en_ollama_assessment", True):
        return _base("disabled", settings)
    if not reading:
        return _base("skipped", settings, error="reading_assessment unavailable")
    if reading.get("status") == "no_voice":
        return _base("no_voice", settings)
    try:
        facts = llm_facts(reading, intonation)
    except Exception as e:  # noqa: BLE001
        logger.warning("Ollama assessment facts failed", exc_info=True)
        return _base("skipped", settings, error=f"facts: {e}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.replace("{reason_lang}", _REASON_LANG.get(lang, "English"))},
        {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
    ]
    t0 = time.perf_counter()
    try:
        content = await _chat(settings, messages)
    except ImportError:
        return _base("unavailable", settings, error="package 'ollama' is not installed")
    except Exception as e:  # noqa: BLE001 -- connection refused, timeout, model not pulled
        logger.warning("Ollama assessment call failed: %s", e)
        return _base("unavailable", settings, error=str(e) or type(e).__name__)
    latency = round((time.perf_counter() - t0) * 1000, 1)
    try:
        criteria, overall, summary = parse_scores(content)
    except ValueError as e:
        logger.warning("Ollama assessment reply unusable: %s", e)
        return _base("invalid_output", settings, latency_ms=latency, error=str(e))

    weighted = _weighted(criteria)
    reading_overall = reading.get("overall_score")
    return _base(
        "scored",
        settings,
        criteria=criteria,
        overall_score=overall,
        weighted_overall=weighted,
        summary=summary or None,
        latency_ms=latency,
        comparison={
            "ollama_overall": overall,
            "ollama_weighted_overall": weighted,
            "reading_overall": reading_overall,
            "gopt_average": (gopt or {}).get("average"),
            "gopt_adjusted_average": (gopt or {}).get("adjusted_average"),
            "legacy_score": reading.get("legacy_score"),
            "delta_vs_reading": (
                round(weighted - reading_overall, 2)
                if weighted is not None and reading_overall is not None else None
            ),
        },
    )
