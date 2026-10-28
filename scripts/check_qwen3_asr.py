"""Check that Ollama can run EN_OLLAMA_ASR_MODEL (frozenlab/qwen3-asr) for
English /evaluate's `ollama_assessment`, on a real recording.

    python scripts/check_qwen3_asr.py reading.wav
    python scripts/check_qwen3_asr.py reading.wav "I would like a glass of water."
    python scripts/check_qwen3_asr.py reading.wav "I would like a glass of water." --full
    python scripts/check_qwen3_asr.py reading.wav "I would like a glass of water." --v2

1. Ollama version (a recent 0.40.x is recommended for the qwen3a audio projector).
2. The model is pulled and reports the audio capability (/api/show).
3. The recording is transcribed; the language, transcript and whether
   per-word confidence (logprobs) came back are printed.
4. With a target: word accuracy of that transcript against it.
   With --full: the whole English evaluate (wav2vec2 + GOPT) and the three
   reports side by side. With --v2: the /evaluate-v2 pipeline (Qwen3-ASR,
   pronunciation evidence, timing, OLLAMA_MODEL feedback).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.english_ollama_assessment import (  # noqa: E402
    AsrError,
    assess_with_ollama,
    score_transcript,
    transcribe,
)
from app.services.qwen3_asr import base_url  # noqa: E402


def _probe(settings) -> bool:
    host = base_url(settings.ollama_host)
    try:
        version = httpx.get(f"{host}/api/version", timeout=5).json().get("version")
    except httpx.HTTPError as e:
        print(f"[x] Ollama not reachable at {host}: {e}")
        return False
    print(f"[ok] Ollama {version} at {host}")
    show = httpx.post(f"{host}/api/show", json={"model": settings.en_ollama_asr_model}, timeout=10)
    if show.status_code == 404:
        print(f"[x] {settings.en_ollama_asr_model} is not pulled: ollama pull {settings.en_ollama_asr_model}")
        return False
    caps = show.json().get("capabilities") or []
    flag = "[ok]" if "audio" in caps else "[!]"
    print(f"{flag} {settings.en_ollama_asr_model} capabilities: {caps}"
          + ("" if "audio" in caps else "  (no 'audio': this Ollama may be too old for qwen3a)"))
    return True


async def _run(args) -> int:
    settings = get_settings()
    if (args.v2 or args.full) and not args.target:
        print("[x] --v2 / --full need the target sentence: check_qwen3_asr.py audio.wav \"Target.\" --v2")
        return 2
    if not _probe(settings):
        return 1
    from src.asr.inference import load_audio

    samples, sr = load_audio(args.audio)
    t0 = time.perf_counter()
    try:
        heard = await transcribe(samples, sr, settings, "English")
    except AsrError as e:
        print(f"[x] {e.status}: {e}")
        return 1
    print(f"[ok] transcribed {len(samples) / sr:.2f}s in {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(f"     language:   {heard['language']}")
    print(f"     transcript: {heard['transcript']!r}")
    conf = heard["word_confidence"]
    print(f"     logprobs:   {'per-word confidence ' + str(conf) if conf else 'not returned (phoneme falls back to forced alignment)'}")
    if not args.target:
        return 0

    if args.v2:
        from app.services.english_asr import get_english_asr_service
        from app.services.english_evaluate_v2 import evaluate_v2, measure_acoustics

        acoustics = measure_acoustics(args.target, args.audio, get_english_asr_service(), settings.en_v2_aligner)
        report = await evaluate_v2(acoustics, settings, with_feedback=True)
        print(json.dumps({
            "status": report["status"],
            "overall_score": report["overall_score"],
            "criteria": {k: (c["score"], c["status"]) for k, c in report["criteria"].items()},
            "errors": [(e["type"], e["word"], e["heard"]) for e in report["errors"]],
            "pronunciation": [(w["text"], (w["pronunciation"] or {}).get("score"),
                               (w["pronunciation"] or {}).get("rival")) for w in report["word_results"]],
            "asr": report["asr"],
            "alignment": report["alignment"],
            "feedback": report["feedback"],
        }, ensure_ascii=False, indent=2))
        return 0

    if not args.full:
        from app.services.english_assessment import AssessmentInputs

        inputs = AssessmentInputs(args.target, None, None, None, None, len(samples) / sr, samples, sr)
        report, _source = score_transcript(inputs, heard["transcript"], conf, heard.get("stream"))
        content = report["criteria"]["content_accuracy"]
        print(f"[ok] content accuracy vs target: {content['score']} ({content['detail']})")
        print("     words:", [(w["word"], w["status"], w["recognized"]) for w in report["word_results"]])
        return 0

    from app.services.english_asr import get_english_asr_service
    from app.services.english_use_cases import EvaluateEnglishUseCase

    result = EvaluateEnglishUseCase(get_english_asr_service()).execute(args.target, args.audio)
    ollama = await assess_with_ollama(result.assessment_inputs, result.reading_assessment,
                                      result.gopt_assessment, settings)
    reading, gopt = result.reading_assessment or {}, result.gopt_assessment or {}
    print(json.dumps({
        "legacy_score": result.score.score,
        "wav2vec2_transcript": result.recognized_text,
        "reading_overall": reading.get("overall_score"),
        "gopt_average": gopt.get("average"),
        "gopt_adjusted_average": gopt.get("adjusted_average"),
        "ollama_status": ollama["status"],
        "qwen3_asr_overall": ollama["overall_score"],
        "criteria": {k: (reading.get("criteria", {}).get(k, {}).get("score"), (ollama.get("criteria") or {}).get(k, {}).get("score"))
                     for k in (reading.get("criteria") or {})},
        "agreement": ollama["agreement"],
        "confidence_source": ollama["confidence_source"],
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="recording (wav / mp3 / webm ...)")
    parser.add_argument("target", nargs="?", help="target sentence the learner read")
    parser.add_argument("--full", action="store_true", help="run the whole English evaluate too")
    parser.add_argument("--v2", action="store_true", help="run /evaluate-v2's pipeline (needs a target)")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
