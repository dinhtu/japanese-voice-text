"""Real-time ASR -> Claude compare runner (phoneme/hiragana/hybrid input).

For each finalized utterance, this script can send:
- phoneme sequence
- hiragana sequence
- hybrid payload (phoneme + hiragana + ASR confidence metadata)

This helps compare which representation yields more stable dialogue behavior.

Environment:
    ANTHROPIC_API_KEY can be placed in .env (loaded automatically).

Usage:
    uv run python scripts/realtime_asr_claude_compare.py --model medium
    uv run python scripts/realtime_asr_claude_compare.py --compare-mode hybrid
"""

import argparse
import collections
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from anthropic import Anthropic
from transformers import Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.realtime_asr_phoneme import (  # noqa: E402
    MODEL_PRESETS,
    SAMPLE_RATE,
    VADSegmenter,
    resolve_inference_dtype,
    transcribe,
)
from src.asr.kana_vocab import KanaVocab  # noqa: E402
from src.asr.model import load_checkpoint  # noqa: E402
from src.asr.phoneme_vocab import PhonemeVocab  # noqa: E402


DEFAULT_SYSTEM_PROMPT = """You are a dialogue assistant receiving noisy real-time ASR.

Input can be either:
- phoneme sequence (space-separated Japanese phonemes), or
- hiragana sequence.

ASR profile and limits:
- Loanwords can be distorted (especially v/b and long-vowel variants).
- Some utterances lose punctuation and word boundaries.
- Confidence metadata is attached; low-confidence tokens include alternatives.

Behavior:
- Infer likely user intent conservatively from ASR metadata.
- By default, avoid clarification questions and proceed with the best-effort interpretation.
- Prefer concise conversational Japanese unless the user asks otherwise.
- If force_clarification is true, ask exactly one short clarification question first.
"""

TRAINING_PROFILE_PROMPT = """Empirical ASR profile (H100 medium training, ep5):
- Domain shift is large: JSUT KER 7.47%, JVS 15.68%, ReazonSpeech 21.65%.
- Frequent kana confusions: ー <-> ん/い/あ/え, い <-> え, small-kana related errors.
- Frequent phoneme confusions: U <-> u, I <-> i, u -> i, d -> t, palatalized drop (my/by/gy -> m/b/g).
- Weak regions: long vowels, small kana, glides/palatalized sounds, noisy speech, loanwords.
- Confidence metadata is meaningful; low-confidence tokens are often true error hotspots.
"""


def parse_args():
    p = argparse.ArgumentParser(description="Real-time ASR -> Claude compare runner")

    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), default="medium")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--pretrained", default=None)

    p.add_argument("--preview-interval", type=float, default=0.3)
    p.add_argument("--preview-min-delta", type=float, default=0.4)
    p.add_argument("--preview-max-audio", type=float, default=6.0)
    p.add_argument("--silence-timeout", type=float, default=0.8)
    p.add_argument("--max-utterance", type=float, default=15.0)
    p.add_argument("--vad-threshold", type=float, default=0.4)
    p.add_argument("--prebuffer-sec", type=float, default=0.5)
    p.add_argument("--device-id", type=int, default=None)

    p.add_argument("--decode", choices=["greedy", "swd"], default="swd")
    p.add_argument("--swd-window", type=int, default=1)
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--input-preproc", choices=["auto", "hf", "fast"], default="auto")
    p.add_argument("--sync-timing", action="store_true")
    p.add_argument("--confidence-threshold", type=float, default=0.65)
    p.add_argument("--alt-topk", type=int, default=3)
    p.add_argument("--max-low-conf", type=int, default=8)

    p.add_argument("--claude-model", default="claude-sonnet-4-5")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--compare-mode",
        choices=["parallel", "phoneme_only", "hiragana_only", "hybrid"],
        default="parallel",
    )
    p.add_argument(
        "--history-mode",
        choices=["session", "utterance"],
        default="session",
        help="session: keep context across turns, utterance: reset context each utterance",
    )
    p.add_argument(
        "--force-clarify-min-conf",
        type=float,
        default=0.52,
        help="Force clarification when confidence_min falls below this threshold",
    )
    p.add_argument(
        "--force-clarify-lowconf-count",
        type=int,
        default=3,
        help="Force clarification when low-confidence token count reaches this value",
    )
    p.add_argument(
        "--clarify-policy",
        choices=["auto", "never", "always"],
        default="auto",
        help="Clarification policy: auto=confidence-based, never=no asking back, always=always ask",
    )
    p.add_argument(
        "--min-kana-chars",
        type=int,
        default=2,
        help="Skip LLM call when kana length is shorter than this value",
    )
    p.add_argument(
        "--min-phoneme-tokens",
        type=int,
        default=2,
        help="Skip LLM call when phoneme token count is shorter than this value",
    )
    p.add_argument(
        "--use-training-profile",
        action="store_true",
        default=True,
        help="Inject compressed ASR profile from training report into system prompt",
    )
    p.add_argument(
        "--no-use-training-profile",
        dest="use_training_profile",
        action="store_false",
    )
    p.add_argument("--output-format", choices=["text", "jsonl"], default="text")
    p.add_argument("--profile-file", type=Path, default=None)
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    return p.parse_args()


def load_env_file(path: Path):
    """Minimal .env loader to avoid extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export "):].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        value = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def extract_message_text(message) -> str:
    parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def build_user_payload(
    channel: str,
    text: str,
    duration_sec: float,
    meta: dict,
    utterance_id: int,
    force_clarification: bool,
    clarification_reasons: list[str],
) -> str:
    low_conf = meta.get("low_confidence_tokens", [])
    payload = {
        "utterance_id": utterance_id,
        "channel": channel,
        "text": text,
        "duration_sec": round(duration_sec, 3),
        "confidence_mean": round(float(meta.get("confidence_mean", 0.0)), 4),
        "confidence_min": round(float(meta.get("confidence_min", 0.0)), 4),
        "low_confidence_tokens": low_conf[:5],
        "force_clarification": force_clarification,
        "clarification_reasons": clarification_reasons,
        "clarify_policy": "force" if force_clarification else "best_effort_no_clarify",
    }
    return json.dumps(payload, ensure_ascii=False)


def build_hybrid_payload(
    hiragana_text: str,
    phoneme_text: str,
    duration_sec: float,
    meta: dict,
    utterance_id: int,
    force_clarification: bool,
    clarification_reasons: list[str],
) -> str:
    low_conf = meta.get("low_confidence_tokens", [])
    payload = {
        "utterance_id": utterance_id,
        "channel": "hybrid",
        "hiragana": hiragana_text,
        "phonemes": phoneme_text,
        "duration_sec": round(duration_sec, 3),
        "confidence_mean": round(float(meta.get("confidence_mean", 0.0)), 4),
        "confidence_min": round(float(meta.get("confidence_min", 0.0)), 4),
        "low_confidence_tokens": low_conf[:5],
        "force_clarification": force_clarification,
        "clarification_reasons": clarification_reasons,
        "clarify_policy": "force" if force_clarification else "best_effort_no_clarify",
        "task": (
            "Use hiragana as primary text signal; use phonemes and low-confidence metadata "
            "as error-correction hints."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def ask_claude(
    client: Anthropic,
    model: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    history: list[dict],
    user_payload: str,
) -> tuple[str, list[dict]]:
    messages = history + [{"role": "user", "content": user_payload}]
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=messages,
    )
    text = extract_message_text(resp)
    new_history = messages + [{"role": "assistant", "content": text}]
    return text, new_history


def should_force_clarification(
    policy: str,
    meta: dict,
    min_conf_threshold: float,
    low_conf_count_threshold: int,
) -> tuple[bool, list[str]]:
    if policy == "never":
        return False, []
    if policy == "always":
        return True, ["policy=always"]

    reasons: list[str] = []
    conf_min = float(meta.get("confidence_min", 0.0))
    low_conf_count = len(meta.get("low_confidence_tokens", []))
    if conf_min < min_conf_threshold:
        reasons.append(f"confidence_min<{min_conf_threshold:.2f}")
    if low_conf_count >= low_conf_count_threshold:
        reasons.append(f"low_conf_count>={low_conf_count_threshold}")
    return (len(reasons) > 0), reasons


def should_skip_llm_call(
    kana_text: str | None,
    phoneme_text: str | None,
    min_kana_chars: int,
    min_phoneme_tokens: int,
) -> tuple[bool, str]:
    kana_len = len((kana_text or "").strip())
    ph_tokens = len((phoneme_text or "").strip().split()) if (phoneme_text or "").strip() else 0
    if kana_len < min_kana_chars and ph_tokens < min_phoneme_tokens:
        return True, f"too_short(kana={kana_len}, phoneme_tokens={ph_tokens})"
    return False, ""


def format_text_result(
    utterance_id: int,
    asr_result: dict,
    force_clarification: bool,
    clarification_reasons: list[str],
    phoneme_resp: str | None,
    hira_resp: str | None,
    hybrid_resp: str | None,
):
    print("\n" + "=" * 72)
    print(f"[utt={utterance_id}] phonemes: {asr_result['phonemes']}")
    if asr_result.get("kana"):
        print(f"[utt={utterance_id}] hiragana: {asr_result['kana']}")
    print(
        f"[asr] conf_mean={asr_result['phoneme_meta']['confidence_mean']:.3f} "
        f"conf_min={asr_result['phoneme_meta']['confidence_min']:.3f} "
        f"decode_ms={asr_result['decode_ms']:.1f}"
    )
    if force_clarification:
        print(f"[asr] force_clarification=True reasons={clarification_reasons}")
    if phoneme_resp is not None:
        print(f"[claude/phoneme] {phoneme_resp}")
    if hira_resp is not None:
        print(f"[claude/hiragana] {hira_resp}")
    if hybrid_resp is not None:
        print(f"[claude/hybrid] {hybrid_resp}")
    print("=" * 72)


def main():
    args = parse_args()
    load_env_file(args.env_file)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found. Put it in .env or environment.")
    client = Anthropic(api_key=api_key)

    preset = MODEL_PRESETS[args.model]
    checkpoint = args.checkpoint or ROOT / preset["checkpoint"]
    pretrained = args.pretrained or preset["pretrained"]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtype = resolve_inference_dtype(device, args.precision)
    if args.input_preproc == "auto":
        use_fast_preproc = device.type == "mps"
    else:
        use_fast_preproc = args.input_preproc == "fast"

    model = load_checkpoint(str(checkpoint), pretrained).to(device=device, dtype=dtype).eval()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(pretrained)
    kana_vocab = KanaVocab()
    phoneme_vocab = PhonemeVocab()

    with torch.inference_mode():
        dummy = torch.randn(1, SAMPLE_RATE, device=device, dtype=dtype)
        model(dummy)
    if device.type == "mps":
        torch.mps.synchronize()

    segmenter = VADSegmenter(
        threshold=args.vad_threshold,
        silence_timeout=args.silence_timeout,
        max_utterance=args.max_utterance,
        prebuffer_sec=args.prebuffer_sec,
    )

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.use_training_profile:
        system_prompt += "\n\n" + TRAINING_PROFILE_PROMPT
    if args.profile_file is not None:
        system_prompt += "\n\nAdditional profile:\n" + args.profile_file.read_text(encoding="utf-8")

    hist_phoneme: list[dict] = []
    hist_hira: list[dict] = []

    audio_queue = collections.deque()
    queue_lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        with queue_lock:
            audio_queue.append(indata[:, 0].copy())

    print(
        f"[init] device={device} dtype={dtype} decode={args.decode} "
        f"compare={args.compare_mode} history={args.history_mode} claude_model={args.claude_model}"
    )
    print("[ready] Speak! Ctrl+C to stop.")

    last_preview = 0.0
    last_preview_samples = 0
    preview_min_delta_samples = max(1, int(args.preview_min_delta * SAMPLE_RATE))
    preview_max_samples = int(args.preview_max_audio * SAMPLE_RATE) if args.preview_max_audio > 0 else 0
    utterance_id = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.05),
            device=args.device_id,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.02)

                chunks = []
                with queue_lock:
                    while audio_queue:
                        chunks.append(audio_queue.popleft())
                if not chunks:
                    continue

                audio_chunk = np.concatenate(chunks)
                utterance, is_final = segmenter.feed(audio_chunk)

                if is_final and utterance is not None:
                    duration_sec = len(utterance) / SAMPLE_RATE
                    if duration_sec < 0.3:
                        continue

                    asr_result = transcribe(
                        model=model,
                        fe=fe,
                        audio=utterance,
                        device=device,
                        dtype=dtype,
                        decode_mode=args.decode,
                        swd_window=args.swd_window,
                        use_fast_preproc=use_fast_preproc,
                        sync_timing=args.sync_timing,
                        phoneme_vocab=phoneme_vocab,
                        kana_vocab=kana_vocab,
                        emit_kana=True,
                        confidence_threshold=args.confidence_threshold,
                        alt_topk=args.alt_topk,
                        max_low_conf=args.max_low_conf,
                    )

                    phoneme_resp = None
                    hira_resp = None
                    hybrid_resp = None
                    force_clarification, clarification_reasons = should_force_clarification(
                        policy=args.clarify_policy,
                        meta=asr_result["phoneme_meta"],
                        min_conf_threshold=args.force_clarify_min_conf,
                        low_conf_count_threshold=args.force_clarify_lowconf_count,
                    )
                    skip_llm, skip_reason = should_skip_llm_call(
                        kana_text=asr_result.get("kana"),
                        phoneme_text=asr_result.get("phonemes"),
                        min_kana_chars=args.min_kana_chars,
                        min_phoneme_tokens=args.min_phoneme_tokens,
                    )
                    hist_ph = hist_phoneme if args.history_mode == "session" else []
                    hist_hi = hist_hira if args.history_mode == "session" else []

                    if not skip_llm and args.compare_mode in ("parallel", "phoneme_only"):
                        payload = build_user_payload(
                            channel="phoneme",
                            text=asr_result["phonemes"],
                            duration_sec=duration_sec,
                            meta=asr_result["phoneme_meta"],
                            utterance_id=utterance_id,
                            force_clarification=force_clarification,
                            clarification_reasons=clarification_reasons,
                        )
                        phoneme_resp, hist_ph = ask_claude(
                            client=client,
                            model=args.claude_model,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            system_prompt=system_prompt,
                            history=hist_ph,
                            user_payload=payload,
                        )
                        if args.history_mode == "session":
                            hist_phoneme = hist_ph

                    if not skip_llm and args.compare_mode in ("parallel", "hiragana_only"):
                        payload = build_user_payload(
                            channel="hiragana",
                            text=asr_result["kana"],
                            duration_sec=duration_sec,
                            meta=asr_result["phoneme_meta"],
                            utterance_id=utterance_id,
                            force_clarification=force_clarification,
                            clarification_reasons=clarification_reasons,
                        )
                        hira_resp, hist_hi = ask_claude(
                            client=client,
                            model=args.claude_model,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            system_prompt=system_prompt,
                            history=hist_hi,
                            user_payload=payload,
                        )
                        if args.history_mode == "session":
                            hist_hira = hist_hi

                    if not skip_llm and args.compare_mode == "hybrid":
                        payload = build_hybrid_payload(
                            hiragana_text=asr_result["kana"],
                            phoneme_text=asr_result["phonemes"],
                            duration_sec=duration_sec,
                            meta=asr_result["phoneme_meta"],
                            utterance_id=utterance_id,
                            force_clarification=force_clarification,
                            clarification_reasons=clarification_reasons,
                        )
                        hybrid_resp, hist_hi = ask_claude(
                            client=client,
                            model=args.claude_model,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            system_prompt=system_prompt,
                            history=hist_hi,
                            user_payload=payload,
                        )
                        if args.history_mode == "session":
                            hist_hira = hist_hi

                    if args.output_format == "jsonl":
                        print(json.dumps({
                            "type": "result",
                            "utterance_id": utterance_id,
                            "asr": asr_result,
                            "llm_skipped": skip_llm,
                            "llm_skip_reason": skip_reason,
                            "force_clarification": force_clarification,
                            "clarification_reasons": clarification_reasons,
                            "claude_phoneme": phoneme_resp,
                            "claude_hiragana": hira_resp,
                            "claude_hybrid": hybrid_resp,
                        }, ensure_ascii=False), flush=True)
                    else:
                        if skip_llm:
                            print("\n" + "=" * 72)
                            print(f"[utt={utterance_id}] phonemes: {asr_result['phonemes']}")
                            if asr_result.get("kana"):
                                print(f"[utt={utterance_id}] hiragana: {asr_result['kana']}")
                            print(f"[llm] skipped: {skip_reason}")
                            print("=" * 72)
                            utterance_id += 1
                            last_preview = 0.0
                            last_preview_samples = 0
                            continue
                        format_text_result(
                            utterance_id=utterance_id,
                            asr_result=asr_result,
                            force_clarification=force_clarification,
                            clarification_reasons=clarification_reasons,
                            phoneme_resp=phoneme_resp,
                            hira_resp=hira_resp,
                            hybrid_resp=hybrid_resp,
                        )

                    utterance_id += 1
                    last_preview = 0.0
                    last_preview_samples = 0
                    continue

                now = time.time()
                if now - last_preview < args.preview_interval:
                    continue
                current = segmenter.get_current_audio()
                if current is None or len(current) / SAMPLE_RATE <= 0.3:
                    continue
                if len(current) - last_preview_samples < preview_min_delta_samples:
                    continue

                clipped = False
                preview_audio = current
                if preview_max_samples > 0 and len(current) > preview_max_samples:
                    preview_audio = current[-preview_max_samples:]
                    clipped = True

                asr_preview = transcribe(
                    model=model,
                    fe=fe,
                    audio=preview_audio,
                    device=device,
                    dtype=dtype,
                    decode_mode=args.decode,
                    swd_window=args.swd_window,
                    use_fast_preproc=use_fast_preproc,
                    sync_timing=args.sync_timing,
                    phoneme_vocab=phoneme_vocab,
                    kana_vocab=kana_vocab,
                    emit_kana=True,
                    confidence_threshold=args.confidence_threshold,
                    alt_topk=args.alt_topk,
                    max_low_conf=args.max_low_conf,
                )
                if args.output_format == "jsonl":
                    print(json.dumps({
                        "type": "preview",
                        "utterance_id": utterance_id,
                        "duration_sec": len(current) / SAMPLE_RATE,
                        "phonemes": asr_preview["phonemes"],
                        "kana": asr_preview["kana"],
                        "confidence_mean": asr_preview["phoneme_meta"]["confidence_mean"],
                        "clipped_preview": clipped,
                    }, ensure_ascii=False), flush=True)
                else:
                    shown = f"...{asr_preview['phonemes']}" if clipped else asr_preview["phonemes"]
                    sys.stdout.write(
                        "\r\033[K"
                        f"\033[90m[preview {len(current)/SAMPLE_RATE:.1f}s] {shown}\033[0m"
                    )
                    sys.stdout.flush()

                last_preview = now
                last_preview_samples = len(current)

    except KeyboardInterrupt:
        print("\n[stop] session ended.")


if __name__ == "__main__":
    main()
