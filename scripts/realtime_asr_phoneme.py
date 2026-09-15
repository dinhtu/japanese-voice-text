"""Real-time phoneme streaming ASR for LLM pipelines.

Architecture:
    Mic -> VAD segmentation -> ASR -> JSONL events

Primary output is phoneme sequence with confidence metadata.
Optional kana output can be added for debugging.

Usage:
    uv run python scripts/realtime_asr_phoneme.py
    uv run python scripts/realtime_asr_phoneme.py --emit-kana
    uv run python scripts/realtime_asr_phoneme.py --decode swd --swd-window 1
"""

import argparse
import collections
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from transformers import Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asr.kana_vocab import KanaVocab
from src.asr.model import load_checkpoint
from src.asr.phoneme_vocab import PhonemeVocab

SAMPLE_RATE = 16000

MODEL_PRESETS = {
    "medium": {
        "checkpoint": "models/checkpoints/best-medium-ep5-inference.pt",
        "pretrained": "reazon-research/japanese-wav2vec2-large",
    },
    "large": {
        "checkpoint": "models/checkpoints/best_large.pt",
        "pretrained": "reazon-research/japanese-wav2vec2-large",
    },
}

BLANK_IDX = 0


def swd_decode(logits: torch.Tensor, window: int = 1) -> torch.Tensor:
    """Spike Window Decoding (SWD) for CTC logits."""
    probs = logits.squeeze(0).softmax(dim=-1)  # (T, V)
    blank_prob = probs[:, BLANK_IDX]
    is_spike = blank_prob < 0.5

    if not is_spike.any():
        return logits.squeeze(0).argmax(dim=-1)

    t = probs.shape[0]
    spike_indices = is_spike.nonzero(as_tuple=True)[0]
    active = torch.zeros(t, dtype=torch.bool, device=logits.device)
    for idx in spike_indices:
        start = max(0, idx.item() - window)
        end = min(t, idx.item() + window + 1)
        active[start:end] = True

    pred_ids = torch.zeros(t, dtype=torch.long, device=logits.device)
    pred_ids[active] = logits.squeeze(0)[active].argmax(dim=-1)
    return pred_ids


def resolve_inference_dtype(device: torch.device, precision: str) -> torch.dtype:
    """Resolve safe inference dtype for the selected device/precision."""
    if precision == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32

    requested = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]

    if device.type == "cpu" and requested != torch.float32:
        print(f"[warn] precision={precision} is not practical on CPU. Falling back to fp32.")
        return torch.float32
    if device.type == "mps" and requested == torch.bfloat16:
        print("[warn] bf16 is unsupported on MPS. Falling back to fp16.")
        return torch.float16
    if device.type == "cuda" and requested == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[warn] bf16 is unsupported on this CUDA device. Falling back to fp16.")
        return torch.float16
    return requested


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Wav2Vec2-style utterance-level normalization."""
    mean = waveform.mean(dim=-1, keepdim=True)
    var = waveform.var(dim=-1, keepdim=True, unbiased=False)
    return (waveform - mean) / torch.sqrt(var + 1e-5)


def build_input_values(
    audio: np.ndarray,
    fe: Wav2Vec2FeatureExtractor,
    device: torch.device,
    dtype: torch.dtype,
    use_fast_preproc: bool,
) -> torch.Tensor:
    """Prepare model input tensor with optional fast path for MPS."""
    if use_fast_preproc:
        iv = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        if fe.do_normalize:
            iv = normalize_waveform(iv)
    else:
        inputs = fe(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=False,
        )
        iv = inputs.input_values

    return iv.to(device=device, dtype=dtype, non_blocking=(device.type != "cpu"))


def parse_args():
    p = argparse.ArgumentParser(description="Real-time phoneme streaming ASR")
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

    p.add_argument("--decode", choices=["greedy", "swd"], default="swd")
    p.add_argument("--swd-window", type=int, default=1)
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--input-preproc", choices=["auto", "hf", "fast"], default="auto")
    p.add_argument("--sync-timing", action="store_true")
    p.add_argument("--device-id", type=int, default=None)

    p.add_argument("--emit-kana", action="store_true")
    p.add_argument("--confidence-threshold", type=float, default=0.65)
    p.add_argument("--alt-topk", type=int, default=3)
    p.add_argument("--max-low-conf", type=int, default=8)
    return p.parse_args()


class VADSegmenter:
    """Silero VAD-based utterance segmenter with pre-buffer."""

    def __init__(self, threshold=0.4, silence_timeout=0.8, max_utterance=15.0, prebuffer_sec=0.5):
        self.vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True,
        )
        self.threshold = threshold
        self.silence_samples = int(silence_timeout * SAMPLE_RATE)
        self.max_samples = int(max_utterance * SAMPLE_RATE)
        self.prebuffer_maxlen = int(prebuffer_sec * SAMPLE_RATE)
        self.vad_chunk = 512

        self._utterance_chunks = []
        self._total_samples = 0
        self._is_speaking = False
        self._silence_count = 0
        self._vad_leftover = np.array([], dtype=np.float32)

        self._prebuffer = collections.deque()
        self._prebuffer_len = 0

    def _push_prebuffer(self, chunk: np.ndarray):
        self._prebuffer.append(chunk)
        self._prebuffer_len += len(chunk)
        while self._prebuffer_len > self.prebuffer_maxlen:
            old = self._prebuffer.popleft()
            self._prebuffer_len -= len(old)

    def _drain_prebuffer(self) -> np.ndarray | None:
        if not self._prebuffer:
            return None
        data = np.concatenate(list(self._prebuffer))
        self._prebuffer.clear()
        self._prebuffer_len = 0
        return data

    def feed(self, audio_chunk: np.ndarray):
        data = np.concatenate([self._vad_leftover, audio_chunk])
        self._vad_leftover = np.array([], dtype=np.float32)

        pos = 0
        while pos + self.vad_chunk <= len(data):
            chunk = data[pos:pos + self.vad_chunk]
            prob = self.vad_model(torch.from_numpy(chunk).float(), SAMPLE_RATE).item()
            is_speech = prob >= self.threshold

            if is_speech:
                if not self._is_speaking:
                    self._is_speaking = True
                    pre = self._drain_prebuffer()
                    if pre is not None:
                        self._utterance_chunks.append(pre)
                        self._total_samples += len(pre)

                self._silence_count = 0
                self._utterance_chunks.append(chunk)
                self._total_samples += len(chunk)
            else:
                if self._is_speaking:
                    self._silence_count += len(chunk)
                    self._utterance_chunks.append(chunk)
                    self._total_samples += len(chunk)
                    if self._silence_count >= self.silence_samples:
                        utterance = np.concatenate(self._utterance_chunks)
                        self._utterance_chunks.clear()
                        self._total_samples = 0
                        self._is_speaking = False
                        self._silence_count = 0
                        self.vad_model.reset_states()
                        self._vad_leftover = data[pos + self.vad_chunk:]
                        return utterance, True
                else:
                    self._push_prebuffer(chunk)

            if self._total_samples >= self.max_samples:
                utterance = np.concatenate(self._utterance_chunks)
                self._utterance_chunks.clear()
                self._total_samples = 0
                self._silence_count = 0
                self._vad_leftover = data[pos + self.vad_chunk:]
                return utterance, True

            pos += self.vad_chunk

        if pos < len(data):
            self._vad_leftover = data[pos:]

        return None, False

    def get_current_audio(self) -> np.ndarray | None:
        if not self._utterance_chunks or not self._is_speaking:
            return None
        return np.concatenate(self._utterance_chunks)


def ctc_token_spans(
    logits: torch.Tensor, decode_mode: str, swd_window: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Return per-token spans and confidence from frame-level CTC outputs."""
    frame_logits = logits.squeeze(0)  # (T, V)
    frame_probs = frame_logits.softmax(dim=-1)

    if decode_mode == "swd":
        pred_ids = swd_decode(logits, window=swd_window)
    else:
        pred_ids = frame_logits.argmax(dim=-1)

    pred_probs = frame_probs.gather(1, pred_ids.unsqueeze(-1)).squeeze(-1)

    spans = []
    t = pred_ids.shape[0]
    i = 0
    while i < t:
        token_id = int(pred_ids[i].item())
        if token_id == BLANK_IDX:
            i += 1
            continue
        j = i + 1
        while j < t and int(pred_ids[j].item()) == token_id:
            j += 1
        conf = float(pred_probs[i:j].mean().item())
        spans.append({
            "token_id": token_id,
            "start_frame": i,
            "end_frame": j - 1,
            "confidence": conf,
        })
        i = j

    return pred_ids, frame_probs, spans


def span_alternatives(
    span: dict, frame_probs: torch.Tensor, vocab, topk: int,
) -> list[dict]:
    """Top-k alternatives for one token span based on averaged frame probs."""
    start = int(span["start_frame"])
    end = int(span["end_frame"]) + 1
    span_mean = frame_probs[start:end].mean(dim=0)

    k = min(max(1, topk + 1), span_mean.shape[0])
    probs, ids = torch.topk(span_mean, k=k)
    out = []
    current_id = int(span["token_id"])
    for prob, idx in zip(probs.tolist(), ids.tolist()):
        token_id = int(idx)
        if token_id == BLANK_IDX or token_id == current_id:
            continue
        token = vocab.itos.get(token_id)
        if not token:
            continue
        out.append({"token": token, "prob": float(prob)})
        if len(out) >= topk:
            break
    return out


def phoneme_payload(
    logits: torch.Tensor,
    phoneme_vocab: PhonemeVocab,
    decode_mode: str,
    swd_window: int,
    confidence_threshold: float,
    alt_topk: int,
    max_low_conf: int,
) -> tuple[str, dict]:
    """Build phoneme string + confidence metadata."""
    _, frame_probs, spans = ctc_token_spans(
        logits=logits,
        decode_mode=decode_mode,
        swd_window=swd_window,
    )

    tokens = []
    token_infos = []
    for i, span in enumerate(spans):
        token_id = int(span["token_id"])
        token = phoneme_vocab.itos.get(token_id)
        if token is None or token == "<blank>":
            continue
        tokens.append(token)
        token_infos.append({
            "index": len(token_infos),
            "token": token,
            "confidence": float(span["confidence"]),
            "start_frame": int(span["start_frame"]),
            "end_frame": int(span["end_frame"]),
            "token_id": token_id,
            "_span": span,
        })

    phoneme_text = " ".join(tokens)
    confs = [x["confidence"] for x in token_infos]
    if confs:
        mean_conf = float(np.mean(confs))
        min_conf = float(np.min(confs))
    else:
        mean_conf = 0.0
        min_conf = 0.0

    low_conf = [x for x in token_infos if x["confidence"] < confidence_threshold]
    low_conf.sort(key=lambda x: x["confidence"])
    low_conf = low_conf[:max_low_conf]

    low_conf_items = []
    for item in low_conf:
        low_conf_items.append({
            "index": item["index"],
            "token": item["token"],
            "confidence": item["confidence"],
            "start_frame": item["start_frame"],
            "end_frame": item["end_frame"],
            "alternatives": span_alternatives(
                item["_span"],
                frame_probs=frame_probs,
                vocab=phoneme_vocab,
                topk=alt_topk,
            ),
        })

    for item in token_infos:
        item.pop("token_id", None)
        item.pop("_span", None)

    metadata = {
        "token_count": len(token_infos),
        "confidence_mean": mean_conf,
        "confidence_min": min_conf,
        "tokens": token_infos,
        "low_confidence_tokens": low_conf_items,
    }
    return phoneme_text, metadata


def transcribe(
    model,
    fe,
    audio: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    decode_mode: str,
    swd_window: int,
    use_fast_preproc: bool,
    sync_timing: bool,
    phoneme_vocab: PhonemeVocab,
    kana_vocab: KanaVocab,
    emit_kana: bool,
    confidence_threshold: float,
    alt_topk: int,
    max_low_conf: int,
) -> dict:
    """Run inference and return structured output for JSON events."""
    iv = build_input_values(
        audio=audio,
        fe=fe,
        device=device,
        dtype=dtype,
        use_fast_preproc=use_fast_preproc,
    )

    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(iv)
    if sync_timing:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
    dt_ms = (time.perf_counter() - t0) * 1000

    phoneme_text, phoneme_meta = phoneme_payload(
        logits=outputs["phoneme_logits"],
        phoneme_vocab=phoneme_vocab,
        decode_mode=decode_mode,
        swd_window=swd_window,
        confidence_threshold=confidence_threshold,
        alt_topk=alt_topk,
        max_low_conf=max_low_conf,
    )

    kana_text = None
    if emit_kana:
        kana_logits = outputs["kana_logits"]
        if decode_mode == "swd":
            kana_ids = swd_decode(kana_logits, window=swd_window)
        else:
            kana_ids = kana_logits.squeeze(0).argmax(dim=-1)
        kana_text = kana_vocab.decode(kana_ids.tolist())

    return {
        "phonemes": phoneme_text,
        "phoneme_meta": phoneme_meta,
        "kana": kana_text,
        "decode_ms": float(dt_ms),
    }


def emit_event(payload: dict):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    args = parse_args()
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

    audio_queue = collections.deque()
    queue_lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        with queue_lock:
            audio_queue.append(indata[:, 0].copy())

    emit_event({
        "type": "session_start",
        "ts": time.time(),
        "sample_rate": SAMPLE_RATE,
        "device": device.type,
        "dtype": str(dtype),
        "decode": args.decode,
        "swd_window": args.swd_window,
        "input_preproc": "fast" if use_fast_preproc else "hf",
    })

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

                    result = transcribe(
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
                        emit_kana=args.emit_kana,
                        confidence_threshold=args.confidence_threshold,
                        alt_topk=args.alt_topk,
                        max_low_conf=args.max_low_conf,
                    )
                    rtf = result["decode_ms"] / (duration_sec * 1000)
                    payload = {
                        "type": "final",
                        "ts": time.time(),
                        "utterance_id": utterance_id,
                        "duration_sec": duration_sec,
                        "decode_ms": result["decode_ms"],
                        "rtf": rtf,
                        "phonemes": result["phonemes"],
                        "phoneme_meta": result["phoneme_meta"],
                    }
                    if result["kana"] is not None:
                        payload["kana"] = result["kana"]
                    emit_event(payload)

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

                result = transcribe(
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
                    emit_kana=args.emit_kana,
                    confidence_threshold=args.confidence_threshold,
                    alt_topk=args.alt_topk,
                    max_low_conf=args.max_low_conf,
                )
                payload = {
                    "type": "preview",
                    "ts": time.time(),
                    "utterance_id": utterance_id,
                    "duration_sec": len(current) / SAMPLE_RATE,
                    "decode_ms": result["decode_ms"],
                    "phonemes": result["phonemes"],
                    "phoneme_meta": {
                        "token_count": result["phoneme_meta"]["token_count"],
                        "confidence_mean": result["phoneme_meta"]["confidence_mean"],
                        "confidence_min": result["phoneme_meta"]["confidence_min"],
                    },
                    "clipped_preview": clipped,
                }
                if result["kana"] is not None:
                    payload["kana"] = result["kana"]
                emit_event(payload)

                last_preview = now
                last_preview_samples = len(current)

    except KeyboardInterrupt:
        current = segmenter.get_current_audio()
        if current is not None and len(current) / SAMPLE_RATE > 0.3:
            result = transcribe(
                model=model,
                fe=fe,
                audio=current,
                device=device,
                dtype=dtype,
                decode_mode=args.decode,
                swd_window=args.swd_window,
                use_fast_preproc=use_fast_preproc,
                sync_timing=args.sync_timing,
                phoneme_vocab=phoneme_vocab,
                kana_vocab=kana_vocab,
                emit_kana=args.emit_kana,
                confidence_threshold=args.confidence_threshold,
                alt_topk=args.alt_topk,
                max_low_conf=args.max_low_conf,
            )
            payload = {
                "type": "final_interrupted",
                "ts": time.time(),
                "utterance_id": utterance_id,
                "duration_sec": len(current) / SAMPLE_RATE,
                "decode_ms": result["decode_ms"],
                "phonemes": result["phonemes"],
                "phoneme_meta": result["phoneme_meta"],
            }
            if result["kana"] is not None:
                payload["kana"] = result["kana"]
            emit_event(payload)

        emit_event({"type": "session_end", "ts": time.time()})


if __name__ == "__main__":
    main()
