"""Real-time ASR from microphone with VAD-based segmentation.

Architecture:
    Mic → pre-buffer → VAD (Silero) → utterance segmentation → ASR → display

Each utterance is detected by VAD with a lookback pre-buffer to capture
speech onset. Transcribed independently and displayed as finalized lines.
Live preview updates while speaking.

Usage:
    uv run python scripts/realtime_asr.py                          # default (medium)
    uv run python scripts/realtime_asr.py --model large             # large (small dataset)
    uv run python scripts/realtime_asr.py --show-phonemes
    uv run python scripts/realtime_asr.py --silence-timeout 0.5

Press Ctrl+C to stop.
"""

import argparse
import collections
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


def swd_decode(logits: torch.Tensor, window: int = 1) -> torch.Tensor:
    """Spike Window Decoding (SWD) for CTC logits."""
    probs = logits.squeeze(0).softmax(dim=-1)  # (T, V)
    blank_prob = probs[:, 0]
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
    p = argparse.ArgumentParser(description="Real-time ASR with VAD")
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), default="medium",
                   help="Model preset (default: medium)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Override checkpoint path")
    p.add_argument("--pretrained", default=None,
                   help="Override pretrained model name")
    p.add_argument("--preview-interval", type=float, default=0.3,
                   help="Live preview update interval (seconds)")
    p.add_argument("--preview-min-delta", type=float, default=0.4,
                   help="Minimum additional audio before next preview decode (seconds)")
    p.add_argument("--preview-max-audio", type=float, default=6.0,
                   help="Max seconds used for preview decode (0 = full utterance)")
    p.add_argument("--silence-timeout", type=float, default=0.8,
                   help="Silence duration to finalize utterance (seconds)")
    p.add_argument("--max-utterance", type=float, default=15.0,
                   help="Max utterance length before forced split (seconds)")
    p.add_argument("--vad-threshold", type=float, default=0.4,
                   help="VAD speech probability threshold")
    p.add_argument("--prebuffer-sec", type=float, default=0.5,
                   help="Pre-buffer to capture speech onset (seconds)")
    p.add_argument("--decode", choices=["greedy", "swd"], default="swd",
                   help="CTC decoding strategy (default: swd)")
    p.add_argument("--swd-window", type=int, default=1,
                   help="SWD window size around spikes")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                   help="Inference precision (default: auto)")
    p.add_argument("--input-preproc", choices=["auto", "hf", "fast"], default="auto",
                   help="Input preprocessing backend (default: auto)")
    p.add_argument("--sync-timing", action="store_true",
                   help="Synchronize accelerator before timing (more accurate, slower)")
    p.add_argument("--show-phonemes", action="store_true")
    p.add_argument("--device-id", type=int, default=None,
                   help="Audio input device ID")
    return p.parse_args()


class VADSegmenter:
    """Silero VAD-based utterance segmenter with pre-buffer.

    Keeps a rolling pre-buffer of recent audio. When speech onset is detected,
    the pre-buffer is prepended so the beginning of speech is not lost.
    """

    def __init__(self, threshold=0.4, silence_timeout=0.8,
                 max_utterance=15.0, prebuffer_sec=0.5):
        self.vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True,
        )
        self.threshold = threshold
        self.silence_samples = int(silence_timeout * SAMPLE_RATE)
        self.max_samples = int(max_utterance * SAMPLE_RATE)
        self.prebuffer_maxlen = int(prebuffer_sec * SAMPLE_RATE)

        # VAD requires 512-sample (32ms) chunks at 16kHz
        self.vad_chunk = 512

        self._utterance_chunks = []   # current utterance audio chunks
        self._total_samples = 0
        self._is_speaking = False
        self._silence_count = 0       # consecutive silence samples
        self._vad_leftover = np.array([], dtype=np.float32)

        # Rolling pre-buffer: keeps last N samples of non-speech audio
        self._prebuffer = collections.deque()
        self._prebuffer_len = 0

    def reset(self):
        self._utterance_chunks.clear()
        self._total_samples = 0
        self._is_speaking = False
        self._silence_count = 0
        self._vad_leftover = np.array([], dtype=np.float32)
        self._prebuffer.clear()
        self._prebuffer_len = 0
        self.vad_model.reset_states()

    def _push_prebuffer(self, chunk: np.ndarray):
        """Add chunk to rolling pre-buffer, evict old data if needed."""
        self._prebuffer.append(chunk)
        self._prebuffer_len += len(chunk)
        while self._prebuffer_len > self.prebuffer_maxlen:
            old = self._prebuffer.popleft()
            self._prebuffer_len -= len(old)

    def _drain_prebuffer(self) -> np.ndarray | None:
        """Get and clear the pre-buffer contents."""
        if not self._prebuffer:
            return None
        data = np.concatenate(list(self._prebuffer))
        self._prebuffer.clear()
        self._prebuffer_len = 0
        return data

    def feed(self, audio_chunk: np.ndarray):
        """Feed audio. Returns (utterance_audio, True) when utterance ends,
        or (None, False) otherwise."""
        data = np.concatenate([self._vad_leftover, audio_chunk])
        self._vad_leftover = np.array([], dtype=np.float32)

        pos = 0
        while pos + self.vad_chunk <= len(data):
            chunk = data[pos:pos + self.vad_chunk]
            prob = self.vad_model(
                torch.from_numpy(chunk).float(), SAMPLE_RATE,
            ).item()

            is_speech = prob >= self.threshold

            if is_speech:
                if not self._is_speaking:
                    # Speech onset! Prepend pre-buffer
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
                        # Utterance complete
                        utterance = np.concatenate(self._utterance_chunks)
                        self._utterance_chunks.clear()
                        self._total_samples = 0
                        self._is_speaking = False
                        self._silence_count = 0
                        self.vad_model.reset_states()
                        self._vad_leftover = data[pos + self.vad_chunk:]
                        return utterance, True
                else:
                    # Not speaking - accumulate in pre-buffer
                    self._push_prebuffer(chunk)

            # Force split if too long
            if self._total_samples >= self.max_samples:
                utterance = np.concatenate(self._utterance_chunks)
                self._utterance_chunks.clear()
                self._total_samples = 0
                self._silence_count = 0
                self._vad_leftover = data[pos + self.vad_chunk:]
                return utterance, True

            pos += self.vad_chunk

        # Save leftover
        if pos < len(data):
            self._vad_leftover = data[pos:]

        return None, False

    def get_current_audio(self) -> np.ndarray | None:
        """Get current in-progress utterance audio for live preview."""
        if not self._utterance_chunks or not self._is_speaking:
            return None
        return np.concatenate(self._utterance_chunks)


def transcribe(
    model,
    fe,
    audio,
    device,
    dtype,
    kana_vocab,
    decode_mode="swd",
    swd_window=1,
    use_fast_preproc=False,
    sync_timing=False,
    phoneme_vocab=None,
):
    """Run inference on audio array."""
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
        kana_logits = outputs["kana_logits"]
        if decode_mode == "swd":
            kana_ids = swd_decode(kana_logits, window=swd_window)
        else:
            kana_ids = kana_logits.squeeze(0).argmax(dim=-1)
    if sync_timing:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
    dt_ms = (time.perf_counter() - t0) * 1000

    kana = kana_vocab.decode(kana_ids.tolist())

    phonemes = None
    if phoneme_vocab:
        ph_ids = outputs["phoneme_logits"].squeeze(0).argmax(dim=-1)
        phonemes = phoneme_vocab.decode(ph_ids.tolist())

    return kana, phonemes, dt_ms


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

    print(
        f"[init] model={args.model}, device={device}, dtype={dtype}, decode={args.decode}, "
        f"preproc={'fast' if use_fast_preproc else 'hf'}"
    )
    print(f"[init] checkpoint={checkpoint}")
    model = load_checkpoint(str(checkpoint), pretrained)
    model.to(device=device, dtype=dtype).eval()

    fe = Wav2Vec2FeatureExtractor.from_pretrained(pretrained)
    kana_vocab = KanaVocab()
    phoneme_vocab = PhonemeVocab()

    # Warmup
    print("[init] Warming up...")
    dummy = torch.randn(1, SAMPLE_RATE, device=device, dtype=dtype)
    with torch.inference_mode():
        model(dummy)
    if device.type == "mps":
        torch.mps.synchronize()

    print("[init] Loading VAD...")
    segmenter = VADSegmenter(
        threshold=args.vad_threshold,
        silence_timeout=args.silence_timeout,
        max_utterance=args.max_utterance,
        prebuffer_sec=args.prebuffer_sec,
    )

    # Shared audio queue
    audio_queue = collections.deque()
    queue_lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        with queue_lock:
            audio_queue.append(indata[:, 0].copy())

    print(f"[config] vad={args.vad_threshold}, silence={args.silence_timeout}s, "
          f"prebuffer={args.prebuffer_sec}s, preview={args.preview_interval}s, "
          f"preview_delta={args.preview_min_delta}s, preview_max={args.preview_max_audio}s, "
          f"sync_timing={args.sync_timing}")
    print("[ready] Speak! Ctrl+C to stop.\n")

    last_preview = 0
    last_preview_samples = 0
    preview_min_delta_samples = max(1, int(args.preview_min_delta * SAMPLE_RATE))
    preview_max_samples = (
        int(args.preview_max_audio * SAMPLE_RATE) if args.preview_max_audio > 0 else 0
    )
    finalized = []

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.05),  # 50ms blocks
            device=args.device_id,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.02)

                # Drain queue
                chunks = []
                with queue_lock:
                    while audio_queue:
                        chunks.append(audio_queue.popleft())
                if not chunks:
                    continue

                audio_chunk = np.concatenate(chunks)
                utterance, is_final = segmenter.feed(audio_chunk)

                if is_final and utterance is not None:
                    dur = len(utterance) / SAMPLE_RATE
                    if dur < 0.3:
                        continue

                    ph_vocab = phoneme_vocab if args.show_phonemes else None
                    kana, phonemes, dt_ms = transcribe(
                        model, fe, utterance, device, dtype, kana_vocab,
                        decode_mode=args.decode, swd_window=args.swd_window,
                        use_fast_preproc=use_fast_preproc, sync_timing=args.sync_timing,
                        phoneme_vocab=ph_vocab,
                    )
                    rtf = dt_ms / (dur * 1000)

                    sys.stdout.write("\r\033[K")
                    print(f"[{dur:.1f}s {dt_ms:.0f}ms RTF={rtf:.2f}] {kana}")
                    if phonemes:
                        print(f"  ph: {phonemes}")
                    finalized.append(kana)
                    last_preview = 0
                    last_preview_samples = 0

                else:
                    # Live preview
                    now = time.time()
                    if now - last_preview >= args.preview_interval:
                        current = segmenter.get_current_audio()
                        if current is not None and len(current) / SAMPLE_RATE > 0.3:
                            if len(current) - last_preview_samples < preview_min_delta_samples:
                                continue
                            preview_audio = current
                            clipped = False
                            if preview_max_samples > 0 and len(current) > preview_max_samples:
                                preview_audio = current[-preview_max_samples:]
                                clipped = True
                            kana, _, dt_ms = transcribe(
                                model, fe, preview_audio, device, dtype, kana_vocab,
                                decode_mode=args.decode, swd_window=args.swd_window,
                                use_fast_preproc=use_fast_preproc, sync_timing=args.sync_timing,
                            )
                            dur = len(current) / SAMPLE_RATE
                            shown = f"...{kana}" if clipped else kana
                            sys.stdout.write(
                                f"\r\033[K\033[90m[{dur:.1f}s {dt_ms:.0f}ms] {shown}\033[0m"
                            )
                            sys.stdout.flush()
                            last_preview = now
                            last_preview_samples = len(current)

    except KeyboardInterrupt:
        current = segmenter.get_current_audio()
        if current is not None and len(current) / SAMPLE_RATE > 0.3:
            kana, _, _ = transcribe(
                model, fe, current, device, dtype, kana_vocab,
                decode_mode=args.decode, swd_window=args.swd_window,
                use_fast_preproc=use_fast_preproc, sync_timing=args.sync_timing,
            )
            finalized.append(kana)

        print(f"\n\n{'='*50}")
        print("Transcription:")
        for line in finalized:
            print(f"  {line}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
