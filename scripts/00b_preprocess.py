"""Pre-process ReazonSpeech: decode audio + compute kana/phoneme labels.

Eliminates the CPU bottleneck (FLAC decode + fugashi + cutlet) from the
training loop so the GPU stays fed.

Usage:
    uv run python scripts/00b_preprocess.py \
        --input data/datasets/reazonspeech/small \
        --output data/datasets/reazonspeech/small_proc
"""

import argparse
import io
import logging
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.asr.kana_converter import JapaneseKanaConverter
from src.asr.kana_vocab import KanaVocab
from src.asr.phoneme_converter import JapanesePhonemeConverter
from src.asr.phoneme_vocab import PhonemeVocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

TARGET_SR = 16_000
MAX_DURATION_S = 15.0


def resolve_num_proc(requested: int | None) -> int:
    if requested is not None:
        return max(1, requested)
    n_cpu = os.cpu_count() or 4
    if n_cpu <= 4:
        return max(1, n_cpu - 1)
    return min(16, max(2, n_cpu - 2))


def decode_audio_to_mono_f32(audio_item) -> tuple[np.ndarray | None, int | None]:
    """Decode audio payload to mono float32 waveform.

    Returns:
        (waveform, sr) on success, (None, None) on decode failure.
    """
    import soundfile as sf

    if not isinstance(audio_item, dict):
        arr = np.asarray(audio_item, dtype=np.float32)
        if arr.size == 0:
            return None, None
        return arr, TARGET_SR

    array = audio_item.get("array")
    if array is not None:
        arr = np.asarray(array, dtype=np.float32)
        if arr.size == 0:
            return None, None
        sr = int(audio_item.get("sampling_rate") or TARGET_SR)
        return arr, sr

    audio_bytes = audio_item.get("bytes")
    if audio_bytes is not None:
        try:
            arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.size > 0:
                return arr, int(sr)
        except Exception:
            pass

    audio_path = audio_item.get("path")
    if audio_path:
        try:
            arr, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.size > 0:
                return arr, int(sr)
        except Exception:
            pass

    return None, None


def to_mono_and_resample(waveform: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if waveform.ndim > 1:
        # soundfile is usually (frames, channels)
        if waveform.shape[0] >= waveform.shape[1]:
            waveform = waveform.mean(axis=1)
        else:
            waveform = waveform.mean(axis=0)

    waveform = np.asarray(waveform, dtype=np.float32)
    if sr != target_sr and sr > 0:
        wav_t = torch.from_numpy(waveform)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        waveform = wav_t.numpy().astype(np.float32, copy=False)
    return waveform


def is_valid_row(row: dict) -> bool:
    return bool(row["is_valid"])


def main():
    p = argparse.ArgumentParser(description="Pre-process dataset: decode audio + compute labels")
    p.add_argument("--input", type=Path, required=True, help="Path to raw HF dataset on disk")
    p.add_argument("--output", type=Path, required=True, help="Path to save preprocessed dataset")
    p.add_argument("--max-duration", type=float, default=MAX_DURATION_S)
    p.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of map workers (default: auto)",
    )
    p.add_argument(
        "--writer-batch-size",
        type=int,
        default=1000,
        help="HF datasets writer batch size",
    )
    args = p.parse_args()
    num_proc = resolve_num_proc(args.num_proc)

    from datasets import Audio, load_from_disk

    log.info(f"Loading dataset from {args.input} ...")
    ds = load_from_disk(str(args.input))
    log.info(f"Loaded {len(ds)} samples")

    # Keep decode=False so broken FLAC rows can be skipped in user code.
    ds = ds.cast_column("audio", Audio(sampling_rate=None, decode=False))

    kana_converter = JapaneseKanaConverter()
    phoneme_converter = JapanesePhonemeConverter()
    kana_vocab = KanaVocab()
    phoneme_vocab = PhonemeVocab()

    max_samples = int(args.max_duration * TARGET_SR)

    def preprocess(sample):
        audio = sample["audio"]
        decoded, sr = decode_audio_to_mono_f32(audio)
        if decoded is None or sr is None:
            return {
                "waveform": np.zeros(1, dtype=np.float32),
                "waveform_len": 0,
                "kana_labels": [],
                "phoneme_labels": [],
                "kana_str": "",
                "phoneme_str": "",
                "is_valid": False,
                "skip_reason": "decode_error",
            }

        waveform = to_mono_and_resample(decoded, sr=sr, target_sr=TARGET_SR)
        if waveform.size == 0:
            return {
                "waveform": np.zeros(1, dtype=np.float32),
                "waveform_len": 0,
                "kana_labels": [],
                "phoneme_labels": [],
                "kana_str": "",
                "phoneme_str": "",
                "is_valid": False,
                "skip_reason": "empty_waveform",
            }

        # Truncate to max duration
        waveform = waveform[:max_samples]

        text = str(sample.get("transcription", ""))
        try:
            kana_str = kana_converter.text_to_kana(text)
            phoneme_str = phoneme_converter.text_to_phonemes(text)
        except Exception:
            return {
                "waveform": np.zeros(1, dtype=np.float32),
                "waveform_len": 0,
                "kana_labels": [],
                "phoneme_labels": [],
                "kana_str": "",
                "phoneme_str": "",
                "is_valid": False,
                "skip_reason": "g2p_error",
            }
        kana_labels = kana_vocab.encode(kana_str)
        phoneme_labels = phoneme_vocab.encode(phoneme_str)

        if len(kana_labels) == 0 or len(phoneme_labels) == 0:
            return {
                "waveform": np.zeros(1, dtype=np.float32),
                "waveform_len": 0,
                "kana_labels": [],
                "phoneme_labels": [],
                "kana_str": "",
                "phoneme_str": "",
                "is_valid": False,
                "skip_reason": "empty_label",
            }

        return {
            "waveform": waveform,
            "waveform_len": int(waveform.shape[0]),
            "kana_labels": kana_labels,
            "phoneme_labels": phoneme_labels,
            "kana_str": kana_str,
            "phoneme_str": phoneme_str,
            "is_valid": True,
            "skip_reason": "",
        }

    log.info("Pre-processing samples (decode audio + compute labels) ...")
    log.info(f"num_proc={num_proc}, writer_batch_size={args.writer_batch_size}")
    t0 = time.time()
    ds_proc = ds.map(
        preprocess,
        remove_columns=["audio", "name", "transcription"],
        desc="Preprocessing",
        num_proc=num_proc,
        writer_batch_size=max(args.writer_batch_size, 1),
    )
    reason_counts = Counter(ds_proc["skip_reason"])
    invalid_count = len(ds_proc) - int(sum(ds_proc["is_valid"]))
    if invalid_count > 0:
        ds_proc = ds_proc.filter(
            is_valid_row,
            num_proc=num_proc,
            desc="Filtering invalid rows",
        )
    ds_proc = ds_proc.remove_columns(["is_valid", "skip_reason"])
    elapsed = time.time() - t0
    log.info(
        "Preprocessing done in %.1fs (valid=%d, skipped=%d, reasons=%s)",
        elapsed,
        len(ds_proc),
        invalid_count,
        dict(reason_counts),
    )

    tmp_output = args.output.parent / f"{args.output.name}.tmp"
    if tmp_output.exists():
        shutil.rmtree(tmp_output)
    log.info(f"Saving to temp path {tmp_output} ...")
    tmp_output.parent.mkdir(parents=True, exist_ok=True)
    ds_proc.save_to_disk(str(tmp_output))
    if args.output.exists():
        shutil.rmtree(args.output)
    tmp_output.rename(args.output)
    log.info(f"Saved to {args.output}")
    log.info("Done.")


if __name__ == "__main__":
    main()
