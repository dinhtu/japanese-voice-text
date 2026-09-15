"""Export transcription-to-kana/phoneme alignment samples as CSV.

Usage:
    uv run python scripts/04_export_alignment_csv.py \
        --data-split small \
        --max-samples 1000 \
        --out data/alignment_samples_small_1000.csv
"""

# ruff: noqa: E402

import argparse
import csv
import io
import logging
import random
import sys
from pathlib import Path

import torchaudio

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.asr.dataset import load_prepared_reazonspeech
from src.asr.kana_converter import JapaneseKanaConverter
from src.asr.phoneme_converter import JapanesePhonemeConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Export alignment CSV from prepared ReazonSpeech")
    p.add_argument("--data-split", default="small", choices=["tiny", "small", "medium"])
    p.add_argument("--dataset-dir", type=Path, default=Path("data/datasets/reazonspeech"))
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def infer_duration_seconds(audio_item) -> float:
    if not isinstance(audio_item, dict):
        return 0.0

    array = audio_item.get("array")
    sampling_rate = audio_item.get("sampling_rate")
    if array is not None and sampling_rate not in (None, 0):
        return float(len(array) / sampling_rate)

    audio_bytes = audio_item.get("bytes")
    if audio_bytes:
        try:
            info = torchaudio.info(io.BytesIO(audio_bytes))
            if info.sample_rate > 0 and info.num_frames > 0:
                return float(info.num_frames / info.sample_rate)
        except Exception:
            return 0.0

    audio_path = audio_item.get("path")
    if audio_path:
        try:
            info = torchaudio.info(audio_path)
            if info.sample_rate > 0 and info.num_frames > 0:
                return float(info.num_frames / info.sample_rate)
        except Exception:
            return 0.0

    return 0.0


def main():
    args = parse_args()

    log.info(f"Loading prepared ReazonSpeech split={args.data_split} from {args.dataset_dir}")
    ds = load_prepared_reazonspeech(args.data_split, args.dataset_dir)
    total_rows = len(ds)

    sample_count = min(max(args.max_samples, 0), total_rows)
    if sample_count == 0:
        raise ValueError("No samples to export. Set --max-samples > 0 and check dataset.")

    rng = random.Random(args.seed)
    indices = rng.sample(range(total_rows), sample_count)

    kana_converter = JapaneseKanaConverter()
    phoneme_converter = JapanesePhonemeConverter()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing {sample_count} rows to {args.out}")

    fieldnames = [
        "index",
        "name",
        "transcription",
        "kana",
        "phoneme",
        "duration_sec",
        "kana_len",
        "phoneme_len",
    ]
    n_empty_kana = 0
    n_empty_phoneme = 0

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_num, idx in enumerate(indices, start=1):
            item = ds[idx]
            transcription = str(item.get("transcription", ""))
            kana = kana_converter.text_to_kana(transcription)
            phoneme = phoneme_converter.text_to_phonemes(transcription)
            duration_sec = infer_duration_seconds(item.get("audio"))

            if not kana:
                n_empty_kana += 1
            if not phoneme:
                n_empty_phoneme += 1

            writer.writerow({
                "index": idx,
                "name": str(item.get("name", "")),
                "transcription": transcription,
                "kana": kana,
                "phoneme": phoneme,
                "duration_sec": f"{duration_sec:.3f}",
                "kana_len": len(kana.split()) if kana else 0,
                "phoneme_len": len(phoneme.split()) if phoneme else 0,
            })

            if row_num % 100 == 0 or row_num == sample_count:
                log.info(f"Processed {row_num}/{sample_count}")

    log.info("Export completed.")
    log.info(f"rows={sample_count} empty_kana={n_empty_kana} empty_phoneme={n_empty_phoneme}")


if __name__ == "__main__":
    main()
