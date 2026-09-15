"""Prepare and cache ReazonSpeech datasets locally.

Usage:
    uv run python scripts/00_prepare_dataset.py --splits small
    uv run python scripts/00_prepare_dataset.py --splits tiny,small --overwrite
    uv run python scripts/00_prepare_dataset.py --splits small --max-samples 5000
"""

# ruff: noqa: E402

import argparse
import io
import json
import logging
import random
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import torchaudio

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.asr.dataset import load_reazonspeech

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

VALID_SPLITS = {"tiny", "small", "medium"}


def parse_args():
    p = argparse.ArgumentParser(description="Download/cache ReazonSpeech and save it to disk")
    p.add_argument("--splits", default="small", help="Comma-separated list: tiny,small,medium")
    p.add_argument("--output-dir", type=Path, default=Path("data/datasets/reazonspeech"))
    p.add_argument("--overwrite", action="store_true", help="Replace existing saved split")
    p.add_argument("--max-samples", type=int, default=None, help="Limit samples per split")
    p.add_argument(
        "--duration-mode",
        choices=["none", "sample", "full"],
        default="sample",
        help="How to estimate total hours",
    )
    p.add_argument(
        "--duration-samples",
        type=int,
        default=1000,
        help="Number of samples used when duration-mode=sample",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-save", action="store_true", help="Only print summary")
    return p.parse_args()


def parse_splits(raw: str) -> list[str]:
    splits = [s.strip() for s in raw.split(",") if s.strip()]
    if not splits:
        raise ValueError("No split specified")
    invalid = [s for s in splits if s not in VALID_SPLITS]
    if invalid:
        valid = ", ".join(sorted(VALID_SPLITS))
        raise ValueError(f"Invalid split: {invalid}. Valid splits: {valid}")
    return splits


def sample_indices(n_rows: int, k: int, seed: int) -> list[int]:
    k = min(max(k, 0), n_rows)
    rng = random.Random(seed)
    if k == n_rows:
        return list(range(n_rows))
    return rng.sample(range(n_rows), k)


def infer_duration_seconds(audio_item) -> float:
    if not isinstance(audio_item, dict):
        return 0.0
    array = audio_item.get("array")
    sr = audio_item.get("sampling_rate")
    if array is not None and sr not in (None, 0):
        return float(len(array) / sr)

    audio_bytes = audio_item.get("bytes")
    if audio_bytes:
        try:
            info = torchaudio.info(io.BytesIO(audio_bytes))
            if info.sample_rate > 0 and info.num_frames > 0:
                return float(info.num_frames / info.sample_rate)
        except Exception:
            return 0.0
    return 0.0


def estimate_total_hours(
    ds,
    duration_mode: str,
    duration_samples: int,
    seed: int,
    audio_column: str = "audio",
) -> tuple[float | None, int]:
    n_rows = len(ds)
    if n_rows == 0 or duration_mode == "none":
        return None, 0

    if duration_mode == "full":
        indices = range(n_rows)
    else:
        indices = sample_indices(n_rows, duration_samples, seed)

    sampled_seconds = 0.0
    sampled = 0
    for idx in indices:
        item = ds[idx]
        sampled_seconds += infer_duration_seconds(item.get(audio_column))
        sampled += 1

    if sampled == 0:
        return None, 0

    if duration_mode == "full":
        total_seconds = sampled_seconds
    else:
        total_seconds = (sampled_seconds / sampled) * n_rows
    return total_seconds / 3600.0, sampled


def save_summary(summary_path: Path, summary: dict):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_split(
    split: str,
    output_dir: Path,
    overwrite: bool,
    max_samples: int | None,
    duration_mode: str,
    duration_samples: int,
    seed: int,
    skip_save: bool,
):
    split_dir = output_dir / split
    summary_path = split_dir / "summary.json"

    if split_dir.exists() and not overwrite and not skip_save:
        log.info(f"[{split}] already exists: {split_dir} (skip, use --overwrite)")
        return

    log.info(f"[{split}] loading from Hugging Face...")
    ds = load_reazonspeech(split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    n_rows = len(ds)
    est_hours, sampled = estimate_total_hours(
        ds=ds,
        duration_mode=duration_mode,
        duration_samples=duration_samples,
        seed=seed,
    )

    log.info(f"[{split}] rows={n_rows}")
    if est_hours is not None:
        if duration_mode == "full":
            log.info(f"[{split}] total_hours={est_hours:.2f} (full scan)")
        else:
            log.info(
                f"[{split}] estimated_hours={est_hours:.2f} "
                f"(sampled {sampled} rows)"
            )

    summary = {
        "split": split,
        "rows": n_rows,
        "estimated_hours": est_hours,
        "duration_mode": duration_mode,
        "duration_sampled_rows": sampled,
        "max_samples": max_samples,
        "saved_to": None if skip_save else str(split_dir),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    if skip_save:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if split_dir.exists() and overwrite:
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[{split}] saving to {split_dir} ...")
    ds.save_to_disk(str(split_dir))
    save_summary(summary_path, summary)
    log.info(f"[{split}] done")


def main():
    args = parse_args()
    splits = parse_splits(args.splits)

    for split in splits:
        prepare_split(
            split=split,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            max_samples=args.max_samples,
            duration_mode=args.duration_mode,
            duration_samples=args.duration_samples,
            seed=args.seed,
            skip_save=args.skip_save,
        )

    log.info("All requested splits completed.")


if __name__ == "__main__":
    main()
