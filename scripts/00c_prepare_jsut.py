"""Prepare JSUT-BASIC5000 for evaluation.

Downloads JSUT corpus, extracts BASIC5000 subset,
and saves as HuggingFace Dataset with 16kHz audio.

Usage:
    uv run python scripts/00c_prepare_jsut.py
    uv run python scripts/00c_prepare_jsut.py --output-dir data/datasets/jsut
"""

import argparse
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

JSUT_URL = "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip"
JSUT_FILENAME = "jsut_ver1.1.zip"


def parse_args():
    p = argparse.ArgumentParser(description="Download and prepare JSUT-BASIC5000")
    p.add_argument("--output-dir", type=Path, default=Path("data/datasets/jsut"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache/jsut"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_transcript(transcript_path: Path) -> dict[str, str]:
    """Parse JSUT transcript_utf8.txt -> {utterance_id: text}."""
    entries = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            utt_id, text = line.split(":", 1)
            entries[utt_id.strip()] = text.strip()
    return entries


def main():
    args = parse_args()
    output_dir = args.output_dir / "basic5000"

    if output_dir.exists() and not args.overwrite:
        log.info(f"Already exists: {output_dir} (use --overwrite to replace)")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Download
    zip_path = args.cache_dir / JSUT_FILENAME
    if not zip_path.exists():
        log.info(f"Downloading JSUT from {JSUT_URL} ...")
        urllib.request.urlretrieve(JSUT_URL, zip_path)
        log.info(f"Downloaded: {zip_path} ({zip_path.stat().st_size / 1e9:.1f} GB)")
    else:
        log.info(f"Using cached: {zip_path}")

    # Extract basic5000 only
    extract_dir = args.cache_dir / "extracted"
    basic5000_dir = extract_dir / "jsut_ver1.1" / "basic5000"
    if not basic5000_dir.exists():
        log.info("Extracting basic5000 from archive...")
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.startswith("jsut_ver1.1/basic5000/")]
            zf.extractall(extract_dir, members)
        log.info(f"Extracted to {basic5000_dir}")

    # Parse transcript
    transcript_path = basic5000_dir / "transcript_utf8.txt"
    entries = parse_transcript(transcript_path)
    log.info(f"Transcript entries: {len(entries)}")

    # Build dataset
    wav_dir = basic5000_dir / "wav"
    names, audio_paths, transcriptions = [], [], []
    for utt_id, text in sorted(entries.items()):
        wav_path = wav_dir / f"{utt_id}.wav"
        if wav_path.exists():
            names.append(utt_id)
            audio_paths.append(str(wav_path))
            transcriptions.append(text)

    log.info(f"Matched {len(names)} / {len(entries)} entries to WAV files")

    ds = datasets.Dataset.from_dict({
        "name": names,
        "audio": audio_paths,
        "transcription": transcriptions,
    })
    # Cast to Audio with 16kHz target (auto-resamples from 48kHz on load)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    # Store without decoding (same pattern as ReazonSpeech preparation)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=None, decode=False))

    # Save
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    log.info(f"Saved {len(ds)} samples to {output_dir}")


if __name__ == "__main__":
    main()
