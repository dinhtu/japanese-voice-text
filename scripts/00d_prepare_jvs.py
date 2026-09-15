"""Prepare JVS (Japanese Versatile Speech) corpus for evaluation.

Downloads JVS corpus from Google Drive, extracts parallel100 subset,
and saves as HuggingFace Dataset with 16kHz audio.

Usage:
    uv run python scripts/00d_prepare_jvs.py
    uv run python scripts/00d_prepare_jvs.py --output-dir data/datasets/jvs
    uv run python scripts/00d_prepare_jvs.py --subset nonpara30
"""

import argparse
import logging
import shutil
import zipfile
from pathlib import Path

import datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Google Drive file ID for jvs_ver1.zip (~3.5GB)
JVS_GDRIVE_ID = "19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt"


def parse_args():
    p = argparse.ArgumentParser(description="Download and prepare JVS corpus")
    p.add_argument("--output-dir", type=Path, default=Path("data/datasets/jvs"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache/jvs"))
    p.add_argument(
        "--subset", default="parallel100",
        choices=["parallel100", "nonpara30"],
        help="JVS subset to prepare",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_transcript(transcript_path: Path) -> dict[str, str]:
    """Parse JVS transcript_utf8.txt -> {utterance_id: text}."""
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
    output_dir = args.output_dir / args.subset

    if output_dir.exists() and not args.overwrite:
        log.info(f"Already exists: {output_dir} (use --overwrite to replace)")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Download from Google Drive
    zip_path = args.cache_dir / "jvs_ver1.zip"
    if not zip_path.exists():
        import gdown

        log.info("Downloading JVS corpus from Google Drive (~3.5GB)...")
        gdown.download(id=JVS_GDRIVE_ID, output=str(zip_path), quiet=False)
        log.info(f"Downloaded: {zip_path} ({zip_path.stat().st_size / 1e9:.1f} GB)")
    else:
        log.info(f"Using cached: {zip_path}")

    # Extract
    extract_dir = args.cache_dir / "extracted"
    jvs_root = extract_dir / "jvs_ver1"
    if not jvs_root.exists():
        log.info("Extracting archive...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        log.info(f"Extracted to {jvs_root}")

    # Collect samples from all speakers
    speaker_dirs = sorted(jvs_root.glob("jvs???"))
    log.info(f"Found {len(speaker_dirs)} speakers")

    names, audio_paths, transcriptions, speaker_ids = [], [], [], []

    for spk_dir in speaker_dirs:
        spk_id = spk_dir.name
        subset_dir = spk_dir / args.subset
        if not subset_dir.exists():
            log.warning(f"Subset {args.subset} not found for {spk_id}, skipping")
            continue

        transcript_path = subset_dir / "transcripts_utf8.txt"
        if not transcript_path.exists():
            log.warning(f"No transcript for {spk_id}/{args.subset}, skipping")
            continue

        entries = parse_transcript(transcript_path)

        # Look for wav files in wav24kHz16bit/ or wav/
        wav_dir = subset_dir / "wav24kHz16bit"
        if not wav_dir.exists():
            wav_dir = subset_dir / "wav"
        if not wav_dir.exists():
            log.warning(f"No wav directory for {spk_id}/{args.subset}, skipping")
            continue

        for utt_id, text in sorted(entries.items()):
            wav_path = wav_dir / f"{utt_id}.wav"
            if wav_path.exists():
                names.append(f"{spk_id}_{utt_id}")
                audio_paths.append(str(wav_path))
                transcriptions.append(text)
                speaker_ids.append(spk_id)

    log.info(f"Collected {len(names)} samples from {len(set(speaker_ids))} speakers")

    # Build dataset
    ds = datasets.Dataset.from_dict({
        "name": names,
        "audio": audio_paths,
        "transcription": transcriptions,
        "speaker_id": speaker_ids,
    })
    # Cast to Audio with 16kHz target (auto-resamples from 24kHz on load)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    # Store without decoding
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=None, decode=False))

    # Save
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    log.info(f"Saved {len(ds)} samples to {output_dir}")


if __name__ == "__main__":
    main()
