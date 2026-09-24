"""Dataset loading for Japanese ASR training.

Supports ReazonSpeech with dual labels: kana (hiragana) + phoneme.
"""

import io
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

from src.asr.augmentation import add_noise, speed_perturbation
from src.asr.kana_converter import JapaneseKanaConverter
from src.asr.kana_vocab import KanaVocab
from src.asr.phoneme_converter import JapanesePhonemeConverter
from src.asr.phoneme_vocab import PhonemeVocab


def load_prepared_jvs(
    subset: str = "parallel100",
    dataset_root: str | Path = "data/datasets/jvs",
):
    """Load a locally prepared JVS subset from disk.

    JVS must be prepared first via scripts/00d_prepare_jvs.py.
    """
    from datasets import load_from_disk

    split_dir = Path(dataset_root) / subset
    if not split_dir.exists():
        raise RuntimeError(
            f"Prepared JVS dataset not found: {split_dir}\n"
            "Run `uv run python scripts/00d_prepare_jvs.py` first."
        )
    return load_from_disk(str(split_dir))


def load_prepared_jsut(
    subset: str = "basic5000",
    dataset_root: str | Path = "data/datasets/jsut",
):
    """Load a locally prepared JSUT subset from disk.

    JSUT must be prepared first via scripts/00c_prepare_jsut.py.
    """
    from datasets import load_from_disk

    split_dir = Path(dataset_root) / subset
    if not split_dir.exists():
        raise RuntimeError(
            f"Prepared JSUT dataset not found: {split_dir}\n"
            "Run `uv run python scripts/00c_prepare_jsut.py` first."
        )
    return load_from_disk(str(split_dir))


def load_prepared_reazonspeech(
    split: str,
    dataset_root: str | Path = "data/datasets/reazonspeech",
):
    """Load a locally prepared ReazonSpeech split from disk.

    This is the runtime path for train/eval scripts. Dataset download is
    intentionally handled by scripts/00_prepare_dataset.py.
    """
    from datasets import load_from_disk

    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        raise RuntimeError(
            f"Prepared dataset not found: {split_dir}\n"
            "Run `uv run python scripts/00_prepare_dataset.py --splits "
            f"{split}` first."
        )
    return load_from_disk(str(split_dir))


class ASRDataset(Dataset):
    """Dataset that wraps HuggingFace datasets for dual CTC training.

    Produces both kana and phoneme labels for each sample.
    """

    def __init__(
        self,
        hf_dataset,
        feature_extractor: Wav2Vec2FeatureExtractor,
        kana_converter: JapaneseKanaConverter | None = None,
        phoneme_converter: JapanesePhonemeConverter | None = None,
        kana_vocab: KanaVocab | None = None,
        phoneme_vocab: PhonemeVocab | None = None,
        max_duration_s: float = 15.0,
        audio_column: str = "audio",
        text_column: str = "transcription",
        target_sr: int = 16_000,
        speed_perturb_prob: float = 0.0,
        noise_prob: float = 0.0,
    ):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.kana_converter = kana_converter or JapaneseKanaConverter()
        self.phoneme_converter = phoneme_converter or JapanesePhonemeConverter()
        self.kana_vocab = kana_vocab or KanaVocab()
        self.phoneme_vocab = phoneme_vocab or PhonemeVocab()
        self.max_duration_s = max_duration_s
        self.audio_column = audio_column
        self.text_column = text_column
        self.target_sr = target_sr
        self.speed_perturb_prob = speed_perturb_prob
        self.noise_prob = noise_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Some rows can contain broken audio payloads. Retry a few nearby rows
        # instead of crashing DataLoader workers.
        last_error = None
        n_rows = len(self.dataset)
        max_attempts = 5
        for offset in range(max_attempts):
            cur_idx = (idx + offset) % n_rows
            try:
                item = self.dataset[cur_idx]

                # Extract audio
                audio_data = item[self.audio_column]
                waveform, sr = self._load_audio(audio_data)
                if waveform.dim() == 2:
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    waveform = waveform.squeeze(0)

                # Resample if needed
                if sr != self.target_sr:
                    waveform = waveform.unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                    waveform = resampler(waveform)
                    waveform = waveform.squeeze(0)

                waveform = waveform.float()

                # Optional train-time augmentation.
                if self.speed_perturb_prob > 0 and random.random() < self.speed_perturb_prob:
                    waveform = speed_perturbation(waveform, sample_rate=self.target_sr)
                if self.noise_prob > 0 and random.random() < self.noise_prob:
                    waveform = add_noise(waveform)

                # Keep memory usage bounded for long utterances.
                if self.max_duration_s > 0:
                    max_samples = int(self.max_duration_s * self.target_sr)
                    waveform = waveform[:max_samples]

                # Feature extraction
                inputs = self.feature_extractor(
                    waveform.numpy(), sampling_rate=self.target_sr, return_tensors="pt",
                )
                input_values = inputs.input_values.squeeze(0)  # (T,)

                # Text → kana labels
                text = item[self.text_column]
                kana_str = self.kana_converter.text_to_kana(text)
                kana_labels = self.kana_vocab.encode(kana_str)

                # Text → phoneme labels
                phoneme_str = self.phoneme_converter.text_to_phonemes(text)
                phoneme_labels = self.phoneme_vocab.encode(phoneme_str)

                return {
                    "input_values": input_values,
                    "kana_labels": torch.tensor(kana_labels, dtype=torch.long),
                    "phoneme_labels": torch.tensor(phoneme_labels, dtype=torch.long),
                    "kana_str": kana_str,
                    "phoneme_str": phoneme_str,
                }
            except Exception as e:
                last_error = e

        raise RuntimeError(
            f"Failed to load sample after {max_attempts} attempts (start_idx={idx})."
        ) from last_error

    def _load_audio(self, audio_data) -> tuple[torch.Tensor, int]:
        import soundfile as sf

        def _as_waveform_and_sr(arr, sr: int):
            waveform = np.asarray(arr, dtype=np.float32)
            # soundfile returns (samples, channels); convert to (channels, samples)
            if waveform.ndim == 2:
                waveform = waveform.T
            return torch.from_numpy(waveform), int(sr)

        if isinstance(audio_data, dict):
            array = audio_data.get("array")
            if array is not None:
                sr = int(audio_data.get("sampling_rate", self.target_sr))
                waveform = torch.from_numpy(np.asarray(array, dtype=np.float32))
                return waveform, sr

            audio_bytes = audio_data.get("bytes")
            if audio_bytes is not None:
                try:
                    arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
                    return _as_waveform_and_sr(arr, sr)
                except Exception:
                    pass

            audio_path = audio_data.get("path")
            if audio_path:
                try:
                    arr, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                    return _as_waveform_and_sr(arr, sr)
                except Exception:
                    pass

            raise RuntimeError("Failed to decode audio with soundfile from both bytes and path.")

        waveform = torch.from_numpy(np.asarray(audio_data, dtype=np.float32))
        return waveform, self.target_sr


class PreprocessedASRDataset(Dataset):
    """Dataset that reads pre-computed waveforms and labels.

    Much faster than ASRDataset: no FLAC decode, no on-the-fly g2p conversion.
    Only does optional augmentation + wav2vec2 normalization.
    """

    def __init__(
        self,
        hf_dataset,
        feature_extractor: Wav2Vec2FeatureExtractor,
        target_sr: int = 16_000,
        speed_perturb_prob: float = 0.0,
        noise_prob: float = 0.0,
    ):
        self.dataset = hf_dataset
        # Use numpy format for 50x faster reads (avoids Python list overhead)
        self.dataset.set_format("numpy")
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.speed_perturb_prob = speed_perturb_prob
        self.noise_prob = noise_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        waveform_np = np.ascontiguousarray(item["waveform"])

        # Optional train-time augmentation (needs torch tensor)
        if self.speed_perturb_prob > 0 and random.random() < self.speed_perturb_prob:
            waveform = speed_perturbation(
                torch.from_numpy(waveform_np), sample_rate=self.target_sr,
            )
            waveform_np = waveform.numpy()
        if self.noise_prob > 0 and random.random() < self.noise_prob:
            waveform_np = add_noise(torch.from_numpy(waveform_np)).numpy()

        # Normalize (same as Wav2Vec2FeatureExtractor but without Python overhead)
        mean = waveform_np.mean()
        std = waveform_np.std()
        if std > 0:
            waveform_np = (waveform_np - mean) / std
        input_values = torch.from_numpy(waveform_np)

        kana_labels = torch.from_numpy(item["kana_labels"].astype(np.int64))
        phoneme_labels = torch.from_numpy(item["phoneme_labels"].astype(np.int64))

        return {
            "input_values": input_values,
            "kana_labels": kana_labels,
            "phoneme_labels": phoneme_labels,
            "kana_str": str(item.get("kana_str", "")),
            "phoneme_str": str(item.get("phoneme_str", "")),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for DataLoader with dynamic padding."""
    # Pad input_values
    input_values = [item["input_values"] for item in batch]
    max_len = max(v.shape[0] for v in input_values)
    padded_inputs = torch.zeros(len(batch), max_len)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    input_lengths = []
    for i, v in enumerate(input_values):
        padded_inputs[i, :v.shape[0]] = v
        attention_mask[i, :v.shape[0]] = 1
        input_lengths.append(v.shape[0])

    # Concatenate kana labels (CTC format)
    kana_labels = [item["kana_labels"] for item in batch]
    kana_target_lengths = [len(lab) for lab in kana_labels]
    all_kana_labels = torch.cat(kana_labels)

    # Concatenate phoneme labels (CTC format)
    phoneme_labels = [item["phoneme_labels"] for item in batch]
    phoneme_target_lengths = [len(lab) for lab in phoneme_labels]
    all_phoneme_labels = torch.cat(phoneme_labels)

    return {
        "input_values": padded_inputs,
        "attention_mask": attention_mask,
        "kana_labels": all_kana_labels,
        "phoneme_labels": all_phoneme_labels,
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "kana_target_lengths": torch.tensor(kana_target_lengths, dtype=torch.long),
        "phoneme_target_lengths": torch.tensor(phoneme_target_lengths, dtype=torch.long),
        "kana_strs": [item["kana_str"] for item in batch],
        "phoneme_strs": [item["phoneme_str"] for item in batch],
    }


_REAZONSPEECH_BASE = "https://corpus.reazon-research.org/"
_REAZONSPEECH_SPLITS = {
    "tiny":   {"tsv": "reazonspeech-v2/tsv/tiny.tsv",   "audio": "reazonspeech-v2/data/{:03x}.tar", "nfiles": 1},
    "small":  {"tsv": "reazonspeech-v2/tsv/small.tsv",  "audio": "reazonspeech-v2/data/{:03x}.tar", "nfiles": 12},
    "medium": {"tsv": "reazonspeech-v2/tsv/medium.tsv", "audio": "reazonspeech-v2/data/{:03x}.tar", "nfiles": 116},
}


def _download_reazonspeech_direct(split: str, cache_dir: str | Path | None = None):
    """Download ReazonSpeech by fetching TSV + tar archives directly.

    This bypasses the HuggingFace datasets loading script (deprecated in
    datasets>=4.0) and downloads audio from corpus.reazon-research.org.
    """
    import logging
    import tarfile
    import tempfile
    import urllib.request
    from time import sleep

    import datasets

    log = logging.getLogger(__name__)

    if split not in _REAZONSPEECH_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Valid: {list(_REAZONSPEECH_SPLITS)}")

    info = _REAZONSPEECH_SPLITS[split]
    cache = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "reazonspeech_cache"
    cache.mkdir(parents=True, exist_ok=True)

    # 1) Download TSV transcript
    tsv_url = _REAZONSPEECH_BASE + info["tsv"]
    tsv_path = cache / f"{split}.tsv"
    if not tsv_path.exists():
        log.info(f"Downloading transcript: {tsv_url}")
        urllib.request.urlretrieve(tsv_url, tsv_path)

    meta = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
    log.info(f"Transcript entries: {len(meta)}")

    # 2) Download & extract tar archives
    audio_dir = cache / f"{split}_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    max_retries = 5
    retry_wait_sec = 10

    for idx in range(info["nfiles"]):
        tar_url = _REAZONSPEECH_BASE + info["audio"].format(idx)
        tar_path = cache / f"{split}_{idx:03x}.tar"
        marker = cache / f"{split}_{idx:03x}.extracted"
        if marker.exists():
            continue
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                if not tar_path.exists():
                    log.info(
                        f"Downloading archive {idx+1}/{info['nfiles']} "
                        f"(attempt {attempt}/{max_retries}): {tar_url}"
                    )
                    urllib.request.urlretrieve(tar_url, tar_path)

                log.info(
                    f"Extracting archive {idx+1}/{info['nfiles']} "
                    f"(attempt {attempt}/{max_retries})..."
                )
                with tarfile.open(tar_path) as tf:
                    tf.extractall(audio_dir)
                tar_path.unlink(missing_ok=True)  # save disk space
                marker.touch()
                success = True
                break
            except Exception as e:
                tar_path.unlink(missing_ok=True)
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Failed archive {idx+1}/{info['nfiles']} after {max_retries} attempts: "
                        f"{tar_url}"
                    ) from e
                log.warning(
                    f"Archive {idx+1}/{info['nfiles']} failed on attempt {attempt}: {e}. "
                    f"Retrying in {retry_wait_sec}s..."
                )
                sleep(retry_wait_sec)

        if not success:
            raise RuntimeError(f"Failed to process archive {idx+1}/{info['nfiles']}: {tar_url}")

    # 3) Build dataset from extracted files
    log.info("Building dataset from extracted audio...")
    names, audio_paths, transcriptions = [], [], []
    for filename, transcription in meta.items():
        audio_path = audio_dir / filename
        if audio_path.exists():
            names.append(filename)
            audio_paths.append(str(audio_path))
            transcriptions.append(transcription)

    log.info(f"Matched {len(names)} / {len(meta)} transcript entries to audio files")

    ds = datasets.Dataset.from_dict({
        "name": names,
        "audio": audio_paths,
        "transcription": transcriptions,
    })
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    # Re-cast to not decode so prepare_dataset can handle it
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=None, decode=False))
    return ds


def load_reazonspeech(split: str = "small", streaming: bool = False, cache_dir: str | Path | None = None):
    """Load ReazonSpeech dataset.

    First tries the HuggingFace datasets loader. If that fails (e.g. because
    dataset scripts are no longer supported), falls back to direct download
    from corpus.reazon-research.org.

    Args:
        split: Dataset size — "tiny" (8.5h), "small" (100h), "medium" (1000h).
        streaming: Whether to stream instead of downloading fully.
        cache_dir: Directory for caching downloaded archives.

    Returns:
        HuggingFace dataset object.
    """
    import logging

    from datasets import Audio, load_dataset

    log = logging.getLogger(__name__)

    # Medium is large and HF downloader often times out on Vast; direct path is resumable.
    if split == "medium" and not streaming:
        log.info("Using direct downloader for ReazonSpeech medium.")
        return _download_reazonspeech_direct(split, cache_dir=cache_dir)

    try:
        ds = load_dataset(
            "reazon-research/reazonspeech",
            name=split,
            split="train",
            streaming=streaming,
            trust_remote_code=True,
        )
        return ds.cast_column("audio", Audio(sampling_rate=None, decode=False))
    except Exception as e:
        msg = str(e).lower()
        if "gated dataset" in msg or "you must be authenticated" in msg:
            raise RuntimeError(
                "ReazonSpeech is gated. Authenticate with Hugging Face first "
                "(`huggingface-cli login`), then accept dataset access terms."
            ) from e

        log.info(f"HuggingFace loader failed ({e}), using direct download...")
        return _download_reazonspeech_direct(split, cache_dir=cache_dir)
