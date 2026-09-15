"""Evaluate a trained kana + phoneme ASR model.

Computes KER, PER, and group confusion analysis.

Usage:
    uv run python scripts/02_evaluate.py --checkpoint models/checkpoints/best.pt
    uv run python scripts/02_evaluate.py --checkpoint models/checkpoints/best.pt --data-split small
    uv run python scripts/02_evaluate.py --checkpoint models/checkpoints/best.pt --dataset jsut
"""

# ruff: noqa: E402

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.asr.dataset import ASRDataset, collate_fn, load_prepared_jsut, load_prepared_jvs, load_prepared_reazonspeech
from src.asr.kana_converter import JapaneseKanaConverter
from src.asr.kana_vocab import KanaVocab
from src.asr.metrics import (
    format_confusion_report,
    kana_confusion_analysis,
    kana_error_rate,
    phoneme_confusion_analysis,
    phoneme_error_rate,
)
from src.asr.model import load_checkpoint
from src.asr.phoneme_converter import JapanesePhonemeConverter
from src.asr.phoneme_vocab import PhonemeVocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate kana + phoneme ASR model")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--pretrained", default="reazon-research/japanese-wav2vec2-base")
    p.add_argument("--inter-ctc-layer", type=int, default=None)
    p.add_argument("--dataset", default="reazonspeech", choices=["reazonspeech", "jsut", "jvs"])
    p.add_argument("--data-split", default="small", choices=["tiny", "small", "medium"])
    p.add_argument("--dataset-dir", type=Path, default=Path("data/datasets/reazonspeech"))
    p.add_argument("--jsut-subset", default="basic5000")
    p.add_argument("--jsut-dir", type=Path, default=Path("data/datasets/jsut"))
    p.add_argument("--jvs-subset", default="parallel100", choices=["parallel100", "nonpara30"])
    p.add_argument("--jvs-dir", type=Path, default=Path("data/datasets/jvs"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None, help="Limit eval samples")
    p.add_argument("--max-duration", type=float, default=15.0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Loading model from {args.checkpoint}")
    model = load_checkpoint(
        str(args.checkpoint), args.pretrained,
        inter_ctc_layer=args.inter_ctc_layer,
    )
    model.to(device)
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.pretrained)
    kana_converter = JapaneseKanaConverter()
    phoneme_converter = JapanesePhonemeConverter()
    kana_vocab = KanaVocab()
    phoneme_vocab = PhonemeVocab()

    if args.dataset == "jsut":
        log.info(f"Loading prepared JSUT ({args.jsut_subset}) from {args.jsut_dir} ...")
        hf_dataset = load_prepared_jsut(args.jsut_subset, args.jsut_dir)
    elif args.dataset == "jvs":
        log.info(f"Loading prepared JVS ({args.jvs_subset}) from {args.jvs_dir} ...")
        hf_dataset = load_prepared_jvs(args.jvs_subset, args.jvs_dir)
    else:
        log.info(f"Loading prepared ReazonSpeech ({args.data_split}) from {args.dataset_dir} ...")
        hf_dataset = load_prepared_reazonspeech(args.data_split, args.dataset_dir)
    if args.max_samples:
        hf_dataset = hf_dataset.select(range(min(args.max_samples, len(hf_dataset))))

    dataset = ASRDataset(
        hf_dataset, feature_extractor,
        kana_converter=kana_converter,
        phoneme_converter=phoneme_converter,
        kana_vocab=kana_vocab,
        phoneme_vocab=phoneme_vocab,
        max_duration_s=args.max_duration,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    all_kana_ref = []
    all_kana_hyp = []
    all_phoneme_ref = []
    all_phoneme_hyp = []

    log.info("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            outputs = model(input_values, attention_mask=attention_mask)
            kana_pred_ids = outputs["kana_logits"].argmax(dim=-1)
            phoneme_pred_ids = outputs["phoneme_logits"].argmax(dim=-1)

            out_lengths = model.get_feat_extract_output_lengths(input_lengths)

            for i in range(kana_pred_ids.shape[0]):
                length = out_lengths[i].item()
                kana_hyp = kana_vocab.decode(kana_pred_ids[i, :length].tolist())
                phoneme_hyp = phoneme_vocab.decode(phoneme_pred_ids[i, :length].tolist())
                all_kana_hyp.append(kana_hyp)
                all_phoneme_hyp.append(phoneme_hyp)
                kana_tokens = batch["kana_strs"][i].split()
                kana_ref = "".join(" " if tok == "<sp>" else tok for tok in kana_tokens)
                all_kana_ref.append(kana_ref)
                all_phoneme_ref.append(batch["phoneme_strs"][i])

            if batch_idx % 50 == 0:
                log.info(f"  Batch {batch_idx + 1}")

    # Compute KER
    ker_scores = [kana_error_rate(r, h) for r, h in zip(all_kana_ref, all_kana_hyp)]
    avg_ker = sum(ker_scores) / len(ker_scores)
    log.info(f"\nAverage KER: {avg_ker:.4f} ({len(ker_scores)} samples)")

    # Compute PER
    per_scores = [phoneme_error_rate(r, h) for r, h in zip(all_phoneme_ref, all_phoneme_hyp)]
    avg_per = sum(per_scores) / len(per_scores)
    log.info(f"Average PER: {avg_per:.4f}")

    # Kana confusion analysis
    print("\n=== Kana (Primary Output) ===")
    kana_analysis = kana_confusion_analysis(all_kana_ref, all_kana_hyp)
    print(format_confusion_report(kana_analysis, "KER"))

    # Phoneme confusion analysis
    print("\n=== Phoneme (InterCTC) ===")
    phoneme_analysis = phoneme_confusion_analysis(all_phoneme_ref, all_phoneme_hyp)
    print(format_confusion_report(phoneme_analysis, "PER"))

    # Show some examples
    print("\n\nSample predictions:")
    print("-" * 70)
    for i in range(min(10, len(all_kana_ref))):
        print(f"KANA REF: {all_kana_ref[i]}")
        print(f"KANA HYP: {all_kana_hyp[i]}")
        print(f"KER: {ker_scores[i]:.3f}")
        print(f"PHON REF: {all_phoneme_ref[i]}")
        print(f"PHON HYP: {all_phoneme_hyp[i]}")
        print(f"PER: {per_scores[i]:.3f}")
        print()


if __name__ == "__main__":
    main()
