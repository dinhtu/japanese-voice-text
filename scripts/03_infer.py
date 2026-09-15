"""Inference demo for kana ASR with Spike Window Decoding (SWD).

Supports file input. Outputs kana sequence (+ optional phoneme from InterCTC).

Usage:
    uv run python scripts/03_infer.py --audio data/test.wav --checkpoint models/checkpoints/best.pt
    uv run python scripts/03_infer.py --audio data/test.wav --checkpoint ... --swd
"""

# ruff: noqa: E402

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.asr.inference import (  # noqa: F401
    DEFAULT_PRETRAINED,
    KanaRecognizer,
    load_audio,
    swd_decode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Kana ASR inference with SWD")
    p.add_argument("--audio", required=True, type=Path, help="Audio file path")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    p.add_argument("--inter-ctc-layer", type=int, default=None)
    p.add_argument("--swd", action="store_true", help="Enable Spike Window Decoding")
    p.add_argument("--swd-window", type=int, default=1, help="SWD window size")
    p.add_argument("--show-phonemes", action="store_true", help="Also show InterCTC phonemes")
    p.add_argument("--fp16", action="store_true", help="Use FP16 inference (recommended for MPS)")
    return p.parse_args()


def main():
    args = parse_args()

    log.info(f"Loading model from {args.checkpoint}")
    recognizer = KanaRecognizer(
        args.checkpoint,
        pretrained=args.pretrained,
        inter_ctc_layer=args.inter_ctc_layer,
        fp16=args.fp16,
    )

    log.info(f"Loading audio: {args.audio}")
    audio_array, sr = load_audio(args.audio)
    log.info(f"Audio duration: {len(audio_array) / sr:.2f}s")

    result = recognizer.transcribe(
        audio_array,
        swd=args.swd,
        swd_window=args.swd_window,
        with_phonemes=args.show_phonemes,
    )

    print(f"\nKana: {result.kana}")
    print(f"Inference: {result.inference_time * 1000:.1f}ms (RTF: {result.rtf:.3f})")
    if args.swd:
        print(f"Decoding: SWD (window={args.swd_window})")

    if args.show_phonemes:
        print(f"Phonemes (InterCTC): {result.phonemes}")


if __name__ == "__main__":
    main()
