"""Pre-download and cache the Korean wav2vec2 CTC model from Hugging Face.

    python scripts/download_korean_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings


def main() -> int:
    settings = get_settings()
    model_id = settings.ko_asr_model
    print(f"Downloading Korean ASR model: {model_id} ...")
    try:
        if "whisper" in model_id.lower():
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            print("Downloading Whisper processor...")
            WhisperProcessor.from_pretrained(model_id)
            print("Downloading Whisper model weights...")
            WhisperForConditionalGeneration.from_pretrained(model_id)
        else:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            print("Downloading Wav2Vec2 processor / tokenizer...")
            Wav2Vec2Processor.from_pretrained(model_id)
            print("Downloading Wav2Vec2 model weights...")
            Wav2Vec2ForCTC.from_pretrained(model_id)
        print("✓ Korean ASR model cached successfully.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error downloading Korean model: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
