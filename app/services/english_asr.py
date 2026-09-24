"""English ASR: facebook/wav2vec2-base-960h (CTC), lazy-loaded."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import Settings, get_settings
from src.asr.inference import load_audio, resolve_device

logger = logging.getLogger(__name__)


@dataclass
class EnglishRecognition:
    text: str
    duration: float
    inference_time: float
    log_probs: np.ndarray | None = None
    vocab: list[str] | None = None


class EnglishASRService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = None
        self._processor = None
        self._device = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if self._model is None:
            model_id = getattr(self.settings, "en_asr_model", "facebook/wav2vec2-base-960h")
            self._device = resolve_device(getattr(self.settings, "en_asr_device", None))
            logger.info("Loading English ASR %s on %s", model_id, self._device)
            self._processor = Wav2Vec2Processor.from_pretrained(model_id)
            self._model = Wav2Vec2ForCTC.from_pretrained(model_id)
            self._model.eval()
        self._model.to(self._device)

    def offload(self) -> None:
        if self._model is None or self._device is None:
            return
        if self._device.type != "cuda":
            return
        from app.core.vram import module_to_cpu

        module_to_cpu(self._model)
        self._model.eval()

    def recognize(self, audio_path: str | Path) -> EnglishRecognition:
        import torch

        self.load()
        samples, sample_rate = load_audio(audio_path)
        duration = float(len(samples) / sample_rate) if sample_rate else 0.0
        inputs = self._processor(
            samples, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self._device)
        kwargs = {}
        mask = getattr(inputs, "attention_mask", None)
        if mask is not None:
            kwargs["attention_mask"] = mask.to(self._device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(input_values, **kwargs).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0].cpu().numpy()
        ids = logits.argmax(dim=-1)
        text = self._processor.batch_decode(ids)[0]
        vocab = [
            self._processor.tokenizer.convert_ids_to_tokens(i)
            for i in range(logits.shape[-1])
        ]
        del logits, input_values
        return EnglishRecognition(
            text=text.strip(),
            duration=duration,
            inference_time=time.perf_counter() - t0,
            log_probs=log_probs,
            vocab=vocab,
        )


@lru_cache
def get_english_asr_service() -> EnglishASRService:
    return EnglishASRService(get_settings())
