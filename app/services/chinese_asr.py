"""Small Chinese CTC ASR service, lazy-loaded and VRAM-friendly."""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import Settings, get_settings
from src.asr.inference import load_audio, resolve_device


@dataclass
class ChineseRecognition:
    text: str
    duration: float
    inference_time: float
    windows: list[tuple[float, float]] | None = None


class ChineseASRService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = self._processor = self._device = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        if self._model is None:
            model_id = self.settings.zh_asr_model
            self._device = resolve_device(self.settings.zh_asr_device)
            self._processor = Wav2Vec2Processor.from_pretrained(model_id)
            self._model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()
        if self._device.type == "cuda":
            self._model.half()
        self._model.to(self._device)

    def offload(self) -> None:
        if self._model is not None and self._device is not None and self._device.type == "cuda":
            from app.core.vram import module_to_cpu
            module_to_cpu(self._model)

    def recognize(self, audio_path: str | Path, align_to: str = "") -> ChineseRecognition:
        import torch
        samples, sample_rate = load_audio(audio_path)
        duration = len(samples) / sample_rate if sample_rate else 0.0
        if duration > 30:
            raise ValueError("Chinese recordings must be 30 seconds or shorter.")
        self.load()
        inputs = self._processor(samples, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        values = inputs.input_values.to(self._device)
        if self._device.type == "cuda":
            values = values.half()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(values).logits
            ids = logits.argmax(dim=-1).cpu()
            windows = self._align(logits, align_to, duration) if align_to else None
        text = self._processor.batch_decode(ids)[0]
        return ChineseRecognition(text=text.strip(), duration=duration, inference_time=time.perf_counter() - t0, windows=windows)

    def _align(self, logits, target: str, duration: float) -> list[tuple[float, float]] | None:
        """CTC character windows; unavailable tokens leave pitch unmeasured."""
        from src.asr.force_align import _viterbi_token_frames

        vocab = self._processor.tokenizer.get_vocab()
        if not target or any(char not in vocab for char in target):
            return None
        log_probs = logits[0].float().log_softmax(dim=-1).cpu().numpy()
        spans = _viterbi_token_frames(
            log_probs, [vocab[char] for char in target], blank=self._model.config.pad_token_id
        )
        if not spans or any(span is None for span in spans):
            return None
        seconds_per_frame = duration / log_probs.shape[0]
        return [(start * seconds_per_frame, end * seconds_per_frame) for start, end in spans]


@lru_cache
def get_chinese_asr_service() -> ChineseASRService:
    return ChineseASRService(get_settings())
