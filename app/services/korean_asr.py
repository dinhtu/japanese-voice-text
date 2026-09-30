"""Lazy, VRAM-friendly Korean ASR service supporting Whisper and Wav2Vec2 CTC."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import Settings, get_settings
from src.asr.inference import load_audio, resolve_device


@dataclass
class KoreanRecognition:
    text: str
    duration: float
    inference_time: float
    windows: list[tuple[float, float]] | None = None


class KoreanASRService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = self._processor = self._device = None

    @property
    def is_whisper(self) -> bool:
        return "whisper" in self.settings.ko_asr_model.lower()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        self._device = resolve_device(self.settings.ko_asr_device)
        model_id = self.settings.ko_asr_model

        if self.is_whisper:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            self._processor = WhisperProcessor.from_pretrained(model_id)
            self._model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
        else:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            self._processor = Wav2Vec2Processor.from_pretrained(model_id)
            self._model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()

        if self._device.type == "cuda" and self.settings.ko_asr_fp16:
            self._model.half()
        else:
            self._model.float()
        self._model.to(self._device)

    def offload(self) -> None:
        if self._model is not None and self._device is not None and self._device.type == "cuda":
            from app.core.vram import module_to_cpu

            module_to_cpu(self._model)

    def recognize(self, audio_path: str | Path, align_to: str = "") -> KoreanRecognition:
        import numpy as np
        import torch

        samples, sample_rate = load_audio(audio_path)
        duration = len(samples) / sample_rate if sample_rate else 0.0
        if duration > 30:
            raise ValueError("Korean recordings must be 30 seconds or shorter.")

        # Audio amplitude peak normalization to prevent low-volume mic drops
        if len(samples) > 0:
            peak = float(np.max(np.abs(samples)))
            if peak > 1e-4:
                samples = (samples / peak) * 0.95

        self.load()
        t0 = time.perf_counter()

        if self.is_whisper:
            inputs = self._processor(samples, sampling_rate=sample_rate, return_tensors="pt")
            features = inputs.input_features.to(self._device)
            if self._device.type == "cuda" and self.settings.ko_asr_fp16:
                features = features.half()
            with torch.inference_mode():
                predicted_ids = self._model.generate(
                    features,
                    language="korean",
                    task="transcribe",
                    no_repeat_ngram_size=3,
                )
            text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            del features, predicted_ids
            return KoreanRecognition(text.strip(), duration, time.perf_counter() - t0, None)

        # Wav2Vec2 CTC path
        inputs = self._processor(samples, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        values = inputs.input_values.to(self._device)
        if self._device.type == "cuda" and self.settings.ko_asr_fp16:
            values = values.half()
        with torch.inference_mode():
            logits = self._model(values).logits
            pad_id = getattr(self._model.config, "pad_token_id", None) or self._processor.tokenizer.pad_token_id
            dec_logits = logits.clone()
            if pad_id is not None:
                dec_logits[..., pad_id] -= 0.6
            ids = dec_logits.argmax(dim=-1).cpu()
            windows = self._align(logits, align_to, duration) if align_to else None
        text = self._processor.batch_decode(ids)[0]
        del logits, dec_logits, values
        return KoreanRecognition(text.strip(), duration, time.perf_counter() - t0, windows)

    def _align(self, logits, target: str, duration: float):
        from src.asr.force_align import _viterbi_token_frames

        vocab = self._processor.tokenizer.get_vocab()
        token_ids = []
        for char in target:
            encoded = self._processor.tokenizer(char, add_special_tokens=False).input_ids
            if len(encoded) != 1 or encoded[0] == self._processor.tokenizer.unk_token_id:
                return None
            token_ids.append(encoded[0])
        probs = logits[0].float().log_softmax(dim=-1).cpu().numpy()
        spans = _viterbi_token_frames(probs, token_ids, blank=self._model.config.pad_token_id)
        if not spans or any(span is None for span in spans):
            return None
        seconds = duration / probs.shape[0]
        return [(start * seconds, end * seconds) for start, end in spans]


@lru_cache
def get_korean_asr_service() -> KoreanASRService:
    return KoreanASRService(get_settings())
