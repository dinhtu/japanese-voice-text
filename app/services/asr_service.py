"""ASR service: a process-wide KanaRecognizer shared by all requests."""

import logging
import threading
from pathlib import Path

from app.core.config import Settings, get_settings
from src.asr.inference import DEFAULT_PRETRAINED, KanaRecognizer, RecognitionResult

logger = logging.getLogger(__name__)

_recognizer: KanaRecognizer | None = None
_lock = threading.Lock()


class ASRService:
    """Thin wrapper around KanaRecognizer with lazy, thread-safe loading.

    The model is heavy (315M params), so it is loaded once and reused. Inference
    itself is serialized with a lock — one MVP process, one model instance.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def load(self) -> KanaRecognizer:
        global _recognizer
        if _recognizer is None:
            with _lock:
                if _recognizer is None:
                    checkpoint = Path(self.settings.checkpoint)
                    if not checkpoint.exists():
                        raise FileNotFoundError(f"ASR checkpoint not found: {checkpoint}")
                    logger.info("Loading ASR checkpoint: %s", checkpoint)
                    _recognizer = KanaRecognizer(
                        checkpoint,
                        pretrained=self.settings.pretrained or DEFAULT_PRETRAINED,
                        inter_ctc_layer=self.settings.inter_ctc_layer,
                        device=self.settings.device,
                        fp16=self.settings.fp16,
                    )
                    logger.info("ASR model ready on %s", _recognizer.device)
        return _recognizer

    @property
    def is_loaded(self) -> bool:
        return _recognizer is not None

    def offload(self) -> None:
        """Move the JP ASR off CUDA. Safe no-op if it was never loaded."""
        with _lock:
            if _recognizer is not None:
                _recognizer.offload()

    def recognize(
        self, audio_path: str | Path, with_timing: bool = False
    ) -> RecognitionResult:
        """Transcribe an audio file to kana.

        `with_timing=True` also computes per-character CTC onset timing
        (see RecognitionResult.char_spans) -- used by /pitch-contour to
        align the learner's real pitch curve to the target's morae instead
        of guessing with equal time division.
        """
        recognizer = self.load()
        with _lock:
            return recognizer.transcribe(audio_path, with_timing=with_timing)


def get_asr_service() -> ASRService:
    """FastAPI dependency."""
    return ASRService()
