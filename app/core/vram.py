"""Free GPU memory after a full evaluate/coach/pitch burst.

The practice page fires those endpoints together. A request counter waits
until the last one finishes, then moves ASR weights back to CPU so the
3060 is empty for the next service (or the next language).
"""

from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_inflight = 0


def empty_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        logger.debug("CUDA cache clear skipped", exc_info=True)


def module_to_cpu(module) -> None:
    """Move every parameter/buffer off CUDA. `.to('cpu')` alone can leave HF caches."""
    import torch

    module.to("cpu")
    for param in module.parameters():
        if param.device.type == "cuda":
            param.data = param.data.cpu()
        param.grad = None
    for name, buf in list(module.named_buffers()):
        if buf is not None and getattr(buf, "device", None) is not None and buf.device.type == "cuda":
            module.register_buffer(name, buf.cpu(), persistent=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _offload_models() -> None:
    try:
        from app.services.asr_service import ASRService

        ASRService().offload()
        logger.info("Japanese ASR moved to CPU")
    except Exception:  # noqa: BLE001
        logger.warning("Japanese ASR offload failed", exc_info=True)
    try:
        from app.services.english_asr import get_english_asr_service

        get_english_asr_service().offload()
    except Exception:  # noqa: BLE001
        logger.debug("English ASR offload skipped", exc_info=True)
    try:
        from app.services.chinese_asr import get_chinese_asr_service

        get_chinese_asr_service().offload()
    except Exception:  # noqa: BLE001
        logger.debug("Chinese ASR offload skipped", exc_info=True)
    try:
        from app.services.pasqa_intonation import offload_pasqa

        offload_pasqa()
    except Exception:  # noqa: BLE001
        logger.debug("PASQA offload skipped", exc_info=True)


@contextmanager
def gpu_session() -> Iterator[None]:
    """Hold a GPU slot for one request. Last exit offloads + empty_cache."""
    global _inflight
    with _lock:
        _inflight += 1
    try:
        yield
    finally:
        idle = False
        with _lock:
            _inflight = max(0, _inflight - 1)
            idle = _inflight == 0
            if idle:
                _offload_models()
        if idle:
            empty_cuda()
            logger.info("VRAM released after pronunciation flow")
