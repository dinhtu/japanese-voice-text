"""PASQA (LY Corp) pitch-accent MOS → 0--100 for intonation_score.

Inference code is vendored at vendor/pasqa (CC0). Weights live in
models/pasqa/ (download with `python scripts/download_pasqa.py`). Missing
weights or s3prl → None, F0 H/L stays in charge. Default device is CPU so
the 753MB head does not sit on the 3060 next to wav2vec2.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_VENDOR_SRC = Path(__file__).resolve().parents[2] / "vendor" / "pasqa" / "src"

_HIRA_TO_KATA = str.maketrans(
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんっー",
    "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンッー",
)


def _ensure_vendor_on_path() -> None:
    src = str(_VENDOR_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def morae_to_katakana(morae: list[str]) -> list[str]:
    return [m.translate(_HIRA_TO_KATA) for m in morae if m]


_loaded_predictor = None


@lru_cache(maxsize=1)
def _predictor(checkpoint: str, device: str):
    global _loaded_predictor
    _ensure_vendor_on_path()
    from pasqa import PasqaPredictor

    logger.info("Loading PASQA from %s on %s", checkpoint, device)
    _loaded_predictor = PasqaPredictor(checkpoint=checkpoint, device=device)
    return _loaded_predictor


def offload_pasqa() -> None:
    """Move PASQA + s3prl SSL off CUDA if a predictor was loaded."""
    if _loaded_predictor is None:
        return
    from app.core.vram import module_to_cpu

    model = getattr(_loaded_predictor, "model", None)
    if model is not None:
        module_to_cpu(model)
    ssl = getattr(model, "ssl_model", None) or getattr(model, "upstream", None)
    if ssl is not None:
        module_to_cpu(ssl)


def score_pasqa(
    audio_path: str | Path,
    morae: list[str],
    checkpoint: str | Path | None,
    device: str = "cpu",
) -> float | None:
    """MOS 1--5 → 0--100. None if weights or s3prl are missing."""
    if not checkpoint:
        return None
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        return None
    kata = morae_to_katakana(morae)
    if not kata:
        return None
    try:
        predictor = _predictor(str(ckpt), device)
        result = predictor.predict(wav_path=str(audio_path), mora=kata)
        mos = float(result["mos"] if isinstance(result, dict) else result)
    except Exception:  # noqa: BLE001
        logger.warning("PASQA inference failed; intonation uses F0 only", exc_info=True)
        return None
    return round(max(0.0, min(100.0, (mos - 1.0) / 4.0 * 100.0)), 2)
