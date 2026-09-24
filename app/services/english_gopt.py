"""English-only GOPT: official MIT weights + wav2vec2 LPP/LPR features.

Kaldi is not installed. Phone-level GOP is computed the same way as
`compute-gop` (mean log-posterior + log-posterior ratio) from the
English wav2vec2-base-960h CTC softmax already used for /en ASR.

Utterance heads (SpeechOcean762 0-10, GOPT stores them /5):
    accuracy, completeness, fluency, prosodic, total
mapped onto the page as pronunciation / rhythm / fluency / intonation / overall.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import ROOT_DIR, Settings, get_settings
from app.services.english_g2p import PHONES, PHONE_TO_LETTER, phone_ids, text_phones
from app.services.gopt_model import (
    GOPT,
    GOPT_FEAT_DIM,
    GOPT_SEQ_LEN,
    LIBRISPEECH_FEAT_MEAN,
    LIBRISPEECH_FEAT_STD,
    strip_dataparallel,
)

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "gopt" / "best_audio_model.pth"
WEIGHTS_URL = (
    "https://github.com/YuanGongND/gopt/raw/master/"
    "pretrained_models/gopt_librispeech/best_audio_model.pth"
)
WEIGHTS_MIN_BYTES = 80_000
# 42-wide LPP/LPR like Librispeech pure-phone GOP; first dim is 1-based phone id.
_PHONE_FEAT = 41


@dataclass(frozen=True)
class GoptScores:
    overall: float
    pronunciation: float
    fluency: float
    rhythm: float
    intonation: float


def download_gopt_weights(dest: Path = DEFAULT_CHECKPOINT) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size >= WEIGHTS_MIN_BYTES:
        return dest
    logger.info("Downloading GOPT weights to %s", dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    if tmp.stat().st_size < WEIGHTS_MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("GOPT weight download looks truncated")
    tmp.replace(dest)
    return dest


def _letter_index(vocab: list[str], letter: str) -> int | None:
    upper = letter.upper()
    if upper in vocab:
        return vocab.index(upper)
    if letter in vocab:
        return vocab.index(letter)
    if "|" in vocab and letter in {"|", "SIL"}:
        return vocab.index("|")
    return None


def _lpp_lpr(log_probs: np.ndarray, phone: str, vocab: list[str]) -> np.ndarray:
    """84-d Kaldi-style vector: [phone_id, LPP…, LPR…]."""
    phone_id = next((i for i, name in enumerate(PHONES) if name == phone), 3)
    lpp = np.mean(log_probs, axis=0)
    n_vocab = lpp.shape[0]
    lpp_pad = np.full(_PHONE_FEAT, -10.0, dtype=np.float64)
    lpp_pad[: min(n_vocab, _PHONE_FEAT)] = lpp[: min(n_vocab, _PHONE_FEAT)]

    canon = _letter_index(vocab, PHONE_TO_LETTER.get(phone, "A"))
    canon_lpp = float(lpp[canon]) if canon is not None else float(lpp.max())
    lpr = np.full(_PHONE_FEAT, 0.0, dtype=np.float64)
    width = min(n_vocab, _PHONE_FEAT)
    lpr[:width] = canon_lpp - lpp[:width]

    feat = np.zeros(GOPT_FEAT_DIM, dtype=np.float64)
    packed = np.concatenate(([float(phone_id + 1)], lpp_pad, lpr))
    feat[: min(GOPT_FEAT_DIM, packed.size)] = packed[:GOPT_FEAT_DIM]
    return feat


def build_gopt_features(
    log_probs: np.ndarray,
    target_text: str,
    vocab: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (1, 50, 84) features and (1, 50) phone ids, or None."""
    phones = text_phones(target_text)
    if not phones or log_probs.size == 0:
        return None
    ids = phone_ids(phones)
    n_frames = log_probs.shape[0]
    n = len(phones)
    feat = np.zeros((GOPT_SEQ_LEN, GOPT_FEAT_DIM), dtype=np.float64)
    phn = np.full(GOPT_SEQ_LEN, -1, dtype=np.int64)
    for i, (phone, pid) in enumerate(zip(phones, ids)):
        start = int(i * n_frames / n)
        end = int((i + 1) * n_frames / n)
        end = max(end, start + 1)
        window = log_probs[start:end]
        if window.size == 0:
            continue
        vec = _lpp_lpr(window, phone, vocab)
        feat[i] = (vec - LIBRISPEECH_FEAT_MEAN) / LIBRISPEECH_FEAT_STD
        phn[i] = min(pid, 38)
    if not np.any(feat[:, 0] != 0):
        return None
    return feat[np.newaxis], phn[np.newaxis]


def _to_100(value: float) -> float:
    # GOPT utterance labels were divided by 5 (0-10 → 0-2).
    return round(max(0.0, min(100.0, float(value) * 50.0)), 2)


class EnglishGoptService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model: GOPT | None = None
        self._failed = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        if not getattr(self.settings, "gopt_enabled", True):
            return False
        try:
            import torch

            path = getattr(self.settings, "gopt_checkpoint", None) or DEFAULT_CHECKPOINT
            path = Path(path)
            if not path.is_file():
                path = download_gopt_weights(path)
            device = str(getattr(self.settings, "gopt_device", "cpu") or "cpu")
            logger.info("Loading GOPT from %s on %s", path, device)
            model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(strip_dataparallel(state), strict=True)
            model.to(device)
            model.eval()
            self._model = model
            return True
        except Exception:  # noqa: BLE001
            self._failed = True
            logger.warning("GOPT unavailable; English keeps wav2vec2+VAD+F0", exc_info=True)
            return False

    def score(self, target_text: str, log_probs: np.ndarray, vocab: list[str]) -> GoptScores | None:
        if not self.load() or self._model is None:
            return None
        built = build_gopt_features(log_probs, target_text, vocab)
        if built is None:
            return None
        import torch

        feat, phn = built
        device = next(self._model.parameters()).device
        with torch.inference_mode():
            u1, u2, u3, u4, u5, *_ = self._model(
                torch.from_numpy(feat).float().to(device),
                torch.from_numpy(phn).to(device),
            )
        return GoptScores(
            pronunciation=_to_100(u1.reshape(-1)[0].item()),
            rhythm=_to_100(u2.reshape(-1)[0].item()),
            fluency=_to_100(u3.reshape(-1)[0].item()),
            intonation=_to_100(u4.reshape(-1)[0].item()),
            overall=_to_100(u5.reshape(-1)[0].item()),
        )


@lru_cache
def get_english_gopt_service() -> EnglishGoptService:
    return EnglishGoptService(get_settings())
