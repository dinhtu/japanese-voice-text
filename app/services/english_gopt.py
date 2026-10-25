"""English-only GOPT: official MIT weights + wav2vec2 LPP/LPR features.

Kaldi is not installed. Phone-level GOP is computed the same way as
`compute-gop` (mean log-posterior + log-posterior ratio) from the
English wav2vec2-base-960h CTC softmax already used for /en ASR.

Utterance heads (SpeechOcean762 0-10, GOPT stores them /5):
    accuracy, completeness, fluency, prosodic, total
Accuracy / fluency / prosodic overlay the page bars when the head
actually fired. Completeness is not rhythm (Nhịp stays local timing).
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import ROOT_DIR, Settings, get_settings
from app.services.english_g2p import (
    PHONE_TO_ID,
    PHONE_TO_LETTER,
    PHONES,
    phone_ids,
    text_phones,
    word_phones,
)
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
    pronunciation: float  # accuracy 0-100
    completeness: float
    fluency: float
    intonation: float  # prosodic
    rhythm: float = 0.0  # unused; GOPT has no rhythm head


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


def _mean_letter_lpp(log_probs: np.ndarray, vocab: list[str]) -> np.ndarray:
    """41-d LPP: mean log-posterior of each Kaldi phone's wav2vec2 letter."""
    frame_mean = np.mean(log_probs, axis=0)
    lpp = np.full(_PHONE_FEAT, -20.0, dtype=np.float64)
    for i, name in enumerate(PHONES):
        idx = _letter_index(vocab, PHONE_TO_LETTER.get(name, "A"))
        if idx is not None and 0 <= idx < frame_mean.shape[0]:
            lpp[i] = float(frame_mean[idx])
    return lpp


def _lpp_lpr(log_probs: np.ndarray, phone: str, vocab: list[str]) -> np.ndarray:
    """84-d Kaldi-style vector: [phone_id, LPP…, LPR…]."""
    phone_id = next((i for i, name in enumerate(PHONES) if name == phone), 3)
    lpp = _mean_letter_lpp(log_probs, vocab)
    canon = next((i for i, name in enumerate(PHONES) if name == phone), 3)
    canon_lpp = float(lpp[canon])
    lpr = canon_lpp - lpp

    feat = np.zeros(GOPT_FEAT_DIM, dtype=np.float64)
    packed = np.concatenate(([float(phone_id + 1)], lpp, lpr))
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
    if not np.any(phn >= 0):
        return None
    return feat[np.newaxis], phn[np.newaxis]


def _gopt_phones(cmu: list[str]) -> list[str]:
    phones = ["SH" if p == "ZH" else p for p in (p.rstrip("012") for p in cmu)]
    return [p for p in phones if p in PHONE_TO_ID]


def word_cmu_phones(spoken: str, initialism: bool = False) -> list[str]:
    """Stress-stripped CMUdict phones of one spoken word (first entry; ZH
    folded into SH like PHONES), or english_g2p's spelling rules. An
    initialism ("U.S.") is its letter names, not the word "us"."""
    from app.services.english_lexicon import _cmudict, _letter_name

    letters = spoken.replace("'", "")
    if initialism:
        names = [_letter_name(ch) for ch in letters]
        if all(names):
            phones = _gopt_phones([p for name in names for p in name])
            if phones:
                return phones
    entries = _cmudict().get(spoken) or _cmudict().get(letters)
    if entries:
        phones = _gopt_phones(entries[0])
        if phones:
            return phones
    return word_phones(letters)


def build_word_gopt_features(
    log_probs: np.ndarray,
    words: list[list[str]],
    vocab: list[str],
    word_frames: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]] | None:
    """Like build_gopt_features, but phones come per word and, with
    `word_frames` (start, end frame per word, from forced alignment), each
    phone's window is a share of its own word instead of a share of the
    whole clip. Returns (1, 50, 84) features, (1, 50) phone ids and, per
    word, the sequence slots of its phones (empty when truncated away)."""
    n_frames = log_probs.shape[0] if log_probs.ndim == 2 else 0
    if n_frames == 0 or not any(words):
        return None
    if word_frames is not None and len(word_frames) != len(words):
        word_frames = None
    if word_frames is None:
        total = sum(len(p) for p in words) or 1
        word_frames, done = [], 0
        for phones in words:
            word_frames.append((int(done * n_frames / total), int((done + len(phones)) * n_frames / total)))
            done += len(phones)
    feat = np.zeros((GOPT_SEQ_LEN, GOPT_FEAT_DIM), dtype=np.float64)
    phn = np.full(GOPT_SEQ_LEN, -1, dtype=np.int64)
    slots: list[list[int]] = []
    slot = 0
    for phones, (lo, hi) in zip(words, word_frames):
        if slot + len(phones) > GOPT_SEQ_LEN:
            slots.append([])  # GOPT sees 50 phones at most
            slot = GOPT_SEQ_LEN
            continue
        lo = min(max(0, int(lo)), n_frames - 1)
        hi = min(max(lo + 1, int(hi)), n_frames)
        own = []
        for k, phone in enumerate(phones):
            start = lo + int(k * (hi - lo) / len(phones))
            end = max(start + 1, lo + int((k + 1) * (hi - lo) / len(phones)))
            vec = _lpp_lpr(log_probs[start:min(end, n_frames)], phone, vocab)
            feat[slot] = (vec - LIBRISPEECH_FEAT_MEAN) / LIBRISPEECH_FEAT_STD
            phn[slot] = min(PHONE_TO_ID.get(phone, PHONE_TO_ID["AH"]), 38)
            own.append(slot)
            slot += 1
        slots.append(own)
    if not np.any(phn >= 0):
        return None
    return feat[np.newaxis], phn[np.newaxis], slots


@dataclass(frozen=True)
class GoptWord:
    index: int
    phones: tuple[str, ...]
    # GOPT word heads (0-100); None when the word fell past 50 phones.
    accuracy: float | None
    stress: float | None
    total: float | None
    phone_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class GoptDetail:
    scores: GoptScores
    words: tuple[GoptWord, ...]
    phone_count: int
    truncated: bool
    feature_source: str  # "forced_alignment" | "proportional"


def _to_100(value: float) -> float:
    """SpeechOcean labels are 0-10; GOPT trains on label/5 so raw ~0-2."""
    raw = float(value)
    scaled = raw * 50.0 if abs(raw) <= 2.5 else raw * 10.0
    return round(max(0.0, min(100.0, scaled)), 2)


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
        acc, comp, flu, pro, tot = (
            u1.reshape(-1)[0].item(),
            u2.reshape(-1)[0].item(),
            u3.reshape(-1)[0].item(),
            u4.reshape(-1)[0].item(),
            u5.reshape(-1)[0].item(),
        )
        logger.info(
            "GOPT raw acc=%.3f comp=%.3f flu=%.3f pro=%.3f tot=%.3f",
            acc, comp, flu, pro, tot,
        )
        return GoptScores(
            pronunciation=_to_100(acc),
            completeness=_to_100(comp),
            fluency=_to_100(flu),
            intonation=_to_100(pro),
            overall=_to_100(tot),
        )

    def score_detailed(
        self,
        words: list[str],
        log_probs: np.ndarray,
        vocab: list[str],
        word_frames: list[tuple[int, int]] | None = None,
        initialisms: list[bool] | None = None,
    ) -> GoptDetail | None:
        """Utterance, word and phone heads for the target's spoken `words`
        (english_text.speech_tokens order). Separate from score(), which
        the legacy aspect bars keep using unchanged."""
        if not self.load() or self._model is None:
            return None
        flags = initialisms if initialisms is not None and len(initialisms) == len(words) else [False] * len(words)
        phones = [word_cmu_phones(w, f) for w, f in zip(words, flags)]
        built = build_word_gopt_features(log_probs, phones, vocab, word_frames)
        if built is None:
            return None
        import torch

        feat, phn, slots = built
        device = next(self._model.parameters()).device
        with torch.inference_mode():
            u1, u2, u3, u4, u5, p, w1, w2, w3 = self._model(
                torch.from_numpy(feat).float().to(device),
                torch.from_numpy(phn).to(device),
            )
        p, w1, w2, w3 = (t.reshape(-1).cpu().numpy() for t in (p, w1, w2, w3))
        out = []
        for i, (word_phones_, own) in enumerate(zip(phones, slots)):
            if not own:
                out.append(GoptWord(i, tuple(word_phones_), None, None, None))
                continue
            # SpeechOcean word labels sit on every phone of the word.
            out.append(GoptWord(
                index=i,
                phones=tuple(word_phones_),
                accuracy=_to_100(float(np.mean(w1[own]))),
                stress=_to_100(float(np.mean(w2[own]))),
                total=_to_100(float(np.mean(w3[own]))),
                phone_scores=tuple(_to_100(float(p[s])) for s in own),
            ))
        scores = GoptScores(
            pronunciation=_to_100(u1.reshape(-1)[0].item()),
            completeness=_to_100(u2.reshape(-1)[0].item()),
            fluency=_to_100(u3.reshape(-1)[0].item()),
            intonation=_to_100(u4.reshape(-1)[0].item()),
            overall=_to_100(u5.reshape(-1)[0].item()),
        )
        return GoptDetail(
            scores=scores,
            words=tuple(out),
            phone_count=sum(len(s) for s in slots),
            truncated=any(ph and not s for ph, s in zip(phones, slots)),
            feature_source=(
                "forced_alignment"
                if word_frames is not None and len(word_frames) == len(words)
                else "proportional"
            ),
        )


@lru_cache
def get_english_gopt_service() -> EnglishGoptService:
    return EnglishGoptService(get_settings())
