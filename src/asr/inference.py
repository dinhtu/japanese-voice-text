"""Reusable kana ASR inference (shared by the CLI and the FastAPI service).

Holds the model/feature-extractor once and transcribes audio files:

    recognizer = KanaRecognizer("models/checkpoints/best.pt")
    result = recognizer.transcribe("test.wav")
    print(result.kana)
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor

from src.asr.kana_vocab import KanaVocab
from src.asr.model import load_checkpoint
from src.asr.phoneme_vocab import PhonemeVocab

DEFAULT_PRETRAINED = "reazon-research/japanese-wav2vec2-base"
TARGET_SAMPLE_RATE = 16_000


def resolve_device(name: str | None = None) -> torch.device:
    """Pick the best available device (cuda > mps > cpu) unless one is given."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _decode(path: str | Path) -> tuple[torch.Tensor, int]:
    """Decode to a (channels, samples) float tensor.

    torchaudio 2.9+ delegates to torchcodec, which needs FFmpeg shared libraries
    that are often missing on Windows; soundfile handles WAV/FLAC without them.
    """
    errors = []
    for backend in (_decode_torchaudio, _decode_soundfile):
        try:
            return backend(path)
        except Exception as e:  # noqa: BLE001 — backend errors vary by format
            errors.append(f"{backend.__name__}: {e}")
    raise RuntimeError(f"Failed to decode audio file ({'; '.join(errors)})")


def _decode_torchaudio(path: str | Path) -> tuple[torch.Tensor, int]:
    return torchaudio.load(str(path))


def _decode_soundfile(path: str | Path) -> tuple[torch.Tensor, int]:
    data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono 16kHz float array.

    Raises:
        RuntimeError: If the file cannot be decoded.
    """
    waveform, sr = _decode(path)

    if waveform.numel() == 0:
        raise RuntimeError("Audio file contains no samples")

    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(waveform)
        sr = TARGET_SAMPLE_RATE

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0).numpy(), sr


def swd_decode(logits: torch.Tensor, window: int = 1) -> torch.Tensor:
    """Spike Window Decoding (SWD) for CTC.

    Instead of argmax over all frames, only considers frames around CTC spikes.
    This reduces computation and can improve accuracy.

    Reference: ICASSP 2025 — "Spike Window Decoding"

    Args:
        logits: (1, T, V) raw logits from the model.
        window: Number of frames on each side of the spike to consider.

    Returns:
        Predicted token indices (1D tensor).
    """
    # Step 1: Find spikes (frames where non-blank is most likely)
    probs = logits.squeeze(0).softmax(dim=-1)  # (T, V)
    blank_prob = probs[:, 0]  # (T,)

    # A spike is where blank probability drops below 0.5
    is_spike = blank_prob < 0.5  # (T,)

    if not is_spike.any():
        # No spikes: return standard greedy decode
        return logits.squeeze(0).argmax(dim=-1)

    # Step 2: Expand spike positions by window
    T = probs.shape[0]
    spike_indices = is_spike.nonzero(as_tuple=True)[0]

    active = torch.zeros(T, dtype=torch.bool, device=logits.device)
    for idx in spike_indices:
        start = max(0, idx.item() - window)
        end = min(T, idx.item() + window + 1)
        active[start:end] = True

    # Step 3: Decode only active frames, set inactive to blank (0)
    pred_ids = torch.zeros(T, dtype=torch.long, device=logits.device)
    pred_ids[active] = logits.squeeze(0)[active].argmax(dim=-1)

    return pred_ids


def _ctc_collapse_with_onsets(
    pred_ids: list[int], blank_idx: int = 0
) -> tuple[list[int], list[int]]:
    """CTC greedy collapse (skip blanks, merge consecutive repeats) that also
    records, for each emitted token, the frame index where its run began.

    Mirrors KanaVocab.decode()'s collapse rule exactly (same skip-blank,
    skip-repeat logic), so the token sequence this returns lines up
    character-for-character with the `kana` string that decode() produces
    from the same `pred_ids` -- that 1:1 correspondence is what lets each
    character of RecognitionResult.kana be paired with an onset frame.
    """
    tokens: list[int] = []
    onsets: list[int] = []
    prev: int | None = None
    for frame_idx, idx in enumerate(pred_ids):
        if idx == blank_idx:
            prev = idx
            continue
        if idx == prev:
            continue
        tokens.append(idx)
        onsets.append(frame_idx)
        prev = idx
    return tokens, onsets


def _char_spans_from_pred_ids(
    pred_ids: list[int], kana: str, num_frames: int, duration: float
) -> list[tuple[float, float]] | None:
    """(start_sec, end_sec) per character of `kana`, from CTC onset frames.

    Each character's window runs from its own onset to the next
    character's onset (or the clip's end for the last character) -- the
    stretch of audio during which the model was "on" that symbol before
    moving to the next one. `num_frames` is the model's own frame count for
    this clip (kana_logits.shape[1]); dividing `duration` by it gives the
    (roughly uniform, thanks to the encoder's fixed conv stride) time each
    frame represents.

    Returns None instead of possibly-misaligned data if the collapsed
    token count doesn't match len(kana) one-for-one -- should not happen
    (same collapse rule as KanaVocab.decode()), but this is speech ASR:
    fail soft rather than hand out timing that might not line up.
    """
    if num_frames == 0:
        return None
    tokens, onset_frames = _ctc_collapse_with_onsets(pred_ids)
    if len(tokens) != len(kana):
        return None
    frame_time = duration / num_frames
    onset_times = [idx * frame_time for idx in onset_frames]
    spans: list[tuple[float, float]] = []
    for i, start in enumerate(onset_times):
        end = onset_times[i + 1] if i + 1 < len(onset_times) else duration
        spans.append((start, max(end, start)))
    return spans


@dataclass
class RecognitionResult:
    """Output of a single transcription."""

    kana: str
    duration: float
    inference_time: float
    phonemes: str | None = None
    char_spans: list[tuple[float, float]] | None = None
    """(start_sec, end_sec) per character of `kana`, from the CTC decode's
    own frame timing. Only populated when transcribe(with_timing=True) was
    used; see _char_spans_from_pred_ids for how it's derived and when it
    falls back to None instead."""
    aligned_char_spans: list[tuple[float, float]] | None = None
    """(start_sec, end_sec) per character of the *target* hiragana passed
    to transcribe(align_to=...), from CTC forced alignment of that string
    onto the kana head. Preferred over char_spans for mora pitch windows
    because it timestamps the sentence the learner was asked to read, not
    whatever greedy decode emitted."""
    phoneme_probs: np.ndarray | None = None
    """(T, phoneme_V) softmax over the InterCTC phoneme head. Populated
    when with_timing or with_phonemes is set -- used for GOP scoring.
    Blank is index 0, same as PhonemeVocab."""

    @property
    def rtf(self) -> float:
        """Real-time factor: inference time / audio duration."""
        return self.inference_time / self.duration if self.duration > 0 else 0.0


class KanaRecognizer:
    """Loads a DualCTC checkpoint once and transcribes audio to kana."""

    def __init__(
        self,
        checkpoint: str | Path,
        pretrained: str = DEFAULT_PRETRAINED,
        inter_ctc_layer: int | None = None,
        device: str | torch.device | None = None,
        fp16: bool = False,
    ):
        self.device = device if isinstance(device, torch.device) else resolve_device(device)
        self.fp16 = fp16

        model = load_checkpoint(str(checkpoint), pretrained, inter_ctc_layer=inter_ctc_layer)
        if fp16:
            model.half()
        model.to(self.device)
        model.eval()
        self.model = model
        self._compute_device = self.device

        # The checkpoint knows which encoder it was fine-tuned from.
        self.pretrained = getattr(model.encoder.config, "_name_or_path", pretrained) or pretrained
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained)
        self.kana_vocab = KanaVocab()
        self.phoneme_vocab = PhonemeVocab()

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        swd: bool = False,
        swd_window: int = 1,
        with_phonemes: bool = False,
        with_timing: bool = False,
        align_to: str | None = None,
    ) -> RecognitionResult:
        """Transcribe an audio file (or a pre-loaded 16kHz mono array) to kana.

        `align_to` is the target hiragana to force-align on the same forward
        pass (see RecognitionResult.aligned_char_spans).
        """
        self.ensure_on_device()
        if isinstance(audio, np.ndarray):
            audio_array, sr = audio, TARGET_SAMPLE_RATE
        else:
            audio_array, sr = load_audio(audio)

        duration = len(audio_array) / sr

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        if self.fp16:
            input_values = input_values.half()

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_values, attention_mask=attention_mask)
            kana_logits = outputs["kana_logits"]  # (1, T, kana_V)

            if swd:
                kana_pred_ids = swd_decode(kana_logits, window=swd_window)
            else:
                kana_pred_ids = kana_logits.squeeze(0).argmax(dim=-1)
        inference_time = time.perf_counter() - t0

        phonemes = None
        phoneme_probs = None
        if with_phonemes or with_timing:
            phoneme_logits = outputs["phoneme_logits"].squeeze(0).float().cpu()
            phoneme_probs = torch.softmax(phoneme_logits, dim=-1).numpy()
            if with_phonemes:
                phonemes = self.phoneme_vocab.decode(phoneme_logits.argmax(dim=-1).tolist())

        kana_ids_list = kana_pred_ids.tolist()
        kana = self.kana_vocab.decode(kana_ids_list)

        char_spans = None
        if with_timing:
            char_spans = _char_spans_from_pred_ids(
                kana_ids_list, kana, kana_logits.shape[1], duration
            )

        aligned_char_spans = None
        if align_to:
            from src.asr.force_align import align_kana

            aligned_char_spans = align_kana(
                kana_logits, self.kana_vocab, align_to, duration
            )

        del outputs, kana_logits, kana_pred_ids, input_values, attention_mask
        return RecognitionResult(
            kana=kana,
            duration=duration,
            inference_time=inference_time,
            phonemes=phonemes,
            char_spans=char_spans,
            aligned_char_spans=aligned_char_spans,
            phoneme_probs=phoneme_probs,
        )

    def ensure_on_device(self) -> None:
        self.model.to(self._compute_device)
        self.device = self._compute_device

    def offload(self) -> None:
        target = self._compute_device
        kind = getattr(target, "type", None) or str(target).split(":")[0]
        if kind != "cuda":
            return
        from app.core.vram import module_to_cpu

        module_to_cpu(self.model)
        self.model.eval()
