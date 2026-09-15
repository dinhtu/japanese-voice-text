"""Wav2Vec2 + Dual CTC model for Japanese kana + phoneme ASR.

Architecture:
    wav2vec2 encoder (frozen CNN + fine-tuned transformer)
    ├── Layer inter_ctc_layer → InterCTC head → phoneme output (auxiliary)
    └── Final layer → CTC head → kana output (primary)

Supports both base (12 layers, hidden=768) and large (24 layers, hidden=1024).
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from src.asr.kana_vocab import KanaVocab
from src.asr.phoneme_vocab import PhonemeVocab


def default_inter_ctc_layer(num_hidden_layers: int) -> int:
    """Middle layer of the encoder, suitable for InterCTC."""
    return num_hidden_layers // 2


class DualCTCModel(nn.Module):
    """Wav2Vec2 encoder with dual CTC heads for kana and phoneme output.

    Implements the Apple Diverse Modeling Units approach (Interspeech 2024)
    adapted for Japanese: phoneme CTC at an intermediate layer, kana CTC
    at the final layer.
    """

    def __init__(
        self,
        encoder: Wav2Vec2Model,
        kana_vocab_size: int,
        phoneme_vocab_size: int,
        hidden_size: int = 768,
        inter_ctc_layer: int = 6,
    ):
        super().__init__()
        self.encoder = encoder
        self.inter_ctc_layer = inter_ctc_layer

        # Final CTC head for kana output
        self.kana_head = nn.Linear(hidden_size, kana_vocab_size)

        # Intermediate CTC head for phoneme output
        self.phoneme_head = nn.Linear(hidden_size, phoneme_vocab_size)

    def forward(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning both kana and phoneme logits.

        Args:
            input_values: (B, T) raw audio waveform.
            attention_mask: Optional attention mask.

        Returns:
            Dict with keys:
                - kana_logits: (B, T', kana_vocab_size)
                - phoneme_logits: (B, T', phoneme_vocab_size)
        """
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Final layer → kana CTC
        final_hidden = outputs.last_hidden_state
        kana_logits = self.kana_head(final_hidden)

        # Intermediate layer → phoneme CTC
        inter_hidden = outputs.hidden_states[self.inter_ctc_layer]
        phoneme_logits = self.phoneme_head(inter_hidden)

        return {
            "kana_logits": kana_logits,
            "phoneme_logits": phoneme_logits,
        }

    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output sequence lengths after wav2vec2 CNN downsampling."""
        return self.encoder._get_feat_extract_output_lengths(input_lengths)


def create_model(
    pretrained: str = "reazon-research/japanese-wav2vec2-base",
    freeze_feature_extractor: bool = True,
    mask_time_prob: float = 0.05,
    inter_ctc_layer: int | None = None,
) -> DualCTCModel:
    """Create a DualCTCModel configured for Japanese kana + phoneme output.

    Args:
        pretrained: HuggingFace model ID (base or large).
        freeze_feature_extractor: Whether to freeze the CNN feature extractor.
        mask_time_prob: SpecAugment time masking probability (training only).
        inter_ctc_layer: Encoder layer for InterCTC. None = auto (num_layers // 2).
    """
    kana_vocab = KanaVocab()
    phoneme_vocab = PhonemeVocab()

    config = Wav2Vec2Config.from_pretrained(pretrained)
    config.mask_time_prob = mask_time_prob

    if inter_ctc_layer is None:
        inter_ctc_layer = default_inter_ctc_layer(config.num_hidden_layers)

    encoder = Wav2Vec2Model.from_pretrained(pretrained, config=config)

    if freeze_feature_extractor:
        encoder.feature_extractor._freeze_parameters()

    model = DualCTCModel(
        encoder=encoder,
        kana_vocab_size=kana_vocab.size,
        phoneme_vocab_size=phoneme_vocab.size,
        hidden_size=config.hidden_size,
        inter_ctc_layer=inter_ctc_layer,
    )

    return model


def load_checkpoint(
    checkpoint_path: str,
    pretrained: str = "reazon-research/japanese-wav2vec2-base",
    inter_ctc_layer: int | None = None,
) -> DualCTCModel:
    """Load a fine-tuned DualCTCModel from checkpoint.

    Supports both legacy (raw state_dict) and new (metadata dict) formats.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # New format: {"model_state_dict": ..., "pretrained": ..., "inter_ctc_layer": ...}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        pretrained = checkpoint.get("pretrained", pretrained)
        if inter_ctc_layer is None:
            inter_ctc_layer = checkpoint.get("inter_ctc_layer")
        state_dict = checkpoint["model_state_dict"]
    else:
        # Legacy format: raw state_dict
        state_dict = checkpoint

    model = create_model(
        pretrained=pretrained,
        mask_time_prob=0.0,
        inter_ctc_layer=inter_ctc_layer,
    )
    model.load_state_dict(state_dict, strict=False)
    model.float()  # Ensure FP32 (weights may be saved in BF16)
    return model
