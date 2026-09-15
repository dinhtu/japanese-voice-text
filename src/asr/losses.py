"""Loss functions for Japanese ASR training.

CR-CTC: Consistency-Regularized CTC for better temporal alignment.
DualCTCLoss: CR-CTC for kana + standard CTC for phoneme InterCTC.
"""

import torch
import torch.nn as nn


class CRCTCLoss(nn.Module):
    """Consistency-Regularized CTC Loss.

    Adds a regularization term that encourages smooth, consistent output
    distributions across adjacent time steps. This mitigates the CTC
    "spike problem" where predictions concentrate on a few frames.

    Loss = CTC(log_probs, targets) + cr_weight * CR(log_probs)

    Reference:
        Huang et al. "CR-CTC: Consistency Regularization on CTC
        for Improved Speech Recognition." ICLR 2025.
    """

    def __init__(
        self,
        blank: int = 0,
        cr_weight: float = 0.1,
        zero_infinity: bool = True,
    ):
        super().__init__()
        self.ctc = nn.CTCLoss(
            blank=blank, reduction="mean", zero_infinity=zero_infinity,
        )
        self.cr_weight = cr_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CR-CTC loss.

        Args:
            log_probs: (T, B, V) log probabilities from model.
            targets: (sum(target_lengths),) concatenated target indices.
            input_lengths: (B,) input sequence lengths.
            target_lengths: (B,) target sequence lengths.

        Returns:
            Scalar loss value.
        """
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        if self.cr_weight > 0 and log_probs.shape[0] > 1:
            cr_loss = self._consistency_regularization(log_probs, input_lengths)
            return ctc_loss + self.cr_weight * cr_loss

        return ctc_loss

    def _consistency_regularization(
        self,
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean KL divergence between adjacent frames."""
        T, B, V = log_probs.shape

        probs_t = log_probs[:-1].exp()       # (T-1, B, V)
        log_probs_t = log_probs[:-1]         # (T-1, B, V)
        log_probs_t1 = log_probs[1:]         # (T-1, B, V)

        kl = (probs_t * (log_probs_t - log_probs_t1)).sum(dim=-1)  # (T-1, B)

        # Mask padded frames
        time_idx = torch.arange(T - 1, device=log_probs.device).unsqueeze(1)
        mask = (time_idx + 1) < input_lengths.unsqueeze(0)

        n_valid = mask.sum().clamp(min=1)
        cr_loss = (kl * mask).sum() / n_valid

        return cr_loss


class DualCTCLoss(nn.Module):
    """Combined loss for dual-output CTC: CR-CTC (kana) + CTC (phoneme InterCTC).

    L = CR-CTC(kana_logits, kana_targets) + inter_weight * CTC(phoneme_logits, phoneme_targets)
    """

    def __init__(
        self,
        blank: int = 0,
        cr_weight: float = 0.1,
        inter_weight: float = 0.3,
    ):
        super().__init__()
        self.kana_ctc = CRCTCLoss(blank=blank, cr_weight=cr_weight)
        self.phoneme_ctc = nn.CTCLoss(
            blank=blank, reduction="mean", zero_infinity=True,
        )
        self.inter_weight = inter_weight

    def forward(
        self,
        kana_log_probs: torch.Tensor,
        kana_targets: torch.Tensor,
        phoneme_log_probs: torch.Tensor,
        phoneme_targets: torch.Tensor,
        input_lengths: torch.Tensor,
        kana_target_lengths: torch.Tensor,
        phoneme_target_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute dual CTC loss.

        Args:
            kana_log_probs: (T, B, kana_V) from final layer.
            kana_targets: Concatenated kana target indices.
            phoneme_log_probs: (T, B, phoneme_V) from intermediate layer.
            phoneme_targets: Concatenated phoneme target indices.
            input_lengths: (B,) output sequence lengths (same for both heads).
            kana_target_lengths: (B,) kana target lengths.
            phoneme_target_lengths: (B,) phoneme target lengths.

        Returns:
            Dict with 'loss' (total), 'kana_loss', 'phoneme_loss'.
        """
        kana_loss = self.kana_ctc(
            kana_log_probs, kana_targets, input_lengths, kana_target_lengths,
        )
        phoneme_loss = self.phoneme_ctc(
            phoneme_log_probs, phoneme_targets, input_lengths, phoneme_target_lengths,
        )

        total = kana_loss + self.inter_weight * phoneme_loss

        return {
            "loss": total,
            "kana_loss": kana_loss,
            "phoneme_loss": phoneme_loss,
        }
