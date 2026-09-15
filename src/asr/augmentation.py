"""Audio augmentation for ASR training.

SpecAugment is handled by wav2vec2's built-in mask_time_prob config.
This module provides additional augmentations.
"""

import numpy as np
import torch
import torchaudio


def speed_perturbation(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
    min_speed: float = 0.9,
    max_speed: float = 1.1,
) -> torch.Tensor:
    """Apply random speed perturbation to audio via resampling.

    Args:
        waveform: (1, T) or (T,) audio tensor.
        sample_rate: Audio sample rate.
        min_speed: Minimum speed factor.
        max_speed: Maximum speed factor.

    Returns:
        Speed-perturbed waveform.
    """
    speed = np.random.uniform(min_speed, max_speed)
    if abs(speed - 1.0) < 0.01:
        return waveform

    was_1d = waveform.dim() == 1
    if was_1d:
        waveform = waveform.unsqueeze(0)

    # Resample to simulate speed change: speed>1 = faster = shorter
    orig_freq = int(sample_rate * speed)
    augmented = torchaudio.functional.resample(waveform, orig_freq, sample_rate)

    if was_1d:
        augmented = augmented.squeeze(0)
    return augmented


def add_noise(
    waveform: torch.Tensor,
    snr_db: float | None = None,
    min_snr_db: float = 10.0,
    max_snr_db: float = 20.0,
) -> torch.Tensor:
    """Add Gaussian noise at a random SNR level.

    Args:
        waveform: Audio tensor.
        snr_db: Specific SNR in dB. If None, randomly sampled from [min, max].
        min_snr_db: Minimum SNR in dB.
        max_snr_db: Maximum SNR in dB.

    Returns:
        Noisy waveform.
    """
    if snr_db is None:
        snr_db = np.random.uniform(min_snr_db, max_snr_db)

    signal_power = waveform.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return waveform + noise
