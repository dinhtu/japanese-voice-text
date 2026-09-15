# Training Report: wav2vec2-large 100h (2026-02-23)

## Summary

| Item | Value |
|---|---|
| Model | `reazon-research/japanese-wav2vec2-large` + Dual CTC heads |
| Parameters | 315.6M |
| Training data | ReazonSpeech small (100h, 62,047 samples) |
| Best val_loss | **1.3588** (step 12,500, epoch 7) |
| Final train_loss | 1.3289 (epoch 15) |
| Training time | 5h 12m (09:32–14:44 UTC) |
| Cost | ~$6.2 (A100 80GB @ $1.19/hr) |

## Evaluation Results (JSUT-BASIC5000)

| Model | KER | PER |
|---|---|---|
| **large (100h, this run)** | **7.5%** | **6.6%** |
| base (100h, prev run) | 17.9% | 19.6% |
| improvement | **-58%** | **-66%** |

Evaluated at intermediate best (val_loss=1.3588, step 12,500).
Final best.pt has the same val_loss — no improvement after epoch 7.

## Infrastructure

| Item | Value |
|---|---|
| Provider | RunPod |
| GPU | NVIDIA A100 80GB PCIe |
| Container disk | 200 GB |
| Pod ID | `vda5z0d1ez1glm` |
| SSH | `213.173.105.10:17144` |

## Hyperparameters

| Parameter | Value |
|---|---|
| Pretrained | `reazon-research/japanese-wav2vec2-large` |
| Encoder layers | 24 (hidden=1024) |
| InterCTC layer | 12 (auto: num_layers // 2) |
| Batch size | 16 (× grad_accum=2 = effective 32) |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Precision | BF16 |
| Epochs | 15 |
| SpecAugment | mask_time_prob=0.05 |
| Speed perturbation | 0.0 (disabled — causes CPU bottleneck) |
| Noise augmentation | 0.0 (disabled) |
| num_workers | 0 (Arrow mmap contention) |
| Freeze CNN | Yes (feature_extractor frozen) |
| Loss weights | kana CTC λ=0.7, phoneme InterCTC λ=0.3 |

## Training Curve

### Train Loss (per epoch)

```
Epoch  avg_loss
  1     9.1736  ████████████████████████████████████████████████████
  2     1.8307  ██████████
  3     1.6104  █████████
  4     1.5084  ████████
  5     1.4503  ████████
  6     1.4067  ████████
  7     1.3755  ███████
  8     1.3592  ███████
  9     1.3429  ███████
 10     1.3365  ███████
 11     1.3295  ███████
 12     1.3316  ███████
 13     1.3276  ███████
 14     1.3282  ███████
 15     1.3289  ███████
```

### Val Loss (best checkpoints)

```
Step     val_loss   Epoch
  500     4.6010     1     ████████████████████████████████████
 1000     4.1051     1     ████████████████████████████████
 1500     2.2210     1     █████████████████
 2000     1.7998     1     ██████████████
 2500     1.6617     2     █████████████
 3000     1.5612     2     ████████████
 3500     1.5561     2     ████████████
 4000     1.4949     3     ███████████
 5000     1.4673     3     ███████████
 5500     1.4477     3     ███████████
 7000     1.4322     4     ███████████
 8500     1.4028     5     ██████████
10500     1.4021     6     ██████████
11000     1.3951     6     ██████████
11500     1.3729     6     ██████████
12500     1.3588     7     ██████████  ← best
```

Val loss plateaued after epoch 7. Epochs 8-15 showed no improvement (~1.36-1.40).

## Checkpoints

| File | Path | Size | Description |
|---|---|---|---|
| best (final) | `models/checkpoints/best_large_final.pt` | 603 MB | Best val_loss=1.3588, BF16 |
| best (interim) | `models/checkpoints/best_large.pt` | 603 MB | Same model (downloaded mid-training) |
| base best | `models/checkpoints/best.pt` | 361 MB | Base model, val_loss=2.13 |

### Checkpoint format

```python
{
    "model_state_dict": OrderedDict,   # BF16 weights
    "pretrained": "reazon-research/japanese-wav2vec2-large",
    "inter_ctc_layer": 12,
}
```

## Kana Error Analysis (JSUT eval, large model)

### Per-group KER

| Group | KER | Substitutions | Deletions | Insertions |
|---|---|---|---|---|
| 長音 (chōon) | 84.1% | 401 | 429 | 5 |
| 半濁音 (handakuten) | 30.7% | 144 | 90 | 2 |
| や行 | 24.5% | 263 | 502 | 27 |
| unknown | 23.5% | 2,402 | 14,128 | 1,461 |
| 小書き (small kana) | 23.2% | 460 | 1,517 | 125 |
| 母音 (vowels) | 22.2% | 1,663 | 3,920 | 255 |
| わ行 | 20.8% | 796 | 1,720 | 168 |
| 濁音 (dakuten) | 20.2% | 2,334 | 1,555 | 110 |
| は行 | 11.6% | 333 | 505 | 83 |
| さ行 | 11.5% | 680 | 1,065 | 69 |
| な行 | 10.2% | 802 | 745 | 80 |
| た行 | 9.5% | 992 | 871 | 84 |
| か行 | 9.2% | 874 | 872 | 118 |
| ま行 | 8.9% | 393 | 296 | 49 |
| ら行 | 8.9% | 677 | 477 | 24 |

### Top kana confusion pairs

| Ref → Hyp | Count |
|---|---|
| ご → も | 218 |
| い → え | 214 |
| を → う | 212 |
| け → き | 189 |
| が → は | 175 |
| (del) → う | 154 |
| お → う | 149 |
| ー → う | 142 |
| え → い | 123 |
| ど → と | 117 |

### Top phoneme confusion pairs

| Ref → Hyp | Count |
|---|---|
| g → w | 533 |
| e → i | 456 |
| a → o | 429 |
| g → m | 389 |
| u → o | 381 |
| o → u | 352 |
| n → m | 323 |
| u → i | 274 |
| d → t | 265 |
| g → n | 228 |

## Key Observations

1. **Large >> Base**: KER 7.5% vs 17.9% (2.4x better), PER 6.6% vs 19.6% (3x better)
2. **Early convergence**: Best val_loss at epoch 7/15. Later epochs didn't improve.
3. **長音 (long vowel) is hardest**: 84% error rate. The model struggles with ー.
4. **Voiced/unvoiced confusion**: ご→も, が→は, ど→と — dakuten discrimination is weak.
5. **Vowel confusion**: い↔え, お↔う — close vowels are mixed up.
6. **No overfitting**: Train loss (1.33) ≈ val loss (1.36) at end. More data would help.

## Recommendations for Next Steps

1. **More data (1000h)**: Val loss not overfit → primary bottleneck is data. ReazonSpeech medium split.
2. **Longer warmup / lower LR**: Try lr=5e-5 with warmup to reduce early instability (epoch 1 loss=9.17).
3. **SpecAugment tuning**: Increase mask_time_prob (currently 0.05) for better generalization.
4. **Fix speed perturbation**: Current CPU bottleneck makes it unusable. Consider GPU-side implementation.
5. **長音 post-processing**: Rule-based correction for ー → う confusion.
6. **Evaluation on ReazonSpeech test set**: JSUT is read speech only; need conversational speech eval.
