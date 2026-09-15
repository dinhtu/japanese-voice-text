# CTC Loss Improvements and Decoding Advances (2024--2026)

## 1. CR-CTC: Consistency Regularization on CTC

**本プロジェクトの `losses.py` に直接関連。**

| Item | Detail |
|------|--------|
| Paper | CR-CTC: Consistency regularization on CTC for improved speech recognition |
| Venue | **ICLR 2025** |
| Authors | Zengwei Yao, Wei Kang, et al. (Daniel Povey group) |
| Encoder | Zipformer |

### Mechanism

1. **Self-distillation**: 異なる SpecAugment ビューから生成した 2 つの CTC 分布間の一貫性を強制
2. **Masked prediction**: マスク領域の文脈的表現学習
3. **Peak suppression**: 極端に peaky な CTC 分布を抑制しオーバーフィッティング低減

### Results

| Dataset | Model | WER |
|---------|-------|-----|
| LibriSpeech test-clean/other | Zipformer-L + Pruned Transducer + CR-CTC | 1.88% / 3.95% |
| Aishell-1 | Zipformer-S + CR-CTC | > CTC/AED with Zipformer-M |

- 純粋 CTC で初めて Transducer / CTC-AED に匹敵する性能
- バッチサイズ半分、エポック半分でもバニラ CTC を上回る
- [arXiv](https://arxiv.org/abs/2410.05101) | [ICLR](https://openreview.net/forum?id=CIs9x2ZRgh) | [PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/432540ba4d35a2010ebfbaa3d5cb4640-Paper-Conference.pdf)

---

## 2. Less Peaky CTC Forced Alignment by Label Priors

| Item | Detail |
|------|--------|
| Paper | Less Peaky and More Accurate CTC Forced Alignment by Label Priors |
| Venue | **ICASSP 2024** |

### Key Idea

Label priors により blank の少ないアライメントパスを強化。CTC モデルが less peaky な posterior を出力し、トークンの onset だけでなく offset も正確に予測。

### Results

- 標準 CTC より **12--40%** の音素/単語境界エラー低減 (TIMIT, Buckeye)
- MFA (Montreal Forced Aligner) に匹敵する性能でありながら、学習パイプラインがシンプルかつ高速
- **TorchAudio でレシピとモデルを公開済み**
- [arXiv](https://arxiv.org/abs/2406.02560)

---

## 3. LCS-CTC: Leveraging Soft Alignments (2025)

- ソフトアライメントを活用した音声転写のロバスト性向上
- [arXiv](https://arxiv.org/html/2508.03937)

---

## 4. Self-Conditioned CTC (Nozaki et al.)

CR-CTC の先駆的研究。

| Item | Detail |
|------|--------|
| Paper | Relaxing the Conditional Independence Assumption of CTC-based ASR |
| Venue | Interspeech 2021 |
| Authors | Jumon Nozaki, Tatsuya Komatsu |

### Method

中間層の CTC 予測を次の層の入力に加算して条件付け。CTC の条件付き独立性仮定を緩和。

### Results

- WSJ で **20% 以上の WER 相対改善**
- デコーディング速度は **30 倍以上** 維持 (vs autoregressive)
- [arXiv](https://arxiv.org/abs/2104.02724) | [ISCA](https://www.isca-archive.org/interspeech_2021/nozaki21_interspeech.html)

---

## 5. Intermediate CTC Loss / InterCTC

### Inter-layer Attention CTC (Interspeech 2024)

- 全 Transformer 層にアテンション重み付き CTC 損失を適用
- 10-20% の相対 WER/CER 改善
- 24 層 → 12 層プルーニングでも性能維持（追加再学習不要）
- [PDF](https://www.isca-archive.org/interspeech_2024/hojo24_interspeech.pdf)

### LLM-Based Intermediate Loss (LAIL)

- 中間層出力を LLM の embedding space にマッピング
- Causal LM loss を補助損失として使用
- [arXiv](https://arxiv.org/html/2506.22846)

---

## 6. Spike Window Decoding (SWD, 2025-01)

| Item | Detail |
|------|--------|
| Paper | Breaking Through the Spike: Spike Window Decoding for Accelerated and Precise ASR |
| Venue | ICASSP 2025 |

### Key Finding

CTC スパイクの左右 **1 フレーム** の隣接フレームに意味情報が集中。

### Results

- CTC greedy デコーディングの **2.17 倍高速化**
- 精度も向上 (AISHELL-1)
- [arXiv](https://arxiv.org/abs/2501.03257)

---

## 7. FlexCTC (2025-08)

GPU 完全バッチ処理の CTC ビームデコーディング。

| Feature | Detail |
|---------|--------|
| Implementation | Python / PyTorch (no C++/CUDA required) |
| GPU optimization | CUDA Graphs で kernel launch overhead 排除 |
| Contextual | N-gram LM fusion + phrase-level boosting |
| Performance | CPU-GPU sync 排除で高速化 |

- [arXiv](https://arxiv.org/abs/2508.07315)

---

## 8. All-in-One ASR (ASRU 2025)

| Item | Detail |
|------|--------|
| Paper | All-in-One ASR: Unifying Encoder-Decoder Models of CTC, Attention, and Transducer |
| Authors | Takafumi Moriya et al. |

- CTC + Attention + Transducer を単一モデルに統合
- Dual-mode (offline / streaming)
- Multi-mode joiner で個別最適化モデルに匹敵する性能を、より小さなフットプリントで実現
- [arXiv](https://arxiv.org/abs/2512.11543)

---

## 9. Apple: Diverse Modeling Units for CTC (Interspeech 2024)

- 音素と書記素の joint training が CTC 精度を相乗的に向上
- 音素ベース CTC システムに直接関連
- [arXiv](https://arxiv.org/abs/2406.03274)

---

## 10. Uncertainty-Aware Self-Training for CTC (AAAI 2025)

- CTC モデルの sequence-level 不確実性推定で擬似ラベルをフィルタリング
- Semi-supervised / low-resource シナリオで有効
- [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34610)

---

## 11. Encoder Architecture Advances

### Zipformer (ICLR 2024 Oral)

CR-CTC と同じ研究グループが開発。

| Feature | Detail |
|---------|--------|
| Structure | U-Net-like multi-rate |
| Normalization | BiasNorm (length information 保持) |
| Activation | SwooshR / SwooshL |
| Optimizer | ScaledAdam |
| Speed | 前世代エンコーダの 50%+ 高速推論 |

- [arXiv](https://arxiv.org/abs/2310.11230)

### E-Branchformer

- 並列 self-attention + convolution branches
- LibriSpeech: **1.81% / 3.65%** WER (外部データなし)
- OWSM v3.1 で採用
- [arXiv](https://arxiv.org/abs/2210.00077)

### Samba-ASR (2025-01)

- **Mamba (SSM)** ベースの初の SOTA ASR
- LibriSpeech Clean: **1.17%** WER
- シーケンス長に対して線形スケーリング（Transformer は二次）
- [arXiv](https://arxiv.org/abs/2501.02832)

### FastConformer (NVIDIA)

- 8x subsampling、depthwise separable convolutions
- Conformer の **2.8 倍高速**、Whisper-large-v3 の **7-10 倍高速**
- [NVIDIA](https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-06-07-fast-conformer/)
