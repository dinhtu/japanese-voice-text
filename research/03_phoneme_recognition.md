# Phoneme-Level Recognition Research (2024--2026)

## 1. XLSR-Wav2Vec2 Phoneme Recognition (Kitahara)

本プロジェクトに最も近い先行実装。

| Item | Detail |
|------|--------|
| Architecture | XLSR-Wav2Vec2 + CTC linear head over phoneme vocabulary |
| Approach | Cross-lingual transfer (53 languages) → target language phoneme labels で fine-tune |
| Applicability | 日本語音素ボキャブラリに差し替えるだけで適用可能 |

- [GitHub](https://github.com/kosuke-kitahara/xlsr-wav2vec2-phoneme-recognition)
- [Colab](https://colab.research.google.com/github/kosuke-kitahara/xlsr-wav2vec2-phoneme-recognition/blob/main/Fine_tuning_XLSR_Wav2Vec2_for_Phoneme_Recognition.ipynb)

---

## 2. IPA-Wav2Vec2-Phoneme-Recognition (Srinath-N-R)

| Item | Detail |
|------|--------|
| Pipeline | Preprocessing → IPA phoneme vocab → CTC training → Evaluation (WER/CER) |
| Architecture | Wav2Vec2 + CTC loss via HuggingFace Trainer API |
| Techniques | Gradient accumulation, mixed-precision, early stopping, custom data collator |

- [GitHub](https://github.com/Srinath-N-R/IPA-Wav2Vec2-Phoneme-Recognition/)

---

## 3. Multilingual Phoneme Recognition Comparison

Wav2Vec2, HuBERT, WavLM の体系的比較 (CTC heads)。

| Finding | Detail |
|---------|--------|
| Cross-lingual transfer | 英語事前学習モデルから他言語の音素認識に転移可能 |
| XLSR-53 | 53 言語で **72% の PER 削減** |

- [GitHub](https://github.com/ASR-project/Multilingual-PR)

---

## 4. Self-Supervised Models for Phoneme Recognition (Interspeech 2024)

フランス語の児童音素認識における比較研究。

| Model | Performance |
|-------|------------|
| Wav2Vec2 | Good |
| HuBERT | Good |
| **WavLM base+** | **Best** (transformer blocks unfrozen 時) |

### Key Findings

- **WavLM base+** が音素認識で最良の性能
- Transformer の**上位層**が ASR / 音素認識タスクに最重要
- ノイズや多様な読み上げタスクへのロバスト性が高い
- [PDF](https://www.isca-archive.org/interspeech_2024/blockmedin24_interspeech.pdf)

---

## 5. Wav2Vec2 Fine-tuning Layer Analysis (2025-03)

Fine-tuning が wav2vec 2.0 の音素正規化にどう作用するかの分析。

| Finding | Detail |
|---------|--------|
| Best layers | **Layers 15--19** が音素認識精度最高 |
| Implication | CNN 凍結 + transformer fine-tune の戦略を支持 |

- [arXiv](https://arxiv.org/html/2503.04814v1)

---

## 6. Wav2Vec2-BERT for CAPT (Interspeech 2025)

- Computer-Assisted Pronunciation Training (発音訓練) 用の音素認識
- Wav2Vec2-BERT を音素レベルでファインチューニング
- [PDF](https://www.isca-archive.org/interspeech_2025/fort25_interspeech.pdf)

---

## 7. Wav2Vec2Phoneme (HuggingFace)

- HuggingFace Transformers 統合の音素認識モデル
- `Wav2Vec2PhonemeCTCTokenizer` で CTC ベースの音素デコーディング
- [Docs](https://huggingface.co/docs/transformers/model_doc/wav2vec2_phoneme)

---

## 8. Apple: Diverse Modeling Units for CTC (Interspeech 2024)

- 音素と書記素の joint training が CTC 精度を相乗的に向上
- 音素ベース CTC システムに直接関連
- [arXiv](https://arxiv.org/abs/2406.03274)

---

## 9. Self-Supervised Models: Foundation Benchmark

### SUPERB Benchmark

15 タスク (phoneme recognition 含む) での音声基盤モデル評価フレームワーク。

| Task | Metric |
|------|--------|
| Phoneme Recognition (PR) | PER |
| ASR | WER |
| Speaker Identification | Accuracy |
| ... (15 tasks total) | ... |

- [arXiv](https://arxiv.org/abs/2404.09385)

### Data2vec 2.0 (Meta)

- wav2vec 2.0 と同等性能を **10.6 倍短い** 事前学習時間で達成
- Contextualized target representations の amortized 計算
- [Meta AI Blog](https://ai.meta.com/blog/ai-self-supervised-learning-data2vec/)

### MCR-Data2vec 2.0 (Interspeech 2023)

- Data2vec 2.0 に consistency regularization を適用（CR-CTC と同じ発想を事前学習に）
- SUPERB benchmark で SOTA
- [arXiv](https://arxiv.org/abs/2306.08463)

---

## 10. Augmentation for Phoneme Recognition

### SpecAugment Best Practices (2024-2025)

標準スタック: **SpecAugment + speed perturbation + noise injection**

| Variant | Innovation |
|---------|-----------|
| Generalized SpecAugment (Gen-SA) | ゼロ以外の値でマスク → ロバスト性向上 |
| Semantic-Aware SpecAugment | wav2vec2 attention heatmap で adaptive masking |
| Phoneme-Aware Augmentation | Phoneme Substitution Matrix で adversarial 変種注入 |

### Speed Perturbation

- HuBERT + Speed Perturbation: WER **21.63%** (augmentation 戦略中最良)

### CR-CTC との関係

CR-CTC の dual-view approach は SpecAugment のマスキングを実質的に置き換え/補完。2 つの異なる augmented view 間の consistency 学習が鍵。

---

## 11. Kana-Level ASR (ひらがな・カタカナ出力 ASR)

### 11.1 ouktlab/espnet_asr_models (2024-2025)

ESPnet ベースの**カタカナ音節 ASR** モデル群。最も体系的なかなレベル ASR 実装。

| Item | Detail |
|------|--------|
| Output unit | カタカナ音節（「コ」「ン」「ニ」「チ」「ワ」） |
| Architecture | Conformer encoder + CTC/Attention hybrid |
| SCT module | Syllable-to-Character Translation — 音節出力から漢字仮名混じり文への変換 |
| Significance | かな単位 ASR が実用的に動作することを実証 |

- [GitHub](https://github.com/ouktlab/espnet_asr_models)

### 11.2 Wav2vec2-xls-r-1b Japanese Hiragana-Katakana (HuggingFace)

| Item | Detail |
|------|--------|
| Model | wav2vec2-xls-r-1b fine-tuned for ひらがな・カタカナ出力 |
| Scale | 1B params（本プロジェクトの 95M と比較して巨大） |
| Output | ひらがな＋カタカナ混在 |

### 11.3 DeepSpeech-based Hiragana ASR

Mozilla DeepSpeech を日本語ひらがな出力用にカスタマイズした実装。ブログ記事レベルだが、ひらがな単位の CTC デコーディングが動作することを確認。

### 11.4 Lenient Evaluation of Japanese Speech Recognition (ACL 2023)

日本語 ASR の評価における**表記揺れ問題**を指摘。同一発話に対して複数の正解表記が存在する（漢字 vs ひらがな vs カタカナ）。**かな出力にすることで、漢字変換由来の表記揺れ問題を根本的に回避できる**ことを示唆。

---

## Key Takeaway

1. **日本語音素レベル出力の ASR モデルは存在しない。** 本プロジェクトの wav2vec2-base + CTC + 日本語音素ボキャブラリは独自性が高い。
2. **かなレベル ASR は先行実装が存在する**が、いずれも「漢字テキスト出力の代替」として位置づけられており、**LLM 対話パイプラインの入力として意図的にかな出力を選択する**アプローチは先行研究が見当たらない。
3. **本プロジェクトの方針**: ひらがな（最終出力）+ 音素（InterCTC 中間出力）の dual-output は新規性が高い。
