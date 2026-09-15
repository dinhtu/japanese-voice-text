# Implications for This Project

調査結果から得られた、本プロジェクトへの具体的な示唆。

---

## 1. Architecture: Dual-Output CTC (Kana + Phoneme)

### 設計思想

現代の ASR (Whisper 等) は語彙的補正が強すぎ、その出力を LLM に渡すと誤りが増幅される。
ASR は「聞こえたものを忠実に報告」し、解釈は frozen LLM に委ねるべき。

- **ひらがな出力**: 語彙的補正ゼロ、LLM が直接読める
- **音素出力**: さらに低レベル、vsr-test と共有可能

### Apple Diverse Modeling Units (Interspeech 2024) の適用

Apple の研究では phoneme + grapheme の joint CTC training が相乗効果を発揮。
日本語への適用:

```
wav2vec2 encoder
  ├── Layer 6 → InterCTC (phoneme, 38 tokens) — 低レベル正則化
  └── Layer 12 → Final CTC (kana, ~85 tokens) — 最終出力
```

### 根拠

- ouktlab (ESPnet) がカタカナ音節 ASR の実用性を実証済み
- wav2vec2-xls-r-1b でひらがな CTC が動作する先行実装あり
- ACL 2023 の表記揺れ研究: かな出力で漢字変換由来の問題を回避
- **LLM 対話入力として意図的にかな出力する先行研究は存在しない** → 新規性

### CR-CTC is well-validated

`losses.py` の CR-CTC は ICLR 2025 採択済み。純粋 CTC で初めて Transducer/AED に匹敵する性能を達成した手法。

---

## 2. Technical Decisions

### 2.1 Base model

`reazon-research/japanese-wav2vec2-base` (35,000h 日本語事前学習, Apache 2.0)

### 2.2 Training data

ReazonSpeech v2 (CDLA-Sharing-1.0)
- Section 3.5: 学習済みモデル (Results) への制限なし
- Section 3.3: 商用利用制限の追加を禁止
- → 学習済みモデルの商用利用 OK

### 2.3 Freeze strategy

- CNN feature extractor: **Frozen** (必須)
- Transformer layers: **All fine-tuned** (wav2vec2-base は 12 層)
- Layer analysis (arXiv 2503.04814): Layers 15--19 が音素認識最良 → 12 層モデルは全層 fine-tune が妥当

### 2.4 InterCTC

- 中間層 (layer 6) に音素 CTC 補助損失
- 10-20% の PER 相対改善が期待
- Apple の Diverse Modeling Units アプローチと組み合わせ

### 2.5 Spike Window Decoding (SWD)

- 推論時に CTC スパイクの左右 1 フレームのみ使用
- **2.17x 高速化** (追加学習なし)
- ICASSP 2025

---

## 3. Deployment Path

### M2 Air 16GB

```
Phase 1 (Current): PyTorch training on RTX 2080
Phase 2: ONNX export + INT8 quantization
Phase 3: sherpa-onnx integration for cross-platform
```

### Model size estimate

```
wav2vec2-base + dual CTC head (~85 kana + 38 phoneme)
  FP32: ~380MB
  INT8: ~95MB  ← M2 Air で余裕
```

---

## 4. Benchmark Targets

| Metric | Target | Reference |
|--------|--------|-----------|
| Kana Error Rate | <15% | XLSR-53 baseline 相当 |
| Phoneme Error Rate (InterCTC) | <15% | 中間層補助メトリクス |
| Inference latency (5s audio) | <500ms | wav2vec2 CTC non-autoregressive |
| Model size (INT8) | <100MB | wav2vec2-base |
| RTF on M2 Air | <0.1 | Non-autoregressive CTC |

---

## 5. Research Opportunities

### 5.1 Kana ASR for LLM dialogue is novel

「語彙的補正を排除し、かな列を LLM に直接入力する」パイプラインは先行研究なし。

### 5.2 Dual-output (kana + phoneme) CTC

Apple の Diverse Modeling Units を日本語 wav2vec2 に適用した例はない。

### 5.3 AVSR integration (asr-test + vsr-test)

- 音素レベル ASR + lip reading の組み合わせは未開拓
- Whisper-Flamingo (gated cross-attention) が参考

### 5.4 CR-CTC + Japanese wav2vec2

- CR-CTC は Zipformer で評価されているが、wav2vec2 での評価例は少ない
- 日本語 wav2vec2 + CR-CTC + dual output の組み合わせは新規性が高い

---

## 6. Action Items (Priority Order)

1. **Dual-output model**: kana (final) + phoneme (InterCTC) の実装
2. **Kana vocabulary**: ひらがな ~83 文字 + blank の語彙定義
3. **Kana converter**: fugashi でテキスト → ひらがな変換
4. **InterCTC loss**: layer 6 補助損失の統合
5. **SWD 実装**: 推論高速化 (`scripts/03_infer.py`)
6. **Training pipeline**: dual-output 対応の学習ループ
7. **ONNX export**: デプロイメント準備
