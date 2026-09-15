# Japanese ASR Models and Datasets (2024--2026)

## 1. ReazonSpeech

### ReazonSpeech v2.0 Corpus

- **Size**: 35,000+ hours of natural Japanese speech (terrestrial TV broadcasts)
- **License**: Apache 2.0
- **Format**: 16kHz FLAC, 7.2M clips
- **Links**: [HuggingFace](https://huggingface.co/datasets/reazon-research/reazonspeech) | [GitHub](https://github.com/reazon-research/ReazonSpeech) | [Paper](https://research.reazon.jp/_static/reazonspeech_nlp2023.pdf)

### ReazonSpeech v2.1 / k2-v2 (2024-08)

Japanese ASR のエッジデプロイにおける最重要リリース。

| Item | Detail |
|------|--------|
| Architecture | Zipformer transducer (Next-gen Kaldi / k2) |
| Parameters | 159M |
| Training | 35,000h ReazonSpeech v2.0 |
| Format | ONNX (INT8 quantized) |
| Deploy | GPU 不要、Linux/macOS/Windows/iOS/Android |
| Performance | JSUT / CV8 / TEDxJP で SOTA |

- Whisper-Large-v3 の 10 分の 1 サイズ
- ストリーミング版と日英バイリンガル版を開発中
- [Blog](https://research.reazon.jp/blog/2024-08-01-ReazonSpeech.html) | [HuggingFace](https://huggingface.co/reazon-research/reazonspeech-k2-v2)

### Japanese Wav2Vec2 Models

| Model | Pretraining | Params | License |
|-------|------------|--------|---------|
| `reazon-research/japanese-wav2vec2-base` | 35Kh ReazonSpeech v2.0 | ~95M | Apache 2.0 |
| `reazon-research/japanese-wav2vec2-large-rs35kh` | 35Kh ReazonSpeech v2.0 | ~300M | Apache 2.0 |

**本プロジェクトのベースモデル候補**: 汎用 wav2vec2-base ではなく `japanese-wav2vec2-base` を使うことで日本語音声に特化した事前学習の恩恵を受けられる。

- [HuggingFace (base)](https://huggingface.co/reazon-research/japanese-wav2vec2-base) | [HuggingFace (large)](https://huggingface.co/reazon-research/japanese-wav2vec2-large-rs35kh)

---

## 2. Kotoba-Whisper (2024)

Whisper Large-v3 の蒸留モデル。

| Version | Training Data | Speed | Feature |
|---------|--------------|-------|---------|
| v1.0 | ReazonSpeech large (1,253h) | 6.3x faster than Large-v3 | CER/WER は Large-v3 以上 |
| v2.0 | ReazonSpeech full (7.2M clips) | Same | Full dataset |
| v2.2 | Same | Same | 話者分離 + 自動句読点 |

- **Architecture**: Full encoder from Whisper Large-v3 + 2-layer decoder
- [HuggingFace](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) | [GitHub](https://github.com/kotoba-tech/kotoba-whisper)

---

## 3. NVIDIA Parakeet-TDT-CTC-0.6B-ja (2025)

| Item | Detail |
|------|--------|
| Architecture | FastConformer TDT-CTC (XL) |
| Parameters | 600M |
| Training | ReazonSpeech v2.0, 300k + 100k fine-tuning steps (32x A100 80GB) |
| CER | JSUT: **6.4%**, CV8 (ja): **7.1%** |
| Innovation | TDT (Token-and-Duration Transducer): blank 予測スキップで高速推論 |
| Self-correction | CER > 10% のサンプルで再学習 |

- [HuggingFace](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja)

---

## 4. SenseVoice (Alibaba, 2024-07)

| Item | Detail |
|------|--------|
| Architecture | Non-autoregressive E2E |
| Training | 300K+ hours multilingual (400K+ total) |
| Languages | 50+ (including Japanese) |
| Speed | Whisper-small の 5 倍、Whisper-large の 15 倍 |
| Extra | 言語識別、感情認識、音声イベント検出 |

- [GitHub](https://github.com/FunAudioLLM/SenseVoice) | [HuggingFace](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)

---

## 5. Qwen3-ASR (Alibaba, 2025-09)

| Item | Detail |
|------|--------|
| Models | 1.7B / 0.6B |
| Base | Qwen3-Omni foundation model |
| Languages | 52 (including Japanese) |
| Performance | Open-source ASR で SOTA、商用 API に匹敵 |
| Japanese | -4.9pp CER improvement (1.7B) |
| Extra | 言語識別、タイムスタンプ、音楽/歌認識 |

- [arXiv](https://arxiv.org/html/2601.21337v1) | [GitHub](https://github.com/QwenLM/Qwen3-ASR) | [HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

---

## 6. Wav2Vec2-BERT (Meta, 2024)

| Item | Detail |
|------|--------|
| Parameters | 580M |
| Pretraining | 4.5M hours, 143+ languages |
| Fine-tuning | CTC-based, MIT license |
| Key | 10 時間のデータで Whisper 並みの性能、推論 10x 高速 |

- [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2-bert) | [Blog](https://huggingface.co/blog/fine-tune-w2v2-bert)

---

## 7. OWSM v3.1 (CMU, Interspeech 2024)

Open Whisper-Style Speech Model。E-Branchformer ベース。

| Scale | Params | Avg Error Rate |
|-------|--------|----------------|
| base | 101M | - |
| small | 367M | - |
| medium | 1.02B | 15.2% (vs OWSM v3: 18.8%) |

- Whisper より低 WER（英語 ASR データ量は 1/6 以下）、推論 25% 高速化
- [arXiv](https://arxiv.org/abs/2401.16658) | [HuggingFace](https://huggingface.co/espnet/owsm_v3.1_ebf)

---

## 8. Other Notable Works

### Efficient Adaptation of Multilingual Models for Japanese ASR (2024-12)

- Whisper-Tiny + LoRA: CER 32.7% → **14.7%** (Whisper-Base の 20.2% を超える)
- [arXiv](https://arxiv.org/abs/2412.10705)

### Benchmarking Japanese ASR with LLM Error Correction (2024-08)

- 日本語初の GER (Generative Error Correction) ベンチマーク
- Multi-pass augmented GER で複数 LLM の訂正を統合
- [arXiv](https://arxiv.org/abs/2408.16180)

### WFST-based Hybrid Japanese ASR (Interspeech 2025)

- wav2vec2 + CTC-WFST decoder、TTS augmented 学習
- 構音障害対応で fine-tuned Whisper より低 CER
- [PDF](https://www.isca-archive.org/interspeech_2025/hojo25_interspeech.pdf)

### Jargonic V2 (2025)

- ドメイン特化日本語 ASR (製造、物流、医療、金融)
- Whisper v3 の CER を半分以下に削減、ドメイン用語の 95% recall
- [Blog](https://aiola.ai/blog/jargonic-japanese-asr/)

---

## Summary: Japanese ASR Model Landscape

| Model | Params | CER/WER | Speed | Edge | Phoneme |
|-------|--------|---------|-------|------|---------|
| ReazonSpeech-k2-v2 | 159M | SOTA (JSUT/CV8/TEDxJP) | Very fast | Yes (ONNX, INT8) | No (char) |
| Parakeet-TDT-CTC-ja | 600M | 6.4% (JSUT) | Fast | Partial | No (char) |
| Kotoba-Whisper v2.2 | ~400M | > Large-v3 | 6.3x faster | whisper.cpp | No (char) |
| SenseVoice-Small | ~200M | > Whisper | 5-15x faster | Yes (ONNX) | No (char) |
| Qwen3-ASR-0.6B | 600M | SOTA OSS | Fast | Partial | No (char) |
| Wav2Vec2-BERT | 580M | Competitive | 10x faster | Yes (ONNX) | Possible |
| XLSR-Wav2Vec2 | 300M | 72% PER reduction | Moderate | Yes (ONNX) | Yes (IPA) |

**Key finding**: 主要モデルはすべて文字レベル出力。音素レベル出力の日本語 ASR は空白地帯。
