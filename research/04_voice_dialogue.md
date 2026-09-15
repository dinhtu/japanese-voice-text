# Real-Time Voice Dialogue and AVSR (2024--2026)

## 1. End-to-End Speech Dialogue Models

### Key Surveys

| Survey | Year | Venue | Link |
|--------|------|-------|------|
| WavChat: A Survey of Spoken Dialogue Models | 2024 | arXiv | [2411.13577](https://arxiv.org/abs/2411.13577) |
| Recent Advances in Speech Language Models | 2025 | ACL Long | [PDF](https://aclanthology.org/2025.acl-long.682.pdf) |
| Full-Duplex Spoken Language Models Survey | 2025 | arXiv | [2509.14515](https://arxiv.org/abs/2509.14515) |

### Model Comparison

| Model | Year | Architecture | Latency | OSS | License |
|-------|------|-------------|---------|-----|---------|
| **Moshi** (Kyutai) | 2024 | Helium 7B + Mimi codec, full-duplex | **200ms** | Yes | Apache 2.0 |
| **GPT-4o Realtime** | 2024 | Native multimodal S2S (WebRTC) | Sub-second | No | API |
| **Gemini Live** | 2024 | Native audio, Gemini 2.5 Flash | <800ms | No | API |
| **GLM-4-Voice** | 2024 | Single-codebook 12.5Hz + flow-matching | - | Yes | - |
| **LLaMA-Omni** | 2024 | Llama-3.1-8B + streaming decoder | Low | Yes | - |
| **Mini-Omni** | 2024 | Multimodal LLM, hear+talk+think | Low | Yes | - |
| **Freeze-Omni** | 2025 | Frozen LLM + speech I/O (ICML 2025) | Low | Yes | - |
| **OmniFlatten** | 2025 | GPT E2E, full-duplex (ACL 2025) | Low | Yes | - |
| **MinMo** (Alibaba) | 2025 | 8B, token interleaving, 1.4M h | STT 100ms, duplex 800ms | Partial | - |
| **PSLM** | 2024 | Parallel text+speech gen (EMNLP) | Reduced | - | - |

---

## 2. Moshi: Deep Dive

本プロジェクトの音声対話ビジョンに最も参考になるモデル。

### Architecture

```
Mimi Codec (12.5Hz, 1.1kbps, 8 RVQ codebooks)
  ├── 1 semantic codebook (knowledge distillation from non-causal model)
  └── 7 acoustic codebooks

Helium LLM (7B params, 2.1T tokens English text)

Multi-stream token generator
  ├── Temporal Transformer
  └── Depth Tokenizer stack
```

### Key Innovations

1. **Inner Monologue**: 音声トークンの prefix として text トークンを予測 → 言語品質が大幅向上
2. **Dual-stream**: システムとユーザーの音声を並列ストリームでモデル化 (full-duplex)
3. **Mimi codec**: SpeechTokenizer (50Hz, 4kbps) を 12.5Hz, 1.1kbps で上回る

### Training Pipeline

1. Unsupervised pre-training
2. Multi-stream post-training with diarization
3. Fisher dataset fine-tuning (duplex)
4. Instruction fine-tuning (synthetic scripts)

- [arXiv](https://arxiv.org/abs/2410.00037) | [GitHub](https://github.com/kyutai-labs/moshi)

---

## 3. Streaming ASR for Dialogue

| System | Latency | Architecture |
|--------|---------|-------------|
| NVIDIA Nemotron Speech | **24ms** median | Cache-aware FastConformer + RNN-T |
| Deepgram Nova-3 | ~150ms | Commercial streaming |
| SenseVoice-Small | 70ms / 10s | Non-autoregressive |
| FastConformer | RTF < 0.2 | 8x subsampling + depthwise separable conv |

### Latency Budget for Voice Dialogue

Human conversational response: **300--500ms**

```
ASR (~150ms) + LLM (~300ms) + TTS (~100ms) = ~550ms (理想)
実際: 800ms -- 2s (stack latency compounding)
```

- [Nemotron HuggingFace](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)

---

## 4. Japanese Voice Dialogue Systems

### NTT Communications LLM Voice Dialogue (2024-12)

- **Architecture**: User speech → STT → LLM (CyberAgent calm3-22b-chat) → TTS
- **Optimization**: 句読点出力時に即座に音声生成開始（ストリーミング最適化）
- [Blog](https://engineers.ntt.com/entry/202411-streaming-dialogue/entry)

### PSLM: Parallel Speech Language Model (EMNLP 2024)

- テキストと音声の並列生成でレイテンシ削減
- 音素レベルの中間表現が対話レイテンシ削減に有効であることを示唆
- [arXiv](https://arxiv.org/abs/2406.12428)

---

## 5. Discrete Speech Tokens for Dialogue

### Discrete vs. Continuous (EMNLP 2025)

- SpeechLLM の入力として離散トークンと連続特徴量を比較
- [PDF](https://aclanthology.org/2025.emnlp-main.1266.pdf)

### Discrete Audio Tokens Survey (2025)

- メルスペクトログラムの代替として離散トークンで学習時間 **<35%** に削減
- ASR, S2S 翻訳, 音声変換, TTS, 音声強調, 音源分離をカバー
- [arXiv](https://arxiv.org/html/2506.10274v3)

### Token Types

| Type | Example | Frame Rate |
|------|---------|-----------|
| Semantic | HuBERT k-means (k=2000) | 50Hz |
| Acoustic | Encodec, DAC | 75Hz |
| Hybrid | Mimi (1 semantic + 7 acoustic) | 12.5Hz |

---

## 6. Audio-Visual Speech Recognition (AVSR)

### SOTA Systems

| Model | Year | Venue | WER (LRS3) |
|-------|------|-------|-----------|
| **Whisper-Flamingo** | 2024 | Interspeech | ASR: 0.68%, AVSR: 0.76% |
| **mWhisper-Flamingo** | 2025 | IEEE SPL | 9 言語 SOTA |
| **Llama-AVSR** | 2025 | ICASSP | SOTA (multiple scales) |
| **Zero-AVSR** | 2025 | ICCV | 82 言語ゼロショット |
| **VALLR** | 2025 | ICCV | Lip reading LM |
| **MMS-LLaMA** | 2025 | ACL Findings | Multi-modal speech |

### Whisper-Flamingo Architecture

```
Audio input ──→ [Whisper encoder (frozen)] ──→ Audio features
                                                    │
Video input ──→ [Visual encoder] ──→ Visual features │
                                         │          ↓
                                    Gated cross-attention
                                         ↓
                                   [Whisper decoder]
```

- [arXiv](https://arxiv.org/abs/2406.10082) | [GitHub](https://github.com/roudimit/whisper-flamingo)

### Japanese AVSR: Significant Gap

- 日本語 AVSR データセットは存在するが限定的
- 日本語特化の SOTA AVSR システムは **未発見**
- 本プロジェクト (asr-test) + vsr-test の組み合わせが未開拓領域をカバーする可能性

---

## 7. BESTOW: Streaming SpeechLLM

CTC と LLM を組み合わせたストリーミング音声理解。

| Feature | Detail |
|---------|--------|
| Architecture | GPT-style + T5-style のハイブリッド |
| Streaming | Read-write policy として定式化 |
| CTC role | CTC pretraining + blank filtering でシーケンス圧縮 |
| Tasks | ASR, AST, SQA, DynamicSuperb |

- [arXiv](https://arxiv.org/abs/2406.19954)
