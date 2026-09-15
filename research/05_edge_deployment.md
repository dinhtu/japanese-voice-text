# Edge / On-Device Deployment (2024--2026)

Target: **M2 Air 16GB** での日本語音素 ASR リアルタイム推論。

---

## 1. sherpa-onnx (k2-fsa)

**最も成熟したオンデバイス日本語 ASR デプロイメントフレームワーク。**

| Item | Detail |
|------|--------|
| Runtime | ONNX Runtime (GPU 不要) |
| Platforms | Linux, macOS, Windows, Android, iOS, HarmonyOS, Raspberry Pi, RISC-V |
| Languages bindings | 12 |
| Offline | 完全オフライン推論 |
| Features | STT, TTS, speaker diarization, VAD |

### Available Japanese Models

| Model | Params | Quantization |
|-------|--------|-------------|
| ReazonSpeech-k2-v2 (Zipformer) | 159M | INT8 |
| SenseVoice (ja/en/zh/ko/yue) | ~200M | INT8 |

- [GitHub](https://github.com/k2-fsa/sherpa-onnx) | [Docs](https://k2-fsa.github.io/sherpa/onnx/index.html)

---

## 2. WhisperKit (Argmax, 2024--2025)

Apple Silicon 向け CoreML ベースの Whisper 推論。

| Item | Detail |
|------|--------|
| Framework | CoreML + Apple Neural Engine (ANE) |
| Features | Real-time streaming, word timestamps, VAD |
| Compression | OD-MBP (Outlier-Decomposed Mixed-Bit Palletization) |
| Size | <1 GB (WER は uncompressed の 1% 以内) |
| Latency | **0.46s** |
| ANE speedup | CPU-only の **3x+** |
| M2 performance | Whisper-medium を ~3.3s で処理 |

- [GitHub](https://github.com/argmaxinc/WhisperKit) | [arXiv](https://arxiv.org/abs/2507.10860)

---

## 3. whisper.cpp (ggml-org)

C/C++ ポートの Whisper (GGML ベース)。

| Item | Detail |
|------|--------|
| Quantization | INT4, INT5, INT8 |
| Apple Silicon | ANE acceleration via CoreML backend |
| M2 performance | ~3.3s for medium model |
| Japanese | 全 Whisper バリアント対応 |

- [GitHub](https://github.com/ggml-org/whisper.cpp)

---

## 4. MLX-Audio (Apple MLX Framework, 2025)

Apple Silicon ネイティブ最適化。

| Item | Detail |
|------|--------|
| Framework | Apple MLX |
| Models | Whisper variants, Qwen3-ASR |
| Features | STT, TTS, STS |
| Advantage | CoreML 変換不要、MLX ネイティブ |

- [GitHub (mlx-audio)](https://github.com/Blaizzy/mlx-audio)
- [GitHub (lightning-whisper-mlx)](https://github.com/mustafaaljadery/lightning-whisper-mlx)

---

## 5. Quantization Strategies

### For wav2vec2 + CTC (本プロジェクト)

| Strategy | Size Reduction | Quality Impact |
|----------|---------------|---------------|
| ONNX export + INT8 dynamic | ~4x | Minimal |
| ONNX export + INT4 | ~8x | Moderate |
| CoreML + ANE | ~4x + ANE accel | Minimal |
| PyTorch quantization (qint8) | ~4x | Minimal |

### Recommended Path for M2 Air

```
PyTorch model
  → torch.onnx.export()
  → ONNX Runtime (INT8 quantized)
  → sherpa-onnx integration
```

**Alternative**: MLX への変換で Apple Silicon 最適化。

---

## 6. Model Size Comparison

| Model | Params | FP32 Size | INT8 Size |
|-------|--------|-----------|-----------|
| wav2vec2-base + CTC head | ~95M | ~380MB | ~95MB |
| wav2vec2-large + CTC head | ~300M | ~1.2GB | ~300MB |
| Whisper-tiny | 39M | ~150MB | ~40MB |
| Whisper-small | 244M | ~960MB | ~240MB |
| ReazonSpeech-k2-v2 | 159M | ~640MB | ~160MB |

**wav2vec2-base (95M) は M2 Air 16GB で余裕を持って動作可能。**

---

## 7. Latency Benchmarks on Apple Silicon

| Model | Device | RTF | Latency (5s audio) |
|-------|--------|-----|-------------------|
| Whisper-medium (whisper.cpp) | M2 Max | - | ~3.3s |
| Whisper-medium (WhisperKit) | M2 | - | ~3.3s |
| Whisper-small (MLX) | M2 Air | - | ~1.5s (est.) |
| wav2vec2-base + CTC | M2 Air | <0.1 (est.) | <0.5s (est.) |

**wav2vec2-base + CTC は非自己回帰モデルであり、Whisper (自己回帰) と比較して根本的にレイテンシが低い。**

---

## 8. Spike Window Decoding for Deployment

SWD (02_ctc_advances.md 参照) は CTC デコーディングを 2.17 倍高速化。

- CTC スパイクの左右 1 フレームのみを使用
- greedy デコーディングの計算量を大幅削減
- エッジデバイスでのリアルタイム推論に直結
- [arXiv](https://arxiv.org/abs/2501.03257)

---

## 9. Deployment Architecture Options

### Option A: ONNX (Recommended)

```
wav2vec2-base + CTC head
  → ONNX export (opset 17)
  → ONNX Runtime (INT8)
  → sherpa-onnx wrapper
  → macOS/iOS/Android
```

**Pros**: Cross-platform, mature ecosystem, INT8 quantization
**Cons**: ONNX opset compatibility issues

### Option B: MLX (Apple-native)

```
wav2vec2-base + CTC head
  → MLX conversion
  → mlx-audio integration
  → macOS only
```

**Pros**: Apple Silicon 最適化、unified memory 活用
**Cons**: macOS のみ、エコシステムが若い

### Option C: CoreML

```
wav2vec2-base + CTC head
  → coremltools conversion
  → CoreML + ANE
  → macOS/iOS
```

**Pros**: ANE ハードウェアアクセラレーション
**Cons**: 変換の複雑さ、デバッグが困難

### Option D: PyTorch Mobile (Prototype)

```
wav2vec2-base + CTC head
  → torch.jit.script / torch.export
  → PyTorch Mobile / ExecuTorch
```

**Pros**: 開発中の PyTorch コードがそのまま使える
**Cons**: モバイル向け最適化が限定的
