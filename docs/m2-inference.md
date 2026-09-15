# M2 Mac 高速推論ガイド (MPS + FP16)

## モデル情報

| 項目 | 値 |
|---|---|
| アーキテクチャ | wav2vec2-large + Dual CTC (InterCTC phoneme + kana) |
| ベースモデル | `reazon-research/japanese-wav2vec2-large` |
| エンコーダ層数 | 24 (hidden=1024) |
| InterCTC層 | 12 (中間層、phoneme出力) |
| パラメータ数 | 315,554,042 (315.6M) |
| Kana語彙 | 84 (82ひらがな + `<sp>` + `<blank>`) |
| Phoneme語彙 | 38 (37音素 + `<blank>`) |

## チェックポイント

| ファイル | パス | サイズ | 説明 |
|---|---|---|---|
| large (暫定best) | `models/checkpoints/best_large.pt` | 603 MB | BF16保存、val_loss=1.3588 (step 12500, epoch 7) |
| base (best) | `models/checkpoints/best.pt` | 361 MB | FP32保存、val_loss=2.13 |

### チェックポイント形式

```python
{
    "model_state_dict": OrderedDict,  # BF16 tensor weights
    "pretrained": "reazon-research/japanese-wav2vec2-large",
    "inter_ctc_layer": 12,
}
```

重みはBF16で保存されている。`load_checkpoint()` 内で `.float()` によりFP32に変換される。

## 学習情報

| 項目 | 値 |
|---|---|
| 学習データ | ReazonSpeech small (100h, 62,047サンプル) |
| GPU | A100 80GB (RunPod) |
| 精度 | BF16 |
| バッチサイズ | 32 |
| エポック数 | 15 (学習中、現在epoch 10) |
| 学習率 | 1e-4 |
| Optimizer | AdamW |
| Loss | CR-CTC (kana, λ=0.7) + InterCTC (phoneme, λ=0.3) |
| SpecAugment | mask_time_prob=0.05 |
| 所要時間 | ~21分/epoch, 合計~5.2h |

## 評価結果 (JSUT-BASIC5000)

| モデル | KER | PER |
|---|---|---|
| **large** (暫定best) | **7.5%** | **6.6%** |
| base (100h) | 17.9% | 19.6% |

## M2 Macでの推論セットアップ

### 前提条件

- macOS 13+ (Ventura以降、MPS対応)
- Python 3.11+
- PyTorch 2.1+ (MPS backend対応)

### 1. 依存関係インストール

```bash
# リポジトリクローン済みの場合
uv sync

# UniDic (初回のみ、fugashi用)
uv run python -m unidic download
```

### 2. MPS推論の実行

現在の `scripts/03_infer.py` はCUDA/CPUの自動切り替えのみ対応。
MPS対応の推論は以下のように実行する:

```bash
# FP32 (安全、まずこれで動作確認)
uv run python scripts/03_infer.py \
    --audio data/test.wav \
    --checkpoint models/checkpoints/best_large.pt \
    --pretrained reazon-research/japanese-wav2vec2-large \
    --show-phonemes
```

**注意**: 現在の `03_infer.py` は `device = "cuda" if torch.cuda.is_available() else "cpu"` でデバイスを選択するため、Mac上ではCPUフォールバックになる。MPS対応が必要。

### 3. MPS + FP16対応の変更点

`scripts/03_infer.py` の `main()` 内でデバイス選択を以下に変更:

```python
# デバイス選択 (MPS対応)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

FP16で推論する場合 (メモリ半減 + M2のFP16ネイティブ対応で高速化):

```python
model.to(device)
model.half()  # FP16化 (MPS上でFP16ネイティブ動作)
model.eval()

# 入力もFP16に
input_values = inputs.input_values.to(device).half()
```

### 4. メモリ見積もり

| 精度 | モデルサイズ | 推論時ピーク (15秒音声) |
|---|---|---|
| FP32 | ~1.26 GB | ~2.5 GB |
| FP16 | ~0.63 GB | ~1.3 GB |

M2 Air 16GBなら余裕。FP16推奨。

### 5. MPS既知の注意点

- **初回推論が遅い**: MPS Metal shader compilation のため初回は数秒かかる。2回目以降は高速。
- **一部opの非対応**: wav2vec2のgroup normなど、一部opがMPSで未対応の場合CPUフォールバックが発生する。PyTorch 2.4+で改善済み。
- **torch.compile未対応**: MPS backendでは `torch.compile` は2024時点で未サポート。将来対応予定。

### 6. RTF目安

| デバイス | 精度 | RTF (目安) |
|---|---|---|
| RTX 2080 CUDA | FP32 | ~0.05 |
| M2 MPS | FP16 | ~0.1-0.2 (推定) |
| M2 CPU | FP32 | ~0.5-1.0 |

RTF < 1.0 ならリアルタイム処理可能。MPS + FP16で十分リアルタイム。

## 次のステップ (性能不足の場合)

1. **ONNX Runtime + CoreML EP**: `torch.onnx.export` → `onnxruntime` with `CoreMLExecutionProvider` で ANE活用
2. **CoreML変換**: `coremltools` で直接CoreMLモデルに変換、ANEフル活用
3. **量子化**: INT8量子化でさらにメモリ・速度改善
