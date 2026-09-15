# H100 Medium Training Report (2026-03-01)

## Summary

- **Model**: `reazon-research/japanese-wav2vec2-large` (317M params)
- **Dataset**: ReazonSpeech medium 1,000h (588,149 train / 30,955 val)
- **Infrastructure**: Vast.ai H100 80GB (instance 32190341)
- **Training**: 5 epochs (early stopped at epoch 5 ~20%, plateau)
- **Total time**: ~7h46m, **Cost**: ~$15
- **Best val_loss**: **1.5185** (step 75,000)

## Configuration

| Parameter | Value |
|---|---|
| Pretrained model | `reazon-research/japanese-wav2vec2-large` |
| Dataset | ReazonSpeech medium (medium_proc, max_duration=15s) |
| Train / Val samples | 588,149 / 30,955 |
| Batch size | 8 (grad_accum=4, effective=32) |
| Learning rate | 5e-5 |
| Warmup steps | 3,000 |
| Gradient clipping | 1.0 |
| Precision | BF16 |
| Eval / Save interval | 5,000 steps |
| Bucket batching | Yes |
| Augmentation | None (speed_perturb=0, noise=0) |
| Num workers | 16 |

## Training History

2回に分けて実行。Run 1 で epoch 1、Run 2 で epoch 2〜5（epoch 5 途中で打ち切り）。

### val_loss 推移

```
Step   val_loss  Epoch   Event
─────  ────────  ──────  ──────────
 5000   1.8505   ep1     best
10000   1.6252   ep1     best
15000   1.6096   ep1     best
20000   1.6084   ep1     best
ep1end  1.6093   ep1     avg_loss=2.0319
25000   1.5767   ep2     best
30000   1.5593   ep2     best
35000   1.5387   ep2     best
ep2end  1.5366   ep2     avg_loss=1.7912, best
40000   1.5368   ep3
45000   1.5283   ep3     best
50000   1.5255   ep3     best
55000   1.5226   ep3     best
ep3end  1.5214   ep3     avg_loss=1.7584, best
60000   1.5213   ep4     best
65000   1.5202   ep4     best
70000   1.5185   ep4     best
75000   1.5185   ep4     best (tied)
ep4end  1.5201   ep4     avg_loss=1.7490
~79000  (killed) ep5     plateau reached
```

### Run details

- **Run 1**: `--resume-from checkpoint-2000.pt --epochs 1`
  - Duration: ~1h51m (06:46 → 08:37 UTC)
- **Run 2**: `--resume-from best.pt --epochs 5`
  - Duration: ~5h55m (09:25 → 15:20 UTC)
  - Killed at epoch 5 ~20% — val_loss had not improved since step 70,000

## Evaluation Results

### Best Model (step 75,000, val_loss=1.5185)

| Dataset | Samples | Overall KER | Overall PER |
|---|---|---|---|
| **JSUT** (studio, single speaker) | 5,000 | **7.47%** | **10.42%** |
| **JVS** (100 speakers) | 9,997 | **15.68%** | **21.43%** |
| **ReazonSpeech** (wild) | 2,637 | **21.65%** | **21.87%** |

### Epoch 1 → Epoch 5 改善比較

| Dataset | KER (ep1) | KER (ep5) | Δ KER | PER (ep1) | PER (ep5) | Δ PER |
|---|---|---|---|---|---|---|
| JSUT | 7.86% | **7.47%** | -0.39 | 12.17% | **10.42%** | -1.75 |
| JVS | 17.57% | **15.68%** | -1.89 | 25.80% | **21.43%** | -4.37 |
| ReazonSpeech | 24.93%* | **21.65%** | -3.28 | 25.86%* | **21.87%** | -3.99 |

*ReazonSpeech: ep1=1,000 samples, ep5=2,637 samples（評価セットサイズが異なる）

### KER by Kana Group (ep5)

| Group | JSUT | JVS | ReazonSpeech |
|---|---|---|---|
| か行 | 5.0% | 7.0% | 17.0% |
| た行 | 4.8% | 7.5% | 16.7% |
| ら行 | 4.8% | 10.2% | 19.4% |
| さ行 | 5.8% | 11.2% | 17.9% |
| ま行 | 5.3% | 11.9% | 21.1% |
| な行 | 5.5% | 13.5% | 19.5% |
| わ行 | 7.0% | 12.0% | 22.5% |
| 濁音 (dakuten) | 8.6% | 15.7% | 21.7% |
| 半濁音 (handakuten) | 8.4% | 16.1% | 26.1% |
| は行 | 8.3% | 18.9% | 25.5% |
| や行 | 11.5% | 19.8% | 30.1% |
| 長音 (ー) | 11.5% | 19.8% | 27.4% |
| 母音 (vowels) | 13.4% | 25.4% | 29.8% |
| 小書き (small) | 11.9% | **28.1%** | 26.1% |

### PER by Phoneme Group (ep5)

| Group | JSUT | JVS | ReazonSpeech |
|---|---|---|---|
| 破裂音 (plosives) | 7.7% | 14.7% | 18.7% |
| 鼻音 (nasals) | 8.5% | 17.9% | 21.2% |
| 流音 (liquids) | 8.0% | 16.3% | 18.6% |
| 破擦音 (affricates) | 10.6% | 24.2% | 18.7% |
| 摩擦音 (fricatives) | 9.3% | 21.9% | 20.9% |
| 母音 (vowels) | 11.4% | 22.9% | 22.5% |
| 半母音 (glides) | 12.9% | 32.2% | 30.0% |
| 拗音 (palatalized) | 15.0% | **33.0%** | 22.8% |

### Top KER Confusion Pairs (ep5)

**JSUT**:
`い→え`(183), `け→き`(96), `ご→も`(89), `ー→ん`(81), `い→ー`(78), `を→ー`(57), `や→あ`(53)

**JVS**:
`ぉ→ほ`(443), `ど→と`(405), `ー→ん`(389), `い→ー`(335), `う→ー`(319), `ー→い`(281), `ー→あ`(277), `じ→ち`(237)

**ReazonSpeech**:
`い→ー`(37), `ー→ん`(34), `い→え`(33), `ん→ー`(30), `え→い`(25), `ね→め`(25), `お→ー`(24)

### Top PER Confusion Pairs (ep5)

**JSUT**:
`i→e`(235), `i→I`(232), `u→U`(211), `U→u`(192), `I→i`(171), `o→u`(158), `a→o`(149)

**JVS**:
`u→i`(1048), `I→i`(577), `u→U`(534), `d→t`(434), `f→h`(356), `U→u`(300), `my→m`(269)

**ReazonSpeech**:
`o→a`(163), `a→o`(128), `I→i`(117), `u→U`(111), `u→o`(105), `U→u`(101), `i→e`(98)

### Sample Predictions (ep5)

**JSUT** (KER 4.3%):
```
REF: みずをまれーしあからかわなくてわならないのです
HYP: みずをまれーしゃあからかわなくてわならないのです
```

**JVS** (KER 1.8%):
```
REF: にゅーいんぐらんどふーわぎゅーにゅーをべーすとしたしろいくりーむすーぷでありぼすとんくらむちゃうだーともよばれる
HYP: にゅーいんぐらんどふーわぎゅーにーをべーすとしたしろいくりーむすーぷでありぼすとんくらむちゃうだーともよばれる
```

**ReazonSpeech** (KER 0.0%):
```
REF: りょーほーにせきにんがあってせーきゅーできるとおもうんですよね
HYP: りょーほーにせきにんがあってせーきゅーできるとおもうんですよね
```

## Analysis

### 強み
- **子音系 (か行・た行・さ行・ら行)** はJSUTで5%未満と高精度
- **破裂音・鼻音・流音の PER** がJSUTで8%未満
- クリーン環境ではKER 7.5%, PER 10.4%で実用域に近い

### 弱点
- **小書き仮名 (ぁぃぅぇぉっ)**: JVS 28.1% — `ぉ→ほ`(443回) が最多混同
- **長音 `ー`**: 全データセットで不安定。`ー→ん`, `ー→あ`, `ー→い`, `ー→え` と多方向に誤る
- **母音 KER**: JSUT 13.4%, JVS 25.4% — 独立母音の認識が弱い
- **拗音 PER**: JVS 33.0% — `my→m`, `by→b`, `gy→g` など拗音マーカー脱落
- **無声化母音 PER**: `U↔u`, `I↔i` の判別がボトルネック（全データセット共通）
- **ノイズ耐性**: JSUT→ReazonSpeechでKER 3倍（7.5%→21.7%）

### ep1→ep5 での改善傾向
- **PER の改善が KER より大きい**: phoneme レベルの精度向上が進んでいる
- **ノイズ環境 (ReazonSpeech, JVS) ほど改善幅が大きい**: 汎化が進んだ
- **JSUT KER はほぼ頭打ち** (7.86%→7.47%): クリーン環境での限界に近い

## Comparison with Previous Models

| Model | Data | Epochs | val_loss | JSUT KER | JVS KER | ReazonSpeech KER |
|---|---|---|---|---|---|---|
| wav2vec2-base + small (100h) | ReazonSpeech small | 20 | 2.13 | N/A | N/A | N/A |
| wav2vec2-large + medium (1000h) ep1 | ReazonSpeech medium | 1 | 1.6093 | 7.86% | 17.57% | 24.93% |
| **wav2vec2-large + medium (1000h) ep5** | **ReazonSpeech medium** | **~5** | **1.5185** | **7.47%** | **15.68%** | **21.65%** |

## Cost

| Item | Value |
|---|---|
| H100 hourly rate | ~$1.87/hr |
| Total GPU time | ~8h |
| A100 (data staging) | ~$2 |
| **Total estimated cost** | **~$17** |

## Files

### Local

| File | Description | Size |
|---|---|---|
| `models/checkpoints/best-medium-ep5.pt` | Final best training checkpoint (step 75000) | 1.8GB |
| `models/checkpoints/best-medium-ep5-inference.pt` | Inference-only checkpoint | 632MB |
| `models/checkpoints/best-medium-ep1-inference.pt` | Epoch 1 inference checkpoint | 631MB |
| `logs/2026-03-01-h100-large-medium-1000h/` | All logs (training, eval, preprocess) | |

### Log files

| File | Content |
|---|---|
| `train_h100_medium_proc_resume.log` | Epoch 1 training log |
| `train_h100_medium_ep2_5.log` | Epoch 2-5 training log |
| `eval_reazonspeech_ep5.log` | ReazonSpeech evaluation (ep5 best) |
| `eval_jsut_ep5.log` | JSUT evaluation (ep5 best) |
| `eval_jvs_ep5.log` | JVS evaluation (ep5 best) |
| `eval_reazonspeech.log` | ReazonSpeech evaluation (ep1 best) |
| `eval_jsut.log` | JSUT evaluation (ep1 best) |
| `eval_jvs.log` | JVS evaluation (ep1 best) |
| `preprocess_h100_medium_clean.log` | Dataset preprocessing log |

## Next Steps

- [ ] Augmentation 追加 (speed perturbation, noise injection) で再学習
- [ ] 小書き仮名・長音の混同対策（ラベル正規化 or 後処理ルール）
- [ ] M2 Air デプロイ用の量子化・ONNX 変換
- [ ] リアルタイム推論パイプラインとの統合テスト
