# hiragana-asr

Lightweight Japanese ASR that outputs **hiragana only** — no hallucination by design.

wav2vec2-large + Dual CTC (InterCTC for phonemes, CR-CTC for kana). 315M parameters, runs real-time on MacBook Air M2.

## Why hiragana?

- **No hallucination**: CTC is structurally incapable of generating content not in the input
- **Lightweight**: 315M params (half of Whisper large-v3), FP16 inference ~630MB
- **Easy to fine-tune**: Simple CTC + wav2vec2 fine-tuning, no complex decoder
- **LLM-friendly**: Pass hiragana to an LLM for kanji conversion and intent understanding

## Model

Available on HuggingFace: [sakasegawa/japanese-wav2vec2-large-hiragana-ctc](https://huggingface.co/sakasegawa/japanese-wav2vec2-large-hiragana-ctc) / [Spaces Demo](https://huggingface.co/spaces/sakasegawa/hiragana-asr)

| Model | Data | JSUT KER | JVS KER | ReazonSpeech KER |
|-------|------|:--------:|:-------:|:----------------:|
| wav2vec2-large + 1,000h (ep5) | ReazonSpeech medium | 7.47% | 15.68% | 21.65% |

## Quick Start


```bash
# Install dependencies
uv sync

# Download UniDic (first time only)
uv run python -m unidic download

# Inference on audio file
uv run python scripts/03_infer.py --audio your_audio.wav

# Real-time ASR from microphone
uv run python scripts/realtime_asr.py
```



## Architecture

```
Audio (16kHz) → CNN Feature Extractor (frozen) → Transformer Encoder (24 layers)
                                                    ├── Layer 12 → Phoneme CTC Head (InterCTC)
                                                    └── Layer 24 → Kana CTC Head (CR-CTC)
```

- **Base encoder**: [reazon-research/japanese-wav2vec2-large](https://huggingface.co/reazon-research/japanese-wav2vec2-large) (pretrained on 35,000h)
- **InterCTC** ([Lee & Watanabe, ICASSP 2021](https://arxiv.org/abs/2102.03216)): Phoneme auxiliary task at layer 12
- **CR-CTC** ([Yao et al., ICLR 2025](https://arxiv.org/abs/2410.05101)): Consistency regularization for smoother CTC output
- **Loss**: `CR-CTC(kana) + 0.3 × CTC(phoneme)`

## Pronunciation Evaluation API

A FastAPI service that scores a spoken recording against a target Japanese text.

```
WAV --> wav2vec2 Dual CTC --> recognized kana --                                                 >-- normalize (pyopenjtalk) --> CER --> score 0-100
admin target text ------------------------------/
```

### Start

```bash
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: <http://localhost:8000/docs> · health check: `GET /health`

Configuration (all optional, via environment variables):

| Variable | Default | Purpose |
|----------|---------|---------|
| `ASR_CHECKPOINT` | `models/checkpoints/best-medium-ep5-inference.pt` | Checkpoint to serve |
| `ASR_DEVICE` | auto (`cuda` > `mps` > `cpu`) | Inference device |
| `ASR_FP16` | `0` | FP16 inference |
| `ASR_EAGER_LOAD` | `1` | Load the model at startup instead of on first request |
| `MAX_AUDIO_MB` | `25` | Upload size limit |
| `CORS_ORIGINS` | `localhost:3000`, `localhost:5173` | Comma-separated allowed origins |

### `POST /api/pronunciation/evaluate`

`multipart/form-data`:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Target Japanese text — kanji, kana or mixed |
| `audio` | file | `.wav` recording of the user reading it |

```bash
curl -X POST "http://localhost:8000/api/pronunciation/evaluate"   -F "text=げんきょうもいちにちがんばりましょう"   -F "audio=@test.wav"
```

```json
{
  "success": true,
  "target_text": "げんきょうもいちにちがんばりましょう",
  "target_hiragana": "げんきょーもいちにちがんばりましょー",
  "recognized_text": "げんきょーもいちにちがんばりましょー",
  "recognized_hiragana": "げんきょーもいちにちがんばりましょー",
  "score": 100,
  "cer": 0.0,
  "distance": 0,
  "audio_duration": 6.34,
  "inference_ms": 1225.2,
  "feedback": { "level": "excellent", "message": "Pronunciation matches the target very closely." },
  "errors": []
}
```

Errors: `400` (empty/unpronounceable `text`, non-WAV upload, oversized file), `422` (undecodable audio or a missing field), `500` (model or inference failure).

### Scoring methodology

Both sides are normalized to the same hiragana *reading* with `pyopenjtalk` before comparison, so orthographic differences do not cost points:

```
今日も一日頑張りましょう  -->  きょーもいちにちがんばりましょー
...がんばりましょう       -->  ...がんばりましょー     (long vowel おう -> おー)
```

Then, over characters of the normalized target:

```
CER        = levenshtein(target, recognized) / len(target)
similarity = max(0, 1 - CER)
score      = round(similarity * 100)
```

`errors[]` carries the character-level diff (`sub` / `del` / `ins` with a position in the target), which is where phoneme-level feedback will slot in later.

### Limitation

**This score measures how closely the ASR transcription matches the target text — not true acoustic pronunciation quality.** The model outputs hiragana only; it does not assess pitch accent, timing, or phoneme articulation. Recording noise, an unusual speaking rate, or plain ASR error will lower the score even when pronunciation is fine, and a fluent-but-wrong reading that happens to transcribe correctly will score well. Treat it as a read-aloud accuracy check.

### Tests

```bash
uv run --extra dev pytest tests/   # scoring + normalization + API (ASR stubbed, no model load)
```

### Web UI

A minimal Next.js front-end for read-aloud practice lives in [`frontend/`](frontend/) — it shows a
target sentence, records from the microphone, and displays the score. See
[frontend/README.md](frontend/README.md).

```bash
cd frontend && npm install && npm run dev   # http://localhost:3000
```

## Training

```bash
# Prepare dataset
uv run python scripts/00_prepare_dataset.py --splits medium

# Train (large model, 1000h)
uv run python scripts/01_train.py \
    --pretrained reazon-research/japanese-wav2vec2-large \
    --data-split medium --dataset-dir data/datasets/reazonspeech \
    --epochs 5 --batch-size 8 --grad-accum 4 --lr 5e-5 --bf16

# Evaluate
uv run python scripts/02_evaluate.py --checkpoint models/checkpoints/best.pt --dataset jsut
uv run python scripts/02_evaluate.py --checkpoint models/checkpoints/best.pt --dataset jvs
```

## Evaluation Datasets

| Dataset | Condition | Utterances |
|---------|-----------|------------|
| [JSUT-BASIC5000](https://sites.google.com/site/shinaborulab/publication/jsut) | Studio, single speaker | 5,000 |
| [JVS parallel100](https://sites.google.com/site/shinaborulab/publication/jvs) | 100 speakers | ~10,000 |
| [ReazonSpeech](https://research.reazon.jp/) | TV broadcast (wild) | ~2,600 |

## Blog Post

Detailed write-up (in Japanese): [ひらがなASRを作った話](https://nyosegawa.github.io/posts/hiragana-asr/)

## License

Apache-2.0. See [LICENSE](LICENSE).

Training data: [ReazonSpeech](https://research.reazon.jp/) (CDLA-Sharing-1.0 — model weights are unrestricted).
