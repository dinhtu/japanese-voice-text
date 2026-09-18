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

## Install

Python 3.11+. `pyopenjtalk` is compiled from source, so install a C++ toolchain first —
on Windows the [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
with the "Desktop development with C++" workload, on macOS/Linux a working `cc` + CMake.
Without it the install fails on `pyopenjtalk`.

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) covers the API, the practice page and the tests.
The training and dataset tools used by `scripts/` (`datasets`, `wandb`, `sounddevice`,
`unidic`, ...) are listed commented-out at the bottom of the same file — uncomment what
you need. `uv sync` also works and installs everything at once.

### Download the model checkpoint (required)

The checkpoint is 603MB, so it is **not** in this repository. Nothing transcribes until
you download it into `models/checkpoints/`:

```bash
hf download sakasegawa/japanese-wav2vec2-large-hiragana-ctc \
    best-medium-ep5-inference.pt --local-dir models/checkpoints
```

`hf` ships with `transformers`, so it is there after the pip install above. Without it,
plain `curl` does the same job:

```bash
curl -L -o models/checkpoints/best-medium-ep5-inference.pt \
  https://huggingface.co/sakasegawa/japanese-wav2vec2-large-hiragana-ctc/resolve/main/best-medium-ep5-inference.pt
```

The file must end up at `models/checkpoints/best-medium-ep5-inference.pt` (631,542,555
bytes) — that is where [app/core/config.py](app/core/config.py) looks by default. To
serve a checkpoint from somewhere else, point `ASR_CHECKPOINT` at it instead.

**If it is missing**, the server still starts and the practice page still loads, but the
log shows `ASR checkpoint not found: ...`, `GET /health` reports `"model_loaded": false`,
and every recording comes back `500 ASR model is not available.`

## Project layout

```
app/
  main.py               FastAPI app: middleware, routers, /health
  api/routes.py         POST /api/pronunciation/evaluate
  api/pages.py          GET / (practice page)
  core/config.py        Settings read from the environment
  schemas/              Response models
  services/             ASR service, normalization, scoring, use case
  constants/            Practice sentences
src/asr/                Model, inference, dataset, kana/phoneme converters
static/  templates/     Practice page assets (no build step)
scripts/                Dataset prep, training, evaluation, realtime demos
models/checkpoints/     Model checkpoint — download it, see Install (not in git)
tests/                  Scoring, normalization, API tests
run.py                  python run.py -> serves the API + page
```

## Quick Start

```bash
# Download UniDic (first time only; needs the training deps)
python -m unidic download

# Inference on audio file
python scripts/03_infer.py --audio your_audio.wav

# Real-time ASR from microphone
python scripts/realtime_asr.py
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
One process serves both the JSON API and the practice web page.

```
WAV --> wav2vec2 Dual CTC --> recognized kana --\
                                                 >-- normalize (pyopenjtalk) --> CER --> score 0-100
admin target text ------------------------------/
```

### Start

```bash
python run.py
# or, with auto-reload while developing:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Practice page: <http://localhost:8000> · interactive API docs: <http://localhost:8000/docs> ·
health check: `GET /health`

> The page records from the microphone, so open it over `localhost` or HTTPS —
> browsers block mic access on plain HTTP hosts.

Configuration (all optional, via environment variables):

| Variable | Default | Purpose |
|----------|---------|---------|
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8000` | Bind address `run.py` listens on (`uvicorn` takes `--host` / `--port` instead) |
| `ASR_CHECKPOINT` | `models/checkpoints/best-medium-ep5-inference.pt` | Checkpoint to serve — [download it first](#download-the-model-checkpoint-required) |
| `ASR_DEVICE` | auto (`cuda` > `mps` > `cpu`) | Inference device |
| `ASR_FP16` | `0` | FP16 inference |
| `ASR_EAGER_LOAD` | `1` | Load the model at startup instead of on first request |
| `MAX_AUDIO_MB` | `25` | Upload size limit |
| `CORS_ORIGINS` | `localhost:3000`, `localhost:5173` | Allowed origins for *external* API clients; the page itself is same-origin |

### `POST /api/pronunciation/evaluate`

`multipart/form-data`:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Target Japanese text — kanji, kana or mixed |
| `audio` | file | `.wav` recording of the user reading it |

```bash
curl -X POST "http://localhost:8000/api/pronunciation/evaluate" \
  -F "text=げんきょうもいちにちがんばりましょう" \
  -F "audio=@test.wav"
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

### `GET /` — practice page

A server-rendered page (Jinja2 + a little vanilla JS, no build step) that shows a target
sentence, records from the microphone, and displays the score. The browser cannot upload
what `MediaRecorder` produces (webm/ogg), so [`static/app.js`](static/app.js)
decodes the recording with the Web Audio API, downmixes to mono, resamples to the model's
16 kHz and writes the RIFF header itself. Recording stops at 30 s to stay under the upload
limit.

Edit the sentences in [`app/constants/practice_texts.py`](app/constants/practice_texts.py) —
kanji, kana or mixed, since the target is normalized to a reading before comparison. The
first entry is the default.

### Tests

```bash
pytest tests/   # scoring + normalization + API (ASR stubbed, no model load)
```

## Training

```bash
# Prepare dataset
python scripts/00_prepare_dataset.py --splits medium

# Train (large model, 1000h)
python scripts/01_train.py \
    --pretrained reazon-research/japanese-wav2vec2-large \
    --data-split medium --dataset-dir data/datasets/reazonspeech \
    --epochs 5 --batch-size 8 --grad-accum 4 --lr 5e-5 --bf16

# Evaluate
python scripts/02_evaluate.py --checkpoint models/checkpoints/best.pt --dataset jsut
python scripts/02_evaluate.py --checkpoint models/checkpoints/best.pt --dataset jvs
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
