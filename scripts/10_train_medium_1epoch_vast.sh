#!/usr/bin/env bash
set -euo pipefail

# Run on a Vast.ai A100 40GB instance after cloning this repository.
# Example:
#   bash scripts/10_train_medium_1epoch_vast.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Sync dependencies"
uv sync

echo "[2/5] Prepare ReazonSpeech medium if missing"
if [ ! -d "data/datasets/reazonspeech/medium" ]; then
  uv run python scripts/00_prepare_dataset.py --splits medium
else
  echo "data/datasets/reazonspeech/medium already exists; skip download."
fi

echo "[3/5] Preprocess medium -> medium_proc if missing"
if [ ! -d "data/datasets/reazonspeech/medium_proc" ]; then
  PREPROC_WRITER_BATCH_SIZE="${PREPROC_WRITER_BATCH_SIZE:-1000}"
  PREPROC_ARGS=(
    --input data/datasets/reazonspeech/medium
    --output data/datasets/reazonspeech/medium_proc
    --max-duration 15.0
    --writer-batch-size "${PREPROC_WRITER_BATCH_SIZE}"
  )
  if [ -n "${PREPROC_NUM_PROC:-}" ]; then
    PREPROC_ARGS+=(--num-proc "${PREPROC_NUM_PROC}")
  fi
  uv run python scripts/00b_preprocess.py "${PREPROC_ARGS[@]}"
else
  echo "data/datasets/reazonspeech/medium_proc already exists; skip preprocessing."
fi

echo "[4/5] Train 1 epoch on medium (wav2vec2-large, BF16)"
TRAIN_ARGS=(
  --pretrained reazon-research/japanese-wav2vec2-large
  --data-split medium
  --dataset-dir data/datasets/reazonspeech
  --epochs 1
  --batch-size 8
  --grad-accum 4
  --lr 5e-5
  --warmup-steps 3000
  --grad-clip 1.0
  --bf16
  --eval-steps 5000
  --save-steps 5000
  --eval-at-epoch-end
  --save-at-epoch-end
  --bucket-batching
  --speed-perturb-prob 0.0
  --noise-prob 0.0
)
if [ -n "${TRAIN_NUM_WORKERS:-}" ]; then
  TRAIN_ARGS+=(--num-workers "${TRAIN_NUM_WORKERS}")
fi
uv run python scripts/01_train.py "${TRAIN_ARGS[@]}"

echo "[5/5] Done"
echo "Check models/checkpoints and training logs for epoch time / stability."
echo "Tip: TRAIN_NUM_WORKERS=8 PREPROC_NUM_PROC=12 bash scripts/10_train_medium_1epoch_vast.sh"
