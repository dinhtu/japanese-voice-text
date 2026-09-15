#!/usr/bin/env bash
set -euo pipefail

# Continue medium training on Vast.ai from an existing checkpoint.
# Example:
#   TARGET_EPOCHS=2 bash scripts/11_resume_medium_vast.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

RESUME_FROM="${RESUME_FROM:-models/checkpoints/final.pt}"
TARGET_EPOCHS="${TARGET_EPOCHS:-2}"

if [ ! -f "$RESUME_FROM" ]; then
  echo "resume checkpoint not found: $RESUME_FROM" >&2
  exit 1
fi

echo "[1/4] Sync dependencies"
uv sync

echo "[2/4] Ensure ReazonSpeech medium exists"
if [ ! -d "data/datasets/reazonspeech/medium" ]; then
  uv run python scripts/00_prepare_dataset.py --splits medium
else
  echo "data/datasets/reazonspeech/medium already exists; skip download."
fi

echo "[3/4] Ensure medium_proc exists"
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

echo "[4/4] Resume training to epoch ${TARGET_EPOCHS} from ${RESUME_FROM}"
TRAIN_ARGS=(
  --pretrained reazon-research/japanese-wav2vec2-large
  --data-split medium
  --dataset-dir data/datasets/reazonspeech
  --epochs "${TARGET_EPOCHS}"
  --resume-from "${RESUME_FROM}"
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

echo "Done. Check models/checkpoints and train logs."
