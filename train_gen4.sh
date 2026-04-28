#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
UV_BIN="${UV_BIN:-uv}"

DATA_DIR="${GEN4_DATASET_DIR:-/path/to/gen4}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-6}"
TRAIN_WORKERS_PER_GPU="${TRAIN_WORKERS_PER_GPU:-6}"
EVAL_WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-HybridLSTM_gen4_no_lstm}"
WANDB_GROUP="${WANDB_GROUP:-no_lstm}"

EXTRA_ARGS=""
if [ "${FAST_DEV:-0}" = "1" ]; then
  EXTRA_ARGS=$(cat <<EOF
training.max_steps=1
training.max_epochs=1
training.limit_train_batches=1
validation.limit_val_batches=1
validation.check_val_every_n_epoch=1
EOF
)
fi

set -- \
  "$UV_BIN" run python train.py \
  "dataset=gen4" \
  "dataset.path=${DATA_DIR}" \
  "+experiment/gen4=no_lstm.yaml" \
  "hardware.gpus=${GPU_IDS}" \
  "batch_size.train=${BATCH_SIZE_PER_GPU}" \
  "batch_size.eval=${BATCH_SIZE_PER_GPU}" \
  "hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU}" \
  "hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}" \
  "wandb.project_name=${WANDB_PROJECT}" \
  "wandb.group_name=${WANDB_GROUP}"

if [ -n "$EXTRA_ARGS" ]; then
  set -- "$@" $EXTRA_ARGS
fi

cd "$ROOT_DIR"
echo "Running: $*"
if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi
"$@"
