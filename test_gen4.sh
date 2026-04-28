#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
UV_BIN="${UV_BIN:-uv}"
# PyTorch 2.6+ defaults torch.load(weights_only=True), which breaks older Lightning checkpoints.
# Use legacy behavior for trusted local checkpoints.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

DATA_DIR="${GEN4_DATASET_DIR:-/path/to/gen4}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/ckpt_files/gen4/gen4_mAP29.ckpt}"
EXPERIMENT_CFG="${EXPERIMENT_CFG:-no_lstm}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
USE_TEST_SET="${USE_TEST_SET:-true}"
CHECKPOINT_STRICT="${CHECKPOINT_STRICT:-true}"

# For no_lstm experiments, checkpoint can be optional (start from scratch)
# If checkpoint provided, escape '=' for Hydra override parser
if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT_HYDRA=""
else
  CHECKPOINT_HYDRA="$(printf '%s' "$CHECKPOINT" | sed 's/=/\\=/g')"
fi

set -- \
  "dataset=gen4" \
  "+experiment/gen4=${EXPERIMENT_CFG}.yaml" \
  "dataset.path=${DATA_DIR}"

if [ -n "$CHECKPOINT_HYDRA" ]; then
  set -- "$@" "checkpoint=${CHECKPOINT_HYDRA}" "checkpoint_load_strict=${CHECKPOINT_STRICT}"
fi

set -- \
  "$@" \
  "use_test_set=${USE_TEST_SET}" \
  "hardware.gpus=${GPU_ID}" \
  "batch_size.eval=${BATCH_SIZE_EVAL}" \
  "hardware.num_workers.eval=${EVAL_WORKERS}"

cd "$ROOT_DIR"
echo "Running: $UV_BIN run python validation.py $*"
if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi
"$UV_BIN" run python validation.py "$@"
