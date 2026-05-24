#!/usr/bin/env bash
set -euo pipefail

# 12 CPU cores, 20GB RAM, small-GPU profile.
# Override any variable before this script if you want a larger/slower run.

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export OPT_N_JOBS="${OPT_N_JOBS:-12}"

export LGBM_USE_GPU="${LGBM_USE_GPU:-1}"
export LGBM_DEVICE_TYPE="${LGBM_DEVICE_TYPE:-gpu}"
export LGBM_GPU_PLATFORM_ID="${LGBM_GPU_PLATFORM_ID:-0}"
export LGBM_GPU_DEVICE_ID="${LGBM_GPU_DEVICE_ID:-0}"
export LGBM_MAX_BIN="${LGBM_MAX_BIN:-31}"
export LGBM_GPU_USE_DP="${LGBM_GPU_USE_DP:-0}"

export OPT_MAX_TRAIN_ROWS="${OPT_MAX_TRAIN_ROWS:-700000}"
export OPT_MAX_FINAL_TRAIN_ROWS="${OPT_MAX_FINAL_TRAIN_ROWS:-1000000}"
export OPT_MAX_EVAL_ROWS="${OPT_MAX_EVAL_ROWS:-350000}"
export OPT_PRED_CHUNK_ROWS="${OPT_PRED_CHUNK_ROWS:-250000}"

export OPT_LGBM_TREES="${OPT_LGBM_TREES:-500}"
export OPT_LGBM_LEAVES="${OPT_LGBM_LEAVES:-63}"
export OPT_LGBM_MIN_CHILD="${OPT_LGBM_MIN_CHILD:-100}"
export OPT_LGBM_SUBSAMPLE="${OPT_LGBM_SUBSAMPLE:-0.80}"
export OPT_LGBM_COLSAMPLE="${OPT_LGBM_COLSAMPLE:-0.75}"
export OPT_EARLY_STOPPING="${OPT_EARLY_STOPPING:-50}"
export OPT_FALLBACK_CPU="${OPT_FALLBACK_CPU:-1}"

# CatBoost is disabled by default because 4GB VRAM is usually too tight.
export OPT_RUN_CATBOOST="${OPT_RUN_CATBOOST:-0}"
export OPT_CATBOOST_USE_GPU="${OPT_CATBOOST_USE_GPU:-0}"
export OPT_CATBOOST_MAX_ROWS="${OPT_CATBOOST_MAX_ROWS:-400000}"
export OPT_CATBOOST_ITERS="${OPT_CATBOOST_ITERS:-300}"
export OPT_CATBOOST_DEPTH="${OPT_CATBOOST_DEPTH:-6}"

if python - <<'PY' >/dev/null 2>&1
import numpy, pandas, pyarrow, lightgbm
PY
then
  exec python srcoptimized/pipeline_20gb.py "$@"
fi

if command -v micromamba >/dev/null 2>&1; then
  exec micromamba run -n rapids-feature python srcoptimized/pipeline_20gb.py "$@"
fi

if [ -x /root/bin/micromamba ]; then
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/opt/micromamba}"
  exec /root/bin/micromamba run -n rapids-feature python srcoptimized/pipeline_20gb.py "$@"
fi

echo "Could not find a Python environment with project dependencies. Activate rapids-feature first." >&2
exit 1
