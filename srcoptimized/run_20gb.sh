#!/usr/bin/env bash
set -euo pipefail

# 12 CPU cores, 20GB RAM, small-GPU profile.
# Override any variable before this script if you want a larger/slower run.

cd "$(dirname "$0")/.."

PROFILE="${CS116_PROFILE:-safe}"
case "${PROFILE}" in
  safe)
    : "${OPT_RUN_CATBOOST:=0}"
    : "${OPT_MAX_TRAIN_ROWS:=1000000}"
    : "${OPT_MAX_FINAL_TRAIN_ROWS:=1500000}"
    : "${OPT_MAX_EVAL_ROWS:=500000}"
    : "${OPT_LGBM_TREES:=700}"
    : "${OPT_LGBM_LEAVES:=63}"
    : "${OPT_LGBM_MIN_CHILD:=80}"
    : "${LGBM_MAX_BIN:=31}"
    : "${OPT_USE_RAW_ONLY:=1}"
    ;;
  stronger)
    : "${OPT_RUN_CATBOOST:=0}"
    : "${OPT_MAX_TRAIN_ROWS:=1500000}"
    : "${OPT_MAX_FINAL_TRAIN_ROWS:=2200000}"
    : "${OPT_MAX_EVAL_ROWS:=600000}"
    : "${OPT_LGBM_TREES:=900}"
    : "${OPT_LGBM_LEAVES:=95}"
    : "${OPT_LGBM_MIN_CHILD:=80}"
    : "${LGBM_MAX_BIN:=31}"
    : "${OPT_USE_RAW_ONLY:=1}"
    ;;
  large58)
    : "${OPT_RUN_CATBOOST:=0}"
    : "${OPT_MAX_TRAIN_ROWS:=2200000}"
    : "${OPT_MAX_FINAL_TRAIN_ROWS:=3200000}"
    : "${OPT_MAX_EVAL_ROWS:=800000}"
    : "${OPT_LGBM_TREES:=1000}"
    : "${OPT_LGBM_LEAVES:=63}"
    : "${OPT_LGBM_MIN_CHILD:=80}"
    : "${LGBM_MAX_BIN:=31}"
    : "${OPT_USE_RAW_ONLY:=1}"
    ;;
  a5000)
    : "${OPT_RUN_CATBOOST:=0}"
    : "${OPT_MAX_TRAIN_ROWS:=3000000}"
    : "${OPT_MAX_FINAL_TRAIN_ROWS:=4500000}"
    : "${OPT_MAX_EVAL_ROWS:=1000000}"
    : "${OPT_LGBM_TREES:=1200}"
    : "${OPT_LGBM_LEAVES:=63}"
    : "${OPT_LGBM_MIN_CHILD:=80}"
    : "${LGBM_MAX_BIN:=31}"
    : "${OPT_USE_RAW_ONLY:=1}"
    ;;
  none)
    ;;
  *)
    echo "Unknown CS116_PROFILE=${PROFILE}; use safe, stronger, large58, a5000, or none." >&2
    exit 2
    ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export OPT_N_JOBS="${OPT_N_JOBS:-12}"

export LGBM_USE_GPU="${LGBM_USE_GPU:-1}"
export LGBM_DEVICE_TYPE="${LGBM_DEVICE_TYPE:-gpu}"
export LGBM_GPU_PLATFORM_ID="${LGBM_GPU_PLATFORM_ID:-0}"
export LGBM_GPU_DEVICE_ID="${LGBM_GPU_DEVICE_ID:-0}"
export LGBM_MAX_BIN="${LGBM_MAX_BIN:-31}"
export LGBM_GPU_USE_DP="${LGBM_GPU_USE_DP:-0}"

export OPT_MAX_TRAIN_ROWS="${OPT_MAX_TRAIN_ROWS:-1000000}"
export OPT_MAX_FINAL_TRAIN_ROWS="${OPT_MAX_FINAL_TRAIN_ROWS:-1500000}"
export OPT_MAX_EVAL_ROWS="${OPT_MAX_EVAL_ROWS:-500000}"
export OPT_PRED_CHUNK_ROWS="${OPT_PRED_CHUNK_ROWS:-250000}"

export OPT_LGBM_TREES="${OPT_LGBM_TREES:-700}"
export OPT_LGBM_LEAVES="${OPT_LGBM_LEAVES:-63}"
export OPT_LGBM_MIN_CHILD="${OPT_LGBM_MIN_CHILD:-80}"
export OPT_LGBM_SUBSAMPLE="${OPT_LGBM_SUBSAMPLE:-0.80}"
export OPT_LGBM_COLSAMPLE="${OPT_LGBM_COLSAMPLE:-0.75}"
export OPT_EARLY_STOPPING="${OPT_EARLY_STOPPING:-50}"
export OPT_FALLBACK_CPU="${OPT_FALLBACK_CPU:-0}"
export OPT_USE_RAW_ONLY="${OPT_USE_RAW_ONLY:-1}"

# CatBoost is disabled by default because 4GB VRAM is usually too tight.
export OPT_RUN_CATBOOST="${OPT_RUN_CATBOOST:-0}"
export OPT_CATBOOST_USE_GPU="${OPT_CATBOOST_USE_GPU:-0}"
export OPT_CATBOOST_MAX_ROWS="${OPT_CATBOOST_MAX_ROWS:-400000}"
export OPT_CATBOOST_ITERS="${OPT_CATBOOST_ITERS:-300}"
export OPT_CATBOOST_DEPTH="${OPT_CATBOOST_DEPTH:-6}"

echo "CS116_PROFILE=${PROFILE}"
echo "OPT_MAX_TRAIN_ROWS=${OPT_MAX_TRAIN_ROWS} OPT_MAX_FINAL_TRAIN_ROWS=${OPT_MAX_FINAL_TRAIN_ROWS} OPT_MAX_EVAL_ROWS=${OPT_MAX_EVAL_ROWS}"
echo "OPT_LGBM_TREES=${OPT_LGBM_TREES} OPT_LGBM_LEAVES=${OPT_LGBM_LEAVES} OPT_LGBM_MIN_CHILD=${OPT_LGBM_MIN_CHILD} LGBM_MAX_BIN=${LGBM_MAX_BIN}"
echo "OPT_USE_RAW_ONLY=${OPT_USE_RAW_ONLY} OPT_RUN_CATBOOST=${OPT_RUN_CATBOOST}"
echo "OPT_ADD_TET_FEATURES=${OPT_ADD_TET_FEATURES:-0} OPT_ADD_EVENT_FEATURES=${OPT_ADD_EVENT_FEATURES:-0}"

run_with_python() {
  set +e
  "$@"
  status=$?
  set -e
  if [ "${status}" -eq 137 ] || [ "${status}" -eq 143 ]; then
    echo "Profile ${PROFILE} failed with status ${status} (likely OOM/Killed)." | tee outputs/profile_failure.log >&2
    exit "${status}"
  fi
  return "${status}"
}

MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-/opt/micromamba/envs/rapids-feature}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-/root/bin/micromamba}"

if [ -x "${MICROMAMBA_BIN}" ] && [ -d "${MAMBA_ENV_PREFIX}" ]; then
  if "${MICROMAMBA_BIN}" run -p "${MAMBA_ENV_PREFIX}" python - <<'PY' >/dev/null 2>&1
import numpy, pandas, pyarrow, lightgbm
PY
  then
    run_with_python "${MICROMAMBA_BIN}" run -p "${MAMBA_ENV_PREFIX}" python srcoptimized/pipeline_20gb.py --profile "${PROFILE}" "$@"
    exit $?
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import numpy, pandas, pyarrow, lightgbm
PY
then
  run_with_python "${PYTHON_BIN}" srcoptimized/pipeline_20gb.py --profile "${PROFILE}" "$@"
  exit $?
fi

if command -v micromamba >/dev/null 2>&1; then
  run_with_python micromamba run -n rapids-feature python srcoptimized/pipeline_20gb.py --profile "${PROFILE}" "$@"
  exit $?
fi

if [ -x /root/bin/micromamba ]; then
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/opt/micromamba}"
  run_with_python /root/bin/micromamba run -n rapids-feature python srcoptimized/pipeline_20gb.py --profile "${PROFILE}" "$@"
  exit $?
fi

echo "Could not find a Python environment with project dependencies. Activate rapids-feature first." >&2
exit 1
