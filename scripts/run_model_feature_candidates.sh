#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONTROL="outputs/submission_best_39_0.pkl"
SOURCE_39="outputs/submission_larger_1800k_gpu_inv_y_raw_scale_best.pkl"
if [ ! -f "${CONTROL}" ]; then
  cp -n "${SOURCE_39}" "${CONTROL}"
fi

PROFILE="${CS116_PROFILE:-a5000}"
COMMON_ENV=(
  CS116_PROFILE="${PROFILE}"
  OPT_WEIGHT_MODE=inv_y
  OPT_BLEND_MODE=raw_only
  OPT_RUN_CATBOOST=0
  LGBM_MAX_BIN=31
  OPT_FALLBACK_CPU=0
)

MICROMAMBA_BIN="${MICROMAMBA_BIN:-/root/bin/micromamba}"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-/opt/micromamba/envs/rapids-feature}"

run_py() {
  "${MICROMAMBA_BIN}" run -p "${MAMBA_ENV_PREFIX}" python "$@"
}

run_cache() {
  local cache_dir="$1"
  local tag="$2"
  shift 2
  env "${COMMON_ENV[@]}" "$@" bash srcoptimized/run_20gb.sh --cache-dir "${cache_dir}" --stage all --force
  env "${COMMON_ENV[@]}" "$@" "${MICROMAMBA_BIN}" run -p "${MAMBA_ENV_PREFIX}" python srcoptimized/inv_y_group_scale_candidates.py \
    --cache-dir "${cache_dir}" \
    --tag "${tag}" \
    --raw-out "outputs/submission_${tag}_inv_y_raw.pkl" \
    --group-out "outputs/submission_${tag}_inv_y_5group_scale.pkl" \
    --metadata-out "outputs/candidate_validation_${tag}.json"
}

run_cache "outputs/srcoptimized_cache_larger_model" "larger_sample" \
  OPT_ADD_TET_FEATURES=0 OPT_ADD_EVENT_FEATURES=0

run_cache "outputs/srcoptimized_cache_larger_tet" "larger_sample_tet" \
  OPT_ADD_TET_FEATURES=1 OPT_ADD_EVENT_FEATURES=0

run_cache "outputs/srcoptimized_cache_larger_tet_event" "larger_sample_tet_event" \
  OPT_ADD_TET_FEATURES=1 OPT_ADD_EVENT_FEATURES=1

run_py scripts/write_submit_plan_model_features.py
run_py scripts/check_model_feature_candidates.py
