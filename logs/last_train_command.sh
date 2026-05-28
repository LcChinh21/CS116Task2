#!/usr/bin/env bash
set -euo pipefail

TAG="two_stage"
CACHE_DIR="outputs/srcoptimized_cache_neg_B"

export CS116_PROFILE=none
export OMP_NUM_THREADS=12
export OPT_N_JOBS=12
export LGBM_USE_GPU=0
export LGBM_DEVICE_TYPE=cpu
export OPT_FALLBACK_CPU=1
export LGBM_MAX_BIN=31

export OPT_ADD_TET_FEATURES=1
export OPT_ADD_EVENT_FEATURES=1
export OPT_WEIGHT_MODE=inv_y
export OPT_BLEND_MODE=raw_only
export OPT_RUN_CATBOOST=0
export OPT_USE_RAW_ONLY=1
export OPT_NEGATIVE_SAMPLE_RATIO=1.0
export OPT_ZERO_WEIGHT=0.3

export OPT_MAX_TRAIN_ROWS=1800000
export OPT_MAX_FINAL_TRAIN_ROWS=2600000
export OPT_MAX_EVAL_ROWS=700000
export OPT_PRED_CHUNK_ROWS=250000
export OPT_LGBM_TREES=900
export OPT_LGBM_LEAVES=63
export OPT_LGBM_MIN_CHILD=80
export OPT_LGBM_SUBSAMPLE=0.80
export OPT_LGBM_COLSAMPLE=0.75
export OPT_EARLY_STOPPING=50

python3 scripts/two_stage_lgbm.py \
  --cache-dir "${CACHE_DIR}" \
  --out "reports/two_stage_validation.md" \
  --json-out "outputs/two_stage_validation.json" \
  --submission-out "outputs/submission_two_stage_lgbm.pkl" \
  --baseline 51.178136 \
  --make-submission-if-better
