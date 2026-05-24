#!/bin/bash
# =========================================================================
# CS116 Task 2 - optimized monthly sale forecasting pipeline
# =========================================================================
# Main split:
# - Train: 2025-01 .. 2025-11 features/targets
# - Validation: 2025-12
# - Final train: 2025-01 .. 2025-12
# - Predict: 2026-01
#
# Useful notebook/debug knobs:
#   OPT_MAX_ROW_GROUPS=2      # read only a few parquet row groups
#   OPT_MAX_TRAIN_ROWS=3000000
#   OPT_RUN_CATBOOST=0
# =========================================================================

set -e

# GPU-first defaults. Override any of these before calling the script if the
# current notebook/kernel uses a different device setup.
export LGBM_USE_GPU="${LGBM_USE_GPU:-1}"
export LGBM_DEVICE_TYPE="${LGBM_DEVICE_TYPE:-gpu}"
export LGBM_GPU_PLATFORM_ID="${LGBM_GPU_PLATFORM_ID:-0}"
export LGBM_GPU_DEVICE_ID="${LGBM_GPU_DEVICE_ID:-0}"
export LGBM_MAX_BIN="${LGBM_MAX_BIN:-63}"
export LGBM_GPU_USE_DP="${LGBM_GPU_USE_DP:-0}"

export OPT_RUN_CATBOOST="${OPT_RUN_CATBOOST:-1}"
export OPT_CATBOOST_USE_GPU="${OPT_CATBOOST_USE_GPU:-1}"
export OPT_CATBOOST_DEVICES="${OPT_CATBOOST_DEVICES:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPT_N_JOBS="${OPT_N_JOBS:--1}"

echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q

echo "GPU config:"
echo "  LightGBM: use_gpu=${LGBM_USE_GPU}, device_type=${LGBM_DEVICE_TYPE}, platform=${LGBM_GPU_PLATFORM_ID}, device=${LGBM_GPU_DEVICE_ID}, max_bin=${LGBM_MAX_BIN}"
echo "  CatBoost: run=${OPT_RUN_CATBOOST}, use_gpu=${OPT_CATBOOST_USE_GPU}, devices=${OPT_CATBOOST_DEVICES}"

echo "[2/3] Running optimized monthly forecast pipeline..."
python src/optimized_forecast_pipeline.py

echo "[3/3] Checking final submission..."
python src/check_submission.py

echo "======================================================================="
echo "DONE. Submit this file:"
echo "outputs/submission_final.csv"
echo "Schema: location, item_id, quantity"
echo "======================================================================="
