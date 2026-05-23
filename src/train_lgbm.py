"""
src/train_lgbm.py
=================
Bước 5: Train LightGBM models (Poisson, Tweedie, L1).

Models được train với time-based validation (Nov và Dec 2025).
Lưu model vào models/, feature importance vào outputs/.
"""

import sys
import os
import logging
import yaml
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

with open(REPO_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

from metrics import evaluate, print_metrics

OUTPUT_DIR = REPO_ROOT / CFG["OUTPUT_DIR"]
MODEL_DIR  = REPO_ROOT / CFG["MODEL_DIR"]
REPORT_DIR = REPO_ROOT / CFG["REPORT_DIR"]
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOCATION_COL = CFG["TX_LOCATION_COL"]
ITEM_COL     = CFG["TX_ITEM_COL"]
TARGET_COL   = CFG["TARGET_COL"]
RANDOM_STATE = CFG["RANDOM_STATE"]

NON_FEATURE_COLS = {
    LOCATION_COL, ITEM_COL, "category",
    TARGET_COL, "sales_next_month", "revenue_next_month",
    "sale_status",
}

def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_lgbm_params() -> dict:
    params = {
        **CFG["LGBM_COMMON"],
        "random_state": RANDOM_STATE,
    }

    gpu_cfg = CFG.get("LGBM_GPU", {}) or {}
    use_gpu = env_flag("LGBM_USE_GPU", bool(gpu_cfg.get("enabled", False)))
    if not use_gpu:
        log.info("LightGBM GPU disabled. Set LGBM_USE_GPU=1 to enable GPU params.")
        return params

    gpu_params = {k: v for k, v in gpu_cfg.items() if k != "enabled" and v is not None}
    params.update(gpu_params)
    params["device_type"] = os.getenv("LGBM_DEVICE_TYPE", params.get("device_type", "gpu"))

    int_env_overrides = {
        "LGBM_GPU_PLATFORM_ID": "gpu_platform_id",
        "LGBM_GPU_DEVICE_ID": "gpu_device_id",
        "LGBM_MAX_BIN": "max_bin",
    }
    for env_name, param_name in int_env_overrides.items():
        value = os.getenv(env_name)
        if value not in (None, ""):
            params[param_name] = int(value)

    log.info(
        "LightGBM GPU enabled: device_type=%s, gpu_platform_id=%s, gpu_device_id=%s, max_bin=%s",
        params.get("device_type"),
        params.get("gpu_platform_id"),
        params.get("gpu_device_id"),
        params.get("max_bin"),
    )
    return params


LGBM_PARAMS_BASE = build_lgbm_params()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def encode_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, dict]:
    """Label encode categorical columns."""
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def load_train_features():
    """Load features_val_nov and features_val_dec as fold-indexed training data."""
    folds = []
    for name, val_month in [("nov", 11), ("dec", 12)]:
        path = OUTPUT_DIR / f"features_val_{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["fold_val_month"] = val_month
            folds.append(df)
        else:
            log.warning(f"Missing {path}, run features.py first")
    if not folds:
        raise FileNotFoundError("No feature files found. Run src/features.py first.")
    return folds


def load_predict_features():
    path = OUTPUT_DIR / "features_predict_jan2026.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/features.py first.")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Build X, y for training
# ---------------------------------------------------------------------------
def prepare_train_fold(fold_df: pd.DataFrame, val_month: int):
    """For a given fold dataframe, split into train and validation."""
    # As folds are already separate parquets, all goes to one
    df = fold_df.copy()
    # Filter out sale_status=0 from training
    if "sale_status" in df.columns:
        df = df[df["sale_status"] != 0]

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COL].values if TARGET_COL in df.columns else df["sales_next_month"].values

    # Location and item as categoricals (encode to int)
    df_meta = df[[LOCATION_COL, ITEM_COL]].copy()

    for col in [LOCATION_COL, ITEM_COL]:
        X[col] = df[col].astype(str).astype("category").cat.codes

    return X, y, df_meta, feature_cols


# ---------------------------------------------------------------------------
# Train a single model
# ---------------------------------------------------------------------------
def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    objective: str = "poisson",
    log_transform: bool = False,
    model_name: str = "lgbm_poisson",
) -> lgb.Booster:
    params = {**LGBM_PARAMS_BASE, "objective": objective}
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.5

    y_tr = np.log1p(y_train) if log_transform else y_train
    y_va = np.log1p(y_val)   if log_transform else y_val

    dtrain = lgb.Dataset(X_train, label=y_tr)
    dval   = lgb.Dataset(X_val,   label=y_va, reference=dtrain)

    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                 lgb.log_evaluation(period=-1)]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=params.pop("n_estimators", 1000),
        valid_sets=[dval],
        callbacks=callbacks,
    )

    model_path = MODEL_DIR / f"{model_name}.txt"
    model.save_model(str(model_path))
    log.info(f"Model saved: {model_path} (trees={model.num_trees()})")
    return model


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_training():
    folds = load_train_features()
    predict_df = load_predict_features()

    # We use fold Nov as a hold-out for fold Dec training, and vice versa.
    # For simplicity: train on Nov-cutoff data, val on Dec-cutoff data.
    # (Two cutoffs give two separate datasets)
    df_nov = folds[0]  # cutoff Oct31, target Nov
    df_dec = folds[1]  # cutoff Nov30, target Dec

    results = {}
    all_fi = []
    predictions_val = []

    model_configs = [
        {"name": "lgbm_poisson", "objective": "poisson",    "log": False},
        {"name": "lgbm_tweedie", "objective": "tweedie",    "log": False},
        {"name": "lgbm_l1",      "objective": "regression_l1", "log": True},
    ]

    for mcfg in model_configs:
        mname = mcfg["name"]
        log.info(f"\n=== Training {mname} ===")

        # Prepare Nov fold (train) → validate Dec
        X_nov, y_nov, meta_nov, feat_cols = prepare_train_fold(df_nov, 11)
        X_dec, y_dec, meta_dec, _         = prepare_train_fold(df_dec, 12)

        # Align columns
        common_cols = [c for c in feat_cols if c in X_dec.columns]
        # Add location/item
        for col in [LOCATION_COL, ITEM_COL]:
            if col not in common_cols:
                common_cols.append(col)
        X_nov_m = X_nov[[c for c in common_cols if c in X_nov.columns]]
        X_dec_m = X_dec[[c for c in common_cols if c in X_dec.columns]]

        model = train_model(
            X_nov_m, y_nov,
            X_dec_m, y_dec,
            objective=mcfg["objective"],
            log_transform=mcfg["log"],
            model_name=mname,
        )

        # Predict on Dec validation
        pred_dec_raw = model.predict(X_dec_m, num_iteration=model.best_iteration)
        if mcfg["log"]:
            pred_dec = np.expm1(pred_dec_raw)
        else:
            pred_dec = pred_dec_raw
        pred_dec = np.clip(pred_dec, 0, None)

        # Build pred df for Dec
        val_names_df = meta_dec.copy()
        val_names_df["prediction"] = pred_dec

        # Evaluate
        gt_df = df_dec[[LOCATION_COL, ITEM_COL, "sales_next_month", "revenue_next_month"]].copy()
        gt_df = gt_df[gt_df["sales_next_month"] > 0]
        gt_df = gt_df.rename(columns={"sales_next_month": "sales", "revenue_next_month": "revenue"})

        metrics = evaluate(
            df_true=gt_df,
            df_pred=val_names_df,
            location_col=LOCATION_COL,
            item_col=ITEM_COL,
        )
        metrics["model"] = mname
        results[mname] = metrics
        print_metrics(metrics, label=f"{mname}/Dec")

        # Feature importance
        fi = pd.DataFrame({
            "feature": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
            "model": mname,
        })
        all_fi.append(fi)

        # Predictions on final (Jan2026)
        predict_copy = predict_df.copy()
        for col in [LOCATION_COL, ITEM_COL]:
            predict_copy[col] = predict_copy[col].astype(str).astype("category").cat.codes
        X_pred = predict_copy[[c for c in common_cols if c in predict_copy.columns]]
        final_pred_raw = model.predict(X_pred, num_iteration=model.best_iteration)
        if mcfg["log"]:
            final_pred = np.expm1(final_pred_raw)
        else:
            final_pred = final_pred_raw
        final_pred = np.clip(final_pred, 0, None)
        predict_df[f"pred_{mname}"] = final_pred
        val_names_df["model"] = mname
        predictions_val.append(val_names_df)

    # Save feature importance
    fi_all = pd.concat(all_fi, ignore_index=True)
    fi_path = OUTPUT_DIR / "feature_importance.csv"
    fi_all.to_csv(fi_path, index=False)
    log.info(f"Feature importance saved: {fi_path}")

    # Save model results
    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    log.info("\n=== Model Results Summary ===")
    log.info(res_df.to_string(index=False))

    # Save model results markdown
    res_md = "# Model Results\n\n" + res_df.to_markdown(index=False) + "\n"
    mdr_path = REPORT_DIR / "model_results.md"
    mdr_path.write_text(res_md)
    log.info(f"Results saved to {mdr_path}")

    # Save predictions for Jan2026 (for blending)
    pred_paths = OUTPUT_DIR / "predictions_lgbm.parquet"
    predict_df.to_parquet(pred_paths, index=False)
    log.info(f"LightGBM predictions saved: {pred_paths}")

    # Save val predictions
    val_preds = pd.concat(predictions_val, ignore_index=True)
    val_preds.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    return predict_df, results


if __name__ == "__main__":
    run_training()
