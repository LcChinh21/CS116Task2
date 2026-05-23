"""
src/blend.py
============
Bước 6: Blend predictions + weight search trên Dec 2025 validation.

Blend: pred = w1*poisson + w2*tweedie + w3*l1 + w4*baseline
Tìm weight tối ưu MAPE Sales trên Dec 2025.
"""

import sys
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

with open(REPO_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

from metrics import evaluate, print_metrics
from baseline import load_purchases, load_items, make_baseline_submission, apply_sale_status

OUTPUT_DIR = REPO_ROOT / CFG["OUTPUT_DIR"]
MODEL_DIR  = REPO_ROOT / CFG["MODEL_DIR"]
REPORT_DIR = REPO_ROOT / CFG["REPORT_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOCATION_COL = CFG["TX_LOCATION_COL"]
ITEM_COL     = CFG["TX_ITEM_COL"]
GRID_STEPS   = CFG.get("BLEND_GRID_STEPS", 5)


# ---------------------------------------------------------------------------
# Load val predictions from LightGBM models (Dec fold assumed)
# ---------------------------------------------------------------------------
def load_lgbm_val_predictions() -> pd.DataFrame:
    """Load model predictions for validation months from features files."""
    dec_feat = OUTPUT_DIR / "features_val_dec.parquet"
    lgbm_preds = OUTPUT_DIR / "predictions_lgbm.parquet"

    if not dec_feat.exists():
        raise FileNotFoundError(f"{dec_feat} missing. Run features.py and train_lgbm.py first.")

    # Load the Dec validation features (which contain model outputs if we ran train)
    df = pd.read_parquet(dec_feat)

    # Try to load model predictions parquet for validation
    # These are stored via train_lgbm: pred_lgbm_poisson, pred_lgbm_tweedie, pred_lgbm_l1
    import lightgbm as lgb
    from sklearn.preprocessing import LabelEncoder

    NON_FEAT = {LOCATION_COL, ITEM_COL, "category",
                "sales_next_month", "revenue_next_month", "sale_status", "fold_val_month"}
    feat_cols = [c for c in df.columns if c not in NON_FEAT]

    # Encode categories
    X = df[feat_cols].copy()
    for col in [LOCATION_COL, ITEM_COL]:
        if col in X.columns:
            X[col] = X[col].astype(str).astype("category").cat.codes

    results = df[[LOCATION_COL, ITEM_COL, "sales_next_month", "revenue_next_month"]].copy()

    model_configs = [
        ("lgbm_poisson", False),
        ("lgbm_tweedie", False),
        ("lgbm_l1",      True),
    ]
    for mname, log_transform in model_configs:
        mpath = MODEL_DIR / f"{mname}.txt"
        if not mpath.exists():
            log.warning(f"Model {mpath} not found, skipping.")
            results[f"pred_{mname}"] = 0.0
            continue
        model = lgb.Booster(model_file=str(mpath))
        feat_names = model.feature_name()
        X_m = X[[c for c in feat_names if c in X.columns]]
        # fill missing
        for c in feat_names:
            if c not in X_m.columns:
                X_m[c] = 0.0
        X_m = X_m[feat_names]
        raw = model.predict(X_m, num_iteration=model.best_iteration)
        pred = np.expm1(raw) if log_transform else raw
        results[f"pred_{mname}"] = np.clip(pred, 0, None)

    return results


# ---------------------------------------------------------------------------
# Build baseline predictions for Dec
# ---------------------------------------------------------------------------
def load_baseline_val(purch: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp("2025-11-30")
    purch_train = purch[purch["date"] <= cutoff]
    sub = make_baseline_submission(cutoff, purch_train)
    sub = apply_sale_status(sub, items)
    return sub.rename(columns={"prediction": "pred_baseline"})


# ---------------------------------------------------------------------------
# Grid search blend weights
# ---------------------------------------------------------------------------
def grid_search_weights(val_df: pd.DataFrame, grid_steps: int = 5) -> dict:
    """
    Search over (w1, w2, w3, w4) grid, normalise to sum=1.
    Return best weights minimising MAPE Sales.
    """
    pred_cols = ["pred_lgbm_poisson", "pred_lgbm_tweedie", "pred_lgbm_l1", "pred_baseline"]
    available = [c for c in pred_cols if c in val_df.columns]

    steps = np.linspace(0, 1, grid_steps + 1)
    best_mape = float("inf")
    best_weights = {}

    log.info(f"Grid search over {len(available)} models × {grid_steps+1}^{len(available)} combinations ...")

    if len(available) == 1:
        return {available[0]: 1.0}

    combos = list(product(steps, repeat=len(available)))
    combos = [(c, s) for c in combos for s in [sum(c)] if s > 0]

    for weights_raw, wsum in combos:
        weights_norm = [w / wsum for w in weights_raw]
        pred = sum(val_df[col].fillna(0) * w for col, w in zip(available, weights_norm))
        pred = np.clip(pred, 0, None)

        gt = val_df[val_df["sales_next_month"] > 0].copy()
        gt_true = gt[[LOCATION_COL, ITEM_COL, "sales_next_month", "revenue_next_month"]].rename(
            columns={"sales_next_month": "sales", "revenue_next_month": "revenue"}
        )
        gt_pred = gt[[LOCATION_COL, ITEM_COL]].copy()
        gt_pred["prediction"] = pred.loc[gt.index]
        metrics = evaluate(
            gt_true,
            gt_pred,
            location_col=LOCATION_COL,
            item_col=ITEM_COL,
            qty_col="sales",
            revenue_col="revenue",
        )
        mape = metrics.get("mape_sales", float("inf"))

        if mape < best_mape:
            best_mape = mape
            best_weights = dict(zip(available, weights_norm))

    log.info(f"Best MAPE: {best_mape:.4f}%  |  Weights: {best_weights}")
    return best_weights, best_mape


# ---------------------------------------------------------------------------
# Apply blend weights and output
# ---------------------------------------------------------------------------
def blend_and_save(predict_df_lgbm: pd.DataFrame, best_weights: dict) -> pd.DataFrame:
    """Apply best blend weights to Jan 2026 prediction features."""
    pred = pd.Series(0.0, index=predict_df_lgbm.index)
    for col, w in best_weights.items():
        if col in predict_df_lgbm.columns:
            pred += predict_df_lgbm[col].fillna(0) * w
        elif col == "pred_baseline":
            # Load baseline for Jan2026 prediction
            pass
    pred = np.clip(pred, 0, None)
    predict_df_lgbm["prediction_blend"] = pred
    return predict_df_lgbm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_blend():
    log.info("Loading validation predictions ...")
    val_df = load_lgbm_val_predictions()

    log.info("Loading purchases for baseline ...")
    purch = load_purchases(sample=CFG.get("DEBUG_SAMPLE", False))
    items = load_items()

    baseline_val = load_baseline_val(purch, items)
    val_df = val_df.merge(baseline_val, on=[LOCATION_COL, ITEM_COL], how="left")
    val_df["pred_baseline"] = val_df["pred_baseline"].fillna(0)

    # Grid search
    result = grid_search_weights(val_df, grid_steps=GRID_STEPS)
    if isinstance(result, tuple):
        best_weights, best_mape = result
    else:
        best_weights = result
        best_mape = None

    # Load Jan2026 predictions
    lgbm_preds_path = OUTPUT_DIR / "predictions_lgbm.parquet"
    if lgbm_preds_path.exists():
        predict_df = pd.read_parquet(lgbm_preds_path)
    else:
        log.warning("predictions_lgbm.parquet missing, using empty baseline")
        predict_df = pd.DataFrame()

    # Build blended prediction for Jan2026
    if not predict_df.empty:
        pred = pd.Series(0.0, index=predict_df.index)
        for col, w in best_weights.items():
            if col == "pred_baseline":
                # Load baseline for final cutoff (Dec 31)
                cutoff = pd.Timestamp("2025-12-31")
                purch_all = purch[purch["date"] <= cutoff]
                base_sub = make_baseline_submission(cutoff, purch_all)
                base_sub = apply_sale_status(base_sub, items)
                predict_df = predict_df.merge(
                    base_sub.rename(columns={"prediction": "pred_baseline"}),
                    on=[LOCATION_COL, ITEM_COL], how="left"
                )
                predict_df["pred_baseline"] = predict_df["pred_baseline"].fillna(0)
            if col in predict_df.columns:
                pred += predict_df[col].fillna(0) * w

        predict_df["prediction_blend"] = np.clip(pred, 0, None)
        blend_path = OUTPUT_DIR / "predictions_blended.parquet"
        predict_df.to_parquet(blend_path, index=False)
        log.info(f"Blended predictions saved: {blend_path}")
    else:
        log.warning("No LightGBM predictions found. Using baseline only.")
        cutoff = pd.Timestamp("2025-12-31")
        purch_all = purch[purch["date"] <= cutoff]
        base_sub = make_baseline_submission(cutoff, purch_all)
        base_sub = apply_sale_status(base_sub, items)
        base_sub["prediction_blend"] = base_sub["prediction"]
        predict_df = base_sub
        predict_df.to_parquet(OUTPUT_DIR / "predictions_blended.parquet", index=False)

    # Save blend weights
    import json
    weights_path = OUTPUT_DIR / "blend_weights.json"
    weights_path.write_text(json.dumps({"weights": best_weights, "val_mape": best_mape}, indent=2))
    log.info(f"Blend weights saved: {weights_path}")

    return predict_df, best_weights


if __name__ == "__main__":
    run_blend()
