"""
src/postprocess.py
==================
Bước 6 (tiếp): Post-processing tối ưu MAPE.

1. Clip predictions (0 ≤ pred ≤ max_90d * 2.0)
2. Floor nhỏ → 0 nếu < threshold
3. sale_status=0 → prediction=0
4. Nếu không có purchase 90d nhưng có view/ATC gần đây → small prediction
5. Xuat submission_final.pkl
"""

import sys
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

with open(REPO_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

from baseline import load_purchases, load_items

OUTPUT_DIR   = REPO_ROOT / CFG["OUTPUT_DIR"]
LOCATION_COL = CFG["TX_LOCATION_COL"]
ITEM_COL     = CFG["TX_ITEM_COL"]
QTY_COL      = CFG["TX_QTY_COL"]
SUBMISSION_PRED_COL = "quantity_pred"
CLIP_MULT    = CFG.get("CLIP_MAX_MULTIPLIER", 2.0)
FLOOR_THR    = CFG.get("FLOOR_THRESHOLD", 0.5)
EPS          = 1e-6


# ---------------------------------------------------------------------------
# Compute max 90d historical sales per location × item
# ---------------------------------------------------------------------------
def compute_max_90d(purch: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    start = cutoff - pd.Timedelta(days=90)
    window = purch[(purch["date"] > start) & (purch["date"] <= cutoff)]
    # daily totals
    daily = window.groupby([LOCATION_COL, ITEM_COL, "date"])[QTY_COL].sum().reset_index()
    max90 = daily.groupby([LOCATION_COL, ITEM_COL])[QTY_COL].max().reset_index()
    max90.rename(columns={QTY_COL: "max_daily_90d"}, inplace=True)
    # Annualise to monthly scale: max daily × 31
    max90["max_monthly_cap"] = max90["max_daily_90d"] * 31 * CLIP_MULT
    return max90[[LOCATION_COL, ITEM_COL, "max_monthly_cap"]]


# ---------------------------------------------------------------------------
# Small prediction from view/ATC conversion if no purchase history
# ---------------------------------------------------------------------------
def estimate_from_engagement(
    sub: pd.DataFrame,
    pred_feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    For rows with prediction=0 but recent ATC activity, give a tiny estimate.
    """
    if "atc_count_7d" not in pred_feat.columns or "atc_to_purchase_rate_28d" not in pred_feat.columns:
        return sub

    engage = pred_feat[[LOCATION_COL, ITEM_COL, "atc_count_7d", "atc_to_purchase_rate_28d"]].copy()
    sub = sub.merge(engage, on=[LOCATION_COL, ITEM_COL], how="left")

    # Only adjust zero predictions with ATC signal
    mask_zero    = sub["prediction"] == 0
    mask_has_atc = sub["atc_count_7d"].fillna(0) > 0
    mask_adj     = mask_zero & mask_has_atc

    conversion = sub.loc[mask_adj, "atc_to_purchase_rate_28d"].fillna(0).clip(0, 1)
    atc_7d     = sub.loc[mask_adj, "atc_count_7d"].fillna(0)
    # Rough estimate: expected monthly purchases = atc_7d * (31/7) * conversion
    sub.loc[mask_adj, "prediction"] = (atc_7d * (31.0 / 7.0) * conversion).clip(lower=0)

    sub.drop(columns=["atc_count_7d", "atc_to_purchase_rate_28d"], inplace=True, errors="ignore")
    n_adj = mask_adj.sum()
    log.info(f"  Engagement fallback applied to {n_adj} rows")
    return sub


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_postprocess():
    # Load blended predictions
    blend_path = OUTPUT_DIR / "predictions_blended.parquet"
    feat_path  = OUTPUT_DIR / "features_predict_jan2026.parquet"

    if blend_path.exists():
        pred_df = pd.read_parquet(blend_path)
        pred_col = "prediction_blend"
    else:
        log.warning("No blended predictions found, falling back to submission_baseline.csv")
        pred_df  = pd.read_csv(OUTPUT_DIR / "submission_baseline.csv")
        pred_col = "prediction"

    items = load_items()
    purch = load_purchases(sample=CFG.get("DEBUG_SAMPLE", False))

    # Rename blended column to prediction
    if pred_col != "prediction":
        pred_df["prediction"] = pred_df[pred_col]

    sub = pred_df[[LOCATION_COL, ITEM_COL, "prediction"]].copy()
    sub["prediction"] = sub["prediction"].fillna(0).clip(lower=0)

    # ---- Step 1: Clip to max historical cap --------------------------------
    log.info("Applying max-cap clipping (90d historical × 2.0) ...")
    cutoff = pd.Timestamp("2025-12-31")
    max90  = compute_max_90d(purch, cutoff)
    sub    = sub.merge(max90, on=[LOCATION_COL, ITEM_COL], how="left")
    sub["max_monthly_cap"] = sub["max_monthly_cap"].fillna(sub["prediction"].max() * 3)
    before_clip = sub["prediction"].copy()
    sub["prediction"] = np.minimum(sub["prediction"], sub["max_monthly_cap"])
    n_clipped = (sub["prediction"] < before_clip).sum()
    log.info(f"  Clipped {n_clipped} predictions")
    sub.drop(columns=["max_monthly_cap"], inplace=True)

    # ---- Step 2: Floor tiny predictions to 0 --------------------------------
    log.info(f"Flooring predictions < {FLOOR_THR} to 0 ...")
    n_floored = (sub["prediction"].between(0, FLOOR_THR, inclusive="neither")).sum()
    sub.loc[sub["prediction"] < FLOOR_THR, "prediction"] = 0.0
    log.info(f"  Floored {n_floored} predictions")

    # ---- Step 3: Engagement fallback (zero preds with ATC signal) -----------
    if feat_path.exists():
        log.info("Applying engagement fallback for zero predictions ...")
        pred_feat = pd.read_parquet(feat_path)
        sub = estimate_from_engagement(sub, pred_feat)

    # ---- Step 4: sale_status = 0 → prediction = 0 --------------------------
    log.info("Applying sale_status=0 filter ...")
    item_status = items[[CFG["ITEM_ID_COL"], CFG["ITEM_SALE_STATUS_COL"]]].rename(
        columns={CFG["ITEM_ID_COL"]: ITEM_COL, CFG["ITEM_SALE_STATUS_COL"]: "sale_status"}
    )
    sub = sub.merge(item_status, on=ITEM_COL, how="left")
    n_zero_status = (sub["sale_status"] == 0).sum()
    sub.loc[sub["sale_status"] == 0, "prediction"] = 0.0
    log.info(f"  sale_status=0 → 0: {n_zero_status} rows")
    sub.drop(columns=["sale_status"], inplace=True)

    # ---- Final clip ≥ 0 -----------------------------------------------------
    sub["prediction"] = sub["prediction"].clip(lower=0)

    # Save
    sub_out = sub[[LOCATION_COL, ITEM_COL, "prediction"]].drop_duplicates(
        subset=[LOCATION_COL, ITEM_COL]
    )
    sub_out = sub_out.rename(columns={"prediction": SUBMISSION_PRED_COL})
    out_path = OUTPUT_DIR / "submission_final.pkl"
    sub_out.to_pickle(out_path)
    log.info(f"Final submission saved: {out_path}")

    # Stats
    print("\n=== Final Submission Statistics ===")
    print(f"  Rows    : {len(sub_out):,}")
    print(f"  Columns : {sub_out.columns.tolist()}")
    print(f"  Min     : {sub_out[SUBMISSION_PRED_COL].min():.4f}")
    print(f"  Max     : {sub_out[SUBMISSION_PRED_COL].max():.4f}")
    print(f"  Mean    : {sub_out[SUBMISSION_PRED_COL].mean():.4f}")
    print(f"  Zeros   : {(sub_out[SUBMISSION_PRED_COL] == 0).sum():,}")
    print(f"  Non-zero: {(sub_out[SUBMISSION_PRED_COL] > 0).sum():,}")
    print("\nTop-10 largest predictions:")
    print(sub_out.nlargest(10, SUBMISSION_PRED_COL).to_string(index=False))

    return sub_out


if __name__ == "__main__":
    run_postprocess()
