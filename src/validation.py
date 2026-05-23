"""
src/validation.py
=================
Bước 3: Time-based validation.

Folds:
  - Fold 1: Train Jan-Oct 2025 → Validate Nov 2025
  - Fold 2: Train Jan-Nov 2025 → Validate Dec 2025
  - Final : Train Jan-Dec 2025 → Predict Jan 2026
"""

import sys
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

with open(REPO_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

from metrics import evaluate, print_metrics
from baseline import load_purchases, load_items, make_baseline_submission, apply_sale_status

DATA_DIR   = REPO_ROOT / CFG["DATA_DIR"]
OUTPUT_DIR = REPO_ROOT / CFG["OUTPUT_DIR"]
REPORT_DIR = REPO_ROOT / CFG["REPORT_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOCATION_COL = CFG["TX_LOCATION_COL"]
ITEM_COL     = CFG["TX_ITEM_COL"]
DATE_COL     = CFG["TX_DATE_COL"]
QTY_COL      = CFG["TX_QTY_COL"]
PURCHASE_EVT = CFG["TX_PURCHASE_EVENT"]
EVENT_COL    = CFG["TX_EVENT_COL"]


# ---------------------------------------------------------------------------
# Build ground truth for a given month
# ---------------------------------------------------------------------------
def build_ground_truth(purch: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Trả về DataFrame [location, item_id, sales, revenue] cho tháng year/month.
    Chỉ gồm các location × item_id có purchased > 0 (chỉ tính location active).
    """
    mask = (purch["date"].dt.year == year) & (purch["date"].dt.month == month)
    target = purch[mask].copy()
    agg = (
        target.groupby([LOCATION_COL, ITEM_COL])
        .agg(sales=(QTY_COL, "sum"), revenue=("revenue", "sum"))
        .reset_index()
    )
    agg = agg[agg["sales"] > 0]
    return agg


# ---------------------------------------------------------------------------
# Validation folds
# ---------------------------------------------------------------------------
FOLDS = [
    {"name": "Val_Nov2025", "train_cutoff": "2025-10-31", "val_year": 2025, "val_month": 11},
    {"name": "Val_Dec2025", "train_cutoff": "2025-11-30", "val_year": 2025, "val_month": 12},
]


def run_fold_baseline(purch_full: pd.DataFrame, fold: dict, items: pd.DataFrame) -> dict:
    """
    Chạy baseline strategy trên một fold.
    """
    cutoff = pd.Timestamp(fold["train_cutoff"])
    # chỉ dùng data <= cutoff cho train
    purch_train = purch_full[purch_full["date"] <= cutoff]

    sub = make_baseline_submission(cutoff, purch_train)
    sub = apply_sale_status(sub, items)

    gt = build_ground_truth(purch_full, fold["val_year"], fold["val_month"])
    # Filter: only evaluate on items with sale_status != 0
    item_ids = items.copy()
    item_ids = item_ids[item_ids[CFG["ITEM_SALE_STATUS_COL"]] != 0][CFG["ITEM_ID_COL"]]
    gt = gt[gt[ITEM_COL].isin(item_ids)]

    metrics = evaluate(
        df_true=gt,
        df_pred=sub,
        location_col=LOCATION_COL,
        item_col=ITEM_COL,
        qty_col="sales",
        revenue_col="revenue",
        pred_col="prediction",
    )
    log.info(f"  [{fold['name']} Baseline] MAE={metrics['mae_sales']:.4f} | MAPE={metrics['mape_sales']:.4f}%")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_validation():
    sample = CFG.get("DEBUG_SAMPLE", False)
    purch  = load_purchases(sample=sample)
    items  = load_items()

    results = []
    for fold in FOLDS:
        log.info(f"=== Running fold: {fold['name']} ===")
        metrics = run_fold_baseline(purch, fold, items)
        metrics["fold"] = fold["name"]
        results.append(metrics)
        print_metrics(metrics, label=fold["name"])

    # Save results table
    results_df = pd.DataFrame(results)
    results_df = results_df[["fold", "mae_sales", "mape_sales", "mae_revenue", "mape_revenue"]]
    log.info("\n" + results_df.to_string(index=False))

    out_path = OUTPUT_DIR / "validation_baseline_results.csv"
    results_df.to_csv(out_path, index=False)
    log.info(f"Results saved to {out_path}")

    return results_df


if __name__ == "__main__":
    run_validation()
