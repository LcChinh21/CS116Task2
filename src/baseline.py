"""
src/baseline.py
===============
Bước 2: Tạo baseline submission.

Baseline strategies:
  1. Last month (Dec 2025) sales per location × item_id
  2. Rolling 28/56/90 ngày → scale lên 31 ngày (Jan 2026)
  3. Fallback: item global avg → 0

Output: outputs/submission_baseline.csv
"""

import os
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

DATA_DIR   = REPO_ROOT / CFG["DATA_DIR"]
OUTPUT_DIR = REPO_ROOT / CFG["OUTPUT_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCATION_COL = CFG["TX_LOCATION_COL"]
ITEM_COL     = CFG["TX_ITEM_COL"]
DATE_COL     = CFG["TX_DATE_COL"]
QTY_COL      = CFG["TX_QTY_COL"]
PRICE_COL    = CFG["TX_PRICE_COL"]
EVENT_COL    = CFG["TX_EVENT_COL"]
PURCHASE_EVT = CFG["TX_PURCHASE_EVENT"]
DAYS_JAN     = 31   # days in January 2026


# ---------------------------------------------------------------------------
# Load purchase data
# ---------------------------------------------------------------------------
def load_purchases(sample: bool = False) -> pd.DataFrame:
    log.info("Loading transaction_full_2025.parquet ...")
    tx = pd.read_parquet(DATA_DIR / CFG["TRANSACTION_FILE"])
    if sample:
        tx = tx.sample(frac=CFG["DEBUG_SAMPLE_FRAC"], random_state=CFG["RANDOM_STATE"])
    purch = tx[tx[EVENT_COL] == PURCHASE_EVT].copy()
    purch["price_num"] = pd.to_numeric(purch[PRICE_COL], errors="coerce")
    purch["revenue"]   = purch["price_num"] * purch[QTY_COL]
    purch["date"]      = purch[DATE_COL].dt.normalize()
    log.info(f"Purchased rows: {len(purch):,}")
    return purch


# ---------------------------------------------------------------------------
# Load items (for sale_status)
# ---------------------------------------------------------------------------
def load_items() -> pd.DataFrame:
    items = pd.read_parquet(DATA_DIR / CFG["ITEMS_FILE"])
    return items[[CFG["ITEM_ID_COL"], CFG["ITEM_SALE_STATUS_COL"]]]


# ---------------------------------------------------------------------------
# Build prediction frame (all location × item_id seen in history)
# ---------------------------------------------------------------------------
def build_prediction_frame(purch: pd.DataFrame) -> pd.DataFrame:
    combos = (
        purch[[LOCATION_COL, ITEM_COL]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    log.info(f"Unique location × item_id combos: {len(combos):,}")
    return combos


# ---------------------------------------------------------------------------
# Strategy 1: Last month (Dec 2025)
# ---------------------------------------------------------------------------
def last_month_baseline(purch: pd.DataFrame) -> pd.DataFrame:
    dec = purch[purch["date"].dt.month == 12]
    agg = (
        dec.groupby([LOCATION_COL, ITEM_COL])[QTY_COL]
        .sum()
        .reset_index()
        .rename(columns={QTY_COL: "pred_lastmonth"})
    )
    return agg


# ---------------------------------------------------------------------------
# Strategy 2: Rolling window baseline
# ---------------------------------------------------------------------------
def rolling_baseline(purch: pd.DataFrame, cutoff_date: pd.Timestamp, days: int) -> pd.DataFrame:
    start = cutoff_date - pd.Timedelta(days=days)
    window = purch[(purch["date"] > start) & (purch["date"] <= cutoff_date)]
    agg = (
        window.groupby([LOCATION_COL, ITEM_COL])[QTY_COL]
        .sum()
        .reset_index()
        .rename(columns={QTY_COL: f"pred_rolling{days}d"})
    )
    # Scale to 31 days (January)
    agg[f"pred_rolling{days}d"] = agg[f"pred_rolling{days}d"] * DAYS_JAN / days
    return agg


# ---------------------------------------------------------------------------
# Fallback: item global average (across all locations) over last 90d
# ---------------------------------------------------------------------------
def item_global_avg(purch: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    start = cutoff_date - pd.Timedelta(days=90)
    window = purch[(purch["date"] > start) & (purch["date"] <= cutoff_date)]
    agg = (
        window.groupby(ITEM_COL)[QTY_COL]
        .mean()
        .reset_index()
        .rename(columns={QTY_COL: "item_global_avg"})
    )
    return agg


# ---------------------------------------------------------------------------
# Assemble baseline submission
# ---------------------------------------------------------------------------
def make_baseline_submission(cutoff_date: pd.Timestamp, purch: pd.DataFrame) -> pd.DataFrame:
    pred_frame = build_prediction_frame(purch)

    last_mo  = last_month_baseline(purch)
    roll28   = rolling_baseline(purch, cutoff_date, 28)
    roll56   = rolling_baseline(purch, cutoff_date, 56)
    roll90   = rolling_baseline(purch, cutoff_date, 90)
    item_avg = item_global_avg(purch, cutoff_date)

    df = pred_frame.copy()
    df = df.merge(last_mo,  on=[LOCATION_COL, ITEM_COL], how="left")
    df = df.merge(roll28,   on=[LOCATION_COL, ITEM_COL], how="left")
    df = df.merge(roll56,   on=[LOCATION_COL, ITEM_COL], how="left")
    df = df.merge(roll90,   on=[LOCATION_COL, ITEM_COL], how="left")
    df = df.merge(item_avg, on=ITEM_COL, how="left")

    # Priority: last month > rolling90 > rolling56 > rolling28 > item_avg > 0
    # Vectorised (much faster than row-apply)
    pred = np.zeros(len(df))
    for col in ["pred_rolling28d", "pred_rolling56d", "pred_rolling90d", "pred_lastmonth"]:
        if col in df.columns:
            has_val = df[col].notna() & (df[col] >= 0)
            pred = np.where(has_val, df[col].fillna(0), pred)
    # item_avg as fallback only where still 0 and not seen in any of above
    if "item_global_avg" in df.columns:
        all_nan = (
            df["pred_lastmonth"].isna() &
            df.get("pred_rolling90d", pd.Series(np.nan, index=df.index)).isna() &
            df.get("pred_rolling56d", pd.Series(np.nan, index=df.index)).isna() &
            df.get("pred_rolling28d", pd.Series(np.nan, index=df.index)).isna()
        )
        pred = np.where(all_nan & df["item_global_avg"].notna(), df["item_global_avg"].fillna(0), pred)

    df["prediction"] = np.clip(pred, 0, None)
    return df[[LOCATION_COL, ITEM_COL, "prediction"]]


# ---------------------------------------------------------------------------
# Apply sale_status filter
# ---------------------------------------------------------------------------
def apply_sale_status(sub: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    merged = sub.merge(
        items.rename(columns={CFG["ITEM_ID_COL"]: ITEM_COL}),
        on=ITEM_COL,
        how="left",
    )
    sale_status_col = CFG["ITEM_SALE_STATUS_COL"]
    if sale_status_col in merged.columns:
        n_zero = (merged[sale_status_col] == 0).sum()
        log.info(f"  sale_status=0 items set to 0: {n_zero}")
        merged.loc[merged[sale_status_col] == 0, "prediction"] = 0.0
    return merged[[LOCATION_COL, ITEM_COL, "prediction"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_baseline():
    sample = CFG.get("DEBUG_SAMPLE", False)
    purch  = load_purchases(sample=sample)
    items  = load_items()

    # Cutoff = last day of Dec 2025 (train data end)
    cutoff = pd.Timestamp("2025-12-31")

    log.info("Building baseline submission ...")
    sub = make_baseline_submission(cutoff, purch)
    sub = apply_sale_status(sub, items)

    # Ensure no negatives
    sub["prediction"] = sub["prediction"].clip(lower=0)

    out_path = OUTPUT_DIR / "submission_baseline.csv"
    sub.to_csv(out_path, index=False)
    log.info(f"Submission saved: {out_path}")

    # Print stats
    print("\n=== Baseline Prediction Statistics ===")
    print(f"  Rows          : {len(sub):,}")
    print(f"  Min           : {sub['prediction'].min():.4f}")
    print(f"  Max           : {sub['prediction'].max():.4f}")
    print(f"  Mean          : {sub['prediction'].mean():.4f}")
    print(f"  Zeros         : {(sub['prediction'] == 0).sum():,}")
    print(f"  Non-zero      : {(sub['prediction'] > 0).sum():,}")

    return sub


if __name__ == "__main__":
    run_baseline()
