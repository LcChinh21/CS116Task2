"""
src/metrics.py
==============
Metric functions cho bài toán Sale Forecasting.

MAPE chính: chỉ tính trên y_true > 0, chỉ tính location có phát sinh giao dịch.
"""

import numpy as np
import pandas as pd
from typing import Optional


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE chỉ tính trên y_true > 0."""
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    return float(np.mean(np.abs(yt - yp) / np.abs(yt)) * 100)


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def evaluate(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    location_col: str = "location",
    item_col: str = "item_id",
    qty_col: str = "sales",
    revenue_col: Optional[str] = "revenue",
    pred_col: str = "prediction",
    pred_revenue_col: Optional[str] = None,
) -> dict:
    """
    Tính MAE/MAPE trên sales và revenue.

    Parameters
    ----------
    df_true : DataFrame, các cột [location, item_id, sales, (revenue)]
        Chỉ chứa những location × item_id có phát sinh giao dịch (y_true > 0).
    df_pred : DataFrame, các cột [location, item_id, prediction, (pred_revenue)]

    Returns
    -------
    dict với keys: mae_sales, mape_sales, mae_revenue, mape_revenue
    """
    merged = df_true.merge(
        df_pred[[location_col, item_col, pred_col]],
        on=[location_col, item_col],
        how="left",
    )
    merged[pred_col] = merged[pred_col].fillna(0).clip(lower=0)

    y_true_qty = merged[qty_col].values.astype(float)
    y_pred_qty = merged[pred_col].values.astype(float)

    result = {
        "mae_sales":  _safe_mae(y_true_qty, y_pred_qty),
        "mape_sales": _safe_mape(y_true_qty, y_pred_qty),
    }

    if revenue_col is not None and revenue_col in merged.columns:
        y_true_rev = merged[revenue_col].values.astype(float)
        # Estimate predicted revenue from price ratio
        if pred_revenue_col and pred_revenue_col in merged.columns:
            y_pred_rev = merged[pred_revenue_col].values.astype(float)
        else:
            # avg_price per item    
            avg_price = df_true.copy()
            avg_price["avg_price"] = avg_price[revenue_col] / avg_price[qty_col].replace(0, np.nan)
            avg_price = avg_price[[item_col, "avg_price"]].groupby(item_col).mean().reset_index()
            merged2 = merged.merge(avg_price, on=item_col, how="left")
            y_pred_rev = (y_pred_qty * merged2["avg_price"].fillna(0)).values

        result["mae_revenue"]  = _safe_mae(y_true_rev, y_pred_rev)
        result["mape_revenue"] = _safe_mape(y_true_rev, y_pred_rev)

    return result


def print_metrics(metrics: dict, label: str = ""):
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}MAE  Sales   : {metrics.get('mae_sales', 'N/A'):.4f}")
    print(f"{prefix}MAPE Sales   : {metrics.get('mape_sales', 'N/A'):.4f}%  ← main metric")
    if "mae_revenue" in metrics:
        print(f"{prefix}MAE  Revenue : {metrics.get('mae_revenue', 'N/A'):.2f}")
        print(f"{prefix}MAPE Revenue : {metrics.get('mape_revenue', 'N/A'):.4f}%")
