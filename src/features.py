"""
src/features.py
===============
Bước 4: Feature engineering.

Tạo feature table theo từng cutoff date cho:
  - Train (multiple folds)
  - Predict Jan 2026

Đơn vị: location × item_id × cutoff_month
Target: tổng purchased quantity của tháng tiếp theo

Không leak tương lai: mọi feature <= cutoff.
"""

import pandas as pd_core
try:
    import cudf as pd
    HAS_CUDF = True
    print("\n[INFO] TÌM THẤY CUDF! ĐANG SỬ DỤNG SỨC MẠNH GPU (VRAM) ĐỂ CHẠY FEATURE ENGINEERING!\n")
except ImportError:
    import pandas as pd
    HAS_CUDF = False
    print("\n[INFO] Không tìm thấy cuDF. Đang chạy bằng Pandas (CPU) bình thường.\n")
import sys
import yaml
import numpy as np
import logging
from typing import Optional
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
EPS = 1e-6


# ---------------------------------------------------------------------------
# Data loading (cached in module level)
# ---------------------------------------------------------------------------
_purch_cache: Optional[pd.DataFrame] = None
_event_cache: Optional[pd.DataFrame] = None
_items_cache: Optional[pd.DataFrame] = None


def load_data(sample: bool = False):
    global _purch_cache, _event_cache, _items_cache

    if _purch_cache is None:
        log.info("Loading transaction_full_2025.parquet ...")
        tx = pd.read_parquet(DATA_DIR / CFG["TRANSACTION_FILE"])
        if sample:
            tx = tx.sample(frac=CFG["DEBUG_SAMPLE_FRAC"], random_state=CFG["RANDOM_STATE"])
        purch = tx[tx[EVENT_COL] == PURCHASE_EVT].copy()
        
        if HAS_CUDF:
            # cuDF to_numeric is stricter, need string cast first if the type is unknown
            purch["price_num"] = purch[PRICE_COL].astype(str).astype(float)
        else:
            purch["price_num"] = pd.to_numeric(purch[PRICE_COL], errors="coerce").fillna(0)
            
        purch["revenue"]   = purch["price_num"] * purch[QTY_COL]
        # dt.floor('D') tương thích với cả pandas và cudf (thay cho dt.normalize())
        purch["date"]      = purch[DATE_COL].dt.floor('D')
        _purch_cache = purch
        log.info(f"Purchases: {len(purch):,} rows")

    if _event_cache is None:
        log.info("Loading event_full_2025.parquet ...")
        ev = pd.read_parquet(DATA_DIR / CFG["EVENT_FILE"])
        ev["date"] = ev[CFG["EV_DATE_COL"]].dt.floor('D')
        _event_cache = ev
        log.info(f"Events: {len(ev):,} rows")

    if _items_cache is None:
        log.info("Loading items.parquet ...")
        items = pd.read_parquet(DATA_DIR / CFG["ITEMS_FILE"])
        _items_cache = items
        log.info(f"Items: {len(items):,} rows")

    return _purch_cache, _event_cache, _items_cache


# ---------------------------------------------------------------------------
# Rolling aggregations helper
# ---------------------------------------------------------------------------
def rolling_sum(df, cutoff: pd_core.Timestamp, days: int,
                group_cols, val_col: str, alias: str):
    """Tổng val_col trong `days` ngày trước cutoff, grouped by group_cols."""
    start = cutoff - pd_core.Timedelta(days=days)
    window = df[(df["date"] > start) & (df["date"] <= cutoff)]
    agg = window.groupby(group_cols)[val_col].sum().reset_index().rename(columns={val_col: alias})
    return agg


def rolling_mean(df, cutoff, days, group_cols, val_col, alias):
    start = cutoff - pd_core.Timedelta(days=days)
    window = df[(df["date"] > start) & (df["date"] <= cutoff)]
    # daily mean: sum / days
    daily_sum = window.groupby(group_cols)[val_col].sum().reset_index()
    daily_sum[alias] = daily_sum[val_col] / days
    return daily_sum[[*group_cols, alias]]


def rolling_std(df, cutoff, days, group_cols, val_col, alias):
    start = cutoff - pd_core.Timedelta(days=days)
    window = df[(df["date"] > start) & (df["date"] <= cutoff)].copy()
    # group by date first, then std across days
    daily = window.groupby([*group_cols, "date"])[val_col].sum().reset_index()
    std_df = daily.groupby(group_cols)[val_col].std().reset_index().rename(columns={val_col: alias})
    return std_df


def nonzero_days(df, cutoff, days, group_cols, val_col, alias):
    start = cutoff - pd_core.Timedelta(days=days)
    window = df[(df["date"] > start) & (df["date"] <= cutoff)].copy()
    daily = window.groupby([*group_cols, "date"])[val_col].sum().reset_index()
    nz = (daily[val_col] > 0).groupby([daily[c] for c in group_cols]).sum()
    nz = nz.reset_index().rename(columns={val_col: alias})
    # alternative if above fails
    daily_pos = daily[daily[val_col] > 0]
    nz = daily_pos.groupby(group_cols)["date"].nunique().reset_index().rename(columns={"date": alias})
    return nz


def days_since_last_sale(df, cutoff, group_cols, val_col, alias="days_since_last_sale"):
    last_sale = (
        df[df[val_col] > 0]
        .groupby(group_cols)["date"]
        .max()
        .reset_index()
        .rename(columns={"date": "last_sale_date"})
    )
    # Extract days carefully for both pandas and cudf
    td = cutoff - last_sale["last_sale_date"]
    last_sale[alias] = td.dt.days
    return last_sale[group_cols + [alias]]


# ---------------------------------------------------------------------------
# Build features at a given cutoff date
# ---------------------------------------------------------------------------
def build_features_at_cutoff(
    purch,
    events,
    items,
    cutoff: pd_core.Timestamp,
    target_month_year: Optional[tuple] = None,   # (year, month)
):
    """
    Tạo feature table tại thời điểm cutoff.
    target_month_year: nếu cung cấp → tính target_sales (không leak).
    """
    log.info(f"Building features at cutoff={cutoff.date()} target={target_month_year}")

    # ---- Universe: all location x item seen before cutoff ------------------
    purch_past = purch[purch["date"] <= cutoff].copy()
    universe   = purch_past[[LOCATION_COL, ITEM_COL]].drop_duplicates().reset_index(drop=True)
    log.info(f"  Universe size: {len(universe):,}")

    feat = universe.copy()

    # ================================================================
    # GROUP A: location × item_id features (purchased)
    # ================================================================
    gc = [LOCATION_COL, ITEM_COL]

    for days, suffix in [(7, "7d"), (14, "14d"), (28, "28d"), (56, "56d"), (90, "90d")]:
        s = rolling_sum(purch_past, cutoff, days, gc, QTY_COL, f"sales_sum_{suffix}")
        feat = feat.merge(s, on=gc, how="left")

    for days, suffix in [(7, "7d"), (14, "14d"), (28, "28d"), (56, "56d"), (90, "90d")]:
        m = rolling_mean(purch_past, cutoff, days, gc, QTY_COL, f"sales_mean_{suffix}")
        feat = feat.merge(m, on=gc, how="left")

    for days, suffix in [(28, "28d"), (56, "56d")]:
        s = rolling_std(purch_past, cutoff, days, gc, QTY_COL, f"sales_std_{suffix}")
        feat = feat.merge(s, on=gc, how="left")

    for days, suffix in [(28, "28d"), (56, "56d"), (90, "90d")]:
        nz = nonzero_days(purch_past, cutoff, days, gc, QTY_COL, f"sales_nonzero_days_{suffix}")
        feat = feat.merge(nz, on=gc, how="left")

    dslast = days_since_last_sale(purch_past, cutoff, gc, QTY_COL)
    feat = feat.merge(dslast, on=gc, how="left")

    # Trend ratios
    feat["trend_7_vs_28"]  = feat["sales_mean_7d"]  / (feat["sales_mean_28d"]  + EPS)
    feat["trend_28_vs_90"] = feat["sales_mean_28d"]  / (feat["sales_mean_90d"]  + EPS)

    # Revenue features
    for days, suffix in [(7, "7d"), (28, "28d"), (90, "90d")]:
        rs = rolling_sum(purch_past, cutoff, days, gc, "revenue", f"revenue_sum_{suffix}")
        feat = feat.merge(rs, on=gc, how="left")

    # avg_price_28d / 90d
    for days, suffix in [(28, "28d"), (90, "90d")]:
        cnt = rolling_sum(purch_past, cutoff, days, gc, QTY_COL, f"_qty_{suffix}")
        rev = rolling_sum(purch_past, cutoff, days, gc, "revenue", f"_rev_{suffix}")
        feat = feat.merge(cnt.rename(columns={f"_qty_{suffix}": f"__q{suffix}"}), on=gc, how="left")
        feat = feat.merge(rev.rename(columns={f"_rev_{suffix}": f"__r{suffix}"}), on=gc, how="left")
        feat[f"avg_price_{suffix}"] = feat[f"__r{suffix}"] / (feat[f"__q{suffix}"] + EPS)
        feat.drop(columns=[f"__q{suffix}", f"__r{suffix}"], inplace=True)

    feat["price_change_28_vs_90"] = feat["avg_price_28d"] / (feat["avg_price_90d"] + EPS)

    # last price
    # Sửa sort_values: không sort toàn bộ dataframe lớn trên GPU để tránh OOM
    # Thay vào đó chỉ giữ lại những dòng có price_num > 0, groupby và lấy date max
    last_price = (
        purch_past[purch_past["price_num"] > 0][[LOCATION_COL, ITEM_COL, "date", "price_num"]]
        .sort_values([LOCATION_COL, ITEM_COL, "date"])
        .drop_duplicates(subset=[LOCATION_COL, ITEM_COL], keep='last')
        .drop(columns=["date"])
        .rename(columns={"price_num": "last_price"})
    )
    feat = feat.merge(last_price, on=gc, how="left")

    # ================================================================
    # GROUP B: item-level (across all locations)
    # ================================================================
    gi = [ITEM_COL]

    for days, suffix in [(7, "7d"), (28, "28d"), (90, "90d")]:
        s = rolling_sum(purch_past, cutoff, days, gi, QTY_COL, f"item_sales_sum_{suffix}")
        feat = feat.merge(s, on=gi, how="left")

    for days, suffix in [(28, "28d"), (90, "90d")]:
        start = cutoff - pd_core.Timedelta(days=days)
        w = purch_past[(purch_past["date"] > start) & (purch_past["date"] <= cutoff)]
        nloc = w.groupby(gi)[LOCATION_COL].nunique().reset_index().rename(
            columns={LOCATION_COL: f"item_num_locations_sold_{suffix}"})
        feat = feat.merge(nloc, on=gi, how="left")

    item_sum28  = feat.groupby(gi)["item_sales_sum_28d"].first().reset_index()
    item_sum90  = feat.groupby(gi)["item_sales_sum_90d"].first().reset_index()
    item_trend  = item_sum28.merge(item_sum90, on=gi, how="left")
    item_trend["item_global_trend_28_vs_90"] = (
        item_trend["item_sales_sum_28d"] / (item_trend["item_sales_sum_90d"] + EPS)
    )
    feat = feat.merge(item_trend[[gi[0], "item_global_trend_28_vs_90"]], on=gi, how="left")

    # ================================================================
    # GROUP C: location-level
    # ================================================================
    gl = [LOCATION_COL]

    for days, suffix in [(7, "7d"), (28, "28d"), (90, "90d")]:
        s = rolling_sum(purch_past, cutoff, days, gl, QTY_COL, f"location_sales_sum_{suffix}")
        feat = feat.merge(s, on=gl, how="left")

    for days, suffix in [(28, "28d"), (90, "90d")]:
        start = cutoff - pd_core.Timedelta(days=days)
        w = purch_past[(purch_past["date"] > start) & (purch_past["date"] <= cutoff)]
        nitems = w.groupby(gl)[ITEM_COL].nunique().reset_index().rename(
            columns={ITEM_COL: f"location_active_items_{suffix}"})
        feat = feat.merge(nitems, on=gl, how="left")

    loc_sum28 = feat.groupby(gl)["location_sales_sum_28d"].first().reset_index()
    loc_sum90 = feat.groupby(gl)["location_sales_sum_90d"].first().reset_index()
    loc_trend = loc_sum28.merge(loc_sum90, on=gl, how="left")
    loc_trend["location_sales_trend_28_vs_90"] = (
        loc_trend["location_sales_sum_28d"] / (loc_trend["location_sales_sum_90d"] + EPS)
    )
    feat = feat.merge(loc_trend[[gl[0], "location_sales_trend_28_vs_90"]], on=gl, how="left")

    # ================================================================
    # GROUP D: Category features (if available)
    # ================================================================
    if CFG["ITEM_CATEGORY_COL"] in items.columns:
        cat_map = items[[CFG["ITEM_ID_COL"], CFG["ITEM_CATEGORY_COL"]]].rename(
            columns={CFG["ITEM_ID_COL"]: ITEM_COL,
                     CFG["ITEM_CATEGORY_COL"]: "category"}
        )
        feat = feat.merge(cat_map, on=ITEM_COL, how="left")

        # category sales
        purch_past_cat = purch_past.merge(cat_map, on=ITEM_COL, how="left")
        for days, suffix in [(28, "28d")]:
            cat_sum = rolling_sum(purch_past_cat, cutoff, days, ["category"], QTY_COL,
                                  f"category_sales_sum_{suffix}")
            feat = feat.merge(cat_sum, on="category", how="left")

        # item share in category
        item_cat_sum = rolling_sum(purch_past_cat, cutoff, 28, [ITEM_COL, "category"], QTY_COL,
                                   "item_cat_qty_28d")
        feat = feat.merge(item_cat_sum, on=[ITEM_COL, "category"], how="left")
        feat["item_share_in_category"] = (
            feat["item_cat_qty_28d"] / (feat["category_sales_sum_28d"] + EPS)
        )
        feat.drop(columns=["item_cat_qty_28d"], inplace=True, errors="ignore")

    # ================================================================
    # GROUP E: Event features (view_item / add_to_cart by item_id)
    # ================================================================
    events_past = events[events["date"] <= cutoff].copy()

    ev_view = events_past[events_past[CFG["EV_EVENT_COL"]] == CFG["EV_VIEW_EVENT"]]
    ev_atc  = events_past[events_past[CFG["EV_EVENT_COL"]] == CFG["EV_ATC_EVENT"]]

    for days, suffix in [(1, "1d"), (3, "3d"), (7, "7d"), (14, "14d"), (28, "28d")]:
        sv = rolling_sum(ev_view, cutoff, days, [ITEM_COL], CFG["EV_QTY_COL"], f"view_count_{suffix}")
        feat = feat.merge(sv, on=ITEM_COL, how="left")
        sa = rolling_sum(ev_atc,  cutoff, days, [ITEM_COL], CFG["EV_QTY_COL"], f"atc_count_{suffix}")
        feat = feat.merge(sa, on=ITEM_COL, how="left")

    # Ratio features
    feat["view_to_atc_rate_28d"]    = feat["atc_count_28d"]   / (feat["view_count_28d"]   + EPS)
    feat["atc_to_purchase_rate_28d"]= feat["sales_sum_28d"]   / (feat["atc_count_28d"]    + EPS)
    feat["view_to_purchase_rate_28d"]= feat["sales_sum_28d"]  / (feat["view_count_28d"]   + EPS)
    feat["recent_view_ratio"]       = feat["view_count_7d"]   / (feat["view_count_28d"]   + EPS)
    feat["recent_atc_ratio"]        = feat["atc_count_7d"]    / (feat["atc_count_28d"]    + EPS)

    # ================================================================
    # GROUP F: Calendar features
    # ================================================================
    target_month = target_month_year[1] if target_month_year else (cutoff.month % 12) + 1
    target_year  = target_month_year[0] if target_month_year else (
        cutoff.year + 1 if cutoff.month == 12 else cutoff.year
    )
    import calendar
    days_in_target = calendar.monthrange(target_year, target_month)[1]

    feat["cutoff_month"]           = cutoff.month
    feat["target_month"]           = target_month
    feat["days_in_target_month"]   = days_in_target
    feat["is_january_target"]      = int(target_month == 1)
    feat["quarter"]                = (target_month - 1) // 3 + 1

    # ================================================================
    # GROUP G: Items metadata
    # ================================================================
    item_meta = items[[CFG["ITEM_ID_COL"], CFG["ITEM_SALE_STATUS_COL"]]].rename(
        columns={CFG["ITEM_ID_COL"]: ITEM_COL,
                 CFG["ITEM_SALE_STATUS_COL"]: "sale_status"}
    )
    feat = feat.merge(item_meta, on=ITEM_COL, how="left")

    # ================================================================
    # TARGET (if specified and not leaking future)
    # ================================================================
    if target_month_year is not None:
        ty, tm = target_month_year
        mask = (purch["date"].dt.year == ty) & (purch["date"].dt.month == tm)
        target_df = (
            purch[mask]
            .groupby([LOCATION_COL, ITEM_COL])
            .agg(sales_next_month=(QTY_COL, "sum"), revenue_next_month=("revenue", "sum"))
            .reset_index()
        )
        feat = feat.merge(target_df, on=[LOCATION_COL, ITEM_COL], how="left")
        feat["sales_next_month"]   = feat["sales_next_month"].fillna(0)
        feat["revenue_next_month"] = feat["revenue_next_month"].fillna(0)

    # ================================================================
    # Fill NaN
    # ================================================================
    fill_zero_cols = [c for c in feat.columns if c not in [LOCATION_COL, ITEM_COL, "category"]]
    feat[fill_zero_cols] = feat[fill_zero_cols].fillna(0)

    log.info(f"  Feature table shape: {feat.shape}")
    return feat


# ---------------------------------------------------------------------------
# Build train + predict tables
# ---------------------------------------------------------------------------
def build_all_features():
    sample = CFG.get("DEBUG_SAMPLE", False)
    purch, events, items = load_data(sample=sample)

    # Validation folds
    folds = [
        # cutoff date, (target_year, target_month), save_name
        (pd_core.Timestamp("2025-10-31"), (2025, 11), "features_val_nov"),
        (pd_core.Timestamp("2025-11-30"), (2025, 12), "features_val_dec"),
        (pd_core.Timestamp("2025-12-31"), None,       "features_predict_jan2026"),
    ]

    all_train_parts = []
    for cutoff, tgt, name in folds:
        feat = build_features_at_cutoff(purch, events, items, cutoff, tgt)
        out = OUTPUT_DIR / f"{name}.parquet"
        # Convert to Pandas before saving if we are in cuDF, for cross-script safety
        if HAS_CUDF:
            feat_pd = feat.to_pandas()
            feat_pd.to_parquet(out, index=False)
            log.info(f"Saved {out} (from cuDF) [{feat.shape}]")
        else:
            feat.to_parquet(out, index=False)
            log.info(f"Saved {out} [{feat.shape}]")
        
        if tgt is not None:
            if HAS_CUDF:
                all_train_parts.append(feat_pd)
            else:
                all_train_parts.append(feat)

    # Combine train
    if all_train_parts:
        if HAS_CUDF:
            train_all = pd_core.concat(all_train_parts, ignore_index=True)
        else:
            train_all = pd.concat(all_train_parts, ignore_index=True)
        train_out = OUTPUT_DIR / "features_train.parquet"
        train_all.to_parquet(train_out, index=False)
        log.info(f"Combined train features saved: {train_out} [{train_all.shape}]")

    return all_train_parts


if __name__ == "__main__":
    build_all_features()
