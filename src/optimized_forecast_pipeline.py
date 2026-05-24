"""
Optimized monthly pipeline for CS116 Task 2 sale forecasting.

Validation:
  - train targets: 2025-04 .. 2025-11, features use months before target
  - validation target: 2025-12
  - final train targets: 2025-04 .. 2025-12
  - final prediction: 2026-01, features use 2025-01 .. 2025-12

The pipeline is intentionally notebook-friendly:
  - one script entry point
  - row-group parquet aggregation before feature building
  - float32 feature matrices
  - elapsed-time logging for each major step
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import resource
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    start_rss = current_rss_mb()
    log.info("START %s | rss=%.1f MB", name, start_rss)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        end_rss = current_rss_mb()
        log.info("DONE  %s in %.1fs | rss=%.1f MB | delta=%.1f MB", name, elapsed, end_rss, end_rss - start_rss)


def current_rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def cleanup() -> None:
    gc.collect()


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()

DATA_DIR = REPO_ROOT / CFG["DATA_DIR"]
OUTPUT_DIR = REPO_ROOT / CFG["OUTPUT_DIR"]
MODEL_DIR = REPO_ROOT / CFG["MODEL_DIR"]
REPORT_DIR = REPO_ROOT / CFG["REPORT_DIR"]
for directory in (OUTPUT_DIR, MODEL_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LOCATION_COL = CFG["TX_LOCATION_COL"]
ITEM_COL = CFG["TX_ITEM_COL"]
DATE_COL = CFG["TX_DATE_COL"]
QTY_COL = CFG["TX_QTY_COL"]
PRICE_COL = CFG["TX_PRICE_COL"]
EVENT_COL = CFG["TX_EVENT_COL"]
PURCHASE_EVT = CFG["TX_PURCHASE_EVENT"]
EV_DATE_COL = CFG["EV_DATE_COL"]
EV_EVENT_COL = CFG["EV_EVENT_COL"]
EV_QTY_COL = CFG["EV_QTY_COL"]
VIEW_EVT = CFG["EV_VIEW_EVENT"]
ATC_EVT = CFG["EV_ATC_EVENT"]
SALE_STATUS_COL = CFG["ITEM_SALE_STATUS_COL"]
ITEM_PRICE_COL = CFG["ITEM_PRICE_COL"]
RANDOM_STATE = env_int("RANDOM_STATE", int(CFG.get("RANDOM_STATE", 42)))
EPS = 1e-6
MONTHS = np.arange(1, 13, dtype=np.int16)


@dataclass
class MonthlyData:
    pairs: pd.DataFrame
    purchase_monthly: pd.DataFrame
    pair_qty: np.ndarray
    pair_rev: np.ndarray
    item_qty: np.ndarray
    item_rev: np.ndarray
    loc_qty: np.ndarray
    loc_rev: np.ndarray
    cat_qty: np.ndarray
    brand_qty: np.ndarray
    item_active_locs: np.ndarray
    loc_active_items: np.ndarray
    view_count: np.ndarray
    atc_count: np.ndarray
    item_price: np.ndarray
    n_items: int
    n_locations: int
    n_categories: int
    n_brands: int


def parquet_schema_names(path: Path) -> List[str]:
    return pq.ParquetFile(path).schema_arrow.names


def choose_existing_column(names: Iterable[str], candidates: Iterable[Optional[str]]) -> Optional[str]:
    available = set(names)
    for candidate in candidates:
        if candidate and candidate in available:
            return candidate
    return None


def iter_parquet_row_groups(path: Path, columns: List[str], max_row_groups: int = 0):
    parquet_file = pq.ParquetFile(path)
    n_groups = parquet_file.metadata.num_row_groups
    if max_row_groups > 0:
        n_groups = min(n_groups, max_row_groups)
    for row_group in range(n_groups):
        table = parquet_file.read_row_group(row_group, columns=columns)
        yield row_group, table.to_pandas()


def to_float32(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str), errors="coerce").fillna(0).astype("float32")


def aggregate_purchases(max_row_groups: int = 0) -> pd.DataFrame:
    path = DATA_DIR / CFG["TRANSACTION_FILE"]
    columns = [LOCATION_COL, ITEM_COL, DATE_COL, QTY_COL, PRICE_COL, EVENT_COL]
    partials: List[pd.DataFrame] = []

    with timer("aggregate purchased rows to location-item-month"):
        for row_group, df in iter_parquet_row_groups(path, columns, max_row_groups=max_row_groups):
            df = df[df[EVENT_COL].astype(str).eq(PURCHASE_EVT)].copy()
            if df.empty:
                continue

            df[LOCATION_COL] = pd.to_numeric(df[LOCATION_COL], errors="coerce").fillna(-1).astype("int32")
            df = df[df[LOCATION_COL] >= 0]
            df[ITEM_COL] = df[ITEM_COL].astype(str)
            df["month_idx"] = pd.to_datetime(df[DATE_COL]).dt.month.astype("int8")
            df[QTY_COL] = pd.to_numeric(df[QTY_COL], errors="coerce").fillna(0).astype("float32")
            df["price_num"] = to_float32(df[PRICE_COL])
            df["revenue"] = (df[QTY_COL] * df["price_num"]).astype("float32")

            grouped = (
                df.groupby([LOCATION_COL, ITEM_COL, "month_idx"], observed=True, sort=False)
                .agg(quantity=(QTY_COL, "sum"), revenue=("revenue", "sum"))
                .reset_index()
            )
            partials.append(grouped)
            if row_group % 10 == 0:
                log.info("  processed transaction row_group=%s partials=%s", row_group, len(partials))
            del df, grouped
            cleanup()

        if not partials:
            raise RuntimeError("No purchased rows were found.")

        monthly = pd.concat(partials, ignore_index=True)
        del partials
        monthly = (
            monthly.groupby([LOCATION_COL, ITEM_COL, "month_idx"], observed=True, sort=False)
            .agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"))
            .reset_index()
        )
        monthly["quantity"] = monthly["quantity"].astype("float32")
        monthly["revenue"] = monthly["revenue"].astype("float32")
        log.info("purchase_monthly shape=%s", monthly.shape)
        return monthly


def aggregate_events(max_row_groups: int = 0) -> pd.DataFrame:
    path = DATA_DIR / CFG["EVENT_FILE"]
    columns = [ITEM_COL, EV_DATE_COL, EV_EVENT_COL, EV_QTY_COL]
    partials: List[pd.DataFrame] = []

    with timer("aggregate view/add-to-cart rows to item-month"):
        for row_group, df in iter_parquet_row_groups(path, columns, max_row_groups=max_row_groups):
            event_name = df[EV_EVENT_COL].astype(str)
            df = df[event_name.isin([VIEW_EVT, ATC_EVT])].copy()
            if df.empty:
                continue

            df[ITEM_COL] = df[ITEM_COL].astype(str)
            df["month_idx"] = pd.to_datetime(df[EV_DATE_COL]).dt.month.astype("int8")
            df[EV_QTY_COL] = pd.to_numeric(df[EV_QTY_COL], errors="coerce").fillna(0).astype("float32")
            df["is_view"] = df[EV_EVENT_COL].astype(str).eq(VIEW_EVT).astype("int8")
            df["is_atc"] = df[EV_EVENT_COL].astype(str).eq(ATC_EVT).astype("int8")
            df["view_qty"] = np.where(df["is_view"].to_numpy(dtype=bool), df[EV_QTY_COL].to_numpy(), 0).astype("float32")
            df["atc_qty"] = np.where(df["is_atc"].to_numpy(dtype=bool), df[EV_QTY_COL].to_numpy(), 0).astype("float32")

            grouped = (
                df.groupby([ITEM_COL, "month_idx"], observed=True, sort=False)
                .agg(
                    view_count=("is_view", "sum"),
                    atc_count=("is_atc", "sum"),
                    view_qty=("view_qty", "sum"),
                    atc_qty=("atc_qty", "sum"),
                )
                .reset_index()
            )
            partials.append(grouped)
            if row_group % 10 == 0:
                log.info("  processed event row_group=%s partials=%s", row_group, len(partials))
            del df, grouped
            cleanup()

        if not partials:
            return pd.DataFrame(columns=[ITEM_COL, "month_idx", "view_count", "atc_count", "view_qty", "atc_qty"])

        monthly = pd.concat(partials, ignore_index=True)
        del partials
        monthly = (
            monthly.groupby([ITEM_COL, "month_idx"], observed=True, sort=False)
            .agg(
                view_count=("view_count", "sum"),
                atc_count=("atc_count", "sum"),
                view_qty=("view_qty", "sum"),
                atc_qty=("atc_qty", "sum"),
            )
            .reset_index()
        )
        for col in ["view_count", "atc_count", "view_qty", "atc_qty"]:
            monthly[col] = monthly[col].astype("float32")
        log.info("event_monthly shape=%s", monthly.shape)
        return monthly


def load_items() -> pd.DataFrame:
    path = DATA_DIR / CFG["ITEMS_FILE"]
    names = parquet_schema_names(path)
    category_col = choose_existing_column(
        names,
        [CFG.get("ITEM_CATEGORY_COL"), "category_l1", "category_lv1", "category", "category_l2", "category_l3"],
    )
    brand_col = choose_existing_column(names, ["brand", "manufacturer"])
    item_price_col = choose_existing_column(names, [ITEM_PRICE_COL, "price"])
    columns = [ITEM_COL, SALE_STATUS_COL]
    for col in [category_col, brand_col, item_price_col]:
        if col and col not in columns:
            columns.append(col)

    with timer("load item metadata"):
        items = pd.read_parquet(path, columns=columns)
        items[ITEM_COL] = items[ITEM_COL].astype(str)
        items[SALE_STATUS_COL] = pd.to_numeric(items[SALE_STATUS_COL], errors="coerce").fillna(0).astype("int8")
        items["category"] = items[category_col].astype(str) if category_col else "__missing__"
        items["brand"] = items[brand_col].astype(str) if brand_col else "__missing__"
        if item_price_col:
            items["item_price"] = to_float32(items[item_price_col])
        else:
            items["item_price"] = np.float32(0.0)
        result = items[[ITEM_COL, SALE_STATUS_COL, "category", "brand", "item_price"]].drop_duplicates(ITEM_COL)
        log.info("items shape=%s category_col=%s brand_col=%s", result.shape, category_col, brand_col)
        return result


def encode_monthly_tables(purchase_monthly: pd.DataFrame, events_monthly: pd.DataFrame, items: pd.DataFrame) -> MonthlyData:
    with timer("encode ids and build monthly arrays"):
        df = purchase_monthly.merge(items, on=ITEM_COL, how="left")
        df[SALE_STATUS_COL] = df[SALE_STATUS_COL].fillna(1).astype("int8")
        df["category"] = df["category"].fillna("__missing__").astype(str)
        df["brand"] = df["brand"].fillna("__missing__").astype(str)
        df["item_price"] = df["item_price"].fillna(0).astype("float32")
        df = df[df[SALE_STATUS_COL] != 0].copy()

        item_cat = pd.Categorical(df[ITEM_COL].astype(str))
        loc_cat = pd.Categorical(df[LOCATION_COL].astype("int32"))
        cat_cat = pd.Categorical(df["category"].astype(str))
        brand_cat = pd.Categorical(df["brand"].astype(str))
        df["item_code"] = item_cat.codes.astype("int32")
        df["location_code"] = loc_cat.codes.astype("int32")
        df["category_code"] = cat_cat.codes.astype("int32")
        df["brand_code"] = brand_cat.codes.astype("int32")

        pair_keys = df[[LOCATION_COL, ITEM_COL]].drop_duplicates().reset_index(drop=True)
        pair_keys["pair_id"] = np.arange(len(pair_keys), dtype=np.int32)
        df = df.merge(pair_keys, on=[LOCATION_COL, ITEM_COL], how="left")

        pairs = (
            df.groupby("pair_id", observed=True, sort=True)
            .agg(
                location=(LOCATION_COL, "first"),
                item_id=(ITEM_COL, "first"),
                location_code=("location_code", "first"),
                item_code=("item_code", "first"),
                category_code=("category_code", "first"),
                brand_code=("brand_code", "first"),
                sale_status=(SALE_STATUS_COL, "first"),
                item_price=("item_price", "first"),
            )
            .reset_index()
            .sort_values("pair_id")
            .reset_index(drop=True)
        )
        hist_price = df[df["quantity"] > 0].copy()
        hist_price["hist_unit_price"] = hist_price["revenue"] / hist_price["quantity"].clip(lower=EPS)
        hist_price = (
            hist_price.groupby("item_code", observed=True)["hist_unit_price"]
            .median()
            .reset_index()
        )
        pairs = pairs.merge(hist_price, on="item_code", how="left")
        pairs["item_price"] = np.where(
            pairs["item_price"].to_numpy(dtype=np.float32) > 0,
            pairs["item_price"].to_numpy(dtype=np.float32),
            pairs["hist_unit_price"].fillna(0).to_numpy(dtype=np.float32),
        )
        pairs.drop(columns=["hist_unit_price"], inplace=True)
        pairs["location"] = pairs["location"].astype("int32")
        for col in ["pair_id", "location_code", "item_code", "category_code", "brand_code"]:
            pairs[col] = pairs[col].astype("int32")
        pairs["item_price"] = pairs["item_price"].astype("float32")

        n_pairs = len(pairs)
        n_items = int(df["item_code"].max()) + 1
        n_locations = int(df["location_code"].max()) + 1
        n_categories = int(df["category_code"].max()) + 1
        n_brands = int(df["brand_code"].max()) + 1

        pair_qty = make_wide_array(df, "pair_id", "quantity", n_pairs)
        pair_rev = make_wide_array(df, "pair_id", "revenue", n_pairs)
        item_qty = make_wide_array(df, "item_code", "quantity", n_items)
        item_rev = make_wide_array(df, "item_code", "revenue", n_items)
        loc_qty = make_wide_array(df, "location_code", "quantity", n_locations)
        loc_rev = make_wide_array(df, "location_code", "revenue", n_locations)
        cat_qty = make_wide_array(df, "category_code", "quantity", n_categories)
        brand_qty = make_wide_array(df, "brand_code", "quantity", n_brands)

        active_pair_month = df[[LOCATION_COL, "pair_id", "item_code", "location_code", "month_idx", "quantity"]].copy()
        active_pair_month = active_pair_month[active_pair_month["quantity"] > 0]
        item_active = (
            active_pair_month.groupby(["item_code", "month_idx"], observed=True)["location_code"]
            .nunique()
            .reset_index(name="active_locs")
        )
        loc_active = (
            active_pair_month.groupby(["location_code", "month_idx"], observed=True)["item_code"]
            .nunique()
            .reset_index(name="active_items")
        )
        item_active_locs = make_wide_array(item_active, "item_code", "active_locs", n_items)
        loc_active_items = make_wide_array(loc_active, "location_code", "active_items", n_locations)

        item_map = pairs[[ITEM_COL, "item_code"]].drop_duplicates(ITEM_COL)
        ev = events_monthly.merge(item_map, on=ITEM_COL, how="inner")
        view_count = make_wide_array(ev, "item_code", "view_count", n_items)
        atc_count = make_wide_array(ev, "item_code", "atc_count", n_items)

        item_price = (
            pairs.groupby("item_code", observed=True)["item_price"]
            .median()
            .reindex(np.arange(n_items), fill_value=0)
            .astype("float32")
            .to_numpy()
        )

        log.info(
            "pairs=%s items=%s locations=%s categories=%s brands=%s",
            n_pairs,
            n_items,
            n_locations,
            n_categories,
            n_brands,
        )

        return MonthlyData(
            pairs=pairs,
            purchase_monthly=df[[LOCATION_COL, ITEM_COL, "pair_id", "item_code", "location_code", "category_code", "brand_code", "month_idx", "quantity", "revenue"]],
            pair_qty=pair_qty,
            pair_rev=pair_rev,
            item_qty=item_qty,
            item_rev=item_rev,
            loc_qty=loc_qty,
            loc_rev=loc_rev,
            cat_qty=cat_qty,
            brand_qty=brand_qty,
            item_active_locs=item_active_locs,
            loc_active_items=loc_active_items,
            view_count=view_count,
            atc_count=atc_count,
            item_price=item_price,
            n_items=n_items,
            n_locations=n_locations,
            n_categories=n_categories,
            n_brands=n_brands,
        )


def make_wide_array(df: pd.DataFrame, key_col: str, value_col: str, n_keys: int) -> np.ndarray:
    if df.empty or value_col not in df.columns:
        return np.zeros((n_keys, 12), dtype=np.float32)
    wide = (
        df.groupby([key_col, "month_idx"], observed=True)[value_col]
        .sum()
        .unstack("month_idx", fill_value=0)
        .reindex(index=np.arange(n_keys), columns=MONTHS, fill_value=0)
    )
    return wide.astype("float32").to_numpy()


def lag_values(history: np.ndarray, target_month: int, lag: int) -> np.ndarray:
    month = target_month - lag
    if 1 <= month <= 12:
        return history[:, month - 1]
    return np.zeros(history.shape[0], dtype=np.float32)


def rolling_window(history: np.ndarray, target_month: int, window: int) -> np.ndarray:
    return np.column_stack([lag_values(history, target_month, lag) for lag in range(1, window + 1)])


def series_features(
    matrix: np.ndarray,
    keys: np.ndarray,
    target_month: int,
    prefix: str,
    full: bool = True,
) -> Dict[str, np.ndarray]:
    hist = matrix[keys]
    features: Dict[str, np.ndarray] = {}
    lag1 = lag_values(hist, target_month, 1)
    lag2 = lag_values(hist, target_month, 2)
    lag3 = lag_values(hist, target_month, 3)

    for lag in [1, 2, 3, 6]:
        features[f"{prefix}_lag{lag}"] = lag_values(hist, target_month, lag).astype("float32")

    roll3 = rolling_window(hist, target_month, 3)
    roll6 = rolling_window(hist, target_month, 6)
    features[f"{prefix}_roll_mean_3"] = roll3.mean(axis=1).astype("float32")
    features[f"{prefix}_roll_mean_6"] = roll6.mean(axis=1).astype("float32")

    if full:
        features[f"{prefix}_roll_median_3"] = np.median(roll3, axis=1).astype("float32")
        features[f"{prefix}_roll_std_3"] = roll3.std(axis=1).astype("float32")
        features[f"{prefix}_roll_max_3"] = roll3.max(axis=1).astype("float32")
        features[f"{prefix}_roll_min_3"] = roll3.min(axis=1).astype("float32")

        hist_len = max(0, min(target_month - 1, 12))
        if hist_len > 0:
            hist_cut = hist[:, :hist_len]
            positive = hist_cut > 0
            month_numbers = np.arange(1, hist_len + 1, dtype=np.int16)
            last_sale_month = np.where(positive, month_numbers, 0).max(axis=1)
            n_sale_months = positive.sum(axis=1)
        else:
            last_sale_month = np.zeros(hist.shape[0], dtype=np.int16)
            n_sale_months = np.zeros(hist.shape[0], dtype=np.int16)
        months_since = np.where(last_sale_month > 0, (target_month - 1) - last_sale_month, target_month - 1)
        features[f"{prefix}_months_since_last_sale"] = months_since.astype("float32")
        features[f"{prefix}_number_of_sale_months"] = n_sale_months.astype("float32")
        features[f"{prefix}_zero_sale_months"] = ((target_month - 1) - n_sale_months).astype("float32")
        features[f"{prefix}_trend_3"] = (lag1 - lag3).astype("float32")
        features[f"{prefix}_growth_ratio"] = (lag1 / (lag2 + 1.0)).astype("float32")
        features[f"{prefix}_lag1_to_roll3"] = (lag1 / (features[f"{prefix}_roll_mean_3"] + 1.0)).astype("float32")

    return features


def build_feature_frame(
    data: MonthlyData,
    target_month: int,
    include_target: bool,
    name: str,
    pair_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    with timer(f"build feature frame {name} target_month={target_month}"):
        if pair_ids is None:
            hist_len = max(0, min(target_month - 1, 12))
            seen = data.pair_qty[:, :hist_len].sum(axis=1) > 0
            pair_ids = np.flatnonzero(seen).astype("int32")

        pairs = data.pairs.iloc[pair_ids]
        item_keys = pairs["item_code"].to_numpy(dtype=np.int32)
        loc_keys = pairs["location_code"].to_numpy(dtype=np.int32)
        cat_keys = pairs["category_code"].to_numpy(dtype=np.int32)
        brand_keys = pairs["brand_code"].to_numpy(dtype=np.int32)

        features: Dict[str, np.ndarray] = {
            "pair_id": pair_ids.astype("int32"),
            "location": pairs["location"].to_numpy(dtype=np.int32),
            "location_code": loc_keys,
            "item_code": item_keys,
            "category_code": cat_keys,
            "brand_code": brand_keys,
            "target_month_idx": np.full(len(pair_ids), target_month, dtype=np.int16),
            "month_sin": np.full(len(pair_ids), math.sin(2 * math.pi * target_month / 12), dtype=np.float32),
            "month_cos": np.full(len(pair_ids), math.cos(2 * math.pi * target_month / 12), dtype=np.float32),
        }

        features.update(series_features(data.pair_qty, pair_ids, target_month, "li_qty", full=True))
        features.update(series_features(data.item_qty, item_keys, target_month, "item_qty", full=True))
        features.update(series_features(data.loc_qty, loc_keys, target_month, "loc_qty", full=True))
        features.update(series_features(data.cat_qty, cat_keys, target_month, "cat_qty", full=True))
        features.update(series_features(data.brand_qty, brand_keys, target_month, "brand_qty", full=True))

        features.update(series_features(data.pair_rev, pair_ids, target_month, "li_rev", full=False))
        features.update(series_features(data.item_rev, item_keys, target_month, "item_rev", full=False))
        features.update(series_features(data.loc_rev, loc_keys, target_month, "loc_rev", full=False))

        features.update(series_features(data.item_active_locs, item_keys, target_month, "item_active_locs", full=False))
        features.update(series_features(data.loc_active_items, loc_keys, target_month, "loc_active_items", full=False))

        features.update(series_features(data.view_count, item_keys, target_month, "view_item", full=False))
        features.update(series_features(data.atc_count, item_keys, target_month, "add_to_cart", full=False))
        features["atc_to_view_lag1"] = (
            features["add_to_cart_lag1"] / (features["view_item_lag1"] + 1.0)
        ).astype("float32")
        features["purchase_to_atc_lag1"] = (
            features["item_qty_lag1"] / (features["add_to_cart_lag1"] + 1.0)
        ).astype("float32")
        features["atc_to_view_roll3"] = (
            features["add_to_cart_roll_mean_3"] / (features["view_item_roll_mean_3"] + 1.0)
        ).astype("float32")
        features["purchase_to_atc_roll3"] = (
            features["item_qty_roll_mean_3"] / (features["add_to_cart_roll_mean_3"] + 1.0)
        ).astype("float32")

        frame = pd.DataFrame(features)
        frame["baseline_lag1"] = frame["li_qty_lag1"].astype("float32")
        frame["baseline_rolling3"] = frame["li_qty_roll_mean_3"].astype("float32")
        frame["baseline_weighted"] = make_weighted_recent_baseline(frame).astype("float32")

        if include_target:
            frame["target"] = data.pair_qty[pair_ids, target_month - 1].astype("float32")
            frame["target_revenue"] = data.pair_rev[pair_ids, target_month - 1].astype("float32")

        log.info("%s shape=%s", name, frame.shape)
        return frame


def make_weighted_recent_baseline(frame: pd.DataFrame) -> np.ndarray:
    pair_recent = (
        0.5 * frame["li_qty_lag1"].to_numpy()
        + 0.3 * frame["li_qty_lag2"].to_numpy()
        + 0.2 * frame["li_qty_lag3"].to_numpy()
    )
    item_recent_total = (
        0.5 * frame["item_qty_lag1"].to_numpy()
        + 0.3 * frame["item_qty_lag2"].to_numpy()
        + 0.2 * frame["item_qty_lag3"].to_numpy()
    )
    item_active_locs = (
        0.5 * frame["item_active_locs_lag1"].to_numpy()
        + 0.3 * frame["item_active_locs_lag2"].to_numpy()
        + 0.2 * frame["item_active_locs_lag3"].to_numpy()
    )
    loc_recent_total = (
        0.5 * frame["loc_qty_lag1"].to_numpy()
        + 0.3 * frame["loc_qty_lag2"].to_numpy()
        + 0.2 * frame["loc_qty_lag3"].to_numpy()
    )
    loc_active_items = (
        0.5 * frame["loc_active_items_lag1"].to_numpy()
        + 0.3 * frame["loc_active_items_lag2"].to_numpy()
        + 0.2 * frame["loc_active_items_lag3"].to_numpy()
    )
    item_fallback = item_recent_total / np.maximum(item_active_locs, 1.0)
    loc_fallback = loc_recent_total / np.maximum(loc_active_items, 1.0)
    fallback = 0.7 * item_fallback + 0.3 * loc_fallback
    return np.where(pair_recent > 0, pair_recent, fallback).clip(0)


def build_training_frames(data: MonthlyData, target_months: List[int], label: str) -> pd.DataFrame:
    frames = [build_feature_frame(data, month, include_target=True, name=f"{label}_m{month}") for month in target_months]
    with timer(f"concat {label} training frames"):
        result = pd.concat(frames, ignore_index=True)
        del frames
        cleanup()
        log.info("%s rows=%s cols=%s positive_rate=%.4f", label, len(result), result.shape[1], (result["target"] > 0).mean())
        return result


def feature_columns(frame: pd.DataFrame) -> List[str]:
    excluded = {"pair_id", "target", "target_revenue"}
    return [col for col in frame.columns if col not in excluded]


def sample_training_rows(frame: pd.DataFrame, max_rows: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    rng = np.random.default_rng(seed)
    positive_idx = frame.index[frame["target"] > 0].to_numpy()
    zero_idx = frame.index[frame["target"] <= 0].to_numpy()
    if len(positive_idx) >= max_rows:
        chosen = rng.choice(positive_idx, size=max_rows, replace=False)
    else:
        n_zero = max_rows - len(positive_idx)
        chosen_zero = rng.choice(zero_idx, size=min(n_zero, len(zero_idx)), replace=False)
        chosen = np.concatenate([positive_idx, chosen_zero])
    rng.shuffle(chosen)
    sampled = frame.loc[chosen].reset_index(drop=True)
    log.info("sampled train rows from %s to %s; positives kept=%s", len(frame), len(sampled), (sampled["target"] > 0).sum())
    return sampled


def train_lightgbm_models(train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str]) -> Tuple[dict, pd.DataFrame]:
    import lightgbm as lgb

    with timer("train LightGBM raw/log models"):
        max_train_rows = env_int("OPT_MAX_TRAIN_ROWS", 0)
        train_fit = sample_training_rows(train_df, max_train_rows)
        X_train = train_fit[features]
        y_train = train_fit["target"].to_numpy(dtype=np.float32)
        X_val = val_df[features]
        y_val = val_df["target"].to_numpy(dtype=np.float32)
        weights = 1.0 / np.maximum(y_train, 1.0)
        val_weights = 1.0 / np.maximum(y_val, 1.0)

        base_params = {
            "n_estimators": env_int("OPT_LGBM_TREES", int(CFG.get("LGBM_COMMON", {}).get("n_estimators", 1200))),
            "learning_rate": env_float("OPT_LGBM_LR", float(CFG.get("LGBM_COMMON", {}).get("learning_rate", 0.04))),
            "num_leaves": env_int("OPT_LGBM_LEAVES", int(CFG.get("LGBM_COMMON", {}).get("num_leaves", 127))),
            "min_child_samples": env_int("OPT_LGBM_MIN_CHILD", 50),
            "subsample": env_float("OPT_LGBM_SUBSAMPLE", 0.85),
            "subsample_freq": 1,
            "colsample_bytree": env_float("OPT_LGBM_COLSAMPLE", 0.85),
            "reg_alpha": env_float("OPT_LGBM_REG_ALPHA", 0.1),
            "reg_lambda": env_float("OPT_LGBM_REG_LAMBDA", 0.5),
            "random_state": RANDOM_STATE,
            "n_jobs": env_int("OPT_N_JOBS", -1),
            "verbosity": -1,
        }
        if env_flag("LGBM_USE_GPU", False):
            base_params.update(
                {
                    "device_type": os.getenv("LGBM_DEVICE_TYPE", "gpu"),
                    "gpu_platform_id": env_int("LGBM_GPU_PLATFORM_ID", 0),
                    "gpu_device_id": env_int("LGBM_GPU_DEVICE_ID", 0),
                    "max_bin": env_int("LGBM_MAX_BIN", 63),
                    "gpu_use_dp": env_flag("LGBM_GPU_USE_DP", False),
                }
            )
            log.info(
                "LightGBM GPU enabled: device_type=%s platform=%s device=%s max_bin=%s gpu_use_dp=%s",
                base_params["device_type"],
                base_params["gpu_platform_id"],
                base_params["gpu_device_id"],
                base_params["max_bin"],
                base_params["gpu_use_dp"],
            )
        else:
            log.info("LightGBM GPU disabled; set LGBM_USE_GPU=1 to enable it.")

        categorical = [col for col in ["location_code", "item_code", "category_code", "brand_code"] if col in features]
        callbacks = [lgb.early_stopping(env_int("OPT_EARLY_STOPPING", 100), verbose=False), lgb.log_evaluation(period=100)]

        raw_model = lgb.LGBMRegressor(objective="regression_l1", **base_params)
        raw_model.fit(
            X_train,
            y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[val_weights],
            eval_metric="l1",
            categorical_feature=categorical,
            callbacks=callbacks,
        )

        log_model = lgb.LGBMRegressor(objective="regression", **base_params)
        log_model.fit(
            X_train,
            np.log1p(y_train),
            sample_weight=weights,
            eval_set=[(X_val, np.log1p(y_val))],
            eval_sample_weight=[val_weights],
            eval_metric="l1",
            categorical_feature=categorical,
            callbacks=callbacks,
        )

        pred_raw = np.clip(raw_model.predict(X_val), 0, None).astype("float32")
        pred_log = np.clip(np.expm1(log_model.predict(X_val)), 0, None).astype("float32")

        models = {"lgbm_raw": raw_model, "lgbm_log": log_model}
        preds = pd.DataFrame({"pair_id": val_df["pair_id"].to_numpy(), "lgbm_raw": pred_raw, "lgbm_log": pred_log})
        return models, preds


def train_catboost_if_possible(train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str]) -> Tuple[Optional[object], Optional[np.ndarray]]:
    if not env_flag("OPT_RUN_CATBOOST", True):
        log.info("CatBoost disabled by OPT_RUN_CATBOOST=0")
        return None, None
    try:
        from catboost import CatBoostRegressor
    except Exception as exc:
        log.warning("CatBoost unavailable: %s", exc)
        return None, None

    with timer("train CatBoost raw model"):
        max_rows = env_int("OPT_CATBOOST_MAX_ROWS", 2_000_000)
        train_fit = sample_training_rows(train_df, max_rows, seed=RANDOM_STATE + 11)
        X_train = train_fit[features]
        y_train = train_fit["target"].to_numpy(dtype=np.float32)
        X_val = val_df[features]
        y_val = val_df["target"].to_numpy(dtype=np.float32)
        weights = 1.0 / np.maximum(y_train, 1.0)
        cat_cols = [col for col in ["location_code", "item_code", "category_code", "brand_code"] if col in features]
        cat_idx = [features.index(col) for col in cat_cols]

        try:
            cat_params = {
                "loss_function": "MAE",
                "iterations": env_int("OPT_CATBOOST_ITERS", 900),
                "learning_rate": env_float("OPT_CATBOOST_LR", 0.05),
                "depth": env_int("OPT_CATBOOST_DEPTH", 8),
                "random_seed": RANDOM_STATE,
                "allow_writing_files": False,
                "verbose": 100,
            }
            if env_flag("OPT_CATBOOST_USE_GPU", env_flag("LGBM_USE_GPU", False)):
                cat_params.update(
                    {
                        "task_type": "GPU",
                        "devices": os.getenv("OPT_CATBOOST_DEVICES", "0"),
                    }
                )
                log.info("CatBoost GPU enabled: devices=%s", cat_params["devices"])
            else:
                log.info("CatBoost GPU disabled; set OPT_CATBOOST_USE_GPU=1 to enable it.")

            model = CatBoostRegressor(
                **cat_params,
            )
            model.fit(
                X_train,
                y_train,
                sample_weight=weights,
                eval_set=(X_val, y_val),
                cat_features=cat_idx,
                use_best_model=True,
            )
            pred = np.clip(model.predict(X_val), 0, None).astype("float32")
            return model, pred
        except Exception as exc:
            log.warning("CatBoost skipped after failure: %s", exc)
            return None, None


def score_predictions(data: MonthlyData, pair_ids: np.ndarray, prediction: np.ndarray, target_month: int) -> dict:
    actual_qty = data.pair_qty[:, target_month - 1]
    actual_rev = data.pair_rev[:, target_month - 1]
    active_locs = set(data.pairs.loc[actual_qty > 0, "location_code"].astype(int).tolist())
    eval_mask = data.pairs["location_code"].isin(active_locs) & (data.pairs["sale_status"] != 0)
    eval_pair_ids = data.pairs.loc[eval_mask, "pair_id"].to_numpy(dtype=np.int32)

    pred_map = pd.Series(prediction.astype("float32"), index=pair_ids)
    pred = pred_map.reindex(eval_pair_ids, fill_value=0).to_numpy(dtype=np.float32)
    truth_qty = actual_qty[eval_pair_ids].astype(np.float32)
    truth_rev = actual_rev[eval_pair_ids].astype(np.float32)
    item_codes = data.pairs.loc[eval_pair_ids, "item_code"].to_numpy(dtype=np.int32)
    pred_rev = pred * data.item_price[item_codes]

    mae_qty = float(np.mean(np.abs(truth_qty - pred)))
    mape_qty = safe_mape(truth_qty, pred)
    mae_rev = float(np.mean(np.abs(truth_rev - pred_rev)))
    mape_rev = safe_mape(truth_rev, pred_rev)
    return {
        "mae_quantity": mae_qty,
        "mape_quantity": mape_qty,
        "mae_revenue": mae_rev,
        "mape_revenue": mape_rev,
    }


def metric_summary(y_true: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0, None)
    y_true = np.asarray(y_true, dtype=np.float64)
    return {
        "mae_quantity": float(np.mean(np.abs(y_true - pred))),
        "mape_quantity": safe_mape(y_true, pred),
    }


def safe_mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - pred[mask]) / np.maximum(np.abs(actual[mask]), EPS)) * 100.0)


def make_validation_table(data: MonthlyData, val_df: pd.DataFrame, pred_cols: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    pair_ids = val_df["pair_id"].to_numpy(dtype=np.int32)
    for name, pred in pred_cols.items():
        metrics = score_predictions(data, pair_ids, np.clip(pred, 0, None), target_month=12)
        rows.append({"model": name, **metrics})
    result = pd.DataFrame(rows).sort_values("mape_quantity")
    return result


def make_validation_predictions(
    data: MonthlyData,
    val_df: pd.DataFrame,
    pred_raw: np.ndarray,
    pred_log: np.ndarray,
    pred_ensemble: np.ndarray,
    pred_baseline: Optional[np.ndarray] = None,
    pred_raw_only_postprocess: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    pair_ids = val_df["pair_id"].to_numpy(dtype=np.int32)
    meta = data.pairs.iloc[pair_ids][["location", "item_id"]].reset_index(drop=True)
    result = meta.copy()
    result["y_true"] = val_df["target"].to_numpy(dtype=np.float32)
    result["pred_raw"] = np.clip(pred_raw, 0, None).astype("float32")
    result["pred_log"] = np.clip(pred_log, 0, None).astype("float32")
    result["pred_ensemble"] = np.clip(pred_ensemble, 0, None).astype("float32")
    if pred_baseline is not None:
        result["pred_baseline"] = np.clip(pred_baseline, 0, None).astype("float32")
    if pred_raw_only_postprocess is not None:
        result["pred_raw_only_postprocess"] = np.clip(pred_raw_only_postprocess, 0, None).astype("float32")
    return result


def grid_search_ensemble(data: MonthlyData, val_df: pd.DataFrame, model_pred: np.ndarray, baseline_pred: np.ndarray) -> Tuple[np.ndarray, dict]:
    pair_ids = val_df["pair_id"].to_numpy(dtype=np.int32)
    best = {"mape_quantity": float("inf")}
    best_pred = baseline_pred.copy()
    alphas = np.linspace(0.0, 1.0, env_int("OPT_ALPHA_STEPS", 21))
    scales = np.round(np.arange(0.75, 1.3001, env_float("OPT_SCALE_STEP", 0.05)), 4)
    with timer("grid search ensemble alpha and scale"):
        for alpha in alphas:
            mixed = alpha * model_pred + (1.0 - alpha) * baseline_pred
            for scale in scales:
                pred = np.clip(mixed * scale, 0, None)
                metrics = score_predictions(data, pair_ids, pred, target_month=12)
                if metrics["mape_quantity"] < best["mape_quantity"]:
                    best = {"alpha": float(alpha), "scale": float(scale), **metrics}
                    best_pred = pred.astype("float32")
        log.info("best ensemble params=%s", best)
    return best_pred, best


def build_q99_caps(data: MonthlyData, train_end_month: int) -> Dict[str, np.ndarray]:
    hist = data.purchase_monthly[data.purchase_monthly["month_idx"] <= train_end_month]
    item_q99 = hist.groupby("item_code", observed=True)["quantity"].quantile(0.99)
    loc_q99 = hist.groupby("location_code", observed=True)["quantity"].quantile(0.99)
    pair_q99 = hist.groupby("pair_id", observed=True)["quantity"].quantile(0.99)
    return {
        "item": item_q99.reindex(np.arange(data.n_items), fill_value=np.inf).to_numpy(dtype=np.float32),
        "location": loc_q99.reindex(np.arange(data.n_locations), fill_value=np.inf).to_numpy(dtype=np.float32),
        "pair": pair_q99.reindex(np.arange(len(data.pairs)), fill_value=np.inf).to_numpy(dtype=np.float32),
    }


def apply_cap(data: MonthlyData, pair_ids: np.ndarray, pred: np.ndarray, caps: Dict[str, np.ndarray], kind: str, mult: float) -> np.ndarray:
    if kind == "none":
        return pred
    pairs = data.pairs.iloc[pair_ids]
    if kind == "item":
        cap = caps["item"][pairs["item_code"].to_numpy(dtype=np.int32)]
    elif kind == "location":
        cap = caps["location"][pairs["location_code"].to_numpy(dtype=np.int32)]
    elif kind == "pair":
        cap = caps["pair"][pair_ids]
    elif kind == "min_item_location":
        item_cap = caps["item"][pairs["item_code"].to_numpy(dtype=np.int32)]
        loc_cap = caps["location"][pairs["location_code"].to_numpy(dtype=np.int32)]
        cap = np.minimum(item_cap, loc_cap)
    else:
        raise ValueError(f"Unknown cap kind: {kind}")
    cap = np.where(np.isfinite(cap) & (cap > 0), cap * mult, np.inf)
    return np.minimum(pred, cap).astype("float32")


def tune_postprocess(data: MonthlyData, val_df: pd.DataFrame, pred: np.ndarray) -> Tuple[np.ndarray, dict]:
    pair_ids = val_df["pair_id"].to_numpy(dtype=np.int32)
    caps = build_q99_caps(data, train_end_month=11)
    best = {"mape_quantity": float("inf"), "clip_kind": "none", "clip_mult": 1.0, "floor": 0.0}
    best_pred = pred.copy()
    clip_kinds = ["none", "pair", "item", "location", "min_item_location"]
    clip_mults = [1.0, 1.25, 1.5, 2.0, 3.0]
    floors = [0.0, 0.1, 0.25, 0.5, 1.0]
    with timer("tune postprocess clipping/floor"):
        for kind in clip_kinds:
            for mult in clip_mults:
                clipped = apply_cap(data, pair_ids, pred.copy(), caps, kind, mult)
                for floor in floors:
                    candidate = clipped.copy()
                    if floor > 0:
                        candidate[candidate < floor] = 0.0
                    metrics = score_predictions(data, pair_ids, candidate, target_month=12)
                    if metrics["mape_quantity"] < best["mape_quantity"]:
                        best = {
                            "clip_kind": kind,
                            "clip_mult": float(mult),
                            "floor": float(floor),
                            **metrics,
                        }
                        best_pred = candidate.astype("float32")
        log.info("best postprocess params=%s", best)
    return best_pred, best


def tune_safe_postprocess_options(data: MonthlyData, val_df: pd.DataFrame, pred: np.ndarray) -> Tuple[np.ndarray, dict, pd.DataFrame]:
    pair_ids = val_df["pair_id"].to_numpy(dtype=np.int32)
    caps = build_q99_caps(data, train_end_month=11)
    options = [
        ("none", "none", 1.0, 0),
        ("global_q99", "global", 1.0, 0),
        ("item_q99", "item", 1.0, 3),
        ("location_q99", "location", 1.0, 3),
    ]
    hist = data.purchase_monthly[data.purchase_monthly["month_idx"] <= 11]
    global_cap = float(hist["quantity"].quantile(0.99)) if not hist.empty else float("inf")
    rows = []
    best_name = "none"
    best_metrics = score_predictions(data, pair_ids, pred, target_month=12)
    best_pred = np.clip(pred, 0, None).astype("float32")

    with timer("tune safe postprocess options"):
        for name, kind, mult, min_hist in options:
            candidate = np.clip(pred.copy(), 0, None).astype("float32")
            if kind == "global" and np.isfinite(global_cap) and global_cap > 0:
                candidate = np.minimum(candidate, global_cap * mult).astype("float32")
            elif kind in {"item", "location"}:
                pairs = data.pairs.iloc[pair_ids]
                if kind == "item":
                    hist_count = hist.groupby("item_code", observed=True)["quantity"].size()
                    counts = hist_count.reindex(np.arange(data.n_items), fill_value=0).to_numpy()
                    raw_cap = caps["item"][pairs["item_code"].to_numpy(dtype=np.int32)]
                    enough_history = counts[pairs["item_code"].to_numpy(dtype=np.int32)] >= min_hist
                else:
                    hist_count = hist.groupby("location_code", observed=True)["quantity"].size()
                    counts = hist_count.reindex(np.arange(data.n_locations), fill_value=0).to_numpy()
                    raw_cap = caps["location"][pairs["location_code"].to_numpy(dtype=np.int32)]
                    enough_history = counts[pairs["location_code"].to_numpy(dtype=np.int32)] >= min_hist
                cap = np.where(enough_history & np.isfinite(raw_cap) & (raw_cap > 0), raw_cap * mult, np.inf)
                candidate = np.minimum(candidate, cap).astype("float32")

            metrics = score_predictions(data, pair_ids, candidate, target_month=12)
            rows.append({"postprocess": name, "clip_kind": kind, "clip_mult": mult, "min_history": min_hist, **metrics})
            if metrics["mape_quantity"] < best_metrics["mape_quantity"]:
                best_name = name
                best_metrics = metrics
                best_pred = candidate

    params = {"postprocess": best_name, **best_metrics}
    table = pd.DataFrame(rows).sort_values("mape_quantity").reset_index(drop=True)
    log.info("safe postprocess best=%s", params)
    return best_pred, params, table


def train_final_lgbm(train_df: pd.DataFrame, features: List[str], val_models: dict) -> dict:
    import lightgbm as lgb

    with timer("train final LightGBM models on Jan-Dec targets"):
        max_train_rows = env_int("OPT_MAX_FINAL_TRAIN_ROWS", env_int("OPT_MAX_TRAIN_ROWS", 0))
        train_fit = sample_training_rows(train_df, max_train_rows, seed=RANDOM_STATE + 21)
        X_train = train_fit[features]
        y_train = train_fit["target"].to_numpy(dtype=np.float32)
        weights = 1.0 / np.maximum(y_train, 1.0)
        categorical = [col for col in ["location_code", "item_code", "category_code", "brand_code"] if col in features]

        final_models = {}
        for name, target_values, objective in [
            ("lgbm_raw", y_train, "regression_l1"),
            ("lgbm_log", np.log1p(y_train), "regression"),
        ]:
            source = val_models[name]
            n_estimators = int(getattr(source, "best_iteration_", None) or getattr(source, "n_estimators", 1000))
            params = source.get_params()
            params.update({"n_estimators": max(50, n_estimators), "objective": objective})
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, target_values, sample_weight=weights, categorical_feature=categorical)
            final_models[name] = model
            model.booster_.save_model(str(MODEL_DIR / f"optimized_{name}.txt"))
        return final_models


def train_final_catboost(train_df: pd.DataFrame, features: List[str], val_cat_model: Optional[object]) -> Optional[object]:
    if val_cat_model is None or not env_flag("OPT_RUN_CATBOOST", True):
        return None
    try:
        from catboost import CatBoostRegressor
    except Exception:
        return None
    with timer("train final CatBoost model"):
        max_rows = env_int("OPT_CATBOOST_MAX_ROWS", 2_000_000)
        train_fit = sample_training_rows(train_df, max_rows, seed=RANDOM_STATE + 31)
        X_train = train_fit[features]
        y_train = train_fit["target"].to_numpy(dtype=np.float32)
        weights = 1.0 / np.maximum(y_train, 1.0)
        cat_cols = [col for col in ["location_code", "item_code", "category_code", "brand_code"] if col in features]
        cat_idx = [features.index(col) for col in cat_cols]
        cat_params = {
            "loss_function": "MAE",
            "iterations": int(getattr(val_cat_model, "best_iteration_", None) or env_int("OPT_CATBOOST_ITERS", 900)),
            "learning_rate": env_float("OPT_CATBOOST_LR", 0.05),
            "depth": env_int("OPT_CATBOOST_DEPTH", 8),
            "random_seed": RANDOM_STATE,
            "allow_writing_files": False,
            "verbose": 100,
        }
        if env_flag("OPT_CATBOOST_USE_GPU", env_flag("LGBM_USE_GPU", False)):
            cat_params.update({"task_type": "GPU", "devices": os.getenv("OPT_CATBOOST_DEVICES", "0")})
            log.info("Final CatBoost GPU enabled: devices=%s", cat_params["devices"])
        else:
            log.info("Final CatBoost GPU disabled; set OPT_CATBOOST_USE_GPU=1 to enable it.")

        try:
            model = CatBoostRegressor(**cat_params)
            model.fit(X_train, y_train, sample_weight=weights, cat_features=cat_idx)
            model.save_model(str(MODEL_DIR / "optimized_catboost.cbm"))
            return model
        except Exception as exc:
            log.warning("Final CatBoost skipped after failure: %s", exc)
            return None


def predict_model_dict(models: dict, X: pd.DataFrame, features: List[str]) -> Dict[str, np.ndarray]:
    preds = {}
    if "lgbm_raw" in models:
        preds["lgbm_raw"] = np.clip(models["lgbm_raw"].predict(X[features]), 0, None).astype("float32")
    if "lgbm_log" in models:
        preds["lgbm_log"] = np.clip(np.expm1(models["lgbm_log"].predict(X[features])), 0, None).astype("float32")
    if "catboost" in models and models["catboost"] is not None:
        preds["catboost"] = np.clip(models["catboost"].predict(X[features]), 0, None).astype("float32")
    return preds


def combine_model_predictions(preds: Dict[str, np.ndarray]) -> np.ndarray:
    available = [values for values in preds.values() if values is not None]
    if not available:
        raise RuntimeError("No model predictions available.")
    return np.mean(np.column_stack(available), axis=1).astype("float32")


def apply_postprocess_with_params(
    data: MonthlyData,
    pair_ids: np.ndarray,
    pred: np.ndarray,
    params: dict,
    train_end_month: int,
) -> np.ndarray:
    caps = build_q99_caps(data, train_end_month=train_end_month)
    result = apply_cap(data, pair_ids, pred.copy(), caps, params.get("clip_kind", "none"), float(params.get("clip_mult", 1.0)))
    floor = float(params.get("floor", 0.0))
    if floor > 0:
        result[result < floor] = 0.0
    return np.clip(result, 0, None).astype("float32")


def save_submission(data: MonthlyData, pred_df: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    with timer("save final submission"):
        pair_ids = pred_df["pair_id"].to_numpy(dtype=np.int32)
        meta = data.pairs.iloc[pair_ids][["location", "item_id", "sale_status"]].copy()
        submission = meta[meta["sale_status"] != 0][["location", "item_id"]].copy()
        submission["prediction"] = prediction[meta["sale_status"].to_numpy() != 0].astype("float64")
        submission["prediction"] = submission["prediction"].clip(lower=0)
        submission = submission.drop_duplicates(["location", "item_id"]).reset_index(drop=True)
        submission["location"] = submission["location"].astype("int64")
        submission["item_id"] = submission["item_id"].astype("string[python]").astype(object)
        submission["prediction"] = submission["prediction"].astype("float64")
        submission.columns = pd.Index(["location", "item_id", "prediction"], dtype=object)

        csv_path = OUTPUT_DIR / "submission_final.csv"
        pkl_path = OUTPUT_DIR / "submission_final.pkl"
        submission.to_csv(csv_path, index=False)
        submission.to_pickle(pkl_path)
        log.info("saved %s rows to %s and %s", len(submission), csv_path, pkl_path)
        return submission


def write_report(validation_table: pd.DataFrame, ensemble_params: dict, postprocess_params: dict) -> None:
    with timer("write validation report"):
        validation_path = OUTPUT_DIR / "optimized_validation_results.csv"
        validation_table.to_csv(validation_path, index=False)
        report = [
            "# Optimized Forecast Validation",
            "",
            validation_table.to_markdown(index=False),
            "",
            "## Selected Ensemble Params",
            "",
            "```json",
            json.dumps(ensemble_params, indent=2),
            "```",
            "",
            "## Selected Postprocess Params",
            "",
            "```json",
            json.dumps(postprocess_params, indent=2),
            "```",
            "",
        ]
        path = REPORT_DIR / "optimized_model_results.md"
        path.write_text("\n".join(report), encoding="utf-8")
        log.info("validation table saved to %s", validation_path)
        log.info("report saved to %s", path)


def write_validation_predictions(validation_predictions: pd.DataFrame, path: Optional[Path] = None) -> None:
    out_path = path or (OUTPUT_DIR / "validation_predictions.csv")
    with timer("write wide validation predictions"):
        validation_predictions.to_csv(out_path, index=False)
        log.info("validation predictions saved to %s rows=%s cols=%s", out_path, len(validation_predictions), validation_predictions.shape[1])


def run_pipeline() -> pd.DataFrame:
    max_row_groups = env_int("OPT_MAX_ROW_GROUPS", 0)
    log.info("OPT_MAX_ROW_GROUPS=%s (0 means all)", max_row_groups)

    purchases = aggregate_purchases(max_row_groups=max_row_groups)
    events = aggregate_events(max_row_groups=max_row_groups)
    items = load_items()
    data = encode_monthly_tables(purchases, events, items)
    del purchases, events, items
    cleanup()

    train_df = build_training_frames(data, list(range(4, 12)), label="train_jan_to_nov")
    val_df = build_feature_frame(data, 12, include_target=True, name="validation_dec")
    features = feature_columns(train_df)
    log.info("feature_count=%s", len(features))

    lgbm_models, lgbm_val_preds = train_lightgbm_models(train_df, val_df, features)
    cat_model, cat_val_pred = train_catboost_if_possible(train_df, val_df, features)
    if cat_val_pred is not None:
        lgbm_val_preds["catboost"] = cat_val_pred

    val_pair_ids = val_df["pair_id"].to_numpy(dtype=np.int32)
    pred_dict = {
        "baseline lag1": val_df["baseline_lag1"].to_numpy(dtype=np.float32),
        "baseline rolling3": val_df["baseline_rolling3"].to_numpy(dtype=np.float32),
        "LightGBM raw": lgbm_val_preds["lgbm_raw"].to_numpy(dtype=np.float32),
        "LightGBM log": lgbm_val_preds["lgbm_log"].to_numpy(dtype=np.float32),
    }
    if "catboost" in lgbm_val_preds.columns:
        pred_dict["CatBoost"] = lgbm_val_preds["catboost"].to_numpy(dtype=np.float32)

    model_pred_val = combine_model_predictions(
        {col: lgbm_val_preds[col].to_numpy(dtype=np.float32) for col in lgbm_val_preds.columns if col != "pair_id"}
    )
    baseline_pred_val = val_df["baseline_weighted"].to_numpy(dtype=np.float32)
    ensemble_val_pred, ensemble_params = grid_search_ensemble(data, val_df, model_pred_val, baseline_pred_val)
    post_val_pred, post_params = tune_postprocess(data, val_df, ensemble_val_pred)
    raw_safe_post_pred, raw_safe_post_params, safe_post_table = tune_safe_postprocess_options(
        data,
        val_df,
        lgbm_val_preds["lgbm_raw"].to_numpy(dtype=np.float32),
    )
    pred_dict["ensemble"] = ensemble_val_pred
    pred_dict["ensemble + postprocess"] = post_val_pred
    pred_dict["raw_only_postprocess"] = raw_safe_post_pred

    validation_table = make_validation_table(data, val_df, pred_dict)
    log.info("\n%s", validation_table.to_string(index=False))
    write_report(validation_table, ensemble_params, post_params)
    safe_post_table.to_csv(OUTPUT_DIR / "postprocess_validation_results.csv", index=False)
    write_validation_predictions(
        make_validation_predictions(
            data,
            val_df,
            pred_raw=lgbm_val_preds["lgbm_raw"].to_numpy(dtype=np.float32),
            pred_log=lgbm_val_preds["lgbm_log"].to_numpy(dtype=np.float32),
            pred_ensemble=ensemble_val_pred,
            pred_baseline=baseline_pred_val,
            pred_raw_only_postprocess=raw_safe_post_pred,
        )
    )

    del train_df
    cleanup()

    final_train_df = build_training_frames(data, list(range(4, 13)), label="final_train_jan_to_dec")
    final_models = train_final_lgbm(final_train_df, features, lgbm_models)
    final_cat = train_final_catboost(final_train_df, features, cat_model)
    if final_cat is not None:
        final_models["catboost"] = final_cat
    del final_train_df
    cleanup()

    pred_df = build_feature_frame(data, 13, include_target=False, name="predict_jan_2026")
    final_model_preds = predict_model_dict(final_models, pred_df, features)
    if env_flag("OPT_USE_RAW_ONLY", False):
        log.info("OPT_USE_RAW_ONLY=1: final prediction uses direct LightGBM raw output")
        final_pred = np.clip(final_model_preds["lgbm_raw"], 0, None).astype("float32")
    else:
        final_model_pred = combine_model_predictions(final_model_preds)
        final_baseline = pred_df["baseline_weighted"].to_numpy(dtype=np.float32)
        final_ensemble = np.clip(
            (ensemble_params["alpha"] * final_model_pred + (1.0 - ensemble_params["alpha"]) * final_baseline)
            * ensemble_params["scale"],
            0,
            None,
        ).astype("float32")
        final_pred = apply_postprocess_with_params(
            data,
            pred_df["pair_id"].to_numpy(dtype=np.int32),
            final_ensemble,
            post_params,
            train_end_month=12,
        )

    submission = save_submission(data, pred_df, final_pred)
    log.info("Final submission columns=%s rows=%s", submission.columns.tolist(), len(submission))
    return submission


if __name__ == "__main__":
    run_pipeline()
