"""
src/data_audit.py
=================
Bước 1: Đọc và kiểm tra schema toàn bộ dữ liệu.
Xuất báo cáo ngắn ra reports/data_audit.md.
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
with open(REPO_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DATA_DIR   = REPO_ROOT / CFG["DATA_DIR"]
REPORT_DIR = REPO_ROOT / CFG["REPORT_DIR"]
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def audit_df(df: pd.DataFrame, name: str) -> dict:
    """Return dict with key stats of a DataFrame."""
    info = {
        "name": name,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "nulls": (len(df) - df.count()).to_dict(),
        "sample": df.head(3),
    }
    for col in ["location", "item_id", "event_type"]:
        if col in df.columns:
            info[f"{col}_nunique"] = df[col].nunique()
            if col == "event_type":
                info["event_type_values"] = df[col].unique().tolist()
    # date range
    for col in ["updated_date", "event_date", "created_date"]:
        if col in df.columns:
            info[f"{col}_min"] = str(df[col].min())
            info[f"{col}_max"] = str(df[col].max())
    return info


def fmt_nulls(nulls: dict) -> str:
    non_zero = {k: v for k, v in nulls.items() if v > 0}
    return str(non_zero) if non_zero else "None"


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def run_audit():
    report_lines = [
        "# Data Audit Report\n",
        "## Mapping từ tên cột thực tế sang ý nghĩa bài toán\n",
        "| File | Cột thực tế | Ý nghĩa |",
        "|------|-------------|---------|",
        "| transaction_full_2025 | location | location (địa điểm bán) |",
        "| transaction_full_2025 | item_id | item_id (mã sản phẩm) |",
        "| transaction_full_2025 | updated_date | ngày giao dịch |",
        "| transaction_full_2025 | quantity | số lượng mua |",
        "| transaction_full_2025 | price | đơn giá |",
        "| transaction_full_2025 | event_type | loại sự kiện (chỉ có 'Purchase') |",
        "| event_full_2025 | item_id | item_id |",
        "| event_full_2025 | event_date | ngày sự kiện |",
        "| event_full_2025 | event_type | view_item / add_to_cart |",
        "| event_full_2025 | customer_id | khách hàng (không có location) |",
        "| items | item_id | item_id |",
        "| items | sale_status | trạng thái bán (0=không bán) |",
        "| items | category_lv1 | ngành hàng cấp 1 |",
        "| items | price | giá chuẩn |",
        "",
        "---",
        "",
    ]

    # -----------------------------------------------------------------------
    # 1. transaction_full_2025
    # -----------------------------------------------------------------------
    log.info("Loading transaction_full_2025.parquet ...")
    tx = pd.read_parquet(DATA_DIR / CFG["TRANSACTION_FILE"])
    if CFG["DEBUG_SAMPLE"]:
        tx = tx.sample(frac=CFG["DEBUG_SAMPLE_FRAC"], random_state=CFG["RANDOM_STATE"])
        log.info(f"DEBUG_SAMPLE mode: {len(tx):,} rows")

    tx_info = audit_df(tx, "transaction_full_2025")

    log.info(f"transaction rows: {tx_info['rows']:,}")
    log.info(f"  location nunique: {tx_info.get('location_nunique', 'N/A')}")
    log.info(f"  item_id  nunique: {tx_info.get('item_id_nunique', 'N/A')}")
    log.info(f"  event_types: {tx_info.get('event_type_values', 'N/A')}")

    tx["price_num"] = pd.to_numeric(tx[CFG["TX_PRICE_COL"]], errors="coerce")
    tx["revenue"] = tx["price_num"] * tx[CFG["TX_QTY_COL"]]
    log.info(f"  revenue range: {tx['revenue'].min():.0f} - {tx['revenue'].max():.0f}")

    report_lines += [
        "## 1. transaction_full_2025.parquet",
        f"- **Rows**: {tx_info['rows']:,}",
        f"- **Columns**: {tx_info['columns']}",
        "- **Dtypes**:",
        "```",
    ] + [f"  {k}: {v}" for k, v in tx_info["dtypes"].items()] + [
        "```",
        f"- **Null counts**: {fmt_nulls(tx_info['nulls'])}",
        f"- **location nunique**: {tx_info.get('location_nunique', 'N/A')}",
        f"- **item_id nunique**: {tx_info.get('item_id_nunique', 'N/A')}",
        f"- **event_type values**: {tx_info.get('event_type_values', 'N/A')}",
        f"- **Date range**: {tx_info.get('updated_date_min')} → {tx_info.get('updated_date_max')}",
        f"- **quantity stats**: min={tx[CFG['TX_QTY_COL']].min()}, max={tx[CFG['TX_QTY_COL']].max()}, mean={tx[CFG['TX_QTY_COL']].mean():.2f}",
        f"- **revenue stats**: min={tx['revenue'].min():.0f}, max={tx['revenue'].max():.0f}, mean={tx['revenue'].mean():.0f}",
        "",
    ]

    # -----------------------------------------------------------------------
    # 4. Monthly purchase summary
    # -----------------------------------------------------------------------
    log.info("Computing monthly purchase summary ...")
    purchase = tx[tx[CFG["TX_EVENT_COL"]] == CFG["TX_PURCHASE_EVENT"]].copy()
    purchase["month"] = purchase[CFG["TX_DATE_COL"]].dt.to_period("M")
    monthly = purchase.groupby("month").agg(
        transactions=("quantity", "count"),
        total_qty=(CFG["TX_QTY_COL"], "sum"),
        total_revenue=("revenue", "sum"),
        active_locations=(CFG["TX_LOCATION_COL"], "nunique"),
        active_items=(CFG["TX_ITEM_COL"], "nunique"),
    ).reset_index()

    report_lines += [
        "## 4. Monthly Purchase Summary (transaction_full_2025)",
        monthly.to_markdown(index=False),
        "",
        "---",
        "## Kết luận quan trọng",
        "- **Target**: `transaction_full_2025`, event_type == 'Purchase', cột `quantity`.",
        "- **Revenue**: `price` × `quantity` (price cần cast sang float).",
        "- **event_full_2025**: Không có `location`, chỉ dùng để feature view_item/ATC theo item_id.",
        "- **sale_status = 0**: Loại khỏi prediction (set = 0).",
        "- **Submission target**: Dự báo tổng `quantity` Purchase tháng 01/2026 theo `location × item_id`.",
        "",
    ]

    import gc
    del tx
    del purchase
    gc.collect()

    # -----------------------------------------------------------------------
    # 2. event_full_2025
    # -----------------------------------------------------------------------
    log.info("Loading event_full_2025.parquet ...")
    ev = pd.read_parquet(DATA_DIR / CFG["EVENT_FILE"])
    ev_info = audit_df(ev, "event_full_2025")

    log.info(f"event rows: {ev_info['rows']:,}")
    log.info(f"  event_types: {ev_info.get('event_type_values', 'N/A')}")

    report_lines += [
        "## 2. event_full_2025.parquet",
        f"- **Rows**: {ev_info['rows']:,}",
        f"- **Columns**: {ev_info['columns']}",
        "- **Dtypes**:",
        "```",
    ] + [f"  {k}: {v}" for k, v in ev_info["dtypes"].items()] + [
        "```",
        f"- **Null counts**: {fmt_nulls(ev_info['nulls'])}",
        f"- **item_id nunique**: {ev_info.get('item_id_nunique', 'N/A')}",
        f"- **event_type values**: {ev_info.get('event_type_values', 'N/A')}",
        f"- **Date range**: {ev_info.get('event_date_min')} → {ev_info.get('event_date_max')}",
        "- **NOTE**: Không có cột `location`. Event chỉ có customer_id.",
        "",
    ]

    # -----------------------------------------------------------------------
    # 3. items
    # -----------------------------------------------------------------------
    log.info("Loading items.parquet ...")
    items = pd.read_parquet(DATA_DIR / CFG["ITEMS_FILE"])
    items_info = audit_df(items, "items")

    log.info(f"items rows: {items_info['rows']:,}")
    if "sale_status" in items.columns:
        log.info(f"  sale_status counts: {items['sale_status'].value_counts().to_dict()}")

    report_lines += [
        "## 3. items.parquet",
        f"- **Rows**: {items_info['rows']:,}",
        f"- **Columns**: {items_info['columns']}",
        "- **Dtypes**:",
        "```",
    ] + [f"  {k}: {v}" for k, v in items_info["dtypes"].items()] + [
        "```",
        f"- **Null counts**: {fmt_nulls(items_info['nulls'])}",
    ]
    if "sale_status" in items.columns:
        report_lines.append(
            f"- **sale_status distribution**: {items['sale_status'].value_counts().to_dict()}"
        )
    if "category_lv1" in items.columns:
        report_lines.append(
            f"- **category_lv1 nunique**: {items['category_lv1'].nunique()}"
        )
    report_lines.append("")

    # -----------------------------------------------------------------------
    # Write report
    # -----------------------------------------------------------------------
    report_path = REPORT_DIR / "data_audit.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log.info(f"Report saved to {report_path}")

    log.info("=== DATA AUDIT COMPLETE ===")
    return None, None, None


if __name__ == "__main__":
    run_audit()
