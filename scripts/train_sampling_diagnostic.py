#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "srcoptimized"))

import pipeline_20gb as pipe  # noqa: E402


def read_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def target_candidate_mask(data, target_month: int) -> np.ndarray:
    hist_len = max(0, min(target_month - 1, 12))
    seen = data.pair_qty[:, :hist_len].sum(axis=1) > 0
    sale_status = data.pairs["sale_status"].to_numpy()
    return seen & (sale_status != 0)


def target_counts_from_data(data, target_month: int) -> dict:
    mask = target_candidate_mask(data, target_month)
    y = data.pair_qty[mask, target_month - 1].astype(np.float32)
    positives = int((y > 0).sum())
    zeros = int((y <= 0).sum())
    return {
        "target_month": target_month,
        "rows": int(mask.sum()),
        "y=0": zeros,
        "y>0": positives,
        "positive_rate": positives / max(int(mask.sum()), 1),
    }


def train_sample_summary(path: Path) -> tuple[dict, pd.DataFrame]:
    df = pd.read_parquet(path, columns=["target", "target_month_idx"])
    overall_pos = int((df["target"] > 0).sum())
    overall_zero = int((df["target"] <= 0).sum())
    overall = {
        "file": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "rows": len(df),
        "y=0": overall_zero,
        "y>0": overall_pos,
        "positive_rate": overall_pos / max(len(df), 1),
    }
    by_month = (
        df.groupby("target_month_idx")["target"]
        .agg(
            rows="size",
            **{
                "y=0": lambda s: int((s <= 0).sum()),
                "y>0": lambda s: int((s > 0).sum()),
                "positive_rate": lambda s: float((s > 0).mean()),
            },
        )
        .reset_index()
        .rename(columns={"target_month_idx": "target_month"})
    )
    return overall, by_month


def validation_summary(cp: pipe.Checkpoints, data) -> tuple[dict, str]:
    if cp.validation.exists():
        val = pd.read_parquet(cp.validation, columns=["target"])
        positives = int((val["target"] > 0).sum())
        zeros = int((val["target"] <= 0).sum())
        return {
            "source": str(cp.validation.relative_to(REPO_ROOT) if cp.validation.is_relative_to(REPO_ROOT) else cp.validation),
            "rows": len(val),
            "y=0": zeros,
            "y>0": positives,
            "positive_rate": positives / max(len(val), 1),
        }, "parquet"

    counts = target_counts_from_data(data, 12)
    counts["source"] = "monthly_data candidate universe; validation_dec.parquet missing"
    return counts, "computed"


def candidate_universe_summary(cp: pipe.Checkpoints, data) -> dict:
    hist_len = 12
    seen = data.pair_qty[:, :hist_len].sum(axis=1) > 0
    sale_status = data.pairs["sale_status"].to_numpy()
    kept = seen & (sale_status != 0)
    filtered_sale_status_zero = int((seen & (sale_status == 0)).sum())
    meta = data.pairs.loc[kept, ["location", "item_id", "sale_status"]]

    result = {
        "seen_pairs_before_target": int(seen.sum()),
        "rows_after_sale_status_filter": int(kept.sum()),
        "sale_status_zero_filtered": filtered_sale_status_zero,
        "active_locations": int(meta["location"].nunique()),
        "active_items": int(meta["item_id"].nunique()),
        "sale_status_values_after_filter": meta["sale_status"].value_counts(dropna=False).sort_index().to_dict(),
    }
    if cp.predict_frame.exists():
        pred = pd.read_parquet(cp.predict_frame, columns=["pair_id"])
        result["predict_frame_file"] = str(cp.predict_frame.relative_to(REPO_ROOT) if cp.predict_frame.is_relative_to(REPO_ROOT) else cp.predict_frame)
        result["predict_frame_rows"] = int(len(pred))
    else:
        result["predict_frame_file"] = "missing"
        result["predict_frame_rows"] = None
    return result


def fmt_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def dict_table(mapping: dict) -> str:
    rows = pd.DataFrame([{"metric": key, "value": fmt_value(value)} for key, value in mapping.items()])
    return rows.to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write train sampling diagnostics for a srcoptimized cache.")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "outputs/srcoptimized_cache_larger_1800k"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports/train_sampling_diagnostic.md"))
    args = parser.parse_args()

    cp = pipe.Checkpoints(Path(args.cache_dir))
    if not cp.monthly_data.exists():
        raise FileNotFoundError(cp.monthly_data)
    if not cp.train_val.exists():
        raise FileNotFoundError(cp.train_val)

    data = read_pickle(cp.monthly_data)
    train_overall, train_by_month = train_sample_summary(cp.train_val)
    val_counts, val_source = validation_summary(cp, data)
    candidate = candidate_universe_summary(cp, data)
    full_month_counts = pd.DataFrame([target_counts_from_data(data, month) for month in range(4, 12)])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Train Sampling Diagnostic",
        "",
        f"Cache: `{cp.cache_dir}`",
        "",
        "## Current Train Sample",
        "",
        dict_table(train_overall),
        "",
        "## Train Positive Rate By Target Month",
        "",
        train_by_month.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Full Candidate Universe By Train Month",
        "",
        full_month_counts.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Validation December",
        "",
        f"Source: `{val_source}`",
        "",
        dict_table(val_counts),
        "",
        "## Submission Candidate Universe",
        "",
        dict_table(candidate),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
