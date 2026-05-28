#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "srcoptimized"))

import pipeline_20gb as pipe  # noqa: E402


BASELINE_MAPE = 51.178136
SCALE_VALUES = np.round(np.arange(0.70, 1.1501, 0.025), 6)
REFINED_GROUP_NAMES = ("very_low", "low", "mid", "high", "very_high")


def read_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_mape(y_true: np.ndarray, pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - pred[mask]) / np.maximum(np.abs(y_true[mask]), pipe.base.EPS)) * 100.0)


def metric_summary(y_true: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0, None)
    y_true = np.asarray(y_true, dtype=np.float64)
    return {
        "mae_quantity": float(np.mean(np.abs(y_true - pred))),
        "mape_quantity": safe_mape(y_true, pred),
        "mean_prediction": float(np.mean(pred)),
        "min_prediction": float(np.min(pred)),
        "pct_pred_lt_0_1": float(np.mean(pred < 0.1) * 100.0),
        "pct_pred_lt_0_5": float(np.mean(pred < 0.5) * 100.0),
        "pct_pred_lt_1_0": float(np.mean(pred < 1.0) * 100.0),
    }


def recent_mean(frame: pd.DataFrame) -> np.ndarray:
    if "li_qty_roll_mean_3" in frame.columns:
        return frame["li_qty_roll_mean_3"].to_numpy(dtype=np.float32)
    cols = [col for col in ("li_qty_lag1", "li_qty_lag2", "li_qty_lag3") if col in frame.columns]
    if cols:
        return frame[cols].mean(axis=1).to_numpy(dtype=np.float32)
    return np.zeros(len(frame), dtype=np.float32)


def refined_group_codes(frame: pd.DataFrame) -> np.ndarray:
    recent = recent_mean(frame)
    codes = np.zeros(len(frame), dtype=np.int8)
    codes[(recent > 0.5) & (recent <= 1.0)] = 1
    codes[(recent > 1.0) & (recent <= 5.0)] = 2
    codes[(recent > 5.0) & (recent <= 20.0)] = 3
    codes[recent > 20.0] = 4
    return codes


def validation_context(data, val_df: pd.DataFrame, pred: np.ndarray, target_month: int = 12) -> dict:
    actual_qty = data.pair_qty[:, target_month - 1]
    active_locs = set(data.pairs.loc[actual_qty > 0, "location_code"].astype(int).tolist())
    eval_mask = data.pairs["location_code"].isin(active_locs) & (data.pairs["sale_status"] != 0)
    eval_pair_ids = data.pairs.loc[eval_mask, "pair_id"].to_numpy(dtype=np.int32)
    y_true = actual_qty[eval_pair_ids].astype(np.float32)

    source_pair_ids = val_df["pair_id"].to_numpy(dtype=np.int32)
    source_pos = pd.Series(np.arange(len(source_pair_ids), dtype=np.int64), index=source_pair_ids)
    position_float = source_pos.reindex(eval_pair_ids).to_numpy()
    valid_position = ~pd.isna(position_float)
    position = np.where(valid_position, position_float, -1).astype(np.int64)

    aligned = np.zeros(len(y_true), dtype=np.float32)
    aligned[valid_position] = np.clip(pred[position[valid_position]], 0, None).astype(np.float32)

    source_codes = refined_group_codes(val_df)
    codes = np.zeros(len(y_true), dtype=np.int8)
    codes[valid_position] = source_codes[position[valid_position]]
    return {"y_true": y_true, "pred": aligned, "codes": codes}


def tune_global(y_true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, dict]:
    best = {"scale": 1.0, "mape_quantity": float("inf"), "mae_quantity": float("inf")}
    best_pred = pred.astype(np.float32)
    for scale in SCALE_VALUES:
        candidate = np.clip(pred * float(scale), 0, None).astype(np.float32)
        metrics = metric_summary(y_true, candidate)
        key = (metrics["mape_quantity"], metrics["mae_quantity"], float(scale))
        best_key = (best["mape_quantity"], best["mae_quantity"], best["scale"])
        if key < best_key:
            best = {"scale": float(scale), **metrics}
            best_pred = candidate
    return best_pred, best


def tune_refined_group(y_true: np.ndarray, pred: np.ndarray, codes: np.ndarray, global_scale: float) -> tuple[np.ndarray, dict]:
    scaled = pred.astype(np.float32).copy()
    positive = y_true > 0
    group_scales: dict[str, float] = {}
    group_mapes: dict[str, float] = {}
    for code, name in enumerate(REFINED_GROUP_NAMES):
        group_mask = codes == code
        score_mask = group_mask & positive
        if not score_mask.any():
            group_scales[name] = float(global_scale)
            group_mapes[name] = float("nan")
            scaled[group_mask] = pred[group_mask] * float(global_scale)
            continue
        y_group = y_true[score_mask].astype(np.float64)
        pred_group = pred[score_mask].astype(np.float64)
        best_scale = float(global_scale)
        best_sum = float("inf")
        best_mape = float("inf")
        for scale in SCALE_VALUES:
            ape = np.abs(y_group - pred_group * float(scale)) / np.maximum(y_group, pipe.base.EPS)
            ape_sum = float(ape.sum())
            if (ape_sum, float(scale)) < (best_sum, best_scale):
                best_sum = ape_sum
                best_scale = float(scale)
                best_mape = float(ape.mean() * 100.0)
        group_scales[name] = best_scale
        group_mapes[name] = best_mape
        scaled[group_mask] = pred[group_mask] * best_scale
    scaled = np.clip(scaled, 0, None).astype(np.float32)
    return scaled, {"group_scales": group_scales, "group_mapes": group_mapes, **metric_summary(y_true, scaled)}


def group_metrics(tag: str, variant: str, y_true: np.ndarray, pred: np.ndarray) -> list[dict]:
    groups = [
        ("y=0", y_true == 0),
        ("0<y<=1", (y_true > 0) & (y_true <= 1)),
        ("1<y<=5", (y_true > 1) & (y_true <= 5)),
        ("y>5", y_true > 5),
    ]
    rows = []
    for name, mask in groups:
        if not mask.any():
            rows.append({"tag": tag, "variant": variant, "y_true_group": name, "rows": 0})
            continue
        metrics = metric_summary(y_true[mask], pred[mask])
        rows.append({"tag": tag, "variant": variant, "y_true_group": name, "rows": int(mask.sum()), **metrics})
    return rows


def evaluate_cache(tag: str, cache_dir: Path) -> tuple[list[dict], list[dict]]:
    cp = pipe.Checkpoints(cache_dir)
    for path in [cp.monthly_data, cp.validation, cp.val_preds]:
        if not path.exists():
            raise FileNotFoundError(path)
    data = read_pickle(cp.monthly_data)
    val_df = pd.read_parquet(cp.validation)
    val_preds = pd.read_parquet(cp.val_preds)
    raw = val_preds["lgbm_raw"].to_numpy(dtype=np.float32)

    ctx = validation_context(data, val_df, raw)
    y_true = ctx["y_true"]
    raw_aligned = ctx["pred"]
    global_pred, global_params = tune_global(y_true, raw_aligned)
    group_pred, group_params = tune_refined_group(y_true, raw_aligned, ctx["codes"], global_params["scale"])

    variants = [
        ("raw", raw_aligned, {"scale": 1.0}),
        ("global_scale", global_pred, {"scale": global_params["scale"]}),
        ("refined_5group", group_pred, {"group_scales": group_params["group_scales"]}),
    ]
    summary_rows = []
    detail_rows = []
    for variant, pred, params in variants:
        metrics = metric_summary(y_true, pred)
        summary_rows.append(
            {
                "tag": tag,
                "cache_dir": display_path(cache_dir),
                "variant": variant,
                **metrics,
                "delta_vs_baseline": metrics["mape_quantity"] - BASELINE_MAPE,
                "params": json.dumps(params, sort_keys=True),
            }
        )
        detail_rows.extend(group_metrics(tag, variant, y_true, pred))
    return summary_rows, detail_rows


def parse_cache_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    tag, raw_path = value.split("=", 1)
    return tag, Path(raw_path)


def write_markdown(out: Path, summary: pd.DataFrame, groups: pd.DataFrame) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Negative Sampling Validation",
        "",
        f"Baseline MAPE: `{BASELINE_MAPE:.6f}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## MAPE By y_true Group",
        "",
        groups.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    global BASELINE_MAPE

    parser = argparse.ArgumentParser(description="Evaluate negative sampling cache validation predictions.")
    parser.add_argument("--cache", action="append", required=True, help="TAG=cache_dir. May be repeated.")
    parser.add_argument("--out", default=str(REPO_ROOT / "reports/negative_sampling_validation.md"))
    parser.add_argument("--json-out", default=str(REPO_ROOT / "outputs/negative_sampling_validation_summary.json"))
    parser.add_argument("--baseline", type=float, default=BASELINE_MAPE)
    args = parser.parse_args()

    BASELINE_MAPE = float(args.baseline)

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    for cache_arg in args.cache:
        tag, cache_dir = parse_cache_arg(cache_arg)
        rows, details = evaluate_cache(tag, cache_dir)
        summary_rows.extend(rows)
        detail_rows.extend(details)

    summary = pd.DataFrame(summary_rows).sort_values(["mape_quantity", "mae_quantity"]).reset_index(drop=True)
    groups = pd.DataFrame(detail_rows).sort_values(["tag", "variant", "y_true_group"]).reset_index(drop=True)
    write_markdown(Path(args.out), summary, groups)
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps({"summary": summary_rows, "groups": detail_rows}, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"wrote {args.out}")
    print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
