#!/usr/bin/env python
"""
Tune a single multiplicative scale on December validation predictions.

Default input:
    outputs/validation_predictions.csv

Expected columns:
    location,item_id,y_true,pred_raw,pred_log,pred_ensemble[,pred_baseline]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "outputs" / "validation_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "scale_tuning_results.csv"


def mape(y_true: np.ndarray, pred: np.ndarray) -> float:
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - pred[mask]) / np.maximum(y_true[mask], 1e-6)) * 100.0)


def mae(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - pred)))


def candidate_pred_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Prediction column {requested!r} not found. Available={df.columns.tolist()}")
        return requested
    for col in ["pred_raw_only_postprocess", "pred_raw", "pred_ensemble", "pred_log"]:
        if col in df.columns:
            return col
    raise ValueError("No usable prediction column found.")


def tune_scale(df: pd.DataFrame, pred_col: str, start: float, stop: float, step: float) -> pd.DataFrame:
    if "y_true" not in df.columns:
        raise ValueError("validation_predictions.csv must contain y_true")
    y_true = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    base_pred = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    scales = np.round(np.arange(start, stop + step / 2.0, step), 6)
    rows = []
    for scale in scales:
        pred = np.clip(base_pred * scale, 0, None)
        rows.append(
            {
                "pred_col": pred_col,
                "scale": float(scale),
                "mape_quantity": mape(y_true, pred),
                "mae_quantity": mae(y_true, pred),
                "mean_prediction": float(pred.mean()),
                "nonzero_prediction": int((pred > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mape_quantity", "mae_quantity", "scale"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune submission scale using local December validation.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="validation_predictions.csv path.")
    parser.add_argument("--pred-col", default=None, help="Prediction column to scale; default prefers pred_raw_only_postprocess then pred_raw.")
    parser.add_argument("--start", type=float, default=0.70)
    parser.add_argument("--stop", type=float, default=1.15)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV path for all scale scores.")
    parser.add_argument("--json-out", default=str(REPO_ROOT / "outputs" / "scale_tuning_top.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    json_path = Path(args.json_out)
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path

    df = pd.read_csv(input_path)
    pred_col = candidate_pred_column(df, args.pred_col)
    results = tune_scale(df, pred_col, args.start, args.stop, args.step)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    top = results.head(args.top_k).copy()
    payload = {
        "input": str(input_path),
        "pred_col": pred_col,
        "top_scales": top["scale"].round(6).tolist(),
        "best_scale": float(top.iloc[0]["scale"]),
        "best_mape_quantity": float(top.iloc[0]["mape_quantity"]),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Local Scale Tuning ===")
    print(f"Input prediction column: {pred_col}")
    print(top.to_string(index=False, formatters={"scale": "{:.3f}".format, "mape_quantity": "{:.6f}".format, "mae_quantity": "{:.6f}".format}))
    print(f"\nSaved all scores: {output_path}")
    print(f"Saved top-scale JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
