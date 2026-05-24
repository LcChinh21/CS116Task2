#!/usr/bin/env python
"""
Create a small, validation-selected set of CS116 Task 2 submission candidates.

Outputs by default:
    submission_control.csv
    submission_raw_only.csv
    submission_scale_best.csv
    submission_scale_best_minus.csv
    submission_scale_best_plus.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.check_submission import check_submission, make_portable_submission_frame, normalize_submission

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs"
OFFICIAL_COLS = ["location", "item_id", "prediction"]


def load_submission(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path, dtype={"item_id": str})
    if "prediction" not in df.columns:
        candidates = [col for col in ["quantity", "quantity_pred", "pred"] if col in df.columns]
        if not candidates:
            raise ValueError(f"No prediction column found in {path}")
        df = df.rename(columns={candidates[0]: "prediction"})
    index_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if index_cols:
        df = df.drop(columns=index_cols)
    return make_portable_submission_frame(df[OFFICIAL_COLS])


def write_submission(df: pd.DataFrame, path: Path, scale: float = 1.0) -> Path:
    out = df.copy()
    out["prediction"] = np.clip(pd.to_numeric(out["prediction"], errors="coerce").fillna(0).to_numpy(dtype=np.float64) * scale, 0, None)
    out = make_portable_submission_frame(out).drop_duplicates(["location", "item_id"], keep="first").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    ok = check_submission(path)
    if not ok:
        raise RuntimeError(f"Generated file failed validation: {path}")
    return path


def read_best_scale(scale_json: Path | None, explicit_scale: float | None) -> float:
    if explicit_scale is not None:
        return float(explicit_scale)
    if scale_json and scale_json.exists():
        payload = json.loads(scale_json.read_text(encoding="utf-8"))
        return float(payload["best_scale"])
    return 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create validation-selected submission variants.")
    parser.add_argument("--base", default=str(OUTPUT_DIR / "submission_final.csv"), help="Control/base submission file.")
    parser.add_argument(
        "--raw-only",
        default=str(OUTPUT_DIR / "submission_raw_only_scale_1.00.csv"),
        help="Raw-only submission file, if available. Falls back to --base.",
    )
    parser.add_argument("--scale-json", default=str(OUTPUT_DIR / "scale_tuning_top.json"), help="JSON from tune_scale_local.py.")
    parser.add_argument("--best-scale", type=float, default=None, help="Override best local scale.")
    parser.add_argument("--delta", type=float, default=0.025, help="Scale neighborhood width.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for generated variants.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = Path(args.base)
    raw_path = Path(args.raw_only)
    scale_json = Path(args.scale_json)
    output_dir = Path(args.output_dir)
    for attr in ["base_path", "raw_path", "scale_json", "output_dir"]:
        path = locals()[attr]
        if not path.is_absolute():
            locals()[attr] = REPO_ROOT / path

    if not base_path.exists():
        raise FileNotFoundError(base_path)

    control_tmp = output_dir / "submission_control.csv"
    if raw_path.exists():
        raw_df = load_submission(raw_path)
    else:
        raw_df = load_submission(base_path)
        raw_path = base_path

    control_df = load_submission(base_path)
    best_scale = read_best_scale(scale_json, args.best_scale)
    minus_scale = max(0.0, best_scale - args.delta)
    plus_scale = best_scale + args.delta

    variants = [
        ("submission_control.csv", control_df, 1.0),
        ("submission_raw_only.csv", raw_df, 1.0),
        ("submission_scale_best.csv", raw_df, best_scale),
        ("submission_scale_best_minus.csv", raw_df, minus_scale),
        ("submission_scale_best_plus.csv", raw_df, plus_scale),
    ]

    written = []
    for filename, df, scale in variants:
        path = output_dir / filename
        written.append((path, scale, write_submission(df, path, scale=scale)))

    # Extra explicit names are handy when manually uploading.
    write_submission(raw_df, output_dir / f"submission_raw_only_scale_{best_scale:.3f}.csv", scale=best_scale)
    write_submission(raw_df, output_dir / f"submission_raw_only_scale_{minus_scale:.3f}.csv", scale=minus_scale)
    write_submission(raw_df, output_dir / f"submission_raw_only_scale_{plus_scale:.3f}.csv", scale=plus_scale)

    print("\n=== Submission Variants ===")
    for path, scale, _ in written:
        print(f"{path.name}: scale={scale:.3f}")
    print(f"Raw source: {raw_path}")
    print(f"Best local scale: {best_scale:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
