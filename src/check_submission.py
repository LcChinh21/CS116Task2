"""
Strict submission validator for CS116 Task 2.

Expected upload schema:
    location,item_id,prediction

The validator intentionally fails files that still use quantity/quantity_pred
or contain an index column. Use --normalize to write a corrected copy before
submitting an older artifact.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import yaml
except Exception:  # pragma: no cover - lightweight fallback for bare Python envs.
    yaml = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if yaml is not None:
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
else:
    CFG = {
        "OUTPUT_DIR": "outputs",
        "DATA_DIR": "data/data",
        "ITEMS_FILE": "items.parquet",
        "TX_LOCATION_COL": "location",
        "TX_ITEM_COL": "item_id",
        "ITEM_ID_COL": "item_id",
        "ITEM_SALE_STATUS_COL": "sale_status",
    }

OUTPUT_DIR = REPO_ROOT / CFG["OUTPUT_DIR"]
DATA_DIR = REPO_ROOT / CFG["DATA_DIR"]
LOCATION_COL = CFG["TX_LOCATION_COL"]
ITEM_COL = CFG["TX_ITEM_COL"]
PRED_COL = "prediction"
OFFICIAL_COLS = [LOCATION_COL, ITEM_COL, PRED_COL]
PORTAL_PKL_COLS = [LOCATION_COL, ITEM_COL, "quantity"]
RENAMABLE_PRED_COLS = ["prediction", "quantity", "quantity_pred", "pred"]


def load_submission(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path, dtype={ITEM_COL: str})


def make_portable_submission_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[LOCATION_COL] = pd.to_numeric(out[LOCATION_COL], errors="raise").astype(np.int64)
    out[ITEM_COL] = out[ITEM_COL].astype("string[python]").astype(object)
    out[PRED_COL] = pd.to_numeric(out[PRED_COL], errors="coerce").fillna(0).clip(lower=0).astype(np.float64)
    out.columns = pd.Index([str(col) for col in out.columns], dtype=object)
    return out[OFFICIAL_COLS]


def normalize_submission(input_path: Path, output_path: Path) -> Path:
    sub = load_submission(input_path)
    index_cols = [col for col in sub.columns if str(col).startswith("Unnamed:")]
    if index_cols:
        sub = sub.drop(columns=index_cols)

    pred_sources = [col for col in RENAMABLE_PRED_COLS if col in sub.columns]
    if not pred_sources:
        raise ValueError(f"No prediction-like column found in {input_path}. Columns={sub.columns.tolist()}")
    pred_source = pred_sources[0]

    missing_keys = [col for col in [LOCATION_COL, ITEM_COL] if col not in sub.columns]
    if missing_keys:
        raise ValueError(f"Missing key columns: {missing_keys}")

    out = sub[[LOCATION_COL, ITEM_COL, pred_source]].rename(columns={pred_source: PRED_COL})
    out = make_portable_submission_frame(out)
    before = len(out)
    out = out.drop_duplicates([LOCATION_COL, ITEM_COL], keep="first").reset_index(drop=True)
    if len(out) != before:
        log.warning("Dropped %s duplicate location-item rows while normalizing", before - len(out))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".pkl", ".pickle"}:
        out.to_pickle(output_path)
    else:
        out.to_csv(output_path, index=False)
    log.info("Normalized %s -> %s rows=%s", input_path, output_path, len(out))
    return output_path


def expected_schema(path: Path, portal_pickle: bool = False) -> tuple[list[str], str]:
    if portal_pickle or path.suffix.lower() in {".pkl", ".pickle"}:
        return PORTAL_PKL_COLS, "quantity"
    return OFFICIAL_COLS, PRED_COL


def check_submission(path: Path, strict: bool = True, check_sale_status: bool = True, portal_pickle: bool = False) -> bool:
    log.info("Checking submission: %s", path)
    if not path.exists():
        log.error("File not found: %s", path)
        return False

    sub = load_submission(path)
    ok = True
    errors: list[str] = []
    warnings_list: list[str] = []

    columns = [str(col) for col in sub.columns]
    required_cols, pred_col = expected_schema(path, portal_pickle=portal_pickle)
    if columns != required_cols:
        errors.append(f"Columns must be exactly {required_cols}; got {columns}")

    index_cols = [col for col in columns if col.startswith("Unnamed:")]
    if index_cols:
        errors.append(f"Index-like columns are not allowed: {index_cols}")

    if pred_col not in sub.columns:
        legacy_cols = [col for col in ["prediction", "quantity", "quantity_pred"] if col in sub.columns]
        if legacy_cols:
            errors.append(f"Rename {legacy_cols[0]} to {pred_col} before submission")
        else:
            errors.append(f"Missing {pred_col} column")

    missing_key_cols = [col for col in [LOCATION_COL, ITEM_COL] if col not in sub.columns]
    if missing_key_cols:
        errors.append(f"Missing key columns: {missing_key_cols}")

    if errors and strict:
        for err in errors:
            log.error("FAIL: %s", err)
        return False

    if pred_col in sub.columns:
        sub[pred_col] = pd.to_numeric(sub[pred_col], errors="coerce")
        present_required_cols = [col for col in required_cols if col in sub.columns]
        nan_counts = sub[present_required_cols].isnull().sum()
        if nan_counts.any():
            errors.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")

        non_finite = ~np.isfinite(sub[pred_col].fillna(np.nan).to_numpy(dtype=float))
        if non_finite.any():
            errors.append(f"Non-finite predictions: {int(non_finite.sum())}")

        n_neg = int((sub[pred_col] < 0).sum())
        if n_neg:
            errors.append(f"Negative predictions: {n_neg}")

    if LOCATION_COL in sub.columns and ITEM_COL in sub.columns:
        n_dup = int(sub.duplicated(subset=[LOCATION_COL, ITEM_COL]).sum())
        if n_dup:
            errors.append(f"Duplicate location-item rows: {n_dup}")

    if check_sale_status and pred_col in sub.columns and ITEM_COL in sub.columns:
        try:
            items = pd.read_parquet(DATA_DIR / CFG["ITEMS_FILE"], columns=[CFG["ITEM_ID_COL"], CFG["ITEM_SALE_STATUS_COL"]])
            item_status = items.rename(
                columns={CFG["ITEM_ID_COL"]: ITEM_COL, CFG["ITEM_SALE_STATUS_COL"]: "sale_status"}
            )
            item_status[ITEM_COL] = item_status[ITEM_COL].astype(str)
            merged = sub.copy()
            merged[ITEM_COL] = merged[ITEM_COL].astype(str)
            merged = merged.merge(item_status, on=ITEM_COL, how="left")
            sale_zero_nonzero = int(((merged["sale_status"] == 0) & (merged[pred_col] > 0)).sum())
            if sale_zero_nonzero:
                errors.append(f"sale_status=0 items with prediction > 0: {sale_zero_nonzero}")
        except Exception as exc:
            warnings_list.append(f"Could not check sale_status: {exc}")

    print("\n=== Submission Check Report ===")
    print(f"  File       : {path}")
    print(f"  Rows       : {len(sub):,}")
    print(f"  Columns    : {columns}")
    if pred_col in sub.columns:
        print(f"  Min        : {sub[pred_col].min():.6f}")
        print(f"  Max        : {sub[pred_col].max():.6f}")
        print(f"  Mean       : {sub[pred_col].mean():.6f}")
        print(f"  Zeros      : {(sub[pred_col] == 0).sum():,}")
        print(f"  Non-zero   : {(sub[pred_col] > 0).sum():,}")

    if errors:
        print("\nERRORS:")
        for err in errors:
            print(f"  - {err}")
        ok = False

    if warnings_list:
        print("\nWARNINGS:")
        for warning in warnings_list:
            print(f"  - {warning}")

    if pred_col in sub.columns:
        print("\nTop-15 largest predictions:")
        print(sub.nlargest(15, pred_col)[[LOCATION_COL, ITEM_COL, pred_col]].to_string(index=False))

    if ok:
        print("\nSubmission looks valid.")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CS116 Task 2 submission CSV/PKL.")
    parser.add_argument("path", nargs="?", default=str(OUTPUT_DIR / "submission_final.csv"), help="Submission file to validate.")
    parser.add_argument("--normalize", action="store_true", help="Write a corrected 3-column prediction file before checking.")
    parser.add_argument(
        "--out",
        default=str(OUTPUT_DIR / "submission_control_prediction_col.csv"),
        help="Output path used with --normalize.",
    )
    parser.add_argument("--no-sale-status-check", action="store_true", help="Skip sale_status=0 validation.")
    parser.add_argument("--official-csv", action="store_true", help="Require location,item_id,prediction even for pickle files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if args.normalize:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        path = normalize_submission(path, out)

    portal_pickle = path.suffix.lower() in {".pkl", ".pickle"} and not args.official_csv
    return 0 if check_submission(path, strict=True, check_sale_status=not args.no_sale_status_check, portal_pickle=portal_pickle) else 1


if __name__ == "__main__":
    raise SystemExit(main())
