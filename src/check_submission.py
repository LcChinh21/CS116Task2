"""
src/check_submission.py
=======================
Bước 7: Kiểm tra submission trước khi nộp.

Kiểm tra:
- Đúng 3 cột: location, item_id, prediction
- Không có NaN
- Không có prediction âm
- Không có duplicate location × item_id
- Top outliers
- sale_status=0 check
"""

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

OUTPUT_DIR   = REPO_ROOT / CFG["OUTPUT_DIR"]
DATA_DIR     = REPO_ROOT / CFG["DATA_DIR"]
LOCATION_COL = CFG["TX_LOCATION_COL"]
ITEM_COL     = CFG["TX_ITEM_COL"]
REQUIRED_COLS = cfg_cols = CFG["SUBMISSION_COLS"]
PRED_COL = REQUIRED_COLS[-1]


def load_submission(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path)


def check_submission(path: Path) -> bool:
    log.info(f"Checking submission: {path}")
    if not path.exists():
        log.error(f"File not found: {path}")
        return False

    sub = load_submission(path)

    ok = True
    errors = []
    warnings_list = []

    # ---- 1. Column check ---------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLS if c not in sub.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    else:
        log.info("✓ All required columns present: location, item_id, prediction")

    extra_cols = [c for c in sub.columns if c not in REQUIRED_COLS]
    if extra_cols:
        warnings_list.append(f"Extra columns (will be ignored): {extra_cols}")

    if errors:
        for e in errors:
            log.error(f"✗ {e}")
        return False

    # ---- 2. NaN check -------------------------------------------------------
    nan_counts = sub[REQUIRED_COLS].isnull().sum()
    if nan_counts.any():
        errors.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
    else:
        log.info("✓ No NaN values")

    # ---- 3. Negative prediction --------------------------------------------
    n_neg = (sub[PRED_COL] < 0).sum()
    if n_neg > 0:
        errors.append(f"Negative predictions: {n_neg}")
    else:
        log.info("✓ No negative predictions")

    # ---- 4. Duplicate check ------------------------------------------------
    n_dup = sub.duplicated(subset=[LOCATION_COL, ITEM_COL]).sum()
    if n_dup > 0:
        errors.append(f"Duplicate location × item_id pairs: {n_dup}")
    else:
        log.info("✓ No duplicates")

    # ---- 5. sale_status check -----------------------------------------------
    try:
        items = pd.read_parquet(DATA_DIR / CFG["ITEMS_FILE"])
        item_status = items[[CFG["ITEM_ID_COL"], CFG["ITEM_SALE_STATUS_COL"]]].rename(
            columns={CFG["ITEM_ID_COL"]: ITEM_COL, CFG["ITEM_SALE_STATUS_COL"]: "sale_status"}
        )
        sub_merged = sub.merge(item_status, on=ITEM_COL, how="left")
        sale_zero_nonzero = sub_merged[
            (sub_merged["sale_status"] == 0) & (sub_merged[PRED_COL] > 0)
        ]
        if len(sale_zero_nonzero) > 0:
            warnings_list.append(f"sale_status=0 items with prediction > 0: {len(sale_zero_nonzero)}")
        else:
            log.info("✓ sale_status=0 items correctly have prediction=0")
    except Exception as e:
        warnings_list.append(f"Could not check sale_status: {e}")

    # ---- Summary -----------------------------------------------------------
    print("\n=== Submission Check Report ===")
    print(f"  File          : {path}")
    print(f"  Rows          : {len(sub):,}")
    print(f"  Columns       : {sub.columns.tolist()}")
    print(f"  {PRED_COL} min: {sub[PRED_COL].min():.4f}")
    print(f"  {PRED_COL} max: {sub[PRED_COL].max():.4f}")
    print(f"  {PRED_COL} mean:{sub[PRED_COL].mean():.4f}")
    print(f"  Zeros         : {(sub[PRED_COL] == 0).sum():,}")
    print(f"  Non-zero      : {(sub[PRED_COL] > 0).sum():,}")

    if errors:
        print("\n❌ ERRORS:")
        for e in errors:
            print(f"  - {e}")
        ok = False
    if warnings_list:
        print("\n⚠ WARNINGS:")
        for w in warnings_list:
            print(f"  - {w}")

    print("\nTop-15 largest predictions (potential outliers):")
    print(sub.nlargest(15, PRED_COL)[[LOCATION_COL, ITEM_COL, PRED_COL]].to_string(index=False))

    if ok and not errors:
        print("\n✅ Submission looks valid!")

    return ok


if __name__ == "__main__":
    # Check final pickle submission by default, fallback to legacy CSV/baseline
    final_path    = OUTPUT_DIR / "submission_final.pkl"
    legacy_path   = OUTPUT_DIR / "submission_final.csv"
    baseline_path = OUTPUT_DIR / "submission_baseline.csv"

    if final_path.exists():
        check_submission(final_path)
    elif legacy_path.exists():
        log.warning("submission_final.pkl not found, checking legacy CSV instead.")
        check_submission(legacy_path)
    elif baseline_path.exists():
        log.warning("submission_final.pkl not found, checking baseline instead.")
        check_submission(baseline_path)
    else:
        log.error("No submission file found. Run baseline.py or postprocess.py first.")
