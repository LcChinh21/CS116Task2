"""
Build deterministic inv_y/raw_only submission candidates from a pipeline cache.

This script does not train models. It uses validation predictions and final
models that already exist in a srcoptimized cache, tunes global/legacy/refined
group scales on December validation, and writes portal pickle submissions with
the location,item_id,quantity schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "srcoptimized"))

import pipeline_20gb as pipe  # noqa: E402

base = pipe.base

SCALE_VALUES = np.round(np.arange(0.70, 1.1501, 0.025), 6)
LEGACY_GROUP_NAMES = ("low_sale", "mid_sale", "high_sale")
REFINED_GROUP_NAMES = ("very_low", "low", "mid", "high", "very_high")


def recent_mean(frame: pd.DataFrame) -> np.ndarray:
    if "li_qty_roll_mean_3" in frame.columns:
        return frame["li_qty_roll_mean_3"].to_numpy(dtype=np.float32)
    cols = [col for col in ("li_qty_lag1", "li_qty_lag2", "li_qty_lag3") if col in frame.columns]
    if cols:
        return frame[list(cols)].mean(axis=1).to_numpy(dtype=np.float32)
    return np.zeros(len(frame), dtype=np.float32)


def legacy_group_codes(frame: pd.DataFrame) -> np.ndarray:
    recent = recent_mean(frame)
    codes = np.zeros(len(frame), dtype=np.int8)
    codes[(recent > 1.0) & (recent <= 5.0)] = 1
    codes[recent > 5.0] = 2
    return codes


def refined_group_codes(frame: pd.DataFrame) -> np.ndarray:
    recent = recent_mean(frame)
    codes = np.zeros(len(frame), dtype=np.int8)
    codes[(recent > 0.5) & (recent <= 1.0)] = 1
    codes[(recent > 1.0) & (recent <= 5.0)] = 2
    codes[(recent > 5.0) & (recent <= 20.0)] = 3
    codes[recent > 20.0] = 4
    return codes


def metric_summary(y_true: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0, None)
    y_true = np.asarray(y_true, dtype=np.float64)
    mask = y_true > 0
    mape = float(np.mean(np.abs(y_true[mask] - pred[mask]) / np.maximum(y_true[mask], base.EPS)) * 100.0)
    mae = float(np.mean(np.abs(y_true - pred)))
    return {"mape_quantity": mape, "mae_quantity": mae}


def validation_context(data, val_df: pd.DataFrame, raw_pred: np.ndarray, target_month: int = 12) -> dict:
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
    aligned[valid_position] = raw_pred[position[valid_position]]

    return {
        "y_true": y_true,
        "aligned": aligned,
        "legacy_codes": align_codes(val_df, legacy_group_codes(val_df), eval_pair_ids, source_pos, valid_position, position),
        "refined_codes": align_codes(val_df, refined_group_codes(val_df), eval_pair_ids, source_pos, valid_position, position),
    }


def align_codes(
    val_df: pd.DataFrame,
    source_codes: np.ndarray,
    eval_pair_ids: np.ndarray,
    source_pos: pd.Series,
    valid_position: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    del val_df, eval_pair_ids, source_pos
    codes = np.zeros(len(valid_position), dtype=np.int8)
    codes[valid_position] = source_codes[position[valid_position]]
    return codes


def tune_global(y_true: np.ndarray, aligned: np.ndarray) -> Tuple[np.ndarray, dict]:
    rows = []
    best_pred = aligned
    best = {"scale": 1.0, "mape_quantity": float("inf"), "mae_quantity": float("inf")}
    for scale in SCALE_VALUES:
        pred = np.clip(aligned * float(scale), 0, None).astype(np.float32)
        metrics = metric_summary(y_true, pred)
        row = {"scale": float(scale), **metrics}
        rows.append(row)
        key = (metrics["mape_quantity"], metrics["mae_quantity"], float(scale))
        best_key = (best["mape_quantity"], best["mae_quantity"], best["scale"])
        if key < best_key:
            best = row
            best_pred = pred
    return best_pred, {**best, "all_scales": rows}


def tune_group(
    y_true: np.ndarray,
    aligned: np.ndarray,
    codes: np.ndarray,
    names: Iterable[str],
    global_scale: float,
) -> Tuple[np.ndarray, dict]:
    scaled = aligned.copy()
    positive = y_true > 0
    group_scales: Dict[str, float] = {}
    group_mapes: Dict[str, float] = {}
    for code, name in enumerate(names):
        group_mask = codes == code
        score_mask = group_mask & positive
        if not score_mask.any():
            group_scales[name] = float(global_scale)
            group_mapes[name] = float("nan")
            scaled[group_mask] = aligned[group_mask] * float(global_scale)
            continue
        y_group = y_true[score_mask].astype(np.float64)
        pred_group = aligned[score_mask].astype(np.float64)
        best_scale = float(global_scale)
        best_sum = float("inf")
        best_mape = float("inf")
        for scale in SCALE_VALUES:
            ape = np.abs(y_group - pred_group * float(scale)) / np.maximum(y_group, base.EPS)
            ape_sum = float(ape.sum())
            if (ape_sum, float(scale)) < (best_sum, best_scale):
                best_sum = ape_sum
                best_scale = float(scale)
                best_mape = float(ape.mean() * 100.0)
        group_scales[name] = best_scale
        group_mapes[name] = best_mape
        scaled[group_mask] = aligned[group_mask] * best_scale
    scaled = np.clip(scaled, 0, None).astype(np.float32)
    metrics = metric_summary(y_true, scaled)
    return scaled, {"group_scales": group_scales, "group_mapes": group_mapes, **metrics}


def apply_group(frame: pd.DataFrame, pred: np.ndarray, scales: dict, refined: bool) -> np.ndarray:
    names = REFINED_GROUP_NAMES if refined else LEGACY_GROUP_NAMES
    codes = refined_group_codes(frame) if refined else legacy_group_codes(frame)
    out = pred.astype(np.float32).copy()
    for code, name in enumerate(names):
        out[codes == code] *= float(scales[name])
    return np.clip(out, 0, None).astype(np.float32)


def make_submission(data, pred_df: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    pair_ids = pred_df["pair_id"].to_numpy(dtype=np.int32)
    meta = data.pairs.iloc[pair_ids][["location", "item_id", "sale_status"]].copy()
    mask = meta["sale_status"].to_numpy() != 0
    submission = meta.loc[mask, ["location", "item_id"]].copy()
    submission["prediction"] = np.clip(prediction[mask].astype(np.float64), 0, None)
    submission = submission.drop_duplicates(["location", "item_id"]).reset_index(drop=True)
    submission["location"] = submission["location"].astype("int64")
    submission["item_id"] = submission["item_id"].astype("string[python]").astype(object)
    return submission[["location", "item_id", "prediction"]]


def to_portal_pickle_frame(submission: pd.DataFrame) -> pd.DataFrame:
    portal = submission.rename(columns={"prediction": "quantity"}).copy()
    portal["location"] = pd.to_numeric(portal["location"], errors="raise").astype(np.int64)
    portal["item_id"] = pd.to_numeric(portal["item_id"], errors="raise").astype(np.int64)
    portal["quantity"] = pd.to_numeric(portal["quantity"], errors="coerce").fillna(0).clip(lower=0).astype(np.float64)
    portal.columns = pd.Index(["location", "item_id", "quantity"], dtype=object)
    return portal[["location", "item_id", "quantity"]]


def write_submission(path: Path, data, pred_df: pd.DataFrame, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    submission = make_submission(data, pred_df, prediction)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        to_portal_pickle_frame(submission).to_pickle(path)
    else:
        submission.to_csv(path, index=False)
    print(f"wrote {path}")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_final_raw_prediction(cp: pipe.Checkpoints, pred_df: pd.DataFrame, features: list[str], tag: str) -> np.ndarray:
    cache_path = base.OUTPUT_DIR / "candidate_final_predictions" / f"{tag}_lgbm_raw.npy"
    if cache_path.exists() and cache_path.stat().st_mtime >= cp.final_models.stat().st_mtime:
        return np.load(cache_path)
    final_models = pipe.read_pickle(cp.final_models)
    pred = pipe.predict_in_chunks(final_models["lgbm_raw"], pred_df, features, log_target=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, pred.astype(np.float32))
    return pred.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune inv_y raw-only global and group-scale candidates from a cache.")
    parser.add_argument("--cache-dir", default=str(base.OUTPUT_DIR / "srcoptimized_cache"))
    parser.add_argument("--tag", default="base")
    parser.add_argument("--raw-out", default="")
    parser.add_argument("--group-out", default="")
    parser.add_argument("--metadata-out", default="")
    parser.add_argument("--plan-out", default="")
    parser.add_argument("--control", default=str(base.OUTPUT_DIR / "submission_best_39_0.pkl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cp = pipe.Checkpoints(Path(args.cache_dir))
    data = pipe.read_pickle(cp.monthly_data)
    val_df = pipe.read_frame(cp.validation)
    val_preds = pd.read_parquet(cp.val_preds)
    pred_df = pipe.read_frame(cp.predict_frame)
    features = json.loads(cp.features.read_text(encoding="utf-8"))

    raw_pred = val_preds["lgbm_raw"].to_numpy(dtype=np.float32)
    ctx = validation_context(data, val_df, raw_pred)
    global_pred, global_params = tune_global(ctx["y_true"], ctx["aligned"])
    legacy_pred, legacy_params = tune_group(
        ctx["y_true"], ctx["aligned"], ctx["legacy_codes"], LEGACY_GROUP_NAMES, global_params["scale"]
    )
    refined_pred, refined_params = tune_group(
        ctx["y_true"], ctx["aligned"], ctx["refined_codes"], REFINED_GROUP_NAMES, global_params["scale"]
    )
    del global_pred, legacy_pred, refined_pred

    results = {
        "tag": args.tag,
        "cache_dir": str(Path(args.cache_dir)),
        "global": {k: v for k, v in global_params.items() if k != "all_scales"},
        "legacy_group": legacy_params,
        "refined_5_group": refined_params,
        "refined_beats_global": refined_params["mape_quantity"] < global_params["mape_quantity"],
        "refined_beats_legacy": refined_params["mape_quantity"] < legacy_params["mape_quantity"],
        "features": {
            "n_features": len(features),
            "event_extra_enabled": any(col.startswith("item_view_") for col in features),
        },
    }

    final_raw = load_final_raw_prediction(cp, pred_df, features, args.tag)
    raw_out = Path(args.raw_out) if args.raw_out else base.OUTPUT_DIR / f"submission_{args.tag}_inv_y_raw.pkl"
    group_out = Path(args.group_out) if args.group_out else base.OUTPUT_DIR / f"submission_{args.tag}_inv_y_5group_scale.pkl"

    raw_final = np.clip(final_raw * float(global_params["scale"]), 0, None).astype(np.float32)

    write_submission(raw_out, data, pred_df, raw_final)
    results["outputs"] = {
        "control_best_39_0": display_path(Path(args.control)),
        "raw": display_path(raw_out),
    }
    if refined_params["mape_quantity"] < global_params["mape_quantity"]:
        refined_final = apply_group(pred_df, final_raw, refined_params["group_scales"], refined=True)
        write_submission(group_out, data, pred_df, refined_final)
        results["outputs"]["refined_5_group"] = display_path(group_out)
        results["selected_group_scale"] = True
    else:
        if group_out.exists():
            group_out.unlink()
        results["selected_group_scale"] = False
        results["skipped_group_out"] = display_path(group_out)

    meta_out = Path(args.metadata_out) if args.metadata_out else base.OUTPUT_DIR / f"candidate_validation_{args.tag}.json"
    meta_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.plan_out:
        plan_out = Path(args.plan_out)
        plan_lines = [
            "# Submit Plan",
            "",
            "1. `outputs/submission_best_39_0.pkl`",
            "   - role: control_best_39_0",
            "   - public score: 39.0",
            "   - reason: locked control, do not overwrite",
            "",
            f"2. `{display_path(raw_out)}`",
            "   - role: larger_sample_inv_y_raw",
            f"   - validation MAPE: {global_params['mape_quantity']:.6f}",
            f"   - scale/postprocess: global scale {global_params['scale']:.3f}",
            "   - reason: new model/feature candidate, not random scale probing",
            "",
            f"3. `{display_path(group_out)}`",
            "   - role: larger_sample_inv_y_raw + 5_group_scale",
            f"   - validation MAPE: {refined_params['mape_quantity']:.6f}",
            f"   - selected: {refined_params['mape_quantity'] < global_params['mape_quantity']}",
            f"   - scales: `{json.dumps(refined_params['group_scales'], sort_keys=True)}`",
            "   - reason: keep only when validation beats raw",
        ]
        plan_out.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
