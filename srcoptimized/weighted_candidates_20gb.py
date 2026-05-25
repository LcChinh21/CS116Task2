"""
Validation-selected weighted LightGBM candidates for the 5-submit budget.

This runner is intentionally narrow:
  - CatBoost is disabled.
  - Three raw LightGBM weight modes are compared on December validation.
  - Scale tuning is local only: 0.70..1.15 step 0.025.
  - Group scale is kept only when it beats the candidate's global scale.
  - It writes exactly four new candidate CSVs; the fifth file is the 38.8 control.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "srcoptimized"))

import pipeline_20gb as pipe  # noqa: E402

base = pipe.base

WEIGHT_MODES = ("none", "inv_y", "inv_sqrt_y")
GROUP_NAMES = ("low_sale", "mid_sale", "high_sale")
SCALE_VALUES = np.round(np.arange(0.70, 1.1501, 0.025), 6)


@dataclass
class EvalContext:
    y_true: np.ndarray
    position: np.ndarray
    valid_position: np.ndarray
    group_code: np.ndarray

    def align_prediction(self, pred: np.ndarray) -> np.ndarray:
        aligned = np.zeros(len(self.y_true), dtype=np.float32)
        aligned[self.valid_position] = pred[self.position[self.valid_position]]
        return aligned


def cleanup() -> None:
    gc.collect()


def write_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def read_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str)) else False:
        return None
    return value


def metric_summary(y_true: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0, None)
    y_true = np.asarray(y_true, dtype=np.float64)
    mask = y_true > 0
    mape = float(np.mean(np.abs(y_true[mask] - pred[mask]) / np.maximum(y_true[mask], base.EPS)) * 100.0)
    mae = float(np.mean(np.abs(y_true - pred)))
    return {"mape_quantity": mape, "mae_quantity": mae}


def sample_weights(y: np.ndarray, mode: str) -> Optional[np.ndarray]:
    return pipe.sample_weights_for_mode(y, mode)


def callbacks():
    import lightgbm as lgb

    return [
        lgb.early_stopping(base.env_int("OPT_EARLY_STOPPING", 50), verbose=False),
        lgb.log_evaluation(period=100),
    ]


def eval_args_for(
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    weight_mode: str,
    weight_target: Optional[np.ndarray] = None,
) -> dict:
    weights = sample_weights(y_eval if weight_target is None else weight_target, weight_mode)
    args = {
        "eval_set": [(X_eval, y_eval)],
        "eval_metric": "l1",
        "callbacks": callbacks(),
    }
    if weights is not None:
        args["eval_sample_weight"] = [weights]
    return args


def train_validation_raw_models(train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str]) -> Tuple[dict, dict]:
    with pipe.timer("train validation raw models for all weight modes"):
        eval_rows = base.env_int("OPT_MAX_EVAL_ROWS", 600_000)
        eval_df = base.sample_training_rows(val_df, eval_rows, seed=base.RANDOM_STATE + 701)

        X_train = train_df[features]
        y_train = train_df["target"].to_numpy(dtype=np.float32)
        X_eval = eval_df[features]
        y_eval = eval_df["target"].to_numpy(dtype=np.float32)
        categorical = [col for col in ["location_code", "item_code", "category_code", "brand_code"] if col in features]
        params = pipe.lgbm_params()

        models: dict = {}
        val_preds: dict = {}
        for mode in WEIGHT_MODES:
            print(f"\n=== Train raw LightGBM weight_mode={mode} ===", flush=True)
            weights = sample_weights(y_train, mode)
            model = pipe.fit_lgbm_model(
                params,
                "regression_l1",
                X_train,
                y_train,
                weights,
                categorical,
                eval_args_for(X_eval, y_eval, mode),
            )
            models[mode] = model
            val_preds[mode] = pipe.predict_in_chunks(model, val_df, features, log_target=False)
            cleanup()

        return models, val_preds


def train_validation_log_model(train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str], weight_mode: str = "inv_y"):
    with pipe.timer(f"train validation log model weight_mode={weight_mode}"):
        eval_rows = base.env_int("OPT_MAX_EVAL_ROWS", 600_000)
        eval_df = base.sample_training_rows(val_df, eval_rows, seed=base.RANDOM_STATE + 702)

        X_train = train_df[features]
        y_train = train_df["target"].to_numpy(dtype=np.float32)
        X_eval = eval_df[features]
        y_eval = eval_df["target"].to_numpy(dtype=np.float32)
        categorical = [col for col in ["location_code", "item_code", "category_code", "brand_code"] if col in features]
        params = pipe.lgbm_params()
        weights = sample_weights(y_train, weight_mode)
        eval_args = eval_args_for(X_eval, np.log1p(y_eval), weight_mode, weight_target=y_eval)
        model = pipe.fit_lgbm_model(
            params,
            "regression",
            X_train,
            np.log1p(y_train),
            weights,
            categorical,
            eval_args,
        )
        pred = pipe.predict_in_chunks(model, val_df, features, log_target=True)
        return model, pred


def recent_mean(frame: pd.DataFrame) -> np.ndarray:
    if "li_qty_roll_mean_3" in frame.columns:
        return frame["li_qty_roll_mean_3"].to_numpy(dtype=np.float32)
    cols = [col for col in ["li_qty_lag1", "li_qty_lag2", "li_qty_lag3"] if col in frame.columns]
    if cols:
        return frame[cols].mean(axis=1).to_numpy(dtype=np.float32)
    return np.zeros(len(frame), dtype=np.float32)


def group_codes_from_frame(frame: pd.DataFrame) -> np.ndarray:
    recent = recent_mean(frame)
    codes = np.zeros(len(frame), dtype=np.int8)
    codes[(recent > 1.0) & (recent <= 5.0)] = 1
    codes[recent > 5.0] = 2
    return codes


def build_eval_context(data, val_df: pd.DataFrame, target_month: int = 12) -> EvalContext:
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

    source_groups = group_codes_from_frame(val_df)
    group_code = np.zeros(len(eval_pair_ids), dtype=np.int8)
    group_code[valid_position] = source_groups[position[valid_position]]
    return EvalContext(y_true=y_true, position=position, valid_position=valid_position, group_code=group_code)


def tune_global_scale(ctx: EvalContext, pred: np.ndarray) -> Tuple[np.ndarray, dict]:
    aligned = ctx.align_prediction(pred)
    rows = []
    best_metrics = {"mape_quantity": float("inf"), "mae_quantity": float("inf")}
    best_scale = 1.0
    best_pred = aligned
    for scale in SCALE_VALUES:
        candidate = np.clip(aligned * scale, 0, None).astype(np.float32)
        metrics = metric_summary(ctx.y_true, candidate)
        rows.append({"scale": float(scale), **metrics})
        key = (metrics["mape_quantity"], metrics["mae_quantity"], float(scale))
        best_key = (best_metrics["mape_quantity"], best_metrics["mae_quantity"], best_scale)
        if key < best_key:
            best_metrics = metrics
            best_scale = float(scale)
            best_pred = candidate
    return best_pred, {"scale": best_scale, **best_metrics, "all_scales": rows}


def tune_group_scale(ctx: EvalContext, pred: np.ndarray, global_params: dict) -> Tuple[np.ndarray, dict]:
    aligned = ctx.align_prediction(pred)
    scaled = aligned.copy()
    group_scales: dict[str, float] = {}
    group_mapes: dict[str, float] = {}
    positive = ctx.y_true > 0

    for code, name in enumerate(GROUP_NAMES):
        group_mask = ctx.group_code == code
        score_mask = group_mask & positive
        if not score_mask.any():
            group_scales[name] = float(global_params["scale"])
            group_mapes[name] = float("nan")
            scaled[group_mask] = aligned[group_mask] * group_scales[name]
            continue

        best_scale = float(global_params["scale"])
        best_sum = float("inf")
        best_group_mape = float("inf")
        y_group = ctx.y_true[score_mask].astype(np.float64)
        pred_group = aligned[score_mask].astype(np.float64)
        for scale in SCALE_VALUES:
            ape = np.abs(y_group - pred_group * scale) / np.maximum(y_group, base.EPS)
            ape_sum = float(ape.sum())
            if (ape_sum, float(scale)) < (best_sum, best_scale):
                best_sum = ape_sum
                best_scale = float(scale)
                best_group_mape = float(ape.mean() * 100.0)
        group_scales[name] = best_scale
        group_mapes[name] = best_group_mape
        scaled[group_mask] = aligned[group_mask] * best_scale

    scaled = np.clip(scaled, 0, None).astype(np.float32)
    metrics = metric_summary(ctx.y_true, scaled)
    keep = metrics["mape_quantity"] < float(global_params["mape_quantity"])
    params = {
        "group_scales": group_scales,
        "group_mapes": group_mapes,
        "mape_quantity": metrics["mape_quantity"],
        "mae_quantity": metrics["mae_quantity"],
        "beats_global": bool(keep),
    }
    return scaled, params


def apply_global_scale(pred: np.ndarray, params: dict) -> np.ndarray:
    return np.clip(pred * float(params["scale"]), 0, None).astype(np.float32)


def apply_group_scale(frame: pd.DataFrame, pred: np.ndarray, params: dict) -> np.ndarray:
    codes = group_codes_from_frame(frame)
    out = pred.astype(np.float32).copy()
    scales = params["group_scales"]
    for code, name in enumerate(GROUP_NAMES):
        out[codes == code] *= float(scales[name])
    return np.clip(out, 0, None).astype(np.float32)


def blend_prediction(mode: str, raw: np.ndarray, log_pred: Optional[np.ndarray], baseline: np.ndarray, raw_mape: float, log_mape: float) -> Tuple[np.ndarray, str]:
    if mode == "raw_only":
        return raw.astype(np.float32), "raw_only"
    if mode == "raw_log":
        if log_pred is None or raw_mape <= log_mape:
            return raw.astype(np.float32), "raw_only"
        return (0.5 * raw + 0.5 * log_pred).astype(np.float32), "raw_log"
    if mode == "raw_baseline_80_20":
        return (0.8 * raw + 0.2 * baseline).astype(np.float32), mode
    if mode == "raw_baseline_70_30":
        return (0.7 * raw + 0.3 * baseline).astype(np.float32), mode
    raise ValueError(mode)


def evaluate_candidates(data, val_df: pd.DataFrame, val_preds: dict, log_pred: Optional[np.ndarray]) -> Tuple[pd.DataFrame, dict]:
    ctx = build_eval_context(data, val_df)
    baseline = val_df["baseline_weighted"].to_numpy(dtype=np.float32)
    log_global = None
    if log_pred is not None:
        _, log_global = tune_global_scale(ctx, log_pred)

    rows = []
    params: dict = {"raw": {}, "blend": {}, "extra_choice": None}
    for mode, pred in val_preds.items():
        global_pred, global_params = tune_global_scale(ctx, pred)
        group_pred, group_params = tune_group_scale(ctx, pred, global_params)
        chosen_postprocess = "group_scale" if group_params["beats_global"] else "global_scale"
        chosen_mape = group_params["mape_quantity"] if group_params["beats_global"] else global_params["mape_quantity"]
        params["raw"][mode] = {
            "global": {k: v for k, v in global_params.items() if k != "all_scales"},
            "group": group_params,
            "chosen_postprocess": chosen_postprocess,
            "chosen_mape_quantity": chosen_mape,
        }
        rows.append(
            {
                "candidate": f"raw_{mode}",
                "weight_mode": mode,
                "blend_mode": "raw_only",
                "postprocess": "global_scale",
                "scale": global_params["scale"],
                "mape_quantity": global_params["mape_quantity"],
                "mae_quantity": global_params["mae_quantity"],
            }
        )
        rows.append(
            {
                "candidate": f"raw_{mode}_group",
                "weight_mode": mode,
                "blend_mode": "raw_only",
                "postprocess": "group_scale",
                "scale": json.dumps(group_params["group_scales"], sort_keys=True),
                "mape_quantity": group_params["mape_quantity"],
                "mae_quantity": group_params["mae_quantity"],
                "beats_global": group_params["beats_global"],
            }
        )

        raw_mape = global_params["mape_quantity"]
        log_mape = log_global["mape_quantity"] if log_global is not None else float("inf")
        for blend_mode in ("raw_log", "raw_baseline_80_20", "raw_baseline_70_30"):
            blended, effective_mode = blend_prediction(blend_mode, pred, log_pred, baseline, raw_mape, log_mape)
            blend_global_pred, blend_global_params = tune_global_scale(ctx, blended)
            params["blend"][f"{mode}:{blend_mode}"] = {
                "weight_mode": mode,
                "requested_blend_mode": blend_mode,
                "effective_blend_mode": effective_mode,
                "global": {k: v for k, v in blend_global_params.items() if k != "all_scales"},
            }
            rows.append(
                {
                    "candidate": f"{mode}_{blend_mode}",
                    "weight_mode": mode,
                    "blend_mode": effective_mode,
                    "postprocess": "global_scale",
                    "scale": blend_global_params["scale"],
                    "mape_quantity": blend_global_params["mape_quantity"],
                    "mae_quantity": blend_global_params["mae_quantity"],
                }
            )

    table = pd.DataFrame(rows).sort_values(["mape_quantity", "mae_quantity"]).reset_index(drop=True)

    extra_rows = table[
        ((table["postprocess"] == "group_scale") & (table.get("beats_global", False) == True))
        | (table["blend_mode"].isin(["raw_baseline_80_20", "raw_baseline_70_30"]))
    ].copy()
    if extra_rows.empty:
        extra_rows = table[table["candidate"].str.endswith("_group")].copy()
    params["extra_choice"] = json_safe(extra_rows.sort_values(["mape_quantity", "mae_quantity"]).iloc[0].to_dict())
    if log_global is not None:
        params["log_global"] = {k: v for k, v in log_global.items() if k != "all_scales"}
    return table, params


def train_final_raw_models(final_train_df: pd.DataFrame, features: List[str], val_models: dict) -> dict:
    with pipe.timer("train final raw models for all weight modes"):
        X_train = final_train_df[features]
        y_train = final_train_df["target"].to_numpy(dtype=np.float32)
        categorical = [col for col in ["location_code", "item_code", "category_code", "brand_code"] if col in features]
        final_models = {}
        for mode in WEIGHT_MODES:
            source = val_models[mode]
            n_estimators = int(getattr(source, "best_iteration_", None) or getattr(source, "n_estimators", 800))
            params = source.get_params()
            params.update({"n_estimators": max(50, n_estimators), "objective": "regression_l1"})
            weights = sample_weights(y_train, mode)
            print(f"\n=== Train final raw LightGBM weight_mode={mode} n_estimators={params['n_estimators']} ===", flush=True)
            model = pipe.fit_lgbm_model(params, "regression_l1", X_train, y_train, weights, categorical, eval_args={})
            final_models[mode] = model
            model.booster_.save_model(str(base.MODEL_DIR / f"optimized_lgbm_raw_{mode}.txt"))
            cleanup()
        return final_models


def predict_final_raw_models(models: dict, pred_df: pd.DataFrame, features: List[str]) -> dict:
    preds = {}
    for mode, model in models.items():
        preds[mode] = pipe.predict_in_chunks(model, pred_df, features, log_target=False)
    return preds


def make_submission_frame(data, pred_df: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    pair_ids = pred_df["pair_id"].to_numpy(dtype=np.int32)
    meta = data.pairs.iloc[pair_ids][["location", "item_id", "sale_status"]].copy()
    mask = meta["sale_status"].to_numpy() != 0
    submission = meta.loc[mask, ["location", "item_id"]].copy()
    submission["prediction"] = np.clip(prediction[mask].astype(np.float64), 0, None)
    submission = submission.drop_duplicates(["location", "item_id"]).reset_index(drop=True)
    submission["location"] = submission["location"].astype("int64")
    submission["item_id"] = submission["item_id"].astype("string[python]").astype(object)
    submission = submission[["location", "item_id", "prediction"]]
    submission.columns = pd.Index(["location", "item_id", "prediction"], dtype=object)
    return submission


def to_portal_pickle_frame(submission: pd.DataFrame) -> pd.DataFrame:
    portal = submission.rename(columns={"prediction": "quantity"}).copy()
    portal["location"] = pd.to_numeric(portal["location"], errors="raise").astype(np.int64)
    portal["item_id"] = pd.to_numeric(portal["item_id"], errors="raise").astype(np.int64)
    portal["quantity"] = pd.to_numeric(portal["quantity"], errors="coerce").fillna(0).clip(lower=0).astype(np.float64)
    return portal[["location", "item_id", "quantity"]]


def write_submission(path: Path, data, pred_df: pd.DataFrame, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    submission = make_submission_frame(data, pred_df, prediction)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        to_portal_pickle_frame(submission).to_pickle(path)
    else:
        submission.to_csv(path, index=False)
    print(f"wrote {path} rows={sum(data.pairs.iloc[pred_df['pair_id'].to_numpy(dtype=np.int32)]['sale_status'].to_numpy() != 0)}")


def candidate_prediction_for_submit(mode: str, pred_df: pd.DataFrame, final_preds: dict, params: dict) -> Tuple[np.ndarray, str, float]:
    raw = final_preds[mode]
    raw_params = params["raw"][mode]
    pred = apply_global_scale(raw, raw_params["global"])
    scale_desc = f"global scale {raw_params['global']['scale']:.3f}"
    return pred, scale_desc, float(raw_params["global"]["mape_quantity"])


def extra_prediction_for_submit(pred_df: pd.DataFrame, final_preds: dict, params: dict) -> Tuple[np.ndarray, str, str, float]:
    choice = params["extra_choice"]
    candidate = str(choice["candidate"])
    weight_mode = str(choice["weight_mode"])
    raw = final_preds[weight_mode]
    baseline = pred_df["baseline_weighted"].to_numpy(dtype=np.float32)

    if str(choice["postprocess"]) == "group_scale":
        group_params = params["raw"][weight_mode]["group"]
        pred = apply_group_scale(pred_df, raw, group_params)
        desc = f"group scales {json.dumps(group_params['group_scales'], sort_keys=True)}"
        return pred, "submission_group_scale_best.csv", desc, float(group_params["mape_quantity"])

    blend_key = None
    for key, value in params["blend"].items():
        if key == f"{weight_mode}:{candidate.removeprefix(weight_mode + '_')}":
            blend_key = key
            break
    if blend_key is None:
        for key, value in params["blend"].items():
            if value["weight_mode"] == weight_mode and value["effective_blend_mode"] == str(choice["blend_mode"]):
                blend_key = key
                break
    if blend_key is None:
        raise RuntimeError(f"Could not resolve blend params for {candidate}")

    blend_params = params["blend"][blend_key]
    requested = blend_params["requested_blend_mode"]
    if requested == "raw_baseline_80_20":
        blended = 0.8 * raw + 0.2 * baseline
    elif requested == "raw_baseline_70_30":
        blended = 0.7 * raw + 0.3 * baseline
    elif requested == "raw_log":
        blended = raw
    else:
        raise ValueError(requested)
    scale = float(blend_params["global"]["scale"])
    pred = np.clip(blended * scale, 0, None).astype(np.float32)
    desc = f"{requested} with global scale {scale:.3f}, weight_mode={weight_mode}"
    return pred, "submission_raw_baseline_blend_best.csv", desc, float(blend_params["global"]["mape_quantity"])


def write_submit_plan(entries: List[dict]) -> None:
    lines = ["# Submit Plan", ""]
    for idx, entry in enumerate(entries, start=1):
        lines.extend(
            [
                f"{idx}. `{entry['file']}`",
                f"   - validation MAPE: {entry['mape']}",
                f"   - scale/postprocess: {entry['postprocess']}",
                f"   - ly do nop: {entry['reason']}",
            ]
        )
    (REPO_ROOT / "submit_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_control_pickle() -> Path:
    control = base.OUTPUT_DIR / "submission_best_38_8.pkl"
    if control.exists():
        return control
    src = base.OUTPUT_DIR / "submission" / "submission_final_Score38_8.pkl"
    df = pd.read_pickle(src)[["location", "item_id", "quantity"]].copy()
    df["location"] = pd.to_numeric(df["location"], errors="raise").astype(np.int64)
    df["item_id"] = pd.to_numeric(df["item_id"], errors="raise").astype(np.int64)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).clip(lower=0).astype(np.float64)
    df.to_pickle(control)
    return control


def checkpoint_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(pd.read_parquet(path, columns=["pair_id"]))


def prepare_frames(cp: pipe.Checkpoints, force_samples: bool):
    data = pipe.stage_aggregate(cp, force=False)
    target_train_rows = base.env_int("OPT_MAX_TRAIN_ROWS", 1_200_000)
    target_final_rows = base.env_int("OPT_MAX_FINAL_TRAIN_ROWS", 1_800_000)
    rebuild_features = force_samples or checkpoint_row_count(cp.train_val) < target_train_rows
    train_df, val_df, features = pipe.stage_features(cp, force=rebuild_features)
    rebuild_final = force_samples or checkpoint_row_count(cp.final_train) < target_final_rows
    if rebuild_final or not cp.final_train.exists():
        final_train_df = pipe.build_sampled_training_frames(
            data,
            months=range(4, 13),
            label="final_train_jan_to_dec",
            total_rows=target_final_rows,
            final_cap=target_final_rows,
            seed_offset=300,
        )
        pipe.write_frame(cp.final_train, final_train_df)
    else:
        final_train_df = pipe.read_frame(cp.final_train)
    if cp.predict_frame.exists():
        pred_df = pipe.read_frame(cp.predict_frame)
    else:
        pred_df = base.build_feature_frame(data, 13, include_target=False, name="predict_jan_2026")
        pipe.write_frame(cp.predict_frame, pred_df)
    return data, train_df, val_df, final_train_df, pred_df, features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weighted raw LightGBM candidates and write a 5-file submit plan.")
    parser.add_argument("--cache-dir", default=str(base.OUTPUT_DIR / "srcoptimized_cache"))
    parser.add_argument("--profile", choices=["safe", "stronger", "none"], default=os.getenv("CS116_PROFILE", "safe"))
    parser.add_argument("--force-samples", action="store_true", help="Rebuild train/final samples even if cached row counts are large enough.")
    parser.add_argument("--reuse-validation", action="store_true", help="Reuse weighted validation models/predictions if present.")
    parser.add_argument("--reuse-final", action="store_true", help="Reuse weighted final models/predictions if present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OPT_RUN_CATBOOST", "0")
    os.environ.setdefault("OPT_MAX_TRAIN_ROWS", "1200000")
    os.environ.setdefault("OPT_MAX_FINAL_TRAIN_ROWS", "1800000")
    os.environ.setdefault("OPT_MAX_EVAL_ROWS", "600000")
    os.environ.setdefault("OPT_LGBM_TREES", "800")
    os.environ.setdefault("OPT_LGBM_LEAVES", "63")
    os.environ.setdefault("LGBM_MAX_BIN", "31")
    os.environ.setdefault("OPT_USE_RAW_ONLY", "1")
    pipe.apply_profile(args.profile)
    pipe.apply_defaults()

    cp = pipe.Checkpoints(Path(args.cache_dir))
    control = ensure_control_pickle()
    data, train_df, val_df, final_train_df, pred_df, features = prepare_frames(cp, force_samples=args.force_samples)

    weighted_dir = base.OUTPUT_DIR / "weighted_candidates"
    val_model_path = weighted_dir / "validation_raw_models.pkl"
    val_pred_path = weighted_dir / "validation_raw_predictions.pkl"
    log_pred_path = weighted_dir / "validation_log_prediction.pkl"
    final_model_path = weighted_dir / "final_raw_models.pkl"
    final_pred_path = weighted_dir / "final_raw_predictions.pkl"

    if args.reuse_validation and val_model_path.exists() and val_pred_path.exists() and log_pred_path.exists():
        val_models = read_pickle(val_model_path)
        val_preds = read_pickle(val_pred_path)
        log_payload = read_pickle(log_pred_path)
        log_pred = log_payload["prediction"]
    else:
        val_models, val_preds = train_validation_raw_models(train_df, val_df, features)
        write_pickle(val_model_path, val_models)
        write_pickle(val_pred_path, val_preds)
        _, log_pred = train_validation_log_model(train_df, val_df, features, weight_mode="inv_y")
        write_pickle(log_pred_path, {"prediction": log_pred})

    table, params = evaluate_candidates(data, val_df, val_preds, log_pred)
    table_path = base.OUTPUT_DIR / "weighted_candidate_validation_results.csv"
    table.to_csv(table_path, index=False)
    params_path = weighted_dir / "candidate_params.json"
    params_path.write_text(json.dumps(json_safe(params), indent=2), encoding="utf-8")
    print("\n=== Weighted Candidate Validation ===")
    print(table.to_string(index=False))

    del train_df, val_df, val_preds, log_pred
    cleanup()

    if args.reuse_final and final_model_path.exists() and final_pred_path.exists():
        final_preds = read_pickle(final_pred_path)
    else:
        final_models = train_final_raw_models(final_train_df, features, val_models)
        write_pickle(final_model_path, final_models)
        final_preds = predict_final_raw_models(final_models, pred_df, features)
        write_pickle(final_pred_path, final_preds)

    del final_train_df
    cleanup()

    output_specs = [
        ("none", base.OUTPUT_DIR / "submission_raw_only_scale_best.pkl", "unweighted raw LightGBM; local best global scale"),
        ("inv_y", base.OUTPUT_DIR / "submission_weighted_inv_y_scale_best.pkl", "MAPE-like inv_y sample weights; local best global scale"),
        ("inv_sqrt_y", base.OUTPUT_DIR / "submission_weighted_inv_sqrt_y_scale_best.pkl", "milder inv_sqrt_y sample weights; local best global scale"),
    ]
    plan_entries = [
        {
            "file": str(control.relative_to(REPO_ROOT)),
            "mape": "51.548672 existing Dec validation from best 38.8 run",
            "postprocess": "control artifact, no overwrite",
            "reason": "giu best public score 38.8 lam control",
        }
    ]

    for mode, path, reason in output_specs:
        pred, scale_desc, mape_value = candidate_prediction_for_submit(mode, pred_df, final_preds, params)
        write_submission(path, data, pred_df, pred)
        plan_entries.append(
            {
                "file": str(path.relative_to(REPO_ROOT)),
                "mape": f"{mape_value:.6f}",
                "postprocess": scale_desc,
                "reason": reason,
            }
        )

    extra_pred, extra_name, extra_desc, extra_mape = extra_prediction_for_submit(pred_df, final_preds, params)
    extra_name = Path(extra_name).with_suffix(".pkl").name
    extra_path = base.OUTPUT_DIR / extra_name
    write_submission(extra_path, data, pred_df, extra_pred)
    plan_entries.append(
        {
            "file": str(extra_path.relative_to(REPO_ROOT)),
            "mape": f"{extra_mape:.6f}",
            "postprocess": extra_desc,
            "reason": "best local candidate among group-scale improvements and raw-baseline blends",
        }
    )

    write_submit_plan(plan_entries)
    print("\nSubmit plan written to submit_plan.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
