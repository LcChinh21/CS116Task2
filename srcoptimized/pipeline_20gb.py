"""
Checkpointed 20GB-RAM pipeline for CS116 Task 2.

This wrapper reuses the production feature/model functions from
src/optimized_forecast_pipeline.py, but splits the workflow into resumable
stages and samples training frames before concatenation. It is intended for
machines around 12 CPU cores, 20GB RAM, and small GPUs.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import optimized_forecast_pipeline as base  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline_20gb")


DEFAULT_ENV = {
    # 20GB RAM / 4GB VRAM profile. Override any of these before running.
    "OMP_NUM_THREADS": "12",
    "OPT_N_JOBS": "12",
    "LGBM_USE_GPU": "1",
    "LGBM_DEVICE_TYPE": "gpu",
    "LGBM_GPU_PLATFORM_ID": "0",
    "LGBM_GPU_DEVICE_ID": "0",
    "LGBM_MAX_BIN": "31",
    "LGBM_GPU_USE_DP": "0",
    "OPT_LGBM_TREES": "500",
    "OPT_LGBM_LEAVES": "63",
    "OPT_LGBM_MIN_CHILD": "100",
    "OPT_LGBM_SUBSAMPLE": "0.80",
    "OPT_LGBM_COLSAMPLE": "0.75",
    "OPT_EARLY_STOPPING": "50",
    "OPT_MAX_TRAIN_ROWS": "700000",
    "OPT_MAX_FINAL_TRAIN_ROWS": "1000000",
    "OPT_MAX_EVAL_ROWS": "350000",
    "OPT_PRED_CHUNK_ROWS": "250000",
    # CatBoost GPU often exceeds 4GB VRAM. Enable explicitly if you want it.
    "OPT_RUN_CATBOOST": "0",
    "OPT_CATBOOST_USE_GPU": "0",
    "OPT_CATBOOST_MAX_ROWS": "400000",
    "OPT_CATBOOST_ITERS": "300",
    "OPT_CATBOOST_DEPTH": "6",
    # If LightGBM GPU raises an exception, retry that model on CPU.
    "OPT_FALLBACK_CPU": "1",
    "OPT_USE_RAW_ONLY": "1",
    "OPT_WEIGHT_MODE": "inv_y",
    "OPT_BLEND_MODE": "raw_only",
}


PROFILE_ENV = {
    "safe": {
        "OPT_RUN_CATBOOST": "0",
        "OPT_MAX_TRAIN_ROWS": "1200000",
        "OPT_MAX_FINAL_TRAIN_ROWS": "1800000",
        "OPT_MAX_EVAL_ROWS": "600000",
        "OPT_LGBM_TREES": "800",
        "OPT_LGBM_LEAVES": "63",
        "OPT_LGBM_MIN_CHILD": "80",
        "LGBM_MAX_BIN": "31",
        "OPT_USE_RAW_ONLY": "1",
        "OPT_WEIGHT_MODE": "inv_y",
        "OPT_BLEND_MODE": "raw_only",
    },
    "stronger": {
        "OPT_RUN_CATBOOST": "0",
        "OPT_MAX_TRAIN_ROWS": "1200000",
        "OPT_MAX_FINAL_TRAIN_ROWS": "1800000",
        "OPT_MAX_EVAL_ROWS": "600000",
        "OPT_LGBM_TREES": "800",
        "OPT_LGBM_LEAVES": "95",
        "OPT_LGBM_MIN_CHILD": "80",
        "LGBM_MAX_BIN": "31",
        "OPT_USE_RAW_ONLY": "1",
        "OPT_WEIGHT_MODE": "inv_y",
        "OPT_BLEND_MODE": "raw_only",
    },
}

WEIGHT_MODES = {"none", "inv_y", "inv_sqrt_y"}
BLEND_MODES = {"raw_only", "raw_log", "raw_baseline_80_20", "raw_baseline_70_30"}


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    start_rss = current_rss_mb()
    log.info("START %s | rss=%.1f MB", name, start_rss)
    try:
        yield
    finally:
        end_rss = current_rss_mb()
        log.info(
            "DONE  %s in %.1fs | rss=%.1f MB | delta=%.1f MB",
            name,
            time.perf_counter() - start,
            end_rss,
            end_rss - start_rss,
        )


def current_rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def apply_defaults() -> None:
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)


def apply_profile(profile: str) -> None:
    if profile == "none":
        return
    if profile not in PROFILE_ENV:
        raise ValueError(f"Unknown profile: {profile}")
    for key, value in PROFILE_ENV[profile].items():
        os.environ.setdefault(key, value)
    log.info("profile=%s env=%s", profile, {key: os.getenv(key) for key in PROFILE_ENV[profile]})


def cleanup() -> None:
    gc.collect()


def write_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    log.info("saved %s", path)


def read_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


class Checkpoints:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.monthly_data = cache_dir / "monthly_data.pkl"
        self.train_val = cache_dir / "train_val_sample.parquet"
        self.validation = cache_dir / "validation_dec.parquet"
        self.final_train = cache_dir / "final_train_sample.parquet"
        self.predict_frame = cache_dir / "predict_jan_2026.parquet"
        self.features = cache_dir / "features.json"
        self.val_models = cache_dir / "validation_lgbm_models.pkl"
        self.cat_model = cache_dir / "validation_cat_model.pkl"
        self.val_preds = cache_dir / "validation_model_predictions.parquet"
        self.ensemble_params = cache_dir / "ensemble_params.json"
        self.post_params = cache_dir / "postprocess_params.json"
        self.raw_post_params = cache_dir / "raw_only_postprocess_params.json"
        self.final_models = cache_dir / "final_models.pkl"
        self.status = cache_dir / "status.json"

    def paths(self) -> Dict[str, Path]:
        return {
            "monthly_data": self.monthly_data,
            "train_val": self.train_val,
            "validation": self.validation,
            "final_train": self.final_train,
            "predict_frame": self.predict_frame,
            "features": self.features,
            "val_models": self.val_models,
            "cat_model": self.cat_model,
            "val_preds": self.val_preds,
            "ensemble_params": self.ensemble_params,
            "post_params": self.post_params,
            "raw_post_params": self.raw_post_params,
            "final_models": self.final_models,
        }


def save_status(cp: Checkpoints, stage: str) -> None:
    payload = {
        "last_stage": stage,
        "env": {key: os.getenv(key) for key in sorted(DEFAULT_ENV)},
        "checkpoints": {name: str(path) for name, path in cp.paths().items() if path.exists()},
    }
    cp.status.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run stage '{hint}' first.")


def stage_aggregate(cp: Checkpoints, force: bool = False):
    if cp.monthly_data.exists() and not force:
        log.info("reuse %s", cp.monthly_data)
        return read_pickle(cp.monthly_data)

    max_row_groups = base.env_int("OPT_MAX_ROW_GROUPS", 0)
    log.info("OPT_MAX_ROW_GROUPS=%s (0 means all)", max_row_groups)
    purchases = base.aggregate_purchases(max_row_groups=max_row_groups)
    events = base.aggregate_events(max_row_groups=max_row_groups)
    items = base.load_items()
    data = base.encode_monthly_tables(purchases, events, items)
    del purchases, events, items
    cleanup()
    write_pickle(cp.monthly_data, data)
    save_status(cp, "aggregate")
    return data


def load_data(cp: Checkpoints):
    require(cp.monthly_data, "aggregate")
    return read_pickle(cp.monthly_data)


def sample_one_month(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0:
        return frame
    return base.sample_training_rows(frame, max_rows=max_rows, seed=seed)


def build_sampled_training_frames(
    data,
    months: Iterable[int],
    label: str,
    total_rows: int,
    final_cap: int,
    seed_offset: int,
) -> pd.DataFrame:
    months = list(months)
    per_month = max(1, int(np.ceil(total_rows / max(len(months), 1)))) if total_rows > 0 else 0
    frames: List[pd.DataFrame] = []
    with timer(f"build sampled training frames {label}"):
        for idx, month in enumerate(months):
            frame = base.build_feature_frame(data, month, include_target=True, name=f"{label}_m{month}")
            sampled = sample_one_month(frame, per_month, base.RANDOM_STATE + seed_offset + idx)
            log.info(
                "%s month=%s sampled rows=%s/%s positives=%s",
                label,
                month,
                len(sampled),
                len(frame),
                int((sampled["target"] > 0).sum()),
            )
            frames.append(sampled)
            del frame, sampled
            cleanup()

        result = pd.concat(frames, ignore_index=True)
        del frames
        cleanup()
        if final_cap > 0 and len(result) > final_cap:
            result = base.sample_training_rows(result, final_cap, seed=base.RANDOM_STATE + seed_offset + 99)
        log.info("%s sampled rows=%s cols=%s positive_rate=%.4f", label, len(result), result.shape[1], (result["target"] > 0).mean())
        return result


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    log.info("saved %s rows=%s cols=%s", path, len(frame), frame.shape[1])


def read_frame(path: Path) -> pd.DataFrame:
    log.info("loading %s", path)
    return pd.read_parquet(path)


def stage_features(cp: Checkpoints, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    data = load_data(cp)
    if cp.train_val.exists() and cp.validation.exists() and cp.features.exists() and not force:
        train_df = read_frame(cp.train_val)
        val_df = read_frame(cp.validation)
        features = json.loads(cp.features.read_text(encoding="utf-8"))
        return train_df, val_df, features

    max_train_rows = base.env_int("OPT_MAX_TRAIN_ROWS", 700_000)
    train_df = build_sampled_training_frames(
        data,
        months=range(4, 12),
        label="train_jan_to_nov",
        total_rows=max_train_rows,
        final_cap=max_train_rows,
        seed_offset=100,
    )
    val_df = base.build_feature_frame(data, 12, include_target=True, name="validation_dec")
    features = base.feature_columns(train_df)
    log.info("feature_count=%s", len(features))

    write_frame(cp.train_val, train_df)
    write_frame(cp.validation, val_df)
    cp.features.write_text(json.dumps(features, indent=2), encoding="utf-8")
    save_status(cp, "features")
    return train_df, val_df, features


def load_features(cp: Checkpoints) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    require(cp.train_val, "features")
    require(cp.validation, "features")
    require(cp.features, "features")
    return read_frame(cp.train_val), read_frame(cp.validation), json.loads(cp.features.read_text(encoding="utf-8"))


def lgbm_params() -> dict:
    params = {
        "n_estimators": base.env_int("OPT_LGBM_TREES", 500),
        "learning_rate": base.env_float("OPT_LGBM_LR", float(base.CFG.get("LGBM_COMMON", {}).get("learning_rate", 0.04))),
        "num_leaves": base.env_int("OPT_LGBM_LEAVES", 63),
        "min_child_samples": base.env_int("OPT_LGBM_MIN_CHILD", 100),
        "subsample": base.env_float("OPT_LGBM_SUBSAMPLE", 0.80),
        "subsample_freq": 1,
        "colsample_bytree": base.env_float("OPT_LGBM_COLSAMPLE", 0.75),
        "reg_alpha": base.env_float("OPT_LGBM_REG_ALPHA", 0.1),
        "reg_lambda": base.env_float("OPT_LGBM_REG_LAMBDA", 0.5),
        "random_state": base.RANDOM_STATE,
        "n_jobs": base.env_int("OPT_N_JOBS", 12),
        "verbosity": -1,
    }
    if base.env_flag("LGBM_USE_GPU", True):
        params.update(
            {
                "device_type": os.getenv("LGBM_DEVICE_TYPE", "gpu"),
                "gpu_platform_id": base.env_int("LGBM_GPU_PLATFORM_ID", 0),
                "gpu_device_id": base.env_int("LGBM_GPU_DEVICE_ID", 0),
                "max_bin": base.env_int("LGBM_MAX_BIN", 31),
                "gpu_use_dp": base.env_flag("LGBM_GPU_USE_DP", False),
            }
        )
        log.info(
            "LightGBM GPU enabled: device_type=%s platform=%s device=%s max_bin=%s gpu_use_dp=%s",
            params["device_type"],
            params["gpu_platform_id"],
            params["gpu_device_id"],
            params["max_bin"],
            params["gpu_use_dp"],
        )
    else:
        log.info("LightGBM GPU disabled")
    return params


def weight_mode_from_env(default: str = "inv_y") -> str:
    mode = os.getenv("OPT_WEIGHT_MODE", default).strip().lower()
    if mode not in WEIGHT_MODES:
        raise ValueError(f"OPT_WEIGHT_MODE must be one of {sorted(WEIGHT_MODES)}; got {mode!r}")
    return mode


def sample_weights_for_mode(y: np.ndarray, mode: str) -> Optional[np.ndarray]:
    if mode == "none":
        return None
    y_safe = np.maximum(np.asarray(y, dtype=np.float32), 1.0)
    if mode == "inv_y":
        return (1.0 / y_safe).astype("float32")
    if mode == "inv_sqrt_y":
        return (1.0 / np.sqrt(y_safe)).astype("float32")
    raise ValueError(f"Unknown weight mode: {mode}")


def blend_mode_from_env(default: str = "raw_only") -> str:
    mode = os.getenv("OPT_BLEND_MODE", default).strip().lower()
    if mode not in BLEND_MODES:
        raise ValueError(f"OPT_BLEND_MODE must be one of {sorted(BLEND_MODES)}; got {mode!r}")
    return mode


def make_fixed_blend(mode: str, raw_pred: np.ndarray, log_pred: np.ndarray, baseline_pred: np.ndarray) -> np.ndarray:
    if mode == "raw_only":
        return raw_pred.astype("float32")
    if mode == "raw_log":
        return (0.5 * raw_pred + 0.5 * log_pred).astype("float32")
    if mode == "raw_baseline_80_20":
        return (0.8 * raw_pred + 0.2 * baseline_pred).astype("float32")
    if mode == "raw_baseline_70_30":
        return (0.7 * raw_pred + 0.3 * baseline_pred).astype("float32")
    raise ValueError(f"Unknown blend mode: {mode}")


def tune_fixed_blend_scale(data, val_df: pd.DataFrame, pred: np.ndarray, blend_mode: str) -> Tuple[np.ndarray, dict]:
    pair_ids = val_df["pair_id"].to_numpy(dtype=np.int32)
    best = {"mape_quantity": float("inf")}
    best_pred = pred.copy()
    scales = np.round(np.arange(0.70, 1.1501, 0.025), 4)
    with timer(f"tune fixed blend scale mode={blend_mode}"):
        for scale in scales:
            candidate = np.clip(pred * scale, 0, None).astype("float32")
            metrics = base.score_predictions(data, pair_ids, candidate, target_month=12)
            if metrics["mape_quantity"] < best["mape_quantity"]:
                best = {"blend_mode": blend_mode, "scale": float(scale), **metrics}
                best_pred = candidate
        log.info("best fixed blend params=%s", best)
    return best_pred, best


def cpu_params(params: dict) -> dict:
    cpu = dict(params)
    for key in ["device_type", "gpu_platform_id", "gpu_device_id", "gpu_use_dp"]:
        cpu.pop(key, None)
    cpu["device_type"] = "cpu"
    cpu["n_jobs"] = base.env_int("OPT_N_JOBS", 12)
    return cpu


def lgbm_categorical_features(params: dict, features: List[str]) -> List[str]:
    categorical = [col for col in ["location_code", "item_code", "category_code", "brand_code"] if col in features]
    if params.get("device_type") == "gpu" and not base.env_flag("OPT_LGBM_GPU_CATEGORICAL", False):
        raw_cols = os.getenv("OPT_LGBM_GPU_CATEGORICAL_COLS", "category_code").strip()
        gpu_categorical = [col.strip() for col in raw_cols.split(",") if col.strip() in features]
        skipped = [col for col in categorical if col not in gpu_categorical]
        log.info(
            "LightGBM GPU categorical features=%s; high-cardinality encoded ids treated as numeric bins=%s",
            gpu_categorical,
            skipped,
        )
        return gpu_categorical
    return categorical


def predict_in_chunks(model, frame: pd.DataFrame, features: List[str], log_target: bool = False) -> np.ndarray:
    chunk_rows = base.env_int("OPT_PRED_CHUNK_ROWS", 250_000)
    chunks: List[np.ndarray] = []
    with timer(f"predict chunks rows={len(frame)} chunk={chunk_rows}"):
        for start in range(0, len(frame), chunk_rows):
            part = frame.iloc[start : start + chunk_rows]
            pred = model.predict(part[features])
            if log_target:
                pred = np.expm1(pred)
            chunks.append(np.clip(pred, 0, None).astype("float32"))
            del part, pred
            cleanup()
    return np.concatenate(chunks).astype("float32")


def fit_lgbm_model(params: dict, objective: str, X_train, y_train, weights, categorical, eval_args: dict):
    import lightgbm as lgb

    params = dict(params)
    params.pop("objective", None)
    model = lgb.LGBMRegressor(objective=objective, **params)
    try:
        model.fit(X_train, y_train, sample_weight=weights, categorical_feature=categorical, **eval_args)
        return model
    except Exception as exc:
        if not base.env_flag("OPT_FALLBACK_CPU", True) or params.get("device_type") != "gpu":
            raise
        log.warning("LightGBM GPU fit failed (%s). Retrying this model on CPU.", exc)
        fallback = lgb.LGBMRegressor(objective=objective, **cpu_params(params))
        fallback.fit(X_train, y_train, sample_weight=weights, categorical_feature=categorical, **eval_args)
        return fallback


def train_lightgbm_models_20gb(train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str]) -> Tuple[dict, pd.DataFrame]:
    import lightgbm as lgb

    with timer("train LightGBM raw/log models 20gb"):
        eval_rows = base.env_int("OPT_MAX_EVAL_ROWS", 350_000)
        eval_df = base.sample_training_rows(val_df, eval_rows, seed=base.RANDOM_STATE + 700)

        X_train = train_df[features]
        y_train = train_df["target"].to_numpy(dtype=np.float32)
        X_eval = eval_df[features]
        y_eval = eval_df["target"].to_numpy(dtype=np.float32)
        weight_mode = weight_mode_from_env()
        weights = sample_weights_for_mode(y_train, weight_mode)
        eval_weights = sample_weights_for_mode(y_eval, weight_mode)
        log.info("OPT_WEIGHT_MODE=%s", weight_mode)
        callbacks = [lgb.early_stopping(base.env_int("OPT_EARLY_STOPPING", 50), verbose=False), lgb.log_evaluation(period=100)]
        eval_args = {
            "eval_set": [(X_eval, y_eval)],
            "eval_metric": "l1",
            "callbacks": callbacks,
        }
        if eval_weights is not None:
            eval_args["eval_sample_weight"] = [eval_weights]
        params = lgbm_params()
        categorical = lgbm_categorical_features(params, features)

        raw_model = fit_lgbm_model(params, "regression_l1", X_train, y_train, weights, categorical, eval_args)

        eval_args_log = dict(eval_args)
        eval_args_log["eval_set"] = [(X_eval, np.log1p(y_eval))]
        log_model = fit_lgbm_model(params, "regression", X_train, np.log1p(y_train), weights, categorical, eval_args_log)

        pred_raw = predict_in_chunks(raw_model, val_df, features, log_target=False)
        pred_log = predict_in_chunks(log_model, val_df, features, log_target=True)
        preds = pd.DataFrame({"pair_id": val_df["pair_id"].to_numpy(dtype=np.int32), "lgbm_raw": pred_raw, "lgbm_log": pred_log})
        return {"lgbm_raw": raw_model, "lgbm_log": log_model}, preds


def stage_train(cp: Checkpoints, force: bool = False):
    data = load_data(cp)
    train_df, val_df, features = load_features(cp)
    if cp.val_models.exists() and cp.val_preds.exists() and cp.ensemble_params.exists() and cp.post_params.exists() and not force:
        log.info("reuse validation model checkpoints")
        lgbm_val_preds = read_frame(cp.val_preds)
    else:
        lgbm_models, lgbm_val_preds = train_lightgbm_models_20gb(train_df, val_df, features)
        cat_model = None
        if base.env_flag("OPT_RUN_CATBOOST", False):
            cat_model, cat_val_pred = base.train_catboost_if_possible(train_df, val_df, features)
            if cat_val_pred is not None:
                lgbm_val_preds["catboost"] = cat_val_pred

        write_pickle(cp.val_models, lgbm_models)
        write_pickle(cp.cat_model, cat_model)
        write_frame(cp.val_preds, lgbm_val_preds)

    pred_dict = {
        "baseline lag1": val_df["baseline_lag1"].to_numpy(dtype=np.float32),
        "baseline rolling3": val_df["baseline_rolling3"].to_numpy(dtype=np.float32),
        "LightGBM raw": lgbm_val_preds["lgbm_raw"].to_numpy(dtype=np.float32),
        "LightGBM log": lgbm_val_preds["lgbm_log"].to_numpy(dtype=np.float32),
    }
    if "catboost" in lgbm_val_preds.columns:
        pred_dict["CatBoost"] = lgbm_val_preds["catboost"].to_numpy(dtype=np.float32)

    baseline_pred_val = val_df["baseline_weighted"].to_numpy(dtype=np.float32)
    raw_pred_val = lgbm_val_preds["lgbm_raw"].to_numpy(dtype=np.float32)
    log_pred_val = lgbm_val_preds["lgbm_log"].to_numpy(dtype=np.float32)
    blend_mode = blend_mode_from_env()
    if blend_mode == "raw_log":
        raw_metrics = base.score_predictions(data, val_df["pair_id"].to_numpy(dtype=np.int32), raw_pred_val, target_month=12)
        log_metrics = base.score_predictions(data, val_df["pair_id"].to_numpy(dtype=np.int32), log_pred_val, target_month=12)
        if raw_metrics["mape_quantity"] <= log_metrics["mape_quantity"]:
            log.info("raw_log requested but raw validates better than log; using raw_only. raw=%.6f log=%.6f", raw_metrics["mape_quantity"], log_metrics["mape_quantity"])
            blend_mode = "raw_only"
    model_pred_val = make_fixed_blend(blend_mode, raw_pred_val, log_pred_val, baseline_pred_val)
    if blend_mode == "raw_only":
        ensemble_val_pred, ensemble_params = base.grid_search_ensemble(data, val_df, model_pred_val, baseline_pred_val)
        ensemble_params["blend_mode"] = blend_mode
    else:
        ensemble_val_pred, ensemble_params = tune_fixed_blend_scale(data, val_df, model_pred_val, blend_mode)
    post_val_pred, post_params = base.tune_postprocess(data, val_df, ensemble_val_pred)
    raw_post_pred, raw_post_params, safe_post_table = base.tune_safe_postprocess_options(
        data,
        val_df,
        lgbm_val_preds["lgbm_raw"].to_numpy(dtype=np.float32),
    )
    pred_dict["ensemble"] = ensemble_val_pred
    pred_dict["ensemble + postprocess"] = post_val_pred
    pred_dict["raw_only_postprocess"] = raw_post_pred

    validation_table = base.make_validation_table(data, val_df, pred_dict)
    log.info("\n%s", validation_table.to_string(index=False))
    base.write_report(validation_table, ensemble_params, post_params)
    safe_post_table.to_csv(base.OUTPUT_DIR / "postprocess_validation_results.csv", index=False)
    base.write_validation_predictions(
        base.make_validation_predictions(
            data,
            val_df,
            pred_raw=lgbm_val_preds["lgbm_raw"].to_numpy(dtype=np.float32),
            pred_log=lgbm_val_preds["lgbm_log"].to_numpy(dtype=np.float32),
            pred_ensemble=ensemble_val_pred,
            pred_baseline=baseline_pred_val,
            pred_raw_only_postprocess=raw_post_pred,
        )
    )

    cp.ensemble_params.write_text(json.dumps(ensemble_params, indent=2), encoding="utf-8")
    cp.post_params.write_text(json.dumps(post_params, indent=2), encoding="utf-8")
    cp.raw_post_params.write_text(json.dumps(raw_post_params, indent=2), encoding="utf-8")
    save_status(cp, "train")


def stage_final_train(cp: Checkpoints, force: bool = False):
    data = load_data(cp)
    require(cp.features, "features")
    require(cp.val_models, "train")
    features = json.loads(cp.features.read_text(encoding="utf-8"))
    val_models = read_pickle(cp.val_models)
    cat_model = read_pickle(cp.cat_model) if cp.cat_model.exists() else None

    if cp.final_models.exists() and not force:
        log.info("reuse %s", cp.final_models)
        return

    if cp.final_train.exists() and not force:
        final_train_df = read_frame(cp.final_train)
    else:
        max_final_rows = base.env_int("OPT_MAX_FINAL_TRAIN_ROWS", 1_000_000)
        final_train_df = build_sampled_training_frames(
            data,
            months=range(4, 13),
            label="final_train_jan_to_dec",
            total_rows=max_final_rows,
            final_cap=max_final_rows,
            seed_offset=300,
        )
        write_frame(cp.final_train, final_train_df)

    final_models = train_final_lgbm_20gb(final_train_df, features, val_models)
    if base.env_flag("OPT_RUN_CATBOOST", False):
        final_cat = base.train_final_catboost(final_train_df, features, cat_model)
        if final_cat is not None:
            final_models["catboost"] = final_cat

    write_pickle(cp.final_models, final_models)
    save_status(cp, "final")


def train_final_lgbm_20gb(train_df: pd.DataFrame, features: List[str], val_models: dict) -> dict:
    with timer("train final LightGBM models 20gb"):
        X_train = train_df[features]
        y_train = train_df["target"].to_numpy(dtype=np.float32)
        weight_mode = weight_mode_from_env()
        weights = sample_weights_for_mode(y_train, weight_mode)
        log.info("final OPT_WEIGHT_MODE=%s", weight_mode)
        params_probe = lgbm_params()
        categorical = lgbm_categorical_features(params_probe, features)
        final_models = {}
        for name, target_values, objective in [
            ("lgbm_raw", y_train, "regression_l1"),
            ("lgbm_log", np.log1p(y_train), "regression"),
        ]:
            source = val_models[name]
            n_estimators = int(getattr(source, "best_iteration_", None) or getattr(source, "n_estimators", 500))
            params = source.get_params()
            params.update({"n_estimators": max(50, n_estimators), "objective": objective})
            if params.get("device_type") == "gpu":
                categorical = lgbm_categorical_features(params, features)
            model = fit_lgbm_model(params, objective, X_train, target_values, weights, categorical, eval_args={})
            final_models[name] = model
            model.booster_.save_model(str(base.MODEL_DIR / f"optimized_{name}.txt"))
        return final_models


def stage_predict(cp: Checkpoints, force: bool = False) -> pd.DataFrame:
    data = load_data(cp)
    require(cp.features, "features")
    require(cp.final_models, "final")
    require(cp.ensemble_params, "train")
    require(cp.post_params, "train")
    features = json.loads(cp.features.read_text(encoding="utf-8"))
    final_models = read_pickle(cp.final_models)
    ensemble_params = json.loads(cp.ensemble_params.read_text(encoding="utf-8"))
    post_params = json.loads(cp.post_params.read_text(encoding="utf-8"))

    if cp.predict_frame.exists() and not force:
        pred_df = read_frame(cp.predict_frame)
    else:
        pred_df = base.build_feature_frame(data, 13, include_target=False, name="predict_jan_2026")
        write_frame(cp.predict_frame, pred_df)

    final_model_preds: Dict[str, np.ndarray] = {}
    if "lgbm_raw" in final_models:
        final_model_preds["lgbm_raw"] = predict_in_chunks(final_models["lgbm_raw"], pred_df, features, log_target=False)
    if "lgbm_log" in final_models:
        final_model_preds["lgbm_log"] = predict_in_chunks(final_models["lgbm_log"], pred_df, features, log_target=True)
    if "catboost" in final_models and final_models["catboost"] is not None:
        final_model_preds["catboost"] = np.clip(final_models["catboost"].predict(pred_df[features]), 0, None).astype("float32")

    if "lgbm_raw" in final_model_preds:
        raw_submission = base.save_submission(data, pred_df, final_model_preds["lgbm_raw"])
        raw_path = base.OUTPUT_DIR / "submission_raw_only.csv"
        raw_submission.to_csv(raw_path, index=False)
        log.info("saved raw-only candidate to %s", raw_path)

    final_baseline = pred_df["baseline_weighted"].to_numpy(dtype=np.float32)
    blend_mode = ensemble_params.get("blend_mode", blend_mode_from_env())
    final_model_pred = make_fixed_blend(
        blend_mode,
        final_model_preds["lgbm_raw"],
        final_model_preds.get("lgbm_log", final_model_preds["lgbm_raw"]),
        final_baseline,
    )
    if "alpha" in ensemble_params:
        final_ensemble = np.clip(
            (ensemble_params["alpha"] * final_model_pred + (1.0 - ensemble_params["alpha"]) * final_baseline)
            * ensemble_params["scale"],
            0,
            None,
        ).astype("float32")
    else:
        final_ensemble = np.clip(final_model_pred * ensemble_params["scale"], 0, None).astype("float32")
    final_control = base.apply_postprocess_with_params(
        data,
        pred_df["pair_id"].to_numpy(dtype=np.int32),
        final_ensemble,
        post_params,
        train_end_month=12,
    )
    control_submission = base.save_submission(data, pred_df, final_control)
    control_path = base.OUTPUT_DIR / "submission_control.csv"
    control_submission.to_csv(control_path, index=False)
    log.info("saved control candidate to %s", control_path)

    if base.env_flag("OPT_USE_RAW_ONLY", True):
        log.info("OPT_USE_RAW_ONLY=1: final submission uses direct LightGBM raw output")
        final_pred = final_model_preds["lgbm_raw"]
    else:
        log.info("OPT_USE_RAW_ONLY=0: final submission uses ensemble + selected postprocess")
        final_pred = final_control

    submission = base.save_submission(data, pred_df, final_pred)
    save_status(cp, "predict")
    return submission


def print_status(cp: Checkpoints) -> None:
    for name, path in cp.paths().items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            log.info("[ok] %-16s %8.1f MB %s", name, size_mb, path)
        else:
            log.info("[--] %-16s %s", name, path)


def run_stage(stage: str, cp: Checkpoints, force: bool) -> None:
    if stage == "status":
        print_status(cp)
    elif stage == "aggregate":
        stage_aggregate(cp, force=force)
    elif stage == "features":
        stage_features(cp, force=force)
    elif stage == "train":
        stage_train(cp, force=force)
    elif stage == "final":
        stage_final_train(cp, force=force)
    elif stage == "predict":
        stage_predict(cp, force=force)
    elif stage == "all":
        stage_aggregate(cp, force=force)
        stage_features(cp, force=force)
        stage_train(cp, force=force)
        stage_final_train(cp, force=force)
        stage_predict(cp, force=force)
    else:
        raise ValueError(stage)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpointed 20GB CS116 Task 2 pipeline.")
    parser.add_argument(
        "--stage",
        choices=["status", "aggregate", "features", "train", "final", "predict", "all"],
        default="all",
        help="Pipeline stage to run. Later stages reuse checkpoints from earlier stages.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(base.OUTPUT_DIR / "srcoptimized_cache"),
        help="Directory for intermediate checkpoints.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild checkpoints for the selected stage.")
    parser.add_argument("--no-defaults", action="store_true", help="Do not apply the built-in 20GB/4GB defaults.")
    parser.add_argument(
        "--profile",
        choices=["none", "safe", "stronger"],
        default=os.getenv("CS116_PROFILE", "safe"),
        help="Resource profile. safe=20GB default, stronger=larger sample; env vars still take precedence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_profile(args.profile)
    if not args.no_defaults:
        apply_defaults()
    cp = Checkpoints(Path(args.cache_dir))
    log.info("cache_dir=%s", cp.cache_dir)
    log.info("RANDOM_STATE=%s OPT_USE_RAW_ONLY=%s", base.RANDOM_STATE, os.getenv("OPT_USE_RAW_ONLY"))
    run_stage(args.stage, cp, force=args.force)


if __name__ == "__main__":
    main()
