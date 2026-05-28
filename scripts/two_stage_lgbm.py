#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "srcoptimized"))

import pipeline_20gb as pipe  # noqa: E402


BASELINE_MAPE = 51.178136


def read_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fit_params() -> dict:
    params = pipe.lgbm_params()
    params.pop("objective", None)
    return params


def metric_summary(y_true: np.ndarray, pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0, None)
    positive = y_true > 0
    mape = float("nan")
    if positive.any():
        mape = float(np.mean(np.abs(y_true[positive] - pred[positive]) / np.maximum(y_true[positive], pipe.base.EPS)) * 100.0)
    return {
        "mae_quantity": float(np.mean(np.abs(y_true - pred))),
        "mape_quantity": mape,
        "mean_prediction": float(np.mean(pred)),
        "min_prediction": float(np.min(pred)),
        "pct_pred_lt_0_1": float(np.mean(pred < 0.1) * 100.0),
        "pct_pred_lt_0_5": float(np.mean(pred < 0.5) * 100.0),
        "pct_pred_lt_1_0": float(np.mean(pred < 1.0) * 100.0),
    }


def validation_context(data, val_df: pd.DataFrame, pred: np.ndarray, target_month: int = 12) -> tuple[np.ndarray, np.ndarray]:
    actual_qty = data.pair_qty[:, target_month - 1]
    active_locs = set(data.pairs.loc[actual_qty > 0, "location_code"].astype(int).tolist())
    eval_mask = data.pairs["location_code"].isin(active_locs) & (data.pairs["sale_status"] != 0)
    eval_pair_ids = data.pairs.loc[eval_mask, "pair_id"].to_numpy(dtype=np.int32)
    source_pair_ids = val_df["pair_id"].to_numpy(dtype=np.int32)
    pred_map = pd.Series(np.clip(pred, 0, None).astype(np.float32), index=source_pair_ids)
    aligned = pred_map.reindex(eval_pair_ids, fill_value=0).to_numpy(dtype=np.float32)
    y_true = actual_qty[eval_pair_ids].astype(np.float32)
    return y_true, aligned


def group_metrics(y_true: np.ndarray, pred: np.ndarray) -> list[dict]:
    groups = [
        ("y=0", y_true == 0),
        ("0<y<=1", (y_true > 0) & (y_true <= 1)),
        ("1<y<=5", (y_true > 1) & (y_true <= 5)),
        ("y>5", y_true > 5),
    ]
    rows = []
    for name, mask in groups:
        if not mask.any():
            rows.append({"y_true_group": name, "rows": 0})
            continue
        rows.append({"y_true_group": name, "rows": int(mask.sum()), **metric_summary(y_true[mask], pred[mask])})
    return rows


def predict_classifier_in_chunks(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    chunk_rows = pipe.base.env_int("OPT_PRED_CHUNK_ROWS", 250_000)
    chunks = []
    for start in range(0, len(frame), chunk_rows):
        part = frame.iloc[start : start + chunk_rows]
        chunks.append(model.predict_proba(part[features])[:, 1].astype(np.float32))
    return np.concatenate(chunks).astype(np.float32)


def train_two_stage(train_df: pd.DataFrame, val_df: pd.DataFrame | None, features: list[str]):
    import lightgbm as lgb

    params = fit_params()
    categorical = pipe.lgbm_categorical_features(params, features)
    callbacks = []
    eval_classifier = None
    eval_regressor = None
    if val_df is not None:
        callbacks = [
            lgb.early_stopping(pipe.base.env_int("OPT_EARLY_STOPPING", 50), verbose=False),
            lgb.log_evaluation(period=100),
        ]
        eval_classifier = val_df
        eval_regressor = val_df[val_df["target"] > 0].copy()

    y_flag = (train_df["target"].to_numpy(dtype=np.float32) > 0).astype(np.int8)
    classifier = lgb.LGBMClassifier(**params)
    classifier_args = {"categorical_feature": categorical}
    if eval_classifier is not None:
        classifier_args.update(
            {
                "eval_set": [(eval_classifier[features], (eval_classifier["target"].to_numpy(dtype=np.float32) > 0).astype(np.int8))],
                "eval_metric": "binary_logloss",
                "callbacks": callbacks,
            }
        )
    classifier.fit(train_df[features], y_flag, **classifier_args)

    positive_train = train_df[train_df["target"] > 0].copy()
    y_pos = positive_train["target"].to_numpy(dtype=np.float32)
    weights = pipe.sample_weights_for_mode(y_pos, pipe.weight_mode_from_env())
    regressor = lgb.LGBMRegressor(objective="regression_l1", **params)
    regressor_args = {"sample_weight": weights, "categorical_feature": categorical}
    if eval_regressor is not None and len(eval_regressor) > 0:
        y_eval = eval_regressor["target"].to_numpy(dtype=np.float32)
        regressor_args.update(
            {
                "eval_set": [(eval_regressor[features], y_eval)],
                "eval_sample_weight": [pipe.sample_weights_for_mode(y_eval, pipe.weight_mode_from_env())],
                "eval_metric": "l1",
                "callbacks": callbacks,
            }
        )
    regressor.fit(positive_train[features], y_pos, **regressor_args)
    return classifier, regressor


def to_portal_pickle_frame(data, pred_df: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    pair_ids = pred_df["pair_id"].to_numpy(dtype=np.int32)
    meta = data.pairs.iloc[pair_ids][["location", "item_id", "sale_status"]].copy()
    mask = meta["sale_status"].to_numpy() != 0
    out = meta.loc[mask, ["location", "item_id"]].copy()
    out["quantity"] = np.clip(prediction[mask].astype(np.float64), 0, None)
    out = out.drop_duplicates(["location", "item_id"]).reset_index(drop=True)
    out["location"] = pd.to_numeric(out["location"], errors="raise").astype(np.int64)
    out["item_id"] = pd.to_numeric(out["item_id"], errors="raise").astype(np.int64)
    out["quantity"] = out["quantity"].astype(np.float64)
    return out[["location", "item_id", "quantity"]]


def ensure_final_train(cp: pipe.Checkpoints, data, force: bool) -> pd.DataFrame:
    if cp.final_train.exists() and not force:
        return pipe.read_frame(cp.final_train)
    max_final_rows = pipe.base.env_int("OPT_MAX_FINAL_TRAIN_ROWS", 1_000_000)
    final_train = pipe.build_sampled_training_frames(
        data,
        months=range(4, 13),
        label="two_stage_final_train_jan_to_dec",
        total_rows=max_final_rows,
        final_cap=max_final_rows,
        seed_offset=300,
    )
    pipe.write_frame(cp.final_train, final_train)
    return final_train


def ensure_predict_frame(cp: pipe.Checkpoints, data, force: bool) -> pd.DataFrame:
    if cp.predict_frame.exists() and not force:
        return pipe.read_frame(cp.predict_frame)
    pred_df = pipe.base.build_feature_frame(data, 13, include_target=False, name="predict_jan_2026")
    pipe.write_frame(cp.predict_frame, pred_df)
    return pred_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run two-stage LightGBM sale classifier + positive regressor.")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "outputs/srcoptimized_cache_neg_B"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports/two_stage_validation.md"))
    parser.add_argument("--json-out", default=str(REPO_ROOT / "outputs/two_stage_validation.json"))
    parser.add_argument("--submission-out", default=str(REPO_ROOT / "outputs/submission_two_stage_lgbm.pkl"))
    parser.add_argument("--baseline", type=float, default=BASELINE_MAPE)
    parser.add_argument("--make-submission-if-better", action="store_true")
    parser.add_argument("--force-final", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("LGBM_USE_GPU", "0")
    os.environ.setdefault("LGBM_DEVICE_TYPE", "cpu")
    os.environ.setdefault("OPT_RUN_CATBOOST", "0")

    cp = pipe.Checkpoints(Path(args.cache_dir))
    data = read_pickle(cp.monthly_data)
    train_df, val_df, features = pipe.load_features(cp)
    classifier, regressor = train_two_stage(train_df, val_df, features)
    prob = predict_classifier_in_chunks(classifier, val_df, features)
    reg_pred = pipe.predict_in_chunks(regressor, val_df, features, log_target=False)
    pred = np.clip(prob * reg_pred, 0, None).astype(np.float32)
    y_true, aligned = validation_context(data, val_df, pred)
    metrics = metric_summary(y_true, aligned)
    metrics["delta_vs_baseline"] = metrics["mape_quantity"] - float(args.baseline)
    metrics["cache_dir"] = display_path(cp.cache_dir)
    metrics["config"] = {
        "classifier": "LightGBM binary sale_flag",
        "regressor": "LightGBM regression_l1 on y>0 rows",
        "final_pred": "classifier_prob * regressor_pred",
        "weight_mode": os.getenv("OPT_WEIGHT_MODE", "inv_y"),
        "zero_weight": os.getenv("OPT_ZERO_WEIGHT", "1.0"),
    }
    groups = group_metrics(y_true, aligned)

    submission_path = Path(args.submission_out)
    if args.make_submission_if_better and metrics["mape_quantity"] < float(args.baseline):
        final_train = ensure_final_train(cp, data, force=args.force_final)
        final_classifier, final_regressor = train_two_stage(final_train, None, features)
        pred_df = ensure_predict_frame(cp, data, force=args.force_final)
        final_prob = predict_classifier_in_chunks(final_classifier, pred_df, features)
        final_reg = pipe.predict_in_chunks(final_regressor, pred_df, features, log_target=False)
        final_pred = np.clip(final_prob * final_reg, 0, None).astype(np.float32)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        to_portal_pickle_frame(data, pred_df, final_pred).to_pickle(submission_path)
        metrics["submission"] = display_path(submission_path)
    else:
        metrics["submission"] = ""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{k: v for k, v in metrics.items() if k != "config"}])
    group_df = pd.DataFrame(groups)
    out.write_text(
        "\n".join(
            [
                "# Two-Stage LightGBM Validation",
                "",
                f"Baseline MAPE: `{float(args.baseline):.6f}`",
                "",
                "## Config",
                "",
                "```json",
                json.dumps(metrics["config"], indent=2),
                "```",
                "",
                "## Summary",
                "",
                summary.to_markdown(index=False, floatfmt=".6f"),
                "",
                "## MAPE By y_true Group",
                "",
                group_df.to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps({"summary": metrics, "groups": groups}, indent=2, allow_nan=True), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"wrote {out}")
    print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
