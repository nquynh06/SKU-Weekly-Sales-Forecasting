"""Train-only preprocessing, explicit feature whitelist, RF and honest metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import Config
from .features import FEATURE_COLUMNS, NUMERIC_FEATURES


def fit_forest(rows: pd.DataFrame, params: dict, config: Config) -> Pipeline:
    train = rows.loc[rows.actual_quantity.notna() & rows.rf_eligible].copy()
    if train.empty:
        raise ValueError("No labeled RF-eligible training rows; check history, coverage and split")
    if not np.isfinite(train[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("RF-eligible training rows contain missing/nonfinite features")
    preprocess = ColumnTransformer([
        ("sku", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["SKU_ID"]),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ], remainder="drop", sparse_threshold=1.0)
    # Config max_features is always a FRACTION; JSON 1 must not mean one column.
    rf_params = {**params, "max_features": float(params["max_features"])}
    forest = RandomForestRegressor(criterion="squared_error", random_state=config.random_state,
                                   n_jobs=config.n_jobs, **rf_params)
    fitted = Pipeline([("preprocess", preprocess), ("forest", forest)])
    return fitted.fit(train[FEATURE_COLUMNS], train.actual_quantity.astype(float))


def predict_rows(model: Pipeline, rows: pd.DataFrame) -> pd.DataFrame:
    result = rows[["SKU_ID", "origin_week", "target_week", "actual_quantity", "baseline_naive", "baseline_ma4",
                   "history_known_last_4", "history_known_window", "rf_eligible"]].copy()
    fallback = rows.baseline_ma4.notna()
    result["qhat"] = rows.baseline_ma4.astype(float)
    result["prediction_status"] = np.where(fallback, "fallback_ma4", "unavailable")
    result["fallback_reason"] = np.where(fallback, "insufficient_contiguous_complete_history", "no_known_quantity_last_4_weeks")
    result["model_id"] = np.where(fallback, "ma4_fallback", "none")
    use_rf = rows.rf_eligible.to_numpy(dtype=bool)
    if use_rf.any():
        result.loc[use_rf, "qhat"] = np.maximum(0.0, model.predict(rows.loc[use_rf, FEATURE_COLUMNS]))
        result.loc[use_rf, "prediction_status"] = "random_forest"
        result.loc[use_rf, "fallback_reason"] = ""
        result.loc[use_rf, "model_id"] = "random_forest"
    known = set(model.named_steps["preprocess"].named_transformers_["sku"].categories_[0])
    result["sku_seen_in_training"] = rows.SKU_ID.isin(known)
    result["prediction_functional"] = "mean"
    return result


def regression_metrics(rows: pd.DataFrame, prediction_column: str) -> dict:
    labeled = rows.actual_quantity.notna()
    available = labeled & rows[prediction_column].notna()
    valid = rows.loc[available]
    n_total = len(rows)
    n_labeled = int(labeled.sum())
    result = {"total_rows": n_total, "labeled_rows": n_labeled, "scored_rows": len(valid),
              "prediction_coverage": float(rows[prediction_column].notna().mean()) if n_total else None,
              "label_coverage": float(labeled.mean()) if n_total else None,
              "scored_coverage_of_labels": len(valid) / n_labeled if n_labeled else None}
    if valid.empty:
        result.update(dict.fromkeys(["mae", "rmse", "wape", "bias", "macro_sku_mae", "macro_sku_wape"], None))
        result["macro_sku_wape_excluded_zero_denominator"] = 0
        return result
    actual = valid.actual_quantity.to_numpy(dtype=float)
    predicted = valid[prediction_column].to_numpy(dtype=float)
    error = predicted - actual
    denominator = float(actual.sum())
    by_sku = pd.DataFrame({"SKU_ID": valid.SKU_ID.to_numpy(), "absolute_error": np.abs(error), "actual": actual})
    grouped = by_sku.groupby("SKU_ID").agg(absolute_error=("absolute_error", "sum"), actual=("actual", "sum"))
    positive = grouped.actual.gt(0)
    result.update({"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
                   "wape": float(np.abs(error).sum() / denominator) if denominator else None,
                   "bias": float(error.sum() / denominator) if denominator else None,
                   "macro_sku_mae": float(by_sku.groupby("SKU_ID").absolute_error.mean().mean()),
                   "macro_sku_wape": float((grouped.loc[positive, "absolute_error"] / grouped.loc[positive, "actual"]).mean()) if positive.any() else None,
                   "macro_sku_wape_excluded_zero_denominator": int((~positive).sum())})
    return result


def evaluate_predictions(rows: pd.DataFrame) -> dict:
    names = {"random_forest_with_fallback": "qhat", "moving_average_4": "baseline_ma4", "naive": "baseline_naive"}
    all_available = rows.actual_quantity.notna() & rows[list(names.values())].notna().all(axis=1)
    common = rows.loc[all_available]
    return {"all_rows": {name: regression_metrics(rows, column) for name, column in names.items()},
            "common_keys": {name: regression_metrics(common, column) for name, column in names.items()},
            "common_key_rows": len(common),
            "by_week": {str(week.date()): {name: regression_metrics(block, column) for name, column in names.items()}
                        for week, block in rows.groupby("target_week")},
            "prediction_status_counts": rows.prediction_status.value_counts().to_dict()}
