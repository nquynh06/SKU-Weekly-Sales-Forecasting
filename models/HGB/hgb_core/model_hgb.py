"""Scikit-learn HistGradientBoostingRegressor pipeline, Val tuning, and explainability."""

from __future__ import annotations
import itertools
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .metrics import compute_regression_metrics


def _expm1_raw(y_log: np.ndarray) -> np.ndarray:
    return np.expm1(y_log)


def build_hgb_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    hgb_params: Dict[str, Any],
    target_transform: str = "log1p",
    seed: int = 42,
) -> Any:
    transformers = [("num", "passthrough", numeric_features)]
    cat_indices: List[int] = []
    if categorical_features:
        transformers.append(
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), categorical_features)
        )
        cat_indices = list(range(len(numeric_features), len(numeric_features) + len(categorical_features)))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    params = dict(hgb_params)
    params["random_state"] = seed
    params["categorical_features"] = cat_indices if cat_indices else None

    effective_transform = "none" if params.get("loss") == "poisson" else target_transform
    pipe = Pipeline([("prep", preprocessor), ("hgb", HistGradientBoostingRegressor(**params))])

    if effective_transform == "log1p":
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=_expm1_raw, check_inverse=False)
    return pipe


def tune_and_train_hgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    all_features: List[str],
    categorical_features: List[str],
    default_params: Dict[str, Any],
    tuning_grid: Dict[str, List[Any]] | None = None,
    tune: bool = True,
    target_transform: str = "log1p",
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    numeric_features = [c for c in all_features if c not in categorical_features]
    X_train, y_train = train_df[all_features], train_df["Actual_Quantity"].values
    X_val, y_val = val_df[all_features], val_df["Actual_Quantity"].values

    candidates: List[Dict[str, Any]] = [dict(default_params)]
    if tune and tuning_grid:
        keys = list(tuning_grid.keys())
        for combo in itertools.product(*[tuning_grid[k] for k in keys]):
            cfg = dict(default_params)
            for k, v in zip(keys, combo):
                cfg[k] = v
            if cfg not in candidates:
                candidates.append(cfg)

    tuning_records = []
    best_mae = float("inf")
    best_params = dict(default_params)

    for idx, params in enumerate(candidates):
        t0 = time.perf_counter()
        model = build_hgb_pipeline(numeric_features, categorical_features, params, target_transform, seed)
        model.fit(X_train, y_train)
        val_metrics = compute_regression_metrics(y_val, model.predict(X_val))
        tuning_records.append(
            {
                "trial": idx + 1,
                "loss": params.get("loss"),
                "learning_rate": params.get("learning_rate"),
                "max_leaf_nodes": params.get("max_leaf_nodes"),
                "min_samples_leaf": params.get("min_samples_leaf"),
                "l2_regularization": params.get("l2_regularization"),
                "val_MAE": val_metrics["MAE"],
                "val_RMSE": val_metrics["RMSE"],
                "val_WAPE_pct": val_metrics["WAPE_pct"],
                "val_R2": val_metrics["R2"],
                "fit_seconds": round(time.perf_counter() - t0, 3),
            }
        )
        if val_metrics["MAE"] < best_mae:
            best_mae = val_metrics["MAE"]
            best_params = dict(params)

    tuning_df = pd.DataFrame(tuning_records).sort_values("val_MAE", ascending=True).reset_index(drop=True)

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    t_final = time.perf_counter()
    final_model = build_hgb_pipeline(numeric_features, categorical_features, best_params, target_transform, seed)
    final_model.fit(train_val_df[all_features], train_val_df["Actual_Quantity"].values)

    best_info = {
        "best_params": best_params,
        "best_val_MAE": round(best_mae, 4),
        "n_trials": len(candidates),
        "final_refit_seconds": round(time.perf_counter() - t_final, 3),
        "target_transform": "none" if best_params.get("loss") == "poisson" else target_transform,
    }
    return final_model, best_info, tuning_df


def compute_feature_importance(model: Any, test_df: pd.DataFrame, all_features: List[str], n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    res = permutation_importance(
        model, test_df[all_features], test_df["Actual_Quantity"].values,
        scoring="neg_mean_absolute_error", n_repeats=n_repeats, random_state=seed, n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": all_features,
            "importance_mean_mae": np.round(res.importances_mean, 5),
            "importance_std": np.round(res.importances_std, 5),
        }
    ).sort_values("importance_mean_mae", ascending=False).reset_index(drop=True)
