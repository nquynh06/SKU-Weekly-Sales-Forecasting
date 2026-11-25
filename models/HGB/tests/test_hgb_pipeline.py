"""Automated verification tests for HGB exact math, Spec compliance, and N-day forecasting."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hgb_core.algorithm_math import ExactHistogramGBDT, compute_gradients_hessians, optimal_leaf_weight, regularized_split_gain
from hgb_core.feature_engineering import build_causal_features, chronological_time_split
from hgb_core.forecaster import forecast_next_n_days
from hgb_core.metrics import compute_regression_metrics, compute_dq_delta_mae
from hgb_core.model_hgb import build_hgb_pipeline


def test_exact_math_formulas():
    g, h = compute_gradients_hessians(np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 25.0]), "squared_error")
    assert np.allclose(g, [2.0, -2.0, -5.0]) and np.allclose(h, [1.0, 1.0, 1.0])
    assert abs(optimal_leaf_weight(-5.0, 3.0, 2.0) - 1.0) < 1e-9
    assert regularized_split_gain(-4.0, 2.0, 4.0, 2.0, 0.0) > 0.0

    rng = np.random.default_rng(42)
    X = rng.normal(size=(180, 4))
    X[rng.random(X.shape) < 0.05] = np.nan
    y = 3.0 * np.nan_to_num(X[:, 0]) + 2.0 * (np.nan_to_num(X[:, 1]) > 0) + rng.normal(scale=0.2, size=180)
    m = ExactHistogramGBDT(n_estimators=20, learning_rate=0.1, max_leaf_nodes=8, min_samples_leaf=10)
    m.fit(X, y)
    assert np.mean(np.abs(y - m.predict(X))) < 0.6


def test_pipeline_causal_and_nday_forecast():
    rng = np.random.default_rng(42)
    weeks = pd.date_range("2024-01-01", periods=50, freq="W-MON")
    rows = []
    for s in range(1, 6):
        for w_i, wk in enumerate(weeks):
            rows.append({"SKU_ID": f"SKU{s:04d}", "week_start": wk, "qty": float(rng.poisson(8.0)),
                         "avg_price": 1000.0, "discount_rate": 0.05, "promo_flag": int(w_i % 5 == 0),
                         "category": "Laptops", "subcategory": "Gaming", "brand": "ASUS", "active_age": float(w_i)})
    panel = pd.DataFrame(rows)
    df_feat, all_features, cat_features = build_causal_features(panel)
    for req in ["lag_1", "lag_2", "lag_4", "lag_8", "rolling_mean_4", "rolling_mean_8", "rolling_std_4", "month", "week_of_year", "active_age"]:
        assert req in all_features

    train_df, val_df, test_df, split_meta = chronological_time_split(df_feat)
    num_features = [c for c in all_features if c not in cat_features]
    model = build_hgb_pipeline(num_features, cat_features, {"max_iter": 40, "learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 10}, "log1p", 42)
    model.fit(train_df[all_features], train_df["Actual_Quantity"].values)

    m_zero = compute_regression_metrics(np.array([0.0, 0.0]), np.array([-1.0, 2.0]))
    assert m_zero["WAPE_pct"] == "N/A" and m_zero["N_clipped_negative"] == 1
    assert compute_dq_delta_mae(2.65, 2.40) == 0.25

    wk_fc, sum_fc, day_fc = forecast_next_n_days(model, panel, all_features, forecast_days=30, dataset_version="D0")
    assert len(wk_fc) == 5 * 5 and (sum_fc["Forecast_Days"] == 30).all() and len(day_fc) == 5 * 30


if __name__ == "__main__":
    test_exact_math_formulas()
    test_pipeline_causal_and_nday_forecast()
    print("ALL UNIT TESTS PASSED!")
