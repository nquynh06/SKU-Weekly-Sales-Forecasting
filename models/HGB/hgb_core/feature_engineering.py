"""Strictly causal feature engineering (Spec 6.2) and chronological panel split (Spec 9.1)."""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def _rolling_slope_4_shifted(series: pd.Series) -> pd.Series:
    y0, y1, y2, y3 = series.shift(4), series.shift(3), series.shift(2), series.shift(1)
    return (-1.5 * y0 - 0.5 * y1 + 0.5 * y2 + 1.5 * y3) / 5.0


def build_causal_features(
    panel: pd.DataFrame,
    lags: List[int] = (1, 2, 4, 8),
    rolling_windows: List[int] = (4, 8),
    categorical_cols: List[str] = ("category", "subcategory", "brand"),
    optional_numeric_cols: List[str] = ("avg_price", "discount_rate", "promo_flag"),
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = panel.sort_values(["SKU_ID", "week_start"]).copy()
    df["Forecast_Week"] = df["week_start"]
    df["Actual_Quantity"] = df["qty"]
    grp = df.groupby("SKU_ID", sort=False)

    for lag in lags:
        df[f"lag_{lag}"] = grp["Actual_Quantity"].shift(lag)

    df["_shifted_qty"] = grp["Actual_Quantity"].shift(1)
    grp_s = df.groupby("SKU_ID", sort=False)
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = grp_s["_shifted_qty"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        if w == 4:
            df[f"rolling_std_{w}"] = grp_s["_shifted_qty"].transform(lambda s: s.rolling(w, min_periods=2).std())
    df = df.drop(columns=["_shifted_qty"])

    df["trend_slope_4"] = grp["Actual_Quantity"].transform(_rolling_slope_4_shifted)

    iso_week = df["Forecast_Week"].dt.isocalendar().week.astype(int)
    df["month"] = df["Forecast_Week"].dt.month.astype(int)
    df["week_of_year"] = iso_week
    df["woy_sin"] = np.sin(2.0 * np.pi * iso_week / 52.0)
    df["woy_cos"] = np.cos(2.0 * np.pi * iso_week / 52.0)

    if "active_age" not in df.columns:
        first_wk = grp["Forecast_Week"].transform("min")
        df["active_age"] = ((df["Forecast_Week"] - first_wk).dt.days // 7).astype(float)
    else:
        df["active_age"] = pd.to_numeric(df["active_age"], errors="coerce")

    numeric_features = [f"lag_{lag}" for lag in lags]
    for w in rolling_windows:
        numeric_features.append(f"rolling_mean_{w}")
        if w == 4:
            numeric_features.append(f"rolling_std_{w}")
    numeric_features += ["trend_slope_4", "month", "week_of_year", "woy_sin", "woy_cos", "active_age"]

    for col in optional_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in ("avg_price", "discount_rate"):
                lagged = f"{col}_lag1"
                df[lagged] = grp[col].shift(1)
                numeric_features.append(lagged)
            else:
                numeric_features.append(col)

    present_cats: List[str] = []
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
            present_cats.append(col)

    all_features = numeric_features + present_cats
    assert "Actual_Quantity" not in all_features and "qty" not in all_features and "Forecast_Week" not in all_features
    return df, all_features, present_cats


def chronological_time_split(
    df_feat: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    gap_weeks: int = 0,
    min_history_weeks: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    model_df = df_feat.dropna(subset=["Actual_Quantity"]).copy()
    if "lag_1" in model_df.columns:
        model_df = model_df.dropna(subset=["lag_1"]).copy()

    if min_history_weeks > 1:
        sku_counts = model_df["SKU_ID"].value_counts()
        valid_skus = sku_counts[sku_counts >= min_history_weeks].index
        if len(valid_skus) > 0:
            model_df = model_df[model_df["SKU_ID"].isin(valid_skus)].copy()

    unique_weeks = np.sort(model_df["Forecast_Week"].unique())
    n_weeks = len(unique_weeks)
    if n_weeks < 6:
        raise ValueError(f"Need at least 6 unique weeks for Train/Val/Test split, found {n_weeks}.")

    n_train = max(2, int(round(n_weeks * train_ratio)))
    n_val = max(1, int(round(n_weeks * val_ratio)))
    while n_train + n_val + 2 * gap_weeks >= n_weeks and n_train > 2:
        n_train -= 1

    train_weeks = unique_weeks[:n_train]
    val_start = n_train + gap_weeks
    val_end = min(val_start + n_val, n_weeks - 1 - gap_weeks)
    val_weeks = unique_weeks[val_start:val_end]
    test_weeks = unique_weeks[val_end + gap_weeks :]

    if len(val_weeks) == 0 or len(test_weeks) == 0:
        n_train = max(2, int(n_weeks * 0.70))
        n_val = max(1, int(n_weeks * 0.15))
        train_weeks = unique_weeks[:n_train]
        val_weeks = unique_weeks[n_train : n_train + n_val]
        test_weeks = unique_weeks[n_train + n_val :]

    train_df = model_df[model_df["Forecast_Week"].isin(train_weeks)].copy()
    val_df = model_df[model_df["Forecast_Week"].isin(val_weeks)].copy()
    test_df = model_df[model_df["Forecast_Week"].isin(test_weeks)].copy()

    assert not model_df.duplicated(subset=["SKU_ID", "Forecast_Week"]).any()
    assert train_df["Forecast_Week"].max() < val_df["Forecast_Week"].min()
    assert val_df["Forecast_Week"].max() < test_df["Forecast_Week"].min()

    split_meta = {
        "n_total_weeks": int(n_weeks),
        "train_weeks": int(len(train_weeks)),
        "val_weeks": int(len(val_weeks)),
        "test_weeks": int(len(test_weeks)),
        "gap_weeks": int(gap_weeks),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_period": f"{pd.Timestamp(train_weeks[0]).date()} to {pd.Timestamp(train_weeks[-1]).date()}",
        "val_period": f"{pd.Timestamp(val_weeks[0]).date()} to {pd.Timestamp(val_weeks[-1]).date()}",
        "test_period": f"{pd.Timestamp(test_weeks[0]).date()} to {pd.Timestamp(test_weeks[-1]).date()}",
    }
    return train_df, val_df, test_df, split_meta
