"""One row per SKU/origin: information through origin, label at origin + 7d."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

NUMERIC_FEATURES = [
    "lag_1", "lag_2", "lag_4", "lag_8", "rolling_mean_4", "rolling_std_4",
    "rolling_mean_8", "rolling_std_8", "zero_fraction_8", "age_weeks",
    "target_week_sin", "target_week_cos", "target_month",
]
FEATURE_COLUMNS = ["SKU_ID", *NUMERIC_FEATURES]


def build_features(panel: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Target-indexed lag_1 = Q_origin; no future label participates in X.

    All SKU calendars are contiguous from their first observed sale. NaNs remain
    unknown, never backfilled. Eligibility requires uninterrupted complete history.
    """
    frames = []
    for sku, history in panel.groupby("SKU_ID", sort=True):
        history = history.sort_values("week_start").reset_index(drop=True)
        if history.week_start.diff().dropna().ne(pd.Timedelta(days=7)).any():
            raise ValueError(f"Non-contiguous weekly panel for {sku}")
        quantity = history.quantity.astype(float)
        row = pd.DataFrame({"SKU_ID": sku, "origin_week": history.week_start,
                            "target_week": history.week_start + pd.Timedelta(days=7)})
        for lag in (1, 2, 4, 8):
            row[f"lag_{lag}"] = quantity.shift(lag - 1)
        for window in (4, 8):
            row[f"rolling_mean_{window}"] = quantity.rolling(window, min_periods=window).mean()
            row[f"rolling_std_{window}"] = quantity.rolling(window, min_periods=window).std(ddof=0)
        row["zero_fraction_8"] = quantity.eq(0).astype(float).where(quantity.notna()).rolling(8, min_periods=8).mean()
        row["age_weeks"] = (history.week_start - history.first_seen_week).dt.days / 7 + 1
        iso_week = row.target_week.dt.isocalendar().week.astype(float)
        row["target_week_sin"] = np.sin(2 * np.pi * iso_week / 52.1775)
        row["target_week_cos"] = np.cos(2 * np.pi * iso_week / 52.1775)
        row["target_month"] = row.target_week.dt.month.astype(float)
        row["history_known_last_4"] = quantity.rolling(4, min_periods=1).count().astype(int)
        row["history_known_window"] = quantity.rolling(config.min_history_weeks, min_periods=1).count().astype(int)
        row["rf_eligible"] = row.history_known_window.ge(config.min_history_weeks)
        row["baseline_naive"] = quantity
        row["baseline_ma4"] = quantity.rolling(4, min_periods=1).mean()
        # Label is used ONLY for fitting/scoring. It is deliberately absent from FEATURE_COLUMNS.
        row["actual_quantity"] = quantity.shift(-1)
        frames.append(row)
    if not frames:
        raise ValueError("Empty SKU panel")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["SKU_ID", "origin_week"]).any():
        raise ValueError("Duplicate SKU/origin feature key")
    return result.sort_values(["target_week", "SKU_ID"]).reset_index(drop=True)
