"""Recursive multi-step forecasting for arbitrary n-day horizon."""

from __future__ import annotations
import math
from typing import Any, List, Tuple
import numpy as np
import pandas as pd


def forecast_next_n_days(
    model: Any,
    panel: pd.DataFrame,
    all_features: List[str],
    lags: List[int] = (1, 2, 4, 8),
    rolling_windows: List[int] = (4, 8),
    forecast_days: int = 28,
    dataset_version: str = "D0",
    model_name: str = "HistGradientBoosting",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if forecast_days <= 0:
        raise ValueError(f"forecast_days must be >= 1, got {forecast_days}")

    n_steps = math.ceil(forecast_days / 7)
    remainder_days = forecast_days % 7
    last_week_weight = 1.0 if remainder_days == 0 else (remainder_days / 7.0)

    df = panel.sort_values(["SKU_ID", "week_start"]).copy()
    max_hist = max(max(lags), max(rolling_windows), 4)

    sku_histories, sku_last_row = {}, {}
    for sku, sub in df.groupby("SKU_ID", sort=False):
        valid_q = sub["qty"].dropna().astype(float).values
        if len(valid_q) == 0:
            continue
        sku_histories[sku] = list(valid_q[-max_hist:])
        sku_last_row[sku] = sub.iloc[-1].to_dict()

    skus = list(sku_histories.keys())
    weekly_rows = []

    for step in range(1, n_steps + 1):
        step_feature_rows, step_meta = [], []
        for sku in skus:
            hist, last = sku_histories[sku], sku_last_row[sku]
            target_week = pd.Timestamp(last["week_start"]) + pd.Timedelta(days=7 * step)

            row = {}
            for lag in lags:
                row[f"lag_{lag}"] = hist[-lag] if len(hist) >= lag else np.nan
            for w in rolling_windows:
                win = hist[-w:]
                row[f"rolling_mean_{w}"] = float(np.mean(win))
                if w == 4:
                    row["rolling_std_4"] = float(np.std(win, ddof=1)) if len(win) >= 2 else 0.0
            row["trend_slope_4"] = ((-1.5 * hist[-4] - 0.5 * hist[-3] + 0.5 * hist[-2] + 1.5 * hist[-1]) / 5.0) if len(hist) >= 4 else np.nan

            iso_w = int(target_week.isocalendar().week)
            row["month"] = int(target_week.month)
            row["week_of_year"] = iso_w
            row["woy_sin"] = float(np.sin(2.0 * np.pi * iso_w / 52.0))
            row["woy_cos"] = float(np.cos(2.0 * np.pi * iso_w / 52.0))

            base_age = last.get("active_age", np.nan)
            row["active_age"] = (float(len(hist)) if pd.isna(base_age) else float(base_age)) + step

            if "avg_price_lag1" in all_features:
                row["avg_price_lag1"] = last.get("avg_price", np.nan)
            if "discount_rate_lag1" in all_features:
                row["discount_rate_lag1"] = last.get("discount_rate", 0.0)
            if "promo_flag" in all_features:
                row["promo_flag"] = int(last.get("promo_flag", 0)) if pd.notna(last.get("promo_flag")) else 0
            for cat_col in ["category", "subcategory", "brand"]:
                if cat_col in all_features:
                    row[cat_col] = str(last.get(cat_col, "Unknown"))

            step_feature_rows.append(row)
            step_meta.append((sku, target_week, last.get("category", "Unknown")))

        preds = np.maximum(model.predict(pd.DataFrame(step_feature_rows)[all_features]), 0.0)
        weight = last_week_weight if step == n_steps else 1.0
        days_in_step = remainder_days if (step == n_steps and remainder_days != 0) else 7

        for (sku, target_week, cat_name), pred_val in zip(step_meta, preds):
            pred_f = float(pred_val)
            sku_histories[sku].append(pred_f)
            weekly_rows.append(
                {
                    "SKU_ID": sku,
                    "Forecast_Week": target_week.strftime("%Y-%m-%d"),
                    "Horizon_Week_Step": step,
                    "Days_Covered": days_in_step,
                    "Predicted_Quantity": round(pred_f, 4),
                    "Prorated_Quantity": round(pred_f * weight, 4),
                    "Category": cat_name,
                    "Model": model_name,
                    "Dataset_Version": dataset_version,
                }
            )

    weekly_fc = pd.DataFrame(weekly_rows)
    summary_fc = (
        weekly_fc.groupby(["SKU_ID", "Category", "Model", "Dataset_Version"], as_index=False)
        .agg(
            Forecast_Days=("Days_Covered", "sum"),
            Forecast_Weeks=("Horizon_Week_Step", "max"),
            Total_Predicted_Quantity=("Prorated_Quantity", "sum"),
            Avg_Weekly_Predicted_Quantity=("Predicted_Quantity", "mean"),
        )
        .sort_values("Total_Predicted_Quantity", ascending=False)
        .reset_index(drop=True)
    )
    summary_fc["Total_Predicted_Quantity"] = summary_fc["Total_Predicted_Quantity"].round(2)
    summary_fc["Avg_Weekly_Predicted_Quantity"] = summary_fc["Avg_Weekly_Predicted_Quantity"].round(2)

    daily_rows = []
    for _, wrow in weekly_fc.iterrows():
        wk_start = pd.Timestamp(wrow["Forecast_Week"])
        daily_rate = wrow["Predicted_Quantity"] / 7.0
        for d in range(int(wrow["Days_Covered"])):
            daily_rows.append(
                {
                    "SKU_ID": wrow["SKU_ID"],
                    "Forecast_Date": (wk_start + pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
                    "Predicted_Daily_Quantity": round(daily_rate, 4),
                    "Model": model_name,
                    "Dataset_Version": dataset_version,
                }
            )
    return weekly_fc, summary_fc, pd.DataFrame(daily_rows)
