"""
Task 3: Shared Preprocessing & Feature Engineering for Inference
----------------------------------------------------------------
Mirrors the exact preprocessing & 10-feature engineering rules used during training:
  1. Standardize OrderStatus (Complete->Completed, Canceled->Cancelled)
  2. Exclude Cancelled orders
  3. Filter out-of-scope dates (keep valid 2023-01-01..2025-12-31 when present)
  4. Quarantine Quantity <= 0 (never auto-fill negative to 0)
  5. Aggregate into SKU-Week panel (Monday-anchored W-SUN weeks)
  6. Construct the exact 10 leakage-free features for the next week (t+1):
     lag_1, lag_2, lag_4, lag_8, rolling_mean_4, rolling_mean_8, rolling_std_4,
     month, week_of_year, active_age
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
    "month",
    "week_of_year",
    "active_age",
]

STATUS_MAP = {
    "Complete": "Completed",
    "Completed": "Completed",
    "Canceled": "Cancelled",
    "Cancelled": "Cancelled",
}


def preprocess_uploaded_sales(
    df: pd.DataFrame,
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Clean uploaded sales CSV and build SKU-Week panel + next-week (t+1) feature matrix.
    Returns: (panel_df, inference_features_df, audit_dict)
    """
    work = df.copy()
    audit: Dict[str, int] = {"raw_rows": len(work)}

    # 1. Deduplicate exact rows
    before_dedup = len(work)
    work = work.drop_duplicates()
    audit["removed_exact_duplicates"] = before_dedup - len(work)

    # 2. Standardize OrderStatus & exclude Cancelled
    if "OrderStatus" in work.columns:
        work["OrderStatus"] = work["OrderStatus"].map(STATUS_MAP).fillna(work["OrderStatus"])
        n_cancel = int((work["OrderStatus"] == "Cancelled").sum())
        audit["excluded_cancelled_rows"] = n_cancel
        work = work[work["OrderStatus"] != "Cancelled"].copy()
    else:
        audit["excluded_cancelled_rows"] = 0

    # 3. Parse dates & exclude out-of-scope timeliness outliers (matching training scope)
    date_col = "OrderDate" if "OrderDate" in work.columns else "WeekStart"
    work["OrderDate_dt"] = pd.to_datetime(work[date_col], errors="coerce")
    valid_dt = work["OrderDate_dt"].notna()
    in_scope = (work["OrderDate_dt"] >= pd.Timestamp(start_date)) & (
        work["OrderDate_dt"] <= pd.Timestamp(end_date)
    )
    if int(in_scope.sum()) > 0:
        valid_dt = valid_dt & in_scope
    audit["excluded_invalid_or_oob_dates"] = int((~valid_dt).sum())
    work = work[valid_dt].copy()

    # 4. Quarantine Quantity <= 0 (preserve rule: do not convert negative to 0)
    work["Quantity"] = pd.to_numeric(work["Quantity"], errors="coerce")
    bad_qty = work["Quantity"].isna() | (work["Quantity"] <= 0)
    audit["quarantined_nonpositive_qty"] = int(bad_qty.sum())
    work = work[~bad_qty].copy()

    audit["clean_valid_rows"] = len(work)
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), audit

    # 5. Aggregate to SKU-Week panel
    work["WeekStart"] = work["OrderDate_dt"].dt.to_period("W-SUN").dt.start_time
    weekly = (
        work.groupby(["SKU_ID", "WeekStart"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Quantity": "Observed_Quantity"})
    )

    all_weeks = pd.date_range(weekly["WeekStart"].min(), weekly["WeekStart"].max(), freq="W-MON")
    covered_weeks = set(weekly["WeekStart"].unique())

    sku_first = weekly.groupby("SKU_ID", as_index=False)["WeekStart"].min().rename(columns={"WeekStart": "FirstWeek"})
    grid = (
        sku_first.assign(_k=1)
        .merge(pd.DataFrame({"WeekStart": all_weeks, "_k": 1}), on="_k")
        .drop(columns=["_k"])
    )
    grid = grid[grid["WeekStart"] >= grid["FirstWeek"]].copy()

    panel = grid.merge(weekly, on=["SKU_ID", "WeekStart"], how="left")
    panel["Label_Status"] = np.where(
        panel["Observed_Quantity"].notna(),
        "VALID",
        np.where(panel["WeekStart"].isin(covered_weeks), "VALID_ZERO", "UNVERIFIED"),
    )
    panel["Quantity"] = np.where(
        panel["Label_Status"] == "VALID",
        panel["Observed_Quantity"],
        np.where(panel["Label_Status"] == "VALID_ZERO", 0.0, np.nan),
    )
    panel = panel.sort_values(["SKU_ID", "WeekStart"]).reset_index(drop=True)
    audit["panel_sku_weeks"] = len(panel)
    audit["unique_skus"] = int(panel["SKU_ID"].nunique())

    # 6. Build exact 10 inference features for next unseen week (t+1)
    last_week = panel["WeekStart"].max()
    forecast_week = last_week + pd.Timedelta(days=7)

    rows = []
    for sku, g in panel.groupby("SKU_ID"):
        g = g.sort_values("WeekStart")
        q = g["Quantity"].fillna(0.0).to_numpy(dtype=float)
        n = len(q)
        first_w = g["FirstWeek"].iloc[0]
        rows.append(
            {
                "SKU_ID": sku,
                "Forecast_Week": forecast_week,
                "lag_1": q[-1] if n >= 1 else np.nan,
                "lag_2": q[-2] if n >= 2 else np.nan,
                "lag_4": q[-4] if n >= 4 else np.nan,
                "lag_8": q[-8] if n >= 8 else np.nan,
                "rolling_mean_4": float(np.mean(q[-4:])) if n >= 1 else 0.0,
                "rolling_mean_8": float(np.mean(q[-8:])) if n >= 1 else 0.0,
                "rolling_std_4": float(np.std(q[-4:], ddof=1)) if n >= 2 else 0.0,
                "month": int(forecast_week.month),
                "week_of_year": int(forecast_week.isocalendar().week),
                "active_age": max(0, int((forecast_week - first_w).days // 7)),
            }
        )
    infer_df = pd.DataFrame(rows)
    return panel, infer_df, audit