"""Evaluation metrics (Spec 9.2) and Data Quality impact Delta MAE (Spec 9.4)."""

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred_raw: np.ndarray) -> Dict[str, Any]:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_raw = np.asarray(y_pred_raw, dtype=np.float64)
    n_clipped = int((y_raw < 0).sum())
    y_p = np.maximum(y_raw, 0.0)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 else 0.0

    denom = float(np.sum(np.abs(y_t)))
    wape_pct: Any = round(float(np.sum(np.abs(y_t - y_p)) / denom * 100.0), 4) if denom > 0 else "N/A"
    pos_mask = y_t > 0
    mape_pos_pct: Any = round(float(np.mean(np.abs((y_t[pos_mask] - y_p[pos_mask]) / y_t[pos_mask])) * 100.0), 4) if pos_mask.any() else "N/A"
    smape_denom = np.abs(y_t) + np.abs(y_p)
    smape_pct = round(float(np.mean(np.where(smape_denom > 0, 2.0 * np.abs(y_t - y_p) / smape_denom, 0.0)) * 100.0), 4)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "WAPE_pct": wape_pct,
        "MAPE_pos_pct": mape_pos_pct,
        "sMAPE_pct": smape_pct,
        "R2": round(r2, 4),
        "Bias": round(float(np.mean(y_p - y_t)), 4),
        "N_samples": int(len(y_t)),
        "N_clipped_negative": n_clipped,
    }


def compute_dq_delta_mae(mae_dk: float, mae_d0: Optional[float]) -> Optional[float]:
    return None if mae_d0 is None else round(float(mae_dk) - float(mae_d0), 4)


def error_breakdown_by_group(pred_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in pred_df.columns:
        return pd.DataFrame()
    rows = []
    for g_val, sub in pred_df.groupby(group_col, observed=False):
        m = compute_regression_metrics(sub["Actual_Quantity"].values, sub["Predicted_Quantity"].values)
        m[group_col] = str(g_val)
        rows.append(m)
    out = pd.DataFrame(rows)
    return out[[group_col, "N_samples", "MAE", "RMSE", "WAPE_pct", "R2", "Bias"]].sort_values("MAE", ascending=False).reset_index(drop=True)
