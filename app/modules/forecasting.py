"""
Task 4 & Feature 4: Pre-trained Model Loader & Next-Week Forecasting
--------------------------------------------------------------------
STRICT RULE (Part 9):
  Models are loaded from pre-saved artifacts (`models/*.joblib` or `models/*.pkl`).
  NO training or hyperparameter tuning occurs during the Streamlit demo.

Supports:
  - Scikit-learn `.joblib` / `.pkl` models (Random Forest, HistGradientBoosting, Pipeline)
  - Saved Moving Average `.pkl` artifact (dict specification, zero pickle-path issues)
Applies non-negative clipping: y_hat = max(0, y_hat_raw) and enriches with
Stock Cover & Inventory Risk if InventoryBalances CSV is provided.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from modules.preprocessing import FEATURE_COLUMNS


def discover_and_load_models(models_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Scan `models/` for pre-trained `.joblib` and `.pkl` files and load them.
    Returns dict mapping display_name -> {"artifact": obj, "path": str, "type": str}.
    """
    loaded: Dict[str, Dict[str, Any]] = {}
    if not models_dir.exists():
        return loaded

    for p in sorted(models_dir.iterdir()):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in {".joblib", ".pkl"}:
            continue
        try:
            if suffix == ".joblib":
                import joblib
                obj = joblib.load(p)
            else:
                with open(p, "rb") as f:
                    obj = pickle.load(f)

            pretty = p.stem.replace("_", " ").title()
            model_type = "dict_ma" if isinstance(obj, dict) and obj.get("model_type") == "moving_average" else "sklearn"
            loaded[f"{pretty} ({p.name})"] = {
                "artifact": obj,
                "path": str(p),
                "type": model_type,
            }
        except Exception as exc:
            loaded[f"[Lỗi nạp] {p.name}"] = {
                "artifact": None,
                "path": str(p),
                "type": "error",
                "error": str(exc),
            }
    return loaded


def predict_next_week(
    model_entry: Dict[str, Any],
    infer_df: pd.DataFrame,
    inventory_df: Optional[pd.DataFrame] = None,
    model_label: str = "",
) -> pd.DataFrame:
    """
    Run inference on the pre-built 10-feature matrix using a pre-trained model.
    Enforces y_hat = max(0, y_hat_raw) and optionally merges InventoryBalances
    to calculate Stock Cover (weeks) and Inventory Risk for Export CSV.
    """
    obj = model_entry["artifact"]
    mtype = model_entry["type"]

    if obj is None:
        raise ValueError(f"Không thể nạp mô hình: {model_entry.get('error', 'Unknown error')}")

    if mtype == "dict_ma":
        w = int(obj.get("window", 4))
        fallback = float(obj.get("fallback_mean", 0.0))
        col = f"rolling_mean_{w}"
        if col in infer_df.columns:
            raw_pred = infer_df[col].fillna(fallback).to_numpy(dtype=float)
        else:
            raw_pred = infer_df["rolling_mean_4"].fillna(fallback).to_numpy(dtype=float)
    else:
        # Standard scikit-learn estimator / Pipeline expecting FEATURE_COLUMNS
        X = infer_df[FEATURE_COLUMNS].copy()
        raw_pred = np.asarray(obj.predict(X), dtype=float)

    # Non-negative clipping rule
    clipped_pred = np.maximum(0.0, raw_pred)

    out = pd.DataFrame(
        {
            "SKU_ID": infer_df["SKU_ID"].values,
            "Forecast_Week": pd.to_datetime(infer_df["Forecast_Week"]).dt.strftime("%Y-%m-%d").values,
            "Last_Week_Actual": np.round(infer_df["lag_1"].fillna(0.0).to_numpy(dtype=float), 2),
            "Rolling_Mean_4W": np.round(infer_df["rolling_mean_4"].to_numpy(dtype=float), 2),
            "Predicted_Quantity": np.round(clipped_pred, 2),
            "Model": model_label or model_entry["path"],
        }
    )

    # Optional Inventory enrichment for Stock Cover & Export CSV
    if inventory_df is not None and not inventory_df.empty and "SKU_ID" in inventory_df.columns:
        agg_cols = {}
        for c in ["AvailableQuantity", "ReorderPointQty", "SafetyStockQty"]:
            if c in inventory_df.columns:
                agg_cols[c] = (c, "sum")
        if agg_cols:
            inv_agg = inventory_df.groupby("SKU_ID", as_index=False).agg(**agg_cols)
            out = out.merge(inv_agg, on="SKU_ID", how="left")
            if "AvailableQuantity" in out.columns:
                out["Stock_Cover_Weeks"] = np.where(
                    out["Predicted_Quantity"] > 0.01,
                    np.round(out["AvailableQuantity"] / out["Predicted_Quantity"], 2),
                    np.nan,
                )
                out["Inventory_Risk"] = np.where(
                    out["Stock_Cover_Weeks"].isna(),
                    "No Active Demand",
                    np.where(
                        out["Stock_Cover_Weeks"] < 1.0,
                        "Critical (<1 tuần)",
                        np.where(
                            out["Stock_Cover_Weeks"] < 2.0,
                            "Warning (1–2 tuần)",
                            np.where(
                                out["Stock_Cover_Weeks"] > 8.0,
                                "Overstock (>8 tuần)",
                                "Healthy (2–8 tuần)",
                            ),
                        ),
                    ),
                )
    return out