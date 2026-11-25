"""Headless-safe diagnostic & forecast plots for HistGradientBoosting."""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted_aggregate(pred_df: pd.DataFrame, save_path: Path, dataset_version: str = "D0") -> None:
    agg = pred_df.groupby("Forecast_Week", as_index=False)[["Actual_Quantity", "Predicted_Quantity"]].sum().sort_values("Forecast_Week")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(agg["Forecast_Week"], agg["Actual_Quantity"], marker="o", linewidth=2, label="Actual Quantity", color="#1f77b4")
    ax.plot(agg["Forecast_Week"], agg["Predicted_Quantity"], marker="s", linestyle="--", linewidth=2, label=f"HGB Predicted ({dataset_version})", color="#d62728")
    ax.set_title(f"HistGradientBoosting ({dataset_version}): Aggregate Weekly Actual vs Predicted (Test Set)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Forecast_Week")
    ax.set_ylabel("Total Units Sold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(fi_df: pd.DataFrame, save_path: Path, top_n: int = 15) -> None:
    if fi_df.empty:
        return
    sub = fi_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(sub["feature"], sub["importance_mean_mae"], xerr=sub["importance_std"], color="#2ca02c", alpha=0.85)
    ax.set_title(f"Top {len(sub)} Features by Permutation Importance (MAE Increase)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Mean MAE Increase When Shuffled")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history_and_forecast(panel: pd.DataFrame, weekly_fc: pd.DataFrame, save_path: Path, forecast_days: int) -> None:
    hist_agg = panel.groupby("week_start", as_index=False)["qty"].sum().sort_values("week_start")
    fut_agg = weekly_fc.groupby("Forecast_Week", as_index=False)["Predicted_Quantity"].sum().sort_values("Forecast_Week")
    fut_agg["Forecast_Week"] = pd.to_datetime(fut_agg["Forecast_Week"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist_agg["week_start"], hist_agg["qty"], linewidth=1.8, label="Historical Weekly Sales", color="#1f77b4")
    ax.plot(fut_agg["Forecast_Week"], fut_agg["Predicted_Quantity"], marker="o", linewidth=2.2, linestyle="--", label=f"HGB Forecast ({forecast_days} Days)", color="#ff7f0e")
    ax.axvline(hist_agg["week_start"].max(), color="gray", linestyle=":", linewidth=1.2)
    ax.set_title(f"Historical Weekly Sales & {forecast_days}-Day HistGradientBoosting Forecast", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Units")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
