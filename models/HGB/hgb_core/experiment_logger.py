"""Experiment management & tracking logger (Spec Section 13)."""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json
import pandas as pd


def append_experiment_log(
    log_csv_path: Path,
    experiment_id: str,
    dataset_version: str,
    model_name: str,
    feature_version: str,
    split_meta: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    seed: int,
    test_metrics: Dict[str, Any],
    training_time_sec: float,
    delta_mae_vs_d0: Optional[float] = None,
) -> pd.DataFrame:
    record = {
        "Experiment_ID": experiment_id,
        "Dataset_Version": dataset_version,
        "Model_Name": model_name,
        "Feature_Version": feature_version,
        "Train_Period": split_meta["train_period"],
        "Validation_Period": split_meta["val_period"],
        "Test_Period": split_meta["test_period"],
        "Hyperparameters": json.dumps(hyperparameters, ensure_ascii=False),
        "Random_Seed": seed,
        "MAE": test_metrics["MAE"],
        "RMSE": test_metrics["RMSE"],
        "WAPE_pct": test_metrics["WAPE_pct"],
        "Delta_MAE_vs_D0": delta_mae_vs_d0 if delta_mae_vs_d0 is not None else "",
        "N_clipped_negative": test_metrics["N_clipped_negative"],
        "Training_Time_sec": round(training_time_sec, 3),
        "Run_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    new_df = pd.DataFrame([record])
    combined = pd.concat([pd.read_csv(log_csv_path), new_df], ignore_index=True) if log_csv_path.exists() else new_df
    combined.to_csv(log_csv_path, index=False)
    return combined
