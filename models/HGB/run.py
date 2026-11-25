#!/usr/bin/env python3
"""Official CLI for HistGradientBoosting SKU-Weekly Sales Forecasting (models/HGB/run.py).

Matches the exact CLI contract of models/random forest/run.py in README.md Section 9E:

  1. Validate raw data contract (27 CSVs in data/raw/):
     python models/HGB/run.py validate --config models/HGB/configs/hgb.json --data-dir data/raw --output-dir reports/hgb/validate_001

  2. Run full HGB training, tuning, evaluation & N-day forecasting directly from data/raw/:
     python models/HGB/run.py run --config models/HGB/configs/hgb.json --data-dir data/raw --output-dir reports/hgb/run_001

  3. Or run on a preprocessed CSV (e.g., sku_week_ground_truth_provisional.csv or D0..D4):
     python models/HGB/run.py run --data-file data/processed/sku_week_ground_truth_provisional.csv --dataset-version D0 --forecast-days 28

  4. Or simply drop CSV(s) into models/HGB/input_data/ and run:
     python models/HGB/run.py
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hgb_core.data_loader import (
    detect_dataset_version_from_name,
    is_raw_data_dir,
    load_data_source,
    validate_raw_contract,
)
from hgb_core.feature_engineering import build_causal_features, chronological_time_split
from hgb_core.model_hgb import build_hgb_pipeline, tune_and_train_hgb, compute_feature_importance
from hgb_core.metrics import compute_regression_metrics, compute_dq_delta_mae, error_breakdown_by_group
from hgb_core.experiment_logger import append_experiment_log
from hgb_core.forecaster import forecast_next_n_days
from hgb_core.visualizer import (
    plot_actual_vs_predicted_aggregate,
    plot_feature_importance,
    plot_history_and_forecast,
)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=str(SCRIPT_DIR / "configs" / "hgb.json"), help="Path to hgb.json config")
    p.add_argument("--data-dir", type=str, default=None, help="Directory containing raw 27 CSVs (e.g. data/raw)")
    p.add_argument("--data-file", type=str, default=None, help="Direct path to a preprocessed CSV file (D0..D4 or Ground Truth)")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory (e.g. reports/hgb/run_001)")
    p.add_argument("--start-week", type=str, default=None, help="Synchronized start week (default 2023-07-03)")
    p.add_argument("--end-week", type=str, default=None, help="Synchronized end week (default 2025-12-22)")


def parse_args() -> argparse.Namespace:
    # Allow running `python models/HGB/run.py` directly without subcommand (defaults to `run`)
    argv = sys.argv[1:]
    if not argv or (argv[0] not in ("validate", "run", "-h", "--help")):
        argv = ["run"] + argv

    parser = argparse.ArgumentParser(description="HistGradientBoosting SKU-Weekly Sales Forecasting (DS317 Group 5)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Validate raw CSV contract & quality gates before training")
    _add_common_args(p_val)

    p_run = sub.add_parser("run", help="Run full HGB pipeline: load, feature eng, split, tune, evaluate & forecast")
    _add_common_args(p_run)
    p_run.add_argument("--dataset-version", type=str, default=None, help="Dataset version tag: D0, D1, D2, D3, D4")
    p_run.add_argument("--experiment-id", type=str, default=None, help="Experiment_ID override")
    p_run.add_argument("--d0-metrics", type=str, default=None, help="Path to D0 metrics JSON to compute Delta_MAE")
    p_run.add_argument("--history-start", type=str, default=None, help="Historical window start date (YYYY-MM-DD)")
    p_run.add_argument("--history-end", type=str, default=None, help="Historical window end date (YYYY-MM-DD)")
    p_run.add_argument("--history-weeks", type=int, default=None, help="Keep only last N weeks of history")
    p_run.add_argument("--demand-scale", type=float, default=None, help="Scale historical demand (default 1.0)")
    p_run.add_argument("--forecast-days", type=int, default=None, help="Number of future days to forecast (e.g. 7, 14, 28, 30)")
    p_run.add_argument("--skus", type=str, nargs="*", default=None, help="Optional subset of SKU_IDs")
    p_run.add_argument("--gap-weeks", type=int, default=None, help="Optional temporal gap weeks between splits")
    p_run.add_argument("--no-tune", action="store_true", help="Skip hyperparameter grid search (fast mode)")
    p_run.add_argument("--run-all-in-input", action="store_true", help="Batch-run all CSVs dropped into models/HGB/input_data/")

    return parser.parse_args(argv)


def resolve_input_source(args: argparse.Namespace) -> Path:
    if args.data_file:
        p = Path(args.data_file)
        if not p.exists():
            p = PROJECT_ROOT / args.data_file
        return p
    if args.data_dir:
        p = Path(args.data_dir)
        if not p.exists():
            p = PROJECT_ROOT / args.data_dir
        return p
    # Auto-discover: 1) models/HGB/input_data/, 2) PROJECT_ROOT/data/raw/
    in_dir = SCRIPT_DIR / "input_data"
    if is_raw_data_dir(in_dir):
        return in_dir
    dropped = sorted(in_dir.glob("*.csv"))
    if dropped:
        return dropped[0]
    default_raw = PROJECT_ROOT / "data" / "raw"
    if is_raw_data_dir(default_raw):
        return default_raw
    raise FileNotFoundError(
        "No input data found! Either pass `--data-dir data/raw` (with the 27 Kaggle CSVs), "
        "pass `--data-file <path.csv>`, or drop a preprocessed CSV into `models/HGB/input_data/`."
    )


def cmd_validate(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else (SCRIPT_DIR / "outputs" / "validate")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = resolve_input_source(args)
    start_w = args.start_week or cfg["data"].get("start_week", "2023-07-03")
    end_w = args.end_week or cfg["data"].get("end_week", "2025-12-22")

    print("=" * 78)
    print(f"  HGB CONTRACT VALIDATION | Source: {src}")
    print("=" * 78)
    if src.is_dir():
        rep = validate_raw_contract(src, start_week=start_w, end_week=end_w)
    else:
        panel, meta = load_data_source(src, start_week=start_w, end_week=end_w)
        rep = {"status": "PASS", "source_file": str(src.resolve()), "meta": meta, "checks": [{"check": "Preprocessed CSV readable", "value": len(panel), "status": "PASS"}]}

    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    if rep.get("checks"):
        pd.DataFrame(rep["checks"]).to_csv(out_dir / "validation_checks.csv", index=False)

    for c in rep.get("checks", []):
        print(f"  [{c['status']}] {c['check']}: {c['value']}")
    print("-" * 78)
    print(f"  OVERALL VALIDATION STATUS: {rep['status']} | Saved to: {out_dir.resolve()}")
    print("=" * 78)
    if rep["status"] != "PASS":
        sys.exit(1)


def run_single(
    src: Path,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    out_dir: Path,
    dataset_version: str,
    mae_d0_ref: Optional[float] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seed = int(cfg["experiment"].get("seed", 42))
    np.random.seed(seed)

    model_name = cfg["experiment"].get("model_name", "HistGradientBoosting")
    feature_version = cfg["experiment"].get("feature_version", "v1.0")
    experiment_id = args.experiment_id or f"EXP_HGB_{dataset_version}_{time.strftime('%Y%m%d_%H%M%S')}"

    start_w = args.start_week or cfg["data"].get("start_week", "2023-07-03")
    end_w = args.end_week or cfg["data"].get("end_week", "2025-12-22")
    history_start = args.history_start if args.history_start is not None else cfg["data"].get("history_start")
    history_end = args.history_end if args.history_end is not None else cfg["data"].get("history_end")
    history_weeks = args.history_weeks if args.history_weeks is not None else cfg["data"].get("history_weeks")
    demand_scale = args.demand_scale if args.demand_scale is not None else float(cfg["forecast"].get("demand_scale", 1.0))
    forecast_days = args.forecast_days if args.forecast_days is not None else int(cfg["forecast"].get("forecast_days", 28))
    selected_skus = args.skus if args.skus else cfg["forecast"].get("selected_skus", [])
    gap_weeks = args.gap_weeks if args.gap_weeks is not None else int(cfg["split"].get("gap_weeks", 0))
    tune = (not args.no_tune) and bool(cfg["model"].get("tune_hyperparameters", True))

    print("=" * 78)
    print(f"  HGB SKU-WEEKLY PIPELINE | Version: {dataset_version} | Source: {src.name}")
    print("=" * 78)

    panel, data_meta = load_data_source(
        source_path=src,
        start_week=start_w,
        end_week=end_w,
        sku_col=cfg["data"]["sku_col"],
        date_col=cfg["data"]["date_col"],
        qty_col=cfg["data"]["qty_col"],
        label_status_col=cfg["data"].get("label_status_col", "label_status"),
        stockout_flag_col=cfg["data"].get("stockout_flag_col", "stockout_flag"),
        fill_missing_weeks_as_zero_for_csv=bool(cfg["data"].get("fill_missing_weeks_as_zero_for_csv", False)),
        history_start=history_start,
        history_end=history_end,
        history_weeks=history_weeks,
        demand_scale=demand_scale,
        selected_skus=selected_skus,
    )
    print(
        f"[1/6] Loaded ({data_meta['input_mode']}) -> {data_meta['panel_rows']:,} SKU-week rows | "
        f"{data_meta['n_skus']} SKUs | {data_meta['n_weeks']} weeks "
        f"({data_meta['selected_date_min']} -> {data_meta['selected_date_max']}) | Scale: {demand_scale}"
    )

    lags = cfg["features"]["lags"]
    rolling_windows = cfg["features"]["rolling_windows"]
    df_feat, all_features, cat_features = build_causal_features(
        panel=panel,
        lags=lags,
        rolling_windows=rolling_windows,
        categorical_cols=cfg["features"]["categorical_cols"],
        optional_numeric_cols=cfg["features"]["optional_numeric_cols"],
    )
    print(f"[2/6] Built {len(all_features)} causal features ({len(cat_features)} native categorical)")

    train_df, val_df, test_df, split_meta = chronological_time_split(
        df_feat=df_feat,
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        gap_weeks=gap_weeks,
        min_history_weeks=int(cfg["data"].get("min_history_weeks", 8)),
    )
    print(
        f"[3/6] Split -> Train: {split_meta['train_rows']:,} ({split_meta['train_period']}) | "
        f"Val: {split_meta['val_rows']:,} | Test: {split_meta['test_rows']:,} ({split_meta['test_period']})"
    )

    t_train = time.perf_counter()
    final_model, best_info, tuning_df = tune_and_train_hgb(
        train_df=train_df,
        val_df=val_df,
        all_features=all_features,
        categorical_features=cat_features,
        default_params=cfg["model"]["default_params"],
        tuning_grid=cfg["model"].get("tuning_grid"),
        tune=tune,
        target_transform=cfg["model"].get("target_transform", "log1p"),
        seed=seed,
    )
    train_sec = round(time.perf_counter() - t_train, 3)
    tuning_df.to_csv(out_dir / f"hgb_{dataset_version}_tuning_log.csv", index=False)
    print(f"[4/6] Trained HGB (tune={tune}, {best_info['n_trials']} trials, {train_sec}s) | Best Val MAE: {best_info['best_val_MAE']}")

    test_pred_raw = final_model.predict(test_df[all_features])
    test_pred_clipped = np.maximum(test_pred_raw, 0.0)
    test_metrics = compute_regression_metrics(test_df["Actual_Quantity"].values, test_pred_raw)

    ma4_pred = test_df["rolling_mean_4"].fillna(test_df["lag_1"]).values
    ma4_metrics = compute_regression_metrics(test_df["Actual_Quantity"].values, ma4_pred)
    delta_mae = compute_dq_delta_mae(test_metrics["MAE"], mae_d0_ref)

    pred_out = pd.DataFrame(
        {
            "SKU_ID": test_df["SKU_ID"].values,
            "Forecast_Week": test_df["Forecast_Week"].dt.strftime("%Y-%m-%d").values,
            "Actual_Quantity": test_df["Actual_Quantity"].values,
            "Predicted_Quantity": np.round(test_pred_clipped, 4),
            "Model": model_name,
            "Dataset_Version": dataset_version,
        }
    )
    pred_out.to_csv(out_dir / f"hgb_{dataset_version}_test_predictions.csv", index=False)
    try:
        pred_out.to_parquet(out_dir / f"hgb_{dataset_version}_test_predictions.parquet", index=False)
    except Exception:
        pass

    diag_out = pred_out.copy()
    if "category" in test_df.columns:
        diag_out["category"] = test_df["category"].values
        error_breakdown_by_group(diag_out, "category").to_csv(out_dir / f"hgb_{dataset_version}_error_by_category.csv", index=False)
    if "stockout_flag" in test_df.columns:
        diag_out["stockout_flag"] = test_df["stockout_flag"].values
        error_breakdown_by_group(diag_out, "stockout_flag").to_csv(out_dir / f"hgb_{dataset_version}_error_by_stockout.csv", index=False)

    fi_df = pd.DataFrame()
    if cfg["output"].get("compute_permutation_importance", True):
        fi_df = compute_feature_importance(final_model, test_df, all_features, int(cfg["output"].get("permutation_repeats", 5)), seed)
        fi_df.to_csv(out_dir / f"hgb_{dataset_version}_feature_importance.csv", index=False)

    append_experiment_log(
        out_dir / "experiment_registry.csv", experiment_id, dataset_version, model_name,
        feature_version, split_meta, best_info["best_params"], seed, test_metrics, train_sec, delta_mae,
    )
    print("[5/6] Evaluated Test & exported Spec-compliant predictions + registry")

    full_train_df = df_feat.dropna(subset=["Actual_Quantity", "lag_1"]).copy()
    numeric_features = [c for c in all_features if c not in cat_features]
    serving_model = build_hgb_pipeline(
        numeric_features, cat_features, best_info["best_params"], cfg["model"].get("target_transform", "log1p"), seed
    )
    serving_model.fit(full_train_df[all_features], full_train_df["Actual_Quantity"].values)

    weekly_fc, summary_fc, daily_fc = forecast_next_n_days(
        serving_model, panel, all_features, lags, rolling_windows, forecast_days, dataset_version, model_name
    )
    weekly_fc.to_csv(out_dir / f"hgb_{dataset_version}_forecast_{forecast_days}d_weekly.csv", index=False)
    summary_fc.to_csv(out_dir / f"hgb_{dataset_version}_forecast_{forecast_days}d_summary.csv", index=False)
    daily_fc.to_csv(out_dir / f"hgb_{dataset_version}_forecast_{forecast_days}d_daily.csv", index=False)

    joblib.dump(
        {
            "model_name": model_name,
            "dataset_version": dataset_version,
            "experiment_id": experiment_id,
            "serving_model": serving_model,
            "eval_model": final_model,
            "all_features": all_features,
            "categorical_features": cat_features,
            "lags": lags,
            "rolling_windows": rolling_windows,
            "best_params": best_info["best_params"],
            "test_metrics": test_metrics,
            "data_meta": data_meta,
            "split_meta": split_meta,
        },
        out_dir / f"hgb_{dataset_version}_pipeline.joblib",
    )

    payload = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "seed": seed,
        "data_meta": data_meta,
        "split_meta": split_meta,
        "tuning_summary": best_info,
        "test_metrics_hgb": test_metrics,
        "test_metrics_ma4_reference": ma4_metrics,
        "delta_mae_vs_d0": delta_mae,
        "forecast_days": forecast_days,
        "total_pipeline_seconds": round(time.perf_counter() - t0, 3),
    }
    with open(out_dir / f"hgb_{dataset_version}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if cfg["output"].get("save_plots", True):
        plot_actual_vs_predicted_aggregate(pred_out, out_dir / f"plot_{dataset_version}_actual_vs_predicted.png", dataset_version)
        if not fi_df.empty:
            plot_feature_importance(fi_df, out_dir / f"plot_{dataset_version}_feature_importance.png")
        plot_history_and_forecast(panel, weekly_fc, out_dir / f"plot_{dataset_version}_forecast_{forecast_days}d.png", forecast_days)

    print(f"[6/6] Forecasted next {forecast_days} days | {len(summary_fc)} SKUs | Total = {summary_fc['Total_Predicted_Quantity'].sum():,.2f} units")
    print("-" * 78)
    print(f"  HGB  ({dataset_version}) -> MAE: {test_metrics['MAE']:.4f} | RMSE: {test_metrics['RMSE']:.4f} | WAPE: {test_metrics['WAPE_pct']}% | R2: {test_metrics['R2']:.4f}")
    print(f"  MA4* ({dataset_version}) -> MAE: {ma4_metrics['MAE']:.4f} | RMSE: {ma4_metrics['RMSE']:.4f} | WAPE: {ma4_metrics['WAPE_pct']}% | R2: {ma4_metrics['R2']:.4f}")
    if delta_mae is not None:
        print(f"  DQ Impact -> Delta_MAE ({dataset_version} - D0): {delta_mae:+.4f}")
    print("=" * 78)
    return payload


def cmd_run(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else (SCRIPT_DIR / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    mae_d0_ref: Optional[float] = None
    if args.d0_metrics and Path(args.d0_metrics).exists():
        with open(args.d0_metrics, "r", encoding="utf-8") as f:
            mae_d0_ref = json.load(f).get("test_metrics_hgb", {}).get("MAE")

    if args.run_all_in_input:
        in_dir = SCRIPT_DIR / "input_data"
        targets = sorted(in_dir.glob("*.csv"), key=lambda p: (0 if detect_dataset_version_from_name(p, "ZZ") == "D0" else 1, p.name.lower()))
        if not targets:
            raise FileNotFoundError(f"No CSV files found in {in_dir}")
    else:
        targets = [resolve_input_source(args)]

    rows = []
    for idx, src in enumerate(targets):
        ds_ver = args.dataset_version if (args.dataset_version and len(targets) == 1) else detect_dataset_version_from_name(src, f"D{idx}")
        res = run_single(src, args, cfg, out_dir, ds_ver, mae_d0_ref=mae_d0_ref)
        if ds_ver == "D0" and mae_d0_ref is None:
            mae_d0_ref = res["test_metrics_hgb"]["MAE"]
        rows.append(
            {
                "Dataset_Version": ds_ver,
                "Source": src.name,
                "HGB_MAE": res["test_metrics_hgb"]["MAE"],
                "HGB_RMSE": res["test_metrics_hgb"]["RMSE"],
                "HGB_WAPE_pct": res["test_metrics_hgb"]["WAPE_pct"],
                "HGB_R2": res["test_metrics_hgb"]["R2"],
                "Delta_MAE_vs_D0": res["delta_mae_vs_d0"] if res["delta_mae_vs_d0"] is not None else (0.0 if ds_ver == "D0" else ""),
                "MA4_MAE": res["test_metrics_ma4_reference"]["MAE"],
            }
        )
    if len(rows) > 1:
        dq_df = pd.DataFrame(rows)
        dq_df.to_csv(out_dir / "dq_comparison_summary.csv", index=False)
        print("\n>>> BATCH DQ SUMMARY (outputs/dq_comparison_summary.csv):")
        print(dq_df.to_string(index=False))
    print(f"\nSaved all artifacts to: {out_dir.resolve()}")


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if args.command == "validate":
        cmd_validate(args, cfg)
    else:
        cmd_run(args, cfg)


if __name__ == "__main__":
    main()
