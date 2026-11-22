"""Orchestration: expanding validation, untouched temporal test, deploy refit."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import uuid

import joblib
import numpy as np
import pandas as pd

from . import CONTRACT_VERSION, __version__
from .config import Config, monday
from .data import load_weekly_sales, sha256_file
from .features import FEATURE_COLUMNS, NUMERIC_FEATURES, build_features
from .model import evaluate_predictions, fit_forest, predict_rows, regression_metrics


def write_json(path: Path, payload: dict):
    def convert(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (pd.Timestamp, Path)):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=convert) + "\n", encoding="utf-8")


def _new_output(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "scikit-learn", "joblib")}


def _manifest(config: Config, files: list[Path], run_id: str, mode: str) -> dict:
    config_json = json.dumps(config.to_dict(), sort_keys=True)
    return {"run_id": run_id, "mode": mode, "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": __version__, "contract_version": CONTRACT_VERSION, "config": config.to_dict(),
            "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
            "inputs": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files],
            "versions": _versions(), "feature_columns": FEATURE_COLUMNS,
            "assumptions": ["Gross completed units using OrderDate and final status snapshot, not net returns or unconstrained demand.",
                            "Naive local source dates: source timezone and ingestion/status history unverified.",
                            "After first observed sale, every complete-feed week is in cohort; lifecycle and stockouts are not verified.",
                            "No price, future promotions, inventory, customer attributes or realized target-week covariates."]}


def _save_data(output: Path, data, features: pd.DataFrame):
    write_json(output / "dq_report.json", data.report)
    data.panel.to_csv(output / "weekly_sales.csv", index=False, date_format="%Y-%m-%d")
    features.to_csv(output / "features.csv", index=False, date_format="%Y-%m-%d")


def _forecast_export(predictions: pd.DataFrame, run_id: str) -> pd.DataFrame:
    result = predictions.drop(columns=["actual_quantity"]).copy()
    result["run_id"] = run_id
    # as_of is an exclusive event-time boundary, NOT a verified ingestion timestamp.
    result["as_of"] = result.origin_week + pd.Timedelta(days=7)
    result["contract_version"] = CONTRACT_VERSION
    return result


def run_pipeline(config: Config, data_dir: str | Path, output_dir: str | Path) -> dict:
    output = _new_output(output_dir)
    started = time.perf_counter()
    run_id = f"rf-{uuid.uuid4().hex[:12]}"
    data = load_weekly_sales(data_dir, config)
    features = build_features(data.panel, config)
    _save_data(output, data, features)
    manifest = _manifest(config, data.input_files, run_id, "run")
    write_json(output / "run_manifest.json", manifest)
    train_end = monday(config.train_end_week)
    val_end = monday(config.validation_end_week)
    end = monday(config.data_end_week)
    validation_weeks = pd.date_range(train_end + pd.Timedelta(days=7), val_end, freq="W-MON")
    blocks = [pd.DatetimeIndex(block) for block in np.array_split(validation_weeks.to_numpy(), config.validation_blocks)]
    validation_results = []
    block_manifest = []
    for block in blocks:
        block_manifest.append({"first_target_week": str(block.min().date()), "last_target_week": str(block.max().date()),
                               "fit_targets_through": str((block.min() - pd.Timedelta(days=7)).date())})
    for candidate_id, params in enumerate(config.candidates):
        pieces = []
        for block_id, block in enumerate(blocks):
            training = features.loc[features.target_week.lt(block.min())]
            model = fit_forest(training, params, config)
            rows = features.loc[features.target_week.isin(block)]
            if rows.empty:
                raise ValueError(f"Validation block {block_id} contains no cohort rows")
            predictions = predict_rows(model, rows)
            predictions["candidate_id"] = candidate_id
            predictions["validation_block"] = block_id
            predictions["fit_targets_through"] = block.min() - pd.Timedelta(days=7)
            pieces.append(predictions)
        all_predictions = pd.concat(pieces, ignore_index=True)
        metrics = regression_metrics(all_predictions, "qhat")
        if metrics["scored_rows"] == 0:
            raise ValueError("Validation has no trustworthy available labels")
        score = metrics["wape"] if metrics["wape"] is not None else metrics["mae"]
        validation_results.append({"candidate_id": candidate_id, "params": params, "selection_score": score,
                                   "metrics": metrics, "predictions": all_predictions})
    selected = min(validation_results, key=lambda item: (item["selection_score"], item["candidate_id"]))
    params = selected["params"]
    validation_predictions = pd.concat([item["predictions"] for item in validation_results], ignore_index=True)
    validation_predictions.to_csv(output / "validation_predictions.csv", index=False, date_format="%Y-%m-%d")
    # One fixed model for the final test; actuals through each origin may update context, never fit parameters.
    test_model = fit_forest(features.loc[features.target_week.le(val_end)], params, config)
    test_rows = features.loc[features.target_week.gt(val_end) & features.target_week.le(end)]
    if test_rows.empty:
        raise ValueError("No test cohort rows")
    test_predictions = predict_rows(test_model, test_rows)
    if not (test_predictions.actual_quantity.notna() & test_predictions.qhat.notna()).any():
        raise ValueError("No scorable test rows with both trustworthy actual and available prediction; inspect coverage")
    test_predictions["fit_targets_through"] = val_end
    test_predictions.to_csv(output / "test_predictions.csv", index=False, date_format="%Y-%m-%d")
    metrics = {"selection": {"criterion": "aggregate validation WAPE (MAE only if all validation actuals sum to zero)",
                              "selected_candidate_id": selected["candidate_id"], "selected_params": params,
                              "candidates": [{k: v for k, v in item.items() if k != "predictions"} for item in validation_results]},
               "validation_selected": evaluate_predictions(selected["predictions"]),
               "test": evaluate_predictions(test_predictions),
               "metric_units": "MAE/RMSE: units; WAPE/bias: fraction, multiply by 100 for percent; null if denominator=0",
               "test_protocol": "Fixed fit through validation_end; rolling H=1 context, no test fit/tuning. Previously explored legacy data: not guaranteed blind."}
    write_json(output / "metrics.json", metrics)
    # Deployment model is deliberately distinct from the model used to produce test scores.
    deploy_model = fit_forest(features.loc[features.target_week.le(end)], params, config)
    artifact = {"contract_version": CONTRACT_VERSION, "pipeline_version": __version__, "model": deploy_model,
                "params": params, "config": config.to_dict(), "feature_columns": FEATURE_COLUMNS,
                "trained_through_target_week": config.data_end_week, "training_run_id": run_id, "versions": _versions()}
    joblib.dump(artifact, output / "model.joblib")
    latest = features.loc[features.origin_week.eq(end)]
    forecast = _forecast_export(predict_rows(deploy_model, latest), run_id)
    forecast.to_csv(output / "forecast.csv", index=False, date_format="%Y-%m-%d")
    preprocess = deploy_model.named_steps["preprocess"]
    pd.DataFrame({"feature": preprocess.get_feature_names_out(), "importance": deploy_model.named_steps["forest"].feature_importances_}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    source = Path(__file__).parent
    source_hashes = {path.name: sha256_file(path) for path in sorted(source.glob("*.py"))}
    split_keys = features[["SKU_ID", "origin_week", "target_week"]].to_csv(index=False, date_format="%Y-%m-%d")
    manifest.update({"status": "completed", "elapsed_seconds": time.perf_counter() - started, "source_sha256": source_hashes,
                     "model_sha256": sha256_file(output / "model.joblib"), "validation_blocks": block_manifest,
                     "split_keys_sha256": hashlib.sha256(split_keys.encode()).hexdigest(),
                     "selected_candidate_id": selected["candidate_id"], "selected_params": params,
                     "test_model_trained_through": config.validation_end_week, "deploy_model_trained_through": config.data_end_week,
                     "forecast_origin_week": config.data_end_week, "forecast_rows": len(forecast),
                     "forecast_status_counts": forecast.prediction_status.value_counts().to_dict(),
                     "artifacts": sorted(path.name for path in output.iterdir() if path.is_file())})
    write_json(output / "run_manifest.json", manifest)
    return {"run_id": run_id, "output_dir": str(output), "selected_params": params, "test": metrics["test"]["all_rows"],
            "forecast_rows": len(forecast), "elapsed_seconds": manifest["elapsed_seconds"]}


def predict_pipeline(config: Config, data_dir: str | Path, output_dir: str | Path, artifact_path: str | Path, origin_week: str) -> dict:
    output = _new_output(output_dir)
    origin = monday(origin_week, "origin_week")
    # joblib uses pickle: only load artifacts you trust; CLI/docs explicitly warn users.
    artifact_path = Path(artifact_path).expanduser().resolve()
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or artifact.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Incompatible model artifact contract")
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Incompatible model artifact feature schema")
    if artifact.get("versions", {}).get("scikit-learn") != _versions()["scikit-learn"]:
        raise ValueError("Artifact scikit-learn version differs; use the original environment or retrain")
    saved_config = artifact["config"]
    for field in ("data_start_week", "min_history_weeks", "assume_complete_weeks"):
        if saved_config[field] != getattr(config, field):
            raise ValueError(f"Inference config changes label/history contract field: {field}")
    if origin < monday(artifact["trained_through_target_week"]):
        raise ValueError("origin_week precedes model training cutoff: historical inference would leak future training labels")
    if origin > monday(saved_config["data_end_week"]) and not (Path(data_dir).expanduser() / "WeeklyCoverage.csv").is_file():
        raise ValueError("Forward inference beyond the artifact snapshot requires WeeklyCoverage.csv; the old complete-feed assumption must not zero-pad unseen future weeks")
    data = load_weekly_sales(data_dir, config, end_week=origin_week)
    features = build_features(data.panel, config)
    _save_data(output, data, features)
    run_id = f"rf-predict-{uuid.uuid4().hex[:12]}"
    forecast = _forecast_export(predict_rows(artifact["model"], features.loc[features.origin_week.eq(origin)]), run_id)
    forecast.to_csv(output / "forecast.csv", index=False, date_format="%Y-%m-%d")
    manifest = _manifest(config, data.input_files, run_id, "predict")
    manifest.update({"status": "completed", "artifact_path": str(artifact_path), "artifact_sha256": sha256_file(artifact_path),
                     "training_run_id": artifact["training_run_id"], "trained_through_target_week": artifact["trained_through_target_week"],
                     "forecast_origin_week": origin_week, "forecast_rows": len(forecast),
                     "forecast_status_counts": forecast.prediction_status.value_counts().to_dict()})
    write_json(output / "run_manifest.json", manifest)
    return {"run_id": run_id, "output_dir": str(output), "forecast_rows": len(forecast)}


def validate_pipeline(config: Config, data_dir: str | Path, output_dir: str | Path) -> dict:
    output = _new_output(output_dir)
    data = load_weekly_sales(data_dir, config)
    features = build_features(data.panel, config)
    _save_data(output, data, features)
    manifest = _manifest(config, data.input_files, f"rf-validate-{uuid.uuid4().hex[:12]}", "validate")
    manifest["status"] = "completed"
    write_json(output / "run_manifest.json", manifest)
    return {"status": "passed", "output_dir": str(output), "sku_count": data.report["sku_count"], "feature_rows": len(features)}
