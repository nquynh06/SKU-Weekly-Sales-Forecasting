"""Offline synthetic fixtures only. These tests do not claim benchmark accuracy."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rf_pipeline.config import Config, monday
from rf_pipeline.data import DataContractError, load_weekly_sales
from rf_pipeline.features import FEATURE_COLUMNS, NUMERIC_FEATURES, build_features
from rf_pipeline.model import regression_metrics
from rf_pipeline.pipeline import predict_pipeline, run_pipeline, validate_pipeline
from rf_pipeline.cli import main


CANDIDATES = [{"n_estimators": 6, "max_depth": 4, "min_samples_leaf": 2, "max_features": 1.0}]


def test_config():
    return Config(data_start_week="2024-01-01", train_end_week="2024-03-25", validation_end_week="2024-04-29",
                  data_end_week="2024-06-03", assume_complete_weeks=True, n_jobs=1, candidates=CANDIDATES,
                  validation_blocks=3)


# Avoid pytest interpreting this config factory as a test returning an object.
test_config.__test__ = False


def make_raw(root: Path, config: Config, cancelled: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    weeks = pd.date_range(config.data_start_week, config.data_end_week, freq="W-MON")
    orders, lines = [], []
    for index, week in enumerate(weeks):
        # SKU B sells only on even weeks; missing completed transactions must become zero.
        for sku in ("A", "B"):
            if sku == "B" and index % 2:
                continue
            order = f"ORDER_{index}_{sku}"
            orders.append({"SalesOrderID": order, "OrderDate": str((week + pd.Timedelta(days=2)).date()), "OrderStatus": " Complete "})
            lines.append({"SalesOrderLineID": f"LINE_{index}_{sku}", "SalesOrderID": order, "LineNumber": 1,
                          "SKU_ID": sku, "Quantity": index % 4 + 1, "ActualSellingPrice": ""})
    # Short-history SKU with one positive sale: must appear in latest inference via MA4, not be silently dropped.
    orders.append({"SalesOrderID": "NEW", "OrderDate": config.data_end_week, "OrderStatus": "Completed"})
    lines.append({"SalesOrderLineID": "NEW", "SalesOrderID": "NEW", "LineNumber": 1, "SKU_ID": "C", "Quantity": 3})
    if cancelled:
        orders.append({"SalesOrderID": "CANCEL", "OrderDate": config.data_end_week, "OrderStatus": " Canceled "})
        lines.append({"SalesOrderLineID": "CANCEL", "SalesOrderID": "CANCEL", "LineNumber": 1, "SKU_ID": "A", "Quantity": 99})
    pd.DataFrame(orders).to_csv(root / "SalesOrders.csv", index=False)
    pd.DataFrame(lines).to_csv(root / "SalesOrderLines.csv", index=False)
    pd.DataFrame({"SKU_ID": ["A", "B", "C"]}).to_csv(root / "ProductVariants.csv", index=False)


class RandomForestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sku-rf-tests-")
        self.root = Path(self.temp.name)
        self.config = test_config()
        self.raw = self.root / "raw"
        make_raw(self.raw, self.config)

    def tearDown(self):
        self.temp.cleanup()

    def read_csv(self, name):
        return pd.read_csv(self.raw / name)

    def save_csv(self, name, frame):
        frame.to_csv(self.raw / name, index=False)

    def test_config_rejects_unknown_keys_bad_dates_and_insufficient_history(self):
        with self.assertRaises(ValueError):
            monday("2024-01-02")
        with self.assertRaises(ValueError):
            replace(self.config, min_history_weeks=4)
        path = self.root / "bad.json"
        path.write_text('{"unexpected": 1}')
        with self.assertRaises(ValueError):
            Config.load(path)

    def test_max_features_integer_one_is_fraction_not_one_column(self):
        from rf_pipeline.model import fit_forest
        features = build_features(load_weekly_sales(self.raw, self.config).panel, self.config)
        for value in (1, 1.0):
            params = {**CANDIDATES[0], "max_features": value}
            fitted = fit_forest(features.loc[features.target_week.le(monday(self.config.train_end_week))], params, self.config)
            forest = fitted.named_steps["forest"]
            self.assertIs(type(forest.max_features), float)
            self.assertEqual(forest.estimators_[0].max_features_, forest.n_features_in_)

    def test_test_period_requires_overlapping_actual_and_prediction(self):
        weeks = pd.date_range(self.config.data_start_week, self.config.data_end_week, freq="W-MON")
        complete = [week <= monday(self.config.validation_end_week) or week == weeks[-1] for week in weeks]
        pd.DataFrame({"week_start": weeks, "is_complete": complete}).to_csv(self.raw / "WeeklyCoverage.csv", index=False)
        output = self.root / "unscorable-test"
        with self.assertRaisesRegex(ValueError, "No scorable test rows"):
            run_pipeline(self.config, self.raw, output)
        self.assertFalse((output / "model.joblib").exists())

    def test_complete_status_alias_and_cancelled_exclusion(self):
        make_raw(self.raw, self.config, cancelled=True)
        data = load_weekly_sales(self.raw, self.config)
        self.assertEqual(data.report["excluded_cancelled_in_window_lines"], 1)
        self.assertTrue(data.panel.quantity.max() < 99)

    def test_exact_duplicates_do_not_inflate_units(self):
        original = load_weekly_sales(self.raw, self.config)
        for name in ("SalesOrders.csv", "SalesOrderLines.csv", "ProductVariants.csv"):
            frame = self.read_csv(name)
            self.save_csv(name, pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
        dedup = load_weekly_sales(self.raw, self.config)
        self.assertEqual(original.report["gross_completed_units"], dedup.report["gross_completed_units"])
        self.assertEqual(dedup.report["tables"]["SalesOrderLines.csv"]["exact_duplicates_removed"], 1)

    def test_conflicting_primary_key_is_hard_error(self):
        frame = self.read_csv("SalesOrderLines.csv")
        extra = frame.iloc[[0]].copy()
        extra["Quantity"] = 100
        self.save_csv("SalesOrderLines.csv", pd.concat([frame, extra], ignore_index=True))
        with self.assertRaisesRegex(DataContractError, "conflicting"):
            load_weekly_sales(self.raw, self.config)

    def test_business_key_is_hard_error(self):
        frame = self.read_csv("SalesOrderLines.csv")
        extra = frame.iloc[[0]].copy()
        extra["SalesOrderLineID"] = "different-id"
        self.save_csv("SalesOrderLines.csv", pd.concat([frame, extra], ignore_index=True))
        with self.assertRaisesRegex(DataContractError, "business key"):
            load_weekly_sales(self.raw, self.config)

    def test_fk_is_hard_error(self):
        frame = self.read_csv("SalesOrderLines.csv")
        frame.loc[0, "SalesOrderID"] = "MISSING"
        self.save_csv("SalesOrderLines.csv", frame)
        with self.assertRaisesRegex(DataContractError, "orphan"):
            load_weekly_sales(self.raw, self.config)

    def test_unknown_sku_is_hard_error(self):
        frame = self.read_csv("SalesOrderLines.csv")
        frame.loc[0, "SKU_ID"] = "UNKNOWN"
        self.save_csv("SalesOrderLines.csv", frame)
        with self.assertRaisesRegex(DataContractError, "unknown SKU"):
            load_weekly_sales(self.raw, self.config)

    def test_quantity_rejects_zero_negative_fraction_and_nan(self):
        frame = self.read_csv("SalesOrderLines.csv")
        for value in (0, -2, 1.5, np.nan, float("inf")):
            with self.subTest(value=value):
                bad = frame.copy()
                bad["Quantity"] = bad.Quantity.astype(float)
                bad.loc[0, "Quantity"] = value
                self.save_csv("SalesOrderLines.csv", bad)
                with self.assertRaisesRegex(DataContractError, "Quantity"):
                    load_weekly_sales(self.raw, self.config)

    def test_unknown_status_is_hard_error(self):
        frame = self.read_csv("SalesOrders.csv")
        frame.loc[0, "OrderStatus"] = "Processing"
        self.save_csv("SalesOrders.csv", frame)
        with self.assertRaisesRegex(DataContractError, "Unknown OrderStatus"):
            load_weekly_sales(self.raw, self.config)

    def test_malformed_date_is_hard_error(self):
        frame = self.read_csv("SalesOrders.csv")
        frame.loc[0, "OrderDate"] = "not-a-date"
        self.save_csv("SalesOrders.csv", frame)
        with self.assertRaisesRegex(DataContractError, "unparseable"):
            load_weekly_sales(self.raw, self.config)

    def test_date_outliers_are_excluded(self):
        frame = self.read_csv("SalesOrders.csv")
        frame.loc[0, "OrderDate"] = "1900-01-01"
        self.save_csv("SalesOrders.csv", frame)
        data = load_weekly_sales(self.raw, self.config)
        self.assertEqual(data.report["excluded_outside_window_lines"], 1)

    def test_missing_feed_requires_explicit_assumption(self):
        with self.assertRaisesRegex(DataContractError, "WeeklyCoverage"):
            load_weekly_sales(self.raw, replace(self.config, assume_complete_weeks=False))

    def test_zero_is_distinct_from_missing_and_no_prelaunch_padding(self):
        weeks = pd.date_range(self.config.data_start_week, self.config.data_end_week, freq="W-MON")
        pd.DataFrame({"week_start": weeks, "is_complete": [False if index == 2 else True for index in range(len(weeks))]}).to_csv(self.raw / "WeeklyCoverage.csv", index=False)
        data = load_weekly_sales(self.raw, replace(self.config, assume_complete_weeks=False))
        b = data.panel.loc[data.panel.SKU_ID.eq("B")].reset_index(drop=True)
        self.assertEqual(b.loc[1, "quantity"], 0)
        self.assertTrue(np.isnan(b.loc[2, "quantity"]))
        self.assertEqual(len(data.panel.loc[data.panel.SKU_ID.eq("C")]), 1)
        features = build_features(data.panel, self.config)
        self.assertFalse(features.loc[features.origin_week.eq(weeks[3]), "rf_eligible"].any())

    def test_missing_coverage_calendar_entry_is_not_zero(self):
        pd.DataFrame({"week_start": [self.config.data_start_week], "is_complete": [True]}).to_csv(self.raw / "WeeklyCoverage.csv", index=False)
        data = load_weekly_sales(self.raw, self.config)
        self.assertTrue(data.panel.loc[data.panel.week_start.gt(monday(self.config.data_start_week)), "quantity"].isna().all())

    def test_lag_alignment_label_horizon_and_no_future_leakage(self):
        data = load_weekly_sales(self.raw, self.config)
        features = build_features(data.panel, self.config)
        chosen = features.loc[features.SKU_ID.eq("A")].iloc[10]
        history = data.panel.loc[data.panel.SKU_ID.eq("A")].set_index("week_start")
        self.assertEqual(chosen.lag_1, history.loc[chosen.origin_week, "quantity"])
        self.assertEqual(chosen.lag_8, history.loc[chosen.origin_week - pd.Timedelta(weeks=7), "quantity"])
        self.assertEqual(chosen.actual_quantity, history.loc[chosen.target_week, "quantity"])
        mutated = data.panel.copy()
        mutated.loc[mutated.week_start.gt(chosen.origin_week), "quantity"] = 98765
        changed = build_features(mutated, self.config)
        left = features.loc[features.origin_week.le(chosen.origin_week), FEATURE_COLUMNS + ["rf_eligible"]].reset_index(drop=True)
        right = changed.loc[changed.origin_week.le(chosen.origin_week), FEATURE_COLUMNS + ["rf_eligible"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
        self.assertNotIn("actual_quantity", FEATURE_COLUMNS)
        self.assertEqual(features.target_week.sub(features.origin_week).unique().tolist(), [pd.Timedelta(days=7)])

    def test_raw_future_quantity_perturbation_cannot_change_past_features(self):
        original = build_features(load_weekly_sales(self.raw, self.config).panel, self.config)
        cutoff = pd.Timestamp("2024-03-11")
        orders = self.read_csv("SalesOrders.csv")
        future_orders = set(orders.loc[pd.to_datetime(orders.OrderDate).gt(cutoff + pd.Timedelta(days=6)), "SalesOrderID"])
        lines = self.read_csv("SalesOrderLines.csv")
        lines.loc[lines.SalesOrderID.isin(future_orders), "Quantity"] = 9999
        self.save_csv("SalesOrderLines.csv", lines)
        changed = build_features(load_weekly_sales(self.raw, self.config).panel, self.config)
        columns = FEATURE_COLUMNS + ["origin_week", "target_week", "rf_eligible"]
        pd.testing.assert_frame_equal(
            original.loc[original.origin_week.le(cutoff), columns].reset_index(drop=True),
            changed.loc[changed.origin_week.le(cutoff), columns].reset_index(drop=True))

    def test_empty_recent_history_returns_unavailable_without_fake_zero(self):
        import joblib
        run = self.root / "run-for-missing"
        run_pipeline(self.config, self.raw, run)
        artifact = joblib.load(run / "model.joblib")
        weeks = pd.date_range(self.config.data_start_week, self.config.data_end_week, freq="W-MON")
        pd.DataFrame({"week_start": weeks, "is_complete": [True] * (len(weeks) - 4) + [False] * 4}).to_csv(self.raw / "WeeklyCoverage.csv", index=False)
        from rf_pipeline.model import predict_rows
        features = build_features(load_weekly_sales(self.raw, self.config).panel, self.config)
        predictions = predict_rows(artifact["model"], features.loc[features.origin_week.eq(weeks[-1])])
        self.assertTrue(predictions.qhat.isna().all())
        self.assertTrue(predictions.prediction_status.eq("unavailable").all())

    def test_unknown_category_has_explicit_flag_and_finite_prediction(self):
        import joblib
        from rf_pipeline.model import predict_rows
        run = self.root / "run-for-new-sku"
        run_pipeline(self.config, self.raw, run)
        artifact = joblib.load(run / "model.joblib")
        features = build_features(load_weekly_sales(self.raw, self.config).panel, self.config)
        rows = features.loc[features.SKU_ID.eq("A") & features.origin_week.eq(monday(self.config.data_end_week))].copy()
        rows["SKU_ID"] = "NEW_UNSEEN_IN_ENCODER"
        predicted = predict_rows(artifact["model"], rows)
        self.assertFalse(predicted.sku_seen_in_training.any())
        self.assertTrue(predicted.qhat.notna().all())
        self.assertTrue(predicted.prediction_status.eq("random_forest").all())

    def test_zero_denominator_metrics_are_null_not_epsilon(self):
        rows = pd.DataFrame({"SKU_ID": ["A", "B"], "actual_quantity": [0.0, 0.0], "qhat": [1.0, 2.0]})
        metrics = regression_metrics(rows, "qhat")
        self.assertIsNone(metrics["wape"])
        self.assertIsNone(metrics["bias"])
        self.assertEqual(metrics["mae"], 1.5)

    def test_end_to_end_temporal_split_artifact_reload_and_fallback(self):
        output = self.root / "run"
        result = run_pipeline(self.config, self.raw, output)
        self.assertEqual(result["forecast_rows"], 3)
        forecast = pd.read_csv(output / "forecast.csv")
        self.assertNotIn("actual_quantity", forecast.columns)
        self.assertTrue(forecast.qhat.ge(0).all())
        self.assertEqual(forecast.set_index("SKU_ID").loc["C", "prediction_status"], "fallback_ma4")
        validation = pd.read_csv(output / "validation_predictions.csv", parse_dates=["target_week", "fit_targets_through"])
        self.assertTrue(validation.target_week.gt(validation.fit_targets_through).all())
        test = pd.read_csv(output / "test_predictions.csv", parse_dates=["target_week", "fit_targets_through"])
        self.assertTrue(test.target_week.gt(monday(self.config.validation_end_week)).all())
        self.assertTrue(test.fit_targets_through.eq(monday(self.config.validation_end_week)).all())
        pred_dir = self.root / "predict"
        predict_pipeline(self.config, self.raw, pred_dir, output / "model.joblib", self.config.data_end_week)
        reloaded = pd.read_csv(pred_dir / "forecast.csv")
        np.testing.assert_allclose(forecast.qhat, reloaded.qhat)
        with self.assertRaisesRegex(ValueError, "precedes model training"):
            predict_pipeline(self.config, self.raw, self.root / "bad-past", output / "model.joblib", "2024-05-27")
        with self.assertRaises(FileExistsError):
            run_pipeline(self.config, self.raw, output)
        # An old pilot assumption may not be silently extended to unseen future weeks.
        with self.assertRaisesRegex(ValueError, "requires WeeklyCoverage"):
            predict_pipeline(self.config, self.raw, self.root / "bad-future", output / "model.joblib", "2024-07-08")
        pd.DataFrame({"week_start": pd.date_range(self.config.data_start_week, self.config.data_end_week, freq="W-MON"),
                      "is_complete": True}).to_csv(self.raw / "WeeklyCoverage.csv", index=False)
        future = self.root / "covered-future"
        predict_pipeline(self.config, self.raw, future, output / "model.joblib", "2024-07-08")
        future_forecast = pd.read_csv(future / "forecast.csv")
        self.assertTrue(future_forecast.qhat.isna().all())
        self.assertTrue(future_forecast.prediction_status.eq("unavailable").all())

    def test_cli_failure_creates_diagnostic_and_nonzero_exit(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config.to_dict()))
        output = self.root / "failed"
        result = main(["validate", "--config", str(config_path), "--data-dir", str(self.root / "missing"), "--output-dir", str(output)])
        self.assertEqual(result, 2)
        self.assertTrue((output / "failure.json").exists())
        result_again = main(["validate", "--config", str(config_path), "--data-dir", str(self.raw), "--output-dir", str(output)])
        self.assertEqual(result_again, 2)


if __name__ == "__main__":
    unittest.main()
