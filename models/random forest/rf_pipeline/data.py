"""Raw CSV contract, cardinality checks, complete-week calendar and SKU panel."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import CONTRACT_VERSION
from .config import Config, monday


class DataContractError(ValueError):
    """A hard data-quality failure; no predictions may be issued for this batch."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


@dataclass
class WeeklyData:
    panel: pd.DataFrame
    report: dict
    input_files: list[Path]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(message: str, report: dict):
    report["status"] = "failed"
    report["error"] = message
    raise DataContractError(message, report)


def _read_table(root: Path, filename: str, required: list[str], key: str, report: dict) -> pd.DataFrame:
    path = root / filename
    if not path.is_file():
        _fail(f"Missing required input: {path}", report)
    try:
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    except (ValueError, OSError) as exc:
        _fail(f"Cannot read {filename}: {exc}", report)
    missing = set(required) - set(frame.columns)
    if missing:
        _fail(f"{filename}: missing columns {sorted(missing)}", report)
    info = {"input_rows": len(frame), "exact_duplicates_removed": int(frame.duplicated().sum())}
    report["tables"][filename] = info
    # Full-row dedup BEFORE selecting columns: different prices/attributes are NOT exact duplicates.
    frame = frame.drop_duplicates().copy()
    for column in required:
        frame[column] = frame[column].str.strip()
    frame = frame[required].copy()
    keys = [column for column in required if column.endswith("ID") or column == "SKU_ID"]
    null_keys = {column: int(frame[column].eq("").sum()) for column in keys if frame[column].eq("").any()}
    if null_keys:
        _fail(f"{filename}: blank keys {null_keys}", report)
    conflicts = int(frame.duplicated(key, keep=False).sum())
    info["conflicting_key_rows"] = conflicts
    if conflicts:
        _fail(f"{filename}: conflicting {key}, {conflicts} rows; resolve upstream", report)
    info["deduplicated_rows"] = len(frame)
    return frame


def _coverage(root: Path, weeks: pd.DatetimeIndex, config: Config, report: dict) -> tuple[pd.Series, list[Path]]:
    path = root / "WeeklyCoverage.csv"
    if not path.exists():
        if not config.assume_complete_weeks:
            _fail("WeeklyCoverage.csv missing. Supply audited coverage or explicitly set assume_complete_weeks=true (assumption, not evidence).", report)
        report["coverage_source"] = "assumed_complete_config_not_verified"
        report["warnings"].append("Completeness is assumed for every configured week; zero labels are conditional on this unverified assumption.")
        return pd.Series(True, index=weeks), []
    frame = _read_table(root, path.name, ["week_start", "is_complete"], "week_start", report)
    dates = pd.to_datetime(frame.week_start, format="%Y-%m-%d", errors="coerce")
    if not frame.week_start.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all() or dates.isna().any() or dates.dt.day_of_week.ne(0).any():
        _fail("WeeklyCoverage.csv: week_start must be ISO Monday dates", report)
    values = frame.is_complete.str.lower().map({"true": True, "1": True, "false": False, "0": False})
    if values.isna().any():
        _fail("WeeklyCoverage.csv: is_complete must be true/false or 1/0", report)
    complete = pd.Series(values.to_numpy(dtype=bool), index=pd.DatetimeIndex(dates))
    # Missing calendar entries are UNKNOWN/incomplete, never silently complete.
    complete = complete.reindex(weeks, fill_value=False)
    report["coverage_source"] = "WeeklyCoverage.csv"
    return complete, [path]


def load_weekly_sales(data_dir: str | Path, config: Config, end_week: str | None = None) -> WeeklyData:
    root = Path(data_dir).expanduser().resolve()
    start = monday(config.data_start_week, "data_start_week")
    end = monday(end_week or config.data_end_week, "end_week")
    report = {"contract_version": CONTRACT_VERSION, "status": "validating", "tables": {}, "warnings": [],
              "data_start_week": str(start.date()), "data_end_week": str(end.date()),
              "label": "gross_completed_units", "timezone": "naive_source_dates_not_verified",
              "point_in_time_limitation": "OrderStatus is final snapshot; no status/ingestion history. Event-time backtest, not verified point-in-time replay."}
    if end < start:
        _fail("end_week precedes data_start_week", report)
    orders = _read_table(root, "SalesOrders.csv", ["SalesOrderID", "OrderDate", "OrderStatus"], "SalesOrderID", report)
    lines = _read_table(root, "SalesOrderLines.csv", ["SalesOrderLineID", "SalesOrderID", "LineNumber", "SKU_ID", "Quantity"], "SalesOrderLineID", report)
    variants = _read_table(root, "ProductVariants.csv", ["SKU_ID"], "SKU_ID", report)
    status = orders.OrderStatus.str.lower().map({"complete": "completed", "completed": "completed", "canceled": "cancelled", "cancelled": "cancelled"})
    if status.isna().any():
        unknown = orders.loc[status.isna(), "OrderStatus"].value_counts().to_dict()
        _fail(f"Unknown OrderStatus values: {unknown}", report)
    orders["status"] = status
    dates = pd.to_datetime(orders.OrderDate, format="%Y-%m-%d", errors="coerce")
    invalid_dates = dates.isna() | ~orders.OrderDate.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    if invalid_dates.any():
        _fail(f"SalesOrders.csv: {int(invalid_dates.sum())} unparseable OrderDate values (require YYYY-MM-DD)", report)
    orders["order_date"] = dates
    quantity = pd.to_numeric(lines.Quantity, errors="coerce").astype(float)
    invalid_quantity = ~np.isfinite(quantity) | quantity.le(0) | quantity.mod(1).ne(0)
    if invalid_quantity.any():
        _fail(f"SalesOrderLines.csv: {int(invalid_quantity.sum())} Quantity values are not finite positive integer units", report)
    lines["quantity"] = quantity
    numbers = pd.to_numeric(lines.LineNumber, errors="coerce").astype(float)
    if (~np.isfinite(numbers) | numbers.le(0) | numbers.mod(1).ne(0)).any():
        _fail("SalesOrderLines.csv: LineNumber must be a positive integer", report)
    lines["line_number"] = numbers
    if lines.duplicated(["SalesOrderID", "line_number"]).any():
        _fail("SalesOrderLines.csv: conflicting (SalesOrderID, LineNumber) business key", report)
    if not lines.SalesOrderID.isin(orders.SalesOrderID).all():
        _fail("SalesOrderLines.csv: orphan SalesOrderID foreign key", report)
    if not lines.SKU_ID.isin(variants.SKU_ID).all():
        _fail("SalesOrderLines.csv: unknown SKU_ID (not in ProductVariants.csv)", report)
    joined = lines.merge(orders[["SalesOrderID", "order_date", "status"]], on="SalesOrderID", how="left", validate="many_to_one")
    if len(joined) != len(lines) or not np.isclose(joined.quantity.sum(), lines.quantity.sum()):
        _fail("Join changed row count or units", report)
    report["join_rows"] = len(joined)
    report["joined_input_units"] = float(joined.quantity.sum())
    in_window = joined.order_date.ge(start) & joined.order_date.lt(end + pd.Timedelta(days=7))
    completed = joined.status.eq("completed")
    report["excluded_outside_window_lines"] = int((~in_window).sum())
    report["excluded_outside_window_units"] = float(joined.loc[~in_window, "quantity"].sum())
    report["excluded_cancelled_in_window_lines"] = int((in_window & ~completed).sum())
    report["excluded_cancelled_in_window_units"] = float(joined.loc[in_window & ~completed, "quantity"].sum())
    report["excluded_outside_window_orders"] = int((~orders.order_date.between(start, end + pd.Timedelta(days=6))).sum())
    valid = joined.loc[in_window & completed].copy()
    if valid.empty:
        _fail("No completed sales in configured full-week window", report)
    valid["week_start"] = valid.order_date - pd.to_timedelta(valid.order_date.dt.day_of_week, unit="D")
    observed = valid.groupby(["SKU_ID", "week_start"], as_index=False).quantity.sum()
    if not np.isclose(observed.quantity.sum(), valid.quantity.sum()):
        _fail("Weekly aggregation did not preserve gross completed units", report)
    weeks = pd.date_range(start, end, freq="W-MON")
    complete, coverage_files = _coverage(root, weeks, config, report)
    # First observed positive sale is causal; no last-sale/future-active-date cohort filtering.
    first_seen = observed.groupby("SKU_ID").week_start.min().rename("first_seen_week")
    grid = pd.MultiIndex.from_product([first_seen.index, weeks], names=["SKU_ID", "week_start"]).to_frame(index=False)
    grid["first_seen_week"] = grid.SKU_ID.map(first_seen)
    grid = grid.loc[grid.week_start.ge(grid.first_seen_week)].copy()
    grid = grid.merge(observed.rename(columns={"quantity": "observed_quantity"}), on=["SKU_ID", "week_start"], how="left", validate="one_to_one")
    grid["is_complete"] = grid.week_start.map(complete).astype(bool)
    grid["quantity"] = grid.observed_quantity.fillna(0.0).where(grid.is_complete, np.nan)
    panel = grid.sort_values(["SKU_ID", "week_start"]).reset_index(drop=True)
    report.update({"status": "passed", "valid_sales_lines": len(valid), "gross_completed_units": float(valid.quantity.sum()),
                   "observed_sku_weeks": len(observed), "sku_count": int(first_seen.size), "panel_rows": len(panel),
                   "complete_calendar_weeks": int(complete.sum()), "incomplete_calendar_weeks": int((~complete).sum()),
                   "missing_panel_labels": int(panel.quantity.isna().sum()),
                   "complete_panel_units": float(panel.quantity.sum()),
                   "observed_units_in_incomplete_weeks": float(panel.loc[~panel.is_complete, "observed_quantity"].sum()),
                   "zero_policy": "zero only after first observed sale and complete global feed; SKU lifecycle/stock availability unverified"})
    files = [root / name for name in ("SalesOrders.csv", "SalesOrderLines.csv", "ProductVariants.csv")] + coverage_files
    return WeeklyData(panel, report, files)
