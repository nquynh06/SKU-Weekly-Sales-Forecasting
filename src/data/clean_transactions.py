"""
Outputs:
    data/processed/clean_sales_transactions.csv
    reports/tables/transaction_cleaning_report.csv
    reports/tables/excluded_sales_lines.csv

Policy:
- Keep only Completed / Complete orders (canonical status = completed).
- Remove *exact* duplicate records before any join.
- Quarantine (do not arbitrarily choose) conflicting duplicate IDs.
- Exclude missing keys, invalid OrderDate, non-numeric or nonpositive Quantity.
- Do not exclude rows solely for zero price or missing UnitCostAtSale.
- Preserve IDs as strings and log each excluded source row with its reason.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ORDERS_REQUIRED = {"SalesOrderID", "OrderDate", "OrderStatus"}
LINES_REQUIRED = {"SalesOrderLineID", "SalesOrderID", "SKU_ID", "Quantity"}
STATUS_MAP = {
    "complete": "completed",
    "completed": "completed",
    "canceled": "cancelled",
    "cancelled": "cancelled",
}


def find_csv(folder: Path, stem: str) -> Path:
    """Accept exactly one case-insensitive file under the supplied data folder."""
    hits = [p for p in folder.rglob("*.csv") if p.stem.casefold() == stem.casefold()]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {stem}.csv inside {folder}, found {len(hits)}: {hits}"
        )
    return hits[0]


def load_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype="string", encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype="string", encoding="cp1252", low_memory=False)
    df.columns = df.columns.str.strip()
    df.insert(0, "_SourceRow", np.arange(2, len(df) + 2))  # CSV header occupies row 1
    return df


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_id(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.eq(""))


def parse_dates(series: pd.Series) -> pd.Series:
    # B-Tech dates can be exported with ISO or mixed text formats.
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):  # older pandas before format='mixed'
        return pd.to_datetime(series, errors="coerce")


def duplicate_drop(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop exact duplicates based on original CSV fields (excluding _SourceRow)."""
    duplicate = df.duplicated(subset=[c for c in df.columns if c != "_SourceRow"])
    return df.loc[~duplicate].copy(), df.loc[duplicate].copy()


def append_reason(reasons: pd.Series, mask: pd.Series, value: str) -> None:
    """The first failed check wins; each source line is counted once."""
    reasons.loc[reasons.eq("") & mask.fillna(False)] = value


def clean_transactions(
    orders_raw: pd.DataFrame,
    lines_raw: pd.DataFrame,
    *,
    min_date: str | None = None,
    max_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (clean lines, excluded raw lines, metrics table). No files are written."""
    for name, df, expected in (
        ("SalesOrders", orders_raw, ORDERS_REQUIRED),
        ("SalesOrderLines", lines_raw, LINES_REQUIRED),
    ):
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    # 1. Deduplicate order *rows*, then quarantine conflicting order IDs.
    orders, duplicate_orders = duplicate_drop(orders_raw)
    orders["SalesOrderID"] = normalize_id(orders["SalesOrderID"])
    conflict_mask = orders["SalesOrderID"].notna() & orders["SalesOrderID"].duplicated(keep=False)
    conflicting_order_ids = set(orders.loc[conflict_mask, "SalesOrderID"].tolist())
    orders = orders.loc[~conflict_mask & orders["SalesOrderID"].notna()].copy()
    if orders["SalesOrderID"].duplicated().any():
        raise AssertionError("SalesOrderID must be unique after conflict quarantine")

    orders["OrderDate"] = parse_dates(orders["OrderDate"])
    orders["OrderStatusCanonical"] = (
        orders["OrderStatus"].astype("string").str.strip().str.lower().map(STATUS_MAP)
    )
    orders["_OrderExists"] = True
    order_cols = ["SalesOrderID", "OrderDate", "OrderStatus", "OrderStatusCanonical", "_OrderExists"]

    # 2. Deduplicate sales-line rows and quarantine conflicting line IDs.
    lines, duplicate_lines = duplicate_drop(lines_raw)
    for col in ("SalesOrderLineID", "SalesOrderID", "SKU_ID"):
        lines[col] = normalize_id(lines[col])
    duplicate_line_pk = lines["SalesOrderLineID"].notna() & lines["SalesOrderLineID"].duplicated(keep=False)
    lines["Quantity"] = pd.to_numeric(lines["Quantity"], errors="coerce")
    lines = lines.merge(
        orders[order_cols], on="SalesOrderID", how="left", validate="many_to_one", sort=False
    )
    if len(lines) + len(duplicate_lines) != len(lines_raw):
        raise AssertionError("Order join unexpectedly changed the line count")

    # 3. Classify exclusions; do not silently drop records.
    reason = pd.Series("", index=lines.index, dtype="string")
    append_reason(reason, lines["SalesOrderLineID"].isna(), "missing_sales_order_line_id")
    append_reason(reason, duplicate_line_pk.reset_index(drop=True), "conflicting_sales_order_line_id")
    append_reason(reason, lines["SalesOrderID"].isna(), "missing_sales_order_id")
    append_reason(reason, lines["SKU_ID"].isna(), "missing_sku_id")
    append_reason(reason, lines["Quantity"].isna() | lines["Quantity"].le(0), "invalid_quantity")
    append_reason(reason, lines["SalesOrderID"].isin(conflicting_order_ids), "conflicting_sales_order_id")
    append_reason(reason, lines["_OrderExists"].isna(), "missing_parent_sales_order")
    append_reason(reason, lines["OrderDate"].isna(), "invalid_order_date")
    append_reason(reason, lines["OrderStatusCanonical"].isna(), "unknown_order_status")
    append_reason(reason, lines["OrderStatusCanonical"].eq("cancelled"), "cancelled_order")

    if min_date is not None:
        parsed_min = pd.Timestamp(min_date)
        append_reason(reason, lines["OrderDate"].dt.normalize().lt(parsed_min), "before_min_date")
    if max_date is not None:
        parsed_max = pd.Timestamp(max_date)
        append_reason(reason, lines["OrderDate"].dt.normalize().gt(parsed_max), "after_max_date")

    lines["_ExcludeReason"] = reason
    excluded = lines.loc[lines["_ExcludeReason"].ne("")].copy()
    valid = lines.loc[lines["_ExcludeReason"].eq("")].copy()

    if not duplicate_lines.empty:
        exact = duplicate_lines.copy()
        exact["_ExcludeReason"] = "exact_duplicate_sales_line"
        excluded = pd.concat([excluded, exact], ignore_index=True, sort=False)

    # 4. Enrich valid transactions only with *past/observed* order metadata.
    valid["WeekStart"] = (
        valid["OrderDate"].dt.normalize()
        - pd.to_timedelta(valid["OrderDate"].dt.dayofweek, unit="D")
    )
    if "ActualSellingPrice" in valid:
        price = pd.to_numeric(valid["ActualSellingPrice"], errors="coerce")
        valid["ZeroActualPriceFlag"] = price.eq(0).fillna(False).astype(bool)
        valid["InvalidActualPriceFlag"] = (price.isna() | price.lt(0)).fillna(True).astype(bool)
    if "UnitCostAtSale" in valid:
        valid["MissingUnitCostFlag"] = valid["UnitCostAtSale"].isna().astype(bool)

    # Keep original field names for compatibility with earlier notebooks.
    internal = ["_OrderExists", "_ExcludeReason"]
    valid = valid.drop(columns=[c for c in internal if c in valid])
    valid = valid.sort_values(["OrderDate", "SalesOrderID", "SalesOrderLineID"], kind="stable")
    valid = valid.reset_index(drop=True)
    excluded = excluded.sort_values("_SourceRow", kind="stable").reset_index(drop=True)

    if len(valid) + len(excluded) != len(lines_raw):
        raise AssertionError("Row reconciliation failed: input != valid + excluded")
    if valid["SalesOrderLineID"].duplicated().any():
        raise AssertionError("Duplicate line ID survived cleaning")
    if not valid["OrderStatusCanonical"].eq("completed").all():
        raise AssertionError("Noncompleted order survived cleaning")
    if valid["Quantity"].isna().any() or valid["Quantity"].le(0).any():
        raise AssertionError("Invalid Quantity survived cleaning")

    def add_metric(key: str, value: object, notes: str = "") -> dict:
        return {"metric": key, "value": value, "notes": notes}

    metrics = [
        add_metric("orders_raw_rows", len(orders_raw)),
        add_metric("orders_exact_duplicates_removed", len(duplicate_orders)),
        add_metric("orders_conflicting_key_rows", int(conflict_mask.sum()), "After exact deduplication"),
        add_metric("orders_conflicting_distinct_ids", len(conflicting_order_ids)),
        add_metric("orders_unique_after_quarantine", len(orders)),
        add_metric("orders_completed_after_normalization", int(orders["OrderStatusCanonical"].eq("completed").sum())),
        add_metric("sales_lines_raw_rows", len(lines_raw)),
        add_metric("sales_lines_exact_duplicates_removed", len(duplicate_lines)),
        add_metric("sales_lines_conflicting_key_rows", int(duplicate_line_pk.sum())),
        add_metric("sales_lines_accepted", len(valid)),
        add_metric("sales_lines_excluded", len(excluded)),
        add_metric("gross_quantity_accepted", float(valid["Quantity"].sum()), "Gross units; returns NOT subtracted"),
        add_metric("output_week_rule", "Monday-Sunday"),
        add_metric("output_date_filter", f"{min_date or 'none'} to {max_date or 'none'}"),
    ]
    for item, count in excluded["_ExcludeReason"].value_counts().items():
        metrics.append(add_metric(f"excluded_reason:{item}", int(count)))

    return valid, excluded, pd.DataFrame(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="clean B-Tech gross sales transactions")
    parser.add_argument("--data-dir", type=Path, default=Path("src/data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--min-date", default=None, help="Optional inclusive YYYY-MM-DD")
    parser.add_argument("--max-date", default=None, help="Optional inclusive YYYY-MM-DD")
    parser.add_argument("--parquet", action="store_true", help="Also write Parquet (requires pyarrow)")
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Data directory not found: {args.data_dir}")
    if args.min_date and args.max_date and pd.Timestamp(args.min_date) > pd.Timestamp(args.max_date):
        parser.error("--min-date cannot be after --max-date")

    orders_path = find_csv(args.data_dir, "SalesOrders")
    lines_path = find_csv(args.data_dir, "SalesOrderLines")
    orders, lines = load_csv(orders_path), load_csv(lines_path)
    clean, excluded, report = clean_transactions(
        orders, lines, min_date=args.min_date, max_date=args.max_date
    )
    report = pd.concat([report, pd.DataFrame([
        {"metric": "sales_orders_sha256", "value": digest(orders_path), "notes": str(orders_path)},
        {"metric": "sales_order_lines_sha256", "value": digest(lines_path), "notes": str(lines_path)},
    ])], ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    clean_path = args.out_dir / "clean_sales_transactions.csv"
    report_path = args.report_dir / "transaction_cleaning_report.csv"
    excluded_path = args.report_dir / "excluded_sales_lines.csv"
    clean.to_csv(clean_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    if args.parquet:
        try:
            clean.to_parquet(args.out_dir / "clean_sales_transactions.parquet", index=False)
        except ImportError as exc:
            raise SystemExit("To write Parquet, install: pip install pyarrow") from exc

    print(f"Orders (raw): {len(orders):,}")
    print(f"Sales lines (raw): {len(lines):,}")
    print(f"Valid sales lines: {len(clean):,}; excluded: {len(excluded):,}")
    print(f"Gross sales quantity: {clean['Quantity'].sum():,.3f}")
    print("Outputs:")
    print(f" - {clean_path}")
    print(f" - {report_path}")
    print(f" - {excluded_path}")

if __name__ == "__main__":
    main()
