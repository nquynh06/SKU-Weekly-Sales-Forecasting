"""DS317 Lab 1-4 preprocessing with traceable quarantine outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clean_transactions import clean_transactions, load_csv


STATUS_ALIASES = {
    "complete": "completed",
    "completed": "completed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}


def normalize_status(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.map(STATUS_ALIASES).fillna("unknown")


def find_primary_key(table_name: str, columns: list[str]) -> str | None:
    candidates = [
        f"{table_name[:-1]}ID" if table_name.endswith("s") else f"{table_name}ID",
        f"{table_name}ID",
    ]
    exact = set(columns)
    for candidate in candidates:
        if candidate in exact:
            return candidate
    if table_name == "SalesOrderLines":
        return "SalesOrderLineID"
    if table_name == "SalesOrders":
        return "SalesOrderID"
    if table_name == "ProductVariants":
        return "SKU_ID"
    if table_name == "Products":
        return "ProductID"
    return None


def iqr_outliers(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 4:
        return pd.Series(False, index=series.index)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & ((numeric < lower) | (numeric > upper))


def preprocess(raw_dir: Path, processed_dir: Path, report_dir: Path) -> dict:
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = processed_dir / "quarantine"

    tables: dict[str, pd.DataFrame] = {}
    missing_rows: list[dict] = []
    duplicate_rows: list[dict] = []
    outlier_rows: list[dict] = []
    inconsistency_rows: list[dict] = []
    summary_rows: list[dict] = []

    clean_transaction_path = processed_dir / "clean_sales_transactions.csv"
    excluded_transaction_path = report_dir / "excluded_sales_lines.csv"
    transaction_report_path = report_dir / "transaction_cleaning_report.csv"

    orders_paths = [p for p in csv_paths if p.stem.casefold() == "salesorders"]
    line_paths = [p for p in csv_paths if p.stem.casefold() == "salesorderlines"]
    if orders_paths and line_paths:
        orders = load_csv(orders_paths[0])
        lines = load_csv(line_paths[0])
        clean_sales, excluded_sales, transaction_report = clean_transactions(
            orders,
            lines,
            min_date=None,
            max_date=None,
        )
        clean_sales.to_csv(
            clean_transaction_path,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
        )
        excluded_sales.to_csv(
            excluded_transaction_path,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
        )
        transaction_report.to_csv(
            transaction_report_path,
            index=False,
            encoding="utf-8-sig",
        )

    for path in csv_paths:
        table_name = path.stem
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if df.empty:
            continue
        before = len(df)
        df.columns = [str(col).strip() for col in df.columns]

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)

        for col in df.columns:
            if any(token in col.lower() for token in ("date", "datetime", "timestamp")):
                converted = pd.to_datetime(df[col], errors="coerce")
                present = df[col].notna().sum()
                if present and converted.notna().sum() / present >= 0.7:
                    df[col] = converted

        status_col = next((col for col in df.columns if "status" in col.lower()), None)
        if status_col:
            df[f"{status_col}_Canonical"] = normalize_status(df[status_col])

        missing = df.isna().sum()
        for col, count in missing.items():
            if int(count):
                missing_rows.append({
                    "table": table_name,
                    "column": col,
                    "missing_cells": int(count),
                    "row_count": before,
                    "missing_rate_pct": round(100 * int(count) / before, 4),
                })

        exact_mask = df.duplicated(keep="first")
        exact_duplicates = df.loc[exact_mask].copy()
        if not exact_duplicates.empty:
            exact_duplicates.insert(0, "_SourceTable", table_name)
            duplicate_rows.append(exact_duplicates.assign(_Issue="exact_duplicate").to_dict("records"))

        df = df.loc[~exact_mask].copy()

        primary_key = find_primary_key(table_name, df.columns.tolist())
        if primary_key:
            conflicting = df[primary_key].notna() & df[primary_key].duplicated(keep=False)
            if conflicting.any():
                rows = df.loc[conflicting].copy()
                for key_value in rows[primary_key].dropna().unique():
                    inconsistency_rows.append({
                        "table": table_name,
                        "check": "conflicting_primary_key",
                        "key": primary_key,
                        "key_value": str(key_value),
                        "affected_rows": int((rows[primary_key] == key_value).sum()),
                    })

        numeric_col_candidates = [
            col for col in df.columns
            if any(token in col.lower() for token in ("quantity", "price", "cost", "amount", "stock", "gb"))
        ]
        for col in numeric_col_candidates:
            mask = iqr_outliers(df[col])
            if mask.any():
                rows = df.loc[mask, [col]].copy()
                rows.insert(0, "_SourceTable", table_name)
                rows.insert(1, "_IssueColumn", col)
                outlier_rows.extend(rows.assign(_Issue="iqr_outlier").to_dict("records"))

        tables[table_name] = df
        summary_rows.append({
            "table": table_name,
            "rows_before": before,
            "rows_after_exact_duplicate_removal": len(df),
            "exact_duplicates_removed": int(exact_mask.sum()),
            "missing_cells": int(df.isna().sum().sum()),
            "columns": len(df.columns),
        })
        df.to_csv(processed_dir / f"{table_name}_cleaned.csv", index=False, encoding="utf-8-sig")

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(report_dir / "missing_values_report.csv", index=False, encoding="utf-8-sig")

    duplicate_dir = quarantine_dir / "duplicates"
    duplicate_dir.mkdir(parents=True, exist_ok=True)
    duplicate_flat = [
        record
        for table_records in duplicate_rows
        for record in table_records
    ]
    pd.DataFrame(duplicate_flat).to_csv(
        duplicate_dir / "exact_duplicates.csv", index=False, encoding="utf-8-sig"
    )

    outlier_dir = quarantine_dir / "outliers"
    outlier_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outlier_rows).to_csv(
        outlier_dir / "iqr_numeric_outliers.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(inconsistency_rows).to_csv(
        report_dir / "inconsistency_report.csv", index=False, encoding="utf-8-sig"
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(report_dir / "preprocessing_summary.csv", index=False, encoding="utf-8-sig")

    join_log: list[dict] = []
    joined_df = None
    join_config = [
        ("SalesOrderLines", "SalesOrders", "SalesOrderID", "SalesOrderID"),
        ("SalesOrderLines", "ProductVariants", "SKU_ID", "SKU_ID"),
        ("ProductVariants", "Products", "ProductID", "ProductID"),
    ]
    if all(name in tables for name in ("SalesOrderLines", "SalesOrders", "ProductVariants", "Products")):
        joined_df = tables["SalesOrderLines"].copy()
        if clean_transaction_path.is_file():
            accepted_ids = pd.read_csv(
                clean_transaction_path,
                usecols=["SalesOrderLineID"],
                dtype={"SalesOrderLineID": "string"},
            )["SalesOrderLineID"]
            joined_df = joined_df.loc[
                joined_df["SalesOrderLineID"].astype("string").isin(accepted_ids)
            ].copy()
        for left_name, right_name, left_key, right_key in join_config:
            right_df = tables[right_name].copy()
            if left_key not in joined_df.columns or right_key not in right_df.columns:
                inconsistency_rows.append({
                    "table": f"{left_name}->{right_name}",
                    "check": "join_key_missing",
                    "key": f"{left_key}/{right_key}",
                    "key_value": "",
                    "affected_rows": len(joined_df),
                })
                continue
            before = len(joined_df)
            joined_df = joined_df.merge(
                right_df,
                left_on=left_key,
                right_on=right_key,
                how="left",
                validate="many_to_one",
                suffixes=("", f"_{right_name}")
            )
            join_log.append({
                "left_table": left_name,
                "right_table": right_name,
                "left_key": left_key,
                "right_key": right_key,
                "rows_before": before,
                "rows_after": len(joined_df),
            })

        joined_df.to_csv(
            processed_dir / "joined_sales_data.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(join_log).to_csv(
        report_dir / "join_log.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(inconsistency_rows).to_csv(
        report_dir / "inconsistency_report.csv", index=False, encoding="utf-8-sig"
    )

    result = {
        "table_count": len(tables),
        "joined_rows": len(joined_df) if joined_df is not None else None,
        "joined_columns": len(joined_df.columns) if joined_df is not None else None,
        "missing_tables": int(missing_df["table"].nunique()) if not missing_df.empty else 0,
        "outlier_rows": len(outlier_rows),
        "exact_duplicate_rows": len(duplicate_flat),
    }
    (report_dir / "preprocessing_run.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data preprocessing for the basic DS317 lab")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()
    result = preprocess(args.raw_dir.resolve(), args.processed_dir.resolve(), args.report_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
