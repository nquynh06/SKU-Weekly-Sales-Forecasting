"""
Feature 2: Data Validation
--------------------------
Checks structure (required columns), data types, invalid dates,
duplicate records, non-standard OrderStatus, and anomalous/negative Quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


REQUIRED_RAW_COLS = ["SKU_ID", "OrderDate", "Quantity"]
REQUIRED_PANEL_COLS = ["SKU_ID", "WeekStart", "Quantity"]


@dataclass
class ValidationIssue:
    level: str        # "ERROR", "WARNING", "INFO"
    column: str
    rule: str
    affected_rows: int
    detail: str


@dataclass
class ValidationReport:
    is_valid: bool
    input_mode: str   # "transaction" | "sku_week" | "invalid"
    n_rows: int
    n_cols: int
    n_skus: int
    schema_table: pd.DataFrame
    issues: List[ValidationIssue] = field(default_factory=list)


def validate_sales_csv(df: pd.DataFrame) -> ValidationReport:
    """Validate uploaded sales CSV structure, dtypes, and anomalous values."""
    issues: List[ValidationIssue] = []

    if df is None or df.empty:
        issues.append(
            ValidationIssue("ERROR", "-", "Non-empty file", 0, "File CSV không có dữ liệu (0 dòng).")
        )
        return ValidationReport(False, "invalid", 0, 0, 0, pd.DataFrame(), issues)

    has_raw = all(c in df.columns for c in REQUIRED_RAW_COLS)
    has_panel = all(c in df.columns for c in REQUIRED_PANEL_COLS)

    if has_raw:
        mode = "transaction"
        date_col = "OrderDate"
    elif has_panel:
        mode = "sku_week"
        date_col = "WeekStart"
    else:
        missing = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
        issues.append(
            ValidationIssue(
                "ERROR",
                ", ".join(missing),
                "Required Columns",
                len(df),
                f"Thiếu cột bắt buộc: {missing}. Cần có (SKU_ID, OrderDate, Quantity) hoặc (SKU_ID, WeekStart, Quantity).",
            )
        )
        mode = "invalid"
        date_col = ""

    # Schema & dtype summary table
    schema_rows = []
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        schema_rows.append(
            {
                "Cột": col,
                "Kiểu dữ liệu": str(df[col].dtype),
                "Giá trị duy nhất": int(df[col].nunique(dropna=True)),
                "Số giá trị thiếu": n_null,
                "Tỷ lệ thiếu (%)": round(n_null / len(df) * 100, 2),
            }
        )
    schema_table = pd.DataFrame(schema_rows)

    if mode == "invalid":
        return ValidationReport(False, mode, len(df), len(df.columns), 0, schema_table, issues)

    # 1. SKU_ID check
    null_sku = int(df["SKU_ID"].isna().sum())
    if null_sku > 0:
        issues.append(
            ValidationIssue("ERROR", "SKU_ID", "Not Null", null_sku, f"Có {null_sku:,} dòng thiếu mã SKU_ID.")
        )

    # 2. Date parsing & out-of-range check
    dt = pd.to_datetime(df[date_col], errors="coerce")
    bad_dt = int(dt.isna().sum())
    if bad_dt > 0:
        issues.append(
            ValidationIssue("WARNING", date_col, "Date Format", bad_dt, f"Có {bad_dt:,} dòng {date_col} sai định dạng ngày.")
        )
    oob_dt = int(((dt < pd.Timestamp("2023-01-01")) | (dt > pd.Timestamp("2025-12-31"))).sum())
    if oob_dt > 0:
        issues.append(
            ValidationIssue(
                "WARNING",
                date_col,
                "Date Scope (2023–2025)",
                oob_dt,
                f"Có {oob_dt:,} dòng ngày nằm ngoài giai đoạn nghiên cứu 2023–2025.",
            )
        )

    # 3. Quantity type, negative & outlier check
    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    bad_qty = int(qty.isna().sum())
    neg_qty = int((qty < 0).sum())
    zero_qty = int((qty == 0).sum())
    if bad_qty > 0:
        issues.append(
            ValidationIssue("ERROR", "Quantity", "Numeric Type", bad_qty, f"Có {bad_qty:,} dòng Quantity không phải số.")
        )
    if neg_qty > 0:
        issues.append(
            ValidationIssue(
                "WARNING",
                "Quantity",
                "Negative Quantity Quarantine",
                neg_qty,
                f"Phát hiện {neg_qty:,} dòng Quantity âm — được cách ly kiểm tra (không tự chuyển thành 0).",
            )
        )
    if zero_qty > 0 and mode == "transaction":
        issues.append(
            ValidationIssue("INFO", "Quantity", "Zero Transaction", zero_qty, f"Có {zero_qty:,} dòng giao dịch có Quantity = 0.")
        )
    if qty.notna().sum() > 10:
        q99 = float(qty.quantile(0.99))
        extreme = int((qty > max(q99 * 3, 20)).sum())
        if extreme > 0:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    "Quantity",
                    "Extreme Outlier",
                    extreme,
                    f"Phát hiện {extreme:,} dòng Quantity vượt xa phân vị 99% (>{max(q99 * 3, 20):.1f}).",
                )
            )

    # 4. OrderStatus consistency check
    if "OrderStatus" in df.columns:
        non_std = ~df["OrderStatus"].isin(["Completed", "Cancelled"])
        n_non_std = int(non_std.sum())
        if n_non_std > 0:
            variants = df.loc[non_std, "OrderStatus"].value_counts(dropna=False).to_dict()
            issues.append(
                ValidationIssue(
                    "WARNING",
                    "OrderStatus",
                    "Status Standardization",
                    n_non_std,
                    f"Phát hiện {n_non_std:,} dòng trạng thái chưa chuẩn hóa ({variants}) — pipeline sẽ tự chuẩn hóa.",
                )
            )
        n_cancel = int(df["OrderStatus"].isin(["Cancelled", "Canceled"]).sum())
        if n_cancel > 0:
            issues.append(
                ValidationIssue(
                    "INFO",
                    "OrderStatus",
                    "Cancelled Orders",
                    n_cancel,
                    f"Có {n_cancel:,} dòng thuộc đơn hàng đã hủy — sẽ được loại khỏi sản lượng bán khi tiền xử lý.",
                )
            )

    # 5. Exact duplicate rows
    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        issues.append(
            ValidationIssue("WARNING", "All Columns", "Uniqueness", dup_rows, f"Phát hiện {dup_rows:,} dòng trùng lặp hoàn toàn.")
        )

    is_valid = not any(i.level == "ERROR" for i in issues)
    n_skus = int(df["SKU_ID"].nunique(dropna=True))
    return ValidationReport(is_valid, mode, len(df), len(df.columns), n_skus, schema_table, issues)