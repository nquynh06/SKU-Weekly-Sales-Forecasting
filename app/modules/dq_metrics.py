"""
Feature 3: Data Quality Dashboard Metrics
-----------------------------------------
Computes the 4 core DQ dimensions highlighted in Part 9:
  1. Completeness   = 1 - (Missing Required Cells / Total Required Cells)
  2. Consistency    = 1 - (Constraint Violations / Total Checks)
  3. Accuracy Proxy = 1 - (Rule Violations / Total Records Checked)
  4. Timeliness     = 1 - (Out-of-scope Records / Total Checked Records)
Plus 2 supplementary checks (Validity, Uniqueness) and prioritized alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class DQAlert:
    dimension: str
    severity: str       # "CRITICAL", "WARNING", "INFO"
    field: str
    affected_rows: int
    affected_pct: float
    message: str


@dataclass
class DQDashboardResult:
    scores: Dict[str, float]          # 0–100 (%)
    summary_df: pd.DataFrame
    missing_by_col_df: pd.DataFrame
    alerts: List[DQAlert] = field(default_factory=list)


def compute_dq_dashboard(
    df: pd.DataFrame,
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
) -> DQDashboardResult:
    """Evaluate 4 core + 2 supplementary DQ dimensions and generate alerts."""
    n = len(df)
    alerts: List[DQAlert] = []
    if n == 0:
        empty = {k: 0.0 for k in ["Completeness", "Consistency", "Accuracy", "Timeliness", "Validity", "Uniqueness"]}
        return DQDashboardResult(empty, pd.DataFrame(), pd.DataFrame(), alerts)

    # 1. Completeness
    candidate_cols = [
        c for c in [
            "SalesOrderID", "OrderDate", "WeekStart", "SKU_ID", "Quantity",
            "OrderStatus", "CustomerID", "SalesLocationID", "UnitCostAtSale",
            "RegularUnitPrice", "ActualSellingPrice",
        ]
        if c in df.columns
    ]
    if not candidate_cols:
        candidate_cols = list(df.columns)

    total_cells = n * len(candidate_cols)
    missing_cells = int(df[candidate_cols].isna().sum().sum())
    completeness = (1.0 - missing_cells / max(total_cells, 1)) * 100.0

    miss_rows = []
    for col in candidate_cols:
        m = int(df[col].isna().sum())
        pct = m / n * 100.0
        miss_rows.append({"Cột": col, "Số giá trị thiếu": m, "Tỷ lệ thiếu (%)": round(pct, 2)})
        if m > 0:
            alerts.append(
                DQAlert(
                    dimension="Completeness",
                    severity="CRITICAL" if pct >= 20 else "WARNING",
                    field=col,
                    affected_rows=m,
                    affected_pct=round(pct, 2),
                    message=f"Cột `{col}` thiếu {m:,} giá trị ({pct:.2f}%).",
                )
            )
    missing_by_col_df = pd.DataFrame(miss_rows).sort_values("Tỷ lệ thiếu (%)", ascending=False)

    # 2. Consistency
    cons_checks = 0
    cons_viols = 0
    if "OrderStatus" in df.columns:
        cons_checks += n
        non_canon = ~df["OrderStatus"].isin(["Completed", "Cancelled"])
        v = int(non_canon.sum())
        cons_viols += v
        if v > 0:
            variants = df.loc[non_canon, "OrderStatus"].value_counts(dropna=False).to_dict()
            alerts.append(
                DQAlert(
                    dimension="Consistency",
                    severity="WARNING",
                    field="OrderStatus",
                    affected_rows=v,
                    affected_pct=round(v / n * 100.0, 2),
                    message=f"Có {v:,} bản ghi trạng thái không đồng nhất: {variants}.",
                )
            )
    if {"ActualSellingPrice", "RegularUnitPrice"}.issubset(df.columns):
        cons_checks += n
        a_p = pd.to_numeric(df["ActualSellingPrice"], errors="coerce")
        r_p = pd.to_numeric(df["RegularUnitPrice"], errors="coerce")
        v = int((a_p > r_p * 1.001).fillna(False).sum())
        cons_viols += v
        if v > 0:
            alerts.append(
                DQAlert(
                    dimension="Consistency",
                    severity="WARNING",
                    field="ActualSellingPrice",
                    affected_rows=v,
                    affected_pct=round(v / n * 100.0, 2),
                    message=f"Có {v:,} bản ghi giá bán thực tế cao hơn giá niêm yết.",
                )
            )
    consistency = (1.0 - cons_viols / max(cons_checks, 1)) * 100.0 if cons_checks > 0 else 100.0

    # 3. Accuracy (Proxy via business rules)
    acc_checks = 0
    acc_viols = 0
    if "Quantity" in df.columns:
        acc_checks += n
        qty = pd.to_numeric(df["Quantity"], errors="coerce")
        v = int((qty.isna() | (qty <= 0)).sum())
        acc_viols += v
        if v > 0:
            alerts.append(
                DQAlert(
                    dimension="Accuracy",
                    severity="CRITICAL",
                    field="Quantity",
                    affected_rows=v,
                    affected_pct=round(v / n * 100.0, 2),
                    message=f"Phát hiện {v:,} bản ghi Quantity <= 0 hoặc không hợp lệ.",
                )
            )
    if "ActualSellingPrice" in df.columns:
        acc_checks += n
        p = pd.to_numeric(df["ActualSellingPrice"], errors="coerce")
        v = int((p.isna() | (p <= 0)).sum())
        acc_viols += v
        if v > 0:
            alerts.append(
                DQAlert(
                    dimension="Accuracy",
                    severity="CRITICAL",
                    field="ActualSellingPrice",
                    affected_rows=v,
                    affected_pct=round(v / n * 100.0, 2),
                    message=f"Có {v:,} bản ghi giá bán <= 0 hoặc bị lỗi.",
                )
            )
    accuracy = (1.0 - acc_viols / max(acc_checks, 1)) * 100.0 if acc_checks > 0 else 100.0

    # 4. Timeliness
    date_col = "OrderDate" if "OrderDate" in df.columns else ("WeekStart" if "WeekStart" in df.columns else None)
    if date_col:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        oob = int((dt.isna() | (dt < pd.Timestamp(start_date)) | (dt > pd.Timestamp(end_date))).sum())
        timeliness = (1.0 - oob / n) * 100.0
        if oob > 0:
            alerts.append(
                DQAlert(
                    dimension="Timeliness",
                    severity="WARNING",
                    field=date_col,
                    affected_rows=oob,
                    affected_pct=round(oob / n * 100.0, 2),
                    message=f"Có {oob:,} bản ghi ngày nằm ngoài phạm vi [{start_date}, {end_date}].",
                )
            )
    else:
        timeliness = 100.0

    # 5. Validity & 6. Uniqueness (supplementary)
    val_viols = 0
    val_checks = 0
    if "SKU_ID" in df.columns:
        val_checks += n
        val_viols += int((df["SKU_ID"].isna() | (df["SKU_ID"].astype(str).str.strip() == "")).sum())
    if date_col:
        val_checks += n
        val_viols += int(pd.to_datetime(df[date_col], errors="coerce").isna().sum())
    validity = (1.0 - val_viols / max(val_checks, 1)) * 100.0 if val_checks > 0 else 100.0

    dup_cnt = int(df.duplicated().sum())
    uniqueness = (1.0 - dup_cnt / n) * 100.0
    if dup_cnt > 0:
        alerts.append(
            DQAlert(
                dimension="Uniqueness",
                severity="WARNING",
                field="All Columns",
                affected_rows=dup_cnt,
                affected_pct=round(dup_cnt / n * 100.0, 2),
                message=f"Phát hiện {dup_cnt:,} dòng trùng lặp hoàn toàn.",
            )
        )

    scores = {
        "Completeness": round(completeness, 2),
        "Consistency": round(consistency, 2),
        "Accuracy": round(accuracy, 2),
        "Timeliness": round(timeliness, 2),
        "Validity": round(validity, 2),
        "Uniqueness": round(uniqueness, 2),
    }

    def _status(v: float) -> str:
        if v >= 95.0:
            return "Đạt (>=95%)"
        if v >= 85.0:
            return "Cảnh báo (85–95%)"
        return "Cần xử lý (<85%)"

    summary_df = pd.DataFrame(
        [
            {"Chiều CLDL": k, "Điểm (%)": v, "Đánh giá": _status(v)}
            for k, v in scores.items()
        ]
    )
    return DQDashboardResult(scores, summary_df, missing_by_col_df, alerts)