from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

REQUIRED_CANDIDATE_COLUMNS = {
    "SKU_ID", "ProductID", "WeekStart", "WeekEnd", "ProductWindowEvidence",
    "HasObservedSale", "SkuWeekVerifiedActive", "SalesLogCoverageVerified",
    "ZeroLabelAuthorized", "observed_gross_qty",
}
REQUIRED_LIFECYCLE_COLUMNS = {
    "SKU_ID", "ProductID", "ActiveFrom", "ActiveTo", "FirstObservedSale",
    "LastObservedSale", "DateConflictSalesLines",
}
OUTSIDE = {"before_active_from", "after_active_to"}
PRIORITY = {
    "within_recorded_product_window": "P1 - verify SKU active + complete logs",
    "open_end_unverified": "P1 - clarify open-ended activity + logs",
    "unverified_product_window": "P1 - investigate missing lifecycle metadata",
    "partial_week_overlap": "P1 - inspect partial lifecycle overlap",
    "before_active_from": "P2 - review out-of-lifecycle exclusion",
    "after_active_to": "P2 - review out-of-lifecycle exclusion",
}


def bool_col(df: pd.DataFrame, field: str) -> pd.Series:
    series = df[field]
    normalized = series.astype("string").str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0"})
    if not valid.all():
        raise ValueError(f"Invalid or blank boolean values in {field}: {(~valid).sum()}")
    return normalized.isin({"true", "1"})


def require(df: pd.DataFrame, fields: set[str], name: str) -> None:
    missing = sorted(fields - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports-dir", type=Path, default=Path("reports/tables"))
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    source = args.reports_dir
    output = args.out_dir or source
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = source / "sku_week_activity_candidates.csv"
    lifecycle_path = source / "sku_lifecycle_audit.csv"
    for path in (candidates_path, lifecycle_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")
    cand = pd.read_csv(candidates_path, dtype={"SKU_ID": "string", "ProductID": "string"}, low_memory=False)
    life = pd.read_csv(lifecycle_path, dtype={"SKU_ID": "string", "ProductID": "string"}, low_memory=False)
    require(cand, REQUIRED_CANDIDATE_COLUMNS, candidates_path.name)
    require(life, REQUIRED_LIFECYCLE_COLUMNS, lifecycle_path.name)

    has_sale = bool_col(cand, "HasObservedSale")
    active_verified = bool_col(cand, "SkuWeekVerifiedActive")
    log_verified = bool_col(cand, "SalesLogCoverageVerified")
    zero_auth = bool_col(cand, "ZeroLabelAuthorized")
    if cand.duplicated(["SKU_ID", "WeekStart"]).any():
        raise ValueError("Duplicate SKU_ID/WeekStart keys in the candidate panel")
    if life.duplicated(["SKU_ID"]).any():
        raise ValueError("Duplicate SKU_ID rows in sku_lifecycle_audit.csv")
    if (zero_auth & (~active_verified | ~log_verified | has_sale)).any():
        raise ValueError("Inconsistent ZeroLabelAuthorized flags detected; do not authorize zero labels")

    conflict_lines = pd.to_numeric(life["DateConflictSalesLines"], errors="coerce").fillna(0)
    conflict_skus = set(life.loc[conflict_lines.gt(0), "SKU_ID"])
    conflict_cols = [c for c in [
        "SKU_ID", "ProductID", "ActiveFrom", "ActiveTo", "FirstObservedSale",
        "LastObservedSale", "before_active_from", "after_active_to", "DateConflictSalesLines",
        "ReviewReason", "ObservedGrossQty",
    ] if c in life.columns]
    conflicts = life.loc[conflict_lines.gt(0), conflict_cols].copy()
    conflicts["review_status"] = "PENDING_REVIEW"
    conflicts["review_notes"] = ""
    conflicts.to_csv(output / "sku_date_conflicts_to_review.csv", index=False, encoding="utf-8-sig")

    no_sale = cand.loc[~has_sale].copy()
    no_sale["has_product_date_conflict"] = no_sale["SKU_ID"].isin(conflict_skus)
    no_sale["metadata_suggests_outside_window"] = no_sale["ProductWindowEvidence"].isin(OUTSIDE)
    no_sale["review_priority"] = no_sale["ProductWindowEvidence"].map(PRIORITY).fillna("P1 - investigate other metadata")
    no_sale.loc[no_sale["has_product_date_conflict"], "review_priority"] = (
        "P0 - resolve conflicting lifecycle dates before approval"
    )
    no_sale["proposed_action"] = no_sale["metadata_suggests_outside_window"].map({
        True: "REVIEW_OUT_OF_SCOPE (not zero)",
        False: "REVIEW_ACTIVITY_AND_LOG_COVERAGE (not zero)",
    })
    no_sale.loc[no_sale["has_product_date_conflict"], "proposed_action"] = "RESOLVE_DATE_CONFLICT_FIRST"
    no_sale["new_label_authorized"] = False  # Never infer from the current reports
    no_sale["manual_review_status"] = "PENDING"
    no_sale.to_csv(output / "no_sale_review_queue.csv", index=False, encoding="utf-8-sig")

    # Review template is intentionally restricted to rows not outside the metadata window.
    # Metadata boundaries are NOT assumed to be ground truth; outside-window rows remain
    # in the full queue for explicit out-of-scope review.
    actionable = no_sale.loc[~no_sale["metadata_suggests_outside_window"]].copy()
    template = actionable[["SKU_ID", "ProductID", "WeekStart", "WeekEnd", "ProductWindowEvidence", "review_priority"]].copy()
    template["sku_activity_evidence_uri"] = ""
    template["sku_activity_evidence_verified"] = ""
    template["complete_week_order_log_evidence_uri"] = ""
    template["complete_week_order_log_verified"] = ""
    template["reviewer"] = ""
    template["review_date"] = ""
    template["decision"] = "PENDING"  # Values: PENDING / ELIGIBLE_ZERO / UNVERIFIED / OUT_OF_SCOPE
    template["notes"] = ""
    template.to_csv(output / "zero_evidence_review_template.csv", index=False, encoding="utf-8-sig")

    counts = no_sale.groupby("ProductWindowEvidence", dropna=False).agg(
        sku_weeks=("SKU_ID", "size"), distinct_skus=("SKU_ID", "nunique"),
        date_conflict_sku_weeks=("has_product_date_conflict", "sum"),
    ).reset_index().sort_values("sku_weeks", ascending=False)
    counts["pct_of_no_sale"] = (counts["sku_weeks"] / len(no_sale) * 100).round(2) if len(no_sale) else 0
    counts.to_csv(output / "no_sale_review_summary.csv", index=False, encoding="utf-8-sig")

    readme = f"""# DS317 – Bước 8: Rà soát bằng chứng Ground Truth

**Trạng thái:** CHƯA PHÊ DUYỆT NHÃN 0 / KHÔNG PHẢI GROUND TRUTH CUỐI CÙNG.

## Kết quả từ báo cáo hiện có

- Tổng SKU–tuần chưa ghi nhận bán: **{len(no_sale):,}**.
- Ngoài khoảng hoạt động theo metadata sản phẩm (cần kiểm tra trước khi loại phạm vi): **{int(no_sale['metadata_suggests_outside_window'].sum()):,}**.
- Còn lại cần xác minh SKU hoạt động và độ đầy đủ logs: **{len(actionable):,}**.
- SKU có xung đột ngày bán so với ActiveFrom/ActiveTo: **{len(conflicts):,}**.
- Nhãn 0 mới được tự động phê duyệt: **0**.

## File đầu ra

- `sku_date_conflicts_to_review.csv`: ưu tiên xem xung đột ngày sản phẩm / giao dịch.
- `no_sale_review_queue.csv`: toàn bộ tuần không bán, kèm nhóm cần xem xét, **không gán 0**.
- `zero_evidence_review_template.csv`: các tuần không bán chưa nằm rõ ngoài cửa sổ metadata; mẫu để nhập bằng chứng.
- `no_sale_review_summary.csv`: số lượng và tỷ lệ theo loại bằng chứng.

## Bằng chứng cần có để chốt

1. **SKU hoạt động trong tuần:** bằng chứng cấp SKU (không chỉ `Products.ActiveFrom/ActiveTo`).
2. **Tuần giao dịch đầy đủ:** bằng chứng nguồn xuất đơn hàng đủ ngày/không thiếu batch, không suy từ việc SKU khác có đơn.
3. **Xung đột ngày:** giải trình trường hợp bán trước ngày bắt đầu hoặc sau ngày kết thúc sản phẩm.
4. **Quy tắc loại ngoài phạm vi:** có căn cứ xác nhận để loại những tuần thực sự trước/sau vòng đời, không để nhãn 0.

Chỉ cập nhật cờ `SkuWeekVerifiedActive` / `SalesLogCoverageVerified` sau khi có minh chứng và người kiểm duyệt phê duyệt. Sau đó cập nhật panel, chạy lại Bước 4–7. Nếu không có bằng chứng, giữ `NA` và bàn giao dưới nhãn *observed-sales provisional ground truth*.
"""
    (output / "ground_truth_evidence_review.md").write_text(readme, encoding="utf-8")

    print(f"No-sale SKU-weeks: {len(no_sale):,}")
    print(f"Metadata outside-window candidates: {int(no_sale['metadata_suggests_outside_window'].sum()):,}")
    print(f"In-window / unresolved candidates: {len(actionable):,}")
    print(f"Conflicting SKU dates: {len(conflicts):,}")
    print(f"Reports written to: {output.resolve()}")

if __name__ == "__main__":
    main()
