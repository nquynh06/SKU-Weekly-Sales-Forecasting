"""DS317 / B-Tech: Step 1 — audit raw CSVs without modifying them.

Usage (from repository root):
    python src/data/audit_sources.py --data-dir data/raw --out-dir reports/tables

Outputs:
    data_inventory.csv        one row per source table
    raw_quality_report.csv    missing, duplication, numeric/date/status checks
    key_relationship_report.csv    orphan foreign-key references

No ground truth labels are created in this stage.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

REQUIRED = ["SalesOrders", "SalesOrderLines", "ProductVariants", "Products"]
OPTIONAL = ["InventoryBalances", "InventoryMovements", "CustomerReturns"]
PRIMARY_KEYS = {
    "SalesOrders": "SalesOrderID",
    "SalesOrderLines": "SalesOrderLineID",
    "ProductVariants": "SKU_ID",
    "Products": "ProductID",
}
FOREIGN_KEYS = [
    ("SalesOrderLines", "SalesOrderID", "SalesOrders", "SalesOrderID"),
    ("SalesOrderLines", "SKU_ID", "ProductVariants", "SKU_ID"),
    ("ProductVariants", "ProductID", "Products", "ProductID"),
]
DATE_COLUMNS = {"OrderDate", "ActiveFrom", "ActiveTo"}
NUMERIC_COLUMNS = {"Quantity", "RegularUnitPrice", "ActualSellingPrice"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_csv(root: Path, stem: str) -> Path | None:
    matches = sorted(p for p in root.rglob("*.csv") if p.stem.casefold() == stem.casefold())
    if len(matches) > 1:
        raise ValueError(f"Có nhiều bản sao {stem}.csv: {matches}. Hãy chỉ giữ một bản nguồn.")
    return matches[0] if matches else None


def add_quality(rows: list[dict], name: str, check: str, count: int, total: int, details: str = "") -> None:
    rows.append({
        "table": name, "check": check,
        "affected_rows": int(count), "total_rows": int(total),
        "rate_pct": round(100 * count / total, 4) if total else None,
        "details": details,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm toán dữ liệu gốc B-Tech (chưa làm sạch)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Không tìm thấy thư mục: {args.data_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inventory: list[dict] = []
    quality: list[dict] = []
    loaded: dict[str, pd.DataFrame] = {}

    for name in REQUIRED + OPTIONAL:
        path = locate_csv(args.data_dir, name)
        if path is None:
            inventory.append({"table": name, "status": "MISSING_REQUIRED" if name in REQUIRED else "MISSING_OPTIONAL"})
            print(f"[{'THIẾU' if name in REQUIRED else 'TÙY CHỌN'}] {name}.csv")
            continue
        try:
            # Keep IDs as strings so leading zeros are preserved during key checks.
            df = pd.read_csv(path, dtype="string", encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype="string", encoding="cp1252", low_memory=False)
        loaded[name] = df
        n = len(df)
        inventory.append({
            "table": name, "status": "OK", "file": str(path),
            "rows": n, "columns": len(df.columns), "column_names": "; ".join(df.columns),
            "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
            "sha256": sha256(path),
        })
        print(f"[OK] {name}: {n:,} dòng, {len(df.columns)} cột")

        add_quality(quality, name, "exact_duplicate_rows", int(df.duplicated().sum()), n)
        pk = PRIMARY_KEYS.get(name)
        if pk and pk in df:
            add_quality(quality, name, f"duplicate_primary_key:{pk}",
                        int(df[pk].notna().sum() - df.loc[df[pk].notna(), pk].nunique()), n)
            add_quality(quality, name, f"missing_primary_key:{pk}", int(df[pk].isna().sum()), n)
        elif pk:
            add_quality(quality, name, f"required_column_missing:{pk}", n, n)

        for col in df.columns:
            missing = int(df[col].isna().sum())
            add_quality(quality, name, f"missing:{col}", missing, n)
            if col in NUMERIC_COLUMNS:
                vals = pd.to_numeric(df[col], errors="coerce")
                nonnumeric = int((df[col].notna() & vals.isna()).sum())
                add_quality(quality, name, f"non_numeric:{col}", nonnumeric, n)
                if col == "Quantity":
                    add_quality(quality, name, "nonpositive_quantity", int((vals <= 0).sum()), n)
                else:
                    add_quality(quality, name, f"negative_price:{col}", int((vals < 0).sum()), n)
                    add_quality(quality, name, f"zero_price:{col}", int((vals == 0).sum()), n)
            if col in DATE_COLUMNS:
                try:
                    dates = pd.to_datetime(df[col], errors="coerce", format="mixed")
                except (TypeError, ValueError):
                    dates = pd.to_datetime(df[col], errors="coerce")
                add_quality(quality, name, f"invalid_date:{col}",
                            int((df[col].notna() & dates.isna()).sum()), n,
                            f"min={dates.min()}; max={dates.max()}")

        if name == "SalesOrders" and "OrderStatus" in df:
            statuses = df["OrderStatus"].fillna("<MISSING>").str.strip().str.lower()
            for status, count in statuses.value_counts().items():
                add_quality(quality, name, f"order_status:{status}", int(count), n)

    rel_report: list[dict] = []
    for child, child_col, parent, parent_col in FOREIGN_KEYS:
        if child not in loaded or parent not in loaded:
            rel_report.append({"child": child, "child_key": child_col, "parent": parent,
                               "parent_key": parent_col, "status": "TABLE_MISSING"})
            continue
        cdf, pdf = loaded[child], loaded[parent]
        if child_col not in cdf or parent_col not in pdf:
            rel_report.append({"child": child, "child_key": child_col, "parent": parent,
                               "parent_key": parent_col, "status": "COLUMN_MISSING"})
            continue
        observed = cdf[child_col].dropna()
        orphan = ~observed.isin(pdf[parent_col].dropna())
        rel_report.append({"child": child, "child_key": child_col, "parent": parent,
                           "parent_key": parent_col, "status": "CHECKED",
                           "nonnull_child_rows": len(observed),
                           "orphan_rows": int(orphan.sum()),
                           "orphan_pct": round(float(orphan.mean() * 100), 4) if len(observed) else None,
                           "parent_duplicate_key_rows": int(pdf[parent_col].dropna().duplicated().sum())})

    pd.DataFrame(inventory).to_csv(args.out_dir / "data_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality).to_csv(args.out_dir / "raw_quality_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rel_report).to_csv(args.out_dir / "key_relationship_report.csv", index=False, encoding="utf-8-sig")
    print(f"\nHoàn tất kiểm tra. Báo cáo tại: {args.out_dir.resolve()}")
    missing_required = [n for n in REQUIRED if n not in loaded]
    if missing_required:
        print("CHƯA ĐỦ DỮ LIỆU ĐỂ QUA BƯỚC 1:", ", ".join(missing_required))
    else:
        print("Đủ 4 bảng bắt buộc để tiếp tục rà soát kết quả bước 1.")


if __name__ == "__main__":
    main()
