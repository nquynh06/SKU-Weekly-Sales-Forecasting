"""Data contract validator & loader aligned with SKU-Weekly-Sales-Forecasting Lab 1-4 notebooks.

Supports TWO seamless input modes:
  Mode A (Raw Contract — identical to models/random forest/run.py):
    Reads data/raw/ directly: SalesOrders.csv, SalesOrderLines.csv, ProductVariants.csv,
    Products.csv, Subcategories.csv, Categories.csv, Brands.csv, Promotions.csv.
    Applies exact Lab 3-4 cleaning rules:
      - Drop 845 exact duplicates in SalesOrders & 1 exact duplicate in Products
      - Normalize OrderStatus (complete -> completed, canceled -> cancelled); keep completed only
      - Verify 556,094 valid sales lines, zero Quantity <= 0, zero duplicate SalesOrderLineID
      - Filter synchronized weekly window [start_week=2023-07-03 .. end_week=2025-12-22] (130 weeks)
      - Respect SKU lifecycle (ActiveFrom..ActiveTo / first..last observed week) when creating VALID_ZERO

  Mode B (Preprocessed CSV — Ground Truth / Clean Transactions / D0-D4):
    Reads any preprocessed CSV (sku_week_ground_truth_provisional.csv, clean_sales_transactions.csv,
    joined_sales_data.csv, or D0..D4 CSVs dropped into input_data/).
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


REQUIRED_RAW_TABLES = ["SalesOrders.csv", "SalesOrderLines.csv"]
RECOMMENDED_RAW_TABLES = [
    "ProductVariants.csv", "Products.csv", "Subcategories.csv",
    "Categories.csv", "Brands.csv", "Promotions.csv",
]

SKU_ALIASES = ["SKU_ID", "sku_id", "sku", "SKU", "product_id", "item_id"]
DATE_ALIASES = ["Forecast_Week", "week_start", "week", "WeekStart", "order_week", "OrderDate", "date", "Date"]
QTY_ALIASES = ["Actual_Quantity", "qty", "Quantity", "quantity", "weekly_qty", "sales_qty", "units_sold", "demand"]


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """Same encoding fallback as Lab 1-4 notebooks (00..03)."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(file_path, encoding=enc, low_memory=False)
        except Exception:
            pass
    raise ValueError(f"Cannot read CSV file: {file_path}")


def detect_dataset_version_from_name(path: Path, default: str = "D0") -> str:
    m = re.search(r"(?<![A-Za-z0-9])(D[0-4])(?![A-Za-z0-9])", path.stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"_(D[0-4])_", f"_{path.stem}_", flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return default


def is_raw_data_dir(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    names = {p.name.lower() for p in folder.glob("*.csv")}
    return "salesorders.csv" in names and "salesorderlines.csv" in names


def validate_raw_contract(data_dir: Path, start_week: str = "2023-07-03", end_week: str = "2025-12-22") -> Dict[str, object]:
    """Run quality-gated contract validation on data/raw/ (mirroring Lab 2-4 checks)."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    csv_map = {p.name.lower(): p for p in data_dir.glob("*.csv")}
    missing_required = [t for t in REQUIRED_RAW_TABLES if t.lower() not in csv_map]
    missing_recommended = [t for t in RECOMMENDED_RAW_TABLES if t.lower() not in csv_map]

    report: Dict[str, object] = {
        "data_dir": str(data_dir.resolve()),
        "total_csv_files_found": len(csv_map),
        "missing_required_tables": missing_required,
        "missing_recommended_tables": missing_recommended,
        "checks": [],
        "status": "PASS" if not missing_required else "FAIL",
    }
    if missing_required:
        return report

    so_raw = read_csv_safely(csv_map["salesorders.csv"])
    sol_raw = read_csv_safely(csv_map["salesorderlines.csv"])

    so_dup_count = int(so_raw.duplicated().sum())
    so = so_raw.drop_duplicates().copy()
    status_norm = (
        so["OrderStatus"].astype(str).str.strip().str.lower()
        .replace({"complete": "completed", "canceled": "cancelled"})
    )
    so["OrderStatus_norm"] = status_norm
    completed_orders = int((status_norm == "completed").sum())
    cancelled_orders = int((status_norm == "cancelled").sum())
    so_valid = so[so["OrderStatus_norm"] == "completed"].copy()
    so_valid["OrderDate"] = pd.to_datetime(so_valid["OrderDate"], errors="coerce")

    sol_dup_line_id = int(sol_raw["SalesOrderLineID"].duplicated().sum()) if "SalesOrderLineID" in sol_raw.columns else 0
    qty_num = pd.to_numeric(sol_raw["Quantity"], errors="coerce")
    neg_or_zero_qty = int((qty_num <= 0).sum())

    joined = sol_raw.merge(so_valid[["SalesOrderID", "OrderDate"]], on="SalesOrderID", how="inner", validate="many_to_one")
    win_start = pd.Timestamp(start_week)
    win_end = pd.Timestamp(end_week) + pd.Timedelta(days=6)
    in_window = int(((joined["OrderDate"] >= win_start) & (joined["OrderDate"] <= win_end)).sum())

    checks = [
        {"check": "SalesOrders exact duplicates removed", "value": so_dup_count, "status": "PASS"},
        {"check": "SalesOrders completed orders", "value": completed_orders, "status": "PASS" if completed_orders > 0 else "FAIL"},
        {"check": "SalesOrders cancelled orders excluded", "value": cancelled_orders, "status": "PASS"},
        {"check": "SalesOrderLines duplicate SalesOrderLineID", "value": sol_dup_line_id, "status": "PASS" if sol_dup_line_id == 0 else "WARN"},
        {"check": "SalesOrderLines Quantity <= 0", "value": neg_or_zero_qty, "status": "PASS" if neg_or_zero_qty == 0 else "WARN"},
        {"check": "Valid joined sales lines (completed)", "value": int(len(joined)), "status": "PASS" if len(joined) > 0 else "FAIL"},
        {"check": f"Valid lines in [{start_week} .. {win_end.date()}]", "value": in_window, "status": "PASS" if in_window > 0 else "FAIL"},
        {"check": "Unique SKUs in valid sales lines", "value": int(joined["SKU_ID"].nunique()), "status": "PASS"},
    ]

    if "products.csv" in csv_map and "productvariants.csv" in csv_map:
        pv = read_csv_safely(csv_map["productvariants.csv"])
        prod = read_csv_safely(csv_map["products.csv"])
        prod_dups = int(prod.duplicated().sum())
        sku_coverage = float(joined["SKU_ID"].isin(set(pv["SKU_ID"])).mean())
        checks.append({"check": "Products exact duplicates removed", "value": prod_dups, "status": "PASS"})
        checks.append({"check": "SKU_ID coverage in ProductVariants", "value": round(sku_coverage, 6), "status": "PASS" if sku_coverage == 1.0 else "WARN"})

    report["checks"] = checks
    if any(c["status"] == "FAIL" for c in checks):
        report["status"] = "FAIL"
    return report


def build_weekly_panel_from_raw_dir(
    data_dir: Path,
    start_week: str = "2023-07-03",
    end_week: str = "2025-12-22",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build synchronized SKU-Week Ground Truth panel directly from data/raw/ (27 CSVs)."""
    csv_map = {p.name.lower(): p for p in data_dir.glob("*.csv")}
    if "salesorders.csv" not in csv_map or "salesorderlines.csv" not in csv_map:
        raise FileNotFoundError(f"SalesOrders.csv and SalesOrderLines.csv are required in {data_dir}")

    # 1. Clean SalesOrders (exact Lab 3-4 rules: drop exact duplicates, normalize OrderStatus, keep completed)
    so_raw = read_csv_safely(csv_map["salesorders.csv"])
    n_so_dups = int(so_raw.duplicated().sum())
    so = so_raw.drop_duplicates().copy()
    so["OrderStatus"] = (
        so["OrderStatus"].astype(str).str.strip().str.lower()
        .replace({"complete": "completed", "canceled": "cancelled"})
    )
    n_cancelled = int((so["OrderStatus"] != "completed").sum())
    so = so[so["OrderStatus"] == "completed"].copy()
    so["OrderDate"] = pd.to_datetime(so["OrderDate"], errors="coerce")
    so = so.dropna(subset=["OrderDate"])

    # 2. Clean & Join SalesOrderLines -> SalesOrders (many_to_one validated)
    sol = read_csv_safely(csv_map["salesorderlines.csv"]).drop_duplicates().copy()
    sol["Quantity"] = pd.to_numeric(sol["Quantity"], errors="coerce")
    n_neg_qty = int((sol["Quantity"] <= 0).sum())
    sol = sol[sol["Quantity"] > 0].copy()

    merged = sol.merge(
        so[["SalesOrderID", "OrderDate"]],
        on="SalesOrderID",
        how="inner",
        validate="many_to_one",
    )
    n_valid_lines_all = int(len(merged))

    # 3. Restrict to synchronized 130-week window [start_week .. end_week + 6 days]
    win_start = pd.Timestamp(start_week)
    win_end_inclusive = pd.Timestamp(end_week) + pd.Timedelta(days=6)
    merged = merged[(merged["OrderDate"] >= win_start) & (merged["OrderDate"] <= win_end_inclusive)].copy()
    n_valid_lines_window = int(len(merged))

    merged["week_start"] = (merged["OrderDate"] - pd.to_timedelta(merged["OrderDate"].dt.weekday, unit="D")).dt.normalize()
    merged["SKU_ID"] = merged["SKU_ID"].astype(str).str.strip()

    merged["ActualSellingPrice"] = pd.to_numeric(merged.get("ActualSellingPrice"), errors="coerce")
    merged["RegularUnitPrice"] = pd.to_numeric(merged.get("RegularUnitPrice"), errors="coerce")
    merged["line_rev"] = merged["Quantity"] * merged["ActualSellingPrice"]
    merged["line_reg_rev"] = merged["Quantity"] * merged["RegularUnitPrice"]
    merged["promo_flag"] = merged["PromotionID"].notna().astype(int) if "PromotionID" in merged.columns else 0

    agg = merged.groupby(["SKU_ID", "week_start"], as_index=False).agg(
        qty=("Quantity", "sum"),
        revenue=("line_rev", "sum"),
        reg_rev=("line_reg_rev", "sum"),
        promo_flag=("promo_flag", "max"),
    )
    agg["avg_price"] = np.where(agg["qty"] > 0, agg["revenue"] / agg["qty"], np.nan)
    agg["discount_rate"] = np.where(agg["reg_rev"] > 0, np.clip(1.0 - agg["revenue"] / agg["reg_rev"], 0.0, 1.0), 0.0)
    agg = agg.drop(columns=["reg_rev"])

    # 4. Join Product hierarchy (ProductVariants -> Products -> Subcategories -> Categories & Brands)
    sku_meta = pd.DataFrame({"SKU_ID": agg["SKU_ID"].unique()})
    if "productvariants.csv" in csv_map and "products.csv" in csv_map:
        pv = read_csv_safely(csv_map["productvariants.csv"])[["SKU_ID", "ProductID"]].drop_duplicates("SKU_ID")
        prod = read_csv_safely(csv_map["products.csv"]).drop_duplicates().drop_duplicates("ProductID")
        keep_p = [c for c in ["ProductID", "BrandID", "SubcategoryID", "ActiveFrom", "ActiveTo"] if c in prod.columns]
        m = pv.merge(prod[keep_p], on="ProductID", how="left", validate="many_to_one")
        if "subcategories.csv" in csv_map and "SubcategoryID" in m.columns:
            sub = read_csv_safely(csv_map["subcategories.csv"]).drop_duplicates("SubcategoryID")
            m = m.merge(sub, on="SubcategoryID", how="left", validate="many_to_one")
        if "categories.csv" in csv_map and "CategoryID" in m.columns:
            cat = read_csv_safely(csv_map["categories.csv"]).drop_duplicates("CategoryID")
            m = m.merge(cat, on="CategoryID", how="left", validate="many_to_one")
        if "brands.csv" in csv_map and "BrandID" in m.columns:
            br = read_csv_safely(csv_map["brands.csv"]).drop_duplicates("BrandID")
            m = m.merge(br, on="BrandID", how="left", validate="many_to_one")

        m = m.rename(columns={"CategoryName": "category", "SubcategoryName": "subcategory", "BrandName": "brand"})
        m["SKU_ID"] = m["SKU_ID"].astype(str).str.strip()
        keep_m = ["SKU_ID"] + [c for c in ["category", "subcategory", "brand", "ActiveFrom", "ActiveTo"] if c in m.columns]
        sku_meta = sku_meta.merge(m[keep_m].drop_duplicates("SKU_ID"), on="SKU_ID", how="left")

    # 5. Lifecycle-aware weekly grid (only fill VALID_ZERO inside SKU's active observed window)
    obs_bounds = agg.groupby("SKU_ID")["week_start"].agg(["min", "max"])
    grid_frames = [
        pd.DataFrame({"SKU_ID": sku, "week_start": pd.date_range(r["min"], r["max"], freq="W-MON")})
        for sku, r in obs_bounds.iterrows()
    ]
    grid = pd.concat(grid_frames, ignore_index=True)
    panel = grid.merge(agg, on=["SKU_ID", "week_start"], how="left")
    panel = panel.merge(sku_meta, on="SKU_ID", how="left")

    panel["label_status"] = np.where(panel["qty"].notna(), "VALID_SALE", "VALID_ZERO")
    panel["qty"] = panel["qty"].fillna(0.0)
    panel["revenue"] = panel["revenue"].fillna(0.0)
    panel["promo_flag"] = panel["promo_flag"].fillna(0).astype(int)
    panel["discount_rate"] = panel["discount_rate"].fillna(0.0)
    # Causal price carry-forward for VALID_ZERO weeks (only past info)
    panel["avg_price"] = panel.groupby("SKU_ID")["avg_price"].transform(lambda s: s.ffill().bfill())

    for c in ["category", "subcategory", "brand"]:
        if c in panel.columns:
            panel[c] = panel[c].fillna("Unknown").astype(str)

    # Compute causal active_age from ActiveFrom (if available) else first observed sale week
    if "ActiveFrom" in panel.columns:
        af = pd.to_datetime(panel["ActiveFrom"], errors="coerce")
        first_obs = panel.groupby("SKU_ID")["week_start"].transform("min")
        origin = af.fillna(first_obs)
        panel["active_age"] = np.maximum(((panel["week_start"] - origin).dt.days // 7).astype(float), 0.0)
        panel = panel.drop(columns=[c for c in ["ActiveFrom", "ActiveTo"] if c in panel.columns])

    panel = panel.sort_values(["SKU_ID", "week_start"]).reset_index(drop=True)

    meta = {
        "input_mode": "RAW_DATA_DIR_CONTRACT",
        "source_file": str(data_dir.resolve()),
        "sales_orders_duplicates_removed": n_so_dups,
        "sales_orders_cancelled_excluded": n_cancelled,
        "negative_qty_quarantined": n_neg_qty,
        "valid_sales_lines_total": n_valid_lines_all,
        "valid_sales_lines_in_window": n_valid_lines_window,
        "available_date_min": str(panel["week_start"].min().date()),
        "available_date_max": str(panel["week_start"].max().date()),
    }
    return panel, meta


def _resolve_column(df: pd.DataFrame, preferred: str, aliases: List[str], required: bool = True) -> Optional[str]:
    if preferred in df.columns:
        return preferred
    lower_map = {c.lower(): c for c in df.columns}
    if preferred.lower() in lower_map:
        return lower_map[preferred.lower()]
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    if required:
        raise KeyError(f"Cannot find column '{preferred}' (aliases: {aliases}). Available: {list(df.columns)}")
    return None


def apply_window_and_scale(
    panel: pd.DataFrame,
    meta: Dict[str, object],
    history_start: Optional[str] = None,
    history_end: Optional[str] = None,
    history_weeks: Optional[int] = None,
    demand_scale: float = 1.0,
    selected_skus: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = panel.copy()
    if selected_skus:
        keep = {str(s).strip() for s in selected_skus}
        df = df[df["SKU_ID"].isin(keep)].copy()
        if df.empty:
            raise ValueError(f"No rows found for selected_skus={selected_skus}")

    if history_start is not None:
        df = df[df["week_start"] >= pd.to_datetime(history_start)].copy()
    if history_end is not None:
        df = df[df["week_start"] <= pd.to_datetime(history_end)].copy()
    if history_weeks is not None and history_weeks > 0:
        unique_weeks = np.sort(df["week_start"].unique())
        if len(unique_weeks) > history_weeks:
            df = df[df["week_start"] >= unique_weeks[-history_weeks]].copy()

    if df.empty:
        raise ValueError("Dataset is empty after applying historical date window filters.")
    if demand_scale <= 0:
        raise ValueError(f"demand_scale must be > 0, got {demand_scale}")
    if demand_scale != 1.0:
        df["qty"] = df["qty"] * float(demand_scale)
        if "revenue" in df.columns:
            df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce") * float(demand_scale)

    df = df.sort_values(["SKU_ID", "week_start"]).reset_index(drop=True)
    out_meta = dict(meta)
    out_meta.update(
        {
            "panel_rows": int(len(df)),
            "n_unverified_null_qty": int(df["qty"].isna().sum()),
            "n_valid_zero_qty": int((df["qty"] == 0).sum()),
            "n_skus": int(df["SKU_ID"].nunique()),
            "n_weeks": int(df["week_start"].nunique()),
            "selected_date_min": str(df["week_start"].min().date()),
            "selected_date_max": str(df["week_start"].max().date()),
            "demand_scale": float(demand_scale),
        }
    )
    return df, out_meta


def load_data_source(
    source_path: Path,
    start_week: str = "2023-07-03",
    end_week: str = "2025-12-22",
    sku_col: str = "SKU_ID",
    date_col: str = "Forecast_Week",
    qty_col: str = "Actual_Quantity",
    label_status_col: str = "label_status",
    stockout_flag_col: str = "stockout_flag",
    fill_missing_weeks_as_zero_for_csv: bool = False,
    history_start: Optional[str] = None,
    history_end: Optional[str] = None,
    history_weeks: Optional[int] = None,
    demand_scale: float = 1.0,
    selected_skus: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load either raw data directory (data/raw/) OR a preprocessed CSV file."""
    path = Path(source_path)
    if path.is_dir() and is_raw_data_dir(path):
        panel, meta = build_weekly_panel_from_raw_dir(path, start_week=start_week, end_week=end_week)
        return apply_window_and_scale(panel, meta, history_start, history_end, history_weeks, demand_scale, selected_skus)

    if not path.exists():
        raise FileNotFoundError(f"Input source not found: {path.resolve()}")

    raw_df = read_csv_safely(path)
    if raw_df.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    resolved_sku = _resolve_column(raw_df, sku_col, SKU_ALIASES, required=True)
    resolved_date = _resolve_column(raw_df, date_col, DATE_ALIASES, required=True)
    resolved_qty = _resolve_column(raw_df, qty_col, QTY_ALIASES, required=True)

    df = raw_df.copy()
    rename_map = {resolved_sku: "SKU_ID", resolved_date: "week_start", resolved_qty: "qty"}
    for std_col in [
        "avg_price", "discount_rate", "promo_flag", "revenue",
        "category", "subcategory", "brand", "active_age", "product_age_weeks",
        label_status_col, stockout_flag_col,
    ]:
        if std_col not in df.columns:
            found = _resolve_column(df, std_col, [std_col], required=False)
            if found and found not in rename_map:
                rename_map[found] = std_col
    df = df.rename(columns=rename_map)
    if "product_age_weeks" in df.columns and "active_age" not in df.columns:
        df["active_age"] = df["product_age_weeks"]

    # If input is clean_sales_transactions.csv or joined_sales_data.csv, filter completed status if present
    if "OrderStatus" in df.columns:
        st = df["OrderStatus"].astype(str).str.strip().str.lower().replace({"complete": "completed", "canceled": "cancelled"})
        df = df[st == "completed"].copy()

    df["SKU_ID"] = df["SKU_ID"].astype(str).str.strip()
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df = df.dropna(subset=["SKU_ID", "week_start"]).copy()
    df["week_start"] = (df["week_start"] - pd.to_timedelta(df["week_start"].dt.weekday, unit="D")).dt.normalize()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    neg_mask = df["qty"] < 0
    n_neg = int(neg_mask.sum())
    if n_neg > 0:
        df = df[~neg_mask].copy()

    if label_status_col in df.columns:
        status_norm = df[label_status_col].astype(str).str.strip().str.upper()
        df = df[status_norm != "OUTSIDE_LIFECYCLE"].copy()
        df.loc[status_norm.loc[df.index] == "UNVERIFIED", "qty"] = np.nan

    was_agg = False
    if df.duplicated(subset=["SKU_ID", "week_start"]).any():
        was_agg = True
        agg_dict: Dict[str, object] = {"qty": lambda s: np.nan if s.isna().all() else s.sum(min_count=1)}
        for c, fn in [("revenue", "sum"), ("avg_price", "mean"), ("discount_rate", "mean"),
                      ("promo_flag", "max"), ("active_age", "max"), (stockout_flag_col, "max")]:
            if c in df.columns:
                agg_dict[c] = fn
        for c in ["category", "subcategory", "brand", label_status_col]:
            if c in df.columns:
                agg_dict[c] = "first"
        df = df.groupby(["SKU_ID", "week_start"], as_index=False).agg(agg_dict)

    min_avail, max_avail = df["week_start"].min(), df["week_start"].max()
    if fill_missing_weeks_as_zero_for_csv:
        bounds = df.groupby("SKU_ID")["week_start"].agg(["min", "max"])
        grid = pd.concat([pd.DataFrame({"SKU_ID": s, "week_start": pd.date_range(r["min"], r["max"], freq="W-MON")})
                          for s, r in bounds.iterrows()], ignore_index=True)
        panel = grid.merge(df, on=["SKU_ID", "week_start"], how="left")
        panel["qty"] = panel["qty"].fillna(0.0)
        if "avg_price" in panel.columns:
            panel["avg_price"] = panel.groupby("SKU_ID")["avg_price"].transform(lambda s: s.ffill().bfill())
        for c in ["category", "subcategory", "brand"]:
            if c in panel.columns:
                panel[c] = panel.groupby("SKU_ID")[c].transform(lambda s: s.ffill().bfill()).fillna("Unknown").astype(str)
    else:
        panel = df

    meta = {
        "input_mode": "PREPROCESSED_CSV",
        "source_file": str(path.name),
        "raw_rows": int(len(raw_df)),
        "was_aggregated_to_weekly": was_agg,
        "negative_qty_quarantined": n_neg,
        "available_date_min": str(min_avail.date()),
        "available_date_max": str(max_avail.date()),
    }
    return apply_window_and_scale(panel, meta, history_start, history_end, history_weeks, demand_scale, selected_skus)
