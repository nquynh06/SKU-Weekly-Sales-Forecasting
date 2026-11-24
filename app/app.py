"""
DS317 – SKU-Weekly Sales Forecasting & Data Quality Dashboard (Streamlit Demo)
Project: SKU-Level Weekly Sales Forecasting with Data Quality Assessment and Improvement
Repository: sku-weekly-sales-forecasting (app/app.py)

Kiến trúc bám sát Mục 9 đặc tả:
- Người dùng CHỈ tải lên dữ liệu CSV (giao dịch bán hàng & tùy chọn tồn kho).
- Các mô hình đã huấn luyện trước (MA4, Random Forest, HistGradientBoosting) và bảng kết quả
  đánh giá (D0–D4) được hệ thống TỰ ĐỘNG NẠP SẴN từ `app/models/` và `app/data/`
  (không yêu cầu người dùng upload model, không train/tuning lại khi demo).
- Giao diện thiết kế theo Nguyên tắc chữ Z (Top-Left KPI -> Top-Right Context -> Center Visual
  -> Bottom-Left Detail -> Bottom-Right Action/Export).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG, ĐƯỜNG DẪN TỰ ĐỘNG & CSS CHỮ Z
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DS317 | SKU-Weekly Sales Forecasting & DQ Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_SEARCH_DIRS = [APP_DIR / "models", PROJECT_ROOT / "models"]
DATA_SEARCH_DIRS = [APP_DIR / "data", PROJECT_ROOT / "reports" / "tables", PROJECT_ROOT / "data"]
SAMPLE_SEARCH_DIRS = [APP_DIR / "sample_data", PROJECT_ROOT / "sample_data"]

CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    /* Thanh Header chữ Z (Top-Left -> Top-Right) */
    .z-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        border-radius: 14px;
        padding: 1.25rem 1.6rem;
        color: #FFFFFF;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.10);
    }
    .z-header-title {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF;
    }
    .z-header-sub {
        font-size: 0.88rem;
        color: #CBD5E1;
        margin-top: 0.25rem;
    }
    .z-badge-group {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        align-items: center;
    }
    .z-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .z-pill-demo { background: rgba(251, 191, 36, 0.18); color: #FDE68A; border: 1px solid rgba(251, 191, 36, 0.4); }
    .z-pill-live { background: rgba(16, 185, 129, 0.20); color: #A7F3D0; border: 1px solid rgba(16, 185, 129, 0.45); }
    .z-pill-info { background: rgba(255, 255, 255, 0.12); color: #E2E8F0; border: 1px solid rgba(255, 255, 255, 0.2); }

    /* Thẻ KPI chữ Z */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #1D4ED8;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .kpi-card.emerald { border-top-color: #059669; }
    .kpi-card.amber   { border-top-color: #D97706; }
    .kpi-card.rose    { border-top-color: #DC2626; }
    .kpi-card.slate   { border-top-color: #475569; }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
        overflow-wrap: break-word;
    }
    .kpi-delta {
        font-size: 0.80rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }
    .delta-pos { color: #059669; }
    .delta-neg { color: #DC2626; }
    .delta-neu { color: #64748B; }

    /* Tiêu đề & mô tả bên trong st.container(border=True) */
    .z-card-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.15rem;
    }
    .z-card-desc {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 0.75rem;
    }

    /* Nền trắng cho st.container(border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        padding: 0.25rem 0.35rem;
    }

    /* Khối CTA góc Bottom-Right (Điểm kết thúc chữ Z) */
    .z-cta-box {
        background: linear-gradient(180deg, #EFF6FF 0%, #FFFFFF 100%);
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .z-cta-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.35rem;
    }
    .z-cta-text {
        font-size: 0.83rem;
        color: #334155;
        line-height: 1.45;
        margin-bottom: 0.35rem;
    }

    /* Thẻ trạng thái mô hình nạp sẵn ở Sidebar */
    .model-status-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        font-size: 0.82rem;
        line-height: 1.6;
        color: #334155;
    }
    .dot-ok { color: #059669; font-weight: 700; }
    .dot-warn { color: #D97706; font-weight: 700; }

    /* Tinh chỉnh Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        background-color: #FFFFFF;
        padding: 0.4rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.55rem 1.05rem;
        font-weight: 600;
        font-size: 0.88rem;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

FEATURE_COLS = [
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
    "month",
    "week_of_year",
    "active_age",
]

VALID_START_DATE = pd.Timestamp("2023-01-01")
VALID_END_DATE = pd.Timestamp("2025-12-31")

STATUS_MAP = {
    "complete": "Completed",
    "completed": "Completed",
    "done": "Completed",
    "delivered": "Completed",
    "canceled": "Cancelled",
    "cancelled": "Cancelled",
    "cancel": "Cancelled",
    "pending": "Pending",
    "processing": "Processing",
    "returned": "Returned",
}


# ---------------------------------------------------------------------------
# 2. TỰ ĐỘNG NẠP MÔ HÌNH ĐÃ HUẤN LUYỆN & BẢNG KẾT QUẢ TỪ HỆ THỐNG (BUILT-IN)
# ---------------------------------------------------------------------------
def _safe_load_artifact(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _find_first_file(dirs: List[Path], patterns: List[str]) -> Optional[Path]:
    for d in dirs:
        if not d.exists():
            continue
        for pat in patterns:
            matches = sorted(d.glob(pat))
            if matches:
                return matches[0]
    return None


@st.cache_resource
def load_builtin_models() -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    specs = [
        ("MA4 (Baseline)", ["*ma4*.pkl", "*ma4*.joblib", "*moving*average*.pkl", "*moving*average*.joblib"]),
        ("Random Forest", ["*random*forest*.pkl", "*random*forest*.joblib", "*rf*.pkl", "*rf*.joblib"]),
        ("HistGradientBoosting", ["*hist*gradient*.pkl", "*hist*gradient*.joblib", "*hgb*.pkl", "*hgb*.joblib"]),
    ]

    for label, patterns in specs:
        found_path = _find_first_file(MODEL_SEARCH_DIRS, patterns)
        if found_path is not None:
            try:
                obj = _safe_load_artifact(found_path)
                registry[label] = {
                    "model": obj,
                    "path": found_path,
                    "status": "loaded",
                    "detail": found_path.name,
                }
            except Exception as exc:
                registry[label] = {
                    "model": None,
                    "path": found_path,
                    "status": "error",
                    "detail": f"Lỗi nạp {found_path.name}: {exc}",
                }
        else:
            if label == "MA4 (Baseline)":
                registry[label] = {
                    "model": None,
                    "path": None,
                    "status": "builtin_formula",
                    "detail": "Công thức chuẩn MA4 (tích hợp sẵn)",
                }
            else:
                registry[label] = {
                    "model": None,
                    "path": None,
                    "status": "missing",
                    "detail": "Chưa tìm thấy file trong app/models/",
                }
    return registry


@st.cache_data
def load_builtin_metrics() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    found = _find_first_file(
        DATA_SEARCH_DIRS,
        ["*model_comparison*.csv", "*metrics*.csv", "*evaluation*.csv", "*.csv"],
    )
    if found is None:
        return None, None
    try:
        df = pd.read_csv(found)
        return df, str(found.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in found.parents else found.name
    except Exception:
        return None, None


def _predict_with_artifact(artifact: Any, target_rows: pd.DataFrame) -> np.ndarray:
    model_obj = artifact
    feat_cols = FEATURE_COLS

    if isinstance(artifact, dict):
        for key in ["model", "estimator", "regressor", "pipeline"]:
            if key in artifact and hasattr(artifact[key], "predict"):
                model_obj = artifact[key]
                break
        for fkey in ["feature_cols", "features", "feature_names"]:
            if fkey in artifact and isinstance(artifact[fkey], (list, tuple)):
                feat_cols = list(artifact[fkey])
                break

    if hasattr(model_obj, "feature_names_in_"):
        feat_cols = list(model_obj.feature_names_in_)

    X_pred = pd.DataFrame(index=target_rows.index)
    for col in feat_cols:
        if col in target_rows.columns:
            X_pred[col] = pd.to_numeric(target_rows[col], errors="coerce").fillna(0.0)
        else:
            X_pred[col] = 0.0

    if hasattr(model_obj, "predict"):
        return np.asarray(model_obj.predict(X_pred), dtype=float)

    window = int(artifact.get("window", 4)) if isinstance(artifact, dict) else 4
    col_ma = "rolling_mean_8" if window >= 8 else "rolling_mean_4"
    return target_rows[col_ma].fillna(0.0).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 3. TIỆN ÍCH HIỂN THỊ UI CHỮ Z
# ---------------------------------------------------------------------------
def render_kpi_card(label: str, value: str, delta_text: str = "", tone: str = "blue", delta_kind: str = "neu") -> None:
    tone_cls = "" if tone == "blue" else tone
    delta_cls = f"delta-{delta_kind}"
    delta_html = f'<div class="kpi-delta {delta_cls}">{delta_text}</div>' if delta_text else ""
    st.markdown(
        f"""
        <div class="kpi-card {tone_cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card_header(title: str, desc: str = "") -> None:
    desc_html = f'<div class="z-card-desc">{desc}</div>' if desc else ""
    st.markdown(f'<div class="z-card-title">{title}</div>{desc_html}', unsafe_allow_html=True)


def style_matplotlib_ax(ax: plt.Axes, title: str = "", ylabel: str = "", xlabel: str = "") -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E1")
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color="#0F172A", loc="left", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color="#475569")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color="#475569")


# ---------------------------------------------------------------------------
# 4. DỮ LIỆU MẪU TÍCH HỢP (DÙNG KHI CHƯA UPLOAD CSV)
# ---------------------------------------------------------------------------
@st.cache_data
def load_or_generate_sample_data(seed: int = 42) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    sample_sales_file = _find_first_file(SAMPLE_SEARCH_DIRS, ["*sales*.csv", "*.csv"])
    sample_inv_file = _find_first_file(SAMPLE_SEARCH_DIRS, ["*inventory*.csv", "*inv*.csv"])

    if sample_sales_file is not None:
        s_df = pd.read_csv(sample_sales_file)
        i_df = pd.read_csv(sample_inv_file) if (sample_inv_file and sample_inv_file != sample_sales_file) else None
        return s_df, i_df, f"Sample: {sample_sales_file.name}"

    rng = np.random.default_rng(seed)
    skus = [f"SKU_{i:04d}" for i in range(1, 21)]
    dates = pd.date_range("2024-01-01", "2025-06-30", freq="D")

    n_rows = 2400
    order_dates = rng.choice(dates, size=n_rows)
    sku_choices = rng.choice(skus, size=n_rows)
    base_prices = {s: float(rng.integers(50, 1200)) for s in skus}

    df = pd.DataFrame(
        {
            "OrderID": [f"ORD_{10000 + i}" for i in range(n_rows)],
            "OrderDate": pd.to_datetime(order_dates),
            "SKU_ID": sku_choices,
            "Quantity": rng.integers(1, 9, size=n_rows).astype(float),
            "ActualSellingPrice": [base_prices[s] * rng.uniform(0.9, 1.1) for s in sku_choices],
            "UnitCostAtSale": [base_prices[s] * rng.uniform(0.6, 0.8) for s in sku_choices],
            "CustomerID": [f"CUST_{rng.integers(1, 500):04d}" for _ in range(n_rows)],
            "SalesLocationID": [f"LOC_{rng.integers(1, 15):02d}" for _ in range(n_rows)],
            "OrderStatus": rng.choice(
                ["Completed", "Complete", "Cancelled", "Canceled", "Completed", "Completed"],
                size=n_rows,
                p=[0.55, 0.25, 0.08, 0.05, 0.04, 0.03],
            ),
        }
    )
    df["Revenue"] = (df["Quantity"] * df["ActualSellingPrice"]).round(2)
    df.loc[rng.choice(n_rows, size=180, replace=False), "CustomerID"] = np.nan
    df.loc[rng.choice(n_rows, size=240, replace=False), "SalesLocationID"] = np.nan
    df.loc[rng.choice(n_rows, size=90, replace=False), "UnitCostAtSale"] = np.nan
    df.loc[rng.choice(n_rows, size=18, replace=False), "Quantity"] = -2.0
    df.loc[rng.choice(n_rows, size=22, replace=False), "Revenue"] *= 1.35
    df.loc[rng.choice(n_rows, size=15, replace=False), "ActualSellingPrice"] *= 6.0
    df.loc[rng.choice(n_rows, size=25, replace=False), "OrderDate"] = pd.Timestamp("2026-08-15")
    df = pd.concat([df, df.sample(12, random_state=seed)], ignore_index=True)

    inv_df = pd.DataFrame({"SKU_ID": skus, "InventoryOnHand": rng.integers(0, 45, size=len(skus))})
    return df, inv_df, "Dữ liệu mẫu tích hợp (Synthetic)"


# ---------------------------------------------------------------------------
# 5. DATA VALIDATION, DATA QUALITY & SHARED PREPROCESSING (MỤC 4, 5, 6)
# ---------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_candidates = {
        "SalesOrderID": "OrderID",
        "ProductVariantID": "SKU_ID",
        "SKU": "SKU_ID",
        "OrderQty": "Quantity",
        "LineTotal": "Revenue",
    }
    out = df.copy()
    for src, dst in rename_candidates.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    return out


def validate_transaction_schema(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    required = ["OrderDate", "SKU_ID", "Quantity"]
    optional_recommended = ["OrderID", "OrderStatus", "ActualSellingPrice", "UnitCostAtSale", "CustomerID", "SalesLocationID"]
    errors: List[str] = []
    warnings: List[str] = []

    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        errors.append(f"Thiếu cột bắt buộc: {', '.join(missing_req)}")

    missing_opt = [c for c in optional_recommended if c not in df.columns]
    if missing_opt:
        warnings.append(f"Không có cột khuyến nghị (một số rule CLDL sẽ bỏ qua): {', '.join(missing_opt)}")

    if "OrderDate" in df.columns:
        invalid_date_cnt = int(pd.to_datetime(df["OrderDate"], errors="coerce").isna().sum())
        if invalid_date_cnt > 0:
            warnings.append(f"Có {invalid_date_cnt:,} dòng không thể parse kiểu ngày ở cột OrderDate.")

    if "Quantity" in df.columns:
        non_num_cnt = int(pd.to_numeric(df["Quantity"], errors="coerce").isna().sum())
        if non_num_cnt > 0:
            warnings.append(f"Có {non_num_cnt:,} dòng không phải giá trị số ở cột Quantity.")

    return len(errors) == 0, errors, warnings


def assess_data_quality(df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    n_rows = len(df)
    if n_rows == 0:
        return {k: 0.0 for k in ["Completeness", "Consistency", "Timeliness", "Accuracy Proxy", "Validity", "Uniqueness"]}, pd.DataFrame()

    alerts: List[Dict[str, object]] = []

    req_cols = [c for c in ["OrderID", "OrderDate", "SKU_ID", "Quantity", "OrderStatus", "CustomerID", "SalesLocationID", "UnitCostAtSale"] if c in df.columns]
    total_cells = n_rows * len(req_cols)
    missing_cells = int(df[req_cols].isna().sum().sum())
    completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 1.0
    for col in req_cols:
        miss_n = int(df[col].isna().sum())
        if miss_n > 0:
            alerts.append(
                {
                    "Chiều CLDL": "Completeness",
                    "Quy tắc": f"Missing {col}",
                    "Vi phạm": miss_n,
                    "Tỷ lệ (%)": round(miss_n / n_rows * 100, 2),
                    "Khuyến nghị xử lý": "Giữ NULL (không tự điền 0)" if col in ["CustomerID", "SalesLocationID", "UnitCostAtSale"] else "Kiểm tra nguồn hoặc cách ly",
                }
            )

    consistency_violations = 0
    if "OrderStatus" in df.columns:
        non_standard_mask = df["OrderStatus"].astype(str).str.strip().str.lower().isin(["complete", "canceled", "cancel"])
        consistency_violations = int(non_standard_mask.sum())
        if consistency_violations > 0:
            alerts.append(
                {
                    "Chiều CLDL": "Consistency",
                    "Quy tắc": "OrderStatus chưa chuẩn hóa (Complete/Canceled)",
                    "Vi phạm": consistency_violations,
                    "Tỷ lệ (%)": round(consistency_violations / n_rows * 100, 2),
                    "Khuyến nghị xử lý": "Chuẩn hóa về Completed / Cancelled",
                }
            )
    consistency = 1.0 - (consistency_violations / n_rows)

    parsed_date = pd.to_datetime(df["OrderDate"], errors="coerce")
    out_of_scope_mask = parsed_date.isna() | (parsed_date < VALID_START_DATE) | (parsed_date > VALID_END_DATE)
    out_of_scope_cnt = int(out_of_scope_mask.sum())
    timeliness = 1.0 - (out_of_scope_cnt / n_rows)
    if out_of_scope_cnt > 0:
        alerts.append(
            {
                "Chiều CLDL": "Timeliness",
                "Quy tắc": f"OrderDate ngoài phạm vi {VALID_START_DATE.year}–{VALID_END_DATE.year}",
                "Vi phạm": out_of_scope_cnt,
                "Tỷ lệ (%)": round(out_of_scope_cnt / n_rows * 100, 2),
                "Khuyến nghị xử lý": "Cách ly khỏi phạm vi chuỗi thời gian",
            }
        )

    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    r1_mask = qty < 0
    if "ActualSellingPrice" in df.columns:
        r1_mask = r1_mask | (pd.to_numeric(df["ActualSellingPrice"], errors="coerce") < 0)

    r2_mask = pd.Series(False, index=df.index)
    if {"Revenue", "ActualSellingPrice"}.issubset(df.columns):
        rev = pd.to_numeric(df["Revenue"], errors="coerce")
        price = pd.to_numeric(df["ActualSellingPrice"], errors="coerce")
        r2_mask = (rev - qty * price).abs() > 1.0

    r3_mask = pd.Series(False, index=df.index)
    if "ActualSellingPrice" in df.columns:
        price = pd.to_numeric(df["ActualSellingPrice"], errors="coerce")
        q1 = df.groupby("SKU_ID")["ActualSellingPrice"].transform(lambda s: pd.to_numeric(s, errors="coerce").quantile(0.25))
        q3 = df.groupby("SKU_ID")["ActualSellingPrice"].transform(lambda s: pd.to_numeric(s, errors="coerce").quantile(0.75))
        iqr = q3 - q1
        r3_mask = (iqr > 0) & ((price < q1 - 1.5 * iqr) | (price > q3 + 1.5 * iqr))

    accuracy_proxy = 1.0 - (int((r1_mask | r2_mask | r3_mask).sum()) / n_rows)
    for label, mask, advice in [
        ("R1: Giá trị đo lường âm (Quantity/Price < 0)", r1_mask, "Cách ly điều tra trả hàng; không tự gán 0"),
        ("R2: Sai lệch toán học Revenue != Quantity * Price", r2_mask, "Kiểm tra chiết khấu hoặc lỗi ghi nhận"),
        ("R3: Ngoại lai giá bán theo IQR của từng SKU", r3_mask, "Gắn cờ bản ghi đáng ngờ để kiểm tra"),
    ]:
        cnt = int(mask.sum())
        if cnt > 0:
            alerts.append(
                {
                    "Chiều CLDL": "Accuracy Proxy",
                    "Quy tắc": label,
                    "Vi phạm": cnt,
                    "Tỷ lệ (%)": round(cnt / n_rows * 100, 2),
                    "Khuyến nghị xử lý": advice,
                }
            )

    validity_cnt = int((qty.isna() | (qty <= 0) | df["SKU_ID"].isna()).sum())
    validity = 1.0 - (validity_cnt / n_rows)
    if validity_cnt > 0:
        alerts.append(
            {
                "Chiều CLDL": "Validity",
                "Quy tắc": "Quantity <= 0 / NULL hoặc thiếu SKU_ID",
                "Vi phạm": validity_cnt,
                "Tỷ lệ (%)": round(validity_cnt / n_rows * 100, 2),
                "Khuyến nghị xử lý": "Cách ly khỏi tổng sản lượng hợp lệ",
            }
        )

    dup_cnt = int(df.duplicated(keep="first").sum())
    uniqueness = 1.0 - (dup_cnt / n_rows)
    if dup_cnt > 0:
        alerts.append(
            {
                "Chiều CLDL": "Uniqueness",
                "Quy tắc": "Bản ghi trùng lặp hoàn toàn (Exact Duplicate)",
                "Vi phạm": dup_cnt,
                "Tỷ lệ (%)": round(dup_cnt / n_rows * 100, 2),
                "Khuyến nghị xử lý": "Loại bỏ sau khi xác minh khóa nghiệp vụ",
            }
        )

    scores = {
        "Completeness": round(completeness * 100, 2),
        "Consistency": round(consistency * 100, 2),
        "Timeliness": round(timeliness * 100, 2),
        "Accuracy Proxy": round(accuracy_proxy * 100, 2),
        "Validity": round(validity * 100, 2),
        "Uniqueness": round(uniqueness * 100, 2),
    }
    return scores, pd.DataFrame(alerts)


def clean_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {"raw_rows": len(df)}
    out = df.copy()

    before = len(out)
    out = out.drop_duplicates()
    stats["removed_duplicates"] = before - len(out)

    if "OrderStatus" in out.columns:
        out["OrderStatus"] = (
            out["OrderStatus"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: STATUS_MAP.get(x, x.title()))
        )
        before = len(out)
        out = out[out["OrderStatus"] != "Cancelled"].copy()
        stats["removed_cancelled"] = before - len(out)
    else:
        stats["removed_cancelled"] = 0

    out["OrderDate"] = pd.to_datetime(out["OrderDate"], errors="coerce")
    before = len(out)
    out = out[
        out["OrderDate"].notna()
        & (out["OrderDate"] >= VALID_START_DATE)
        & (out["OrderDate"] <= VALID_END_DATE)
    ].copy()
    stats["removed_invalid_dates"] = before - len(out)

    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    before = len(out)
    out = out[out["Quantity"].notna() & (out["Quantity"] > 0) & out["SKU_ID"].notna()].copy()
    stats["quarantined_qty_or_sku"] = before - len(out)

    stats["cleaned_rows"] = len(out)
    return out, stats


def build_sku_week_panel_and_features(
    cleaned_df: pd.DataFrame,
    assume_full_coverage: bool = True,
) -> pd.DataFrame:
    df = cleaned_df.copy()
    df["WeekStart"] = df["OrderDate"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    weekly_sales = (
        df.groupby(["SKU_ID", "WeekStart"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Quantity": "Actual_Quantity"})
    )
    global_max_week = weekly_sales["WeekStart"].max()

    panels: List[pd.DataFrame] = []
    for sku, grp in weekly_sales.groupby("SKU_ID"):
        sku_start = grp["WeekStart"].min()
        full_weeks = pd.date_range(sku_start, global_max_week, freq="W-MON")
        merged = pd.DataFrame({"SKU_ID": sku, "WeekStart": full_weeks}).merge(grp, on=["SKU_ID", "WeekStart"], how="left")
        has_tx = merged["Actual_Quantity"].notna()
        if assume_full_coverage:
            merged["Label_Status"] = np.where(has_tx, "VALID", "VALID_ZERO")
            merged["Actual_Quantity"] = merged["Actual_Quantity"].fillna(0.0)
        else:
            merged["Label_Status"] = np.where(has_tx, "VALID", "UNVERIFIED")
        merged["active_age"] = np.arange(len(merged), dtype=float)
        panels.append(merged)

    panel = pd.concat(panels, ignore_index=True).sort_values(["SKU_ID", "WeekStart"]).reset_index(drop=True)

    next_week_ts = global_max_week + pd.Timedelta(days=7)
    next_rows = (
        panel.groupby("SKU_ID", as_index=False)["active_age"]
        .max()
        .assign(WeekStart=next_week_ts, Actual_Quantity=np.nan, Label_Status="FORECAST_TARGET")
    )
    next_rows["active_age"] = next_rows["active_age"] + 1.0
    panel = pd.concat([panel, next_rows], ignore_index=True).sort_values(["SKU_ID", "WeekStart"]).reset_index(drop=True)

    g = panel.groupby("SKU_ID")["Actual_Quantity"]
    panel["lag_1"] = g.shift(1)
    panel["lag_2"] = g.shift(2)
    panel["lag_4"] = g.shift(4)
    panel["lag_8"] = g.shift(8)

    shifted = g.shift(1)
    panel["rolling_mean_4"] = shifted.groupby(panel["SKU_ID"]).transform(lambda s: s.rolling(4, min_periods=1).mean())
    panel["rolling_mean_8"] = shifted.groupby(panel["SKU_ID"]).transform(lambda s: s.rolling(8, min_periods=1).mean())
    panel["rolling_std_4"] = shifted.groupby(panel["SKU_ID"]).transform(lambda s: s.rolling(4, min_periods=2).std()).fillna(0.0)

    panel["month"] = panel["WeekStart"].dt.month.astype(float)
    panel["week_of_year"] = panel["WeekStart"].dt.isocalendar().week.astype(float)
    return panel


def run_forecast_next_week(
    panel: pd.DataFrame,
    model_name: str,
    model_entry: Dict[str, Any],
    dataset_version: str = "D0",
    inventory_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, int]:
    target_rows = panel[panel["Label_Status"] == "FORECAST_TARGET"].copy()
    if target_rows.empty:
        return pd.DataFrame(), 0

    if model_entry["model"] is not None:
        raw_pred = _predict_with_artifact(model_entry["model"], target_rows)
    elif model_name == "MA4 (Baseline)":
        raw_pred = target_rows["rolling_mean_4"].fillna(0.0).to_numpy(dtype=float)
    else:
        raise ValueError(f"Mô hình {model_name} chưa sẵn sàng trong thư mục app/models/.")

    clipped_cnt = int((raw_pred < 0).sum())
    pred_non_neg = np.maximum(0.0, raw_pred)

    result = pd.DataFrame(
        {
            "SKU_ID": target_rows["SKU_ID"].values,
            "Forecast_Week": target_rows["WeekStart"].dt.strftime("%Y-%m-%d").values,
            "Actual_Quantity": np.nan,
            "Predicted_Quantity": np.round(pred_non_neg, 2),
            "Model": model_name,
            "Dataset_Version": dataset_version,
        }
    )

    if inventory_df is not None and {"SKU_ID", "InventoryOnHand"}.issubset(inventory_df.columns):
        inv_agg = inventory_df.groupby("SKU_ID", as_index=False)["InventoryOnHand"].sum()
        result = result.merge(inv_agg, on="SKU_ID", how="left")
        result["Stock_Cover_Weeks"] = np.where(
            result["Predicted_Quantity"] > 0,
            np.round(result["InventoryOnHand"] / result["Predicted_Quantity"], 2),
            np.nan,
        )
        result["Stockout_Risk"] = np.where(
            result["InventoryOnHand"].isna(),
            "N/A",
            np.where(result["InventoryOnHand"] < result["Predicted_Quantity"], "Cao (< 1 tuần)", "Ổn định"),
        )

    return result, clipped_cnt


# ---------------------------------------------------------------------------
# 6. GIAO DIỆN CHÍNH STREAMLIT (BỐ CỤC CHỮ Z + TỰ ĐỘNG NẠP MÔ HÌNH SẴN)
# ---------------------------------------------------------------------------
def main() -> None:
    model_registry = load_builtin_models()
    metrics_df, metrics_src = load_builtin_metrics()

    # ---------------- SIDEBAR: CHỈ DÀNH CHO UPLOAD DỮ LIỆU CSV ----------------
    with st.sidebar:
        st.markdown("### 📂 Dữ liệu đầu vào (CSV)")
        data_source_mode = st.radio(
            "Chọn nguồn dữ liệu",
            ["Upload dữ liệu CSV", "Dùng dữ liệu mẫu có sẵn"],
            index=1,
            help="Người dùng chỉ cần tải lên file CSV giao dịch bán hàng (và tồn kho nếu có).",
        )

        uploaded_sales = None
        uploaded_inv = None
        if data_source_mode == "Upload dữ liệu CSV":
            uploaded_sales = st.file_uploader(
                "Tải file giao dịch bán hàng (.csv)",
                type=["csv"],
                help="Tối thiểu gồm: OrderDate, SKU_ID, Quantity.",
            )
            uploaded_inv = st.file_uploader(
                "Tải file tồn kho hiện tại (.csv - tùy chọn)",
                type=["csv"],
                help="Gồm cột SKU_ID và InventoryOnHand để tính Stock Cover.",
            )

        assume_full_cov = st.checkbox(
            "Xác nhận đủ độ phủ (tuần trống → VALID_ZERO)",
            value=True,
            help="Nếu bỏ chọn, tuần không có giao dịch sẽ giữ NULL (UNVERIFIED) theo Mục 6.1.",
        )

        st.divider()
        st.markdown("### 🤖 Mô hình tích hợp sẵn")
        st.caption("Hệ thống tự động nạp sẵn từ `app/models/` và `app/data/` (không train lại khi demo).")

        status_lines = []
        for m_name, info in model_registry.items():
            ok = info["status"] in ("loaded", "builtin_formula")
            dot = '<span class="dot-ok">●</span>' if ok else '<span class="dot-warn">○</span>'
            status_lines.append(f"{dot} <b>{m_name}</b><br>&nbsp;&nbsp;&nbsp;<span style='color:#64748B;'>{info['detail']}</span>")

        m_dot = '<span class="dot-ok">●</span>' if metrics_df is not None else '<span class="dot-warn">○</span>'
        m_txt = metrics_src if metrics_src else "Chưa có CSV trong app/data/"
        status_lines.append(f"{m_dot} <b>Bảng so sánh mô hình</b><br>&nbsp;&nbsp;&nbsp;<span style='color:#64748B;'>{m_txt}</span>")

        st.markdown(f'<div class="model-status-box">{"<br>".join(status_lines)}</div>', unsafe_allow_html=True)

    is_sample = data_source_mode == "Dùng dữ liệu mẫu có sẵn"
    if is_sample:
        raw_df, inv_df, data_label = load_or_generate_sample_data()
    else:
        if uploaded_sales is None:
            st.markdown(
                """
                <div class="z-header">
                    <div>
                        <div class="z-header-title">📊 SKU-Weekly Sales Forecasting & Data Quality</div>
                        <div class="z-header-sub">DS317 · Hệ thống dự báo sản lượng & giám sát chất lượng dữ liệu bán lẻ điện tử</div>
                    </div>
                    <div class="z-badge-group">
                        <span class="z-pill z-pill-info">📂 Chờ tải lên dữ liệu CSV</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info("Tải lên tệp CSV giao dịch ở thanh bên trái để hệ thống tự động kiểm tra chất lượng dữ liệu và dự báo bằng các mô hình đã huấn luyện sẵn.")
            st.stop()
        raw_df = pd.read_csv(uploaded_sales)
        inv_df = pd.read_csv(uploaded_inv) if uploaded_inv is not None else None
        data_label = uploaded_sales.name

    raw_df = normalize_columns(raw_df)
    is_valid, schema_errors, schema_warnings = validate_transaction_schema(raw_df)

    if is_valid:
        raw_scores, raw_alerts = assess_data_quality(raw_df)
        cleaned_df, clean_stats = clean_transactions(raw_df)
        clean_scores, _ = assess_data_quality(cleaned_df)
        avg_dq_clean = round(float(np.mean(list(clean_scores.values()))), 1)
    else:
        raw_scores, raw_alerts, cleaned_df, clean_stats, clean_scores, avg_dq_clean = {}, pd.DataFrame(), pd.DataFrame(), {}, {}, 0.0

    # =========================================================================
    # Z-PATTERN TOP BAR (ĐIỂM 1 TOP-LEFT -> ĐIỂM 2 TOP-RIGHT)
    # =========================================================================
    mode_pill = (
        f'<span class="z-pill z-pill-demo">🧪 {data_label}</span>'
        if is_sample
        else f'<span class="z-pill z-pill-live">● CSV: {data_label}</span>'
    )
    loaded_model_cnt = sum(1 for v in model_registry.values() if v["status"] in ("loaded", "builtin_formula"))
    model_pill = f'<span class="z-pill z-pill-info">🤖 {loaded_model_cnt}/3 Mô hình sẵn sàng</span>'
    dq_pill = f'<span class="z-pill z-pill-info">DQ sau làm sạch: {avg_dq_clean:.1f}%</span>' if is_valid else ""

    st.markdown(
        f"""
        <div class="z-header">
            <div>
                <div class="z-header-title">📊 SKU-Weekly Sales Forecasting & Data Quality</div>
                <div class="z-header-sub">DS317 · Electronics Sales Dataset (B-Tech Style) · Pre-trained Models Built-in</div>
            </div>
            <div class="z-badge-group">
                {mode_pill}
                {model_pill}
                {dq_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if schema_errors:
        for err in schema_errors:
            st.error(err)
        st.stop()

    if cleaned_df.empty:
        st.error("Không còn bản ghi hợp lệ nào sau bước tiền xử lý để tạo bảng SKU-Week.")
        st.stop()

    panel_df = build_sku_week_panel_and_features(cleaned_df, assume_full_coverage=assume_full_cov)

    tab_dq, tab_fc, tab_cmp, tab_val, tab_exp = st.tabs(
        [
            "🛡️ Data Quality Dashboard",
            "📈 SKU-Week Forecasting",
            "⚖️ Model Comparison (D0–D4)",
            "🔍 Data Validation & Profile",
            "⬇️ Export CSV",
        ]
    )

    # =========================================================================
    # TAB 1: DATA QUALITY DASHBOARD (NGUYÊN TẮC CHỮ Z)
    # =========================================================================
    with tab_dq:
        k1, k2, k3, k4 = st.columns(4)
        for col_ui, (dim, tone) in zip(
            [k1, k2, k3, k4],
            [("Completeness", "blue"), ("Consistency", "emerald"), ("Timeliness", "amber"), ("Accuracy Proxy", "rose")],
        ):
            delta = round(clean_scores[dim] - raw_scores[dim], 2)
            dkind = "pos" if delta > 0 else ("neg" if delta < 0 else "neu")
            with col_ui:
                render_kpi_card(
                    label=dim,
                    value=f"{clean_scores[dim]:.2f}%",
                    delta_text=f"{delta:+.2f}% so với Raw ({raw_scores[dim]:.1f}%)",
                    tone=tone,
                    delta_kind=dkind,
                )

        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

        center_left, center_right = st.columns([1.85, 1.15], gap="medium")
        with center_left:
            with st.container(border=True):
                render_card_header(
                    "So sánh 6 chiều chất lượng dữ liệu (Raw vs. Cleaned)",
                    "Đánh giá mức độ cải thiện trên 4 chiều trọng tâm & 2 chiều bổ sung",
                )
                dims = list(raw_scores.keys())
                fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=120)
                x = np.arange(len(dims))
                w = 0.36
                ax.bar(x - w / 2, [raw_scores[k] for k in dims], w, label="Trước xử lý (Raw)", color="#94A3B8")
                b2 = ax.bar(x + w / 2, [clean_scores[k] for k in dims], w, label="Sau xử lý (Cleaned)", color="#1D4ED8")
                ax.set_xticks(x)
                ax.set_xticklabels(dims, rotation=12, ha="right")
                ax.set_ylim(0, 118)
                style_matplotlib_ax(ax, ylabel="Điểm CLDL (%)")
                ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=9)
                for bar in b2:
                    h = bar.get_height()
                    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#1E3A8A")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with center_right:
            with st.container(border=True):
                render_card_header("Tóm tắt quy trình làm sạch (Mục 5.2)", "Tỷ lệ bản ghi hợp lệ sau 4 bước kiểm soát")
                r_sub1, r_sub2 = st.columns(2)
                with r_sub1:
                    render_kpi_card("Validity", f"{clean_scores['Validity']:.1f}%", f"Raw: {raw_scores['Validity']:.1f}%", tone="emerald", delta_kind="pos")
                with r_sub2:
                    render_kpi_card("Uniqueness", f"{clean_scores['Uniqueness']:.1f}%", f"Raw: {raw_scores['Uniqueness']:.1f}%", tone="slate", delta_kind="pos")
                st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)
                audit_df = pd.DataFrame(
                    [
                        {"Bước làm sạch": "Loại Exact Duplicate", "Số dòng": clean_stats["removed_duplicates"]},
                        {"Bước làm sạch": "Loại đơn Cancelled", "Số dòng": clean_stats["removed_cancelled"]},
                        {"Bước làm sạch": "Ngày ngoài 2023–2025", "Số dòng": clean_stats["removed_invalid_dates"]},
                        {"Bước làm sạch": "Cách ly Quantity <= 0", "Số dòng": clean_stats["quarantined_qty_or_sku"]},
                    ]
                )
                st.dataframe(audit_df, use_container_width=True, hide_index=True, height=180)
                kept_pct = round(clean_stats["cleaned_rows"] / max(clean_stats["raw_rows"], 1) * 100, 1)
                st.caption(f"Giữ lại **{clean_stats['cleaned_rows']:,} / {clean_stats['raw_rows']:,}** bản ghi hợp lệ (**{kept_pct}%**).")

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        bot_left, bot_right = st.columns([2.1, 0.9], gap="medium")
        with bot_left:
            with st.container(border=True):
                render_card_header(
                    "Danh sách cảnh báo vi phạm trên dữ liệu gốc (Raw)",
                    "Chi tiết từng quy tắc kiểm tra trước khi đưa vào tạo đặc trưng dự báo",
                )
                if raw_alerts.empty:
                    st.success("Không phát hiện vi phạm CLDL nào trên dữ liệu đầu vào.")
                else:
                    st.dataframe(raw_alerts.sort_values("Vi phạm", ascending=False), use_container_width=True, hide_index=True)

        with bot_right:
            dq_export = pd.DataFrame(
                [{"Dimension": k, "Raw_Score_Pct": raw_scores[k], "Cleaned_Score_Pct": clean_scores[k]} for k in raw_scores.keys()]
            )
            st.markdown(
                f"""
                <div class="z-cta-box">
                    <div class="z-cta-title">✅ Hành động đề xuất</div>
                    <div class="z-cta-text">
                        Phát hiện <b>{len(raw_alerts)} nhóm cảnh báo</b> trên dữ liệu gốc.
                        Sau làm sạch, điểm CLDL trung bình đạt <b>{avg_dq_clean:.1f}%</b> và đã tự động đồng bộ sang tab Dự báo.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                label="⬇️ Tải báo cáo CLDL (.csv)",
                data=dq_export.to_csv(index=False).encode("utf-8-sig"),
                file_name="data_quality_summary.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

    # =========================================================================
    # TAB 2: SKU-WEEK FORECASTING (NGUYÊN TẮC CHỮ Z)
    # =========================================================================
    with tab_fc:
        ready_models = [name for name, info in model_registry.items() if info["status"] in ("loaded", "builtin_formula")]

        ctrl1, ctrl2, ctrl3 = st.columns([1.4, 1.0, 1.6])
        selected_model = ctrl1.selectbox("Chọn mô hình đã huấn luyện sẵn", ready_models)
        selected_version = ctrl2.selectbox("Phiên bản dữ liệu", ["D0", "D1", "D2", "D3", "D4"], index=0)
        sku_options = sorted(panel_df["SKU_ID"].unique().tolist())
        chosen_sku = ctrl3.selectbox("Chọn mã SKU để xem chuỗi thời gian", sku_options)

        try:
            forecast_df, clipped_cnt = run_forecast_next_week(
                panel=panel_df,
                model_name=selected_model,
                model_entry=model_registry[selected_model],
                dataset_version=selected_version,
                inventory_df=inv_df,
            )
        except Exception as exc:
            st.error(f"Không thể chạy dự báo với mô hình **{selected_model}**: {exc}")
            st.stop()

        st.session_state["latest_forecast_df"] = forecast_df
        st.session_state["latest_model_name"] = selected_model
        st.session_state["latest_version"] = selected_version

        high_risk_cnt = int((forecast_df["Stockout_Risk"] == "Cao (< 1 tuần)").sum()) if "Stockout_Risk" in forecast_df.columns else 0

        fk1, fk2, fk3, fk4 = st.columns(4)
        with fk1:
            render_kpi_card("Tuần dự báo (t+1)", forecast_df["Forecast_Week"].iloc[0] if not forecast_df.empty else "N/A", f"Nguồn: {model_registry[selected_model]['detail']}", tone="blue")
        with fk2:
            render_kpi_card("Tổng sản lượng dự báo", f"{forecast_df['Predicted_Quantity'].sum():,.1f}", f"Trên {len(forecast_df):,} SKU hoạt động", tone="emerald", delta_kind="pos")
        with fk3:
            render_kpi_card("SKU rủi ro hết hàng", f"{high_risk_cnt:,}", "Stock Cover < 1 tuần" if "Stockout_Risk" in forecast_df.columns else "Chưa tải file tồn kho", tone="rose" if high_risk_cnt > 0 else "slate", delta_kind="neg" if high_risk_cnt > 0 else "neu")
        with fk4:
            render_kpi_card("Clipping âm về 0", f"{clipped_cnt}", "Quy tắc Mục 8.3: max(0, y_hat)", tone="slate")

        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            render_card_header(
                f"Lịch sử 24 tuần & Dự báo tuần kế tiếp — {chosen_sku}",
                f"Đường xanh: sản lượng thực tế quan sát được · Điểm đỏ: dự báo tuần t+1 từ mô hình {selected_model}",
            )
            sku_hist = panel_df[(panel_df["SKU_ID"] == chosen_sku) & (panel_df["Label_Status"] != "FORECAST_TARGET")].tail(24)
            sku_fc = forecast_df[forecast_df["SKU_ID"] == chosen_sku]

            fig_fc, ax_fc = plt.subplots(figsize=(11.5, 3.6), dpi=120)
            ax_fc.plot(sku_hist["WeekStart"], sku_hist["Actual_Quantity"], marker="o", linewidth=2.2, markersize=5, label="Thực tế (Actual)", color="#1D4ED8")
            ax_fc.fill_between(sku_hist["WeekStart"], sku_hist["Actual_Quantity"], alpha=0.08, color="#1D4ED8")

            if not sku_fc.empty:
                fc_date = pd.to_datetime(sku_fc["Forecast_Week"].iloc[0])
                fc_val = float(sku_fc["Predicted_Quantity"].iloc[0])
                ax_fc.scatter([fc_date], [fc_val], color="#DC2626", s=110, zorder=5, label=f"Dự báo ({selected_model}): {fc_val:.2f}")
                ax_fc.annotate(f"{fc_val:.1f}", xy=(fc_date, fc_val), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold", color="#DC2626")
                if not sku_hist.empty:
                    ax_fc.plot([sku_hist["WeekStart"].iloc[-1], fc_date], [sku_hist["Actual_Quantity"].iloc[-1], fc_val], linestyle="--", linewidth=1.8, color="#DC2626", alpha=0.8)

            max_y = max(
                float(sku_hist["Actual_Quantity"].max()) if not sku_hist.empty else 1.0,
                float(sku_fc["Predicted_Quantity"].iloc[0]) if not sku_fc.empty else 1.0,
            )
            ax_fc.set_ylim(0, max_y * 1.25 if max_y > 0 else 5)
            style_matplotlib_ax(ax_fc, ylabel="Quantity (Sản phẩm)")
            ax_fc.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=9)
            fig_fc.tight_layout()
            st.pyplot(fig_fc, use_container_width=True)
            plt.close(fig_fc)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        fc_bot_l, fc_bot_r = st.columns([2.1, 0.9], gap="medium")
        with fc_bot_l:
            with st.container(border=True):
                render_card_header(
                    "Bảng xếp hạng dự báo theo SKU (Tuần kế tiếp)",
                    "Sắp xếp giảm dần theo sản lượng dự báo để ưu tiên kiểm tra kế hoạch tồn kho",
                )
                st.dataframe(forecast_df.sort_values("Predicted_Quantity", ascending=False), use_container_width=True, hide_index=True, height=290)

        with fc_bot_r:
            st.markdown(
                f"""
                <div class="z-cta-box">
                    <div class="z-cta-title">📦 Hỗ trợ quyết định tồn kho</div>
                    <div class="z-cta-text">
                        Dự báo được chạy trực tiếp bằng mô hình <b>{selected_model}</b> đã huấn luyện sẵn trong hệ thống.<br><br>
                        Nhấn nút bên dưới để xuất bảng dự báo chuẩn Mục 2.4 (kèm thông tin tồn kho nếu có).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                label="⬇️ Xuất kết quả dự báo (.csv)",
                data=forecast_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"sku_weekly_forecast_{selected_version}_{selected_model.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

    # =========================================================================
    # TAB 3: MODEL COMPARISON & DATA QUALITY IMPACT (TỰ ĐỘNG NẠP TỪ app/data/)
    # =========================================================================
    with tab_cmp:
        if metrics_df is None or metrics_df.empty:
            with st.container(border=True):
                render_card_header(
                    "Chưa tìm thấy bảng kết quả đánh giá trong app/data/",
                    "Đặt tệp CSV kết quả đánh giá (ví dụ app/data/model_comparison.csv) để hệ thống tự động hiển thị tại đây.",
                )
        else:
            # Chuẩn hóa linh hoạt tên cột (hỗ trợ cả WAPE và WAPE_pct)
            mdf = metrics_df.copy()
            col_rename = {}
            for c in mdf.columns:
                cl = c.strip().lower()
                if cl in ("model", "model_name", "algorithm"):
                    col_rename[c] = "Model_Name"
                elif cl in ("dataset", "dataset_version", "version", "scenario"):
                    col_rename[c] = "Dataset_Version"
                elif cl == "mae":
                    col_rename[c] = "MAE"
                elif cl == "rmse":
                    col_rename[c] = "RMSE"
                elif cl in ("wape", "wape_pct", "wape_%"):
                    col_rename[c] = "WAPE"
            mdf = mdf.rename(columns=col_rename)

            # [Z-1 -> Z-2]: Top KPIs từ kết quả tốt nhất
            ref_rows = mdf[mdf["Dataset_Version"].astype(str).str.upper().str.startswith("D0")] if "Dataset_Version" in mdf.columns else mdf
            if ref_rows.empty:
                ref_rows = mdf
            best_row = ref_rows.sort_values("MAE").iloc[0] if ("MAE" in ref_rows.columns and not ref_rows.empty) else None

            if best_row is not None:
                best_model_display = str(best_row.get("Model_Name", "N/A")).replace("_", " ")
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    render_kpi_card("Mô hình tốt nhất", best_model_display, f"Nguồn: {metrics_src}", tone="emerald")
                with mc2:
                    render_kpi_card("MAE tốt nhất", f"{float(best_row['MAE']):.4f}", "Metric đánh giá chính", tone="blue")
                with mc3:
                    rmse_val = f"{float(best_row['RMSE']):.4f}" if "RMSE" in best_row and pd.notna(best_row["RMSE"]) else "N/A"
                    render_kpi_card("RMSE", rmse_val, "Phạt mạnh sai số lớn", tone="slate")
                with mc4:
                    wape_val = f"{float(best_row['WAPE']):.2f}%" if "WAPE" in best_row and pd.notna(best_row["WAPE"]) else "N/A"
                    render_kpi_card("WAPE", wape_val, "Sai số tương đối tổng sản lượng", tone="amber")
                st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

            # [Z-CENTER]: Biểu đồ so sánh MAE (chống chồng chữ & tự động thích ứng 1 hoặc nhiều Dataset_Version)
            if {"Model_Name", "MAE"}.issubset(mdf.columns):
                n_versions = mdf["Dataset_Version"].nunique() if "Dataset_Version" in mdf.columns else 1

                if "Dataset_Version" in mdf.columns and n_versions > 1:
                    d0_ref = mdf[mdf["Dataset_Version"].astype(str).str.upper() == "D0"][["Model_Name", "MAE"]].rename(columns={"MAE": "MAE_D0"})
                    impact_df = mdf.merge(d0_ref, on="Model_Name", how="left") if not d0_ref.empty else mdf.copy()
                    if "MAE_D0" in impact_df.columns:
                        impact_df["Delta_MAE"] = (impact_df["MAE"] - impact_df["MAE_D0"]).round(4)

                    with st.container(border=True):
                        render_card_header(
                            "So sánh MAE giữa các mô hình qua kịch bản D0–D4",
                            "Tác động của các nhóm lỗi CLDL (D1–D4) lên sai số dự báo so với dữ liệu chuẩn D0",
                        )
                        pivot_mae = impact_df.pivot_table(index="Dataset_Version", columns="Model_Name", values="MAE", aggfunc="mean")
                        fig_cmp, ax_cmp = plt.subplots(figsize=(10.5, 4.0), dpi=120)
                        pivot_mae.plot(kind="bar", ax=ax_cmp, color=["#64748B", "#2563EB", "#059669", "#D97706"][: len(pivot_mae.columns)], width=0.68)
                        max_mae = float(pivot_mae.max().max()) if not pivot_mae.empty else 1.0
                        ax_cmp.set_ylim(0, max_mae * 1.24)
                        style_matplotlib_ax(ax_cmp, ylabel="MAE (thấp hơn là tốt hơn)", xlabel="Phiên bản dữ liệu")
                        ax_cmp.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=len(pivot_mae.columns), fontsize=9)
                        for container in ax_cmp.containers:
                            ax_cmp.bar_label(container, fmt="%.2f", padding=3, fontsize=8, color="#334155")
                        plt.xticks(rotation=0)
                        fig_cmp.tight_layout()
                        st.pyplot(fig_cmp, use_container_width=True)
                        plt.close(fig_cmp)

                    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                    with st.container(border=True):
                        render_card_header("Bảng kết quả đánh giá & ΔMAE chi tiết")
                        st.dataframe(impact_df, use_container_width=True, hide_index=True)
                else:
                    # Trường hợp hiện mới có 1 phiên bản (ví dụ D0_Demo): vẽ biểu đồ cột gọn theo từng mô hình
                    ver_label = str(mdf["Dataset_Version"].iloc[0]) if "Dataset_Version" in mdf.columns else "D0"
                    plot_df = mdf.sort_values("MAE", ascending=True).reset_index(drop=True)

                    with st.container(border=True):
                        render_card_header(
                            f"So sánh MAE giữa các mô hình ({ver_label})",
                            "Giá trị MAE càng thấp độ chính xác càng cao · Tự động mở rộng sang nhóm cột D0–D4 khi bổ sung các phiên bản khác",
                        )
                        fig_cmp, ax_cmp = plt.subplots(figsize=(9.5, 3.8), dpi=120)
                        colors = ["#059669" if i == 0 else "#2563EB" if i == 1 else "#64748B" for i in range(len(plot_df))]
                        x_labels = [str(m).replace("_", " ") for m in plot_df["Model_Name"]]
                        bars = ax_cmp.bar(x_labels, plot_df["MAE"], width=0.42, color=colors)
                        max_mae = float(plot_df["MAE"].max()) if not plot_df.empty else 1.0
                        ax_cmp.set_ylim(0, max_mae * 1.24)
                        style_matplotlib_ax(ax_cmp, ylabel="MAE (thấp hơn là tốt hơn)", xlabel="Mô hình dự báo")
                        for bar in bars:
                            h = bar.get_height()
                            ax_cmp.annotate(
                                f"{h:.4f}",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 5),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=9.5,
                                fontweight="bold",
                                color="#0F172A",
                            )
                        fig_cmp.tight_layout()
                        st.pyplot(fig_cmp, use_container_width=True)
                        plt.close(fig_cmp)

                    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                    with st.container(border=True):
                        render_card_header("Bảng kết quả đánh giá chi tiết")
                        st.dataframe(mdf, use_container_width=True, hide_index=True)
            else:
                with st.container(border=True):
                    st.dataframe(mdf, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 4: DATA VALIDATION & PROFILING
    # =========================================================================
    with tab_val:
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            render_kpi_card("Tổng bản ghi (Raw)", f"{len(raw_df):,}", "Dữ liệu trước làm sạch", tone="blue")
        with v2:
            render_kpi_card("Số thuộc tính", f"{raw_df.shape[1]}", "Cột dữ liệu đầu vào", tone="slate")
        with v3:
            render_kpi_card("Số SKU duy nhất", f"{raw_df['SKU_ID'].nunique():,}", "Mã sản phẩm quan sát", tone="emerald")
        with v4:
            render_kpi_card("Trạng thái Schema", "Hợp lệ", f"{len(schema_warnings)} cảnh báo nhẹ", tone="emerald" if not schema_warnings else "amber")

        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)
        for warn in schema_warnings:
            st.warning(warn)

        val_l, val_r = st.columns([1.15, 1.35], gap="medium")
        with val_l:
            with st.container(border=True):
                render_card_header("Hồ sơ kiểu dữ liệu & tỷ lệ thiếu")
                profile_df = pd.DataFrame(
                    {
                        "Cột": raw_df.columns,
                        "Kiểu dữ liệu": raw_df.dtypes.astype(str).values,
                        "Số thiếu": raw_df.isna().sum().values,
                        "Tỷ lệ thiếu (%)": (raw_df.isna().mean() * 100).round(2).values,
                        "Duy nhất": raw_df.nunique(dropna=True).values,
                    }
                )
                st.dataframe(profile_df, use_container_width=True, hide_index=True, height=340)

        with val_r:
            with st.container(border=True):
                render_card_header("Xem trước 20 dòng dữ liệu đầu tiên")
                st.dataframe(raw_df.head(20), use_container_width=True, height=340)

    # =========================================================================
    # TAB 5: EXPORT CSV TỔNG HỢP
    # =========================================================================
    with tab_exp:
        export_df = st.session_state.get("latest_forecast_df", pd.DataFrame())
        exp_model = st.session_state.get("latest_model_name", "MA4")
        exp_ver = st.session_state.get("latest_version", "D0")

        with st.container(border=True):
            render_card_header(
                "Xuất gói dữ liệu đầu ra chuẩn Mục 2.4 & Mục 9",
                "Gồm bảng dự báo SKU-Week (SKU_ID, Forecast_Week, Actual_Quantity, Predicted_Quantity, Model, Dataset_Version + tồn kho nếu có) và bảng tổng hợp CLDL.",
            )
            if not export_df.empty:
                st.dataframe(export_df.head(15), use_container_width=True, hide_index=True)

            ce1, ce2 = st.columns(2)
            if not export_df.empty:
                ce1.download_button(
                    label="⬇️ Tải bảng dự báo SKU-Week (.csv)",
                    data=export_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"sku_weekly_forecast_{exp_ver}_{exp_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary",
                )
            dq_export = pd.DataFrame(
                [{"Dimension": k, "Raw_Score_Pct": raw_scores[k], "Cleaned_Score_Pct": clean_scores[k]} for k in raw_scores.keys()]
            )
            ce2.download_button(
                label="⬇️ Tải báo cáo điểm CLDL (.csv)",
                data=dq_export.to_csv(index=False).encode("utf-8-sig"),
                file_name="data_quality_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()