"""
Phần 9 — Ứng dụng Streamlit Demo & Dashboard Chất lượng Dữ liệu (DS317)
=======================================================================
Đáp ứng trọn vẹn 5 nhiệm vụ và 6 chức năng demo theo đặc tả:
  1. Upload dữ liệu          : Tải CSV giao dịch (+ tùy chọn CSV tồn kho) hoặc dùng dữ liệu mẫu
  2. Data Validation         : Kiểm tra cấu trúc, kiểu dữ liệu, ngày lỗi, giá trị bất thường
  3. Data Quality Dashboard  : Hiển thị Completeness, Consistency, Accuracy, Timeliness & cảnh báo
  4. Forecasting             : Tiền xử lý đồng nhất + nạp mô hình đã huấn luyện sẵn (KHÔNG train lại)
  5. Model Comparison        : Hiển thị kết quả đánh giá các mô hình đã chạy (MAE, RMSE, WAPE, ΔMAE)
  6. Export CSV              : Xuất CSV gồm SKU, tuần dự báo, sản lượng dự báo & thông tin tồn kho

Khởi chạy:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from modules.dq_metrics import compute_dq_dashboard
from modules.forecasting import discover_and_load_models, predict_next_week
from modules.preprocessing import preprocess_uploaded_sales
from modules.validation import validate_sales_csv

APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "models"
DATA_DIR = APP_ROOT / "data"
SAMPLE_DIR = APP_ROOT / "sample_data"

st.set_page_config(
    page_title="DS317 — SKU Sales Forecasting & Data Quality Dashboard",
    page_icon="📊",
    layout="wide",
)

# ==============================================================================
# CACHED LOADERS (PRE-TRAINED MODELS & EVALUATION TABLES)
# ==============================================================================
@st.cache_resource
def get_pretrained_models():
    return discover_and_load_models(MODELS_DIR)


def load_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


# ==============================================================================
# SIDEBAR — INPUT DATA & PRE-TRAINED MODEL SELECTION
# ==============================================================================
st.sidebar.title("⚙️ Cấu hình Demo (Phần 9)")
st.sidebar.caption("Dự báo sản lượng SKU–tuần & Giám sát CLDL")

input_mode = st.sidebar.radio(
    "1. Chọn nguồn dữ liệu CSV:",
    options=["Dùng dữ liệu mẫu thật (30 SKU)", "Tải file CSV từ máy tính"],
    index=0,
)

sales_df: Optional[pd.DataFrame] = None
inv_df: Optional[pd.DataFrame] = None

if input_mode == "Dùng dữ liệu mẫu thật (30 SKU)":
    sales_df = load_optional_csv(SAMPLE_DIR / "sales_sample.csv")
    inv_df = load_optional_csv(SAMPLE_DIR / "inventory_sample.csv")
    if sales_df is None:
        st.sidebar.error("Không tìm thấy `sample_data/sales_sample.csv`.")
else:
    up_sales = st.sidebar.file_uploader(
        "Tải CSV giao dịch bán hàng (SKU_ID, OrderDate, Quantity)",
        type=["csv"],
    )
    up_inv = st.sidebar.file_uploader(
        "Tải CSV tồn kho InventoryBalances (tùy chọn)",
        type=["csv"],
    )
    if up_sales is not None:
        sales_df = pd.read_csv(up_sales)
    if up_inv is not None:
        inv_df = pd.read_csv(up_inv)

st.sidebar.divider()
models_registry = get_pretrained_models()
if not models_registry:
    st.sidebar.error("Chưa có file mô hình `.joblib` hoặc `.pkl` trong thư mục `models/`.")
    selected_model_key = None
else:
    selected_model_key = st.sidebar.selectbox(
        "2. Chọn mô hình đã huấn luyện sẵn:",
        options=list(models_registry.keys()),
    )
    st.sidebar.info(
        "Mô hình được nạp trực tiếp từ thư mục `models/` — không huấn luyện hay tinh chỉnh lại trong lúc trình diễn."
    )

# ==============================================================================
# HEADER & 6-FEATURE NAVIGATION TABS
# ==============================================================================
st.title("📊 SKU-Weekly Sales Forecasting & Data Quality Dashboard")
st.caption(
    "Sản phẩm Demo Phần 9 — Tích hợp Data Validation, Dashboard CLDL, Tiền xử lý dùng chung, "
    "Dự báo bằng mô hình đã huấn luyện sẵn, So sánh mô hình và Xuất CSV."
)

tab_upload, tab_dq, tab_forecast, tab_compare, tab_export = st.tabs(
    [
        "1. Upload & Data Validation",
        "2. Data Quality Dashboard",
        "3. Forecasting (Tuần kế tiếp)",
        "4. Model Comparison",
        "5. Export CSV",
    ]
)

# Compute shared pipeline outputs once per upload/model selection
val_report = validate_sales_csv(sales_df) if sales_df is not None else None
dq_result = compute_dq_dashboard(sales_df) if sales_df is not None else None

panel_df = None
infer_df = None
prep_audit = None
forecast_df = None

if sales_df is not None and val_report is not None and val_report.is_valid:
    panel_df, infer_df, prep_audit = preprocess_uploaded_sales(sales_df)
    if selected_model_key and not infer_df.empty:
        entry = models_registry[selected_model_key]
        if entry["artifact"] is not None:
            forecast_df = predict_next_week(
                entry,
                infer_df,
                inventory_df=inv_df,
                model_label=selected_model_key,
            )

# ==============================================================================
# TAB 1: UPLOAD DỮ LIỆU & DATA VALIDATION
# ==============================================================================
with tab_upload:
    st.subheader("1. Kiểm tra dữ liệu đầu vào (Upload & Data Validation)")
    if sales_df is None or val_report is None:
        st.warning("Vui lòng chọn dữ liệu mẫu hoặc tải file CSV từ thanh bên trái.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số dòng dữ liệu", f"{val_report.n_rows:,}")
        c2.metric("Số cột", f"{val_report.n_cols}")
        c3.metric("Số SKU duy nhất", f"{val_report.n_skus:,}")
        c4.metric("Kết quả kiểm tra", "Hợp lệ ✅" if val_report.is_valid else "Không hợp lệ ❌")

        st.markdown("**Cảnh báo & Kết quả kiểm tra quy tắc:**")
        if not val_report.issues:
            st.success("Không phát hiện lỗi cấu trúc, kiểu dữ liệu hoặc giá trị bất thường.")
        else:
            for issue in val_report.issues:
                msg = f"**[{issue.column} — {issue.rule}]** {issue.detail}"
                if issue.level == "ERROR":
                    st.error(msg)
                elif issue.level == "WARNING":
                    st.warning(msg)
                else:
                    st.info(msg)

        col_s, col_p = st.columns([1, 1.3])
        with col_s:
            st.markdown("**Cấu trúc cột & Kiểu dữ liệu:**")
            st.dataframe(val_report.schema_table, use_container_width=True, hide_index=True)
        with col_p:
            st.markdown("**Xem trước 10 dòng đầu tiên:**")
            st.dataframe(sales_df.head(10), use_container_width=True)

        if prep_audit is not None:
            st.markdown("**Nhật ký tiền xử lý đồng nhất với quy trình huấn luyện:**")
            audit_df = pd.DataFrame(
                [{"Bước / Chỉ số": k, "Số lượng": v} for k, v in prep_audit.items()]
            )
            st.dataframe(audit_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: DATA QUALITY DASHBOARD
# ==============================================================================
with tab_dq:
    st.subheader("2. Dashboard Theo dõi Chất lượng Dữ liệu (CLDL)")
    if dq_result is None:
        st.warning("Chưa có dữ liệu đầu vào để đánh giá chất lượng dữ liệu.")
    else:
        st.markdown("**4 chiều chất lượng dữ liệu trọng tâm:**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Completeness (Đầy đủ)", f"{dq_result.scores['Completeness']:.2f}%")
        k2.metric("Consistency (Nhất quán)", f"{dq_result.scores['Consistency']:.2f}%")
        k3.metric("Accuracy (Chính xác)", f"{dq_result.scores['Accuracy']:.2f}%")
        k4.metric("Timeliness (Kịp thời)", f"{dq_result.scores['Timeliness']:.2f}%")

        s1, s2 = st.columns(2)
        s1.metric("Validity (Hợp lệ — bổ sung)", f"{dq_result.scores['Validity']:.2f}%")
        s2.metric("Uniqueness (Duy nhất — bổ sung)", f"{dq_result.scores['Uniqueness']:.2f}%")

        col_left, col_right = st.columns([1.1, 1.2])
        with col_left:
            st.markdown("**Bảng điểm 6 chiều CLDL:**")
            st.dataframe(dq_result.summary_df, use_container_width=True, hide_index=True)
        with col_right:
            st.markdown("**Biểu đồ điểm chất lượng dữ liệu (%):**")
            st.bar_chart(dq_result.summary_df.set_index("Chiều CLDL")[["Điểm (%)"]])

        st.markdown("**Tỷ lệ giá trị thiếu theo từng trường dữ liệu:**")
        st.dataframe(dq_result.missing_by_col_df, use_container_width=True, hide_index=True)

        st.markdown("**Danh sách cảnh báo CLDL:**")
        if dq_result.alerts:
            alerts_df = pd.DataFrame(
                [
                    {
                        "Chiều CLDL": a.dimension,
                        "Mức độ": a.severity,
                        "Trường dữ liệu": a.field,
                        "Số dòng ảnh hưởng": a.affected_rows,
                        "Tỷ lệ (%)": a.affected_pct,
                        "Mô tả cảnh báo": a.message,
                    }
                    for a in dq_result.alerts
                ]
            )
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
        else:
            st.success("Dữ liệu không phát sinh cảnh báo vi phạm CLDL.")

# ==============================================================================
# TAB 3: FORECASTING (DỰ BÁO TUẦN KẾ TIẾP THEO SKU)
# ==============================================================================
with tab_forecast:
    st.subheader("3. Dự báo sản lượng tuần kế tiếp theo SKU")
    if forecast_df is None or panel_df is None:
        st.warning("Vui lòng nạp dữ liệu hợp lệ (Tab 1) và chọn mô hình đã lưu ở thanh bên trái.")
    else:
        fw = forecast_df["Forecast_Week"].iloc[0]
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Tuần dự báo (t+1)", fw)
        f2.metric("Số SKU dự báo", f"{len(forecast_df):,}")
        f3.metric("Tổng sản lượng dự báo", f"{forecast_df['Predicted_Quantity'].sum():,.1f}")
        if "Inventory_Risk" in forecast_df.columns:
            risk_n = int(
                forecast_df["Inventory_Risk"].isin(["Critical (<1 tuần)", "Warning (1–2 tuần)"]).sum()
            )
            f4.metric("SKU nguy cơ thiếu hàng (<2 tuần)", f"{risk_n:,}")
        else:
            f4.metric("Mô hình đang dùng", selected_model_key.split(" (")[0])

        # Interactive per-SKU history + next-week forecast chart
        sku_list = sorted(forecast_df["SKU_ID"].unique())
        chosen_sku = st.selectbox("Chọn mã SKU để xem chuỗi lịch sử và điểm dự báo tuần kế tiếp:", sku_list)

        hist = panel_df[panel_df["SKU_ID"] == chosen_sku][["WeekStart", "Quantity"]].copy()
        hist = hist.rename(columns={"WeekStart": "Tuần", "Quantity": "Thực tế"})
        hist["Dự báo (t+1)"] = np.nan

        pred_val = float(forecast_df.loc[forecast_df["SKU_ID"] == chosen_sku, "Predicted_Quantity"].iloc[0])
        next_pt = pd.DataFrame([{"Tuần": pd.Timestamp(fw), "Thực tế": np.nan, "Dự báo (t+1)": pred_val}])
        chart_df = pd.concat([hist, next_pt], ignore_index=True).set_index("Tuần")
        st.line_chart(chart_df)

        st.markdown("**Bảng kết quả dự báo theo từng SKU (đã đảm bảo không âm):**")
        st.dataframe(
            forecast_df.sort_values("Predicted_Quantity", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# ==============================================================================
# TAB 4: MODEL COMPARISON (SO SÁNH CÁC MÔ HÌNH ĐÃ CHẠY)
# ==============================================================================
with tab_compare:
    st.subheader("4. So sánh kết quả đánh giá các mô hình đã chạy")
    comp_df = load_optional_csv(DATA_DIR / "model_comparison.csv")
    if comp_df is None or comp_df.empty:
        st.info("Chưa tìm thấy tệp `data/model_comparison.csv`. Hãy xuất bảng đánh giá từ nhóm mô hình vào đây.")
    else:
        st.markdown(
            "Bảng tổng hợp kết quả đánh giá trên cùng tập Test (`MAE`, `RMSE`, `WAPE`, `ΔMAE` so với `D0`). "
            "Khi nhóm hoàn tất chạy Random Forest và HistGradientBoosting, chỉ cần cập nhật file `data/model_comparison.csv`."
        )
        # Filter by Dataset Version if present
        if "Dataset_Version" in comp_df.columns:
            versions = ["Tất cả"] + sorted(comp_df["Dataset_Version"].dropna().unique().tolist())
            chosen_ver = st.selectbox("Lọc theo phiên bản dữ liệu (D0–D4):", versions)
            view_df = comp_df if chosen_ver == "Tất cả" else comp_df[comp_df["Dataset_Version"] == chosen_ver]
        else:
            view_df = comp_df

        st.dataframe(view_df, use_container_width=True, hide_index=True)

        # Bar chart of MAE across models/versions (exclude pending NaN rows)
        plot_src = view_df.dropna(subset=["MAE"]).copy()
        if not plot_src.empty:
            plot_src["Nhãn"] = (
                plot_src["Model"].astype(str) + " (" + plot_src["Dataset_Version"].astype(str) + ")"
                if "Dataset_Version" in plot_src.columns
                else plot_src["Model"].astype(str)
            )
            st.markdown("**Biểu đồ so sánh MAE & RMSE:**")
            st.bar_chart(plot_src.set_index("Nhãn")[["MAE", "RMSE"]])

# ==============================================================================
# TAB 5: EXPORT CSV
# ==============================================================================
with tab_export:
    st.subheader("5. Xuất kết quả dự báo ra tệp CSV")
    if forecast_df is None:
        st.warning("Chưa có dữ liệu dự báo để xuất file.")
    else:
        st.markdown(
            "Tệp CSV xuất ra bao gồm: `SKU_ID`, `Forecast_Week`, `Predicted_Quantity`, `Model` "
            "và tự động đính kèm thông tin tồn kho (`AvailableQuantity`, `Stock_Cover_Weeks`, `Inventory_Risk`) nếu có."
        )
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        csv_data = forecast_df.to_csv(index=False).encode("utf-8-sig")
        out_name = f"forecast_sku_week_{forecast_df['Forecast_Week'].iloc[0]}.csv"
        st.download_button(
            label="⬇️ Tải xuống kết quả dự báo (.csv)",
            data=csv_data,
            file_name=out_name,
            mime="text/csv",
        )