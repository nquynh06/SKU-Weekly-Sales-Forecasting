# Phần 9 — Ứng dụng Streamlit Demo & Dashboard CLDL (DS317)

Gói sản phẩm bàn giao cho **Giai đoạn 9 (Deployment & Demo)** gồm đầy đủ 5 sản phẩm đầu ra theo yêu cầu:
1. **Dashboard CLDL** (Completeness, Consistency, Accuracy, Timeliness + cảnh báo tự động).
2. **Ứng dụng Streamlit** (`app.py`) tích hợp đủ 6 chức năng: Upload CSV, Data Validation, Data Quality Dashboard, Forecasting, Model Comparison, Export CSV.
3. **Mô hình đã lưu** (`models/ma4_pretrained.pkl` sẵn sàng chạy ngay; hỗ trợ gắn thêm `random_forest.joblib` và `hist_gradient_boosting.joblib`).
4. **README & requirements.txt**.
5. **Kịch bản quay Video demo dự phòng** (trong tài liệu thuyết minh đi kèm).

## 1. Cấu trúc thư mục
```text
Part9_Streamlit_Demo/
├── app.py                      # Ứng dụng Streamlit chính (5 tab bao phủ 6 chức năng)
├── requirements.txt            # Thư viện cần cài đặt
├── README.md                   # Hướng dẫn chạy & ghép mô hình của nhóm
├── modules/
│   ├── validation.py           # Kiểm tra cấu trúc, kiểu dữ liệu, giá trị bất thường
│   ├── dq_metrics.py           # Tính 4 chiều CLDL trọng tâm (+2 bổ sung) và cảnh báo
│   ├── preprocessing.py        # Tiền xử lý & tạo 10 đặc trưng đồng nhất lúc huấn luyện
│   └── forecasting.py          # Nạp mô hình đã lưu (.pkl/.joblib) và dự báo tuần kế tiếp
├── models/
│   ├── ma4_pretrained.pkl      # Mô hình MA4 đã huấn luyện sẵn từ Electronics Sales Dataset
│   └── README_MODELS.md        # Hướng dẫn bỏ thêm file .joblib của RF / HistGradientBoosting
├── data/
│   └── model_comparison.csv    # Bảng kết quả đánh giá các mô hình đã chạy (hiển thị ở Tab 4)
└── sample_data/
    ├── sales_sample.csv        # Dữ liệu mẫu thật (30 SKU) để upload kiểm thử ngay
    └── inventory_sample.csv    # Dữ liệu tồn kho mẫu (30 SKU) để tính Stock Cover & Export CSV
```

## 2. Cách cài đặt và chạy nhanh
```bash
pip install -r requirements.txt
streamlit run app.py
```
Mở trình duyệt tại `http://localhost:8501`. Bạn có thể chọn **"Dùng dữ liệu mẫu thật (30 SKU)"** ở thanh bên trái để kiểm tra ngay toàn bộ 6 chức năng mà không cần chuẩn bị thêm file.

## 3. Cách ghép mô hình & bảng so sánh từ các thành viên làm Phần 7–8
- **Ghép mô hình đã huấn luyện:** Chép file `random_forest.joblib` và `hist_gradient_boosting.joblib` (được huấn luyện trên đúng 10 đặc trưng: `lag_1`, `lag_2`, `lag_4`, `lag_8`, `rolling_mean_4`, `rolling_mean_8`, `rolling_std_4`, `month`, `week_of_year`, `active_age`) vào thư mục `models/`. Ứng dụng tự động nhận diện và hiển thị trong hộp chọn ở thanh bên trái (**không train hay tuning lại khi demo**).
- **Cập nhật bảng Model Comparison:** Chép bảng kết quả đánh giá của nhóm (các cột `Model`, `Dataset_Version`, `MAE`, `RMSE`, `WAPE_pct`, `Delta_MAE_vs_D0`) đè lên file `data/model_comparison.csv`. Tab 4 sẽ tự động cập nhật bảng và biểu đồ.
