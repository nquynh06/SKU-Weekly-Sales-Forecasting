# Hướng dẫn lưu và nạp mô hình đã huấn luyện (`models/`)

Ứng dụng Streamlit (`app.py`) tuân thủ nguyên tắc **chỉ nạp mô hình đã huấn luyện sẵn**, tuyệt đối không huấn luyện hay tuning lại khi trình diễn.

## Quy ước ghép mô hình
1. Sau khi huấn luyện xong Random Forest hoặc HistGradientBoosting trên đúng 10 đặc trưng đầu vào:
   `['lag_1', 'lag_2', 'lag_4', 'lag_8', 'rolling_mean_4', 'rolling_mean_8', 'rolling_std_4', 'month', 'week_of_year', 'active_age']`
2. Lưu mô hình bằng `joblib.dump(model, "models/random_forest_d0.joblib")` hoặc `joblib.dump(model, "models/hist_gradient_boosting_d0.joblib")`.
3. Khởi chạy lại `streamlit run app.py` — mô hình mới sẽ tự động xuất hiện trong menu chọn mô hình ở Sidebar.
