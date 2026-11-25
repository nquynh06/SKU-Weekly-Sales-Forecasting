# SKU-Weekly-Sales-Forecasting

## 1. Cấu trúc thư mục

```text
SKU-Weekly-Sales-Forecasting/
│
├── data/
│   ├── raw/                    # 27 CSV gốc, không chỉnh sửa
│   └── processed/              # CSV đã làm sạch và bảng JOIN
│
├── data/notebooks/
│   ├── 00_data_collection.ipynb
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_quality_assessment.ipynb
│   └── 03_data_preprocessing.ipynb
│
├── reports/                    # Báo cáo inventory, EDA, quality, preprocessing
├── plots/                      # Biểu đồ được sinh ra từ notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## 2. Cài đặt

Chạy lệnh tại thư mục gốc dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install jupyter
```

## 3. Chạy dữ liệu đầu vào

Sau khi clone repository, thư mục `data/raw/` có thể trống vì dữ liệu đã được
thêm vào `.gitignore`. Tải 27 file CSV của
Electronics Sales Dataset – B-Tech Style vào `data/raw/`, với tên file trùng tên
bảng, ví dụ `SalesOrders.csv`, `SalesOrderLines.csv` và `ProductVariants.csv`.

## 4. Chạy toàn bộ giai đoạn Lab 1–4

Mở VS Code tại thư mục gốc dự án, chọn kernel `.venv`, mở từng notebook và chạy
**Run All** theo thứ tự:

1. `data/notebooks/00_data_collection.ipynb`
2. `data/notebooks/01_data_understanding.ipynb`
3. `data/notebooks/02_data_quality_assessment.ipynb`
4. `data/notebooks/03_data_preprocessing.ipynb`

Luôn khởi động lại kernel nếu thay đổi môi trường Python hoặc chuyển máy.

## 5. File đầu ra chính

| Mục | Đường dẫn |
|---|---|
| Inventory 27 bảng | `reports/data_inventory.xlsx` |
| Quan hệ khóa ứng viên | `reports/relationship_candidates.xlsx` |
| Báo cáo hiểu dữ liệu | `reports/data_understanding_report.xlsx` |
| Biểu đồ EDA | `plots/*.png` |
| Báo cáo chất lượng | `reports/data_quality_report.xlsx` |
| Báo cáo tiền xử lý | `reports/preprocessing_report.xlsx` |
| 27 bảng đã làm sạch | `data/processed/*_cleaned.csv` |
| Dữ liệu JOIN bán hàng | `data/processed/joined_sales_data.csv` |
| Dòng trùng cần cách ly | `data/processed/quarantine/duplicates/` |

## 6. Quan hệ JOIN đã xác nhận

Bảng JOIN gốc là `SalesOrderLines`, với:

- `SalesOrderLines.SalesOrderID` → `SalesOrders.SalesOrderID`
- `SalesOrderLines.SKU_ID` → `ProductVariants.SKU_ID`
- `ProductVariants.ProductID` → `Products.ProductID`

Kết quả JOIN được lưu trong `data/processed/joined_sales_data.csv`.

## 7. Tái tạo data sau khi clone

Git intentionally does not track `data/raw/`, `data/processed/` và `plots/`.
Người khác khi clone project cần:

1. Tải lại 27 CSV vào `data/raw/`.
2. Cài dependency.
3. Chạy lần lượt 4 notebook.
4. Chạy tiếp các pipeline ở `models/` hoặc script trong `src/`.

Không commit dữ liệu raw hoặc processed nếu repository policy yêu cầu dữ liệu
ngoài Git. `data_sources.md` ghi nguồn và điều kiện sử dụng thay cho bản dữ liệu
được bỏ qua.

## 8. Chạy preprocessing có kiểm soát

Sau khi chạy 4 notebook, có thể chạy pipeline tiền xử lý phục vụ:

- tách dòng trùng hoàn toàn;
- thống kê giá trị thiếu theo bảng/cột;
- chuẩn hóa trạng thái đơn hàng;
- phát hiện quantity/giá/tồn kho ngoài IQR;
- kiểm tra khóa liên bảng;
- JOIN SalesOrderLines → SalesOrders → ProductVariants → Products.

```powershell
python src/data/preprocess_lab.py --raw-dir data/raw --processed-dir data/processed --report-dir reports/tables
```

Phép imputation không được thực hiện mặc định. Giá trị thiếu được giữ nguyên và
được báo cáo. Ngoại lai chỉ được cách ly để rà soát, không bị clip hoặc xóa ngay.

Kết quả kiểm định hiện tại:

- `clean_sales_transactions.csv`: 556.094 dòng hợp lệ.
- Không có dòng trạng thái khác `completed`.
- Không có `Quantity <= 0`.
- Không có `SalesOrderLineID` trùng.
- Bảng JOIN giữ nguyên 556.094 dòng sau ba phép JOIN.

## 9. Thứ tự chạy khi clone repository

Sau khi clone, dữ liệu raw/processed không có sẵn trên Git. Luồng chạy tối thiểu:

### A. Tải dữ liệu nguồn

Tải 27 CSV từ Kaggle hoặc snapshot đã được nhóm thống nhất, rồi đặt vào
`data/raw/`. Không sửa file trong thư mục này.

### B. Chỉ tạo báo cáo Lab 1–4

Mở VS Code, chọn `.venv`, chạy notebook theo thứ tự:

1. `00_data_collection.ipynb`
2. `01_data_understanding.ipynb`
3. `02_data_quality_assessment.ipynb`
4. `03_data_preprocessing.ipynb`

Sau nhóm này, các file chính nằm trong `reports/`, `plots/`,
`data/processed/` và các thư mục quarantine.

### C. Tạo dữ liệu Silver và kiểm tra nguồn

Chạy trực tiếp bằng lệnh sau thay vì chạy lại notebook 01–02:

```powershell
python src/data/audit_sources.py --data-dir data/raw --out-dir reports/tables
python src/data/clean_transactions.py --data-dir data/raw --out-dir data/processed --report-dir reports/tables
python src/data/preprocess_lab.py --raw-dir data/raw --processed-dir data/processed --report-dir reports/tables
```

File quan trọng cần giữ local:

- `data/processed/clean_sales_transactions.csv`
- `reports/tables/transaction_cleaning_report.csv`
- `reports/tables/excluded_sales_lines.csv`

### D. Tạo Gold SKU-tuần

```powershell
python src/data/sku_lifecycle.py --raw-dir data/raw --clean-file data/processed/clean_sales_transactions.csv --report-dir reports/tables --start-week 2023-07-03 --end-week 2025-12-22
python src/data/build_ground_truth.py --candidates reports/tables/sku_week_activity_candidates.csv --clean-file data/processed/clean_sales_transactions.csv --out-file data/processed/sku_week_ground_truth_provisional.csv --report-file reports/tables/ground_truth_build_report.csv
python src/quality/data_quality.py --raw-quality reports/tables/raw_quality_report.csv --key-report reports/tables/key_relationship_report.csv --clean-report reports/tables/transaction_cleaning_report.csv --lifecycle-summary reports/tables/sku_lifecycle_summary.csv --candidates reports/tables/sku_week_activity_candidates.csv --ground-truth data/processed/sku_week_ground_truth_provisional.csv --report-dir reports/tables
python src/quality/ground_truth_eda.py --ground-truth data/processed/sku_week_ground_truth_provisional.csv --dq-file reports/tables/data_quality_6_dimensions.csv --figure-dir reports/figures --table-dir reports/tables
```

Checkpoint bắt buộc sau bước này:

- `data/processed/sku_week_ground_truth_provisional.csv`
- `reports/tables/ground_truth_build_report.csv`
- `reports/tables/data_quality_6_dimensions.csv`
- `reports/tables/sku_lifecycle_summary.csv`

### E. Chạy Random Forest

Pipeline RF đọc `data/raw/` trực tiếp theo contract; không bắt buộc chạy lại
`joined_sales_data.csv`.

```powershell
python "models/random forest/run.py" validate --config "models/random forest/configs/random_forest.json" --data-dir data/raw --output-dir reports/random_forest/validate_001
python "models/random forest/run.py" run --config "models/random forest/configs/random_forest.json" --data-dir data/raw --output-dir reports/random_forest/run_001
```

## 10. Những file nên giữ lại

Khi làm lại trên máy khác, chỉ cần giữ:

1. `data/raw/27 CSV` hoặc một snapshot/hash manifest trên Google Drive.
2. `data/processed/clean_sales_transactions.csv`.
3. `data/processed/sku_week_ground_truth_provisional.csv`.
4. `reports/tables/` vì đây là audit trail nhỏ.
5. `reports/` và `plots/` nếu cần thuyết minh.
6. `reports/random_forest/` khi đã có kết quả thực nghiệm.

Không cần giữ lại:

- `data/processed/*_cleaned.csv` nếu đã có clean transaction và không dùng JOIN thô.
- `joined_sales_data.csv` nếu pipeline dùng raw contract.
- notebook output cache hoặc thư mục `.venv`.
- dataset trùng lặp và IQR outlier đã sinh lại được.

Không nên commit các file trên lên GitHub. Có thể lưu snapshot dưới Google
Drive và ghi checksum/ngày tải trong `reports/data_sources.md`; báo cáo nhỏ,
notebook và source code mới được push lên GitHub.