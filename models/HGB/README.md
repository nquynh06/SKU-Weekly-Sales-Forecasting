# models/HGB — HistGradientBoosting SKU-Weekly Sales Forecasting

Pipeline độc lập cho mô hình **HistGradientBoostingRegressor**, đồng bộ 100% cấu trúc & lệnh chạy với `models/random forest/run.py` trong repo `SKU-Weekly-Sales-Forecasting`.  
Xem sơ đồ & tài liệu workflow chi tiết tại: `../../docs/hgb_pipeline_workflow.md`.

---

## 1. Cài đặt (chỉ cần 1 lần)
```powershell
pip install -r models/HGB/requirements.txt
python models/HGB/tests/test_hgb_pipeline.py
```

---

## 2. Chạy trực tiếp trên `data/raw/` (giống hệt Random Forest ở README Mục 9E)

```powershell
# Bước 1: Kiểm định dữ liệu gốc (27 CSV trong data/raw/)
python models/HGB/run.py validate --config models/HGB/configs/hgb.json --data-dir data/raw --output-dir reports/hgb/validate_001

# Bước 2: Chạy đầy đủ (có tuning trên Validation) & dự báo 28 ngày tới
python models/HGB/run.py run --config models/HGB/configs/hgb.json --data-dir data/raw --output-dir reports/hgb/run_001

# Hoặc chạy nhanh không tuning (~15s trên full 553.261 dòng dữ liệu thật):
python models/HGB/run.py run --config models/HGB/configs/hgb.json --data-dir data/raw --output-dir reports/hgb/run_fast --no-tune
```

---

## 3. Chạy trên file CSV đã tiền xử lý / D0–D4 / Chọn khoảng lịch sử & N ngày dự báo

```powershell
# Cách 1: Truyền file CSV trực tiếp (VD: sku_week_ground_truth_provisional.csv hoặc D0..D4)
python models/HGB/run.py run --data-file data/processed/sku_week_ground_truth_provisional.csv --dataset-version D0 --forecast-days 30

# Cách 2: Thả file CSV vào models/HGB/input_data/ rồi chạy thẳng:
python models/HGB/run.py

# Cách 3: Chọn khoảng thời gian lịch sử, scale nhu cầu & dự báo N ngày bất kỳ:
python models/HGB/run.py run --data-dir data/raw --history-start 2024-01-01 --history-end 2025-12-22 --demand-scale 1.10 --forecast-days 30
```

---

## 4. Kết quả kiểm chứng trên Full Dataset thật (`Electronics Sales Dataset`, 130 tuần)
- **HGB (`--no-tune`):** `MAE = 2.4790` | `RMSE = 3.5618` | `WAPE = 30.2545%` | `R2 = 0.7756`
- **MA4 (Baseline):** `MAE = 2.7711` | `RMSE = 3.9689` | `WAPE = 33.8199%` | `R2 = 0.7214`

---

## 5. Push nhánh riêng & Mở Pull Request vào `main`
```powershell
git checkout main
git pull origin main
git checkout -b feat/hgb-weekly-pipeline
git add models/HGB/ docs/hgb_pipeline_workflow.md docs/figures/hgb_pipeline_workflow.png
git commit -m "feat(hgb): add quality-gated HistGradientBoosting weekly SKU forecasting pipeline and workflow docs"
git push -u origin feat/hgb-weekly-pipeline
```
Mở Pull Request vào `main` và tag Thư ký **`@nquynh06`** duyệt.
