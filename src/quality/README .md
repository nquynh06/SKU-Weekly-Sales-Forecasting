# Ground Truth & Data Quality — DS317

## Đã cập nhật

- Xây dựng `WeeklyQuantity` từ các dòng bán thuộc đơn hoàn tất; **không trừ hàng trả**.
- Chỉ gán nhãn `0` nếu có bằng chứng SKU đang hoạt động và log giao dịch của tuần đầy đủ. Tuần chưa xác minh giữ `NA`.
- Bổ sung báo cáo **6 chiều chất lượng**, EDA, kiểm định và gói bàn giao có checksum.
- Không huấn luyện mô hình ở module này. Dữ liệu nhằm phục vụ MA4, Random Forest và HistGradientBoosting theo **rolling one-step**.

## Chức năng các file

| Script | Vai trò |
|---|---|
| `data_quality.py` | Đánh giá 6 chiều chất lượng; thống kê độ phủ nhãn và vấn đề cần xử lý |
| `ground_truth_eda.py` | Tạo biểu đồ, bảng thống kê SKU–tuần (không coi `NA` là `0`) |
| `validate_ground_truth.py` | Kiểm tra khóa, nhãn, đối soát Quantity và báo cáo nghiệm thu |
| `finalize_ground_truth.py` | **Điểm chạy chính:** gọi Bước 4 → 5 → 6 → 7, xuất gói bàn giao |
| `review_ground_truth_evidence.py` | **Tùy chọn:** lập danh sách SKU/tuần cần bổ sung bằng chứng |
| `apply_ground_truth_evidence.py` | **Tùy chọn:** áp dụng kết quả rà soát đã có bằng chứng và người duyệt |

Các bước chuẩn bị dữ liệu nằm tại `src/data/` (như `clean_transactions.py`, `sku_lifecycle.py`, `build_ground_truth.py`).

## Run

pip install -r requirements.txt

python src/quality/finalize_ground_truth.py

python src/quality/validate_ground_truth.py

Lệnh `finalize_ground_truth.py` cần có:

- `reports/tables/sku_week_activity_candidates_reviewed.csv` (kết quả Bước 9).
- `clean_sales_transactions.csv` tại `data/processed/` **hoặc** `src/data/processed/`.
- Các báo cáo `raw_quality_report.csv`, `key_relationship_report.csv`, `transaction_cleaning_report.csv`, `sku_lifecycle_summary.csv` trong `reports/tables/`.
- File `src/data/build_ground_truth.py` và các script trong bảng trên.

Khi **cần rà soát bằng chứng mới**, thực hiện trước khi chạy `finalize_ground_truth.py`:

```powershell
python src/quality/review_ground_truth_evidence.py
# Điền và duyệt zero_evidence_review_template.csv bằng bằng chứng thực tế
python src/quality/apply_ground_truth_evidence.py
python src/quality/finalize_ground_truth.py
```

## Output

- `deliverables/ground_truth_review/data/`: Ground Truth SKU–tuần.
- `deliverables/ground_truth_review/tables/`: Data Quality, EDA và Validation reports.
- `deliverables/ground_truth_review/figures/`: Biểu đồ EDA.
- `deliverables/ground_truth_review/`: `README.md`, `handoff_status.json`, `source_manifest.json`, `logs/`.
- `deliverables/DS317_GroundTruth_Handoff_PROVISIONAL.zip`: Gói bàn giao.

**Trạng thái hiện tại:** `PROVISIONAL_BLOCKED` — 48.943 tuần chưa được xác minh nhãn `0`. Tổng Quantity quan sát được đã đối soát là **618.745**; không được tự động đổi `NA` thành `0` hoặc coi đây là ground truth đầy đủ.

