# Random Forest — dự báo lượng bán SKU–tuần

Pipeline local cho **cuối tuần t → lượng bán quan sát được của SKU trong tuần t+1**, H=1, đơn vị sản phẩm/SKU/tuần. Nhãn là gross completed units, không phải giá/doanh thu, net returns hoặc nhu cầu không bị giới hạn tồn kho.

**Trạng thái:** pipeline local của dự án SKU Weekly Sales Forecasting; giả định dữ liệu và ngưỡng local cần người dùng/nhóm duyệt. Contract hiện tại `quantity_rf_v1`, pipeline `1.0.0`. Không mặc định Random Forest thắng baseline và không coi saved-model smoke test là bằng chứng đạt KPI nghiệp vụ. Không có thao tác Git hoặc tự triển khai production trong pipeline.

## Tài liệu và workflow

- [Tài liệu đầy đủ tiếng Việt](<../../docs/random_forest_pipeline.md>): bảng/cột input, DQ/zero/missing, feature availability, temporal split, metrics, schema outputs, inference và checklist chờ duyệt.
- [Workflow PNG](<../../docs/figures/random_forest_workflow.png>) · [SVG](<../../docs/figures/random_forest_workflow.svg>) · [Nguồn Mermaid](<../../docs/figures/random_forest_workflow.mmd>).

## Input tối thiểu

Đặt cùng thư mục raw CSV:

| Bảng | Cột bắt buộc |
| --- | --- |
| `SalesOrders.csv` | `SalesOrderID`, `OrderDate`, `OrderStatus` |
| `SalesOrderLines.csv` | `SalesOrderLineID`, `SalesOrderID`, `LineNumber`, `SKU_ID`, `Quantity` |
| `ProductVariants.csv` | `SKU_ID` |
| `WeeklyCoverage.csv` — tùy chọn | `week_start`, `is_complete` |

Status sau strip/lower: `complete`/`completed` được giữ, `canceled`/`cancelled` bị loại có báo cáo. `Quantity` và `LineNumber` phải hữu hạn, nguyên, dương; `OrderDate` là ISO date-only `YYYY-MM-DD`. Malformed date, lỗi khóa/FK/Quantity/trạng thái không biết là lỗi cứng; ngày parse được ngoài cửa sổ bị loại có đếm dòng/units. Exact duplicate xét toàn dòng raw trước chọn cột; duplicate key còn lại bị từ chối, không tùy ý giữ first/last.

Thiếu coverage chỉ chạy khi config cho phép `assume_complete_weeks=true`; đây là **giả định local chưa được duyệt, không chứng thực feed đủ**. Sau first valid sale, tuần nguồn đủ nhưng không bán là 0; nguồn thiếu là missing kể cả có giao dịch dương. Không bfill, không cắt panel tại ngày bán cuối tương lai.

Không join giá/promo/inventory/returns/customer hoặc nhãn phụ. Không thêm cột numeric tùy ý vào X.

## Chạy từ repository root

Dùng Python environment riêng theo [requirements.txt](<requirements.txt>) (tham chiếu Python 3.12.14; pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0). Saved artifact cần đúng version scikit-learn đã tạo nó; không tự load khác version. Dùng dấu nháy vì thư mục model có khoảng trắng. Cách tạo environment và cài dependency được nêu ở [tài liệu cách chạy](<../../docs/random_forest_pipeline.md#10-cách-chạy-tại-local>).

```bash
# 1. Kiểm tra input/DQ; không huấn luyện.
python 'models/random forest/run.py' validate \
  --config 'models/random forest/configs/random_forest.json' \
  --data-dir '/duong-dan/snapshot/data/raw' \
  --output-dir 'reports/random_forest/validate_001'

# 2. Validation theo thời gian, test khóa, refit deploy và lưu artifact.
python 'models/random forest/run.py' run \
  --config 'models/random forest/configs/random_forest.json' \
  --data-dir '/duong-dan/snapshot/data/raw' \
  --output-dir 'reports/random_forest/local_run_001'

# 3. Nạp saved model, không train lại; origin → target + 7 ngày.
python 'models/random forest/run.py' predict \
  --config 'models/random forest/configs/random_forest.json' \
  --data-dir '/duong-dan/snapshot/data/raw' \
  --artifact 'reports/random_forest/local_run_001/model.joblib' \
  --origin-week '2025-12-22' \
  --output-dir 'reports/random_forest/predict_001'
```

Mỗi output directory phải **chưa tồn tại**, không overwrite. Origin `2025-12-22` là tuần đã chốt đến hết Chủ nhật 28/12, forecast target `2025-12-29`. `as_of` bằng `2025-12-29` là exclusive event cutoff, không phải ingest timestamp. Lệnh predict từ chối model fit sau origin, schema/contract khác, version scikit-learn khác hoặc đổi config `data_start_week`/`min_history_weeks`/`assume_complete_weeks`. Origin vượt `data_end_week` của artifact bắt buộc có `WeeklyCoverage.csv`; không tự kéo dài giả định complete sang tuần tương lai chưa có feed. Chỉ load artifact `joblib` đáng tin cậy vì định dạng pickle có thể thực thi mã.

## Protocol mặc định local

- Dữ liệu tuần `2023-07-03`–`2025-12-22`; split theo **target_week**, không random split hàng.
- Train qua `2025-03-24`; validation `2025-03-31`–`2025-08-04`, 3 rolling blocks mở rộng; test `2025-08-11`–`2025-12-22`.
- RF pooled + SKU one-hot train-only; 13 numeric features lag 1/2/4/8, rolling 4/8, zero-rate, tuổi SKU và calendar. RF cần `min_history_weeks=8` tuần liên tục known, không chỉ đủ tuổi lịch. SKU lạ vẫn đi RF nếu đủ history; `sku_seen_in_training=false` cảnh báo identity chưa được học.
- 2 cấu hình 80 cây/depth10 hoặc14; chọn bằng WAPE validation tổng hợp của RF+fallback (MAE nếu tổng actual bằng 0), không tune bằng test.
- Test một fit qua `validation_end_week`. **Artifact deploy refit tới `data_end_week` sau test là model khác**, không chấm lại test bằng artifact đã học test. Dữ liệu legacy từng được khám phá nên không bảo đảm test blind theo toàn lịch sử dự án.
- Naïve = quantity origin (origin missing → naïve missing). History ngắn/gappy → MA4 trên known last4; không có known week → `prediction_status=unavailable`, `qhat` trống.
- Báo `all_rows` với subset/coverage riêng và `common_keys` là giao actual + cả 3 prediction known để so cùng mẫu. MAE/RMSE theo units, WAPE/bias theo fraction; cả hai null khi tổng actual bằng 0. Có macro-SKU MAE/WAPE và coverage.

## Kiểm thử

```bash
python -m unittest discover \
  -s 'models/random forest/tests' -p 'test_*.py' -v
```

[Bộ unittest](<tests/test_random_forest.py>) dùng fixtures synthetic để kiểm correctness/DQ/cutoff/reload/fallback, không phải benchmark accuracy thực. Kết quả pass và runtime lấy từ lần chạy thật, không hardcode số test trong README.

## Outputs

Một `run` đầy đủ ghi DQ, weekly panel, features, validation/test predictions, forecast, metrics, feature importance, saved model và run manifest. `validate` chỉ xuất DQ/panel/features/manifest; `predict` thêm forecast nhưng không train hoặc tạo metrics/model mới. CSV dùng **`SKU_ID` chữ hoa**, predictions là wide columns `qhat` + `baseline_ma4` + `baseline_naive`. Xem [schema và ngữ nghĩa outputs](<../../docs/random_forest_pipeline.md#9-output-và-schema>) trước khi nối BI. Prediction missing không được tự fill thành 0; phải giữ status/fallback_reason và provenance.

## Checklist rà soát và phê duyệt

- [ ] Người dùng/nhóm duyệt quantity contract, feed completeness, cohort/zero policy, mốc split và fallback.
- [ ] Đối chiếu code/config/schema, tests + real-data smoke/reload và runtime máy đích; tách inference khỏi full training/evaluation khi đo demo.
- [ ] Đọc assumptions/giới hạn, giữ riêng kết quả synthetic và real; không tuyên bố KPI chưa đo.
- [ ] Rà diff cuối; không xuất bản raw data/model/cache ngoài chủ đích. `.matplotlib` chỉ là font cache, không chép/đưa vào Git.
- [x] Người dùng đã yêu cầu xuất bản ngày 22/11/2026; tạo nhánh `feat/random-forest-weekly-pipeline`.
- [ ] Push và mở PR vào `main` sau khi xác thực GitHub; tag đúng tài khoản thư ký để duyệt, không tự merge.

Bước nghiệp vụ sau forecast là người lập kế hoạch rà soát; không tự ra lệnh mua, tối ưu lượng nhập, khẳng định giảm tồn kho hoặc tăng lợi nhuận.
