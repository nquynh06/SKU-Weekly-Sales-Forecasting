# Random Forest — kết quả chạy local và checklist chờ duyệt

**Ngày kiểm tra:** 22/11/2026 (Asia/Ho_Chi_Minh). **Repo làm chính:** SKU-Weekly-Sales-Forecasting local, remote `https://github.com/nquynh06/SKU-Weekly-Sales-Forecasting.git`.

**Trạng thái:** đã triển khai và chạy kiểm thử local. Ngày 22/11/2026, người dùng đã yêu cầu tạo nhánh riêng, commit/push, mở PR vào `main` và tag thư ký. Nhánh triển khai là `feat/random-forest-weekly-pipeline`; không commit trực tiếp vào `main`. Xuất bản GitHub cần tài khoản đã xác thực và username thư ký chính xác; trạng thái PR/review là công việc riêng, không được suy từ việc có code hoặc commit local. Không tự merge PR.

## 1. Đối chiếu đủ ba yêu cầu

| Yêu cầu | Đã làm | Nơi kiểm tra |
| --- | --- | --- |
| Dựng pipeline Random Forest phù hợp bài toán, trong folder được yêu cầu | Pipeline quantity SKU–tuần H=1, raw → DQ → panel → features → validation → test → saved model → next-week forecast | [README mô hình](<../models/random forest/README.md>), [entrypoint](<../models/random forest/run.py>), [mã nguồn](<../models/random forest/rf_pipeline/>), [config](<../models/random forest/configs/random_forest.json>) |
| Viết docs pipeline và vẽ workflow | Tài liệu tiếng Việt đối chiếu code, luồng train và saved-model inference, hình vector/raster và nguồn sửa được | [Tài liệu pipeline](<random_forest_pipeline.md>), [SVG](<figures/random_forest_workflow.svg>), [PNG](<figures/random_forest_workflow.png>), [Mermaid](<figures/random_forest_workflow.mmd>) |
| Nêu rõ cần file/bảng gì, output gì, xử lý thế nào để ra dự đoán | Danh sách cột/khóa/input, quality gate, nhãn gross-completed units, zero vs missing, lag/rolling/calendar, RF/fallback, output schemas và lệnh chạy | [Data contract, workflow và hướng dẫn](<random_forest_pipeline.md>) |
| Nhánh riêng → PR main → tag thư ký | Đã được người dùng yêu cầu; nhánh `feat/random-forest-weekly-pipeline` đã tạo. Push/PR cần xác thực GitHub, tag cần đúng username thư ký; không tự đoán người duyệt | Checklist mục 8 |

## 2. Đã đọc gì từ DS317 và dùng theo hướng nào?

- Đã đọc phạm vi hiện tại cùng đặc tả nghiệp vụ/protocol quantity trong repo DS317, kiểm tra schema và thống kê các bảng raw. Không sửa nguồn DS317.
- Mục tiêu giữ đúng: **cuối tuần t dự báo tổng số sản phẩm bán hợp lệ theo SKU trong đúng tuần t+1**. Không dự báo giá, doanh thu tháng hoặc nhãn rủi ro tồn kho.
- Không dùng model RF/doanh thu hay processed/interim cũ làm quantity gold. Tự dựng nhãn từ các bảng raw và đối soát lại tổng units.
- Chi tiết và provenance local nằm trong [tài liệu pipeline](<random_forest_pipeline.md>); repo GitHub không phụ thuộc đường dẫn máy cá nhân để hiểu đặc tả, các lệnh có thể thay `--data-dir`.

## 3. Input → xử lý → output tóm tắt

**Bắt buộc trong `--data-dir`:**

1. `SalesOrders.csv`: `SalesOrderID`, `OrderDate`, `OrderStatus`.
2. `SalesOrderLines.csv`: `SalesOrderLineID`, `SalesOrderID`, `LineNumber`, `SKU_ID`, `Quantity`.
3. `ProductVariants.csv`: `SKU_ID` (kiểm tra FK, không dùng full catalog để chọn cohort quá khứ).

**Coverage tùy chọn:** `WeeklyCoverage.csv`: `week_start`, `is_complete`. Nếu không có, chỉ tiếp tục khi config nêu rõ giả định `assume_complete_weeks=true`. Config local hiện bật giả định này; nó chưa được kiểm chứng/phê duyệt và không được hiểu là đã đo feed completeness.

**Cách xử lý:** dedup toàn dòng → kiểm khóa/cột/FK/status/quantity/date → loại cancelled và ngày ngoài full-week window → join many-to-one bảo toàn units → sum gross completed quantity theo SKU/tuần thứ Hai → panel sau first-sale, không cắt ở last-sale tương lai → zero chỉ khi feed đầy đủ, thiếu là NaN → features quá khứ + calendar → one-hot fit trên train + Random Forest → nonnegative qhat; thiếu lịch sử thì MA4 hoặc unavailable.

**Output chính:** bảng forecast có `SKU_ID`, `origin_week`, `target_week`, `qhat`, baseline, cờ lịch sử, `prediction_status`, `fallback_reason`, `model_id`, `run_id`, `as_of`, `contract_version`; saved preprocessor/model và artifacts đối soát/đánh giá. Actual tuần đích không nằm trong output forecast hoặc feature allowlist của mô hình.

## 4. Lần chạy dữ liệu thật

**Run ID:** `rf-65298bf87425`; pipeline `1.0.0`, contract `quantity_rf_v1`.

[Manifest](<../reports/random_forest/approved_for_review_run/run_manifest.json>), [DQ report](<../reports/random_forest/approved_for_review_run/dq_report.json>), [bản kiểm tra artifact](<../reports/random_forest/approved_for_review_run/verification_summary.json>) là artifacts **chỉ ở local, được ignore khỏi Git**. Các link này sẽ không mở trên GitHub nếu người dùng không tự tái chạy; bảng tóm tắt bên dưới vẫn nằm trong tài liệu để review PR sau này.

| Hạng mục | Số đo |
| --- | ---: |
| Raw SalesOrders / SalesOrderLines / ProductVariants | 322.577 / 579.572 / 902 dòng |
| Duplicate toàn dòng đơn hàng loại bỏ | 845 |
| Dòng bán completed hợp lệ | 553.261 |
| Tổng gross-completed units | 618.745 |
| SKU đã có lịch sử hợp lệ | 871 |
| SKU–tuần có giao dịch quan sát | 68.317 |
| Panel sau first-sale + zero policy | 84.639 hàng |
| Calendar weeks | 130 |
| Dòng validation mỗi ứng viên | 14.609 |
| Dòng test | 16.730 |

Đối soát đã pass: units trước join bằng units sau join; joined units = units ngoài cửa sổ + units cancelled trong cửa sổ + gross-completed units; tổng panel known units bằng tổng giao dịch hợp lệ theo giả định coverage hiện tại. Khóa SKU/origin không trùng; target = origin + 7 ngày; target dự đoán sau training cutoff; source hashes khớp mã đang chạy.

### 4.1. Split và model đã chọn

- Cửa sổ raw: tuần bắt đầu `2023-07-03` đến `2025-12-22` (bao gồm đủ tuần cuối tới hết `2025-12-28`).
- Train ban đầu: target ≤ `2025-03-24`.
- Validation 3 block mở rộng: `2025-03-31…2025-05-12`; `2025-05-19…2025-06-23`; `2025-06-30…2025-08-04`.
- Final test: target `2025-08-11…2025-12-22`; forest đánh giá fit qua `2025-08-04`, cố định trong test. Actual mới quan sát chỉ cập nhật context H=1.
- Cấu hình chọn bằng validation: 80 cây, `max_depth=10`, `min_samples_leaf=5`, `max_features=0.8`, seed 317. Không chọn theo test.
- Sau đánh giá, model deploy được refit target qua `2025-12-22`; không lấy artifact deploy chấm lại test.

### 4.2. Metrics đã đo, không lấy điểm bài toán cũ

**Validation — cấu hình được chọn:**

| Phương pháp | MAE (units) | RMSE (units) | WAPE |
| --- | ---: | ---: | ---: |
| RF + fallback MA4 | 2,0816 | 3,2620 | 31,7630% |
| MA4 | 2,1805 | 3,4370 | 33,2717% |
| Naïve | 2,7640 | 4,5071 | 42,1767% |

**Final temporal test — cùng 16.730 khóa:**

| Phương pháp | MAE (units) | RMSE (units) | WAPE | Macro-SKU WAPE |
| --- | ---: | ---: | ---: | ---: |
| RF + fallback MA4 | 2,0823 | 3,3071 | 31,9988% | 53,5376% |
| MA4 | 2,2390 | 3,5855 | 34,4080% | 52,6287% |
| Naïve | 2,7230 | 4,4811 | 41,8452% | 60,5109% |

Đọc kết quả trung thực:

- RF giảm khoảng **7,00% tương đối WAPE test** so với MA4, nhưng validation giảm khoảng **4,53%** — chưa đạt ngưỡng thử nghiệm 5% validation trong đặc tả đề xuất (ngưỡng đó cũng chưa được duyệt).
- Macro-SKU WAPE của RF **kém MA4**; không nói RF thắng mọi chỉ số/mọi SKU. Macro-WAPE loại 152 SKU có tổng actual test bằng 0 và báo riêng số loại, không thêm epsilon. Macro-SKU MAE RF = 2,1260, MA4 = 2,2791; bias RF ≈ −1,5972% tổng actual.
- Test system gồm 16.224 hàng RF và 506 hàng fallback MA4; tên score là **RF+fallback**, không giả vờ 100% do RF tạo ra.
- Prediction/label coverage ở lần chạy này đều 100% **trên cohort đã có first sale và dưới giả định feed đủ**. Không phải coverage toàn catalog hoặc bằng chứng dữ liệu thực không thiếu.
- Đây là backtest local trên final snapshot, không chứng minh production accuracy, tính ý nghĩa thống kê, lợi nhuận/tồn kho cải thiện hay vô hiệu hóa giới hạn status/ingestion history. Dữ liệu cũ đã được khảo sát trước nên không gọi test hoàn toàn blind.

Chi tiết theo tuần/coverage/macro nằm trong [metrics local](<../reports/random_forest/approved_for_review_run/metrics.json>) và [prediction test local](<../reports/random_forest/approved_for_review_run/test_predictions.csv>).

### 4.3. Dự báo và saved-model inference

[Forecast local](<../reports/random_forest/approved_for_review_run/forecast.csv>) có **871 hàng** cho origin `2025-12-22`, target `2025-12-29`: **849 RF + 22 fallback MA4**. Không có actual của tuần đích trong forecast. Đây là replay minh họa từ snapshot 2025, **không phải dự báo thời gian hiện tại tháng 11/2026**; muốn dự báo tuần hiện tại cần snapshot mới đã đóng tuần.

Ví dụ: SKU `SKU0000001` có quantity origin 12, MA4 11,75 và RF forecast ≈ 9,6294 units cho tuần đích; số thực không bị làm tròn trước scoring.

- Full run bàn giao trong repo (load/DQ/features, 2 candidates × 3 validation fits, test fit, deploy refit, export): **143,76 giây**. Các lượt kiểm tra trước dao động khoảng **145,50–180,63 giây**; không bảo đảm mọi lần ≤180 giây hoặc xem training là yêu cầu demo inference.
- Saved-model CLI predict bản bàn giao trên cùng raw, tính cả startup: **36,77 giây**, 871 hàng (các lượt trước khoảng 26,47–28,34 giây; runtime phụ thuộc tải máy).
- Reload artifact cho kết quả khớp trong tolerance `rtol=atol=1e-12`; sai khác lớn nhất khoảng `1,42e-14` do số học floating point, không nói bitwise identical.
- Bằng chứng: [manifest predict](<../reports/random_forest/approved_for_review_predict/run_manifest.json>), [reload verification](<../reports/random_forest/approved_for_review_predict/reload_verification.json>) local.

Chỉ load model artifact đáng tin cậy vì joblib/pickle có khả năng thực thi mã. Inference từ chối origin trước training cutoff và version/schema không tương thích. Origin sau `data_end_week` của artifact bắt buộc có bảng coverage; không tự kéo dài giả định complete của snapshot cũ để biến tuần tương lai thành zero.

## 5. Kiểm thử

**23/23 unittest pass**, gồm:

- Config/date/feature horizon.
- Exact duplicate không tăng units; conflicting PK/business key và orphan FK phải dừng.
- Status alias completed/cancelled; unknown status không lọt whitelist.
- Quantity zero/negative/fraction/NaN/Inf bị từ chối; date malformed vs out-of-window tách riêng.
- Complete-week zero khác missing feed; thiếu coverage không tự được gọi complete; không pad trước first sale.
- Lag/rolling alignment, nhãn t+1; thay đổi quantity tương lai ở panel **và raw** không đổi features quá khứ.
- All-zero actual → WAPE/bias null, không epsilon.
- Short-history MA4, không có lịch sử gần → unavailable/ô qhat trống, không ép 0.
- Unknown SKU trong encoder được gắn cờ, vẫn dự đoán RF nếu history đủ.
- Temporal validation/test cutoff, saved model reload; từ chối backdated inference và overwrite output cũ.
- CLI failure trả exit code 2 và diagnostics; không coi run lỗi là thành công.
- Test labels và prediction phải giao nhau trên ít nhất một hàng; coverage gap không được làm run evaluation thành công với zero scored rows.
- `max_features` mang nghĩa tỷ lệ: JSON `1` và `1.0` đều được chuyển thành float 100% feature trước truyền sklearn.

Review độc lập không phát hiện P0/P1 hoặc leakage target/encoder/split trong phần mã đã đọc; hai edge cases P2 nêu trên đã được sửa và thêm regression tests. Đây là phạm vi review kỹ thuật, không phải chứng nhận dữ liệu point-in-time đã hoàn chỉnh.

[Test source](<../models/random forest/tests/test_random_forest.py>) dùng fixtures synthetic offline, **không** được hiểu là score dữ liệu thật. [Log local](<../reports/random_forest/review_tests.log>) là log lần chạy trong repo. `git diff --check` pass sau khi rà các file tracked.

Môi trường đã dùng: Python 3.12.14, pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0. [Requirements RF](<../models/random forest/requirements.txt>) pin dependency tương ứng; chưa kiểm chứng một clean install Colab/máy khác, không tuyên bố universal compatibility.

## 6. Chạy lại

Từ repo root, dùng Python có dependencies trong [requirements RF](<../models/random forest/requirements.txt>). Các output dir dưới đây phải **chưa tồn tại**; đổi tên nếu đã chạy:

```bash
python -m unittest discover -s "models/random forest/tests" -p "test_*.py" -v

python "models/random forest/run.py" validate \
  --config "models/random forest/configs/random_forest.json" \
  --data-dir /path/to/raw \
  --output-dir reports/random_forest/my_validation

python "models/random forest/run.py" run \
  --config "models/random forest/configs/random_forest.json" \
  --data-dir /path/to/raw \
  --output-dir reports/random_forest/my_run

python "models/random forest/run.py" predict \
  --config "models/random forest/configs/random_forest.json" \
  --data-dir /path/to/raw \
  --artifact reports/random_forest/my_run/model.joblib \
  --origin-week 2025-12-22 \
  --output-dir reports/random_forest/my_predict
```

Ở máy hiện tại có thể dùng `/home/hexx/DS317_DoAn_KhaiPhaDuLieu/.venv/bin/python` và raw dir `/home/hexx/DS317_DoAn_KhaiPhaDuLieu/data/raw`; code không hardcode hai đường dẫn này.

## 7. Các giả định người dùng/nhóm cần xem và duyệt

- [ ] Gross completed units (không trừ returns), dùng OrderDate và status snapshot, có đúng định nghĩa cần nộp không?
- [ ] Feed completeness: bổ sung bảng coverage có căn cứ hoặc chấp nhận công khai giả định local; không gọi giả định là đã đo DQ.
- [ ] Sau first sale, SKU vẫn nằm trong cohort đến cutoff khi không có lifecycle đáng tin cậy; zero vs unknown và availability/tồn kho là giới hạn.
- [ ] RF cần 8 tuần complete liên tục; short/gappy history dùng MA4, unknown SKU dùng encoder ignore với cờ; có chấp nhận policy này không?
- [ ] Split pilot/3 validation blocks/2 candidates được chấp nhận trước các benchmark tiếp theo; không chỉnh theo điểm test.
- [ ] Điểm validation chưa đạt ngưỡng đề xuất 5%; macro-SKU WAPE chưa hơn MA4. Giữ kết quả âm hay giao thêm tuning/ablation chỉ dựa validation?
- [ ] Dữ liệu demo 2025 là replay, không dự báo live 2026; timezone, status/ingestion/lifecycle chưa có lịch sử point-in-time.

## 8. Checklist xuất bản GitHub và review

- [x] Người dùng đã xem bản bàn giao local và yêu cầu tiếp tục quy trình GitHub ngày 22/11/2026.
- [x] Đã tạo nhánh riêng `feat/random-forest-weekly-pipeline`; base khớp `origin/main` tại `599745e`.
- [x] Đã kiểm tra working tree/remote và phạm vi file; model/raw/per-SKU predictions/artifacts local được ignore, không đưa vào commit.
- [ ] Push nhánh và mở PR vào `main` sau khi xác thực tài khoản GitHub có quyền.
- [ ] Xác nhận username GitHub thư ký rồi tag/request review đúng người.
- [ ] Thư ký kiểm tra PR và duyệt. Không tự merge hoặc vượt qua phê duyệt.

Nhánh/commit local, push, PR, request review và approval là các trạng thái khác nhau. Chỉ báo hoàn thành từng bước khi có kết quả Git/GitHub tương ứng; tài liệu kỹ thuật không thay xác nhận reviewer đã duyệt.
