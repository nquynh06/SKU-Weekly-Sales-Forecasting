# Pipeline Random Forest dự báo lượng bán SKU–tuần

> **Trạng thái:** tài liệu kỹ thuật cho pipeline local của dự án SKU Weekly Sales Forecasting, không phải báo cáo nghiệm thu hay chứng cứ model thắng baseline. Không thay đổi phạm vi bài toán DS317 hoặc tự phê duyệt data contract. Các ngưỡng và giả định được tách rõ ở mục 13; branch/commit/push/PR/tag chỉ thực hiện khi có yêu cầu/phê duyệt riêng của người dùng.

## 1. Bài toán và ranh giới

Cuối tuần \(t\), dự báo tổng `Quantity` hợp lệ của từng SKU trong **đúng tuần kế tiếp** \(t+1\). Đơn vị là **sản phẩm / SKU / tuần**, grain là SKU–tuần, horizon cố định \(H=1\). Đây là lượng bán quan sát được (*observed sales*), không phải giá, doanh thu, sản lượng sản xuất hay nhu cầu không bị giới hạn bởi tồn kho.

Định nghĩa nhãn local là **gross completed units**:

\[
Q_{i,w}=\sum_{r:\,SKU(r)=i,\;OrderDate(r)\in[w,w+7\,ngày),\;r\;hợp\;lệ}Quantity(r).
\]

Tuần bắt đầu vào **thứ Hai**, kết thúc trước thứ Hai tuần sau. `origin_week=t` là thứ Hai bắt đầu tuần vừa chốt; `target_week=t+7 ngày`. Ví dụ origin `2025-12-22` cho dự báo tuần đích `2025-12-29`; origin không phải ngày hệ thống chạy và cũng không có nghĩa dữ liệu mới chỉ đến đầu ngày 22/12. Snapshot nghiệp vụ phải chốt đủ tới hết Chủ nhật `2025-12-28` trước khi dùng origin này.

Dự báo là số thực không âm; **không làm tròn âm thầm** thành đơn vị nguyên. Random Forest regression với squared-error nhắm giá trị trung bình có điều kiện; không gọi output là median, quantile hay xác suất hết hàng.

### 1.1. Có và không có trong lần triển khai này

- Có: quality gate; dựng panel SKU–tuần; lag/rolling/calendar đúng cutoff; global pooled Random Forest; baseline naïve/MA4; validation theo thời gian; test độc lập; saved preprocessing + model; CLI inference; artifacts kiểm toán.
- Không có: tối ưu tồn kho/đặt hàng/giá, return-netting, forecast nhiều bước đệ quy, quantiles, attribution nhân quả, dashboard tương tác, ingest service theo thời gian thực hay giám sát production liên tục.
- Không join giá/khuyến mãi/tồn kho/khách hàng để tránh đưa các trường chưa có chứng cứ *point-in-time availability* vào model. Pipeline RF này không tự thay cam kết benchmark rộng hơn trong đề cương.
- Người phụ trách ngành hàng dùng bảng forecast làm **đầu vào rà soát**; quyết định điều phối/đặt hàng vẫn do người có trách nhiệm và cần thêm stock, hàng về, lead time và policy đã xác minh.

### 1.2. Căn cứ phạm vi

Đã đối chiếu trực tiếp hai nguồn local, chỉ đọc và không sửa. **Các đường dẫn DS317 dưới đây là provenance của máy phát triển; các file nguồn này không được kèm trong repository SKU hoặc có sẵn trên GitHub.** Nội dung cần thiết đã được tóm tắt độc lập trong tài liệu này để người đọc không cần truy cập các đường dẫn local:

1. [Phạm vi đề tài hiện tại](</home/hexx/DS317_DoAn_KhaiPhaDuLieu/DE_TAI_HIEN_TAI.md>): quantity SKU–tuần, H=1, DQ là kiểm soát hỗ trợ, không tự chuyển sang giá/đa nhiệm.
2. [Đặc tả bài toán, nghiệp vụ, gap và mục tiêu](</home/hexx/DS317_DoAn_KhaiPhaDuLieu/outputs/reports/kiem_tra_repo_san_luong_20261115/02_DINH_NGHIA_BAI_TOAN_NGHIEP_VU_GAP_MUC_TIEU.md>): cutoff, zero/missing, cohort tại origin, benchmark cùng keys, saved inference, giới hạn tuyên bố.

Nguồn DS317 là căn cứ phạm vi, không được hiểu là đã phê duyệt mọi cấu hình RF local. Yêu cầu demo ≤180 giây, mục tiêu cải thiện validation 5% và coverage 95% trong đặc tả không phải số đo đã đạt của pipeline này; cần phân biệt yêu cầu giảng viên với ngưỡng nhóm đề xuất và đo thực tế riêng.

## 2. Cấu trúc triển khai và workflow

Thực thi từ thư mục gốc repository SKU Weekly Sales Forecasting; CLI được nêu ở mục 10. Các module Python nằm dưới thư mục [rf_pipeline](<../models/random forest/rf_pipeline/>), gồm `config`, `data`, `features`, `model`, `pipeline`, `cli`; entrypoint là [run.py](<../models/random forest/run.py>). Config local là [random_forest.json](<../models/random forest/configs/random_forest.json>); dependency được pin tại [requirements.txt](<../models/random forest/requirements.txt>).

Phiên bản pipeline hiện tại là `1.0.0`, contract triển khai là **`quantity_rf_v1`**. Đây là contract local của RF, không tự tuyên bố thay thế/phê duyệt toàn bộ data contract quantity cho đề tài. Mô tả dưới đã đối chiếu trực tiếp các module trên, không chỉ dựa vào thiết kế ban đầu.

![Luồng quality gate, train/validation/test và saved-model inference](<figures/random_forest_workflow.png>)

- [Workflow SVG](<figures/random_forest_workflow.svg>): bản vector, phóng lớn được.
- [Nguồn Mermaid](<figures/random_forest_workflow.mmd>): mô tả cùng luồng, sửa văn bản được.
- [Script vẽ SVG/PNG bằng matplotlib](<figures/render_random_forest_workflow.py>): dùng phông DejaVu Sans có sẵn; không cần `dot`, `mmdc`, trình duyệt hoặc tải font.

Luồng tổng quát:

1. Đọc config và snapshot CSV; ghi provenance/hash.
2. Quality gate kiểm khóa, FK, trạng thái, ngày và `Quantity`; lỗi cứng dừng chạy, không trả forecast giả hợp lệ.
3. Dựng panel đến cutoff, phân biệt zero và missing, không dùng ngày bán cuối tương lai.
4. Tạo feature bằng dữ liệu đến origin; không có actual tuần đích trong X.
5. Với `run`: validation rolling → chọn ứng viên → test khóa → refit deploy → lưu artifact và forecast tuần kế tiếp.
6. Với `predict`: nạp saved artifact đáng tin cậy → kiểm contract/cutoff → cùng code tạo feature → RF hoặc fallback → output có trạng thái.
7. Người dùng rà soát kết quả; actual tuần đích chỉ được ghép khi đánh giá sau đó.

## 3. Hợp đồng input: bảng, cột và kiểu dữ liệu

Tất cả input là CSV trong cùng `--data-dir`. Tên bảng/cột **phân biệt đúng tên** như dưới; không tự đổi `ProductID` thành `SKU_ID`. CSV được đọc bằng dtype chuỗi, không coi token NA mặc định là null; các cột bắt buộc được strip khoảng trắng đầu/cuối sau khi loại duplicate toàn dòng. Khóa được giữ là định danh chuỗi để không mất số 0 đầu. Ngày nguồn phải là **ISO date-only `YYYY-MM-DD`**, không nhận timestamp/timezone hoặc chuỗi ngày nhập nhằng; pipeline không tự suy timezone nguồn.

### 3.1. Bảng bắt buộc

| Bảng | Cột bắt buộc | Ngữ nghĩa / yêu cầu |
| --- | --- | --- |
| `SalesOrders.csv` | `SalesOrderID` | Khóa chính đơn hàng, không null/rỗng; một đơn chỉ có một nội dung hợp lệ |
| | `OrderDate` | Ngày giao dịch ISO `YYYY-MM-DD` để gán tuần; không dùng ngày giao hàng/return thay thế |
| | `OrderStatus` | Trạng thái được normalize; chỉ nhận nhóm hoàn tất hoặc hủy được quy định ở mục 4 |
| `SalesOrderLines.csv` | `SalesOrderLineID` | Khóa chính dòng đơn, không null/rỗng |
| | `SalesOrderID` | FK bắt buộc đến đơn hàng; không orphan hoặc join nhiều-nhiều |
| | `LineNumber` | Số hữu hạn, nguyên, **>0**; khóa nghiệp vụ cùng `SalesOrderID`. Hai giá trị như `1` và `1.0` chuẩn hóa cùng số dòng nên không được trùng |
| | `SKU_ID` | FK đến danh mục biến thể; không thay bằng `ProductID` |
| | `Quantity` | Số hữu hạn, nguyên, **>0** cho dòng gross sale hợp lệ |
| `ProductVariants.csv` | `SKU_ID` | Khóa danh mục biến thể, không null/rỗng; dùng xác minh FK, không dùng danh mục toàn kỳ để dựng cohort quá khứ |

Join duy nhất để tạo sales là `SalesOrderLines.SalesOrderID → SalesOrders.SalesOrderID`, có kiểm cardinality. `ProductVariants.SKU_ID` phục vụ xác minh membership/FK. Một SKU có trong danh mục nhưng chưa có giao dịch hợp lệ tại origin **không tự sinh prediction** chỉ vì xuất hiện trong snapshot danh mục cuối kỳ.

### 3.2. Bảng tùy chọn: độ đầy đủ nguồn theo tuần

| Bảng | Cột | Quy tắc |
| --- | --- | --- |
| `WeeklyCoverage.csv` | `week_start` | Ngày thứ Hai của tuần, grain **toàn nguồn theo tuần**, không có SKU |
| | `is_complete` | Nhận `true`/`false` hoặc `1`/`0` sau strip/lower; cờ xác nhận feed tuần đủ/thiếu phải có căn cứ nghiệp vụ, không suy từ số giao dịch dương |

`week_start` phải duy nhất và là ISO thứ Hai; giá trị ngày/cờ không hợp lệ là lỗi cứng. Tuần trong cửa sổ cấu hình nhưng vắng ở bảng coverage được coi là **incomplete**, không được tự mặc định đầy đủ. Bản ghi coverage ngoài cửa sổ không được dùng cho panel hiện tại.

Nếu không có bảng này, pipeline chỉ được phép tiếp tục khi config bật `assume_complete_weeks=true`. [Config local](<../models/random forest/configs/random_forest.json>) **đang bật true**; đây là **giả định local công khai, chưa được nhóm duyệt**, không phải bằng chứng snapshot không thiếu feed. Mặc định bảo thủ của lớp `Config` là false, nhưng CLI đọc file JSON đã nêu. Khi có bảng coverage, bảng đó quyết định và cờ config không biến tuần thiếu thành đủ. Độ đủ toàn nguồn theo tuần không chứng minh độ đủ từng cửa hàng/kênh/SKU; đây là giới hạn của schema hiện tại.

### 3.3. Cột/bảng không sử dụng

Các cột dư có thể được lưu trong raw nhưng không được auto-select làm X. Không có cột tùy chọn nào tự bật feature ngoài contract. Các trường `Price`, `UnitPrice`, `Discount`, `Promotion`, `Inventory`, `CustomerID`, returns, `Products` metadata hoặc nhãn rủi ro không được dùng trong model local này. Không cần điều kiện giá dương để giữ một dòng quantity hợp lệ.

Cấm đưa `y_qty`, actual tuần đích, cột `FutureSales_*`, `cov_fwd`, giá/promo thực hiện tuần sau hoặc thông tin ngày bán cuối toàn kỳ vào X. Chỉ calendar tuần đích có thể biết chắc trước và lịch sử được phép dùng.

## 4. Quy tắc chất lượng dữ liệu và tạo nhãn

### 4.1. Quality gate

| Kiểm tra | Xử lý |
| --- | --- |
| Thiếu bảng/cột bắt buộc; khóa null/rỗng | Dừng, ghi lỗi có ngữ cảnh; không tiếp tục bằng cách tự bịa khóa |
| Dòng trùng hoàn toàn | So **toàn dòng raw trước khi chọn cột/strip**; loại bản lặp đúng nghĩa, đếm trong DQ; không cộng gấp đôi units |
| Cùng PK còn lặp sau full-row dedup; cùng `(SalesOrderID, LineNumber)` sau chuẩn hóa số | Lỗi cứng, không tùy ý giữ first/last. Hai dòng khác cột giá dư cũng không phải exact duplicate, dù model không dùng giá |
| FK đơn hàng hoặc SKU không tồn tại; join nổ cardinality | Lỗi cứng; không drop âm thầm dòng orphan |
| `Quantity` không phải số hữu hạn/nguyên hoặc ≤0 | Lỗi cứng theo gross-units contract; không thay bằng 0 và không cắt quantity lớn hợp lệ theo outlier heuristic |
| `LineNumber` không phải số hữu hạn/nguyên hoặc ≤0 | Lỗi cứng trước kiểm business key |
| Trạng thái ngoài vocabulary, kể cả trống | Lỗi cứng; không tự coi trạng thái chưa biết là completed |
| `OrderDate` không parse được ISO date-only | **Lỗi cứng**, không loại âm thầm |
| Trạng thái hủy với ngày trong cửa sổ | Loại khỏi gross completed units; đếm `excluded_cancelled_in_window_lines` |
| Ngày parse được nhưng ngoài `[data_start_week, data_end_week + 7 ngày)` | Loại khỏi nhãn, đếm lines/orders ngoài cửa sổ; không đổi thành sales bằng 0 |

Vocabulary sau strip/lower: `complete` / `completed` được map thành `completed`; `canceled` / `cancelled` được map thành `cancelled`. Không dùng snapshot cuối kỳ để khẳng định trạng thái đó đã biết tại mọi origin quá khứ. Nếu nguồn thực có trạng thái khác (pending/processing/returned...), phải chốt mapping mới trước khi chạy; không lách rule bằng sửa raw không trace.

Các kiểm tra cứng chạy trên **toàn bảng được cung cấp trước khi lọc ngày/hủy**. Vì vậy Quantity lỗi ở dòng cancelled hoặc ngoài cửa sổ vẫn làm batch thất bại; không dựa vào exclusion để bỏ qua lỗi raw. Sau lọc nếu không còn completed sale nào trong cửa sổ, pipeline cũng từ chối. Khi `predict` nhận raw chứa giao dịch sau origin, chúng bị loại khỏi panel theo cửa sổ, nhưng vẫn chịu các raw checks nói trên.

### 4.2. Panel tuần và cohort theo origin

- Một tuần là `[thứ Hai 00:00, thứ Hai kế tiếp 00:00)`. `data_start_week`/`data_end_week` là **week_start**, không phải ngày đầu/cuối giao dịch tùy ý.
- SKU bắt đầu được theo dõi từ tuần có **first valid completed sale**; tại origin chỉ dùng SKU có first sale ≤ origin.
- Panel kéo dài đến tuần as-of/cutoff, kể cả SKU đã ngừng có giao dịch. **Không cắt tại last sale toàn kỳ**, không dùng số tuần bán toàn kỳ để chọn SKU quá khứ.
- Tuần trước first sale không được pad zero thành lịch sử bán hàng của SKU.
- Sau first sale, nếu nguồn tuần đầy đủ, không có dòng bán nghĩa là `0`. Nếu tuần nguồn thiếu, quantity đánh giá là missing (`NaN`) **kể cả có một phần giao dịch dương**; phần dương không chứng minh đã nạp đủ.
- Không backward-fill và không forward-fill quantity; không biến unknown thành 0 để tăng số hàng train.
- Actual missing không tham gia tính sai số; phải báo coverage/mẫu không chấm điểm, không chỉ bỏ dòng khó khỏi báo cáo.

### 4.3. Đối soát và lineage

DQ report hiện ghi số dòng raw/dedup, key conflicts, dòng join và `joined_input_units`, dòng/units loại ngoài cửa sổ (`excluded_outside_window_lines`, `excluded_outside_window_units`), dòng/units hủy trong cửa sổ (`excluded_cancelled_in_window_lines`, `excluded_cancelled_in_window_units`), `valid_sales_lines`, `gross_completed_units`, số SKU/panel, số tuần complete/incomplete, `missing_panel_labels`, `complete_panel_units` và `observed_units_in_incomplete_weeks`. Không có item-level quarantine CSV; cần thêm nếu quy trình nghiệp vụ đòi truy nguyên từng dòng. Kiểm join bảo toàn row count và units trước lọc; aggregation tuần bảo toàn tổng gross completed units. Chênh giữa gross units và complete-panel units do tuần incomplete phải được đọc cùng báo cáo, không coi là thất thoát join.

Giữ một quy tắc gross quantity nhất quán; không cộng rồi trừ returns không có event-time riêng. Coverage source/warnings hiện rõ trong DQ; manifest lưu config, raw hashes và assumptions chung.

Không gọi event-time (`OrderDate`) là ingest-time. Khi thiếu timestamp nạp/trạng thái lịch sử, backtest chỉ là hồi cứu trên snapshot theo giả định availability, không phải chứng cứ mô phỏng đầy đủ batch vận hành thật.

## 5. Feature availability và chống leakage

Hàng feature được đánh chỉ số bằng `target_week=w`; `origin_week=w−7 ngày`. Vì dự báo **cuối tuần origin**, dữ liệu tuần origin đã được biết. Không dịch lag thừa một tuần.

Allowlist X là **`SKU_ID` + đúng 13 numeric features** dưới đây; các cột diagnostics/actual không lọt vào X nhờ `FEATURE_COLUMNS` và `ColumnTransformer(remainder="drop")`.

| Tên feature thực tế | Định nghĩa tại hàng target `w` | Tính sẵn có |
| --- | --- | --- |
| `lag_1`, `lag_2`, `lag_4`, `lag_8` | `Q[w−1]`, `Q[w−2]`, `Q[w−4]`, `Q[w−8]` | `lag_1=Q_origin`; chỉ quantity quá khứ đã chốt |
| `rolling_mean_4`, `rolling_std_4` | Mean và std **ddof=0** của `Q[w−4:w−1]` | Đủ 4 tuần known mới có giá trị; không bỏ gap cho RF |
| `rolling_mean_8`, `rolling_std_8` | Mean và std **ddof=0** của `Q[w−8:w−1]` | Đủ 8 tuần known mới có giá trị |
| `zero_fraction_8` | Số tuần quantity =0 / 8 trong `Q[w−8:w−1]` | Missing giữ unknown, không tính thành zero |
| `age_weeks` | `(origin_week − first_seen_week)/7 ngày + 1` | First sale đã thấy tại origin; tuần đầu có tuổi 1, không dùng last sale |
| `target_week_sin`, `target_week_cos` | `sin/cos(2π × ISO_week(target_week) / 52.1775)` | Calendar tuần đích đã biết trước; chu kỳ là công thức xấp xỉ được khóa trong code |
| `target_month` | Tháng 1–12 của `target_week`, numeric | Biết trước, không phụ thuộc realized sales/price/promo |
| `SKU_ID` qua one-hot | Category của SKU trong lần fit hiện tại | Encoder fit train-only, `handle_unknown="ignore"`; SKU lạ có block one-hot toàn 0 |

Các cột `history_known_last_4`, `history_known_window`, `rf_eligible`, `baseline_naive`, `baseline_ma4`, `actual_quantity`, keys và trạng thái là diagnostics/routing/evaluation, **không phải feature RF**. `actual_quantity` ở hàng target bằng quantity tuần kế tiếp và chỉ dùng fit/score.

RF cần **`min_history_weeks` tuần liên tục với quantity biết được** trước target; config local đặt **8**, validator chỉ cho integer ≥8. Với `min_history_weeks>8`, eligibility kiểm toàn cửa sổ dài hơn dù numeric window lớn nhất vẫn 8. Một tuần zero có nguồn đủ là known; một tuần thiếu feed là unknown. Chỉ “đã tồn tại 8 tuần” chưa đủ nếu có gap. Training còn yêu cầu actual target known; inference không cần actual target.

Preprocessing ở trong saved artifact; không fit one-hot bằng toàn panel trước split. Naïve, MA4 và RF/fallback có cùng hàng `(SKU_ID, origin_week, target_week)`; score so sánh công bằng nằm trong `common_keys`, còn `all_rows` cho biết coverage/subset riêng (mục 8). Không so WAPE subset riêng rồi ngụ ý cùng cohort chấm điểm.

## 6. Chia train / validation / test theo thời gian

Cấu hình local đang lưu trong [random_forest.json](<../models/random forest/configs/random_forest.json>):

| Phân đoạn | Tuần đích / cửa sổ |
| --- | --- |
| Dữ liệu đầu vào | Từ `2023-07-03` đến hết tuần bắt đầu `2025-12-22` |
| Train ban đầu | Target tuần ≤ `2025-03-24`, sau khi lọc eligibility |
| Validation | Target từ `2025-03-31` đến `2025-08-04`, chia **3 block rolling** |
| Test khóa | Target từ `2025-08-11` đến `2025-12-22` |
| Forecast sau run | Origin `2025-12-22` → target `2025-12-29` |

Toàn SKU dùng **cùng mốc target_week**, không random split theo hàng. Giai đoạn đầu panel phục vụ lag nên không phải mọi tuần trong cửa sổ raw đều sinh được mẫu train.

### 6.1. Rolling validation

1. Chia 19 tuần validation bằng `numpy.array_split` thành 3 block tăng dần, không xáo trộn: **31/03–12/05/2025 (7 tuần)**, **19/05–23/06/2025 (6 tuần)**, **30/06–04/08/2025 (6 tuần)**. Với config khác, đọc ranh giới thực từ `validation_blocks` trong manifest.
2. Trong từng block, fit forest bằng các target trước block; **giữ cố định forest trong block**. Cutoff fit của 3 block mặc định lần lượt là `2025-03-24`, `2025-05-12`, `2025-06-23`.
3. Feature của mỗi origin trong block vẫn dùng actual lịch sử đã quan sát tới origin đó; không forecast recursive toàn block bằng chính qhat cũ.
4. Sang block tiếp theo, mở rộng train bằng target đã có trước block mới rồi fit lại.
5. Thử đúng 2 ứng viên: candidate 0 có `n_estimators=80`, `max_depth=10`; candidate 1 có `n_estimators=80`, `max_depth=14`. Cả hai dùng `min_samples_leaf=5`, `max_features=0.8`, `criterion="squared_error"`, `random_state=317`, `n_jobs=-1`. Seed cố định giúp tái lập ở cùng môi trường; không bảo đảm bitwise identical giữa mọi version/CPU.
6. `max_features` trong config luôn có nghĩa là **tỷ lệ `(0,1]`**; code ép sang float trước truyền sklearn, nên JSON `1` và `1.0` đều là 100% feature, không phải một cột.
7. Chọn theo **WAPE tổng hợp mọi hàng validation có actual và qhat**, không phải trung bình đơn giản WAPE từng block/tuần. Score là hệ **RF có MA4 fallback**, không chỉ các hàng RF. Nếu tổng actual được score bằng 0 dùng MAE; nếu không có hàng nào score thì dừng. Hòa score chọn candidate id nhỏ hơn. Không dùng epsilon và không nhìn test để đổi depth, cohort, split hoặc zero policy.

MA4/naïve không phải ứng viên forest được tuning. Validation có thể cho thấy RF kém baseline; phải giữ kết quả âm, không mặc định tuyên bố RF có ích hơn.

### 6.2. Test và model deploy là hai đối tượng khác nhau

- Sau khi chọn ứng viên, fit một model đánh giá bằng các target đến `validation_end_week`; dự đoán test theo từng origin với forest cố định. Test không được dùng tuning. Nếu không có ít nhất một hàng có **cả actual đáng tin cậy lẫn qhat available**, run dừng thay vì báo evaluation thành công với mọi metric null. Vì dataset legacy có thể đã từng được khám phá ở nghiên cứu trước, “test khóa” ở đây nghĩa là **khóa trong lần triển khai này**, không bảo đảm là blind/unseen theo toàn bộ lịch sử dự án; điều này cũng được ghi trong `metrics.json`.
- Sau đánh giá, **refit deploy** cấu hình đã chọn bằng target hợp lệ đến hết `data_end_week`. Artifact deploy này có thể dùng dữ liệu từng là test để dự báo tương lai; vì vậy không lấy chính artifact refit để chấm lại test rồi báo là kết quả holdout.
- Feature importance của model deploy không phải giải thích nhân quả, không phải bằng chứng chọn feature không leakage; one-hot có thể gây thiên lệch importance và SKU mới cần kiểm tra riêng.

## 7. Baseline, fallback và trạng thái dự báo

- **Naïve:** `baseline_naive=Q_origin` đúng tuần vừa chốt; nếu origin missing thì naïve missing. **Không tìm tuần known cũ hơn** thay cho origin.
- **MA4:** `baseline_ma4` là trung bình các tuần **biết được** trong 4 tuần ngay trước target; có ≥1 tuần known thì có thể xuất. Mẫu số là số tuần known, không cố định 4 khi có gap.
- **RF path:** đủ `min_history_weeks` liên tục known → dự báo từ saved preprocessing + forest, ràng buộc `max(0, prediction)`. `prediction_status=random_forest`, `model_id=random_forest`, `fallback_reason` rỗng.
- **Fallback:** history ngắn/gappy → MA4 nếu có ≥1 tuần known trong 4 tuần cuối. Khi đó `prediction_status=fallback_ma4`, `model_id=ma4_fallback`, `fallback_reason=insufficient_contiguous_complete_history`.
- **Unavailable:** không có tuần known nào trong 4 tuần cuối → `prediction_status=unavailable`, `model_id=none`, `fallback_reason=no_known_quantity_last_4_weeks`, `qhat` trống.
- Không tự tạo zero cho dòng unavailable. `0.0` là dự báo số không; ô trống là không có dự báo, hai trạng thái khác nhau.
- SKU chưa xuất hiện trước origin không vào cohort. SKU **chưa nằm trong vocabulary artifact nhưng đã có lịch sử đủ tại inference vẫn đi RF**: encoder `handle_unknown="ignore"` tạo one-hot toàn 0; `sku_seen_in_training=false` cảnh báo model chưa học identity này. Cờ này không tự buộc fallback, cũng không chứng minh model generalize tốt.
- `prediction_functional=mean` cho output hệ RF/MA4; không phải dự báo phân vị. Baseline naïve được giữ trong cột riêng, không có một hàng status riêng cho naïve.

Kết quả RF có fallback phải báo cả tỷ lệ đi đường RF/fallback/unavailable. Score của hệ RF+fallback không được viết thành score thuần RF nếu có dòng MA4 thực hiện dự báo.

## 8. Metrics và diễn giải

`metrics.json` có `selection`, `validation_selected`, `test`, `metric_units` và `test_protocol`. Tên phương pháp là `random_forest_with_fallback` (cột `qhat`), `moving_average_4` (`baseline_ma4`), `naive` (`baseline_naive`). Validation/test dùng chung hàng cohort, nhưng prediction availability có thể khác; schema tách rõ:

- **`all_rows`:** mỗi phương pháp score trên hàng có actual và prediction của chính nó; kèm coverage riêng. Không coi các score này luôn cùng subset.
- **`common_keys`:** chỉ hàng actual known **và cả ba prediction đều available**; đây là phần dùng so sánh trực tiếp cùng keys. `common_key_rows` là số hàng của giao này.
- **`by_week`:** metrics mỗi phương pháp theo `target_week` trên subset available của chính phương pháp trong tuần đó, **không có common intersection riêng theo tuần** trong implementation hiện tại.
- **`prediction_status_counts`:** số hàng RF/fallback/unavailable của hệ RF. Score chọn candidate lưu riêng ở `selection.candidates`, tổng hợp validation RF+fallback như mục 6.1.

Với \(N\) là số hàng thực sự được score cho phương pháp/subset:

\[
MAE=\frac1N\sum|\widehat Q-Q|,\quad
RMSE=\sqrt{\frac1N\sum(\widehat Q-Q)^2},\quad
WAPE=\frac{\sum|\widehat Q-Q|}{\sum Q},\quad
bias=\frac{\sum(\widehat Q-Q)}{\sum Q}.
\]

- MAE/RMSE có đơn vị sản phẩm. **WAPE và bias là fraction** (nhân 100 nếu trình bày %); tổng actual bằng 0 → **null / N/A**, không chia epsilon hoặc xuất 0% giả. Bias dương nghĩa dự báo cao, âm nghĩa thấp; không phải mean signed error theo đơn vị.
- `macro_sku_mae`: tính MAE từng SKU rồi trung bình đều giữa SKU có mẫu score. `macro_sku_wape`: trung bình WAPE từng SKU chỉ khi tổng actual của SKU >0; số SKU bị loại vì mẫu số 0 nằm ở `macro_sku_wape_excluded_zero_denominator`.
- Coverage gồm `total_rows`, `labeled_rows`, `scored_rows`, `prediction_coverage` (=prediction available / tổng hàng), `label_coverage` (=actual known / tổng hàng), `scored_coverage_of_labels` (=hàng score / actual known). Rỗng mẫu thì tỷ lệ/metrics tương ứng là null; điểm tốt trên subset nhỏ không chứng minh hệ thống tốt.
- Theo tuần giúp thấy dispersion và tuần bất thường. Không gọi point estimate là confidence interval hay bằng chứng ý nghĩa thống kê nếu chưa có phương pháp uncertainty thích hợp.
- Không báo tỷ lệ tăng lợi nhuận, giảm tồn kho hoặc đáp ứng service-level khi không đo policy/cost/lead time và outcome thực tế.

## 9. Output và schema

Mỗi `--output-dir` phải là thư mục **mới**, không overwrite artifacts run cũ. CSV dùng ngày `YYYY-MM-DD`, boolean theo serializer pandas, missing numeric là ô trống. Các tên dưới là schema triển khai hiện tại, không alias `SKU_ID` thành `sku_id`.

| Artifact | Lệnh tạo | Grain / mục đích |
| --- | --- | --- |
| `dq_report.json` | `validate`, `run`, `predict` | Rule DQ, dòng/units, exclusions, completeness và assumptions |
| `weekly_sales.csv` | Cả ba | `(SKU_ID, week_start)`; schema ở mục 9.1 |
| `features.csv` | Cả ba | `(SKU_ID, origin_week, target_week)`; actual/eligibility/diagnostics không phải X |
| `validation_predictions.csv` | `run` | Một hàng SKU–origin–candidate; `qhat` và hai baseline **wide columns**, không ba hàng model riêng |
| `test_predictions.csv` | `run` | Một hàng SKU–origin của cấu hình đã chọn; `qhat` và baseline cùng keys |
| `forecast.csv` | `run`, `predict` | Một hàng SKU tại origin yêu cầu, không xuất actual target |
| `metrics.json` | `run` | Selection, validation_selected và test metrics như mục 8 |
| `feature_importance.csv` | `run` | Đúng hai cột `feature`, `importance`, sắp giảm dần; tên sau preprocessing |
| `model.joblib` | `run` | Artifact deploy refit; không lưu lại test model riêng |
| `run_manifest.json` | Cả ba | Run/config/input hashes/version/provenance; trường thêm tùy mode |
| `failure.json` | Lệnh lỗi sau khi tạo output dir | `status`, `error_type`, `error`, và `dq_report` nếu là lỗi data contract; artifacts đã viết trước lỗi vẫn có thể còn, không được coi là run hoàn tất |

CLI trả exit code **0** khi thành công, **2** khi bắt lỗi contract/config/path/model thông thường. Output directory đã tồn tại thì trả 2 mà không ghi đè diagnostics. Config lỗi trước khi tạo folder có thể chỉ có stderr, không nhất thiết có `failure.json`. Với DataContractError, DQ report lỗi được ghi nếu output folder đã được tạo; manifest đã tồn tại sẽ được cập nhật `status=failed`. Consumer phải kiểm `status=completed`, không chỉ kiểm có CSV/model.

### 9.1. Schema weekly panel và feature table

`weekly_sales.csv` có đúng các cột sau:

| Cột | Kiểu / ý nghĩa |
| --- | --- |
| `SKU_ID` | Chuỗi SKU, giữ số 0 đầu |
| `week_start` | ISO Monday date; key cùng SKU |
| `first_seen_week` | Tuần first valid completed sale trong cửa sổ input; không phải ngày listing được xác minh |
| `observed_quantity` | Tổng units quan sát từ dòng hợp lệ; trống nếu tuần không có giao dịch |
| `is_complete` | Boolean độ đủ nguồn toàn tuần |
| `quantity` | Nhãn: 0 nếu không có giao dịch và nguồn complete; tổng observed nếu complete; trống nếu incomplete |

`features.csv` gồm keys `SKU_ID`, `origin_week`, `target_week`; **13 cột numeric** tên chính xác ở mục 5; và `history_known_last_4` (số known trong last4), `history_known_window` (số known trong cửa sổ min_history), `rf_eligible` (boolean), `baseline_naive`, `baseline_ma4`, `actual_quantity`. Hàng có thể không đủ điều kiện RF nhưng vẫn có để báo cohort/fallback. Feature table chứa cả hàng target sau data_end cho inference, actual trống ở hàng đó; mọi fit/split phải lọc target và label như code, không fit toàn bảng CSV tùy ý.

### 9.2. Exact prediction / forecast schema

Các cột cơ sở do `predict_rows` tạo:

| Trường | Kiểu / ý nghĩa |
| --- | --- |
| `SKU_ID` | Chuỗi SKU; **chữ hoa đúng tên**, không phải ProductID |
| `origin_week` | Ngày thứ Hai bắt đầu tuần đã chốt |
| `target_week` | `origin_week + 7 ngày` |
| `actual_quantity` | Nhãn để đánh giá, trống nếu unknown; bị **loại khỏi `forecast.csv`** |
| `baseline_naive` | Quantity origin hoặc trống nếu origin unknown |
| `baseline_ma4` | Mean known trong last4 hoặc trống nếu không có known |
| `history_known_last_4` | Integer 0–4, số known last4 |
| `history_known_window` | Integer 0–min_history, số known cửa sổ eligibility |
| `rf_eligible` | Boolean đủ min_history tuần known liên tục |
| `qhat` | Float ≥0 hoặc trống khi unavailable; RF hoặc MA4 tùy status |
| `prediction_status` | Đúng một trong `random_forest`, `fallback_ma4`, `unavailable` |
| `fallback_reason` | Rỗng; `insufficient_contiguous_complete_history`; hoặc `no_known_quantity_last_4_weeks` |
| `model_id` | `random_forest`, `ma4_fallback` hoặc `none`; đây là route thực sự dự báo, không phải một id duy nhất của saved artifact |
| `sku_seen_in_training` | Boolean có SKU trong vocabulary encoder của model đang dùng |
| `prediction_functional` | Chuỗi `mean` |

Các phần mở rộng theo output:

- **Validation:** thêm `candidate_id` (0-based), `validation_block` (0-based), `fit_targets_through`. Khóa duy nhất `(SKU_ID, origin_week, candidate_id)`; target suy ra từ origin. Có tất cả candidate; không thêm `run_id`/`as_of` trực tiếp, nối provenance qua manifest.
- **Test:** thêm `fit_targets_through=validation_end_week`; khóa `(SKU_ID, origin_week)`. Không có `run_id`/`as_of` trực tiếp; manifest ghi cutoff test/deploy.
- **Forecast:** bỏ `actual_quantity`, thêm `run_id`, `as_of`, `contract_version=quantity_rf_v1`; khóa `(SKU_ID, origin_week)`. `as_of=origin_week+7 ngày` là **exclusive event-time cutoff**: ví dụ `2025-12-29` bao trọn tuần origin 22/12 đến hết 28/12. Không phải ingest timestamp hay thời gian tạo run. `created_at_utc` nằm trong manifest.

CSV rỗng/NaN cần được consumer BI đọc là missing, không tự `fillna(0)` ở màn hình. Baseline nằm cùng hàng output, không có một hàng riêng mang `model_id=naive`/`moving_average_4`.

### 9.3. Saved artifact và manifest

`model.joblib` là dictionary gồm `contract_version`, `pipeline_version`, `model` (sklearn Pipeline preprocessing+forest), `params`, `config`, `feature_columns`, `trained_through_target_week`, `training_run_id`, `versions`. `trained_through_target_week` là cutoff fit target theo config; nếu một số target thiếu nhãn, không có nghĩa model được học các nhãn missing đó. Không sửa dictionary để bỏ kiểm version/cutoff.

Manifest chung có `run_id`, `mode`, `status`, `created_at_utc`, `pipeline_version`, `contract_version`, `config`, `config_sha256`, `inputs` (path/SHA256/bytes), `versions`, `feature_columns`, `assumptions`.

- Mode **run** thêm runtime, hash các module Python, model hash, split-keys hash, validation block boundaries, candidate được chọn, cutoff test/deploy, forecast origin/rows/status counts và danh sách artifacts.
- Mode **predict** thêm artifact path/hash, training run/cutoff, forecast origin/rows/status counts; không train lại và không tạo metrics/test/importance/model mới.
- Mode **validate** chỉ có manifest chung cùng DQ/panel/features; chưa chứng minh có đủ labeled RF train hoặc model đạt score. Không có source/model hashes đầy đủ như mode run.

### 9.4. Ranh giới nghiệp vụ của output

Forecast hỗ trợ danh sách SKU cần rà soát, có thể đối chiếu delta so MA4 ở lớp BI sau. Pipeline không tự phát hành lệnh mua và không cung cấp quantity tối ưu nhập. DQ/fallback là tín hiệu cần rà nguồn, không là nhãn ground truth “sản phẩm rủi ro” hay classifier thứ hai.

## 10. Cách chạy tại local

Dùng Python environment riêng; các version đã kiểm local nằm tại [requirements.txt](<../models/random forest/requirements.txt>): `pandas==3.1.1`, `numpy==2.5.4`, `scikit-learn==1.9.1`, `joblib==1.6.0`; `matplotlib==3.11.2` chỉ cần khi vẽ lại hình. Môi trường tham chiếu là Python `3.12.14`; exact version thực tế của mỗi run nằm trong manifest. Saved inference yêu cầu **đúng version scikit-learn đã tạo artifact**, không chỉ cùng major version. Không ghi vào DS317 hoặc cài gói vào môi trường của repo khác.

Ví dụ tạo môi trường riêng trong repo SKU (chỉ thực hiện khi cần, không phải bước tự động của pipeline):

```bash
python3.12 -m venv .venv-rf
.venv-rf/bin/python -m pip install -r 'models/random forest/requirements.txt'
```

Các lệnh dưới chạy từ repository root sau khi kích hoạt environment hoặc thay `python` bằng `.venv-rf/bin/python`; thay input bằng snapshot thực của người dùng.

### 10.1. Validate input

```bash
python 'models/random forest/run.py' validate \
  --config 'models/random forest/configs/random_forest.json' \
  --data-dir '/duong-dan/snapshot/data/raw' \
  --output-dir 'reports/random_forest/validate_001'
```

`validate` kiểm data contract/quality và dựng panel/features, xuất DQ + panel + features + manifest; **không fit model**, không bảo đảm đã có đủ labeled RF rows cho mọi validation block, không phải chứng nhận feed thực sự đầy đủ ngoài dữ liệu được cung cấp. Đọc DQ report và assumptions, không chỉ nhìn exit code. Có thể xem cú pháp bằng `python 'models/random forest/run.py' --help` hoặc thêm `run --help`.

### 10.2. Train + validation + test + lưu deploy artifact

```bash
python 'models/random forest/run.py' run \
  --config 'models/random forest/configs/random_forest.json' \
  --data-dir '/duong-dan/snapshot/data/raw' \
  --output-dir 'reports/random_forest/local_run_001'
```

Đường dẫn output phải chưa tồn tại. Không chạy lại cùng folder để “cập nhật” model. Dùng tên run mới và lưu config/raw snapshot tương ứng để so sánh.

### 10.3. Predict bằng saved model, không train lại

```bash
python 'models/random forest/run.py' predict \
  --config 'models/random forest/configs/random_forest.json' \
  --data-dir '/duong-dan/snapshot/data/raw' \
  --artifact 'reports/random_forest/local_run_001/model.joblib' \
  --origin-week '2025-12-22' \
  --output-dir 'reports/random_forest/predict_001'
```

Với artifact deploy fit đến `2025-12-22`, origin này tạo forecast tuần `2025-12-29`. Lệnh `predict` **từ chối** origin trước `trained_through_target_week`; dùng model fit sau origin là leakage dù feature không chứa actual tương lai. Cũng từ chối contract version/feature allowlist khác, version scikit-learn khác, hoặc thay các config fields `data_start_week`, `min_history_weeks`, `assume_complete_weeks` so với artifact. Các split/candidate fields khác không làm lệnh predict train lại.

Có thể dự báo origin mới hơn cutoff fit: lệnh đọc panel đến `--origin-week` thay cho config `data_end_week`, nhưng vẫn giữ model cố định. **Origin vượt `data_end_week` của artifact bắt buộc có `WeeklyCoverage.csv`**, không cho kéo dài giả định complete cũ sang các tuần chưa nhìn thấy. Tuần mới thiếu entry coverage là unknown/incomplete; chỉ đánh complete khi có căn cứ snapshot đã chốt. Nguồn thiếu giữ missing/fallback/unavailable, không forward-fill tuần chưa chốt để xin forecast nhiều bước. Cohort có thể thay đổi theo first sale đã thấy trong snapshot mới; xem cờ `sku_seen_in_training`.

`joblib`/pickle có thể thực thi mã khi load: **chỉ nạp artifact từ nguồn đáng tin cậy do pipeline/nhóm tạo và kiểm soát**, không nhận model upload tùy ý. Saved preprocessing phải được dùng lại; không tái fit encoder trên input inference.

### 10.4. Vẽ lại workflow

```bash
MPLCONFIGDIR='docs/figures/.matplotlib' \
  python 'docs/figures/render_random_forest_workflow.py'
```

Dùng environment có matplotlib và phông DejaVu Sans sẵn có; không cần `dot`/`mmdc` hoặc cài font. Script chỉ ghi SVG/PNG và font cache cạnh script. Thư mục `.matplotlib` là **cache local, không phải deliverable, không chép/đưa vào Git**. Ảnh PNG bàn giao đã được mở bằng `read_image` để kiểm tiếng Việt và layout; SVG giữ text vector.

### 10.5. Chạy bộ kiểm thử

Dùng cùng environment RF, từ repository root:

```bash
python -m unittest discover \
  -s 'models/random forest/tests' \
  -p 'test_*.py' -v
```

[Bộ kiểm thử](<../models/random forest/tests/test_random_forest.py>) dùng fixtures synthetic và standard-library `unittest`; không cần pytest. Các kiểm tra gồm data contract/key/FK/quantity/status/ngày, zero vs missing, cutoff/lag/target, perturb future raw không đổi past features, temporal split, artifact reload, fallback/unavailable không fake zero, encoder SKU lạ, mẫu số metric bằng 0 và CLI failure/không overwrite. Fixture chỉ kiểm correctness, **không phải benchmark accuracy trên dataset thực**. Không hardcode số test hoặc tuyên bố pass trong hướng dẫn này; đọc kết quả thực của lần chạy do Lead thực hiện. Full training/evaluation, saved inference và startup/demo là những phép đo runtime khác nhau, phải ghi riêng.

## 11. Saved-model inference: checklist kỹ thuật

1. Xác minh artifact đáng tin cậy và metadata/version, run id, data cutoff.
2. Snapshot inference phải có các bảng/cột bắt buộc và nguồn đến origin; không “mượn” actual tuần đích từ raw để tạo feature.
3. Chạy lại quality gate như training; lỗi cứng từ chối cả run thay vì trả số dự báo không trace.
4. Dựng panel chỉ đến origin; cohort từ first sale đã thấy, không từ last sale hoặc thống kê toàn kỳ.
5. Tạo feature qua cùng code và cùng naming/order; transform bằng preprocessor đã lưu.
6. Kiểm eligibility, chạy RF/fallback/unavailable theo rule; không fit forest/encoder trong lệnh predict.
7. Xuất forecast có keys/status/fallback_reason, DQ và manifest; kiểm khóa duy nhất, target = origin + 7, qhat không âm hoặc missing có lý do.
8. Đọc baseline/coverage cùng forecast; chỉ join actual sau khi target đóng tuần để monitor, không sửa dự báo lịch sử.

Một smoke test reload cần so dự báo trước/sau load trên cùng snapshot và kiểm trace/schema. Việc CLI chạy được không thay kiểm thử temporal leakage, zero/missing và reconcile quantity.

## 12. Kiểm tra cần đọc trước khi sử dụng kết quả

- [ ] Đầu vào không phải fixture/synthetic bị gọi nhầm là dữ liệu thực; có dataset card và quyền sử dụng.
- [ ] Rule DQ pass; lỗi/exclusions/units đã được rà soát, không che dòng lỗi bằng lọc không trace.
- [ ] Căn cứ feed completeness được cung cấp; nếu chưa có, người đọc thấy rõ assumption local.
- [ ] Cohort theo origin, không last-sale filter; SKU ngừng bán vẫn được giữ đến cutoff.
- [ ] Missing/zero được phân biệt; positive partial week vẫn missing nếu feed thiếu.
- [ ] Split theo target_week; 3 validation blocks; one-hot fit train-only; không test tuning.
- [ ] RF/MA4/naïve cùng keys; báo coverage và fallback chứ không chỉ score subset.
- [ ] Model deploy refit được tách khỏi model đánh giá test; saved inference không fit sau origin.
- [ ] Forecast schema, qhat missing/status và model/run/as-of đủ cho BI consume an toàn.
- [ ] Runtime demo được đo trên máy đích; chưa đo thì không khẳng định đạt ≤180 giây.

## 13. Giả định DQ, giới hạn và hạng mục cần duyệt

### 13.1. Giả định local chưa được duyệt thay nhóm

1. `assume_complete_weeks=true` khi thiếu coverage là giả định pilot, **không chứng thực độ đủ nguồn**. Cần feed ledger hoặc đối soát nguồn chính thức trước dùng vận hành.
2. Gross completed units, status vocabulary và Quantity nguyên dương phải được xác nhận; nếu chọn net returns cần contract khác và tính lại nhãn, không trộn chung.
3. First valid sale là proxy bắt đầu theo dõi; chưa có lịch vòng đời/listing theo thời điểm. SKU mới chưa có bán không được forecast dù có trong danh mục cuối kỳ.
4. Ngưỡng RF 8 tuần liên tục, fallback MA4, 2 ứng viên RF, mốc train/validation/test và cách chọn metric là cấu hình local; cần khóa sau pilot train/validation, trước final test.
5. Lịch tuần theo timezone/ngày nguồn; nếu raw có nhiều timezone hoặc ngày giao dịch không rõ phải chuẩn hóa nghiệp vụ trước. Event time không chứng minh ingest completeness.
6. Snapshot `OrderStatus`/danh mục cuối kỳ có thể không phản ánh trạng thái đã biết tại origin quá khứ; phải công bố hạn chế point-in-time, không khẳng định đã khử hoàn toàn leakage từ raw snapshot.

### 13.2. Giới hạn mô hình và triển khai

- Chưa có availability/lost-sales đáng tin cậy: không suy lượng bán thấp là nhu cầu thấp, không khôi phục demand censored bởi stockout.
- Không có price/promo/inventory/metadata point-in-time: giới hạn sức dự báo, nhưng tốt hơn đưa future covariates chưa chứng minh vào X.
- Pooled RF + SKU one-hot có thể khó generalize SKU mới, panel rất rộng có thể tốn RAM; cần kiểm vocabulary, memory và runtime khi scale.
- Window 8 tuần/fallback MA4 có thể bỏ lỡ seasonality dài, launch/end-of-life, promo shock và regime change; không cam kết extrapolation ngoài miền lịch sử.
- Coverage toàn nguồn theo tuần không đủ chỉ ra missing từng SKU/kênh; quality gate kiểm được contract, không xác nhận toàn bộ tính đúng nghiệp vụ.
- WAPE panel có volume weighting; macro/profile và coverage phải được xem cùng. Không có gain validation thì không ép kết luận “RF tốt hơn”.
- Local files/manifest không phải production registry, access control, data retention, CI/CD hay monitoring service. Dashboard/worklist và đánh giá usability là công việc tiếp theo, chưa được chứng minh qua artifacts RF.

### 13.3. Checklist review kỹ thuật và xuất bản

- [ ] Duyệt quantity contract, status mapping, policy ngày ngoài cửa sổ, zero/missing và completeness evidence.
- [ ] Duyệt cohort/first-sale proxy, ngưỡng 8 tuần và fallback/coverage.
- [ ] Duyệt split, RF candidate grid, metric selection và protocol không chạm test.
- [ ] Duyệt input/output schema, saved artifact safety, assumptions và giới hạn tuyên bố.
- [ ] Đối chiếu smoke tests và runtime trên dữ liệu/máy đích; chốt kết quả nào là synthetic, kết quả nào là real.
- [ ] Rà soát tài liệu/workflow và diff cuối trong repository SKU; không đưa raw data, saved model không chủ định hoặc `.matplotlib` cache vào thay đổi xuất bản.
- [x] Người dùng đã yêu cầu quy trình xuất bản ngày 22/11/2026; mã được chuẩn bị trên nhánh `feat/random-forest-weekly-pipeline` cho PR vào `main`.
- [ ] Hoàn tất push/PR và tag đúng thư ký khi có xác thực/tài khoản reviewer; không tự merge. Không có thao tác Git nào là một phần tự động của tài liệu hoặc CLI này.

## 14. Trạng thái bàn giao

Tài liệu và sơ đồ là tài sản triển khai, không là kết quả thực nghiệm. Chỉ công bố model/training/benchmark đã chạy khi có outputs, manifest và kết quả kiểm tra thực tế do Lead bàn giao. Những phần chưa được đo/duyệt vẫn giữ là assumption hoặc việc chờ làm; không chuyển thành tuyên bố đã đạt.
