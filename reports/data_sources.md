
# Data Sources

## Dataset
- Tên: Electronics Sales Dataset (B-Tech Style)
- Nguồn: https://www.kaggle.com/datasets/fahdelatiby/electronics-sales-dataset-b-tech-style
- Ngày truy cập: 2026-11-25
- Phương thức tải: Tải thủ công từ Kaggle và lưu CSV vào data/raw/
- Giấy phép: Chưa xác minh - cần kiểm tra trên trang Kaggle

## Quy mô dữ liệu
- Số file CSV tìm thấy: 27
- Số bảng đọc thành công: 27
- Số file dự kiến theo yêu cầu: 27

## Bảo toàn dữ liệu
Các file trong `data/raw/` được giữ nguyên trạng.
Không thực hiện chỉnh sửa, xóa dòng hoặc thay thế giá trị tại giai đoạn thu thập.

## Lưu ý về quyền riêng tư
Cần kiểm tra các trường có thể định danh cá nhân như email, số điện thoại,
địa chỉ hoặc thông tin khách hàng. Chỉ sử dụng trường cần thiết cho mục tiêu
phân tích và tuân thủ điều kiện sử dụng của nguồn dữ liệu.

## Quan hệ giữa các bảng
Xem `relationship_candidates.xlsx`.
Các cột chung trong file này chỉ là khóa ứng viên, cần xác nhận bằng
ngữ nghĩa nghiệp vụ và kiểm tra khóa trước khi JOIN.
