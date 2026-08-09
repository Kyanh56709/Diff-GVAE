# Plan — Đối chiếu repo với PROJECT_REVIEW.md & tài liệu

## Câu hỏi
"Tôi vừa hoàn thành PROJECT_REVIEW.md; các hàm để run nằm trong training_local.ipynb.
Hãy check với tài liệu: những gì đã làm và những gì còn thiếu."

## Cách trả lời (sub-questions)
1. Từng mục trong bảng Problems Found (§14) của PROJECT_REVIEW.md đã được xử lý chưa?
2. Từng mục trong Recommended Fix Plan (§15) đã hoàn thành chưa?
3. `training_local.ipynb` có thực sự chứa pipeline duy trì (maintained pipeline) như tài liệu mô tả không?
4. Các kết quả số trong PROJECT_REVIEW §11 còn tái lập được từ repo không?
5. Các yêu cầu còn mở trong README/CLAUDE.md/REPORT_TODO.md/PUBLICATION_CHECKLIST.md?

## Bằng chứng cần thu thập
- Chạy test suite (`pytest -q`).
- Chạy `review_fixes_2026_07/verify_fixes.ipynb` end-to-end (smoke-scale).
- Chạy `data_validation.py` trên cả 4 file `.pt` (canonical, deprecated, configs, data/).
- Chạy `raw_data_audit.py`.
- Đọc code trực tiếp (codegraph + grep) tại các dòng được PROJECT_REVIEW dẫn chiếu.
- Phân tích cú pháp + phụ thuộc file của training_local.ipynb (48 cells).

## Tiêu chí "đã làm"
- Có bằng chứng thực thi (test pass / notebook execute 0 lỗi / script chạy ra kết quả đúng).
- Không dùng "có vẻ đã sửa" — mọi khẳng định phải có file:dòng hoặc output lệnh.
