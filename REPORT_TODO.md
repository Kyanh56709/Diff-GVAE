# CÁC THÔNG TIN CẦN BỔ SUNG CHO BÁO CÁO

## 1. Thông tin hành chính

| Mục | Trạng thái |
|---|---|
| Tên sinh viên | cần bổ sung |
| Mã số sinh viên | cần bổ sung |
| Lớp/khoa | cần bổ sung |
| Tên học phần | cần bổ sung |
| Giảng viên hướng dẫn | cần bổ sung |
| Tên trường/đơn vị | cần bổ sung |
| Ngày nộp | cần bổ sung |

## 2. Thông tin dữ liệu

| Mục | Trạng thái (cập nhật 2026-10-04) |
|---|---|
| Ý nghĩa `binary_label` | **Xong** — `1` = non-responder (185), `0` = responder (62); owner xác nhận 2026-08-06. Đã ghi vào `FINAL_REPORT.md` §3.1. |
| Mô tả cohort NSCLC | **Còn mở** — nguồn dữ liệu, tiêu chí chọn bệnh nhân, thời gian thu thập, inclusion/exclusion. |
| Định nghĩa clinical / pathology / radiology features | **Xong** — `research/2026-08-06-data-dictionary/`; thứ tự slot trong `data/build_ln_pc_ihc_g.py`. Canonical r32 có 32 radiomics/lesion. |
| Raw-to-graph preprocessing script | **Xong** — `data/build_ln_pc_ihc_g.py` (tái lập bit-exact; `research/2026-10-03-a1-build-script/`). |
| Quy trình tạo similarity edges | **Xong** — cosine > 0.8 chỉ trên feature, không đọc nhãn (A4). |
| Xử lý missing values raw | **Xong (cần viết vào Methods)** — median/mode trên cohort cố định khi build, chỉ trên feature; scaler trong pipeline fit trên train fold (`tests/test_train_fold_only_fitting.py`). |
| Label polarity của `data/data_247.pt` | **Xong** — file cũ, đã cách ly vào `deprecated/`. |

## 3. Thông tin thực nghiệm

| Mục | Trạng thái (cập nhật 2026-10-04) |
|---|---|
| Protocol chọn final run | **Xong** — A7: `research/2026-10-03-canonical-r32/reports/final_results_package_r32.md`. |
| Held-out test set | **Còn mở** — mới có pooled-OOF; chưa có nested CV hay test set độc lập (B2). |
| Hardware/runtime | **Còn mở** (G5). |
| Seed sweep/repeated CV | **Một phần** — GVAE 5 seed (42–46); DDPM mới 1 seed. |
| Confidence interval | **Xong** — bootstrap 95% CI cho pooled-OOF. |
| Threshold selection protocol | **Còn mở** (B4) — AUC/AUPRC là metric chính; cần chốt cách báo cáo F1/BA. |
| Per-sample output | **Một phần** — GVAE: `oof_arrays.npz` (chưa lưu patient ID); DDPM: chỉ `summary.json` theo fold. |
| Accuracy aggregate trong best DDPM artifact | **Lỗi thời** — bộ số chốt r32 dùng ROC/PR/BA, không dùng `outputs/best_gvae_ddpm_result.json`. |

## 4. Thông tin phương pháp

| Mục | Lý do cần bổ sung |
|---|---|
| Lý do chọn GVAE | Cần viết thêm lập luận khoa học/lâm sàng nếu nộp báo cáo chính thức. |
| Lý do chọn `concat_mu` thay vì `concat_z`/`fused_cls_mu` | Project review yêu cầu `concat_mu`, nhưng nên bổ sung lý do thiết kế. |
| Lý do chọn conditional DDPM | Cần bổ sung lập luận vì sao sinh latent có điều kiện theo class. |
| Tiêu chí đánh giá synthetic latent | Có MMD/kNN/coverage trong code, nhưng cần chọn metric nào là chính. |
| Xử lý legacy DDPM-as-classifier | **Xong** — đã cách ly vào `deprecated/`. |

## 5. Tài liệu tham khảo

| Mục | Trạng thái |
|---|---|
| Citation multimodal learning | cần bổ sung |
| Citation Graph Variational Autoencoder | cần bổ sung |
| Citation DDPM | cần bổ sung |
| Citation GATv2Conv / graph neural network nếu cần | cần bổ sung |
| Citation ROC-AUC, PR-AUC, balanced accuracy/F1 nếu giảng viên yêu cầu | cần bổ sung |
| Format tài liệu tham khảo theo quy định trường | cần bổ sung |

## 6. Việc nên làm trước khi nộp

Đã xong: chốt nhãn, chọn run final (A7), data dictionary, script tạo `HeteroData`, test suite (96 passed, 2026-10-04), pin requirements + `configs/config.py`.

Còn lại:
1. Điền thông tin hành chính (§1).
2. Mô tả cohort (§2).
3. Bổ sung tài liệu tham khảo học thuật đúng format yêu cầu (§5).
4. Lý do thiết kế ở §4 (GVAE, `concat_mu`, conditional DDPM, metric chính cho synthetic latent).
