# DÀN Ý BÁO CÁO ĐỒ ÁN CUỐI KỲ

## Tên đề tài

Diff-GVAE: Học biểu diễn đa phương thức bằng GVAE và tăng cường dữ liệu latent bằng DDPM cho bài toán dự đoán đáp ứng điều trị miễn dịch PD-(L)1 ở bệnh nhân NSCLC.

## 1. Trang bìa

- Tên trường/khoa: cần bổ sung.
- Tên học phần: cần bổ sung.
- Tên đề tài.
- Sinh viên, mã số sinh viên, lớp: cần bổ sung.
- Giảng viên hướng dẫn: cần bổ sung.
- Thời gian nộp: cần bổ sung.

## 2. Mục lục

- Liệt kê các chương/phụ lục của báo cáo.

## 3. Giới thiệu đề tài

- Bối cảnh NSCLC và điều trị miễn dịch PD-(L)1.
- Bài toán phân lớp đáp ứng điều trị bằng `binary_label`.
- Lý do cần multimodal learning: clinical, pathology, radiology.
- Mục tiêu project: GVAE là mô hình dự đoán chính; DDPM chỉ sinh latent tổng hợp.

## 4. Cơ sở lý thuyết

- Multimodal learning.
- GVAE:
  - encoder theo view.
  - `mu`, `logvar`, reparameterization.
  - reconstruction, KL, contrastive learning.
  - fusion và classifier head.
- DDPM:
  - forward noising.
  - denoising/noise prediction MSE.
  - conditional sampling theo class.
  - vai trò generator-only trong project.
- Metric:
  - AUC/ROC-AUC.
  - PR-AUC.
  - Accuracy.
  - Balanced Accuracy.
  - Precision, Recall, F1-score.
  - MMD/coverage/kNN distance cho synthetic latent.

## 5. Dữ liệu và tiền xử lý

- File chính: `data_ln_pc_ihc_g.pt`.
- Cấu trúc `HeteroData`:
  - node types: `patient`, `lesion`.
  - edge types: clinical/pathology/radiology similarity và patient-has-lesion.
- Thống kê:
  - 247 patient.
  - 333 lesion.
  - clinical `(247, 22)`.
  - pathology `(247, 15)`.
  - lesion radiology `(333, 34)`.
  - `binary_label`: 62 lớp 0, 185 lớp 1.
  - `pathology_mask=True`: 105.
  - `radiology_mask=True`: 187.
- Missing modality và mask.
- Rủi ro:
  - chưa có raw-to-graph script.
  - `data/data_247.pt` có feature dimension khác và label polarity ngược.

## 6. Phương pháp đề xuất

- Pipeline GVAE:
  - load data.
  - stratified k-fold.
  - per-view encoder.
  - radiology lesion attention aggregation.
  - fusion bằng CLS-token attention.
  - classifier head dự đoán response.
- Loss:
  - BCEWithLogitsLoss.
  - reconstruction feature.
  - reconstruction graph.
  - KL.
  - cross-view contrastive loss.
- Latent extraction:
  - trích `clinical_mu`, `pathology_mu`, `radiology_mu`.
  - tạo `concat_mu`.
  - không dùng classifier logits/probabilities.
- DDPM:
  - train conditional DDPM trên train-fold `concat_mu`.
  - sinh synthetic latent theo class.
  - downstream classifier đánh giá augmentation.

## 7. Thiết kế và cài đặt hệ thống

- `models/gvae_model.py`.
- `models/gvae_components.py`.
- `models/ddpm.py`.
- `training/train_gvae.py`.
- `training/latent_ddpm_augmentation.py`.
- `utils/data_utils.py`.
- `utils/latent_extraction.py`.
- `utils/classification_eval.py`.
- Runner:
  - `outputs/gvae/train_gvae_runner.py`.
  - `outputs/gvae/train_conditional_ddpm_augmentation_runner.py`.
- Tests trong `tests/`.

## 8. Thực nghiệm và kết quả

- Dataset và split: 5-fold stratified CV trên 247 patient.
- GVAE run chính: `gvae_latent_quality_codex_20260615_204355`.
- Kết quả GVAE:
  - ROC-AUC 0.6894.
  - PR-AUC 0.8523.
  - Accuracy 0.7409.
  - Balanced Accuracy 0.7265.
  - Precision 0.8826.
  - Recall 0.7568.
  - F1-score 0.8101.
- Conditional DDPM run:
  - `conditional_latent_ddpm_from_gvae_latent_quality_codex_20260615_204355_20260615_213250`.
  - `ddpm_is_classifier = false`.
  - `latent_key = concat_mu`.
- Downstream comparison:
  - real `concat_mu` only.
  - real + synthetic by `both_classes`, ratio 0.5.
  - real + synthetic by `minority_only`, ratio 2.0.
  - real + synthetic by `minority_only`, ratio 0.5.
- Diễn giải:
  - DDPM augmentation chưa vượt GVAE direct prediction trong kết quả chính.
  - filtered branches có thể loại hết synthetic samples.

## 9. Đánh giá, hạn chế và hướng phát triển

- Điểm mạnh:
  - multimodal GVAE rõ ràng.
  - latent extraction tách classifier output.
  - DDPM generator-only metadata rõ.
  - train-only scaler/PCA trong nhánh conditional augmentation.
- Hạn chế:
  - thiếu raw data/preprocessing provenance.
  - label semantics cần bổ sung.
  - file dữ liệu thứ hai có label polarity ngược.
  - README/config còn yếu.
  - legacy DDPM-as-classifier path còn tồn tại.
  - chưa có held-out test độc lập.
- Hướng phát triển:
  - raw-to-HeteroData script.
  - data validation.
  - chuẩn hóa config.
  - repeated CV/seed sweep.
  - nested CV/held-out test.
  - cải thiện DDPM synthetic quality.
  - bổ sung citation học thuật.

## 10. Kết luận

- GVAE là thành phần dự đoán chính.
- DDPM chỉ sinh latent tổng hợp từ `concat_mu`.
- Kết quả hiện có cho thấy GVAE có hiệu năng phân lớp có ý nghĩa, DDPM augmentation cần tiếp tục tinh chỉnh.

## 11. Tài liệu tham khảo

- `PROJECT_REVIEW.md`.
- `CLAUDE.md`.
- Source files trong `models/`, `training/`, `utils/`.
- Artifact metrics trong `outputs/`.
- Tài liệu học thuật: cần bổ sung.
