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

| Mục | Lý do cần bổ sung |
|---|---|
| Ý nghĩa chính xác của `binary_label=0` và `binary_label=1` | Source code có label nhưng không mô tả rõ lớp nào là responder/non-responder. |
| Mô tả cohort NSCLC | Chưa thấy thông tin về nguồn dữ liệu, tiêu chí chọn bệnh nhân, thời gian thu thập, inclusion/exclusion criteria. |
| Định nghĩa clinical features | Code có 22 cột clinical, nhưng chưa có data dictionary từng cột. |
| Định nghĩa pathology features | Project review nói pathology là GLCM texture features, nhưng chưa có danh sách feature chi tiết. |
| Định nghĩa radiology lesion features | File chính có lesion feature 34 chiều, nhưng chưa có data dictionary. |
| Raw-to-graph preprocessing script | Không tìm thấy script tạo `data_ln_pc_ihc_g.pt` từ raw data. |
| Quy trình tạo similarity edges | Cần bổ sung cách tạo `similar_to_clinical`, `similar_to_pathology`, `similar_to_radiology`. |
| Xử lý missing values raw | Chưa thấy tài liệu mô tả imputation/normalization trước khi serialize graph. |
| Xác nhận label polarity của `data/data_247.pt` | File này có `binary_label` ngược với `data_ln_pc_ihc_g.pt`; cần xác nhận đây là file cũ, file lỗi hay label convention khác. |

## 3. Thông tin thực nghiệm

| Mục | Lý do cần bổ sung |
|---|---|
| Protocol chọn final run | Repo có nhiều run GVAE; cần ghi rõ run nào là final và vì sao. |
| Held-out test set | Artifact hiện có là 5-fold train/validation; chưa thấy held-out test độc lập. |
| Hardware/runtime | Cần bổ sung CPU/GPU, RAM, thời gian train nếu báo cáo yêu cầu. |
| Seed sweep/repeated CV | Hiện có run theo seed chính; cần bổ sung để đánh giá độ ổn định. |
| Confidence interval | Chưa có CI/bootstrap cho metric. |
| Threshold selection protocol | Thresholded metrics có thể được tối ưu trên validation; cần quy định protocol báo cáo cuối cùng. |
| Per-sample output cho DDPM/downstream | Nên lưu đầy đủ score, label, patient id để audit ROC/PR curve và calibration. |
| Accuracy aggregate trong best DDPM artifact | `outputs/best_gvae_ddpm_result.json` không lưu mean accuracy; báo cáo đã tính lại từ `summary.json`, nên nên cập nhật artifact nếu cần tái lập trực tiếp. |

## 4. Thông tin phương pháp

| Mục | Lý do cần bổ sung |
|---|---|
| Lý do chọn GVAE | Cần viết thêm lập luận khoa học/lâm sàng nếu nộp báo cáo chính thức. |
| Lý do chọn `concat_mu` thay vì `concat_z`/`fused_cls_mu` | Project review yêu cầu `concat_mu`, nhưng nên bổ sung lý do thiết kế. |
| Lý do chọn conditional DDPM | Cần bổ sung lập luận vì sao sinh latent có điều kiện theo class. |
| Tiêu chí đánh giá synthetic latent | Có MMD/kNN/coverage trong code, nhưng cần chọn metric nào là chính. |
| Xử lý legacy DDPM-as-classifier | Nên loại bỏ hoặc đưa vào deprecated để tránh chạy nhầm. |

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

1. Xác nhận ý nghĩa lớp positive/negative của `binary_label`.
2. Chọn một run GVAE final và một run DDPM augmentation final để đưa vào báo cáo.
3. Bổ sung data dictionary cho clinical/pathology/radiology features.
4. Bổ sung hoặc mô tả quy trình tạo `HeteroData`.
5. Chạy lại hoặc xác minh test suite nếu cần báo cáo tính đúng đắn cài đặt.
6. Cập nhật README/config để người khác có thể tái lập thực nghiệm.
7. Bổ sung tài liệu tham khảo học thuật đúng format yêu cầu.
