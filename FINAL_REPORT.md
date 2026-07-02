# BÁO CÁO ĐỒ ÁN CUỐI KỲ

## Đề tài

**Diff-GVAE: Học biểu diễn đa phương thức bằng GVAE và tăng cường dữ liệu latent bằng DDPM cho bài toán dự đoán đáp ứng điều trị miễn dịch PD-(L)1 ở bệnh nhân NSCLC**

**Sinh viên thực hiện:** cần bổ sung  
**Mã số sinh viên:** cần bổ sung  
**Lớp:** cần bổ sung  
**Giảng viên hướng dẫn:** cần bổ sung  
**Học phần:** cần bổ sung  
**Đơn vị:** cần bổ sung  
**Thời gian:** cần bổ sung

---

# Mục lục

1. Giới thiệu đề tài  
2. Cơ sở lý thuyết  
3. Dữ liệu và tiền xử lý  
4. Phương pháp đề xuất  
5. Thiết kế và cài đặt hệ thống  
6. Thực nghiệm và kết quả  
7. Đánh giá, hạn chế và hướng phát triển  
8. Kết luận  
9. Tài liệu tham khảo

---

# 1. Giới thiệu đề tài

## 1.1. Bối cảnh

Project `Diff-GVAE` tập trung vào bài toán dự đoán đáp ứng điều trị miễn dịch PD-(L)1 ở bệnh nhân ung thư phổi không tế bào nhỏ, NSCLC. Theo cấu trúc project và `PROJECT_REVIEW.md`, hệ thống khai thác dữ liệu đa phương thức gồm:

- `clinical`: đặc trưng lâm sàng cấp bệnh nhân.
- `pathology`: đặc trưng mô bệnh học cấp bệnh nhân.
- `radiology`: đặc trưng hình ảnh cấp lesion, sau đó được tổng hợp lên cấp bệnh nhân.

Bài toán được cài đặt như bài toán phân lớp nhị phân với nhãn `patient.binary_label` trong file dữ liệu chính `data_ln_pc_ihc_g.pt`.

## 1.2. Mục tiêu

Mục tiêu chính của project là xây dựng pipeline học biểu diễn đa phương thức bằng Graph Variational Autoencoder, GVAE, để dự đoán đáp ứng điều trị. GVAE là mô hình dự đoán chính: mô hình học latent representation cho từng modality, hợp nhất các latent view và đưa ra logit dự đoán response.

DDPM không được dùng làm classifier trong pipeline đúng hiện tại. Vai trò của DDPM là học phân phối latent và sinh dữ liệu giả, hay dữ liệu tổng hợp, trong latent space. Latent đầu vào dành cho DDPM là:

```text
concat_mu = concat([clinical_mu, pathology_mu, radiology_mu])
```

Trong đó `clinical_mu`, `pathology_mu`, `radiology_mu` là vector mean do encoder của GVAE sinh ra cho từng modality.

## 1.3. Phạm vi báo cáo

Báo cáo này chỉ dựa trên thông tin có thật trong project và file `PROJECT_REVIEW.md`. Những thông tin chưa tìm thấy trong source code, artifact hoặc tài liệu nội bộ được ghi rõ là **cần bổ sung**.

---

# 2. Cơ sở lý thuyết

## 2.1. Multimodal learning

Multimodal learning là hướng tiếp cận học máy trong đó mô hình khai thác nhiều nguồn thông tin khác nhau cho cùng một đối tượng. Trong project này, một bệnh nhân có thể có dữ liệu lâm sàng, mô bệnh học và hình ảnh. Các modality không hoàn toàn đồng nhất: clinical và pathology là đặc trưng cấp bệnh nhân, trong khi radiology ban đầu là đặc trưng cấp lesion.

Các thách thức chính trong project:

- Không phải bệnh nhân nào cũng có đủ modality.
- Radiology cần được tổng hợp từ nhiều lesion về một vector cấp bệnh nhân.
- Cần học latent representation vừa phục vụ dự đoán response, vừa có thể dùng cho tăng cường dữ liệu latent.

Trong code, missing modality được xử lý bằng `missing_strategy`; run hiện hành dùng `learnable`, tức là view vắng mặt có thể được thay bằng embedding học được.

## 2.2. Graph Variational Autoencoder, GVAE

GVAE trong project là mô hình chính để học biểu diễn đa phương thức và dự đoán đáp ứng điều trị. Với từng modality, `ViewEncoder` ánh xạ feature và graph structure thành tham số của phân phối Gaussian latent:

```text
mu, logvar = encoder(x_view, edge_index_view, edge_attr_view)
```

Trong huấn luyện, latent sample `z` được tạo bằng reparameterization:

```text
z = mu + eps * exp(0.5 * logvar)
```

Mỗi view có:

- Encoder sinh `mu` và `logvar`.
- Attribute decoder tái tạo feature.
- Structure decoder tái tạo adjacency/subgraph.
- Projection head cho contrastive loss giữa các view.

Sau khi có latent của các view, `FusionAndClassifierHead` dùng CLS token và multi-head attention để hợp nhất các view thành representation chung. Classifier MLP sau đó sinh một logit nhị phân. Logit này là đầu ra dự đoán response của GVAE.

## 2.3. DDPM

DDPM trong pipeline đúng hiện tại là conditional latent-space generator. Mô hình nhận latent `concat_mu` của train fold, thêm nhiễu theo timestep và học dự đoán noise bằng MSE. Khi sampling, DDPM bắt đầu từ noise và đảo ngược qua các timestep để sinh latent tổng hợp theo class condition.

Trong project, `ConditionalDDPM` và denoising network `DenoiseDCAMLP`/`DenoiseUNet` phục vụ việc sinh latent. Summary của run conditional DDPM ghi rõ scheme điều kiện:

- token 0: unconditional token.
- token 1: negative label token.
- token 2: positive label token.

DDPM không sinh logits, không sinh probability dự đoán response và không được báo cáo như classifier. Kết quả liên quan đến DDPM trong pipeline đúng phải được hiểu là:

```text
Hiệu năng của downstream classifier sau khi train bằng real concat_mu
hoặc real concat_mu + DDPM synthetic latents.
```

## 2.4. Metric đánh giá

Project sử dụng các metric phân lớp nhị phân sau:

- **AUC / ROC-AUC:** đo khả năng xếp hạng điểm dự đoán giữa hai lớp.
- **PR-AUC / Average Precision:** đo diện tích dưới đường precision-recall; cần đọc kèm positive prevalence vì dữ liệu trong project lệch về lớp positive.
- **Accuracy:** tỷ lệ dự đoán đúng trên tổng số mẫu.
- **Balanced Accuracy:** trung bình giữa sensitivity và specificity, hữu ích khi dữ liệu lệch lớp.
- **Precision:** tỷ lệ dự đoán positive đúng trong tất cả mẫu được dự đoán positive.
- **Recall / Sensitivity:** tỷ lệ positive thật được mô hình phát hiện.
- **F1-score:** trung bình điều hòa giữa precision và recall.

Với latent tổng hợp, project còn có metric chất lượng phân phối như MMD, coverage, class-wise mean/covariance distance và same-class kNN distance. Các metric này đánh giá synthetic latent, không phải hiệu năng dự đoán lâm sàng.

---

# 3. Dữ liệu và tiền xử lý

## 3.1. File dữ liệu chính

Runner hiện hành dùng file:

```text
data_ln_pc_ihc_g.pt
```

File này là một `torch_geometric.data.HeteroData` gồm hai node type:

- `patient`
- `lesion`

Các edge type:

- `('patient', 'similar_to_clinical', 'patient')`
- `('patient', 'similar_to_pathology', 'patient')`
- `('patient', 'has_lesion', 'lesion')`
- `('patient', 'similar_to_radiology', 'patient')`

Thông tin inspect trực tiếp từ `data_ln_pc_ihc_g.pt`:

| Thành phần | Giá trị |
|---|---:|
| Số patient | 247 |
| Số lesion | 333 |
| `patient.x_clinical` | `(247, 22)` |
| `patient.x_pathology` | `(247, 15)` |
| `lesion.x` | `(333, 34)` |
| `patient.binary_label` | 62 mẫu lớp 0, 185 mẫu lớp 1 |
| `pathology_mask=True` | 105 patient |
| `radiology_mask=True` | 187 patient |

Ý nghĩa lâm sàng chính xác của `binary_label=0` và `binary_label=1` chưa được mô tả rõ trong source code. Phần này **cần bổ sung**.

## 3.2. Các modality

### Clinical

Clinical là feature cấp patient. Theo `CLAUDE.md` và code reconstruction loss, 22 cột clinical được chia thành:

- cột 0 đến 4: feature liên tục.
- cột 5 đến 21: feature nhị phân.

Trong reconstruction loss, phần liên tục dùng MSE, phần nhị phân dùng BCEWithLogitsLoss với pos weight theo train fold.

### Pathology

Pathology là feature cấp patient. Dữ liệu chính có `patient.x_pathology` kích thước `(247, 15)`. `PROJECT_REVIEW.md` mô tả đây là các texture feature GLCM. Danh sách và ý nghĩa từng feature **cần bổ sung**.

### Radiology

Radiology ban đầu là feature cấp lesion, `lesion.x` kích thước `(333, 34)`. Trong GVAE, radiology không được đưa trực tiếp vào classifier ở cấp lesion. `RadiologyLesionAttentionAggregator` tổng hợp lesion features thành embedding cấp patient bằng attention theo lesion của từng patient. Embedding radiology cấp patient sau đó đi qua radiology VAE encoder.

## 3.3. Missing modality

Project có `pathology_mask` và `radiology_mask` để xác định bệnh nhân có modality tương ứng. Clinical được giả định có cho tất cả bệnh nhân trong batch. Khi modality vắng mặt, GVAE dùng missing embedding theo `missing_strategy`.

## 3.4. Tiền xử lý

Trong source code có utility `utils/data_utils.py` để:

- trích subgraph và feature theo view.
- tạo dense adjacency cho reconstruction loss.
- hỗ trợ PCA theo train fold nếu được cấu hình.

Tuy nhiên, project không có script raw-to-`HeteroData` để tái tạo `data_ln_pc_ihc_g.pt` từ dữ liệu clinical/pathology/radiology gốc. Vì vậy, các bước tiền xử lý thấp hơn như làm sạch raw data, tạo similarity edges, mã hóa feature và tạo label đều **cần bổ sung** nếu báo cáo yêu cầu đầy đủ quy trình dữ liệu.

## 3.5. Rủi ro dữ liệu

Repo có thêm file:

```text
data/data_247.pt
```

File này cũng có 247 patient và 333 lesion nhưng feature dimension khác:

- `patient.x_clinical`: `(247, 64)`
- `patient.x_pathology`: `(247, 137)`
- `lesion.x`: `(333, 1671)`

Quan trọng hơn, `binary_label` trong file này có phân bố ngược với `data_ln_pc_ihc_g.pt`: 185 mẫu lớp 0 và 62 mẫu lớp 1. Nếu thay đổi file dữ liệu mà không ghi rõ label polarity, kết quả có thể bị diễn giải sai. Báo cáo này xem `data_ln_pc_ihc_g.pt` là dữ liệu chính vì runner hiện hành mặc định load file này.

---

# 4. Phương pháp đề xuất

## 4.1. Tổng quan pipeline đúng hiện tại

Pipeline đúng hiện tại:

1. Load `data_ln_pc_ihc_g.pt` dưới dạng `HeteroData`.
2. Chia patient bằng stratified k-fold theo `patient.binary_label`.
3. Huấn luyện GVAE trên train fold.
4. Đánh giá GVAE classifier head trên validation fold.
5. Lưu checkpoint GVAE và metric.
6. Trích latent encoder từ checkpoint GVAE, trong đó DDPM dùng `concat_mu`.
7. Huấn luyện conditional DDPM trên train-fold `concat_mu`.
8. Sinh synthetic latent theo class và tỷ lệ augmentation.
9. Train downstream classifier trên real latent hoặc real + synthetic latent.
10. Đánh giá downstream classifier trên validation fold.

## 4.2. GVAE là mô hình dự đoán chính

GVAE nhận dữ liệu đa phương thức và sinh response logit. Loss huấn luyện GVAE gồm:

- Classification BCE cho `binary_label`.
- Cross-view contrastive loss giữa các view có mặt.
- Attribute reconstruction loss cho feature của từng view.
- Structure reconstruction loss cho graph adjacency/subgraph.
- KL divergence cho latent distribution.

Run hiện hành `gvae_latent_quality_codex_20260615_204355` có cấu hình chính:

- `n_splits = 5`
- `epochs = 80`
- `batch_size = 64`
- `device = cpu`
- `checkpoint_metric = latent_quality`
- `early_stopping_metric = latent_quality`
- `top_k_gvae_checkpoints = 3`
- `d_embed = 64`
- `missing_strategy = learnable`
- `pretrain_epochs = 80`
- `pretrain_use_pos_weight = true`
- `pretrain_val_split = 0.2`
- `vectorized_contrastive = true`

## 4.3. Cách GVAE học latent representation

Với mỗi view, GVAE thực hiện:

```text
x_view, edge_view -> ViewEncoder -> mu_view, logvar_view
mu_view, logvar_view -> reparameterization -> z_view
z_view -> decoder -> tái tạo feature và structure
mu_view -> projection head -> contrastive loss
z_view của các view -> fusion -> classifier head -> response logit
```

Radiology có bước riêng:

```text
lesion.x + patient-has-lesion edges
-> RadiologyLesionAttentionAggregator
-> patient-level radiology embedding
-> radiology ViewEncoder
```

## 4.4. Cách trích xuất `mu` từ GVAE

Latent extraction nằm trong `utils/latent_extraction.py`. Hàm `_extract_split_latents` gọi `get_separate_view_latent_params`, sau đó tạo:

- `per_view_mu`
- `per_view_logvar`
- `view_mask`
- `stacked_mu`
- `stacked_z`
- `concat_mu`
- `concat_z`
- `fused_cls_mu`
- `labels`
- `patient_indices`

DDPM trong pipeline đúng chỉ nên dùng:

```text
concat_mu = stacked_mu.reshape(N, 3 * d_embed)
```

Theo thiết kế của project, `concat_mu` là vector nối các mean latent theo thứ tự clinical, pathology, radiology. Artifact latent ghi `classifier_head_used_as_ddpm_input = False`, và note trong code khẳng định classifier logits/probabilities không phải DDPM input.

## 4.5. Cách DDPM học và sinh latent tổng hợp

Pipeline DDPM đúng hiện tại nằm trong `training/latent_ddpm_augmentation.py`. DDPM:

- nhận train-fold latent `concat_mu`.
- fit `StandardScaler` chỉ trên train latent.
- dùng conditional DDPM để học noise prediction MSE.
- condition theo class label.
- sinh synthetic latent theo các mode `minority_only`, `responder_only`, `both_classes`.
- inverse transform latent về scale ban đầu bằng scaler đã fit trên train.

Sau khi sinh latent, project có thể lọc synthetic samples bằng same-class train kNN distance. Ngưỡng lọc được fit từ train fold, không dùng validation/test.

## 4.6. Downstream evaluation

Sau khi có synthetic latent, project train downstream classifier trên:

- real train `concat_mu` only.
- real train `concat_mu` + DDPM synthetic latent.
- real train `concat_mu` + filtered DDPM synthetic latent.
- tùy chọn PCA train-fold trước khi DDPM và downstream classifier.

Classifier downstream mặc định trong runner là logistic regression với `class_weight="balanced"`. Kết quả này đánh giá hiệu quả của tăng cường latent, không phải hiệu quả dự đoán trực tiếp của DDPM.

---

# 5. Thiết kế và cài đặt hệ thống

## 5.1. Cấu trúc repository

| Đường dẫn | Vai trò |
|---|---|
| `models/gvae_model.py` | Wrapper GVAE, forward pass, missing view handling, fusion/classifier, latent extraction helper |
| `models/gvae_components.py` | `ViewEncoder`, decoder, fusion head, radiology attention aggregator |
| `models/ddpm.py` | `ConditionalDDPM`, `UnconditionalDDPM`, DCA/denoise components |
| `models/unet.py` | Denoising network cho DDPM |
| `training/train_gvae.py` | Training GVAE theo k-fold, losses, metric và checkpoint ranking |
| `training/latent_ddpm_augmentation.py` | Pipeline đúng cho conditional latent DDPM augmentation |
| `training/train_pipeline.py` | Pipeline legacy DDPM-as-classifier, bị chặn mặc định |
| `training/train_ddpm.py` | Helper training DDPM |
| `utils/data_utils.py` | Trích subgraph theo view, PCA train-fold |
| `utils/latent_extraction.py` | Load checkpoint GVAE và xuất latent artifact cho DDPM |
| `utils/classification_eval.py` | Metric, threshold, curve data và artifact phân lớp |
| `outputs/gvae/train_gvae_runner.py` | Runner GVAE hiện hành |
| `outputs/gvae/train_conditional_ddpm_augmentation_runner.py` | Runner conditional latent DDPM augmentation |
| `PROJECT_REVIEW.md` | Audit pipeline, rủi ro, kết quả và khuyến nghị |
| `CLAUDE.md` | Hướng dẫn repo và mô tả pipeline |

## 5.2. GVAE components

`ViewEncoder` dùng GATv2Conv để mã hóa feature và graph structure của từng view thành `mu` và `logvar`. `StructureDecoder` dùng inner product để tái tạo adjacency. `AttributeDecoder` là MLP có residual blocks để tái tạo feature.

`FusionAndClassifierHead` dùng CLS token, multi-head attention, feed-forward block và classifier MLP. Đầu ra của classifier là một logit duy nhất cho bài toán phân lớp nhị phân.

`RadiologyLesionAttentionAggregator` tổng hợp lesion feature bằng context-aware attention. Mỗi patient có nhiều lesion sẽ được tổng hợp thành một embedding radiology cấp patient.

## 5.3. Training GVAE

Training GVAE được thực hiện trong `kfold_train_gvae`:

- Lấy label stratification từ `patient.binary_label`.
- Tạo train/validation fold bằng `StratifiedKFold`.
- Tính `pos_weight` cho BCE từ train fold.
- Nếu cấu hình, pretrain radiology aggregator với train indices.
- Train GVAE theo batch patient.
- Đánh giá validation fold.
- Lưu checkpoint tốt nhất và top-k checkpoints.

Metric GVAE được lưu trong:

```text
outputs/gvae/metrics/<run_id>/summary.json
outputs/gvae/metrics/<run_id>/fold_metrics.csv
```

## 5.4. Latent extraction

Latent extraction đọc checkpoint GVAE, khởi tạo model từ `model_config`, load `model_state_dict`, sau đó trích latent cho train/val/test split. Các artifact được lưu trong `outputs/latent_for_ddpm` hoặc thư mục output của conditional DDPM pipeline.

## 5.5. Conditional latent DDPM augmentation

Runner chính:

```text
outputs/gvae/train_conditional_ddpm_augmentation_runner.py
```

Một số tham số mặc định:

- `--data-path data_ln_pc_ihc_g.pt`
- `--checkpoint-selector rank`
- `--rank 1`
- `--latent-key concat_mu`
- `--ratios 0.25,0.50,1.00,2.00`
- `--augmentation-modes both_classes`
- `--epochs 120`
- `--timesteps 250`
- `--classifier logistic_regression`

Run result được lưu trong:

```text
outputs/conditional_latent_ddpm/<run_id>/
```

## 5.6. Testing

Thư mục `tests/` có test cho:

- GVAE forward/backward compatibility.
- View encoder và logvar clamp.
- Radiology aggregator.
- Radiology zero-lesion passthrough.
- Contrastive loss vectorized so với loop.
- Latent extraction chỉ dùng encoder latent, không dùng classifier logits.
- Conditional DDPM augmentation modes.
- Same-class train kNN filtering.
- Train-only PCA.
- Smoke test training loop.

Báo cáo này không chạy lại test suite; thông tin trên được tổng hợp từ file test hiện có.

---

# 6. Thực nghiệm và kết quả

## 6.1. Thiết lập thực nghiệm

Project có artifact thực nghiệm đã lưu trong `outputs/`. Báo cáo này dùng các kết quả có thật sau:

- GVAE run hiện hành: `gvae_latent_quality_codex_20260615_204355`
- Conditional latent DDPM run: `conditional_latent_ddpm_from_gvae_latent_quality_codex_20260615_204355_20260615_213250`
- File tổng hợp best result: `outputs/best_gvae_ddpm_result.json`

Dữ liệu chính: `data_ln_pc_ihc_g.pt`, 247 patient, 5-fold stratified cross-validation.

## 6.2. Kết quả GVAE direct prediction

Kết quả từ `outputs/gvae/metrics/gvae_latent_quality_codex_20260615_204355/summary.json`:

| Metric | Mean | Std |
|---|---:|---:|
| ROC-AUC | 0.6894 | 0.0537 |
| PR-AUC | 0.8523 | 0.0326 |
| Accuracy | 0.7409 | 0.0712 |
| Balanced Accuracy | 0.7265 | 0.0418 |
| Precision | 0.8826 | 0.0325 |
| Recall | 0.7568 | 0.1209 |
| F1-score | 0.8101 | 0.0641 |
| Brier score | 0.2289 | 0.0129 |

Đây là kết quả của GVAE classifier head, tức mô hình dự đoán chính trong project.

Project cũng có run GVAE trước đó `gvae_20260614_171332` với `checkpoint_metric = auc`, `epochs = 200`, `top_k_gvae_checkpoints = 5`. Kết quả run này:

| Metric | Mean | Std |
|---|---:|---:|
| ROC-AUC | 0.7469 | 0.0440 |
| PR-AUC | 0.9020 | 0.0197 |
| Accuracy | 0.6927 | 0.0683 |
| Balanced Accuracy | 0.6027 | 0.0767 |
| Precision | 0.8173 | 0.0501 |
| Recall | 0.7784 | 0.2035 |
| F1-score | 0.7813 | 0.0787 |

Hai run này khác nhau về checkpoint metric và cấu hình epoch, nên không nên xem như so sánh ablation công bằng nếu chưa có protocol rõ ràng hơn.

## 6.3. Kết quả downstream và DDPM latent augmentation

Kết quả DDPM augmentation được đọc từ `outputs/best_gvae_ddpm_result.json` và `summary.json` của run conditional DDPM. Tất cả dòng DDPM dưới đây là **kết quả downstream classifier sau latent augmentation**, không phải DDPM classifier.

| Thiết lập | ROC-AUC | PR-AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1-score | Synthetic count TB | MMD TB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Downstream real `concat_mu` only | 0.6705 | 0.8562 | 0.6793 | 0.6872 | 0.9024 | 0.6757 | 0.7293 | 0.0 | cần bổ sung |
| Best by ROC-AUC: `both_classes`, ratio 0.5, unfiltered | 0.6805 | 0.8597 | 0.6638 | 0.6901 | 0.8988 | 0.6378 | 0.7142 | 98.6 | 0.0509 |
| Best by Balanced Accuracy: `minority_only`, ratio 2.0, unfiltered | 0.6747 | 0.8590 | 0.7404 | 0.7052 | 0.8824 | 0.7784 | 0.7993 | 99.2 | 0.0519 |
| Best by PR-AUC: `minority_only`, ratio 0.5, unfiltered | 0.6753 | 0.8631 | 0.6598 | 0.6930 | 0.9047 | 0.6270 | 0.7120 | 24.6 | 0.0934 |

Ghi chú: các giá trị accuracy downstream được tính từ `fold_results` trong `summary.json` bằng trung bình metric `best_balanced_accuracy_threshold` của từng fold, vì `outputs/best_gvae_ddpm_result.json` không lưu sẵn mean accuracy.

## 6.4. Diễn giải kết quả

Kết quả GVAE direct prediction của run `gvae_latent_quality_codex_20260615_204355` có balanced accuracy cao hơn run `gvae_20260614_171332`, trong khi run cũ có ROC-AUC và PR-AUC cao hơn. Điều này cho thấy việc chọn checkpoint theo metric khác nhau có thể thay đổi trade-off giữa ranking metric và thresholded metric.

Với DDPM augmentation, các cấu hình tốt nhất theo ROC-AUC/PR-AUC/Balanced Accuracy chỉ cải thiện nhẹ so với downstream real `concat_mu` only, và không vượt GVAE direct prediction trong bảng kết quả chính. Vì vậy, kết quả hiện tại ủng hộ cách hiểu DDPM là nhánh augmentation cần tinh chỉnh thêm, không phải thành phần thay thế GVAE classifier head.

`outputs/best_gvae_ddpm_result.json` có ghi chú rằng filtered branches có thể trùng real-only khi toàn bộ generated samples bị loại bởi same-class kNN distance filtering. Điều này cho thấy synthetic latent hiện tại có thể nằm khá xa phân phối real train latent trong một số cấu hình.

## 6.5. Lưu ý về legacy DDPM-as-classifier

Repo vẫn còn code legacy trong `training/train_pipeline.py` và một số runner cũ. Nhánh này huấn luyện DDPM riêng cho responder/non-responder và biến denoising loss thành score/probability. Code đã chặn mặc định nếu không bật `allow_deprecated_ddpm_classifier`.

Báo cáo này không xem nhánh legacy đó là pipeline chính, vì nó mâu thuẫn với mục tiêu hiện tại: DDPM không phải classifier và không được dùng để dự đoán response.

---

# 7. Đánh giá, hạn chế và hướng phát triển

## 7.1. Điểm mạnh

- Project có pipeline GVAE rõ ràng cho học biểu diễn đa phương thức và dự đoán response.
- Có xử lý missing modality thông qua mask và missing embedding.
- Radiology được xử lý ở mức lesion-to-patient bằng attention aggregator.
- Latent extraction tách bạch encoder latent với classifier logits/probabilities.
- Conditional DDPM augmentation ghi rõ `ddpm_is_classifier = False`.
- Scaler/PCA trong nhánh conditional augmentation được fit trên train fold.
- Có test cho nhiều thành phần quan trọng, đặc biệt latent extraction và DDPM augmentation.

## 7.2. Hạn chế

- Chưa có script tạo `data_ln_pc_ihc_g.pt` từ raw data, nên provenance và data leakage trong graph construction chưa audit đầy đủ.
- Chưa có mô tả rõ ý nghĩa của lớp 0/1 trong `binary_label`.
- Có hai file graph với label polarity ngược nhau, tạo nguy cơ diễn giải sai.
- `README.md` còn quá ngắn; `configs/config.py` trong repo hiện tại rỗng.
- Các setting thực nghiệm nằm nhiều trong runner/script, chưa tập trung thành config tái lập hoàn chỉnh.
- Legacy DDPM-as-classifier path vẫn tồn tại, dễ gây nhầm lẫn nếu chạy nhầm runner.
- DDPM augmentation chưa cho thấy cải thiện rõ ràng so với GVAE direct prediction trong kết quả hiện có.
- Kết quả hiện tại là cross-validation validation fold; chưa thấy held-out test set độc lập.
- Thresholded metrics có thể optimistic nếu threshold được chọn trên cùng validation split.
- Chưa có confidence interval/repeated CV/seed sweep để đánh giá độ ổn định.

## 7.3. Hướng phát triển

- Bổ sung script raw-to-`HeteroData` và data card cho dataset.
- Chuẩn hóa label polarity và ghi rõ lớp positive/negative.
- Đưa legacy DDPM-as-classifier vào thư mục deprecated hoặc loại bỏ khỏi runner chính.
- Giới hạn API DDPM chỉ nhận `concat_mu` nếu đây là thiết kế cuối cùng.
- Bổ sung config tập trung cho model, train, latent extraction và DDPM augmentation.
- Đánh giá trên held-out test set độc lập hoặc nested cross-validation.
- Thêm repeated CV/seed sweep và bootstrap confidence interval.
- Cải thiện filtering/sampling DDPM để synthetic latent gần phân phối real train latent hơn.
- Lưu đầy đủ per-sample predictions, ROC/PR curves và confusion matrices cho downstream DDPM experiments.
- Bổ sung tài liệu tham khảo học thuật chính thức cho multimodal learning, GVAE, DDPM và metric.

---

# 8. Kết luận

Project `Diff-GVAE` xây dựng một pipeline đa phương thức cho bài toán dự đoán đáp ứng điều trị PD-(L)1 ở bệnh nhân NSCLC. Thành phần trung tâm là GVAE, có khả năng học latent representation riêng cho clinical, pathology và radiology, sau đó hợp nhất các view để dự đoán response bằng classifier head.

DDPM trong thiết kế đúng không phải classifier. DDPM chỉ được dùng để học phân phối và sinh latent tổng hợp trong không gian `concat_mu`. Kết quả DDPM vì vậy cần được báo cáo như hiệu năng của downstream classifier sau khi thêm synthetic latent, không phải hiệu năng dự đoán của DDPM.

Kết quả thực nghiệm hiện có cho thấy GVAE direct prediction đạt ROC-AUC 0.6894, PR-AUC 0.8523, balanced accuracy 0.7265 và F1-score 0.8101 trong run hiện hành theo `latent_quality`. Conditional DDPM augmentation có một số cấu hình đạt chỉ số downstream tốt hơn real latent only ở từng metric, nhưng chưa vượt GVAE direct prediction trong bảng kết quả chính. Hướng phát triển quan trọng nhất là chuẩn hóa dữ liệu, bổ sung provenance, loại bỏ nhầm lẫn DDPM-as-classifier, và cải thiện/thẩm định synthetic latent augmentation bằng protocol thực nghiệm chặt chẽ hơn.

---

# 9. Tài liệu tham khảo

## 9.1. Tài liệu nội bộ trong project

1. `PROJECT_REVIEW.md` - audit pipeline, kết quả, rủi ro và recommended fix plan.
2. `CLAUDE.md` - mô tả repo, data format, key functions, architecture notes và training configuration.
3. `models/gvae_model.py` - cài đặt GVAE, forward pass và latent extraction helper.
4. `models/gvae_components.py` - cài đặt ViewEncoder, decoder, fusion head và radiology attention aggregator.
5. `models/ddpm.py` - cài đặt ConditionalDDPM và các thành phần denoising.
6. `training/train_gvae.py` - training loop GVAE, loss, metric và checkpoint ranking.
7. `training/latent_ddpm_augmentation.py` - pipeline conditional latent DDPM augmentation.
8. `utils/latent_extraction.py` - trích xuất latent artifact cho DDPM.
9. `utils/classification_eval.py` - tính metric phân lớp và artifact evaluation.
10. `outputs/gvae/metrics/gvae_latent_quality_codex_20260615_204355/summary.json` - kết quả GVAE run hiện hành.
11. `outputs/best_gvae_ddpm_result.json` - tổng hợp kết quả GVAE và conditional DDPM augmentation.

## 9.2. Tài liệu học thuật cần bổ sung

- Cần bổ sung citation chính thức cho multimodal learning.
- Cần bổ sung citation chính thức cho Graph Variational Autoencoder.
- Cần bổ sung citation chính thức cho Denoising Diffusion Probabilistic Model.
- Cần bổ sung citation chính thức cho ROC-AUC, PR-AUC và các metric phân lớp nếu giảng viên yêu cầu format tài liệu tham khảo học thuật.
