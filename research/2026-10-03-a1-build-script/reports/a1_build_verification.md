# A1 — Script rebuild raw → `data_ln_pc_ihc_g.pt` (2026-10-03)

**Trạng thái:** XONG — script portable đã viết, chạy và **khớp canonical 100%** (mọi tensor `maxdiff = 0.0`, gồm 4 edge set + edge_attr, đúng cả thứ tự edge).

- Script: `data/build_ln_pc_ihc_g.py` (không cần file Windows; chỉ dùng 3 CSV thô trong `data/`).
- Log verify: `research/2026-10-03-a1-build-script/evidence/build_verify_output.txt`.
- Script phục vụ điều tra mapping: `evidence/reverse_engineer_mappings.py`, `evidence/probe_radiology_artifacts.py`.

## Cách chạy

```bash
.venv/bin/python data/build_ln_pc_ihc_g.py \
    --out data/ln_pc_ihc_g_rebuilt.pt \
    --verify data_ln_pc_ihc_g.pt
```

`--verify` in bảng đối chiếu từng trường và trả exit code 0/1 (14/14 mục PASS).

## Kết quả verify (trích log)

| Trường | Kết quả | Delta |
|---|---|---|
| `main_index`, `x_clinical_columns` | PASS | — |
| `x_clinical` (247, 22) | PASS | maxdiff = 0.000e+00 |
| `x_pathology` (247, 15) | PASS | maxdiff = 0.000e+00 |
| `lesion.x` (333, 34) | PASS | maxdiff = 0.000e+00 |
| `pathology_mask` = 105, `radiology_mask` = 187 | PASS | — |
| `binary_label`, `event`, `y` | PASS | — |
| `similar_to_clinical` (1152) | PASS | attr maxdiff = 0.000e+00 |
| `similar_to_pathology` (1016) | PASS | attr maxdiff = 0.000e+00 |
| `has_lesion` (333) | PASS | — |
| `similar_to_radiology` (652) | PASS | attr maxdiff = 0.000e+00 |

## Pipeline tái lập

1. **Cohort (247):** `clinical_features_tmb.csv`, giữ các dòng `TMB.notna()`; thứ tự bệnh nhân = thứ tự dòng CSV.
2. **Clinical (22 = 5 + 6 + 7 + 4):**
   - 5 continuous `[albumin, dnlr, TMB, tumor_burden, clinical_pdl1_score]`: median impute (chỉ `tumor_burden` và `clinical_pdl1_score` có 1 NaN) → `clip(>=0)` → `log1p` → `RobustScaler` (refit khớp `data/clinical_scaler.pkl`; residual ≤ 2.2e-7).
   - 6 gene flags `[EGFR, ERBB2, BRAF, MET, STK11, ARID1A]_driver`: TRUE/FALSE → 1/0, NaN → 0 (khớp tuyệt đối).
   - `io_drug` MultiLabelBinarizer → 7 cột (`io_drug_*`), đúng thứ tự alphabet (khớp tuyệt đối).
   - `ecog` one-hot → `ecog_0..3` (khớp tuyệt đối).
   - Các cột thô bị loại khỏi canonical: `age`, `pack_years`, `smoking_status`, `sex`, `histo`, `pdl1_tiss_site`, `ALK/ROS1/RET_driver` (3 gene sau constant trong cohort).
3. **Pathology (15):** 15 cột GLCM dưới đây, `median impute → clip(>=0) → log1p → RobustScaler` fit trên 105 bệnh nhân có dữ liệu, zero-pad phần còn lại.
4. **Radiology (34):** 32 cột radiomics + **2 cột index** (xem cảnh báo dưới), `median impute → (log1p với 26 cột / raw với 6 cột) → RobustScaler` fit trên 333 lesion. Aggregated signature `[mean, std, min, max]` (std=0 nếu 1 lesion) → cosine > 0.8.
5. **Edges:** cosine **> 0.8** (strict), cả hai chiều, bỏ self-loop — xác nhận lại ngưỡng thật là **0.8, không phải 0.7** như tài liệu cũ.
6. **Labels:** `pfs → y` (float32, giữ NaN nếu có), `pfs_censor → event`, `label → binary_label` (long).

### Mapping pathology (slot | cột nguồn | biến đổi)

| slot | cột nguồn | biến đổi |
|---|---|---|
| 0 | `original_glcm_ClusterShade_channel_1_skewness` | log1p(clip>=0) → RobustScaler |
| 1 | `original_glcm_ClusterProminence_channel_1_kurtosis` | log1p(clip>=0) → RobustScaler |
| 2 | `original_glcm_JointAverage_channel_1_kurtosis` | log1p(clip>=0) → RobustScaler |
| 3 | `original_glcm_SumAverage_channel_1_kurtosis` | log1p(clip>=0) → RobustScaler |
| 4 | `original_glcm_Idm_channel_1_kurtosis` | log1p(clip>=0) → RobustScaler |
| 5 | `original_glcm_ClusterProminence_channel_1_variance` | log1p(clip>=0) → RobustScaler |
| 6 | `original_glcm_Autocorrelation_channel_1_kurtosis` | log1p(clip>=0) → RobustScaler |
| 7 | `original_glcm_ClusterShade_channel_1_variance` | log1p(clip>=0) → RobustScaler |
| 8 | `original_glcm_JointAverage_channel_1_skewness` | log1p(clip>=0) → RobustScaler |
| 9 | `original_glcm_SumAverage_channel_1_skewness` | log1p(clip>=0) → RobustScaler |
| 10 | `original_glcm_Imc2_channel_1_variance` | log1p(clip>=0) → RobustScaler |
| 11 | `original_glcm_DifferenceVariance_channel_1_skewness` | log1p(clip>=0) → RobustScaler |
| 12 | `original_glcm_MCC_channel_1_kurtosis` | log1p(clip>=0) → RobustScaler |
| 13 | `original_glcm_Autocorrelation_channel_1_lognorm_fit_p2` | log1p(clip>=0) → RobustScaler |
| 14 | `original_glcm_Contrast_channel_1_kurtosis` | log1p(clip>=0) → RobustScaler |

*(Review 2026-10-03 sáng nay đếm 9/15; con số đúng là 15/15 — 6 slot từng bị coi là "không khớp" thực ra khớp với residual ≤ 1.1e-7.)*

### Mapping radiology (slot | cột nguồn | biến đổi)

| slot | cột nguồn | biến đổi |
|---|---|---|
| 0 | **rank theo thứ tự file** của dòng lesion (1..333) | RobustScaler |
| 1 | **`lesion_index`** | RobustScaler |
| 2 | `logarithm_firstorder_Range` | log1p(clip>=0) → RobustScaler |
| 3 | `exponential_firstorder_RobustMeanAbsoluteDeviation` | log1p(clip>=0) → RobustScaler |
| 4 | `lbp-3D-k_glcm_Autocorrelation` | log1p(clip>=0) → RobustScaler |
| 5 | `lbp-3D-k_glcm_JointAverage` | log1p(clip>=0) → RobustScaler |
| 6 | `lbp-3D-k_glcm_SumAverage` | log1p(clip>=0) → RobustScaler |
| 7 | `exponential_firstorder_90Percentile` | log1p(clip>=0) → RobustScaler |
| 8 | `exponential_firstorder_MeanAbsoluteDeviation` | log1p(clip>=0) → RobustScaler |
| 9 | `exponential_firstorder_InterquartileRange` | log1p(clip>=0) → RobustScaler |
| 10 | `lbp-3D-k_glcm_ClusterTendency` | log1p(clip>=0) → RobustScaler |
| 11 | `wavelet-HLL_gldm_DependenceNonUniformityNormalized` | log1p(clip>=0) → RobustScaler |
| 12 | `lbp-3D-m2_firstorder_90Percentile` | log1p(clip>=0) → RobustScaler |
| 13 | `lbp-3D-k_glcm_MaximumProbability` | log1p(clip>=0) → RobustScaler |
| 14 | `lbp-3D-k_glcm_SumSquares` | log1p(clip>=0) → RobustScaler |
| 15 | `lbp-3D-m2_gldm_LowGrayLevelEmphasis` | log1p(clip>=0) → RobustScaler |
| 16 | `lbp-3D-m2_firstorder_Uniformity` | log1p(clip>=0) → RobustScaler |
| 17 | `wavelet-HLL_gldm_DependenceVariance` | raw → RobustScaler |
| 18 | `lbp-3D-m2_ngtdm_Strength` | log1p(clip>=0) → RobustScaler |
| 19 | `lbp-3D-k_glszm_GrayLevelNonUniformityNormalized` | log1p(clip>=0) → RobustScaler |
| 20 | `lbp-3D-k_glszm_GrayLevelVariance` | log1p(clip>=0) → RobustScaler |
| 21 | `lbp-3D-k_glszm_SizeZoneNonUniformityNormalized` | log1p(clip>=0) → RobustScaler |
| 22 | `logarithm_firstorder_RootMeanSquared` | raw → RobustScaler |
| 23 | `wavelet-HHL_glcm_InverseVariance` | log1p(clip>=0) → RobustScaler |
| 24 | `lbp-3D-m2_gldm_HighGrayLevelEmphasis` | log1p(clip>=0) → RobustScaler |
| 25 | `lbp-3D-k_glcm_ClusterProminence` | log1p(clip>=0) → RobustScaler |
| 26 | `wavelet-HLH_glcm_SumEntropy` | log1p(clip>=0) → RobustScaler |
| 27 | `logarithm_firstorder_Mean` | raw → RobustScaler |
| 28 | `lbp-3D-m2_glcm_Imc2` | log1p(clip>=0) → RobustScaler |
| 29 | `lbp-3D-m1_glszm_HighGrayLevelZoneEmphasis` | log1p(clip>=0) → RobustScaler |
| 30 | `wavelet-HLH_gldm_LargeDependenceLowGrayLevelEmphasis` | raw → RobustScaler |
| 31 | `wavelet-HLH_glcm_Imc1` | raw → RobustScaler |
| 32 | `lbp-3D-m1_glszm_LowGrayLevelZoneEmphasis` | log1p(clip>=0) → RobustScaler |
| 33 | `wavelet-HLH_glcm_Imc2` | raw → RobustScaler |

## Phát hiện quan trọng

1. **Radiology "34 features" thực chất = 32 radiomics + 2 index artifact:**
   - slot 0 = RobustScaler của **rank 1..333 theo thứ tự dòng trong file** (không phải thứ tự node; `u + 167 = rank`, `u/166 = giá trị`, kiểm tra `u` là hoán vị của `-166..166`);
   - slot 1 = RobustScaler của **`lesion_index`** (chính xác `0.25*li − 0.5` với `li ∈ 1..6`).
   - Hai slot này là định danh, không mang thông tin ảnh. Mô tả "34 radiomics features" trong paper/report hiện tại là **không chính xác** về mặt ngữ nghĩa; nên ghi chú khi viết manuscript. Việc loại bỏ chúng sẽ đổi input của GVAE (34→32 chiều) và mọi similarity radiology → cần quyết định riêng, **không** đổi canonical bây giờ (mọi artifact downstream đang khớp với bản 34 chiều).
2. **Ngưỡng similarity = 0.8 strict** cho cả 3 view; tài liệu cũ ghi 0.7 là sai (đã sửa trong `final_results_package.md` và `PUBLICATION_CHECKLIST.md`).
3. **Không cần dữ liệu Windows gốc**: 15/15 pathology và 34/34 radiology tái lập được từ 3 CSV trong repo.
4. **Không có label trong construction**: script chỉ đọc các cột feature; impute median/mode, scaler và edges đều là hàm của feature. Đây là bằng chứng số cho A2/A4 ở mức build-time (lưu ý A2/A3 phát biểu "train-fold only" áp cho scaler *trong* pipeline — phần đó đã có `tests/test_train_fold_only_fitting.py`).
5. `DataFrame.replace`/BLAS: script xử lý gene không dùng `replace` (tránh FutureWarning) và chỉ chặn RuntimeWarning `matmul` giả do Accelerate trên macOS (numpy#22487); không thay đổi số.

## Giới hạn

- Feature lists được **suy ngược bằng khớp số học** (residual ≤ 1e-7) chứ chưa có script gốc để đối chiếu tên; nếu tìm thấy `patient_glcm_features.csv` / parquet gốc trên máy Windows thì nên diff lại danh sách cột để chốt provenance.
- Script verify chạy trên cùng máy (sklearn 1.6.1 vs bản gốc 1.7.2) — kết quả vẫn khớp 0.0; nếu chạy trên máy khác nên chạy lại `--verify`.
- Chưa thêm test tự động vào `tests/` (build ~vài giây); nếu muốn đưa vào CI có thể thêm smoke test.

## Liên hệ checklist A1–A4

- **A1:** đóng — script + verification ở trên.
- **A2/A3 (imputation/scaling):** bước build diễn ra *trước* khi chia fold; median/mode và RobustScaler được fit trên toàn cohort đóng băng nhưng **chỉ đọc cột feature**, không đọc `binary_label`/`y`/`event`. Mọi scaler *trong pipeline* (sau chia fold) đã có test train-fold-only: `tests/test_train_fold_only_fitting.py`. Khi viết Methods cần nêu rõ phạm vi build-time vs in-pipeline.
- **A4 (edges không dùng outcome):** đóng — cả 3 similarity graph là cosine trên ma trận feature; script không tham chiếu cột nhãn; edge set + `edge_attr` khớp canonical 100%.
