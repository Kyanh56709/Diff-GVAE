# C2 baselines + B5 DeLong — canonical r32 (2026-10-04)

**Câu hỏi:** GVAE (pooled-OOF, canonical r32) có tốt hơn các baseline đơn giản không dùng graph, khi chạy trên **cùng split, cùng seed** và kiểm định bằng DeLong?

**Trả lời ngắn:** Không. GVAE head không vượt baseline nào. Trung bình 5 seed, logistic regression **chỉ dùng clinical** đạt ROC-AUC cao nhất (0.707 so với 0.664 của GVAE head). Tuy nhiên sau hiệu chỉnh Holm thì không chênh lệch nào giữa GVAE head và baseline có ý nghĩa thống kê (p_holm nhỏ nhất 0.157). Với N = 247, khoảng tin cậy 95% của ΔAUC rộng khoảng ±0.065–0.10, nên thử nghiệm này chỉ phát hiện được chênh lệch cỡ 0.07–0.10 trở lên.

## 1. Protocol

- Graph: `data_ln_pc_ihc_g_r32.pt` (247 BN; positive = `binary_label` 1 = non-responder, 185/247).
- Split giống hệt `kfold_evaluate_gvae_classifier` (`training/train_gvae.py:2034`):
  - outer: `StratifiedKFold(5, shuffle=True, random_state=seed)`, seed 42–46;
  - inner: `train_test_split(outer_train, test_size=0.2, stratify, random_state=4200)`.
- Baseline chọn hyperparameter theo ROC-AUC trên inner-val (fit trên inner-train), rồi chấm điểm outer-test. Hai biến thể refit:
  - `outer_train`: refit trên toàn bộ outer-train (chuẩn thông thường; baseline thấy nhiều hơn GVAE khoảng 25% dữ liệu);
  - `inner_train`: chỉ fit trên inner-train (cân bằng dữ liệu với GVAE, vốn chỉ train trên inner-train).
- Scaler nằm trong `sklearn.Pipeline`, nên chỉ fit trên train.
- Số GVAE lấy nguyên từ các run pooled-OOF đã chốt `research/2026-10-03-radiology-artifact-ablation/output/drop_both32_seed{42..46}/` (A7). Đã kiểm tra thứ tự nhãn trong `oof_arrays.npz` trùng `binary_label` của graph ở cả 5 seed.
- Metric: pooled-OOF ROC-AUC và PR-AUC, bootstrap 2000 lần (seed 4200).
- Kiểm định: DeLong paired hai phía (`review_fixes_2026_07/delong.py`, fast DeLong của Sun & Xu 2014, có test đối chiếu với cài đặt O(n²) trong `tests/test_delong.py`). Holm hiệu chỉnh trên 5 baseline trong từng seed. Thêm một test trên điểm trung bình 5 seed của mỗi mô hình (`seed_mean`).

### Feature cấp bệnh nhân (không dùng graph)

| Block | Chiều | Nội dung |
|---|---|---|
| clinical | 22 | `x_clinical` |
| pathology | 16 | `x_pathology` (0 khi thiếu) + cờ có pathology |
| radiology | 65 | mean và max của `lesion.x` theo lesion của BN (2 × 32, 0 khi không có lesion) + cờ có lesion |
| all | 103 | ghép ba block (đầu vào late fusion) |

### Mô hình và grid

| Tên | Feature | Mô hình | Grid |
|---|---|---|---|
| `logreg_clinical` | clinical | LogisticRegression L2, class_weight balanced | C ∈ {0.01, 0.1, 1, 10} |
| `logreg_concat` | all | như trên | C ∈ {0.01, 0.1, 1, 10} |
| `svm_rbf_concat` | all | SVC RBF, gamma scale, balanced; điểm = decision_function | C ∈ {0.1, 1, 10} |
| `gbdt_concat` | all | HistGradientBoosting, 200 iter, balanced | max_depth ∈ {2, 3} × lr ∈ {0.05, 0.1} |
| `mlp_late_fusion` | all | MLPClassifier, 1000 iter | hidden ∈ {(32), (64, 32)} × alpha ∈ {1e-3, 1e-2} |

**Thay thế so với checklist C2:** venv không có xgboost, nên mình dùng `HistGradientBoostingClassifier` của sklearn (cùng họ gradient-boosted trees) để không phải thêm dependency vào bộ requirements đã pin. DyAM **chưa** được cài đặt lại.

## 2. Kết quả

### 2a. ROC-AUC theo seed (pooled-OOF)

| Mô hình | 42 | 43 | 44 | 45 | 46 | mean ± sd | PR-AUC mean |
|---|---|---|---|---|---|---|---|
| GVAE head | 0.6431 | 0.6678 | 0.6533 | 0.6681 | 0.6852 | 0.6635 ± 0.0161 | 0.8297 |
| GVAE probe | 0.6495 | 0.6119 | 0.6670 | 0.6370 | 0.6125 | 0.6356 ± 0.0238 | 0.8084 |
| *refit = outer_train* | | | | | | | |
| logreg_clinical | 0.7190 | 0.7255 | 0.6869 | 0.7058 | 0.6973 | **0.7069 ± 0.0157** | 0.8626 |
| logreg_concat | 0.6727 | 0.7465 | 0.6526 | 0.6971 | 0.6330 | 0.6804 ± 0.0440 | 0.8471 |
| svm_rbf_concat | 0.7252 | 0.7379 | 0.6231 | 0.6689 | 0.6808 | 0.6872 ± 0.0461 | 0.8527 |
| gbdt_concat | 0.6906 | 0.7050 | 0.6862 | 0.6847 | 0.6345 | 0.6802 ± 0.0268 | 0.8562 |
| mlp_late_fusion | 0.7058 | 0.7224 | 0.7055 | 0.6829 | 0.6435 | 0.6920 ± 0.0305 | 0.8704 |
| *refit = inner_train* | | | | | | | |
| logreg_clinical | 0.7133 | 0.7056 | 0.6755 | 0.7064 | 0.6866 | **0.6975 ± 0.0158** | 0.8593 |
| logreg_concat | 0.7239 | 0.7346 | 0.6248 | 0.6979 | 0.6254 | 0.6813 ± 0.0530 | 0.8430 |
| svm_rbf_concat | 0.6861 | 0.6979 | 0.6035 | 0.6346 | 0.6393 | 0.6523 ± 0.0390 | 0.8332 |
| gbdt_concat | 0.6828 | 0.6690 | 0.6745 | 0.6426 | 0.5905 | 0.6519 ± 0.0375 | 0.8373 |
| mlp_late_fusion | 0.6743 | 0.7004 | 0.7119 | 0.6659 | 0.6479 | 0.6801 ± 0.0260 | 0.8636 |

### 2b. DeLong: GVAE head so với từng baseline

ΔAUC = GVAE head − baseline. "Số seed head tốt hơn" và "số seed p < 0.05" tính trên 5 seed (p chưa hiệu chỉnh). Cột seed_mean so sánh điểm trung bình 5 seed (AUC của GVAE head khi lấy trung bình điểm 5 seed: 0.6889).

| Refit | Baseline | ΔAUC trung bình (5 seed) | Số seed head tốt hơn | Số seed p < 0.05 | seed_mean ΔAUC [95% CI] | seed_mean p | p_holm |
|---|---|---|---|---|---|---|---|
| outer_train | logreg_clinical | −0.043 | 0/5 | 0 | −0.025 [−0.104, 0.054] | 0.53 | 1.0 |
| outer_train | logreg_concat | −0.017 | 2/5 | 1 | −0.011 [−0.087, 0.065] | 0.78 | 1.0 |
| outer_train | svm_rbf_concat | −0.024 | 2/5 | 1 | −0.018 [−0.083, 0.047] | 0.59 | 1.0 |
| outer_train | gbdt_concat | −0.017 | 1/5 | 0 | −0.013 [−0.086, 0.061] | 0.73 | 1.0 |
| outer_train | mlp_late_fusion | −0.029 | 1/5 | 0 | −0.028 [−0.107, 0.051] | 0.49 | 1.0 |
| inner_train | logreg_clinical | −0.034 | 0/5 | 0 | −0.021 [−0.098, 0.055] | 0.58 | 1.0 |
| inner_train | logreg_concat | −0.018 | 2/5 | 0 | −0.013 [−0.082, 0.056] | 0.72 | 1.0 |
| inner_train | svm_rbf_concat | +0.011 | 3/5 | 0 | −0.005 [−0.073, 0.064] | 0.89 | 1.0 |
| inner_train | gbdt_concat | +0.012 | 2/5 | 1 | +0.009 [−0.063, 0.082] | 0.80 | 1.0 |
| inner_train | mlp_late_fusion | −0.017 | 2/5 | 0 | −0.012 [−0.084, 0.060] | 0.75 | 1.0 |

Trên cả 50 test từng seed của GVAE head (2 biến thể × 5 seed × 5 baseline): p chưa hiệu chỉnh nhỏ nhất là 0.032; p_holm nhỏ nhất là 0.157 (outer_train) và 0.161 (inner_train).

### 2c. GVAE probe (linear probe trên `mu`)

Probe thấp hơn mọi baseline ở seed_mean: ΔAUC từ −0.04 đến −0.06 (p_holm ≥ 0.48) với `outer_train`, và từ −0.02 đến −0.05 (p_holm ≥ 0.95) với `inner_train`. Các test từng seed có p_holm < 0.05 đều rơi vào seed 43, nơi probe đạt 0.612: 5/5 baseline ở `outer_train`, và 1/5 (`logreg_concat`, Δ −0.123, p_holm 0.006) ở `inner_train`.

## 3. Diễn giải

1. **Không có bằng chứng GVAE tốt hơn baseline đơn giản.** Head thấp hơn logistic regression clinical-only ở cả 5/5 seed và cả hai biến thể refit, dù không có ý nghĩa thống kê. Mô hình late-fusion đơn giản (MLP, LR, SVM trên 103 feature ghép) cũng ngang hoặc cao hơn GVAE.
2. **Fusion chưa thêm được giá trị so với clinical.** Thêm pathology và radiology vào logistic regression không cải thiện (0.680 so với 0.707 với outer_train), và phương sai giữa các seed tăng (sd 0.044 so với 0.016).
3. **Kiểm định kém lực.** CI 95% của ΔAUC rộng khoảng ±0.065–0.10. Kết luận đúng là "không phân biệt được", **không phải** "tương đương". Muốn khẳng định chênh lệch 0.03–0.04 thì cần cohort lớn hơn nhiều, hoặc external validation (D2/D3).
4. **Hệ quả cho manuscript (I7):** không được claim GVAE vượt baseline. Nên báo cáo trung thực rằng một baseline clinical-only đạt hiệu năng tương đương hoặc cao hơn, rồi chuyển trọng tâm sang những gì GVAE thêm được ngoài AUC (biểu diễn latent, xử lý modality thiếu), kèm bằng chứng tương ứng.
5. Lấy trung bình điểm của 5 seed làm GVAE head tăng từ 0.664 lên 0.689 (seed ensemble). Nhưng baseline cũng tăng tương tự (logreg_clinical lên 0.714), nên không đổi kết luận.

## 4. Giới hạn của phân tích

- GVAE dùng config cố định (`pipeline_phase4a_oof.sh`), còn baseline có grid nhỏ chọn trên inner-val. Cả hai chỉ dùng inner-val để chọn, nên không có leakage, nhưng mức tuning khác nhau.
- Pooled-OOF gộp điểm của 5 mô hình theo fold, mỗi mô hình có thang điểm khác nhau. Điều này ảnh hưởng mọi mô hình như nhau, kể cả GVAE. Riêng SVM dùng decision_function, nên nhạy với chuyện này hơn.
- Chưa có DyAM và chưa có xgboost (đã thay bằng HistGradientBoosting).
- Trên máy này, numpy 2.0.2 + Accelerate bắn warning giả `overflow/invalid value encountered in matmul`. Đã kiểm chứng: feature toàn hữu hạn (max |x| = 59), và `X @ w` khớp `einsum` tới 1e-14. Vì vậy run chính dùng `-W ignore::RuntimeWarning`. Script vẫn assert không có NaN trong điểm OOF.
- Hai lần chạy `outer_train` cho kết quả giống hệt nhau (deterministic).

## 5. Tái lập

```bash
.venv/bin/python -W ignore::RuntimeWarning research/2026-10-04-baselines-delong/scripts/run_baselines.py --refit outer_train
.venv/bin/python -W ignore::RuntimeWarning research/2026-10-04-baselines-delong/scripts/run_baselines.py --refit inner_train
```

Output được ghi vào `research/2026-10-04-baselines-delong/output/{outer_train,inner_train}/`: `metrics_per_seed.csv`, `delong_tests.csv`, `summary.csv`, `chosen_params.json`, cùng `<model>_seed<s>/oof_arrays.npz`. Các file này không vào git (theo `.gitignore` của repo); số liệu đầy đủ nằm trong báo cáo này. Mỗi biến thể chạy khoảng 5–6 phút trên CPU. Test: `tests/test_delong.py`, `tests/test_run_baselines.py`.
