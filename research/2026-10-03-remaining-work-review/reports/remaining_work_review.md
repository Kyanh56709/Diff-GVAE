# Review công việc còn lại — Diff-GVAE (2026-10-03)

> **Cập nhật 2026-10-03 (chiều):** §2 đã lỗi thời — A1 đã đóng bằng `data/build_ln_pc_ihc_g.py` (khớp bit-exact; pathology **15/15**, radiology 34/34 = 32 radiomics + 2 index artifact). Xem `research/2026-10-03-a1-build-script/reports/a1_build_verification.md` và `research/2026-10-03-radiology-artifact-ablation/`.

**Bối cảnh:** Tiếp nối phiên `ses_001b67fdeffevG5Umu4BnUzzAn` (2026-09-08, "Nghiên cứu các task còn lại") — phiên đó kết thúc ở chỗ: repo không có script raw→graph, user chọn *viết lại script*, rồi cung cấp `data_creation.ipynb` nhưng chưa thực thi gì thêm. Báo cáo này kiểm tra lại toàn bộ bằng chứng hiện trạng trước khi thực thi.

**Phạm vi kiểm tra (bằng chứng chạy trực tiếp trên repo ngày 2026-10-03):**
- Test suite, git history/status, cấu trúc `data_ln_pc_ihc_g.pt`, đối chiếu giá trị graph với 4 CSV thô trong `data/`.
- Rà toàn bộ `PUBLICATION_CHECKLIST.md`, `REPORT_TODO.md`, `PROJECT_REVIEW.md`, `research/2026-08-06-*`.
- Tìm artifact cho từng hạng mục B–J (rg/find) và quét placeholder trong manuscript `EJC_Research_Paper.pdf`.

---

## 1. Kết luận ngắn

1. **Việc còn sót lớn nhất (A1 — script raw→graph) vẫn mở, NHƯNG đã đủ bằng chứng để thực thi và kiểm chứng**: mọi thành phần của `data_ln_pc_ihc_g.pt` gần như khớp 1:1 với các CSV thô trong repo (clinical 22/22, radiology 32/34, pathology 9/15, labels/masks/cohort khớp tuyệt đối). Không cần dữ liệu Windows gốc cho phần lớn các chiều.
2. Chưa có gì được commit sau 2026-08-10 (`4d3be44`); các artifact canonical + pooled-OOF + DDPM diagnosis vẫn còn nguyên trong `outputs/`.
3. Toàn bộ B–J của checklist xuất bản vẫn mở/đang dở; chi tiết ở §3.

---

## 2. Task trọng tâm — tái tạo raw→graph (`data_ln_pc_ihc_g.pt`)

### 2.1 Trạng thái

| Hạng mục | Kết quả kiểm tra |
|---|---|
| Script build trong repo | **Không có** — `utils/preprocessing.py` = 0 byte (kể từ initial commit); không có script nào tạo `similar_to_*`; chỉ có `data_creation.ipynb` (git-ignored, của user, 9 cell = 5 phiên bản build) |
| `data_creation.ipynb` | Chạy trên máy Windows (`C:\Users\lekya\...`); nguồn: `clinical_features_tmb.csv`, `glcm_features.csv`, `radiology_node_features.parquet`; có 5 biến thể output (`multi_view_pdl1_data_lesions_366_thresh_08_robust.pt`, `..._with_genomics.pt`, `..._genomics.pt`, `..._247.pt`, `scaled_..._247_log1p.pt`). Các cell dùng threshold 0.7 (riêng bản genomics dùng 0.8), có fallback tạo file dummy rỗng → **chưa port được nguyên trạng** |
| Canonical data | 247 patients / 333 lesions / x_clinical (247,22) / x_pathology (247,15) / lesion x (333,34); masks 105 (pathology) & 187 (radiology); edges: 1152 clinical (576 cặp), 1016 pathology (508 cặp), 652 radiology (326 cặp), 333 has_lesion |

### 2.2 Bằng chứng tái tạo được từ repo (chạy trực tiếp, 2026-10-03)

| Thành phần canonical | Nguồn trong repo | Độ khớp |
|---|---|---|
| Cohort 247 | `clinical_features_tmb.csv`: `cohort=='discovery'` (246) + 1 dòng `cohort=NaN` | Khớp 247/247 |
| Labels | `label`→`binary_label`, `pfs`→`y`, `pfs_censor`→`event` (cùng CSV) | Khớp tuyệt đối |
| Clinical 22 chiều | `x_clinical_unscaled_ln_pc_ihc_g.csv` + `clinical_scaler.pkl` (log1p 5 cột → RobustScaler) | **22/22 cột, maxdiff ≈ 5e-8 – 2e-7** |
| Gene one-hot 6 cột | 9 gene trong notebook; `ALK/ROS1/RET` toàn `False` trong cohort 247 → bị loại | Giải thích được vì sao còn 6 |
| Pathology 15 chiều | `glcm_features.csv` (157 BN × 137 feature) | **9/15 chiều khớp chính xác** dạng `log1p(cột) → affine` (residual ≤ 1e-7): ClusterProminence_kurtosis/variance, ClusterShade_variance, JointAverage_skewness, SumAverage_skewness, Imc2_variance, DifferenceVariance_skewness, Autocorrelation_lognorm_fit_p2, Contrast_kurtosis |
| Radiology lesion 34 chiều | `radiology_features.csv` (431 lesion × 1670 feature) | **32/34 chiều khớp chính xác** (raw hoặc log1p → affine, residual ≤ 1e-6) — kiểm định trên 89 bệnh nhân 1-lesion (không mơ hồ thứ tự) |
| Mask pathology = 105 | = số BN có GLCM trong cohort | Khớp |
| Mask radiology = 187; 333 lesion | = BN có ≥1 lesion trong cohort; phân bố lesion/BN {1:89, 2:60, 3:28, 4:10} | Khớp tuyệt đối |
| Ngưỡng similarity | min cosine trên cạnh thật: clinical **0.8000282**, pathology **0.8000482** | **Ngưỡng thật = 0.8**, không phải 0.7 như `final_results_package.md` ghi |

### 2.3 Phần chưa khớp / còn thiếu để tái tạo chính xác 100%

1. **6/15 chiều pathology** (index 0, 2, 3, 4, 6, 12) — không có cột nào trong `glcm_features.csv` khớp với các họ biến đổi raw/log1p/signed-log/sqrt (best R² chỉ 0.44–0.98). Nhiều khả năng đến từ extraction khác (kênh deconvoluted PD-L1 theo Vanguri) hoặc cột bị biến đổi khác.
2. **2/34 chiều radiology** (index 0, 1) — chưa match.
3. **Đặc trưng similarity của radiology** (aggregated mean/std/min/max) không được lưu trong graph → muốn xác minh 326 cặp cạnh radiology phải tái dựng lại rồi so khớp tập cạnh.
4. **Mâu thuẫn ngưỡng**: tài liệu/notebook ghi 0.7, dữ liệu thật là 0.8 → khi viết script phải chốt 0.8 và sửa docs.
5. `x_clinical_columns` trong graph không kèm danh sách tên 15/34 feature → phải chốt danh sách qua bước matching (đã có 9 + 32 ứng viên).

### 2.4 Kế hoạch thực thi đề xuất cho task data creation

1. Viết script portable `data/build_ln_pc_ihc_g.py` (log = notebook cell 5/8, bỏ Windows path, bỏ dummy fallback): clinical từ CSV + scaler pkl; pathology/radiology từ CSV với danh sách cột đã match; nhãn từ `label/pfs/pfs_censor`.
2. Chạy, đối chiếu từng trường với `data_ln_pc_ihc_g.pt` (dims, giá trị, masks, edge sets, nhãn) và xuất bảng delta.
3. Xử 6+2 chiều chưa khớp: grid-search toàn bộ cột với họ biến đổi mở rộng; nếu bế tắc → hỏi user file gốc trên máy Windows (parquet + glcm gốc) để đối chiếu.
4. Sau khi khớp 100%: cập nhật docs (threshold 0.8, feature list), đóng A1–A4, commit script + verification log.

---

## 3. Checklist A–J — trạng thái đã kiểm chứng

### A. Internal validity
| Mục | Trạng thái | Bằng chứng |
|---|---|---|
| A1 script raw→graph | **MỞ** (khả thi, xem §2) | `utils/preprocessing.py` 0 byte; không có trong git history |
| A2/A3 imputation/scaling | MỞ (blocked A1); scaler *trong pipeline* đã chứng minh train-fold-only | `tests/test_train_fold_only_fitting.py` pass |
| A4 similarity edges không dùng label | MỞ (blocked A1) | — |
| A5 message-passing split | Code có, tài liệu hóa Methods chưa | `train_gvae.py` (split train→train-subgraph) |
| A6/A7 canonical run + hòa giải 2 bộ số | **XONG về cơ bản** | `outputs/gvae/{metrics,checkpoints}` có 3 run frozen; `final_results_package.md` §4 giải thích chênh protocol (0.6894→0.6065) |
| A8 quarantine `configs/data_247.pt` | **MỞ** | File vẫn tồn tại (2.5 MB); `data/data_247.pt` đã chuyển `deprecated/` |

### B. Statistical rigor
| Mục | Trạng thái | Bằng chứng |
|---|---|---|
| B1 bootstrap CI | **XONG** | `oof_metrics_with_ci.json`, `oof_arrays.npz` (247), `oof_roc_pr.png` |
| B2 held-out/nested CV | **MỘT PHẦN** | Protocol pooled-OOF đã tách selection/evaluation: `kfold_evaluate_gvae_classifier` (train_gvae.py:2034–2058) — "outer folds report-only; inner validation split drives early stopping + checkpoint selection"; chạy qua `pipeline_phase4a_oof.sh`. Còn thiếu: nested CV cho chọn hyperparameter + locked held-out test độc lập |
| B3 multi-seed ≥5 | **MỞ** (tool có, chưa chạy số cuối) | `review_fixes_2026_07/seed_sweep.py`; không có output |
| B4 threshold protocol | MỘT PHẦN | Caveat đã ghi trong `classification_eval.py`; protocol chưa chốt |
| B5 significance tests | **MỞ** | Không có DeLong/paired test |
| B6 AUPRC vào manuscript | MỞ | Số có (PR-AUC 0.8106 OOF), draft chưa cập nhật |

### C. Core experiments
| Mục | Trạng thái | Bằng chứng |
|---|---|---|
| C1 ablations + CI | **MỞ** (không có artifact ablation) | `find outputs research -iname "*ablation*"` → 0 |
| C2 baselines (XGB/SVM/DyAM/late-fusion) | **MỞ** | Không có script |
| C3 hyperparam sensitivity | MỘT PHẦN | `gvae_sweep_results_v2.csv` (10/216 combos, 3-fold) |
| C4 N=366 robustness | **MỞ** | Chỉ nhắc trong diagnosis doc |
| C5 pooling ablation | **MỞ** | Không có |

### D. External validation
| Mục | Trạng thái | Bằng chứng |
|---|---|---|
| D1 tìm dataset | **XONG** (science-reviewer WARN→fixed) | `research/2026-08-06-external-datasets/reports/report.md` (I3LUNG Zenodo 17535424 S-tier) |
| D2/D3 I3LUNG adapter + external check | **MỞ** | Không có code `.py` nào cho I3LUNG |
| D4 TCIA (stretch) | MỞ | — |
| D5/D6 write-up + domain shift | MỞ | — |

### E. Generative claim
| Mục | Trạng thái | Bằng chứng |
|---|---|---|
| E1 controls SMOTE/jitter | **MỞ** | Không có code |
| E2 TSTR held-out | MỞ (phụ thuộc B2) | A4 chỉ trên val folds |
| E3 MMD + permutation p/KS/Frobenius | MỘT PHẦN | MMD có (0.032 @600ep); p-value/KS/Frobenius chưa |
| E4 diversity/near-copy | MỘT PHẦN | `near_dup=0.0` ✅; precision-recall generative chưa |
| E5 centroid-shift cosine | **MỞ** | Số vẫn chưa có |
| E6 reframe hypothesis | MỞ | — |

### F. Interpretability
- **F1 (SHAP/IG)**: chưa có. **F2 (lesion attention)**: chưa có.
- **F3 (UMAP/t-SNE)**: code **có** (TSNE/UMAP/PCA trong `training/latent_ddpm_augmentation.py:812–837`) và artifact runtime có (e.g. `outputs/conditional_latent_ddpm/.../projections/tsne_projection.png`); chưa có figure publication-quality "colored by response + driver-gene".

### G. Reporting standards
G1 TRIPOD+AI, G2 CLAIM: **chưa có**. G3 data-availability: README có provenance, chưa có statement chính thức + repo link. G4 ethics: chưa. G5 compute disclosure: chưa.

### H. Reproducibility & code
| Mục | Trạng thái | Bằng chứng |
|---|---|---|
| H1 pinned env + config | **MỞ một phần** | `requirements.txt` không pin version; `configs/config.py` = 0 byte |
| H2 one-command reproduction | **MỞ** | Chỉ có pipeline scripts rời trong research/ |
| H3 deprecated quarantine | **XONG** | `deprecated/` + `test_deprecated_layout.py` |
| H4 data dictionary | **XONG** | `research/2026-08-06-data-dictionary/` (build script + 3 CSV + summary) |
| H5 per-sample outputs | **XONG (GVAE OOF)** | `oof_arrays.npz`: y_true/head_probs/probe_probs (247,) + `oof_per_fold.csv` |
| H6 test suite | **XONG** | `72 passed` (15.3s, chạy lại 2026-10-03) |

### I. Manuscript (`EJC_Research_Paper.pdf`, 31 trang)
- **Placeholder còn nguyên**: "(Chua Sua)" ×1 (cuối Conclusion) + "Chua sua" ×1 ("2. Related Works") — case-insensitive ×2; "(Check lai sau)" ×1 (pathology 15 features); "... dien sau" ×1 (§3.3.7 Stage-2 hyperparameters); "Ket qua cua em Loan, se check lai sau" ×1 (§4.1.4).
- **AUC chưa hòa giải**: "0.806" ×4 và "0.867" ×4 cùng tồn tại.
- Figures/cross-refs/author list/limitations: chưa kiểm tra sâu, chưa thấy bằng chứng đã xử lý.

### J. Submission logistics
Chưa có cover letter, chọn journal, preprint — chưa làm.

### REPORT_TODO (báo cáo tiếng Việt `FINAL_REPORT.md`)
- Admin (tên SV, MSSV, GVHD, ngày…) **vẫn "cần bổ sung"** (7 mục).
- Các mục "cần bổ sung" khác đã lỗi thời (label polarity đã chốt 2026-08-06; data dictionary đã có) nhưng report chưa cập nhật.
- Số liệu chính vẫn là bộ cũ 0.6894 (chưa thay bằng số OOF 0.6065).

---

## 4. Housekeeping phát hiện thêm

1. `research/2026-08-06-project-review-audit/reports/final_results_package.md` **untracked** và lỗi thời: mục "Còn mở" #1 (filter_quantile) và #3 (PCA branch) đã xong sau đó (`62fbcfa`, `b137505`, `c7e54bf`); đồng thời ghi "cosine ≥ 0.7" trong khi dữ liệu thật là 0.8 → cần cập nhật + commit.
2. `CLAUDE.md:19` vẫn trỏ `training.ipynb` (repo chỉ có `training_local.ipynb`).
3. `data/data_247_scaled_ln_pc_ihc_g.pt` còn trong `data/` (chỉ notebook legacy dùng; không script duy trì nào tham chiếu) — nên quyết định quarantine nốt.
4. `configs/data_247.pt` (A8) + `configs/config.py` rỗng (H1) — gộp xử lý một lượt.
5. `.kilo/worktrees/rose-pitcher` (không thuộc git tracked) là worktree cũ ở detached commit `0392718`; nên dọn nếu không dùng (chỉ là housekeeping local).

---

## 5. Critical path đề xuất (cập nhật)

1. **A1 — viết + chạy script tái tạo raw→graph** (đã đủ bằng chứng; xem §2.4) → đóng A2–A4; sửa docs ngưỡng 0.8.
2. **A8 + H1 housekeeping** (quarantine `configs/data_247.pt`, pin requirements, điền `configs/config.py`).
3. **B2 + B3** (nested CV cho chọn hyperparameter + multi-seed; protocol OOF đã có, tool seed sweep đã có, CPU ~13 phút/run).
4. **E1** (SMOTE/jitter controls) — quyết định "denoising effect" là headline hay footnote.
5. **D2/D3** (I3LUNG adapter — bước mapping đã được report external vạch sẵn).
6. **C1/C2** (ablations + baselines) nếu đủ compute.
7. **I1–I9 + G1–G5 + REPORT_TODO** (manuscript + reporting) sau khi số liệu chốt.

---

## 6. Limitations của review này

- Các kết luận match dữ liệu (9/15, 32/34) kiểm trên **tập con xác định được thứ tự** (105 BN pathology; 89 BN 1-lesion radiology) — chưa chứng minh đầy đủ cho BN nhiều lesion (cần bước assignment khi thực thi A1).
- Chưa xác minh đặc trưng similarity radiology (không được lưu trong graph) → chưa thể tuyên bố khớp 100% cạnh radiology.
- Chưa kiểm tra nội dung `EJC_Research_Paper.pdf` ngoài các placeholder đã grep; checklist I5–I9 cần soát riêng khi làm manuscript.

## 7. Xác minh độc lập (claim-verifier, 2026-10-03)

Một subagent độc lập đã chạy lại toàn bộ 10 claim chính: **8 verified, 0 refuted, 2 partial** → verdict: **PARTIAL**.

- Xác nhận nguyên vẹn: không có build script tracked (preprocessing.py = empty blob từ initial commit); clinical 22/22 khớp (maxdiff ≤ 2.2e-7, 17 cột còn lại bằng raw chính xác); 5 mapping pathology nêu trong report khớp R²=1 (residual ≤ 9.3e-8) và 6 dims kia best R² < 0.999; 32/34 radiology dims ≤ 1e-4; ngưỡng cạnh chính xác 0.8000282051 (clinical) / 0.8000481282 (pathology); 72 tests pass; configs/data_247.pt 2,529,980 byte; OOF npz + 3 run dirs + DDPM artifacts đủ.
- **Hiệu chỉnh 1 (B2/F3 trong report này):** code held-out-OOF **có** (`train_gvae.py:2034`, outer folds report-only + inner-val selection) và code UMAP/t-SNE **có** (`latent_ddpm_augmentation.py:812–837`, kèm artifact PNG/CSV trong DDPM runs) — mục B2 và F3 đã được cập nhật tương ứng; phần thực sự thiếu là nested CV cho hyperparameter + figure publication-quality (F3) + SHAP/IG & attention (F1/F2).
- **Hiệu chỉnh 2 (I):** "(Chua Sua)" là 1 exact + 1 "Chua sua" (case-insensitive 2), không phải 2 exact — nội dung không đổi.
- Claim-verifier cũng xác nhận `data_creation.ipynb` là artifact build duy nhất tìm thấy (git-ignored), khớp kết luận §2.
