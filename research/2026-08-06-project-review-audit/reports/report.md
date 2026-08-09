# Audit: PROJECT_REVIEW.md vs hiện trạng repo — 2026-08-06

> **CẬP NHẬT 2026-08-06 (sau xác nhận của user):** `binary_label=1` = **non-responder**; data canonical = `/Users/admin/Diff-GVAE/data_ln_pc_ihc_g.pt` ({0:62, 1:185} → 62 responder, 185 non-responder). Xem mục 8.

## 1. Trả lời (tóm tắt)

Khoảng 2/3 các vấn đề trong PROJECT_REVIEW.md đã được xử lý và có bằng chứng thực thi (49/49 test pass, verify_fixes.ipynb chạy 0 lỗi, data_validation chạy đúng trên 4 file dữ liệu). Phần còn thiếu tập trung vào: (a) script dựng graph từ raw data vẫn chưa có (chỉ có script audit), (b) `configs/config.py` vẫn rỗng và còn file `configs/data_247.pt` chưa bị quarantine, (c) các chi tiết code chưa sửa (clinical indices hardcode `5:22`, checkpoint đặt tên `_auc_`, runner mặc định chọn run mới nhất theo mtime), (d) toàn bộ artifact kết quả (§11) không còn tồn tại trong repo nên các con số không thể tái lập, (e) `training_local.ipynb` chứa pipeline legacy (dùng data_247_scaled, DDPM trên toàn bộ latents, cell lỗi NameError) chứ không phải pipeline duy trì — pipeline duy trì nằm ở `training/latent_ddpm_augmentation.py` + `outputs/gvae/*.py` mà notebook không gọi.

## 2. Ma trận đối chiếu Problems Found (§14)

| # | Vấn đề (PROJECT_REVIEW) | Trạng thái | Bằng chứng |
|---|---|---|---|
| High | Label polarity ngược giữa 2 file .pt | **ĐÃ XỬ LÝ** (quarantine + validation) | `deprecated/data_247.pt` (dim 64, {0:185,1:62}) — data_validation chạy ra `RESULT: FAIL` đúng như mô tả; canonical PASS {0:62,1:185}; README ghi rõ "sole canonical". Còn mở: nghĩa positive-class cần owner xác nhận. |
| High | Không có script dựng graph từ raw | **MỘT PHẦN** | `raw_data_audit.py` chạy được: 247/247 clinical rows khớp graph; 3 mục vẫn OPAQUE (similarity edges, GLCM upstream, attention aggregation). Script dựng graph vẫn KHÔNG tồn tại (A1 còn mở). |
| High | `train_pipeline.py` DDPM-as-classifier | **ĐÃ XỬ LÝ** | Toàn bộ file chuyển sang `deprecated/train_pipeline.py`; `deprecated/README.md` giải thích lý do; test `test_deprecated_layout.py` + verify cell 15 pass. |
| Med | `train_gvae_ddpm_runner.py` opt-in deprecated | **ĐÃ XỬ LÝ** | Đã chuyển `deprecated/train_gvae_ddpm_runner.py`. |
| Med | `train_ddpm_from_gvae_checkpoints_runner.py` | **ĐÃ XỬ LÝ** | Đã chuyển `deprecated/train_ddpm_from_gvae_checkpoints_runner.py`. |
| Med | `latent_extraction.py:326` cho phép latent key ≠ concat_mu | **ĐÃ XỬ LÝ** | `extract_recommended_latents_for_ddpm(..., allow_non_concat_mu: bool = False)` + `_validate_ddpm_latent_key` (utils/latent_extraction.py:295) raise ValueError trừ khi opt-in; `test_latent_key_guard.py` + verify cell 5 pass. |
| Med | `train_pipeline.py:548` `_select_ddpm_latents` | **ĐÃ XỬ LÝ** | File đã deprecated toàn bộ. |
| Med | `run_gvae.py:75` metric cũ | **ĐÃ XỬ LÝ** | `run_gvae.py` mặc định `checkpoint_metric`/`early_stopping_metric = "latent_quality"`, không còn `auc_pr_balanced_accuracy`; `test_run_gvae_defaults.py` + verify cell 7 pass. |
| Med | Clinical indices hardcode 0:5 / 5:22 | **CHƯA XỬ LÝ** | `training/train_gvae.py:1222, 1225-1226, 1991` vẫn `[:, 5:22]` / `arange(0,5)` / `arange(5,22)`. Rủi ro được giảm nhẹ nhờ data_validation, nhưng code vẫn giả định 22 chiều. |
| Med | Best threshold chọn trên cùng split | **MỘT PHẦN** | `build_binary_classification_diagnostics` (utils/classification_eval.py:209) đã ghi `threshold_metric_caveat` (verify cell 9 pass); việc chọn/report trên cùng split vẫn tồn tại — cần protocol báo cáo (B4). |
| Med | Runner mặc định `latest_gvae_run_id` theo mtime | **CHƯA XỬ LÝ** | `outputs/gvae/train_conditional_ddpm_augmentation_runner.py:41,144` vẫn fallback theo `st_mtime`, không cảnh báo. |
| Med | `configs/config.py` rỗng | **CHƯA XỬ LÝ** | File 0 byte (H1 còn mở). |
| Med | `train_ddpm.py:15` tự split bên trong | **CHƯA XỬ LÝ (nhưng đã cô lập)** | `train_single_conditional_ddpm` vẫn `train_test_split` (train_ddpm.py:24-27). Pipeline duy trì (`latent_ddpm_augmentation.py`) KHÔNG dùng hàm này — chỉ deprecated/ và các cell legacy trong notebook dùng. |
| Low | Checkpoint đặt tên `_auc_` | **CHƯA XỬ LÝ** | `training/train_gvae.py:1031` vẫn `f"{metric_name_for_path}_auc_{...}.pt"`. |
| Low | `metrics_at_threshold` dùng `>` thay vì `>=` | **ĐÃ XỬ LÝ** | `utils/classification_eval.py:105` giờ là `(scores >= threshold)`; verify cell 3 pass. |
| Low | CLAUDE.md trỏ tới `training.ipynb` | **CHƯA XỬ LÝ** | CLAUDE.md:19 vẫn ghi `training.ipynb`; repo chỉ có `training_local.ipynb`. |
| Low | README chỉ có title | **ĐÃ XỬ LÝ** | README nay ghi provenance dữ liệu, cảnh báo polarity, lệnh validation; `test_docs_canonical_data.py` pass. |
| Low | Path quá dài (Windows) | **ĐÃ HẾT** | `outputs/conditional_latent_ddpm` không còn tồn tại; `git status` sạch, không warning. |

## 3. Ma trận Recommended Fix Plan (§15)

| Mục | Trạng thái | Bằng chứng |
|---|---|---|
| 1. Quyết định nghĩa positive-class | **MỞ (cần owner)** | README ghi "OWNER TO CONFIRM: binary_label=1 assumed responder". |
| 1. Quarantine data_247 | **ĐÃ LÀM** — còn sót `configs/data_247.pt` | deprecated/ đầy đủ; `configs/data_247.pt` (64 dim clinical, polarity giống canonical — biến thể thứ 3) vẫn nằm tại configs/, chưa quarantine. |
| 1. Script validation dữ liệu | **ĐÃ LÀM** | `review_fixes_2026_07/data_validation.py` — PASS canonical, FAIL cả 2 file 64-dim; `test_data_validation.py` pass. |
| 2. Script dựng graph | **CHƯA LÀM** | Chỉ có `raw_data_audit.py` (audit, không dựng). A1 mở. |
| 2. Scalers/PCA/DDPM train-fold only | **ĐÃ LÀM (đường duy trì)** | `latent_ddpm_augmentation.py:279-280` (downstream scaler), `:325-326` (DDPM scaler), `apply_train_only_pca` `:858-881`; `test_train_fold_only_fitting.py` pass. |
| 3. Loại/đưa legacy vào deprecated | **ĐÃ LÀM** | deprecated/ chứa 3 runner + train_pipeline + data_247; test pass. |
| 3. Giới hạn DDPM input = concat_mu | **ĐÃ LÀM** | Guard opt-in (xem §2). |
| 3. Cập nhật run_gvae.py | **ĐÃ LÀM** | latent_quality mặc định. |
| 4. Gắn nhãn bảng DDPM là downstream | **ĐÃ LÀM** | summary có `ddpm_is_classifier: False` + `ablation_labels`; verify cell 11 pass. |
| 4. Báo cáo GVAE direct riêng | **MỘT PHẦN** | `kfold_evaluate_gvae_classifier` (train_gvae.py:1950) tách pooled-OOF; nhưng chưa có run cuối để kiểm chứng. |
| 4. Ưu tiên AUC/PR-AUC, ghi rõ nguồn threshold | **MỘT PHẦN** | Caveat đã ghi; protocol báo cáo chưa chốt (B4). |
| 4. Best-result artifact kèm ranking criterion | **CHƯA KIỂM CHỨNG** | Không có `best_result*.json` nào trong repo (outputs rỗng) — verify cell 13 ra FOLLOW-UP. |
| 5. Retune filter_quantile (giữ 0 mẫu) | **CHƯA LÀM** | verify cell 25: "Follow-up recorded: retune filter_quantile on a full run". |
| 5. Bootstrap CI | **ĐÃ LÀM (công cụ)** | `bootstrap_ci.py` + `kfold_evaluate_gvae_classifier` pooled-OOF có CIs; verify cell 21 pass (smoke). Chưa áp dụng lên số cuối. |
| 5. Seed sweeps | **ĐÃ LÀM (công cụ)** | `seed_sweep.py` + aggregation; verify cell 23 pass (smoke). Chưa chạy cho số cuối. |

## 4. training_local.ipynb — có phải nơi chứa "các hàm để run"?

**Kết luận: notebook là bản thực nghiệm legacy, KHÔNG phải pipeline duy trì.**

- Đúng: chứa cell chạy được cho `run_gvae_sweep`, `kfold_train_gvae`, `kfold_evaluate_gvae_classifier` (pooled-OOF + bootstrap CI), `sweep_pretrain_recipes`.
- Pipeline duy trì theo PROJECT_REVIEW §7/§12/§16 (`run_conditional_latent_augmentation_pipeline`, `train_conditional_ddpm_on_latents`) **không xuất hiện trong notebook** (rg toàn notebook = 0 hit).
- Cell legacy dùng đường không an toàn: cell 15-20, 29, 34-35, 41-42 gọi `train_single_conditional_ddpm` trên **toàn bộ latents** (self-split nội bộ, rủi ro leakage Medium #12), load `data/data_247_scaled_ln_pc_ihc_g.pt` (không phải file canonical), dùng checkpoint cũ `best_model_fold_7.pth`/`best_model_fold_3.pth`/`best_model_fold_1 (2).pth`, và quy ước polarity `0=Responder` ngược với giả định đang chờ xác nhận của README (`1=responder`).
- Lỗi thực thi khi chạy tuần tự: **cell 14** dùng `OUTPUT_PATH` chưa từng được định nghĩa (NameError — xác nhận bằng AST scan toàn notebook); **cell 22** phụ thuộc `synthetic_latents_numpy` chỉ được định nghĩa ở cell 47; **cell 35** load `best_model_fold_1 (2).pth` không tồn tại.
- Toàn bộ artifact trung gian của các cell 9-47 (`dca_ddpm_input_*.pt`, `synthetic_*.pt/csv`, `ddpm_responder.pth`, `interaction_labels/`, `batch_interaction_results_robust/`, `coverage_density_results/`) **không tồn tại** trong working tree → các cell này không chạy lại được nếu không chạy lại GVAE trước.
- Chỉ 2/48 cell có `assert`; commit f385721 "every notebook section ends with an assertion" thực ra là sửa `verify_fixes.ipynb` (11/13 cell có assert) chứ không phải `training_local.ipynb`.

## 5. Các con số trong PROJECT_REVIEW §11

**KHÔNG TÁI LẬP ĐƯỢC từ hiện trạng repo.** Run `gvae_latent_quality_codex_20260615_204355`, `outputs/conditional_latent_ddpm/`, `outputs/best_gvae_ddpm_result.json` đều không còn tồn tại (outputs/ chỉ còn 2 runner script + 1 báo cáo diagnostic legacy). PUBLICATION_CHECKLIST A6 ghi nhận hai bộ số phân kỳ (notebook ~0.81-0.87 vs PROJECT_REVIEW ~0.69) chưa được hòa giải; chưa chọn run canonical (A7).

## 6. Còn thiếu (tổng hợp, theo thứ tự ưu tiên)

1. **Script raw→graph dựng `data_ln_pc_ihc_g.pt`** (A1, High) — chưa có; similarity edges/GLCM/attention vẫn OPAQUE.
2. **Xác nhận nghĩa positive-class** — ✅ **ĐÃ XÁC NHẬN 2026-08-06: `binary_label=1` = non-responder** (0 = responder, 62/247). Mục này chuyển từ "mở" sang "đã xong" — việc còn lại là cập nhật tài liệu/code đang mang giả định ngược (mục 8).
3. **Chọn run canonical + lưu artifact cuối** — outputs rỗng; A6/A7 mở.
4. **`configs/config.py` vẫn rỗng** (H1); **`configs/data_247.pt` chưa quarantine** (thêm vào deprecated/ hoặc xóa).
5. **Code chưa sửa**: clinical indices `5:22` hardcode (train_gvae.py:1222/1991); checkpoint `_auc_` (train_gvae.py:1031); runner mtime fallback (train_conditional_ddpm_augmentation_runner.py:41).
6. **Tài liệu lệch hiện trạng**: CLAUDE.md trỏ `training.ipynb`; PROJECT_REVIEW §12 lệnh chạy bằng `.venv-win312` + runner (máy hiện tại dùng `.venv`, entry point thực tế là notebook + outputs/gvae runners).
7. **Notebook hygiene**: cell 14 NameError, cell 22 thứ tự, cell 35 file thiếu; các cell legacy dùng data/scaler không canonical — cần đánh dấu rõ hoặc cắt khỏi notebook chính.
8. **Theo dõi còn mở**: retune `filter_quantile` (giữ 0 mẫu), field ranking-criterion trong best_result chưa kiểm chứng.
9. **PUBLICATION_CHECKLIST B-J** (mục tiêu xuất bản): toàn bộ `[ ]` chưa tick — CI/seed áp lên số cuối, held-out test/nested CV, baselines, TSTR controls, external validation, TRIPOD, data dictionary, per-sample outputs.

## 7. Điểm mạnh đã xác nhận

- 49/49 test pass (pytest).
- `verify_fixes.ipynb` chạy end-to-end 0 lỗi, 11 cell có assert pass.
- `data_validation.py` phát hiện đúng polarity/dim trên cả 4 file .pt.
- `raw_data_audit.py` chạy, khớp 1:1 clinical CSV ↔ graph.
- Leakage train-fold-only được kiểm soát ở đường duy trì (scaler DDPM/downstream/PCA đều fit train).

## 8. Hệ quả của xác nhận `binary_label=1` = non-responder (2026-08-06)

### 8.1 Ý nghĩa dữ liệu (đã chốt)

- `data_ln_pc_ihc_g.pt` (canonical, confirmed): {0: 62, 1: 185} → **62 responder (class 0, minority), 185 non-responder (class 1, majority)**.
- `deprecated/data_247.pt` {0: 185, 1: 62} → theo cùng convention: **185 responder / 62 non-responder** — ngược cân bằng với canonical (và khác dim clinical 64) → giữ quarantine, không thể đảo label để dùng lại (vì dim khác).
- `configs/data_247.pt` {0: 62, 1: 185} + dim 64 → polarity GIỐNG canonical nhưng dim clinical sai → vẫn phải quarantine/xóa (chưa làm).
- `data/data_247_scaled_ln_pc_ihc_g.pt` dim 22 + {0:62,1:185} → cùng convention với canonical, nhưng là bản scaled, không được README/tài liệu mô tả; notebook legacy (cell 15-20) dùng file này.

### 8.2 BUG ngữ nghĩa trong code augmentation (mới phát hiện)

`training/latent_ddpm_augmentation.py:134` — `_classes_for_augmentation`:
- `responder_only` → sinh class **1** = **non-responder** (tên gọi ngược nghĩa).
- `minority_only` → sinh class 0 = **responder** (đúng nghĩa).

⇒ Trong PROJECT_REVIEW §8 (dòng "responder_only: generate only class 1") và §11, mọi dòng nhắc "responder" cần sửa/ghi chú theo convention đã xác nhận. Cụ thể §11: "Best DDPM augmentation by balanced accuracy: `minority_only` ratio 2.0" = oversample **responder**; "Best by ROC AUC: `both_classes` ratio 0.5" = sinh cả 2 class. Đề xuất rename mode `responder_only` → `nonresponder_only` (hoặc thêm alias + ghi chú) để khỏi gây nhầm khi báo cáo.

### 8.3 Metric polarity (cần ghi rõ trong PROJECT_REVIEW §11/§16)

- Toàn bộ metric hiện tại (ROC AUC, PR AUC, F1, sensitivity/specificity, threshold, `pos_weight` = n_neg/n_pos với class 1 là "positive") được tính với **class 1 = non-responder làm positive class**.
- ROC AUC đối xứng với label flip (không đổi), nhưng **PR AUC, precision/recall, sensitivity/specificity KHÔNG đối xứng** — nếu paper framing "dự đoán responder" thì phải flip label (`1 - y`) trước khi tính các metric này, hoặc diễn giải lại.
- `pos_weight` trong train_gvae.py: `n_negative / n_positive` với n_positive = class 1 → đang ưu tiên recall cho **non-responder**. Nếu muốn ưu tiên bắt đúng responder (class 0) phải đảo công thức.
- PROJECT_REVIEW §16 cần thêm bước: "chốt positive class cho báo cáo (responder = class 0); flip label nếu cần trước khi tính PR-AUC/threshold metrics".

### 8.4 Tài liệu/code cần sửa ngay do xác nhận này

| File | Hiện tại | Cần sửa |
|---|---|---|
| `README.md:11-15` | "binary_label = 1 is assumed to mean *responder* ... OWNER TO CONFIRM" | 1 = **non-responder** (đã xác nhận 2026-08-06); 0 = responder (62/247, minority). Bỏ cụm "OWNER TO CONFIRM". |
| `tests/test_docs_canonical_data.py` | `assert "OWNER TO CONFIRM" in txt` | Sửa assertion theo README mới (nếu không test sẽ fail). |
| `review_fixes_2026_07/data_validation.py` | in class_counts không kèm nghĩa | (tùy chọn) thêm chú thích "label 1 = non-responder". |
| `PROJECT_REVIEW.md` §8/§11/§14/§15/§16 | giả định 1=responder hoặc không chốt | Cập nhật như mục 8.2/8.3; §15 item 1 đổi từ "cần quyết định" → "đã xác nhận, cập nhật docs". |
| `PUBLICATION_CHECKLIST.md` A8 | "binary_label=1 = responder?" | Trả lời: KHÔNG — 1 = non-responder. |
| `REPORT_TODO.md` §2 | "Ý nghĩa chính xác binary_label chưa mô tả" | Đã có câu trả lời: 0 = responder, 1 = non-responder. |

### 8.5 Còn thiếu (danh sách cập nhật sau xác nhận)

1. Script raw→graph dựng `data_ln_pc_ihc_g.pt` (A1, High) — chưa có; similarity edges/GLCM/attention vẫn OPAQUE.
2. Chọn run canonical + lưu artifact cuối (A6/A7) — outputs rỗng; số §11 không tái lập; hai bộ số phân kỳ chưa hòa giải.
3. `configs/config.py` vẫn rỗng (H1); `configs/data_247.pt` chưa quarantine (dim 64, polarity giống canonical).
4. Code chưa sửa: clinical indices `5:22` hardcode (train_gvae.py:1222/1991); checkpoint `_auc_` (train_gvae.py:1031); runner mtime fallback (train_conditional_ddpm_augmentation_runner.py:41); rename `responder_only` (latent_ddpm_augmentation.py:134).
5. Tài liệu lệch hiện trạng: CLAUDE.md trỏ `training.ipynb`; PROJECT_REVIEW §12 lệnh chạy `.venv-win312`; README + test_docs + PROJECT_REVIEW §8/§11/§14/§15/§16 chưa cập nhật convention mới (mục 8.4).
6. Notebook hygiene: cell 14 NameError (`OUTPUT_PATH`), cell 22 thứ tự, cell 35 file `best_model_fold_1 (2).pth` thiếu; cell legacy dùng `data_247_scaled` + `train_single_conditional_ddpm` trên toàn bộ latents.
7. Theo dõi còn mở: retune `filter_quantile` (giữ 0 mẫu), field ranking-criterion trong best_result chưa kiểm chứng.
8. PUBLICATION_CHECKLIST B-J: CI/seed lên số cuối, held-out/nested CV, baselines, TSTR controls, external validation, TRIPOD, data dictionary, per-sample outputs.
