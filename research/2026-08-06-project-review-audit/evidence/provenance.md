# Provenance — audit PROJECT_REVIEW 2026-08-06

Mọi khẳng định trong `reports/report.md` đều truy ra từ một trong các mục dưới đây.

## Lệnh đã chạy (thực thi thật, không phỏng đoán)

| Kết quả | Lệnh | Input | Output/checksum | Ngày |
|---|---|---|---|---|
| 49/49 tests pass | `.venv/bin/python -m pytest -q` (2 lần) | tests/ (24 file) | 49 collected, 49 passed, 0 failed | 2026-08-06 |
| verify_fixes.ipynb execute, 0 errors | `.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace verify_fixes.ipynb` (cwd=review_fixes_2026_07) | review_fixes_2026_07/verify_fixes.ipynb | 11 code cells có assert, TOTAL ERRORS: 0 | 2026-08-06 |
| data_validation PASS | `.venv/bin/python review_fixes_2026_07/data_validation.py data_ln_pc_ihc_g.pt` | data_ln_pc_ihc_g.pt | PASS; 247 pat, dim 22, {0:62,1:185}, mask 105/187, 333 lesions | 2026-08-06 |
| data_validation FAIL (đúng kỳ vọng) | `... deprecated/data_247.pt` | deprecated/data_247.pt | FAIL: dim 64 + polarity {0:185,1:62} | 2026-08-06 |
| data_validation FAIL (dim) | `... configs/data_247.pt` | configs/data_247.pt | FAIL: dim 64 (polarity giống canonical {0:62,1:185}) | 2026-08-06 |
| data_validation PASS | `... data/data_247_scaled_ln_pc_ihc_g.pt` | data/data_247_scaled_ln_pc_ihc_g.pt | PASS (dim 22, {0:62,1:185}) — file non-canonical | 2026-08-06 |
| raw_data_audit chạy | `.venv/bin/python review_fixes_2026_07/raw_data_audit.py` | data CSVs + graph | 247/247 clinical rows khớp; 3 OPAQUE items | 2026-08-06 |

## Checksum sha256 (shasum -a 256)

| File | sha256 |
|---|---|
| data_ln_pc_ihc_g.pt | 73e79bbe1a37946ad4626be1efa1a58531148f294235c7e4849adc1d3461fd7b |
| deprecated/data_247.pt | 04733ee5c562c32783238dae0f42a15f0cf2ab0eaaf7bc4e33117bfee0f8fc81 |
| configs/data_247.pt | cbeb8516f552f52591f127392391c4285b695733993d4aeb7f999dd0216cfb4a |
| training_local.ipynb | 20d86d2a5079bfb9da354bd1f592bff12d9049fb4e20a995e81b0da08e0ad37c |
| review_fixes_2026_07/verify_fixes.ipynb | a748cb515002eac3a685998fcd0b86f46ae5bc9a89feb560d8a63c9e91bc3a5a |
| review_fixes_2026_07/data_validation.py | 98dc37ccf51123c4eff042b88784a7b8b0407d72d2b40315a5cd5a3c2ed1c3da |
| run_gvae.py | 490cf83744319f21c745c4d2c45aa1a32b58508fe9e7256a4a59ac8e628db471 |

## Đọc code (codegraph explore + rg, ngày 2026-08-06)

- utils/latent_extraction.py:295 `_validate_ddpm_latent_key`; :319 `allow_non_concat_mu: bool = False`; :353 guard call.
- utils/classification_eval.py:105 `preds = (scores >= threshold).astype(int)`; :209 `build_binary_classification_diagnostics` (threshold_metric_caveat).
- training/train_gvae.py:117 `_compute_latent_quality_metrics` (scaler/probe fit train-only, :136-138); :219 `_checkpoint_sort_key`; :1031 `_auc_` filename; :1222/:1225-1226/:1991 hardcode `5:22`/`arange(0,5)`/`arange(5,22)`; :1950 `kfold_evaluate_gvae_classifier` (pooled-OOF + bootstrap CIs).
- training/train_ddpm.py:24-27 `train_test_split` nội bộ trong `train_single_conditional_ddpm`; callers: deprecated + notebook (rg toàn repo).
- training/latent_ddpm_augmentation.py:279-280 (downstream scaler), :325-326 (DDPM scaler), :858-881 `apply_train_only_pca`.
- outputs/gvae/train_conditional_ddpm_augmentation_runner.py:41,144 `latest_gvae_run_id` theo st_mtime, không warning.
- training_local.ipynb: map 48 cells bằng AST (`ast.parse`); cell 14 dùng `OUTPUT_PATH` không định nghĩa; cell 22 phụ thuộc `synthetic_latents_numpy` (chỉ định nghĩa cell 47); cell 35 load `best_model_fold_1 (2).pth` (không tồn tại); rg `run_conditional_latent_augmentation_pipeline|latent_ddpm_augmentation` trong notebook = 0 hit; 2/48 cell có assert.
- verify_fixes.ipynb: 11/13 code cells chứa assert (xác nhận commit f385721 sửa file này, không phải training_local.ipynb — `git show --stat f385721`).
- git: `git log origin/main..HEAD` = 17 commits ahead; `git status --short` chỉ còn untracked (.codegraph/, .serena/, PUBLICATION_CHECKLIST.md, graphify-out/).

## Không kiểm chứng được (UNAVAILABLE)

- Các con số §11 PROJECT_REVIEW (0.6894/0.6705/0.6805/0.6747...): artifact tương ứng không còn trong repo (outputs/ chỉ có 2 runner + 1 báo cáo legacy) — không tái lập được, không khẳng định đúng/sai.
- Nghĩa positive-class `binary_label=1`: cần owner xác nhận (README ghi rõ là mở).
- Imputation/scaling bên ngoài graph (A2/A3 PUBLICATION_CHECKLIST): không chứng minh được vì thiếu script dựng graph.

## Cập nhật 2026-08-06 — xác nhận của user

- User xác nhận: `binary_label=1` = **non-responder**; data canonical = `/Users/admin/Diff-GVAE/data_ln_pc_ihc_g.pt`.
- Bằng chứng code: `training/latent_ddpm_augmentation.py:132-136` (`sed -n '115,160p'`) — `_classes_for_augmentation`: `responder_only` → `(1,)`, `minority_only` → class có count nhỏ nhất; cùng file :120 danh sách mode hợp lệ.
- Bằng chứng test docs: `cat tests/test_docs_canonical_data.py` — assert `"OWNER TO CONFIRM" in txt` (sẽ fail nếu sửa README mà không sửa test).
- Suy diễn (được đánh dấu rõ, không phải kết quả chạy): với convention 1=non-responder, deprecated/data_247.pt {0:185,1:62} tương ứng 185 responder / 62 non-responder; metric hiện tại lấy class 1 làm positive.
