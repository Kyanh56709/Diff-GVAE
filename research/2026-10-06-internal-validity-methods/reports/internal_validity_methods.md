# A2/A3/A5 + B4/B6 — Nội bộ hợp lệ (Methods) và protocol thống kê

> **Ngày:** 2026-10-06 · **Người làm:** owner-side agent · **Trạng thái:** đóng A2, A3, A5, B4, B6.
> **Mục đích:** gom bằng chứng code + đoạn Methods paste được cho owner, cho các mục nội-bộ-hợp-lệ còn lại của nhóm A và protocol B4/B6.

---

## 1. A2 + A3 — Imputation và scaling fit trên train fold

### Bằng chứng (đã kiểm chứng trên code hiện tại)

| Bước | Fit trên | Vị trí |
|---|---|---|
| Imputation + scaler **lúc build** (clinical/pathology/radiology) | Toàn cohort cố định, **chỉ feature** (không đọc nhãn) | `data/build_ln_pc_ihc_g.py`: clinical `:184-190` (median→clip≥0→log1p→RobustScaler), pathology `:246-251`, radiology `:296-310` |
| PCA per-fold (clinical/pathology) | **Train indices của fold** | `utils/data_utils.py:162` (`pca.fit(original_features[train_indices_np])`) |
| PCA per-fold (lesion) | **Lesion của bệnh nhân train** | `utils/data_utils.py:187` |
| Downstream scaler | **Train split** | `tests/test_train_fold_only_fitting.py` (spy + test chống-leak) |
| Probe scaler trong pooled-OOF | **Inner-train** | `training/train_gvae.py:2265` (`StandardScaler().fit(Z_tr)`) |

Build-time imputation/scaler là một phần của dataset đóng băng (feature-only); **mọi** fitter trong pipeline (PCA, probe scaler, downstream scaler) chỉ chạm train/inner-train. Không có đường nào fit trên val/test.

### Đoạn Methods (đề xuất, tiếng Anh)

> Preprocessing never uses outcome information. Raw clinical, pathology and radiology features were median-imputed (with one-hot/categorical handled explicitly) and standardized with `RobustScaler` once, over the frozen study cohort, when the canonical patient graph was built (`data/build_ln_pc_ihc_g.py`); that script reads feature columns only and never touches `binary_label`/`y`/`event`. Every transformation that could otherwise leak is fit on training data only: per-fold PCA for the clinical/pathology patient features and for lesion features is fit on that fold's training indices (`utils/data_utils.py::preprocess_fold_data_with_pca`), and the downstream/probe `StandardScaler`s are fit on the training (or inner-training) split only (`tests/test_train_fold_only_fitting.py`; `training/train_gvae.py:2265`).

---

## 2. A5 — Message-passing split giữ nguyên trong mọi đường báo cáo

### Bằng chứng

`get_view_subgraph_and_features` (`utils/data_utils.py:8`) giữ **chỉ** các cạnh similarity có **cả hai đầu** nằm trong tập index truyền vào:

```python
mask_src_in_subset = torch.isin(src_nodes_global, global_indices_of_subset_in_batch)
mask_dst_in_subset = torch.isin(dst_nodes_global, global_indices_of_subset_in_batch)
edge_selection_mask = mask_src_in_subset & mask_dst_in_subset   # data_utils.py:82-84
```

Hàm này được gọi ở **mọi** đường:

| Đường | Call site | Hệ quả |
|---|---|---|
| Forward train/val | `GVAE.forward` — `models/gvae_model.py:109` | train→train-subgraph, val→val-subgraph |
| Encoder-only (latent/probe) | `GVAE.get_mus_only` — `models/gvae_model.py:244` | subgraph của đúng tập index |
| Pooled-OOF | `kfold_evaluate_gvae_classifier` gọi `model(fold_data, test_idx)` và `get_separate_view_mus(..., inner_tr/inner_val/test_idx)` | test tách khỏi train/val; ngưỡng chỉ từ inner-val |
| Latent extraction cho DDPM | `utils/latent_extraction.py::_extract_split_latents` gọi `get_separate_view_latent_params(model, full_data, indices)` cho **từng split** | latent train/val/test tách subgraph; DDPM chỉ dùng `concat_mu` của train |

Có test bao phủ đúng logic lọc/remap: `tests/test_subgraph_remap.py` (so với tham chiếu, `mask_s & mask_d`).

**Kết luận:** không có thông tin nào đi qua cạnh nối giữa các split. A5 giữ nguyên trong training, validation, pooled-OOF, latent extraction và DDPM.

### Đoạn Methods (đề xuất)

> Message passing is confined to the split under evaluation. For any set of patient indices, the similarity subgraph retains only edges whose source and destination are both inside that set (`utils/data_utils.py:82-84`), and this routine is used by both the supervised forward pass (`GVAE.forward`) and the encoder-only latent extraction (`GVAE.get_mus_only`). Training batches therefore propagate over train→train edges, validation over val→val, and test over test→test; DDPM latents are extracted per split (`utils/latent_extraction.py`). No signal crosses splits through the graph (see `tests/test_subgraph_remap.py`).

---

## 3. B4 — Protocol chọn threshold

- **Metric chính là threshold-free:** ROC-AUC và PR-AUC (không phụ thuộc ngưỡng).
- Khi báo F1 / balanced accuracy: ngưỡng chọn để **maximize F1 trên inner-validation**, không bao giờ trên test — `training/train_gvae.py::_threshold_max_f1`, dùng tại `:2243` trong `kfold_evaluate_gvae_classifier`.
- `utils/classification_eval.py` đã ghi rõ caveat (`threshold_metric_caveat`): F1/BA ở best-threshold chọn trên chính split báo cáo là **lạc quan**; ưu tiên `roc_auc`/`pr_auc`.

**Protocol chốt:** báo AUC + AUPRC làm đầu (kèm bootstrap CI); F1/BA chỉ báo kèm ngưỡng chọn trên inner-val, ghi chú rõ.

---

## 4. B6 — AUPRC

Đã có trong package canonical (`research/2026-10-03-canonical-r32/reports/final_results_package_r32.md` §2a), prevalence 0.749:

| Estimator | PR-AUC seed 42 [95% CI] | 5-seed mean ± sd |
|---|---|---|
| head | 0.8063 [0.7421–0.8803] | 0.8297 ± 0.0133 |
| probe | 0.8221 [0.7600–0.8899] | 0.8084 ± 0.0234 |

`kfold_evaluate_gvae_classifier` cũng đã tính `head_auprc`/`probe_auprc` kèm CI. **Việc còn lại thuộc bản thảo:** đưa PR-AUC vào bảng headline của manuscript (gắn với I4) — đây là sửa trong PDF, owner làm.

---

## 5. Giới hạn

- A2/A3/A5 kiểm chứng bằng **đọc code + test hiện có**, không chạy lại end-to-end (việc chạy lại canonical nằm ở A7/pooled-OOF đã có).
- A5 không kiểm tra đường `deprecated/` (không dùng cho số báo cáo).
- B6 là số đã có; "prominent trong manuscript" chưa thực hiện được ở đây vì manuscript là PDF ngoài repo.

## 6. Bằng chứng tái tra

`data/build_ln_pc_ihc_g.py:184-190,246-251,296-310` · `utils/data_utils.py:82-84,162,187` · `models/gvae_model.py:109,244` · `utils/latent_extraction.py:144-189` · `training/train_gvae.py:2243,2265` · `utils/classification_eval.py:271-275` · `tests/test_train_fold_only_fitting.py` · `tests/test_subgraph_remap.py` · `research/2026-10-03-canonical-r32/reports/final_results_package_r32.md` §2a.
