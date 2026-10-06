# C1 — Ablations (config-only) với CI + DeLong

> **Ngày:** 2026-10-06 · **Người làm:** owner-side agent · **Trạng thái:** C1 **một phần** (7/8 arm; còn no-GNN/MLP).
> **Kết luận một dòng:** không arm nào khác `full` có ý nghĩa thống kê sau Holm (p_holm nhỏ nhất ≈ 0.099 cho `radiology_only`); `full` **không** tốt hơn các mô hình đơn giản hơn (pathology-only/clinical-only nominally cao hơn), và radiology là view yếu nhất.

---

## 1. Câu hỏi

`PUBLICATION_CHECKLIST` C1: ablations no-contrastive / no-GNN / unimodal / bimodal / full, kèm CI và ý nghĩa thống kê ("confirm each drop is significant").

## 2. Protocol

`research/2026-10-06-c1-ablations/scripts/run_ablations.py` — dùng đúng `kfold_evaluate_gvae_classifier` (pooled-OOF, canonical r32), seeds 42–46, base config copy từ `run_oof.py`. Tất cả arm dùng **cùng split** cho mỗi seed (chỉ `random_seed` khác nội dung arm); DeLong ghép cặp trên cùng bệnh nhân.

Arm biểu diễn bằng config:
| Arm | Thay đổi |
|---|---|
| `full` | canonical (3 view, cross_cl=0.5) |
| `no_contrastive` | `loss_weights.cross_cl=0`, anneal cross_cl = 0 |
| `clinical_only` / `pathology_only` / `radiology_only` | `view_configs` còn 1 view |
| `clinical_pathology` / `clinical_radiology` / `pathology_radiology` | `view_configs` còn 2 view |

Chưa làm (cần cờ opt-in ở `models/`): **no-GNN (MLP)** và **C5** (attention vs mean/max pooling).

## 3. Kết quả

Pooled-OOF trên 247 BN, seed 42–46. `Δ = full − arm` (âm = arm tốt hơn full).

| Arm | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | Δ ROC (mean) | seed-mean Δ | p_holm |
|---|---|---|---|---|---|
| **full** | **0.6531 ± 0.0113** | 0.8287 ± 0.0034 | — | — | — |
| no_contrastive | 0.6633 ± 0.0131 | 0.8337 ± 0.0177 | −0.0102 | −0.0169 | 1.0 |
| clinical_only | 0.6615 ± 0.0147 | **0.8461** ± 0.0230 | −0.0084 | −0.0201 | 1.0 |
| pathology_only | **0.6790 ± 0.0265** | 0.8292 ± 0.0156 | −0.0259 | 0.0123 | 1.0 |
| radiology_only | **0.5286 ± 0.0129** | 0.7720 ± 0.0168 | +0.1245 | 0.1378 | 0.0994 |
| clinical_pathology | 0.6622 ± 0.0125 | 0.8283 ± 0.0057 | −0.0091 | 0.0064 | 1.0 |
| clinical_radiology | 0.6253 ± 0.0372 | 0.8252 ± 0.0229 | +0.0278 | −0.0044 | 1.0 |
| pathology_radiology | 0.6333 ± 0.0215 | 0.8163 ± 0.0151 | +0.0198 | 0.0244 | 1.0 |

## 4. Kết luận trung thực

1. **Không ablation nào có ý nghĩa sau Holm.** p_holm nhỏ nhất là **0.0994** (`radiology_only` vs full; trước Holm p=0.0142 ở seed-mean). Mọi arm khác p_holm = 1.0. Với N=247 và 5 seed, các chênh lệch quan sát nằm trong nhiễu — không được tuyên bố "drop có ý nghĩa".
2. **`full` không phải arm tốt nhất.** `pathology_only` (0.679) và `clinical_only`/`clinical_pathology`/`no_contrastive` đều **nominally cao hơn** full (0.653); PR-AUC cao nhất là `clinical_only` (0.846). Nghĩa là tín hiệu chính nằm ở clinical/pathology; thêm radiology không giúp, thậm chí làm giảm.
3. **Radiology là view yếu nhất.** `radiology_only` = 0.529 (gần random), và mọi arm chứa radiology (clinical_radiology, pathology_radiology) đều thấp hơn full. Phù hợp với việc chỉ 187/247 BN có radiology và view này nhiễu.
4. **Contrastive loss không giúp.** `no_contrastive` (0.663) ≥ `full` (0.653) — bỏ contrastive không làm giảm kết quả (chênh không ý nghĩa).
5. Nhất quán với các phát hiện trước: baseline đơn giản (clinical LR 0.707) ≥ GVAE head (0.664); không có so sánh nào của GVAE đạt ý nghĩa.

**Cách viết vào bài:** ablation là *kiểm tra độ bền vững*, không phải bằng chứng "full tốt nhất"; ghi trung thực rằng các drop không đạt ý nghĩa và view/alignment không cải thiện trên cohort này.

## 5. Giới hạn

- **Chưa có no-GNN (MLP)** và **aggregator pooling (C5)** — cần thêm cờ opt-in + test ở `models/`; C1 chưa đóng hoàn toàn.
- Arm dùng chung split nhưng khác RNG nội bộ arm → DeLong ghép cặp hợp lệ trên cùng bệnh nhân; so sánh theo seed là hợp lệ.
- `full` ở đây (0.6531 ± 0.0113) khớp comparator canonical độc lập cùng protocol trong `research/2026-10-06-nested-cv-hparam/` (0.6531 ± 0.0113) — cross-check tốt.
- PR-AUC trong bảng là trung bình các PR-AUC per-seed (không phải PR-AUC của seed-mean scores).

## 6. Tái lập

```bash
.venv/bin/python research/2026-10-06-c1-ablations/scripts/run_ablations.py \
    --data data_ln_pc_ihc_g_r32.pt --seeds 42,43,44,45,46 --tag ablation
# output (git-ignored): research/2026-10-06-c1-ablations/output/ablation/{summary,metrics_per_seed,delong_tests}.csv
```

Lịch sử review: gate `code-reviewer` vòng 1 phát hiện lỗi `IndexError` ở tầng summary (arm `full` không có dòng DeLong) → đã sửa; vòng 2 **PASS**.
