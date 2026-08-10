# DDPM Diagnosis Experiment — A1 (2026-08-10)

> Session kế tiếp của Diff-GVAE. Mục tiêu: chẩn đoán và cải thiện chất lượng latent DDPM
> sinh từ GVAE. Mọi so sánh với baseline run `...20260809_184454` (epochs 120, guidance 3.0).
> Setup chung: 1 fold (fold_1) × both_classes × ratio 0.25, checkpoint rank 1
> (`gvae_bestparam_ranked_20260807_095233`), latent `concat_mu` 96-dim, timesteps 250,
> seed 42, device cpu, `--filter-synthetic --filter-quantile 0.95`.

## 1. Baseline (120 epochs, guidance 3.0) — các vấn đề

| Metric | Baseline | Nhận xét |
|---|---|---|
| MMD | 0.310 | cao |
| Coverage | 0.0 | 0 mẫu sinh phủ real |
| Diversity gen/real pairwise | 29.05 / 10.61 (ratio 2.74) | over-dispersed 2.7× |
| NN mean (gen→real, k=1) | 18.95 | xa real |
| near_duplicate_fraction | 0.0 | không memorization ✓ |
| gen global std | 2.34 vs real 1.28 | over-dispersed 1.8× |

## 2. Các nhánh diagnosis đã chạy (mỗi nhánh ~2–3 phút CPU, seed 42)

| Nhánh | Run dir (suffix) | Loss cuối |
|---|---|---|
| 300ep, g1.0 | `20260810_151834` | 0.347 (ep300) |
| 400ep, g1.0 | `20260810_152113` | 0.307 (ep400) |
| 500ep, g1.0 | `20260810_152609` | 0.298 (ep500) |
| 400ep, g2.0 | `20260810_152946` | 0.307 (ep400) |
| **600ep, g1.0 (chốt)** | `20260810_153523` | 0.274 (ep600) |

Loss chưa plateau ở mọi nhánh (vẫn giảm đều ~0.03/50ep). Guidance 2.0 (400ep) cho kết quả
gần như **giống hệt** guidance 1.0 (400ep) → guidance_scale không phải knob hữu ích ở chế độ này.

## 3. Kết quả quality metrics

| Metric | Baseline (120ep,g3) | 300ep g1 | 400ep g1 | 500ep g1 | 400ep g2 | **600ep g1** |
|---|---|---|---|---|---|---|
| MMD | 0.310 | 0.099 | 0.066 | 0.045 | 0.067 | **0.032** (-90%) |
| Coverage | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.056** (>0 lần đầu) |
| Diversity gen | 29.05 | 19.69 | 17.46 | 16.03 | 17.46 | **14.99** |
| Ratio gen/real | 2.74 | 1.86 | 1.65 | 1.51 | 1.65 | **1.41** (< 1.5 ✓) |
| NN mean | 18.95 | 11.91 | 9.98 | 8.88 | 9.99 | **8.04** (-58%) |
| near_dup | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** ✓ |
| gen std | 2.34 | 1.75 | 1.63 | 1.55 | 1.63 | **1.50** (real 1.28) |

Xu hướng đơn điệu theo epochs: mọi chỉ số cải thiện liên tục 120 → 300 → 400 → 500 → 600.

## 4. Giải thích coverage = 0.0 (các nhánh < 600ep)

Coverage metric (`training/latent_ddpm_augmentation.py:_coverage`):
coverage = tỷ lệ real points có nearest-generated distance ≤ **p95 của real-NN distances**.

Chẩn đoán trên run 500ep (197 real vs 49 generated, 96-dim):
- Real NN distances: mean 2.72, median 2.72, **p95 = 4.92** (real dày đặc, 196 neighbors mỗi point)
- Gen→real NN distances: mean 7.38, **min 5.08** — chỉ cách ngưỡng 3%!
- Coverage theo các ngưỡng: t=6 → 16.8%, t=8 → 68.0%, t=10 → 99.5%

Kết luận: coverage 0.0 = ngưỡng p95-real-NN rất khắt khe (4.92) + 49 mẫu sinh trong 96-dim
+ over-dispersion còn lại. Không phải sinh sai vùng (mean_abs_mean_difference 0.14, NN mean
8.04 < pairwise real 10.61 ở 600ep). 600ep đưa min NN xuống dưới ngưỡng → coverage 0.056.

## 5. Downstream (fold 1, chỉ mang tính tham khảo — 1 fold nhiễu)

| Branch | real-only ROC | +aug ROC | synth count |
|---|---|---|---|
| 300ep | 0.6570 | 0.6237 | 49 |
| 400ep | 0.6570 | 0.6549 | 49 |
| 500ep | 0.6570 | 0.6549 | 49 |
| **600ep** | 0.6570 | **0.6694** | 49 |
| 600ep filtered (q0.95) | 0.6570 | 0.6528 | **1** (filter giữ được mẫu đầu tiên) |

## 6. Kết luận & quyết định A2

**A1 CẢI THIỆN — chạy A2 với epochs 600, guidance 1.0.**
- Tất cả mục tiêu A1 đạt tại 600ep: coverage > 0 (0.056), ratio gen/real < 1.5 (1.41),
  NN mean giảm mạnh (18.9 → 8.04), MMD -90%, không memorization.
- Guidance 1.0 (không phải 2.0/3.0) — guidance không ảnh hưởng chất lượng sinh ở đây.
- Filter quantile 0.95 giờ giữ được 1 mẫu ở 600ep — vẫn quá chặt, cần A3 retune 0.7–0.9
  sau A2.
- Loss chưa plateau ở 600ep (0.274) — nếu A2 sau này cần thêm, có thể nâng epochs,
  nhưng đã đạt mọi ngưỡng mục tiêu nên chốt 600.

## 7. Artifacts

- Logs: `research/2026-08-06-project-review-audit/output/raw/ddpm_a1*.log`
- Runs: `outputs/conditional_latent_ddpm/conditional_latent_ddpm_from_gvae_bestparam_ranked_20260807_095233_20260810_*`
- Chốt config A2: epochs 600, timesteps 250, guidance 1.0, seed 42, rank 1, device cpu

---

# PHẦN 2 — A2 full rerun + A3 filter retune + A4 TSTR (cùng ngày)

## 8. A2 — Full rerun (3 modes × 4 ratios × 5 folds, epochs 600, guidance 1.0)

Run: `conditional_latent_ddpm_from_gvae_bestparam_ranked_20260807_095233_20260810_153849`
(~6 phút CPU, log `pipeline_phase5.log`). Real-only baseline không đổi
(0.7104/0.8694/0.7202) vì deterministic từ GVAE.

### 8a. Unfiltered branches

| Branch | ROC-AUC | PR-AUC | BA | synthetic TB | coverage (mean) | MMD (mean) |
|---|---|---|---|---|---|---|
| Real only | 0.7104 | 0.8694 | 0.7202 | 0 | — | — |
| both_classes r0.25 | 0.7050 | 0.8732 | 0.7087 | 49 | 0.031 | 0.045 |
| both_classes r0.5 | 0.6673 | 0.8432 | 0.6920 | 98.6 | 0.024 | 0.040 |
| both_classes r1.0 | 0.6809 | 0.8500 | 0.6938 | 197.6 | 0.071 | 0.034 |
| both_classes r2.0 | 0.6529 | 0.8302 | 0.7000 | 395.2 | 0.131 | 0.025 |
| minority_only r0.25 | 0.7024 | 0.8687 | 0.7180 | 12 | 0.011 | 0.090 |
| minority_only r0.5 | 0.6798 | 0.8464 | 0.6915 | 24.6 | 0.012 | 0.068 |
| minority_only r1.0 | 0.6719 | 0.8441 | 0.6976 | 49.6 | 0.025 | 0.054 |
| minority_only r2.0 | 0.6579 | 0.8364 | 0.6870 | 99.2 | 0.024 | 0.040 |
| nonresponder_only r0.25 | 0.6982 | 0.8671 | 0.7156 | 37 | 0.013 | 0.056 |
| nonresponder_only r0.5 | 0.6989 | 0.8602 | 0.7037 | 74 | 0.062 | 0.044 |
| nonresponder_only r1.0 | 0.6903 | 0.8600 | 0.6935 | 148 | 0.122 | 0.034 |
| nonresponder_only r2.0 | 0.6981 | 0.8655 | 0.7132 | 296 | 0.110 | 0.027 |

**Chất lượng latent cải thiện lớn so baseline 20260809_184454** (coverage 0→0.01–0.13
mọi branch; MMD 0.31→0.025–0.09). Downstream vẫn ≈ real-only — chênh trong nhiễu
(real-only std ROC 0.0475), chưa branch nào thắng rõ.

### 8b. Filtered branches (q0.95) — giờ giữ được mẫu (baseline giữ 0)

| Branch | ROC-AUC | synth TB/fold |
|---|---|---|
| both_classes r0.25 / r0.5 / r1.0 / r2.0 | 0.7055 / 0.7074 / **0.7130** / 0.6826 | 1.8 / 4.0 / 8.0 / 17.4 |
| minority_only r0.25 / r0.5 / r1.0 / r2.0 | 0.6908 / 0.6997 / 0.6950 / 0.7129 | 1.2 / 2.2 / 4.6 / 9.0 |
| nonresponder_only r0.25 / r0.5 / r1.0 / r2.0 | 0.7073 / 0.7069 / 0.7071 / 0.7097 | 0.8 / 2.0 / 6.2 / 6.0 |

## 9. A3 — filter_quantile retune (post-hoc trên latents A2, không retrain)

Script: `research/2026-08-06-project-review-audit/scripts/a3_filter_quantile_retune.py`
(sửa dụng đúng `filter_synthetic_latents_by_knn` + `train_downstream_classifier` của
pipeline). Output: `.../20260810_153849/a3_filter_quantile_retune.json`.

**Mọi quantile 0.70–0.99 giờ giữ được mẫu** (synthetic_count > 0). Threshold = quantile
của real-NN distances per class → **quantile càng cao giữ càng nhiều** (khắt khe hơn
không giúp): both_classes r1.0: q0.70 giữ 0.4/fold, q0.95 giữ 8.0, q0.97 giữ 13.6,
q0.99 giữ 22.2.

**Ghi chú phương pháp (sau code-review):** các cell kept=0 được evaluate như pipeline
(degenerate → real-only baseline ROC), KHÔNG drop folds — nếu không, mean bị lệch
(các quantile thấp giữ 0 mẫu nhiều nhất → ROC bị inflate). Kết quả dưới đây là
like-for-like 5/5 folds mọi cell.

Best cells theo ROC (mean 5 folds, band nhiễu real-only std 0.0475):
- both_classes r1.0 **q0.97**: 13.6/fold, ROC 0.7144 (+0.004 so real-only 0.7104)
- minority_only r2.0 q0.90: 6.0/fold, ROC 0.7193 (+0.009)
- nonresponder_only r0.5 q0.99: 8.8/fold, ROC 0.7212 (+0.011)
- Tất cả nằm trong nhiễu (±0.01), không cell nào thắng có ý nghĩa thống kê.

**Kết luận A3:** giữ quantile 0.95–0.97 (mặc định 0.95 ổn, 0.97 giữ nhiều hơn ~1.7× với
ROC tương đương); hạ xuống 0.7–0.9 KHÔNG có lợi (giữ ít mẫu hơn, ROC không tốt hơn).

## 10. A4 — TSTR control (train synthetic-only → test val real)

Script: `research/2026-08-06-project-review-audit/scripts/a4_tstr_control.py`.
Output: `.../20260810_153849/a4_tstr_control.json`.
Chỉ có nghĩa với both_classes (minority_only/nonresponder_only sinh 1 class → binary
downstream không train được).

| Branch | n_train | ROC-AUC | PR-AUC |
|---|---|---|---|
| REAL_ONLY (baseline) | 198 | 0.7104 | 0.8694 |
| both_classes r0.25 | 49 | 0.5832 | 0.8066 |
| both_classes r0.5 | 99 | 0.5510 | 0.7832 |
| both_classes r1.0 | 197 | 0.6223 | 0.8200 |
| both_classes r2.0 | 395 | 0.6159 | 0.8070 |

**Kết luận A4:** synthetic mang tín hiệu phân lớp THẬT (TSTR ROC 0.55–0.62, PR 0.78–0.82,
đều >> 0.5/random) nhưng kém real-only ~0.09–0.16 ROC → augmentation chỉ học được phần
tín hiệu, chất lượng chưa bằng dữ liệu thật. Khớp với kết luận trung thực chung:
augmentation chưa thắng real-only có ý nghĩa thống kê.

## 11. Tổng kết A1–A4

1. **A1**: 600 epochs là knob quyết định — mọi chỉ số chất lượng latent đạt mục tiêu
   (coverage > 0, ratio < 1.5, MMD -90%, NN -58%, không memorization).
2. **A2**: cải thiện chất lượng không chuyển thành thắng downstream — aug ≈ real-only
   (trong nhiễu). Filtered path hoạt động trở lại (giữ 1.8–17.4/fold ở q0.95).
3. **A3**: q0.95–0.97 là điểm cân bằng tốt; quantile thấp hơn không giúp.
4. **A4**: TSTR chứng minh synthetic học được tín hiệu thật nhưng chưa đủ thay thế real.
5. Còn lại: A5 (PCA branch) chưa chạy; manuscript B1–B3 chờ owner gỡ defer.

## 12. A5 — PCA branch (--pca-components 32, 600ep, fold 1, both_classes r0.25)

Run: `...20260810_161120` (log `ddpm_a5_pca32.log`). Lần đầu test v2 flag PCA.

### 12a. Quality metrics (trong không gian PCA 32-dim — không so trực tiếp với 96-dim)

| Metric | Non-PCA 600ep (96-dim) | PCA 32 (600ep) |
|---|---|---|
| MMD | 0.032 | **0.0228** |
| Coverage | 0.056 | **0.284** (5.1×) |
| Diversity ratio gen/real | 1.41 | **1.24** |
| NN mean | 8.04 | **6.75** |
| near_dup | 0.0 | 0.0 ✓ |
| gen std vs real | 1.50 vs 1.28 | 1.73 vs 1.39 (32-dim) |

### 12b. Downstream (fold 1)

| Branch | ROC | PR | synth kept (q0.95) |
|---|---|---|---|
| real_only (pca_32) | 0.6611 | 0.8403 | 0 |
| +aug unfiltered | 0.6632 | 0.8453 | 49 |
| +aug filtered q0.95 | 0.6570 | 0.8366 | **10.0** (20% giữ — vs 1 mẫu non-PCA) |

### 12c. Kết luận A5

PCA 32 cải thiện mạnh chất lượng latent (coverage 0.284, ratio 1.24) và làm filtered
path q0.95 hoạt động hiệu quả (giữ 20% generated so với ~2% non-PCA). Downstream vẫn
≈ real-only (chênh nhiễu) — nhưng nếu pipeline cuối dùng filtered path, **PCA 32 là
nền tốt hơn**.

## 13. Tổng kết toàn bộ A1–A5

| # | Task | Kết quả chính |
|---|---|---|
| A1 | Diagnosis epochs/guidance | 600ep/g1.0: coverage 0.056, MMD 0.032, ratio 1.41, NN 8.04 — cải thiện rõ; guidance không phải knob |
| A2 | Full rerun 600ep | Coverage > 0 mọi branch; downstream ≈ real-only (trong nhiễu); filtered giữ mẫu |
| A3 | filter_quantile retune | q0.95–0.97 tối ưu; q thấp hơn không giúp; best cell both_classes r1.0 q0.97 ROC 0.7144 |
| A4 | TSTR control | ROC 0.55–0.62 vs 0.7104 real — synthetic học được tín hiệu thật, kém real |
| A5 | PCA 32 | Coverage 0.284, ratio 1.24, filter q0.95 giữ 20% — nền tốt hơn cho filtered path |

**Kết luận trung thực chung:** epochs 600 + guidance 1.0 (+ PCA 32 nếu dùng filter) khắc
phục hoàn toàn vấn đề chất lượng latent của baseline 120ep/g3.0. Tuy vậy, augmentation
vẫn chưa thắng real-only có ý nghĩa thống kê trên downstream (chênh trong CI ~±0.05).
Đây là giới hạn của phương pháp trên cỡ mẫu 247, không phải lỗi config.
