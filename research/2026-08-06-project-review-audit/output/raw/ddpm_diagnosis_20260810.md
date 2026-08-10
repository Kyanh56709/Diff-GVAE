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
