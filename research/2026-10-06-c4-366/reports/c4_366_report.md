# C4 — Complete-cohort (N=366) robustness under the canonical pipeline

> **Ngày:** 2026-10-06 · **Trạng thái:** đóng C4.
> **Kết luận một dòng:** chạy lại cohort đầy đủ 366 BN dưới pipeline canonical cho head AUC **0.6406 ± 0.0137** (AUPRC 0.8175) — **gần bằng** kết quả 247 canonical (0.6531 ± 0.0113); con số legacy "0.711 ± 0.080" **không** tái lập được dưới protocol hiện tại.

---

## 1. Dữ liệu & protocol

- Cohort 366 = toàn bộ `data/clinical_features_tmb.csv` (366 BN có label/pfs); canonical 247 = subset TMB not-null. TMB thiếu ở 119 BN → median impute (đúng như luật impute sẵn có).
- Cờ opt-in mới `--cohort {tmb,all}` trong `data/build_ln_pc_ihc_g.py` (mặc định `tmb` = canonical, không đổi hành vi); test `tests/test_build_ln_pc_ihc_g.py`.
- Graph: `data/build_ln_pc_ihc_g.py --cohort all --drop-radiology-artifacts both` → 366 BN, clinical 366×22, pathology mask 157, radiology 431 lesion/233 patient, lesion 32.
- Đánh giá: `run_oof.py` (pooled-OOF canonical), seeds 42–46.

## 2. Kết quả (N=366, prevalence 0.746)

| Seed | head ROC-AUC | head AUPRC | probe AUC |
|---|---|---|---|
| 42 | 0.6614 | 0.8406 | 0.6609 |
| 43 | 0.6390 | 0.8075 | 0.6151 |
| 44 | 0.6371 | 0.8068 | 0.6559 |
| 45 | 0.6234 | 0.8069 | 0.6809 |
| 46 | 0.6422 | 0.8259 | 0.6557 |
| **Mean ± sd** | **0.6406 ± 0.0137** | **0.8175 ± 0.0152** | 0.6537 ± 0.0239 |

seed 42 head AUC 0.6614 [95% CI 0.5945–0.7250].

## 3. So sánh & kết luận trung thực

| Cohort | head ROC-AUC |
|---|---|
| canonical N=247 | 0.6531 ± 0.0113 |
| complete N=366 | 0.6406 ± 0.0137 |

- Kết quả **gần bằng** (chênh 0.012, trong khoảng sd) → kết luận head **bền vững** khi mở rộng ra toàn cohort.
- **Legacy "0.711 ± 0.080" không tái lập.** Dưới pipeline canonical, complete-cohort chỉ đạt ~0.64, không phải 0.711. Nhiều khả năng con số cũ dùng tiền xử lý/protocol/threshold khác (nghiệm thu khác) — **không** nên trích 0.711 như kết quả hiện hành. Ghi rõ trong bài.
- AUC không tăng khi thêm 119 BN thiếu TMB → tín hiệu chính vẫn ở cohort TMB-notna; nhất quán với kết luận "TMB/cohort không tạo thêm sức mạnh phân loại".

## 4. Giới hạn

- 119 BN thêm có TMB/`clinical_pdl1_score`/`tumor_burden` thiếu (median impute) và mask pathology/radiology khác → không so trực tiếp từng-BN với legacy.
- Một cohort; 5 seed.

## 5. Tái lập

```bash
.venv/bin/python data/build_ln_pc_ihc_g.py --out data_ln_pc_ihc_g_full366_r32.pt \
    --cohort all --drop-radiology-artifacts both
.venv/bin/python research/2026-10-03-radiology-artifact-ablation/scripts/run_oof.py \
    --data data_ln_pc_ihc_g_full366_r32.pt --tag full366 --seed <42..46>
# output (git-ignored): research/2026-10-03-radiology-artifact-ablation/output/full366_seed{42..46}/oof_arrays.npz
```
