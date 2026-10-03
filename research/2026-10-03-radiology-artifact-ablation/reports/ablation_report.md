# Ablation — 2 cột index artifact trong radiology (2026-10-03)

**Câu hỏi:** Slot 0 (thứ tự dòng lesion trong file) và slot 1 (`lesion_index`) của `lesion.x` không phải radiomics, nhưng lại tương quan với nhãn (xem phần "Vì sao chạy"). Hiệu năng GVAE có phụ thuộc vào chúng không?

**Kết luận:** **Không.** Bỏ hai slot này không làm giảm AUC; trung bình còn tăng nhẹ. Mọi khoảng tin cậy paired của ΔAUC ở head classifier đều chứa 0. → Có thể bỏ artifact mà không mất hiệu năng.

## Vì sao chạy

Trên 187 bệnh nhân có radiology (`binary_label`, prevalence 0.749):

| Đặc trưng | AUC | p (hoán vị 5000 lần) |
|---|---|---|
| Slot 0 (file-rank, trung bình theo BN) | 0.406 | 0.048 |
| Số lesion (= max slot 1) | 0.400 | 0.020 |
| Slot 0, chỉ trong nhóm cùng số lesion 2–4 (hoán vị phân tầng) | 0.332 | 0.034 |
| BN 1 lesion: slot 0 | 0.497 | — |
| Logistic regression CV (5-fold), chỉ slot 0–1 | 0.595 | — |
| Logistic regression CV (5-fold), 32 radiomics | 0.586 | — |

Thứ tự dòng trong `radiology_features.csv` không sắp theo `main_index`, nên "file-rank" là một định danh có tương quan với nhãn. Nó còn đi vào similarity signature của radiology (ảnh hưởng tới A4).

## Thiết lập

- Graph: `data/build_ln_pc_ihc_g.py --drop-radiology-artifacts {none,file_rank,both}`; các slot còn lại giữ nguyên giá trị (RobustScaler theo từng cột; đã test trong `tests/test_build_ln_pc_ihc_g.py`).

  | Biến thể | lesion dim | edge radiology |
  |---|---|---|
  | `full34` (canonical) | 34 | 652 |
  | `drop_rank33` | 33 | 718 |
  | `drop_both32` | 32 | 720 |

- Đánh giá: pooled-OOF `kfold_evaluate_gvae_classifier`, config sao chép nguyên từ `pipeline_phase4a_oof.sh` (`scripts/run_oof.py`); seed 42/43/44; `OMP_NUM_THREADS=3`.
- So sánh: paired bootstrap ΔAUC so với `full34` cùng seed (cùng fold, cùng bệnh nhân), 2000 lần (`scripts/compare.py`).

## Kết quả

OOF ROC-AUC (head = fusion classifier, probe = linear probe trên latent):

| Score | Biến thể | seed 42 | seed 43 | seed 44 | mean ± sd | AUPRC mean |
|---|---|---|---|---|---|---|
| head | full34 | 0.6076 | 0.6476 | 0.6488 | 0.635 ± 0.024 | 0.815 |
| head | drop_rank33 | 0.6283 | 0.6492 | 0.6613 | 0.646 ± 0.017 | 0.836 |
| head | drop_both32 | 0.6431 | 0.6678 | 0.6533 | 0.655 ± 0.012 | 0.826 |
| probe | full34 | 0.6325 | 0.6394 | 0.6555 | 0.643 ± 0.012 | 0.821 |
| probe | drop_rank33 | 0.6793 | 0.6386 | 0.6552 | 0.658 ± 0.021 | 0.825 |
| probe | drop_both32 | 0.6495 | 0.6119 | 0.6670 | 0.643 ± 0.028 | 0.819 |

Paired ΔAUC so với full34 (95% CI): head trong khoảng +0.002 → +0.036, **mọi CI đều chứa 0**; probe trong khoảng −0.028 → +0.047, chỉ có 1/6 CI loại trừ 0 (drop_rank33, seed 42, +0.047 [0.007, 0.088]). Chi tiết: `output/paired_delta_auc.csv`, `output/per_run_auc.csv`.

## Ghi chú

- **Tái lập:** `full34` seed 42 cho head AUC 0.6076, trong khi run canonical ngày 2026-08-09 ghi 0.6065 (cùng config). Chênh lệch 0.001 nhiều khả năng do số thread CPU (OMP=3 so với mặc định) → pipeline không hoàn toàn deterministic giữa các cấu hình thread. Cần ghi chú khi viết phần reproducibility (H2).
- Chênh lệch giữa các seed (sd tới 0.024) lớn hơn hoặc ngang với hiệu ứng của ablation → vẫn nên báo cáo kết quả theo nhiều seed (B3).
- Chỉ đánh giá GVAE classifier/probe; chưa chạy lại phần DDPM augmentation trên graph đã bỏ artifact.

## Cập nhật

Owner đã chọn đổi canonical sang `drop_both32` (2026-10-03). Đã chạy thêm seed 45/46 (head 32 chiều > 34 chiều ở 5/5 seed, mean 0.664 so với 0.640) → xem `research/2026-10-03-canonical-r32/reports/canonical_r32_report.md`.

## Quyết định cần chốt (owner) — đã chốt, xem phần Cập nhật

1. Đổi canonical sang `drop_both32` (chỉ giữ radiomics thuần; số lesion nếu cần thì đưa vào clinical như một feature có tên rõ ràng), **hoặc** `drop_rank33` (giữ `lesion_index`), **hoặc** giữ 34 và nêu trong Limitations kèm ablation này.
2. Nếu đổi canonical: chạy lại run canonical GVAE + OOF + DDPM, cập nhật mọi số liệu đã chốt (A7) và docs.
