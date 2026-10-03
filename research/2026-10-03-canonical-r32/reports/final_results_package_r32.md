# FINAL RESULTS PACKAGE — Diff-GVAE, canonical r32 (2026-10-03)

Thay thế `research/2026-08-06-project-review-audit/reports/final_results_package.md` (graph 34 chiều).

## 1. Dữ liệu
- Graph: `data_ln_pc_ihc_g_r32.pt`: 247 BN, 333 lesion; clinical 22, pathology 15, radiology **32** radiomics/lesion; cạnh cosine > 0.8 (radiology 720 edge). Build: `python data/build_ln_pc_ihc_g.py --out data_ln_pc_ihc_g_r32.pt --drop-radiology-artifacts both`.
- Labels: `binary_label` 1 = non-responder (185), 0 = responder (62); positive = 1.
- Lý do đổi: 2 slot index (file-rank, lesion_index) tương quan với nhãn; xem `research/2026-10-03-radiology-artifact-ablation/reports/ablation_report.md`.

## 2. GVAE
### 2a. Pooled-OOF (SỐ CHÍNH) — seed 42 [95% bootstrap CI] và trung bình 5 seed (42–46)

| Estimator | Metric | seed 42 [95% CI] | 5-seed mean ± sd |
|---|---|---|---|
| head | roc_auc | 0.6431 [0.5628-0.7272] | 0.6635 ± 0.0161 |
| head | pr_auc | 0.8063 [0.7421-0.8803] | 0.8297 ± 0.0133 |
| head | balanced_accuracy | 0.6285 [0.5613-0.6981] | 0.6493 ± 0.0147 |
| head | f1 | 0.8022 [0.7558-0.8454] | 0.7771 ± 0.0182 |
| probe | roc_auc | 0.6495 [0.5670-0.7313] | 0.6356 ± 0.0238 |
| probe | pr_auc | 0.8221 [0.7600-0.8899] | 0.8084 ± 0.0234 |
| probe | balanced_accuracy | 0.5878 [0.5131-0.6583] | 0.6224 ± 0.0241 |
| probe | f1 | 0.7331 [0.6787-0.7845] | 0.7719 ± 0.0278 |

### 2b. Runs val-fold mean (5-fold)
| Run | Graph | ROC-AUC | PR-AUC | BA | F1 |
|---|---|---|---|---|---|
| canonical_r32_20261003_111618 | r32 | 0.6169 | 0.7999 | 0.6770 | 0.7861 |
| gvae_bestparam_ranked_r32_20261003_124706 | r32 | 0.6542 | 0.8437 | 0.7015 | 0.7178 |

## 3. DDPM latent augmentation (concat_mu, 600ep, g1.0, q0.95, rank 1 của gvae_bestparam_ranked_r32_20261003_124706)
**Nguồn gốc checkpoint GVAE (tái lập):** run `gvae_bestparam_ranked_r32_20261003_124706` được tạo bằng `run_bestparam_ranked_r32.py` bản commit `6cdfcad`, khi script chưa đặt `train_config['run_id']`, nên checkpoint được lưu với tiền tố mặc định `seed_42_*`. Để runner DDPM (glob `{run_id}_fold_*_rank_1_*.pt`) tìm thấy, 5 file `seed_42_fold_{1..5}_rank_1_*.pt` đã được **sao chép** sang tên `gvae_bestparam_ranked_r32_20261003_124706_fold_{1..5}_rank_1_*.pt` (đã kiểm tra `cmp`: giống hệt từng byte). Commit `a63be5d` sửa script để đặt tên đúng ngay từ đầu. Chạy lại script đã sửa (run kiểm tra `gvae_bestparam_ranked_r32_20261003_133903`) **không** cho lại cùng checkpoint, vì pipeline không deterministic hoàn toàn (vd. latent_quality rank 1 fold 1: 0.5370 so với 0.5172; xem mục "Còn mở" 4 của `canonical_r32_report.md`). Run `133903` chỉ để kiểm tra tên file, không dùng cho số liệu nào ở đây. Nó là run GVAE mới nhất, nên runner DDPM luôn phải được gọi với `--gvae-run-id` tường minh.

### 3a. A2 full — run conditional_latent_ddpm_from_gvae_bestparam_ranked_r32_20261003_124706_20261003_125733
| Branch | ROC-AUC (±sd) | PR-AUC | BA | synthetic TB/fold | coverage | MMD | folds |
|---|---|---|---|---|---|---|---|
| real_only | 0.6982 ± 0.0359 | 0.8647 | 0.7046 | 0.0 | — | — | 5 |
| both_classes r1 (filtered) | 0.7055 ± 0.0461 | 0.8694 | 0.7044 | 12.8 | 0.104 | 0.045 | 5 |
| both_classes r0.25 | 0.7045 ± 0.0469 | 0.8675 | 0.7060 | 49.0 | 0.033 | 0.054 | 5 |
| both_classes r2 (filtered) | 0.7035 ± 0.0437 | 0.8692 | 0.7043 | 28.8 | 0.188 | 0.025 | 5 |
| both_classes r0.5 | 0.7016 ± 0.0744 | 0.8748 | 0.7020 | 98.6 | 0.076 | 0.046 | 5 |
| nonresponder_only r0.25 (filtered) | 0.7000 ± 0.0371 | 0.8659 | 0.7066 | 2.2 | 0.024 | 0.132 | 5 |
| nonresponder_only r1 (filtered) | 0.6964 ± 0.0422 | 0.8637 | 0.7044 | 8.6 | 0.087 | 0.040 | 5 |
| minority_only r0.5 (filtered) | 0.6960 ± 0.0453 | 0.8632 | 0.7046 | 2.2 | 0.006 | 0.222 | 5 |
| nonresponder_only r0.5 (filtered) | 0.6957 ± 0.0451 | 0.8665 | 0.7046 | 6.4 | 0.082 | 0.094 | 5 |
| minority_only r2 (filtered) | 0.6951 ± 0.0355 | 0.8684 | 0.7075 | 7.4 | 0.045 | 0.091 | 5 |
| nonresponder_only r0.25 | 0.6944 ± 0.0530 | 0.8658 | 0.7078 | 37.0 | 0.019 | 0.061 | 5 |
| nonresponder_only r2 (filtered) | 0.6938 ± 0.0362 | 0.8675 | 0.6958 | 20.8 | 0.202 | 0.035 | 5 |
| minority_only r0.25 (filtered) | 0.6937 ± 0.0396 | 0.8643 | 0.6967 | 0.8 | 0.000 | — | 5 |
| both_classes r0.5 (filtered) | 0.6935 ± 0.0539 | 0.8648 | 0.6913 | 6.8 | 0.076 | 0.060 | 5 |
| both_classes r0.25 (filtered) | 0.6932 ± 0.0418 | 0.8652 | 0.6967 | 4.6 | 0.037 | 0.100 | 5 |
| nonresponder_only r0.5 | 0.6876 ± 0.0663 | 0.8638 | 0.6959 | 74.0 | 0.082 | 0.038 | 5 |
| nonresponder_only r1 | 0.6841 ± 0.0774 | 0.8653 | 0.6918 | 148.0 | 0.089 | 0.037 | 5 |
| nonresponder_only r2 | 0.6800 ± 0.1005 | 0.8648 | 0.6899 | 296.0 | 0.208 | 0.028 | 5 |
| minority_only r0.25 | 0.6797 ± 0.0588 | 0.8578 | 0.6837 | 12.0 | 0.001 | 0.110 | 5 |
| minority_only r1 (filtered) | 0.6763 ± 0.0640 | 0.8562 | 0.6907 | 4.4 | 0.035 | 0.094 | 5 |
| minority_only r1 | 0.6725 ± 0.0595 | 0.8550 | 0.6770 | 49.6 | 0.038 | 0.061 | 5 |
| minority_only r0.5 | 0.6602 ± 0.0618 | 0.8451 | 0.6845 | 24.6 | 0.006 | 0.079 | 5 |
| minority_only r2 | 0.6569 ± 0.0876 | 0.8553 | 0.6670 | 99.2 | 0.047 | 0.045 | 5 |
| both_classes r1 | 0.6523 ± 0.0751 | 0.8515 | 0.6720 | 197.6 | 0.111 | 0.036 | 5 |
| both_classes r2 | 0.6466 ± 0.1167 | 0.8429 | 0.6697 | 395.2 | 0.195 | 0.028 | 5 |

### 3b. A5 PCA 32 full — run conditional_latent_ddpm_from_gvae_bestparam_ranked_r32_20261003_124706_20261003_130455
| Branch | ROC-AUC (±sd) | PR-AUC | BA | synthetic TB/fold | coverage | MMD | folds |
|---|---|---|---|---|---|---|---|
| real_only | 0.7075 ± 0.0271 | 0.8680 | 0.7206 | 0.0 | — | — | 5 |
| minority_only r1 (filtered) | 0.7113 ± 0.0312 | 0.8623 | 0.7344 | 18.6 | 0.287 | 0.033 | 5 |
| minority_only r0.5 | 0.7079 ± 0.0411 | 0.8687 | 0.7215 | 24.6 | 0.132 | 0.086 | 5 |
| minority_only r0.25 | 0.7071 ± 0.0175 | 0.8735 | 0.7189 | 12.0 | 0.081 | 0.097 | 5 |
| nonresponder_only r0.25 (filtered) | 0.7048 ± 0.0253 | 0.8667 | 0.7152 | 9.4 | 0.169 | 0.114 | 5 |
| both_classes r0.5 (filtered) | 0.7035 ± 0.0280 | 0.8696 | 0.7165 | 29.4 | 0.404 | 0.022 | 5 |
| minority_only r0.5 (filtered) | 0.7030 ± 0.0338 | 0.8632 | 0.7063 | 7.2 | 0.132 | 0.075 | 5 |
| minority_only r0.25 (filtered) | 0.7005 ± 0.0297 | 0.8711 | 0.7053 | 4.8 | 0.080 | 0.137 | 5 |
| minority_only r2 | 0.6979 ± 0.0461 | 0.8642 | 0.7265 | 99.2 | 0.394 | 0.054 | 5 |
| nonresponder_only r2 (filtered) | 0.6951 ± 0.0539 | 0.8648 | 0.7105 | 73.8 | 0.579 | 0.015 | 5 |
| nonresponder_only r0.5 (filtered) | 0.6944 ± 0.0580 | 0.8655 | 0.7146 | 22.4 | 0.358 | 0.030 | 5 |
| minority_only r2 (filtered) | 0.6831 ± 0.0572 | 0.8512 | 0.6983 | 31.4 | 0.385 | 0.034 | 5 |
| both_classes r0.25 | 0.6822 ± 0.0605 | 0.8568 | 0.6946 | 49.0 | 0.313 | 0.043 | 5 |
| nonresponder_only r1 (filtered) | 0.6814 ± 0.0392 | 0.8572 | 0.6893 | 43.8 | 0.524 | 0.020 | 5 |
| both_classes r0.25 (filtered) | 0.6800 ± 0.0552 | 0.8583 | 0.6924 | 17.2 | 0.310 | 0.028 | 5 |
| nonresponder_only r0.25 | 0.6784 ± 0.0486 | 0.8556 | 0.6986 | 37.0 | 0.170 | 0.049 | 5 |
| both_classes r0.5 | 0.6780 ± 0.0552 | 0.8651 | 0.6883 | 98.6 | 0.410 | 0.043 | 5 |
| minority_only r1 | 0.6778 ± 0.0399 | 0.8492 | 0.7017 | 49.6 | 0.291 | 0.053 | 5 |
| both_classes r1 (filtered) | 0.6763 ± 0.0483 | 0.8540 | 0.6822 | 64.2 | 0.643 | 0.012 | 5 |
| nonresponder_only r1 | 0.6692 ± 0.0460 | 0.8556 | 0.6841 | 148.0 | 0.525 | 0.041 | 5 |
| nonresponder_only r0.5 | 0.6562 ± 0.0733 | 0.8479 | 0.6817 | 74.0 | 0.358 | 0.049 | 5 |
| nonresponder_only r2 | 0.6558 ± 0.0601 | 0.8437 | 0.6768 | 296.0 | 0.582 | 0.039 | 5 |
| both_classes r2 (filtered) | 0.6553 ± 0.0651 | 0.8421 | 0.6742 | 126.6 | 0.693 | 0.011 | 5 |
| both_classes r1 | 0.6455 ± 0.0777 | 0.8404 | 0.6593 | 197.6 | 0.645 | 0.030 | 5 |
| both_classes r2 | 0.6238 ± 0.0613 | 0.8286 | 0.6599 | 395.2 | 0.699 | 0.030 | 5 |

### 3c. A3 filter_quantile retune

Aggregate A3 (nguyên văn từ `output/logs/a3_r32.log`):

```
=== AGGREGATE (mean over folds) ===
mode               ratio    q kept_mean  roc_mean
both_classes         100 0.70       0.0    0.6982 (5/5 folds)
both_classes         100 0.80       1.4    0.6982 (5/5 folds)
both_classes         100 0.90       6.6     0.698 (5/5 folds)
both_classes         100 0.95      12.8    0.7055 (5/5 folds)
both_classes         100 0.97      16.2    0.7064 (5/5 folds)
both_classes         100 0.99      32.0     0.682 (5/5 folds)
both_classes         200 0.70       1.0    0.6955 (5/5 folds)
both_classes         200 0.80       4.4    0.6986 (5/5 folds)
both_classes         200 0.90      14.8    0.6985 (5/5 folds)
both_classes         200 0.95      28.8    0.7035 (5/5 folds)
both_classes         200 0.97      36.8    0.7029 (5/5 folds)
both_classes         200 0.99      65.4    0.6911 (5/5 folds)
both_classes          25 0.70       0.4    0.6986 (5/5 folds)
both_classes          25 0.80       1.2    0.6936 (5/5 folds)
both_classes          25 0.90       2.6    0.6919 (5/5 folds)
both_classes          25 0.95       4.6    0.6932 (5/5 folds)
both_classes          25 0.97       5.4      0.69 (5/5 folds)
both_classes          25 0.99       9.4    0.6875 (5/5 folds)
both_classes          50 0.70       0.4     0.699 (5/5 folds)
both_classes          50 0.80       1.4     0.695 (5/5 folds)
both_classes          50 0.90       3.2    0.6908 (5/5 folds)
both_classes          50 0.95       6.8    0.6935 (5/5 folds)
both_classes          50 0.97       8.0    0.6871 (5/5 folds)
both_classes          50 0.99      14.4    0.6859 (5/5 folds)
minority_only        100 0.70       0.4    0.6968 (5/5 folds)
minority_only        100 0.80       1.0    0.6973 (5/5 folds)
minority_only        100 0.90       2.4    0.6782 (5/5 folds)
minority_only        100 0.95       4.4    0.6763 (5/5 folds)
minority_only        100 0.97       6.0    0.6676 (5/5 folds)
minority_only        100 0.99       8.6    0.6739 (5/5 folds)
minority_only        200 0.70       0.6    0.7005 (5/5 folds)
minority_only        200 0.80       1.6    0.6931 (5/5 folds)
minority_only        200 0.90       3.4    0.6926 (5/5 folds)
minority_only        200 0.95       7.4    0.6951 (5/5 folds)
minority_only        200 0.97       9.2    0.6998 (5/5 folds)
minority_only        200 0.99      16.2    0.7038 (5/5 folds)
minority_only         25 0.70       0.0    0.6982 (5/5 folds)
minority_only         25 0.80       0.0    0.6982 (5/5 folds)
minority_only         25 0.90       0.4    0.6937 (5/5 folds)
minority_only         25 0.95       0.8    0.6937 (5/5 folds)
minority_only         25 0.97       1.2    0.6937 (5/5 folds)
minority_only         25 0.99       1.6    0.6932 (5/5 folds)
minority_only         50 0.70       0.0    0.6982 (5/5 folds)
minority_only         50 0.80       0.6    0.7004 (5/5 folds)
minority_only         50 0.90       1.0     0.695 (5/5 folds)
minority_only         50 0.95       2.2     0.696 (5/5 folds)
minority_only         50 0.97       2.4    0.6942 (5/5 folds)
minority_only         50 0.99       4.0    0.6841 (5/5 folds)
nonresponder_only    100 0.70       0.2    0.6986 (5/5 folds)
nonresponder_only    100 0.80       0.8    0.6986 (5/5 folds)
nonresponder_only    100 0.90       4.8    0.7016 (5/5 folds)
nonresponder_only    100 0.95       8.6    0.6964 (5/5 folds)
nonresponder_only    100 0.97      12.4    0.6914 (5/5 folds)
nonresponder_only    100 0.99      23.8    0.6897 (5/5 folds)
nonresponder_only    200 0.70       1.4     0.696 (5/5 folds)
nonresponder_only    200 0.80       3.0     0.696 (5/5 folds)
nonresponder_only    200 0.90       9.8    0.6948 (5/5 folds)
nonresponder_only    200 0.95      20.8    0.6938 (5/5 folds)
nonresponder_only    200 0.97      26.0    0.6959 (5/5 folds)
nonresponder_only    200 0.99      46.4      0.69 (5/5 folds)
nonresponder_only     25 0.70       0.2    0.6982 (5/5 folds)
nonresponder_only     25 0.80       0.2    0.6982 (5/5 folds)
nonresponder_only     25 0.90       0.8    0.6986 (5/5 folds)
nonresponder_only     25 0.95       2.2       0.7 (5/5 folds)
nonresponder_only     25 0.97       3.4    0.6977 (5/5 folds)
nonresponder_only     25 0.99       6.0    0.6945 (5/5 folds)
nonresponder_only     50 0.70       0.6    0.7011 (5/5 folds)
nonresponder_only     50 0.80       1.4     0.698 (5/5 folds)
nonresponder_only     50 0.90       3.2    0.6971 (5/5 folds)
nonresponder_only     50 0.95       6.4    0.6957 (5/5 folds)
nonresponder_only     50 0.97       8.0    0.7002 (5/5 folds)
nonresponder_only     50 0.99      14.8    0.6954 (5/5 folds)
```

Top-3 cell ROC-AUC (mean over folds): (1) `both_classes` ratio 1.0, q0.97 — kept 16.2 — **0.7064**; (2) `both_classes` ratio 1.0, q0.95 — kept 12.8 — **0.7055**; (3) `minority_only` ratio 2.0, q0.99 — kept 16.2 — **0.7038**. So với real_only của A2 (0.6982 ± 0.0359): cell tốt nhất hơn +0.0082, nhỏ hơn 1 sd (0.0359). Retune filter không tạo khác biệt có ý nghĩa; cell q0.95 `both_classes` r1 trùng đúng hàng filtered mặc định của bảng A2 (0.7055).

### 3d. A4 TSTR

Aggregate A4 (nguyên văn từ `output/logs/a4_r32.log`):

```
=== TSTR AGGREGATE (mean over folds) ===
mode               ratio  n_train     ROC      PR      BA
REAL_ONLY              -      198  0.6982  0.8647  0.6298
both_classes         100      198  0.5485  0.8032  0.4752
both_classes         200      395  0.5586  0.7978  0.5241
both_classes          25       49  0.5419  0.8027  0.5091
both_classes          50       99  0.6324  0.8366  0.5946
```

TSTR: mọi nhánh augmentation đều thấp hơn REAL_ONLY (0.6982); nhánh tốt nhất `both_classes` ratio 0.5 đạt ROC 0.6324 (n_train 99).

## 4. Kết luận trung thực
- A2 (full): real_only ROC 0.6982 ± 0.0359 (sd). Nhánh tốt nhất `both_classes` r1 (filtered) 0.7055 ± 0.0461, chênh +0.0073 — nhỏ hơn 1 sd. Nhánh unfiltered tốt nhất `both_classes` r0.25 0.7045, chênh +0.0063. **Không nhánh nào vượt real_only quá 1 sd.**
- A5 (PCA 32 full): real_only 0.7075 ± 0.0271. Nhánh tốt nhất `minority_only` r1 (filtered) 0.7113 ± 0.0312, chênh +0.0038 — nhỏ hơn 1 sd. **Không nhánh nào vượt real_only quá 1 sd.**
- A3 (filter retune trên A2): cell tốt nhất 0.7064, chênh +0.0082 so với real_only A2 — nhỏ hơn 1 sd.
- A4 (TSTR trên A2): augmentation làm giảm ROC so với REAL_ONLY (tốt nhất 0.6324 so với 0.6982).
- Kết luận: DDPM latent augmentation trên r32 là trung tính đến tiêu cực — **không branch nào vượt real_only của cùng run quá 1 sd**. Không so sánh số DDPM giữa r32 và 34 chiều vì real-only baseline khác nhau; mọi so sánh ở đây chỉ trong cùng một run (augmented so với real_only).

## 5. Thay đổi so với package 34 chiều
Chỉ so sánh số GVAE pooled-OOF: cùng protocol, cùng config, chỉ khác graph (so sánh 5 seed đầy đủ ở `canonical_r32_report.md`). Số DDPM **không** được so giữa hai graph: real-only baseline và checkpoint GVAE nguồn khác nhau, nên mọi so sánh DDPM chỉ có nghĩa trong cùng một run (§4).

| Mục | 34 chiều (2026-08-10) | r32 (2026-10-03) |
|---|---|---|
| OOF head ROC (seed 42) | 0.6065 | 0.6431 |
| OOF probe ROC (seed 42) | 0.6519 | 0.6495 |
| DDPM: có nhánh nào vượt real_only của cùng run quá 1 sd? | Không | Không |
