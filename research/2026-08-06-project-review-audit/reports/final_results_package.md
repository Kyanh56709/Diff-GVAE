# FINAL RESULTS PACKAGE — Diff-GVAE (2026-08-10)

Gói số liệu đã verify, sẵn sàng đưa vào FINAL_REPORT.md. Mọi số trỏ tới artifact (claim-verifier: Final verdict PASS, 9/9).

## 1. Dữ liệu & convention
- Data: `data_ln_pc_ihc_g.pt` — 247 NSCLC PD-(L)1 (nguồn Vanguri 2022 Nat Cancer, MSK-MIND), 333 lesions.
- Labels: `binary_label` 1 = **non-responder** (185), 0 = **responder** (62). Positive class trong mọi metric = label 1; khi report theo "responder" phải flip label (1-y) cho PR-AUC/F1/sens/spec.
- 3 views: clinical 22 (5 cont + 17 binary), pathology 15 (GLCM), radiology 34 (32 lesion radiomics + 2 index artifact, attention-aggregated); edges cosine **> 0.8**/view + has_lesion (ngưỡng thật đã xác minh 2026-10-03 bởi A1 rebuild; bản ghi "0.7" trước đây là sai — xem `research/2026-10-03-a1-build-script/`).

## 2. Kết quả GVAE

### 2a. Hyperparameter sweep (10/10 trial, 3-fold, epochs 100, CPU)
- Grid: lr [1e-3, 5e-4] × cross_cl [0.1, 0.2, 0.5] × d_embed [16, 32, 64] × hidden [32, 64, 128] × attn [32, 64] × heads [4, 8] (216 combos, random 10).
- **Best: lr 1e-3, cross_cl 0.5, d_embed 32, hidden 128, attn 32, heads 8 → mean_auc 0.7208 ± 0.0769** (mean_f1 0.7580).
- Artifact: `research/2026-08-06-project-review-audit/output/raw/gvae_sweep_results_v2.csv`, `best_params.json`.

### 2b. Runs so sánh (val-fold mean, 5-fold)
| Run | ROC-AUC | PR-AUC | BA | F1 |
|---|---|---|---|---|
| Canonical baseline (epochs 80, seed 0) | 0.6250 | 0.8235 | 0.6772 | 0.6985 |
| Best-params (epochs 80, seed 42) | 0.6548 | 0.8357 | 0.7081 | 0.8186 |
| **Ranked + v2 flags** (logvar_clamp, zero-lesion, top_k=3) | **0.6789** | 0.8641 | 0.7036 | 0.7851 |

### 2c. Pooled-OOF + bootstrap CI (leakage-free; SỐ CUỐI)
| Estimator | ROC-AUC [95% CI] | PR-AUC [95% CI] | BA [95% CI] | F1 [95% CI] |
|---|---|---|---|---|
| **Fusion head** | **0.6065 [0.5244–0.6889]** | 0.8106 [0.7483–0.8749] | 0.6015 [0.5350–0.6722] | 0.7845 [0.7374–0.8302] |
| **Linear probe (frozen mu)** | **0.6519 [0.5710–0.7301]** | 0.8259 [0.7594–0.8900] | 0.6122 [0.5421–0.6797] | 0.7657 [0.7169–0.8108] |

## 3. DDPM latent augmentation (downstream trên concat_mu, run ...20260809_184454)
- DDPM conditional, concat_mu 96-dim, timesteps 250, epochs 120, seed 42; DDPM chỉ generator (ddpm_is_classifier=False).
- Modes: minority_only (=responder, class 0), nonresponder_only (class 1), both_classes × ratios 0.25/0.5/1.0/2.0.
| Branch | ROC-AUC | PR-AUC | BA | synthetic TB |
|---|---|---|---|---|
| Real only | 0.7104 | 0.8694 | 0.7202 | 0 |
| both_classes r0.25 (unfiltered) | **0.7115** | 0.8754 | 0.7166 | 49 |
| minority_only r1.0 (unfiltered) | 0.7060 | 0.8717 | **0.7287** | 49.6 |
| nonresponder_only r0.25 (unfiltered) | 0.7115 | 0.8734 | 0.7241 | 37 |
| TẤT CẢ filtered (quantile 0.95) | = real only | | | **0** (filter quá chặt) |

Kết luận trung thực: augmentation ≈ real-only (chênh trong nhiễu CI); filtered giữ 0 mẫu → cần retune quantile nếu dùng filter.

## 4. Reproduction: số cũ (FINAL_REPORT §6.2) vs mới (OOF)
| Metric | Cũ | Mới (OOF head) | Delta |
|---|---|---|---|
| ROC-AUC | 0.6894 | 0.6065 | -12.02% |
| PR-AUC | 0.8523 | 0.8106 | -4.89% |
| BA | 0.7265 | 0.6015 | -17.20% |
| F1 | 0.8101 | 0.7845 | -3.16% |

Lý do chênh: protocol cũ = val-fold mean + threshold tối ưu; protocol mới = pooled-OOF + inner-val selection (trung thực hơn). Artifact số cũ đã mất (không tái lập được).

## 5. Figures
- `research/2026-08-06-project-review-audit/output/figures/oof_roc_pr.png` (ROC/PR pooled OOF: head vs probe)
- `research/2026-08-06-project-review-audit/output/figures/auc_comparison.png` (5 mức: GVAE direct / OOF head / OOF probe / downstream real / +aug)

## 6. Dataset external (science-reviewer đã review, WARN → fixed)
- **I3LUNG** (Zenodo 17535424): NSCLC tri-modal duy nhất công khai — 2.075 BN (391 tri-modal), clinical+pyradiomics+pathology FM embeddings+genomics, RECIST 4-class, CC-BY-NC-4.0. S-tier.
- A-tier ngoài NSCLC: IMvigor210 (urothelial 348), IMmotion150 (RCC 263), Gide 2019 (melanoma 91). NSCLC clinical-only: Hellmann CheckMate 012 (75).
- Loại trừ (cùng cohort project): cBioPortal lung_msk_mind_2020 + Synapse syn26642505.
- Chi tiết: `research/2026-08-06-external-datasets/reports/report.md`.

## 7. Còn mở (theo dõi)
1. Retune `filter_quantile` (0.95 → 0.7–0.9) — filter hiện giữ 0 mẫu.
2. Seed sweep đầy đủ (3–5 seeds) cho val-fold numbers.
3. Nhánh PCA (--pca-components 32) chưa chạy.
4. MPS migration (native scatter + float64) — backlog, không bắt buộc (CPU 13 phút/run).
5. External validation I3LUNG (adapter HeteroData + pipeline) — giá trị cao cho báo cáo.
