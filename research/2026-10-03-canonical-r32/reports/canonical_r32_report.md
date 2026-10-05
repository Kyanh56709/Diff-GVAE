# Đổi canonical sang graph 32 chiều — phần GVAE (2026-10-03)

**Quyết định (owner, 2026-10-03):** canonical = `data_ln_pc_ihc_g_r32.pt`: graph cũ bỏ 2 slot index radiology (file-rank của lesion, `lesion_index`). Làm từng bước: GVAE trước, DDPM sau. File 34 chiều `data_ln_pc_ihc_g.pt` được giữ nguyên để tái lập các kết quả đã chốt trước ngày này.

## Build

```bash
.venv/bin/python data/build_ln_pc_ihc_g.py --out data_ln_pc_ihc_g_r32.pt --drop-radiology-artifacts both
.venv/bin/python review_fixes_2026_07/data_validation.py data_ln_pc_ihc_g_r32.pt   # RESULT: PASS
```

So với bản 34 chiều: clinical, pathology, nhãn, mask, các edge clinical/pathology và `has_lesion` giữ nguyên (test `tests/test_build_ln_pc_ihc_g.py`). Chỉ có `lesion.x` (333×32) và `similar_to_radiology` (652 → 720 edge) thay đổi.

## Kết quả

### Pooled-OOF, 5 seed (config `pipeline_phase4a_oof.sh`; scripts ở `research/2026-10-03-radiology-artifact-ablation/`)

| Score | Graph | 42 | 43 | 44 | 45 | 46 | mean ± sd | AUPRC |
|---|---|---|---|---|---|---|---|---|
| head | 34 (cũ) | 0.6076 | 0.6476 | 0.6488 | 0.6357 | 0.6618 | 0.640 ± 0.021 | 0.821 |
| head | **32 (mới)** | 0.6431 | 0.6678 | 0.6533 | 0.6681 | 0.6852 | **0.664 ± 0.016** | 0.830 |
| probe | 34 (cũ) | 0.6325 | 0.6394 | 0.6555 | 0.6225 | 0.6341 | 0.637 ± 0.012 | 0.814 |
| probe | 32 (mới) | 0.6495 | 0.6119 | 0.6670 | 0.6370 | 0.6125 | 0.636 ± 0.024 | 0.808 |

- Head: 32 > 34 ở **5/5 seed** (ΔAUC +0.004 → +0.036, mean +0.023). Paired CI từng seed đều chứa 0. Sign test 5/5: p = 0.0625 (hai phía) → **không giảm, có xu hướng tăng, chưa có ý nghĩa thống kê**.
- Probe: không đổi (0.637 so với 0.636).

### Run GVAE canonical (cùng flag với `run_gvae_canonical.sh`: 80 ep, pretrain 80, 5-fold, latent_quality, top-k 3, seed 0)

| Run | Graph | mean AUC (fold) | std | PR-AUC | bal. acc | Brier |
|---|---|---|---|---|---|---|
| `canonical_20260806_162328` | 34 | 0.625 | 0.074 | 0.824 | 0.677 | 0.225 |
| `canonical_r32_20261003_111618` | 32 | 0.617 | 0.134 | 0.800 | 0.677 | 0.219 |

Chênh lệch nằm trong nhiễu giữa các fold (std 0.07–0.13, chỉ 1 seed). Run này dùng để lấy checkpoint (rank 1–3) cho DDPM; con số báo cáo chính vẫn nên là pooled-OOF nhiều seed ở trên.

## Đã chuyển sang graph mới

`configs/config.py` (`CANONICAL_GRAPH`, `LEGACY_GRAPH_34`), `outputs/gvae/train_gvae_runner.py` (flag mới `--data-path`, mặc định là r32; ghi `data_path` vào `run_config.json`), `outputs/gvae/train_conditional_ddpm_augmentation_runner.py` (mặc định `--data-path`), `run_gvae.py`, `run_sweep.py`, `review_fixes_2026_07/raw_data_audit.py`, `README.md`, `CLAUDE.md`, `tests/test_docs_canonical_data.py`.

Cố ý **không** đổi: các script/report trong `research/2026-08-06-*` và `research/2026-10-03-a1-build-script/` (ghi chép lại những gì đã chạy trên graph 34 chiều), `deprecated/`.

## Còn mở

1. ~~**DDPM** (bước tiếp theo): chạy lại augmentation trên checkpoint `canonical_r32_20261003_111618` (rank 1). Run DDPM cũ dùng run `gvae_bestparam_ranked_*` với `best_params.json` tune trên graph 34 chiều → cần quyết định: tune lại hay dùng lại best_params.~~ **Xong 2026-10-03:** dùng lại `best_params.json`, chạy GVAE bestparam ranked trên r32 + DDPM A2/A5/A3/A4; số liệu tại `research/2026-10-03-canonical-r32/reports/final_results_package_r32.md`.
2. `training_local.ipynb` (notebook của owner, không được git theo dõi) vẫn load `data_ln_pc_ihc_g.pt` ở 5 chỗ.
3. ~~Các số liệu đã chốt trong `final_results_package.md`, `FINAL_REPORT.md` và manuscript vẫn là số của graph 34 chiều. Cập nhật sau khi DDPM xong (A7).~~ **Xong 2026-10-03:** đã re-freeze A7 trên canonical r32. **A6 đã đóng 2026-10-05:** số 0.81–0.87 của notebook được truy vết là protocol val-fold trên graph 34 chiều (`gvae_sweep_results.csv`, mean_auc 0.8116), không tái lập được bằng code HEAD — xem `research/2026-10-05-a6-notebook-reconciliation/reports/a6_reconciliation.md`; số chốt mới tại `research/2026-10-03-canonical-r32/reports/final_results_package_r32.md`.
4. Tái lập: pipeline không deterministic hoàn toàn khi đổi số thread (seed 42 / 34 chiều: 0.6076 so với 0.6065 trước đây).
