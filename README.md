# Diff-GVAE

## Data provenance (READ FIRST)

- **`data_ln_pc_ihc_g_r32.pt` is the sole canonical data file** (since
  2026-10-03) used by every maintained runner. 247 patients, `x_clinical`
  dim 22 (cols 0-4 continuous, 5-21 binary), lesion `x` dim 32 (radiomics
  only), class counts `{0: 62, 1: 185}`. Build it from the raw CSVs with
  `python data/build_ln_pc_ihc_g.py --out data_ln_pc_ihc_g_r32.pt --drop-radiology-artifacts both`.
- `data_ln_pc_ihc_g.pt` (lesion dim 34) is the **previous** canonical graph,
  kept only to reproduce results frozen before 2026-10-03. Its two extra
  radiology slots are index artifacts (lesion file-order rank, `lesion_index`)
  that correlate with the label; dropping them does not lower GVAE AUC
  (`research/2026-10-03-radiology-artifact-ablation/`). `python
  data/build_ln_pc_ihc_g.py --verify data_ln_pc_ihc_g.pt` rebuilds it bit-exact.
- The raw CSVs under `data/` are git-ignored (source: MSK-MIND / Vanguri et al.
  2022); they must be obtained separately to run the build script.
- The old `data/data_247.pt` had **inverted `binary_label`** and 64 clinical
  columns; it has been moved to `deprecated/data_247.pt`. Do not train on it.
- **Positive-class meaning (confirmed 2026-08-06):** `binary_label = 1` =
  **non-responder (185/247)**; `binary_label = 0` = **responder (62/247)**.
  Metrics are computed with class 1 as positive; for responder-framed
  reporting of PR-AUC/F1/sensitivity/specificity, flip labels (1-y) or
  reinterpret.
- Validate any graph before training: `python review_fixes_2026_07/data_validation.py data_ln_pc_ihc_g_r32.pt`.
