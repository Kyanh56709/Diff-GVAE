# Diff-GVAE

## Data provenance (READ FIRST)

- **`data_ln_pc_ihc_g.pt` is the sole canonical data file** used by
  `training_local.ipynb` and every maintained runner. 247 patients,
  `x_clinical` dim 22 (cols 0-4 continuous, 5-21 binary), class counts
  `{0: 62, 1: 185}`.
- The old `data/data_247.pt` had **inverted `binary_label`** and 64 clinical
  columns; it has been moved to `deprecated/data_247.pt`. Do not train on it.
- **Positive-class meaning (confirmed 2026-08-06):** `binary_label = 1` =
  **non-responder (185/247)**; `binary_label = 0` = **responder (62/247)**.
  Metrics are computed with class 1 as positive; for responder-framed
  reporting of PR-AUC/F1/sensitivity/specificity, flip labels (1-y) or
  reinterpret.
- Validate any graph before training: `python review_fixes_2026_07/data_validation.py data_ln_pc_ihc_g.pt`.
