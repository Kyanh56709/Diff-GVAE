#!/bin/bash
# Pipeline phase 3b: DDPM augmentation only (rank-1 checkpoints of the ranked best-params run).
set -e
cd /Users/admin/Diff-GVAE
PY=.venv/bin/python
OUT=research/2026-08-06-project-review-audit/output/raw
LOG=$OUT/pipeline_phase3.log
RUN_ID=$(cat "$OUT/bestparam_ranked_run_id.txt")
mkdir -p "$OUT"

echo "=== $(date) phase3b: DDPM augmentation on $RUN_ID (CPU) ===" >> "$LOG"
$PY outputs/gvae/train_conditional_ddpm_augmentation_runner.py \
  --data-path data_ln_pc_ihc_g.pt \
  --gvae-run-id "$RUN_ID" \
  --checkpoint-root outputs/gvae/checkpoints \
  --checkpoint-selector rank --rank 1 \
  --max-folds 5 \
  --latent-key concat_mu \
  --augmentation-modes minority_only,nonresponder_only,both_classes \
  --ratios 0.25,0.5,1.0,2.0 \
  --filter-synthetic --filter-quantile 0.95 \
  --epochs 120 --timesteps 250 --seed 42 --device cpu \
  >> "$LOG" 2>&1

echo "=== $(date) PHASE3 DONE ===" >> "$LOG"
