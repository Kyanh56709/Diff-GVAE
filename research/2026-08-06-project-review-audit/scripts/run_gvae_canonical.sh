#!/bin/bash
# Canonical GVAE run chain (part 1) — main worktree, persistent background.
set -e
cd /Users/admin/Diff-GVAE
PY=.venv/bin/python
OUT=research/2026-08-06-project-review-audit/output/raw
LOG=$OUT/canonical_gvae.log
mkdir -p "$OUT"
echo "=== $(date) data_validation ===" >> "$LOG"
$PY review_fixes_2026_07/data_validation.py data_ln_pc_ihc_g.pt >> "$LOG" 2>&1
echo "=== $(date) GVAE canonical run (epochs 80, pretrain 80, 5-fold, latent_quality, top-k 3, seed 0) ===" >> "$LOG"
$PY outputs/gvae/train_gvae_runner.py \
  --epochs 80 --pretrain-epochs 80 --n-splits 5 \
  --checkpoint-metric latent_quality --early-stopping-metric latent_quality \
  --top-k 3 --seed 0 --run-prefix canonical >> "$LOG" 2>&1
echo "=== $(date) DONE ===" >> "$LOG"
# Record the run_id (latest metrics dir)
ls -t outputs/gvae/metrics | head -1 > "$OUT/canonical_run_id.txt"
echo "RUN_ID: $(cat "$OUT/canonical_run_id.txt")" >> "$LOG"
