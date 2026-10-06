#!/bin/bash
# DDPM chain on canonical r32: A2 full rerun, A5 PCA-32 full rerun, then A3/A4 post-hoc on A2.
# Config = the 2026-08-10 frozen config (ddpm_diagnosis_20260810.md), only graph + GVAE run differ.
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=research/2026-10-03-canonical-r32
mkdir -p "$D/output/logs"
PY=.venv/bin/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
GVAE_RUN=$(cat $D/output/bestparam_ranked_r32_run_id.txt)
COMMON=(--data-path data_ln_pc_ihc_g_r32.pt --gvae-run-id "$GVAE_RUN"
        --checkpoint-root outputs/gvae/checkpoints --checkpoint-selector rank --rank 1 --max-folds 5
        --latent-key concat_mu --augmentation-modes minority_only,nonresponder_only,both_classes
        --ratios 0.25,0.5,1.0,2.0 --filter-synthetic --filter-quantile 0.95
        --epochs 600 --timesteps 250 --guidance-scale 1.0 --seed 42 --device cpu)

echo "=== $(date) A2 full (GVAE $GVAE_RUN) ==="
$PY outputs/gvae/train_conditional_ddpm_augmentation_runner.py "${COMMON[@]}" > $D/output/logs/ddpm_a2_r32.log 2>&1
grep '^RUN_ID=' $D/output/logs/ddpm_a2_r32.log | cut -d= -f2 > $D/output/ddpm_a2_r32_run_id.txt

echo "=== $(date) A5 PCA 32 full ==="
$PY outputs/gvae/train_conditional_ddpm_augmentation_runner.py "${COMMON[@]}" --pca-components 32 > $D/output/logs/ddpm_a5_r32.log 2>&1
grep '^RUN_ID=' $D/output/logs/ddpm_a5_r32.log | cut -d= -f2 > $D/output/ddpm_a5_r32_run_id.txt

A2=$(cat $D/output/ddpm_a2_r32_run_id.txt)
echo "=== $(date) A3 filter retune on $A2 ==="
$PY research/2026-08-06-project-review-audit/scripts/a3_filter_quantile_retune.py --run-id "$A2" > $D/output/logs/a3_r32.log 2>&1
echo "=== $(date) A4 TSTR on $A2 ==="
$PY research/2026-08-06-project-review-audit/scripts/a4_tstr_control.py --run-id "$A2" > $D/output/logs/a4_r32.log 2>&1
echo "=== $(date) DDPM_R32_DONE ==="
