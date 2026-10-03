#!/bin/bash
# Canonical switch to the 32-slot radiology graph (data_ln_pc_ihc_g_r32.pt), GVAE part.
#  1. canonical GVAE run (same flags as research/2026-08-06-project-review-audit/scripts/run_gvae_canonical.sh)
#  2. pooled-OOF seeds 45/46 for r32 and the old 34-slot graph (seeds 42-44 already in
#     research/2026-10-03-radiology-artifact-ablation/output/) -> 5-seed comparison.
set -e
cd "$(dirname "$0")/../../.."
D=research/2026-10-03-canonical-r32
A=research/2026-10-03-radiology-artifact-ablation
PY=.venv/bin/python
$PY review_fixes_2026_07/data_validation.py data_ln_pc_ihc_g_r32.pt > $D/output/logs/data_validation.log 2>&1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
(
  $PY outputs/gvae/train_gvae_runner.py --data-path data_ln_pc_ihc_g_r32.pt \
    --epochs 80 --pretrain-epochs 80 --n-splits 5 \
    --checkpoint-metric latent_quality --early-stopping-metric latent_quality \
    --top-k 3 --seed 0 --device cpu --run-prefix canonical_r32 > $D/output/logs/canonical_gvae.log 2>&1
  echo "finished canonical GVAE: $(grep RUN_ID= $D/output/logs/canonical_gvae.log)"
) &
for seed in 45 46; do
  for v in "drop_both32:data_ln_pc_ihc_g_r32.pt" "full34:data_ln_pc_ihc_g.pt"; do
    echo "${v%%:*} ${v#*:} $seed"
  done
done | xargs -P 2 -L 1 sh -c "$PY $A/scripts/run_oof.py --tag \$0 --data \$1 --seed \$2 > $A/output/logs/\$0_seed\$2.log 2>&1; echo \"finished \$0 seed \$2: \$(tail -1 $A/output/logs/\$0_seed\$2.log)\""
wait
echo ALL_DONE
