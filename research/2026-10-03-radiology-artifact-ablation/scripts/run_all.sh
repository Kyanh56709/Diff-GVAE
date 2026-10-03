#!/bin/bash
# 3 graph variants x 3 seeds, 3 jobs at a time (CPU). Logs in output/logs/.
cd "$(dirname "$0")/../../.."
D=research/2026-10-03-radiology-artifact-ablation
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for seed in 42 43 44; do
  for v in "full34:data_ln_pc_ihc_g.pt" "drop_rank33:$D/data/graph_drop_file_rank.pt" "drop_both32:$D/data/graph_drop_both.pt"; do
    echo "${v%%:*} ${v#*:} $seed"
  done
done | xargs -P 3 -L 1 sh -c '.venv/bin/python '"$D"'/scripts/run_oof.py --tag $0 --data $1 --seed $2 > '"$D"'/output/logs/$0_seed$2.log 2>&1; echo "finished $0 seed $2: $(tail -1 '"$D"'/output/logs/$0_seed$2.log)"'
echo ALL_DONE
