#!/bin/bash
# Pipeline phase 3: best-params GVAE ranked rerun -> DDPM augmentation (CPU).
set -e
cd /Users/admin/Diff-GVAE
PY=.venv/bin/python
OUT=research/2026-08-06-project-review-audit/output/raw
LOG=$OUT/pipeline_phase3.log
mkdir -p "$OUT"

echo "=== $(date) phase3 start; waiting for phase-2 DONE ===" >> "$LOG"
for i in $(seq 1 900); do
  if grep -q "PHASE2 DONE" "$OUT/pipeline_phase2.log" 2>/dev/null; then break; fi
  sleep 60
done

# 1. Ranked rerun of best-params GVAE (top_k=3 so rank checkpoints exist for DDPM)
echo "=== $(date) best-params ranked rerun (top_k=3) ===" >> "$LOG"
$PY - <<'EOF' >> "$LOG" 2>&1
import torch, json, time
from training.sweep_gvae import update_configs_with_params
from training.train_gvae import kfold_train_gvae

data = torch.load('data_ln_pc_ihc_g.pt', weights_only=False)
DIM_CLINICAL = data['patient'].x_clinical.shape[1]
DIM_PATHOLOGY = data['patient'].x_pathology.shape[1]
DIM_RADIOLOGY = data['lesion'].x.shape[1]
best = json.load(open('research/2026-08-06-project-review-audit/output/raw/best_params.json'))
# Cast dims/heads back to int (JSON round-trip -> floats -> nn.Linear TypeError).
for _k in ('d_embed', 'hidden_channels_vae', 'attention_hidden_dim', 'heads'):
    best[_k] = int(best[_k])

train_config = {
    'data_path': 'data_ln_pc_ihc_g.pt',
    'device': torch.device('cpu'),
    'n_splits': 5, 'epochs': 80, 'pretrain_epochs': 100,
    'patience': 30, 'patience_early_stopping': 30,
    'lr': 0.001, 'wd': 1e-4, 'batch_size': 64,
    'loss_weights': {'class': 1.0, 'cross_cl': 0.2,
                     'rec_attr': {'clinical': 1.0, 'pathology': 1.0, 'radiology': 1.0},
                     'rec_struct': 0.1, 'kl': 0.00001},
    'annealing': {'kl': {'start_weight': 0.0, 'end_weight': 0.00001, 'start_epoch': 20, 'end_epoch': 300},
                  'cross_cl': {'start_weight': 0.1, 'end_weight': 0.1, 'start_epoch': 20, 'end_epoch': 100}},
    'pca_config': {'clinical': 16, 'pathology': 8},
    'lesion_pca_config': {'n_components': 15},
    'cross_cl_temp': 0.1, 'grad_clip_norm': 1.0, 'print_every_k_epochs': 10,
    'random_seed': 42, 'vectorized_contrastive': True,
    'checkpoint_metric': 'latent_quality', 'early_stopping_metric': 'latent_quality',
    'top_k_gvae_checkpoints': 3,
    'save_best_fold_model': True,
    'checkpoint_dir': 'outputs/gvae/checkpoints/bestparam_ranked',
    'metrics_dir': 'outputs/gvae/metrics/bestparam_ranked',
}
model_config = {
    'view_configs': {
        'clinical': {'in_channels': DIM_CLINICAL, 'hidden_channels_vae': 64, 'heads': 8, 'dropout': 0.3, 'num_gnn_layers': 2, 'edge_dim': 1},
        'pathology': {'in_channels': DIM_PATHOLOGY, 'hidden_channels_vae': 64, 'heads': 8, 'dropout': 0.3, 'num_gnn_layers': 2, 'edge_dim': 1},
        'radiology': {'in_channels': 32, 'hidden_channels_vae': 64, 'heads': 8, 'dropout': 0.3, 'num_gnn_layers': 2, 'edge_dim': 1},
    },
    'radiology_aggregator_config': {'lesion_feature_dim': DIM_RADIOLOGY, 'aggregated_output_dim': 32,
                                    'attention_hidden_dim': 64, 'dropout': 0.3},
    'fusion_config': {'fused_dim': 32, 'num_fusion_heads': 8, 'fusion_ffn_multiplier': 5},
    'classifier_config': {'classifier_hidden_dim': 32},
    'projection_head_config': {'hidden_dim': 32, 'output_dim': 32, 'dropout': 0.3},
    'd_embed': 32, 'missing_strategy': 'learnable',
    'logvar_clamp': (-4.0, 2.0), 'radiology_zero_lesion_passthrough': True,
}
mc, tc = update_configs_with_params(model_config, train_config, best)
run_id = 'gvae_bestparam_ranked_' + time.strftime('%Y%m%d_%H%M%S')
tc['checkpoint_dir'] = f'outputs/gvae/checkpoints/{run_id}'
tc['metrics_dir'] = f'outputs/gvae/metrics/{run_id}'
print('FINAL_CONFIG:', json.dumps({'params': best}, indent=2))
summary, df_results, roc_data = kfold_train_gvae(data, mc, tc)
print('SUMMARY:', json.dumps(summary, indent=2))
print('RUN_ID:', run_id)
json.dump({'run_id': run_id, 'best_params': best, 'summary': summary},
          open(f'outputs/gvae/metrics/{run_id}/bestparam_meta.json', 'w'), indent=2)
open('research/2026-08-06-project-review-audit/output/raw/bestparam_ranked_run_id.txt', 'w').write(run_id)
EOF

# 2. DDPM augmentation from rank-1 checkpoints of the ranked best-params run
RUN_ID=$(cat "$OUT/bestparam_ranked_run_id.txt")
echo "=== $(date) DDPM augmentation on $RUN_ID (CPU) ===" >> "$LOG"
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
