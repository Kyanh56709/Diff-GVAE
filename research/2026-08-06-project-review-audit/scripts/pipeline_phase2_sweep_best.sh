#!/bin/bash
# Pipeline phase 2: hyperparameter sweep -> best params -> full GVAE run (main worktree, persistent).
set -e
cd /Users/admin/Diff-GVAE
PY=.venv/bin/python
OUT=research/2026-08-06-project-review-audit/output/raw
LOG=$OUT/pipeline_phase2.log
mkdir -p "$OUT"

# 0. Wait for phase-1 (canonical baseline GVAE) to finish — marker in canonical_gvae.log
echo "=== $(date) phase2 start; waiting for phase-1 DONE marker ===" >> "$LOG"
for i in $(seq 1 720); do
  if grep -q "=== .* DONE ===" "$OUT/canonical_gvae.log" 2>/dev/null; then
    echo "=== $(date) phase-1 done ===" >> "$LOG"; break
  fi
  sleep 60
done

# 1. GVAE hyperparameter sweep (random search, resumable CSV) — CPU forced (torch_scatter has no MPS support)
echo "=== $(date) GVAE sweep: 10 random trials, 3-fold, 100 epochs (CPU) ===" >> "$LOG"
$PY - <<'EOF' >> "$LOG" 2>&1
import torch, json
from training.sweep_gvae import run_gvae_sweep

data = torch.load('data_ln_pc_ihc_g.pt', weights_only=False)
DIM_CLINICAL = data['patient'].x_clinical.shape[1]
DIM_PATHOLOGY = data['patient'].x_pathology.shape[1]
DIM_RADIOLOGY = data['lesion'].x.shape[1]

train_config = {
    'data_path': 'data_ln_pc_ihc_g.pt',
    'device': torch.device('cpu'),  # CPU forced: torch_scatter crashes on MPS (src must be CPU tensor)
    'n_splits': 3, 'epochs': 100, 'pretrain_epochs': 100,
    'patience': 30, 'patience_early_stopping': 30,
    'lr': 0.001, 'wd': 1e-4, 'batch_size': 32,
    'loss_weights': {'class': 1.0, 'cross_cl': 0.2,
                     'rec_attr': {'clinical': 1.0, 'pathology': 1.0, 'radiology': 1.0},
                     'rec_struct': 0.1, 'kl': 0.00001},
    'annealing': {'kl': {'start_weight': 0.0, 'end_weight': 0.00001, 'start_epoch': 20, 'end_epoch': 300},
                  'cross_cl': {'start_weight': 0.1, 'end_weight': 0.1, 'start_epoch': 20, 'end_epoch': 100}},
    'pca_config': {'clinical': 16, 'pathology': 8},
    'lesion_pca_config': {'n_components': 15},
    'cross_cl_temp': 0.1, 'grad_clip_norm': 1.0, 'print_every_k_epochs': 50,
    'random_seed': 420, 'vectorized_contrastive': True,
}
model_config = {
    'view_configs': {
        'clinical': {'in_channels': DIM_CLINICAL, 'hidden_channels_vae': 64, 'heads': 8, 'dropout': 0.5, 'num_gnn_layers': 2, 'edge_dim': 1},
        'pathology': {'in_channels': DIM_PATHOLOGY, 'hidden_channels_vae': 64, 'heads': 8, 'dropout': 0.5, 'num_gnn_layers': 2, 'edge_dim': 1},
        'radiology': {'in_channels': 32, 'hidden_channels_vae': 64, 'heads': 8, 'dropout': 0.5, 'num_gnn_layers': 2, 'edge_dim': 1},
    },
    'radiology_aggregator_config': {'lesion_feature_dim': DIM_RADIOLOGY, 'aggregated_output_dim': 32,
                                    'attention_hidden_dim': 64, 'dropout': 0.3},
    'fusion_config': {'fused_dim': 32, 'num_fusion_heads': 8, 'fusion_ffn_multiplier': 5},
    'classifier_config': {'classifier_hidden_dim': 32},
    'projection_head_config': {'hidden_dim': 32, 'output_dim': 32, 'dropout': 0.5},
    'd_embed': 32, 'missing_strategy': 'learnable',
}
param_grid = {
    'lr': [1e-3, 5e-4],
    'cross_cl': [0.1, 0.2, 0.5],
    'd_embed': [16, 32, 64],
    'hidden_channels_vae': [32, 64, 128],
    'attention_hidden_dim': [32, 64],
    'heads': [4, 8],
}
csv_out = 'research/2026-08-06-project-review-audit/output/raw/gvae_sweep_results_v2.csv'
df = run_gvae_sweep(data=data, base_model_config=model_config, base_train_config=train_config,
                    param_grid=param_grid, n_trials=10, n_splits=3, epochs=100,
                    output_csv=csv_out, random_seed=420)
df_ok = df.dropna(subset=['mean_auc'])
if len(df_ok) == 0:
    print('FATAL: all sweep trials failed (mean_auc NaN) — see trial error column')
    raise SystemExit(2)
best = df_ok.sort_values('mean_auc', ascending=False).iloc[0]
best_params = {k: best[k] for k in ['lr', 'cross_cl', 'd_embed', 'hidden_channels_vae', 'attention_hidden_dim', 'heads']}
_int_keys = ('d_embed', 'hidden_channels_vae', 'attention_hidden_dim', 'heads')
json.dump({k: (int(v) if k in _int_keys else float(v)) for k, v in best_params.items()},
          open('research/2026-08-06-project-review-audit/output/raw/best_params.json', 'w'), indent=2)
print('BEST_PARAMS:', best_params, '| mean_auc:', float(best['mean_auc']), 'std:', float(best.get('std_auc', float('nan'))))
EOF

# 2. Full GVAE run with the best sweep params (5-fold, latent_quality, top-k 3, seed 42)
echo "=== $(date) GVAE best-params full run (5-fold) ===" >> "$LOG"
$PY - <<'EOF' >> "$LOG" 2>&1
import torch, json, time, copy
from training.sweep_gvae import update_configs_with_params
from training.train_gvae import kfold_train_gvae

data = torch.load('data_ln_pc_ihc_g.pt', weights_only=False)
DIM_CLINICAL = data['patient'].x_clinical.shape[1]
DIM_PATHOLOGY = data['patient'].x_pathology.shape[1]
DIM_RADIOLOGY = data['lesion'].x.shape[1]
best = json.load(open('research/2026-08-06-project-review-audit/output/raw/best_params.json'))
# Cast dims/heads back to int — JSON round-trip turned them into floats and
# nn.Linear/GATv2Conv reject float dims (TypeError at torch.empty).
for _k in ('d_embed', 'hidden_channels_vae', 'attention_hidden_dim', 'heads'):
    best[_k] = int(best[_k])

train_config = {  # full-quality settings (not sweep-lite) — CPU forced
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
    'save_best_fold_model': True,
    'checkpoint_dir': 'outputs/gvae/checkpoints/bestparam',
    'metrics_dir': 'outputs/gvae/metrics/bestparam',
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
run_id = 'gvae_bestparam_' + time.strftime('%Y%m%d_%H%M%S')
tc['checkpoint_dir'] = f'outputs/gvae/checkpoints/{run_id}'
tc['metrics_dir'] = f'outputs/gvae/metrics/{run_id}'
print('FINAL_CONFIG:', json.dumps({'params': best}, indent=2))
summary, df_results, roc_data = kfold_train_gvae(data, mc, tc)
print('SUMMARY:', json.dumps(summary, indent=2))
print('RUN_ID:', run_id)
json.dump({'run_id': run_id, 'best_params': best, 'summary': summary},
          open(f'outputs/gvae/metrics/{run_id}/bestparam_meta.json', 'w'), indent=2)
open('research/2026-08-06-project-review-audit/output/raw/bestparam_run_id.txt', 'w').write(run_id)
EOF

echo "=== $(date) PHASE2 DONE ===" >> "$LOG"
