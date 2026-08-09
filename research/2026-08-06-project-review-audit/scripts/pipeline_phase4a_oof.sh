#!/bin/bash
# Pipeline phase 4a: pooled-OOF evaluation + bootstrap CIs on the ranked best-params run (CPU).
set -e
cd /Users/admin/Diff-GVAE
PY=.venv/bin/python
OUT=research/2026-08-06-project-review-audit/output
RAW=$OUT/raw
TAB=$OUT/tables
mkdir -p "$TAB"
LOG=$RAW/pipeline_phase4.log

echo "=== $(date) phase4a: pooled-OOF eval (ranked run config) ===" >> "$LOG"
$PY - <<'EOF' >> "$LOG" 2>&1
import torch, json, numpy as np
from training.train_gvae import kfold_evaluate_gvae_classifier
from review_fixes_2026_07.bootstrap_ci import bootstrap_metric_cis

data = torch.load('data_ln_pc_ihc_g.pt', weights_only=False)
DIM_CLINICAL = data['patient'].x_clinical.shape[1]
DIM_PATHOLOGY = data['patient'].x_pathology.shape[1]
DIM_RADIOLOGY = data['lesion'].x.shape[1]

train_config = {
    'data_path': 'data_ln_pc_ihc_g.pt',
    'device': torch.device('cpu'),
    'n_splits': 5, 'epochs': 80, 'pretrain_epochs': 100,
    'patience': 30, 'patience_early_stopping': 30,
    'lr': 0.001, 'wd': 1e-4, 'batch_size': 64,
    'loss_weights': {'class': 1.0, 'cross_cl': 0.5,
                     'rec_attr': {'clinical': 1.0, 'pathology': 1.0, 'radiology': 1.0},
                     'rec_struct': 0.1, 'kl': 0.00001},
    'annealing': {'kl': {'start_weight': 0.0, 'end_weight': 0.00001, 'start_epoch': 20, 'end_epoch': 300},
                  'cross_cl': {'start_weight': 0.1, 'end_weight': 0.1, 'start_epoch': 20, 'end_epoch': 100}},
    'pca_config': {'clinical': 16, 'pathology': 8},
    'lesion_pca_config': {'n_components': 15},
    'cross_cl_temp': 0.1, 'grad_clip_norm': 1.0, 'print_every_k_epochs': 20,
    'random_seed': 42, 'vectorized_contrastive': True,
    'inner_val_split': 0.2, 'selection_metric': 'inner_val_auc',
    'select_after_epoch': 20, 'n_bootstrap': 2000, 'eval_seed': 4200,
}
model_config = {
    'view_configs': {
        'clinical': {'in_channels': DIM_CLINICAL, 'hidden_channels_vae': 128, 'heads': 8, 'dropout': 0.3, 'num_gnn_layers': 2, 'edge_dim': 1},
        'pathology': {'in_channels': DIM_PATHOLOGY, 'hidden_channels_vae': 128, 'heads': 8, 'dropout': 0.3, 'num_gnn_layers': 2, 'edge_dim': 1},
        'radiology': {'in_channels': 32, 'hidden_channels_vae': 128, 'heads': 8, 'dropout': 0.3, 'num_gnn_layers': 2, 'edge_dim': 1},
    },
    'radiology_aggregator_config': {'lesion_feature_dim': DIM_RADIOLOGY, 'aggregated_output_dim': 32,
                                    'attention_hidden_dim': 32, 'dropout': 0.3},
    'fusion_config': {'fused_dim': 32, 'num_fusion_heads': 8, 'fusion_ffn_multiplier': 5},
    'classifier_config': {'classifier_hidden_dim': 32},
    'projection_head_config': {'hidden_dim': 32, 'output_dim': 32, 'dropout': 0.3},
    'd_embed': 32, 'missing_strategy': 'learnable',
    'logvar_clamp': (-4.0, 2.0), 'radiology_zero_lesion_passthrough': True,
}
summary, per_fold_df, oof = kfold_evaluate_gvae_classifier(data, model_config, train_config)

np.savez_compressed('research/2026-08-06-project-review-audit/output/tables/oof_arrays.npz',
                    y_true=oof['y_true'], head_probs=oof['head_probs'], probe_probs=oof['probe_probs'])

ci = {}
for name, probs in [('head', oof['head_probs']), ('probe', oof['probe_probs'])]:
    ci[name] = bootstrap_metric_cis(oof['y_true'], probs, n_boot=2000, seed=4200)
json.dump({'summary': summary, 'ci': ci},
          open('research/2026-08-06-project-review-audit/output/tables/oof_metrics_with_ci.json', 'w'), indent=2, default=float)
per_fold_df.to_csv('research/2026-08-06-project-review-audit/output/tables/oof_per_fold.csv', index=False)
print('OOF_DONE')
print('HEAD AUC:', round(float(ci['head']['roc_auc']['point']), 4),
      'CI', [round(float(x), 4) for x in ci['head']['roc_auc'][['ci_low', 'ci_high']]])
print('HEAD PR-AUC:', round(float(ci['head']['pr_auc']['point']), 4))
print('PROBE AUC:', round(float(ci['probe']['roc_auc']['point']), 4))
EOF

echo "=== $(date) PHASE4A DONE ===" >> "$LOG"
