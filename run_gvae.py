import torch
from training.train_gvae import kfold_train_gvae

data = torch.load("data_ln_pc_ihc_g.pt", map_location="cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = {
    "view_configs": {
        "clinical": {
            "in_channels": data["patient"].x_clinical.shape[1],
            "hidden_channels_vae": 16,
            "heads": 4,
            "dropout": 0.5,
            "num_gnn_layers": 2,
            "edge_dim": 1,
        },
        "pathology": {
            "in_channels": data["patient"].x_pathology.shape[1],
            "hidden_channels_vae": 16,
            "heads": 4,
            "dropout": 0.5,
            "num_gnn_layers": 2,
            "edge_dim": 1,
        },
        "radiology": {
            "in_channels": 32,
            "hidden_channels_vae": 16,
            "heads": 4,
            "dropout": 0.5,
            "num_gnn_layers": 2,
            "edge_dim": 1,
        },
    },
    "radiology_aggregator_config": {
        "lesion_feature_dim": data["lesion"].x.shape[1],
        "aggregated_output_dim": 32,
        "attention_hidden_dim": 32,
        "dropout": 0.3,
    },
    "fusion_config": {"num_fusion_heads": 4, "fusion_ffn_multiplier": 4},
    "classifier_config": {"classifier_hidden_dim": 32},
    "projection_head_config": {"hidden_dim": 32, "output_dim": 64, "dropout": 0.5},
    "d_embed": 64,
    "missing_strategy": "learnable",
}

train_config = {
    "device": device,
    "n_splits": 5,
    "epochs": 100,
    "patience": 30,
    "patience_early_stopping": 30,
    "lr": 1e-3,
    "wd": 1e-4,
    "loss_weights": {
        "class": 1.0,
        "cross_cl": 0.2,
        "rec_attr": {"clinical": 1.0, "pathology": 1.0, "radiology": 1.0},
        "rec_struct": 0.1,
        "kl": 1e-7,
    },
    "annealing": {
        "kl": {"start_weight": 0.0, "end_epoch": 20},
        "cross_cl": {"start_weight": 0.1, "end_epoch": 20},
    },
    "cross_cl_temp": 0.1,
    "grad_clip_norm": 1.0,
    "batch_size": 64,
    "balanced_batch_sampling": True,
    "pos_weight_strategy": "auto",
    "classification_loss_multiplier": 2.0,
    "print_every_k_epochs": 10,
    "random_seed": 42,
    "save_best_fold_model": True,
    "checkpoint_metric": "auc_pr_balanced_accuracy",
    "early_stopping_metric": "auc_pr_balanced_accuracy",
    "checkpoint_dir": "outputs/gvae/checkpoints/manual_run",
    "metrics_dir": "outputs/gvae/metrics/manual_run",
    "classification_threshold_selection_metric": "balanced_accuracy",
    "pretrain_epochs": 100,
}

data = data.to(device)
summary, df_results, roc_data = kfold_train_gvae(data, model_config, train_config)
print(summary)
print(df_results)
