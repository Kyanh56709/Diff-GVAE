import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_gvae import kfold_train_gvae


def json_ready(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Train GVAE folds and rank checkpoints for DDPM latent augmentation."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--pretrain-epochs", type=int, default=400)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--pretrain-patience", type=int, default=30)
    parser.add_argument("--checkpoint-metric", default="latent_quality")
    parser.add_argument("--early-stopping-metric", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-prefix", default="gvae_latent_quality")
    args = parser.parse_args()

    output_root = Path("outputs/gvae")
    run_id = f"{args.run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = output_root / "checkpoints" / run_id
    metrics_dir = output_root / "metrics" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir.mkdir(parents=True, exist_ok=False)

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data = torch.load("data_ln_pc_ihc_g.pt", map_location="cpu", weights_only=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

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
        "logvar_clamp": [-6.0, 2.0],
    }

    train_config = {
        "device": device,
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "patience": args.patience,
        "patience_early_stopping": args.early_stopping_patience,
        "lr": 5e-4,
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
        "batch_size": args.batch_size,
        "balanced_batch_sampling": True,
        "pos_weight_strategy": "auto",
        "classification_loss_multiplier": 2.0,
        "print_every_k_epochs": 10,
        "random_seed": seed,
        "save_best_fold_model": True,
        "checkpoint_metric": args.checkpoint_metric,
        "early_stopping_metric": args.early_stopping_metric or args.checkpoint_metric,
        "top_k_gvae_checkpoints": args.top_k,
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_dir": str(metrics_dir),
        "run_id": run_id,
        "overwrite_checkpoints": False,
        "classification_threshold_selection_metric": "balanced_accuracy",
        "threshold_strategy": "fixed",
        "classification_threshold": 0.5,
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_use_pos_weight": True,
        "pretrain_val_split": 0.2,
        "pretrain_patience": args.pretrain_patience,
        "pretrain_seed": seed,
        "vectorized_contrastive": True,
    }

    run_config_path = metrics_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"run_id": run_id, "model_config": json_ready(model_config), "train_config": json_ready(train_config)},
            f,
            indent=2,
        )

    print(f"RUN_ID={run_id}", flush=True)
    print(f"DEVICE={device}", flush=True)
    print(f"CHECKPOINT_DIR={checkpoint_dir}", flush=True)
    print(f"METRICS_DIR={metrics_dir}", flush=True)

    data = data.to(device)
    summary, df_results, roc_data = kfold_train_gvae(data, model_config, train_config)

    summary_path = metrics_dir / "summary.json"
    fold_metrics_path = metrics_dir / "fold_metrics.csv"
    roc_data_path = metrics_dir / "roc_data.pt"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, indent=2)
    df_results.to_csv(fold_metrics_path, index=False)
    torch.save(json_ready(roc_data), roc_data_path)

    print(f"SUMMARY_PATH={summary_path}", flush=True)
    print(f"FOLD_METRICS_PATH={fold_metrics_path}", flush=True)
    print(f"ROC_DATA_PATH={roc_data_path}", flush=True)
    print("SUMMARY_JSON=" + json.dumps(json_ready(summary), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
