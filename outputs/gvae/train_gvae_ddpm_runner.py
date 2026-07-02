import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

DEPRECATED_PIPELINE_NOTICE = (
    "Deprecated for the paper objective: this runner treats DDPM as a "
    "loss-based classifier. Use outputs/gvae/train_conditional_ddpm_augmentation_runner.py "
    "for conditional latent-space augmentation."
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_pipeline import kfold_gvae_ddpm_generative_classifier


def json_ready(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def flatten_candidate_results(results):
    rows = []
    for result in results.get("all_candidate_results", []):
        metrics_0_5 = result.get("threshold_0_5_metrics") or result.get("metrics", {})
        best_metrics = result.get("best_threshold_metrics", {})
        best_f1_metrics = result.get("best_f1_threshold_metrics", {})
        best_bal_metrics = result.get("best_balanced_accuracy_threshold_metrics", {})
        gvae_metrics_0_5 = result.get("gvae_classifier_metrics_threshold_0_5", {})
        gvae_best_metrics = result.get("gvae_classifier_metrics_best_threshold", {})
        class_distribution = result.get("class_distribution", {})
        validation_distribution = (
            class_distribution.get("val")
            or class_distribution.get("validation")
            or {}
        )
        rows.append(
            {
                "fold": result.get("fold"),
                "rank_by_gvae_val_auc": result.get("rank_by_val_auc"),
                "epoch": result.get("epoch"),
                "status": result.get("status"),
                "gvae_val_auc": result.get("gvae_val_auc"),
                "gvae_val_loss": result.get("gvae_val_loss"),
                "ddpm_latent_representation": result.get("ddpm_latent_representation"),
                "ddpm_latent_dim": result.get("ddpm_latent_dim"),
                "val_positive_rate": validation_distribution.get("positive_rate"),
                "ddpm_auc": metrics_0_5.get("auc"),
                "ddpm_pr_auc": metrics_0_5.get("pr_auc"),
                "ddpm_f1_threshold_0_5": metrics_0_5.get("f1"),
                "ddpm_balanced_accuracy_threshold_0_5": metrics_0_5.get("balanced_accuracy"),
                "ddpm_precision_threshold_0_5": metrics_0_5.get("precision"),
                "ddpm_recall_threshold_0_5": metrics_0_5.get("recall"),
                "ddpm_sensitivity_threshold_0_5": metrics_0_5.get("sensitivity"),
                "ddpm_specificity_threshold_0_5": metrics_0_5.get("specificity"),
                "ddpm_brier_score": metrics_0_5.get("brier_score"),
                "best_threshold": best_metrics.get("threshold"),
                "best_threshold_selection_metric": best_metrics.get("threshold_selection_metric"),
                "ddpm_f1_best_threshold": best_metrics.get("f1"),
                "ddpm_balanced_accuracy_best_threshold": best_metrics.get("balanced_accuracy"),
                "ddpm_precision_best_threshold": best_metrics.get("precision"),
                "ddpm_recall_best_threshold": best_metrics.get("recall"),
                "ddpm_sensitivity_best_threshold": best_metrics.get("sensitivity"),
                "ddpm_specificity_best_threshold": best_metrics.get("specificity"),
                "ddpm_best_f1_threshold": best_f1_metrics.get("threshold"),
                "ddpm_best_balanced_accuracy_threshold": best_bal_metrics.get("threshold"),
                "gvae_classifier_auc": gvae_metrics_0_5.get("auc"),
                "gvae_classifier_pr_auc": gvae_metrics_0_5.get("pr_auc"),
                "gvae_classifier_f1_threshold_0_5": gvae_metrics_0_5.get("f1"),
                "gvae_classifier_balanced_accuracy_threshold_0_5": (
                    gvae_metrics_0_5.get("balanced_accuracy")
                ),
                "gvae_classifier_f1_best_threshold": gvae_best_metrics.get("f1"),
                "gvae_classifier_balanced_accuracy_best_threshold": (
                    gvae_best_metrics.get("balanced_accuracy")
                ),
                "gvae_classifier_best_threshold": gvae_best_metrics.get("threshold"),
                "gvae_classifier_eval_dir": result.get("gvae_classifier_eval_dir"),
                "normalized_average_precision": result.get("normalized_average_precision"),
                "ddpm_eval_dir": result.get("eval_dir"),
                "gvae_checkpoint_path": result.get("gvae_checkpoint_path"),
                "latents_path": result.get("latents_path"),
            }
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if "--allow-deprecated-loss-classifier" not in sys.argv:
        raise SystemExit(f"ERROR={DEPRECATED_PIPELINE_NOTICE}")
    sys.argv.remove("--allow-deprecated-loss-classifier")

    output_root = Path("outputs/gvae_ddpm")
    run_id = f"gvae_ddpm_concat_mu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics_dir = output_root / "metrics" / run_id
    artifact_dir = output_root / "artifacts"
    metrics_dir.mkdir(parents=True, exist_ok=False)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data = torch.load("data_ln_pc_ihc_g.pt", map_location="cpu", weights_only=False)
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
        "logvar_clamp": [-6.0, 2.0],
    }

    train_config = {
        "device": device,
        "n_splits": 5,
        "epochs": 200,
        "patience": 30,
        "patience_early_stopping": 40,
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
        "batch_size": 64,
        "balanced_batch_sampling": True,
        "pos_weight_strategy": "auto",
        "classification_loss_multiplier": 2.0,
        "print_every_k_epochs": 10,
        "random_seed": seed,
        "checkpoint_metric": "auc_pr_balanced_accuracy",
        "early_stopping_metric": "auc_pr_balanced_accuracy",
        "top_k_gvae_checkpoints": 5,
        "classification_threshold_selection_metric": "balanced_accuracy",
        "ddpm_selection_metric": "pr_auc",
        "ddpm_threshold_selection_metric": "balanced_accuracy",
        "ddpm_latent_representation": "concat_mu",
        "latent_sample_seed": seed,
        "artifact_dir": str(artifact_dir),
        "latent_output_dir": "outputs/latent_for_ddpm",
        "run_id": run_id,
        "overwrite_checkpoints": False,
        "allow_deprecated_ddpm_classifier": True,
        "threshold_strategy": "fixed",
        "classification_threshold": 0.5,
        "pretrain_epochs": 400,
        "pretrain_use_pos_weight": True,
        "pretrain_val_split": 0.2,
        "pretrain_patience": 30,
        "pretrain_seed": seed,
        "vectorized_contrastive": True,
    }

    ddpm_config = {
        "latent_dim": model_config["d_embed"] * len(model_config["view_configs"]),
        "timesteps": 250,
        "eval_timesteps": 50,
        "epochs": 120,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout_prob": 0.2,
        "dim_mults": [1, 2],
    }
    pca_config = {}

    run_config_path = metrics_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "model_config": json_ready(model_config),
                "train_config": json_ready(train_config),
                "ddpm_config": json_ready(ddpm_config),
                "pca_config": pca_config,
            },
            f,
            indent=2,
        )

    print(f"RUN_ID={run_id}", flush=True)
    print(f"DEVICE={device}", flush=True)
    print(f"METRICS_DIR={metrics_dir}", flush=True)
    print(f"ARTIFACT_DIR={artifact_dir / run_id}", flush=True)
    print(f"LATENT_OUTPUT_DIR={Path('outputs/latent_for_ddpm') / run_id}", flush=True)

    data = data.to(device)
    results = kfold_gvae_ddpm_generative_classifier(
        data,
        model_config,
        train_config,
        ddpm_config,
        pca_config,
    )

    summary_path = metrics_dir / "summary.json"
    comparison_path = metrics_dir / "checkpoint_ddpm_comparison.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(results), f, indent=2)
    comparison_rows = flatten_candidate_results(results)
    write_csv(comparison_path, comparison_rows)

    print(f"SUMMARY_PATH={summary_path}", flush=True)
    print(f"COMPARISON_TABLE_PATH={comparison_path}", flush=True)
    print("SUMMARY_JSON=" + json.dumps(json_ready(results), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
