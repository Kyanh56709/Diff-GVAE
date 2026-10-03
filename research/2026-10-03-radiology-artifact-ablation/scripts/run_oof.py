"""Pooled-OOF GVAE evaluation on one graph variant (radiology-artifact ablation).

Config is copied verbatim from
research/2026-08-06-project-review-audit/scripts/pipeline_phase4a_oof.sh
(the canonical pooled-OOF run, head AUC 0.6065) so only the input graph differs.

Usage:
    .venv/bin/python research/2026-10-03-radiology-artifact-ablation/scripts/run_oof.py \
        --data data_ln_pc_ihc_g.pt --tag full34 --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from review_fixes_2026_07.bootstrap_ci import bootstrap_metric_cis  # noqa: E402
from training.train_gvae import kfold_evaluate_gvae_classifier  # noqa: E402

OUT_ROOT = ROOT / "research/2026-10-03-radiology-artifact-ablation/output"


def configs(data, seed):
    dim_clinical = data["patient"].x_clinical.shape[1]
    dim_pathology = data["patient"].x_pathology.shape[1]
    dim_radiology = data["lesion"].x.shape[1]
    train_config = {
        "device": torch.device("cpu"),
        "n_splits": 5, "epochs": 80, "pretrain_epochs": 100,
        "patience": 30, "patience_early_stopping": 30,
        "lr": 0.001, "wd": 1e-4, "batch_size": 64,
        "loss_weights": {"class": 1.0, "cross_cl": 0.5,
                         "rec_attr": {"clinical": 1.0, "pathology": 1.0, "radiology": 1.0},
                         "rec_struct": 0.1, "kl": 0.00001},
        "annealing": {"kl": {"start_weight": 0.0, "end_weight": 0.00001, "start_epoch": 20, "end_epoch": 300},
                      "cross_cl": {"start_weight": 0.1, "end_weight": 0.1, "start_epoch": 20, "end_epoch": 100}},
        "pca_config": {"clinical": 16, "pathology": 8},
        "lesion_pca_config": {"n_components": 15},
        "cross_cl_temp": 0.1, "grad_clip_norm": 1.0, "print_every_k_epochs": 20,
        "random_seed": seed, "vectorized_contrastive": True,
        "inner_val_split": 0.2, "selection_metric": "inner_val_auc",
        "select_after_epoch": 20, "n_bootstrap": 2000, "eval_seed": 4200,
    }
    model_config = {
        "view_configs": {
            "clinical": {"in_channels": dim_clinical, "hidden_channels_vae": 128, "heads": 8, "dropout": 0.3, "num_gnn_layers": 2, "edge_dim": 1},
            "pathology": {"in_channels": dim_pathology, "hidden_channels_vae": 128, "heads": 8, "dropout": 0.3, "num_gnn_layers": 2, "edge_dim": 1},
            "radiology": {"in_channels": 32, "hidden_channels_vae": 128, "heads": 8, "dropout": 0.3, "num_gnn_layers": 2, "edge_dim": 1},
        },
        "radiology_aggregator_config": {"lesion_feature_dim": dim_radiology, "aggregated_output_dim": 32,
                                        "attention_hidden_dim": 32, "dropout": 0.3},
        "fusion_config": {"fused_dim": 32, "num_fusion_heads": 8, "fusion_ffn_multiplier": 5},
        "classifier_config": {"classifier_hidden_dim": 32},
        "projection_head_config": {"hidden_dim": 32, "output_dim": 32, "dropout": 0.3},
        "d_embed": 32, "missing_strategy": "learnable",
        "logvar_clamp": (-4.0, 2.0), "radiology_zero_lesion_passthrough": True,
    }
    return model_config, train_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = torch.load(ROOT / args.data, weights_only=False)
    out = OUT_ROOT / f"{args.tag}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    model_config, train_config = configs(data, args.seed)
    summary, per_fold_df, oof = kfold_evaluate_gvae_classifier(data, model_config, train_config)

    np.savez_compressed(out / "oof_arrays.npz", y_true=oof["y_true"],
                        head_probs=oof["head_probs"], probe_probs=oof["probe_probs"])
    ci = {name: bootstrap_metric_cis(oof["y_true"], oof[f"{name}_probs"], n_boot=2000, seed=4200)
          for name in ("head", "probe")}
    json.dump({"data": args.data, "lesion_dim": int(data["lesion"].x.shape[1]),
               "radiology_edges": int(data["patient", "similar_to_radiology", "patient"].edge_index.shape[1]),
               "seed": args.seed, "summary": summary, "ci": ci},
              open(out / "oof_metrics_with_ci.json", "w"), indent=2, default=float)
    per_fold_df.to_csv(out / "oof_per_fold.csv", index=False)
    print("OOF_DONE", args.tag, "seed", args.seed,
          "HEAD AUC", round(float(ci["head"]["roc_auc"]["point"]), 4),
          "PROBE AUC", round(float(ci["probe"]["roc_auc"]["point"]), 4))


if __name__ == "__main__":
    main()
