"""A6 — reproduce the legacy `majority/val-fold` protocol that produced ~0.81-0.87.

The notebook `training_local.ipynb` cell 2 ran `run_gvae_sweep` -> `kfold_train_gvae`
on the legacy 34-dim graph `data_ln_pc_ihc_g.pt` and reported mean_auc=0.8116,
mean_f1=0.8787. This script runs `kfold_train_gvae` with the notebook's exact config
and the winning sweep hyperparameters, writing ALL outputs to an isolated dir so the
canonical r32 artifacts under outputs/ are never touched.

Run:
  .venv/bin/python research/2026-10-05-a6-notebook-reconciliation/scripts/run_legacy_kfold_repro.py \
      --epochs 200 --seed 42
"""
import argparse
import copy
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from training.train_gvae import kfold_train_gvae  # noqa: E402
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output"


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception as exc:  # git unavailable is non-fatal; provenance is best-effort
        return f"unavailable ({exc})"


def build_configs(dim_clinical, dim_pathology, dim_radiology, run_dir):
    train_config = {
        "data_path": "data_ln_pc_ihc_g.pt",
        "device": torch.device("cpu"),
        "n_splits": 5,
        "epochs": 200,
        "pretrain_epochs": 150,
        "patience": 30,
        "patience_early_stopping": 30,
        "lr": 0.001,
        "wd": 1e-5,
        "batch_size": 64,
        "loss_weights": {
            "class": 1.0,
            "cross_cl": 0.5,  # winning sweep trial
            "rec_attr": {"clinical": 0.5, "pathology": 0.5, "radiology": 0.5},
            "rec_struct": 0.05,
            "kl": 0.001,
        },
        "annealing": {"kl": {"start_weight": 0.0, "end_weight": 0.001, "start_epoch": 0, "end_epoch": 100}},
        "pca_config": {"clinical": 16, "pathology": 8},
        "lesion_pca_config": {"n_components": 15},
        "cross_cl_temp": 0.1,
        "grad_clip_norm": 1.0,
        "print_every_k_epochs": 10,
        "vectorized_contrastive": True,
        # NOTE: inner_val_split / selection_metric / select_after_epoch are inert here —
        # they belong to kfold_evaluate_gvae_classifier. They are kept only because the
        # notebook config defines them; kfold_train_gvae selects on the outer val fold.
        "inner_val_split": 0.2,
        "selection_metric": "inner_val_auc",
        "select_after_epoch": 20,
        # isolation: never write into canonical outputs/; per-run subdir avoids
        # two runs sharing eval dirs (which would leave dangling path pointers).
        "save_best_fold_model": False,
        "checkpoint_dir": str(run_dir / "legacy_checkpoints"),
        "classification_eval_dir": str(run_dir / "legacy_classification_eval"),
    }
    model_config = {
        "view_configs": {
            "clinical": {"in_channels": dim_clinical, "hidden_channels_vae": 128, "heads": 8, "dropout": 0.3, "num_gnn_layers": 2, "edge_dim": 1},
            "pathology": {"in_channels": dim_pathology, "hidden_channels_vae": 128, "heads": 8, "dropout": 0.3, "num_gnn_layers": 2, "edge_dim": 1},
            "radiology": {"in_channels": 32, "hidden_channels_vae": 128, "heads": 8, "dropout": 0.3, "num_gnn_layers": 2, "edge_dim": 1},
        },
        "radiology_aggregator_config": {"lesion_feature_dim": dim_radiology, "aggregated_output_dim": 32, "attention_hidden_dim": 64, "dropout": 0.3},
        "fusion_config": {"fused_dim": 16, "num_fusion_heads": 4, "fusion_ffn_multiplier": 4},
        "classifier_config": {"classifier_hidden_dim": 32},
        "projection_head_config": {"hidden_dim": 32, "output_dim": 16, "dropout": 0.3},
        "d_embed": 16,
        "missing_strategy": "learnable",
        "logvar_clamp": (-4.0, 2.0),
        "radiology_zero_lesion_passthrough": True,
    }
    return model_config, train_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-metric", default=None,
                    help="Override train_config['checkpoint_metric'] (e.g. 'auc' to mimic the legacy val-AUC selection).")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = torch.load(ROOT / "data_ln_pc_ihc_g.pt", weights_only=False)
    dims = (data["patient"].x_clinical.shape[1], data["patient"].x_pathology.shape[1], data["lesion"].x.shape[1])
    print(f"legacy graph dims: clinical={dims[0]}, pathology={dims[1]}, radiology_lesion={dims[2]}")

    metric_tag = args.checkpoint_metric or "latent_quality"
    run_dir = OUT / f"seed{args.seed}_ep{args.epochs}_{metric_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_config, train_config = build_configs(*dims, run_dir)
    train_config["epochs"] = args.epochs
    train_config["random_seed"] = args.seed
    if args.checkpoint_metric:
        train_config["checkpoint_metric"] = args.checkpoint_metric
        train_config["early_stopping_metric"] = args.checkpoint_metric

    summary, df, roc_data = kfold_train_gvae(data, model_config, train_config)

    # Post-conditions: a partial/degenerate callee result must NOT be archived as a
    # valid run (a stale artifact under the fixed filename would masquerade as fresh).
    if not summary or "mean_auc" not in summary:
        raise RuntimeError(f"kfold_train_gvae returned no mean_auc (summary keys={list(summary)}).")
    if not np.isfinite(float(summary["mean_auc"])):
        raise RuntimeError(f"non-finite mean_auc: {summary['mean_auc']}.")
    if len(df) != train_config["n_splits"]:
        raise RuntimeError(f"expected {train_config['n_splits']} fold rows, got {len(df)}.")
    if len(roc_data) != train_config["n_splits"]:
        raise RuntimeError(f"expected {train_config['n_splits']} per-fold roc entries, got {len(roc_data)}.")

    # Legacy metrics: val-fold mean of (AUC at the selected checkpoint, F1 at val-tuned threshold)
    legacy = {k: float(v) for k, v in summary.items() if k in ("mean_auc", "std_auc", "mean_f1", "std_f1", "mean_accuracy", "std_accuracy", "mean_best_threshold")}

    # Pool the per-fold val predictions to isolate metric-definition vs split effects.
    y_true = np.concatenate([r["y_true"] for r in roc_data])
    y_prob = np.concatenate([r["y_pred_probs"] for r in roc_data])
    pooled = {
        "pooled_val_auc": float(roc_auc_score(y_true, y_prob)),
        "n": int(len(y_true)),
    }

    # Fold-divergence annotation: the legacy protocol can SELECT a diverged checkpoint
    # (finite-but-exploded val loss) and still count its AUC toward the mean. Flag any
    # such fold so the artifact is self-documenting rather than silently averaged.
    if "best_val_loss_at_selected_epoch" not in df.columns:
        raise RuntimeError(
            "df is missing 'best_val_loss_at_selected_epoch'; divergence detection "
            "would be silently disabled. Update the script if the callee renamed it."
        )
    diverged = [
        {"fold": i + 1, "val_loss": r.get("best_val_loss_at_selected_epoch"), "auc": r.get("auc")}
        for i, r in enumerate(df.to_dict("records"))
        if r.get("best_val_loss_at_selected_epoch") is not None
        and np.isfinite(float(r["best_val_loss_at_selected_epoch"]))
        and float(r["best_val_loss_at_selected_epoch"]) > 1e4
    ]

    metric_tag = train_config.get("checkpoint_metric", "latent_quality")
    result = {
        "protocol": f"legacy kfold_train_gvae (val-fold, best {metric_tag} ckpt)",
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_metric": metric_tag,
        "eval_dir": str(run_dir / "legacy_classification_eval"),
        "legacy_summary": legacy,
        "pooled_from_val_arrays": pooled,
        "diverged_folds": diverged,
        "diverged_fold_count": len(diverged),
        # Provenance: a fixed filename means a crashed rerun leaves the previous file
        # untouched; these fields let a reader tell which run/code produced this artifact.
        "run_completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "per_fold": df.to_dict("records"),
    }
    # Atomic write (tmp -> replace) so a crash mid-write can never produce a partial
    # file at the expected path; tmp is removed on failure.
    out_path = OUT / f"legacy_repro_seed{args.seed}_ep{args.epochs}_{metric_tag}.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(result, indent=1, default=str))
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print("\nLEGACY SUMMARY:", json.dumps(legacy, indent=1))
    print("POOLED(val arrays):", pooled)
    if diverged:
        print(f"WARNING: {len(diverged)} fold(s) selected a diverged checkpoint (val_loss>1e4): {diverged}")


if __name__ == "__main__":
    main()
