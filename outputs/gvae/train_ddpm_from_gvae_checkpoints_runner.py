import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

DEPRECATED_PIPELINE_NOTICE = (
    "Deprecated for the paper objective: this runner treats DDPM as a "
    "loss-based classifier by comparing responder/non-responder diffusion "
    "losses. Use outputs/gvae/train_conditional_ddpm_augmentation_runner.py "
    "for conditional latent-space augmentation."
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_pipeline import (  # noqa: E402
    _evaluate_gvae_candidate_with_ddpm,
    _prepare_artifact_path,
)


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


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
                "rank_by_gvae_checkpoint_metric": result.get("rank_by_val_auc"),
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
                "normalized_average_precision": result.get("normalized_average_precision"),
                "ddpm_eval_dir": result.get("eval_dir"),
                "gvae_classifier_eval_dir": result.get("gvae_classifier_eval_dir"),
                "gvae_checkpoint_path": result.get("gvae_checkpoint_path"),
                "latents_path": result.get("latents_path"),
            }
        )
    return rows


def checkpoint_to_candidate(checkpoint):
    return {
        "model_state_dict": checkpoint["model_state_dict"],
        "rank_by_val_auc": int(
            checkpoint.get(
                "rank_by_gvae_checkpoint_metric",
                checkpoint.get("rank_by_val_auc", 0),
            )
        ),
        "rank_by_gvae_checkpoint_metric": int(
            checkpoint.get(
                "rank_by_gvae_checkpoint_metric",
                checkpoint.get("rank_by_val_auc", 0),
            )
        ),
        "epoch": int(checkpoint.get("epoch", -1)),
        "val_auc": float(
            checkpoint.get(
                "validation_auc",
                (checkpoint.get("validation_metrics") or {}).get("auc", np.nan),
            )
        ),
        "val_pr_auc": checkpoint.get("validation_pr_auc"),
        "val_balanced_accuracy": checkpoint.get("validation_balanced_accuracy"),
        "val_f1": checkpoint.get("validation_f1"),
        "val_loss": float(
            checkpoint.get(
                "validation_loss",
                checkpoint.get("selected_val_loss", checkpoint.get("best_val_loss", np.nan)),
            )
        ),
        "checkpoint_metric": checkpoint.get("checkpoint_metric"),
        "validation_diagnostics": checkpoint.get("validation_diagnostics"),
    }


def checkpoint_sort_key(path):
    match = re.search(r"_fold_(\d+)_rank_(\d+)_", path.name)
    if not match:
        return (999, 999, path.name)
    return (int(match.group(1)), int(match.group(2)), path.name)


def summarize_results(
    selected_candidate_results,
    all_candidate_results,
    selection_metric,
    latent_representation,
):
    if not selected_candidate_results:
        return {
            "ddpm_selection_metric": selection_metric,
            "ddpm_latent_representation": latent_representation,
            "selected_gvae_checkpoints": [],
            "all_candidate_results": all_candidate_results,
        }

    final_selected = max(
        selected_candidate_results,
        key=lambda r: r["metrics"][selection_metric],
    )

    def mean_std(metric_name, source="threshold_0_5_metrics"):
        values = [
            result[source].get(metric_name, np.nan)
            for result in selected_candidate_results
        ]
        return float(np.nanmean(values)), float(np.nanstd(values))

    mean_auc, std_auc = mean_std("auc")
    mean_pr_auc, std_pr_auc = mean_std("pr_auc")
    mean_f1, std_f1 = mean_std("f1")
    mean_accuracy, std_accuracy = mean_std("accuracy")
    mean_balanced_accuracy, std_balanced_accuracy = mean_std("balanced_accuracy")
    mean_precision, std_precision = mean_std("precision")
    mean_recall, std_recall = mean_std("recall")
    mean_sensitivity, std_sensitivity = mean_std("sensitivity")
    mean_specificity, std_specificity = mean_std("specificity")
    mean_brier_score, std_brier_score = mean_std("brier_score")

    mean_f1_best, std_f1_best = mean_std("f1", "best_threshold_metrics")
    mean_accuracy_best, std_accuracy_best = mean_std("accuracy", "best_threshold_metrics")
    mean_balanced_accuracy_best, std_balanced_accuracy_best = mean_std(
        "balanced_accuracy",
        "best_threshold_metrics",
    )
    mean_precision_best, std_precision_best = mean_std("precision", "best_threshold_metrics")
    mean_recall_best, std_recall_best = mean_std("recall", "best_threshold_metrics")
    mean_sensitivity_best, std_sensitivity_best = mean_std(
        "sensitivity",
        "best_threshold_metrics",
    )
    mean_specificity_best, std_specificity_best = mean_std(
        "specificity",
        "best_threshold_metrics",
    )

    return {
        "ddpm_selection_metric": selection_metric,
        "ddpm_latent_representation": latent_representation,
        "final_selected_gvae_checkpoint": final_selected.get("selected_checkpoint_path"),
        "final_selected_source_checkpoint": final_selected.get("gvae_checkpoint_path"),
        "final_selected_latents_path": final_selected.get("latents_path"),
        "final_selected_fold": final_selected.get("fold"),
        "final_selected_rank_by_gvae_checkpoint_metric": final_selected.get("rank_by_val_auc"),
        "final_selected_ddpm_metrics_threshold_0_5": final_selected.get("threshold_0_5_metrics"),
        "final_selected_ddpm_metrics_best_threshold": final_selected.get("best_threshold_metrics"),
        "final_selected_class_distribution": final_selected.get("class_distribution"),
        "final_selected_prediction_distribution": final_selected.get("prediction_distribution"),
        "final_selected_confusion_matrix_threshold_0_5": (
            final_selected.get("confusion_matrix_threshold_0_5")
        ),
        "final_selected_confusion_matrix_best_threshold": (
            final_selected.get("confusion_matrix_best_threshold")
        ),
        "final_selected_evaluation_artifacts": final_selected.get("evaluation_artifacts"),
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_pr_auc": mean_pr_auc,
        "std_pr_auc": std_pr_auc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_balanced_accuracy": mean_balanced_accuracy,
        "std_balanced_accuracy": std_balanced_accuracy,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
        "mean_sensitivity": mean_sensitivity,
        "std_sensitivity": std_sensitivity,
        "mean_specificity": mean_specificity,
        "std_specificity": std_specificity,
        "mean_brier_score": mean_brier_score,
        "std_brier_score": std_brier_score,
        "mean_f1_best_threshold": mean_f1_best,
        "std_f1_best_threshold": std_f1_best,
        "mean_accuracy_best_threshold": mean_accuracy_best,
        "std_accuracy_best_threshold": std_accuracy_best,
        "mean_balanced_accuracy_best_threshold": mean_balanced_accuracy_best,
        "std_balanced_accuracy_best_threshold": std_balanced_accuracy_best,
        "mean_precision_best_threshold": mean_precision_best,
        "std_precision_best_threshold": std_precision_best,
        "mean_recall_best_threshold": mean_recall_best,
        "std_recall_best_threshold": std_recall_best,
        "mean_sensitivity_best_threshold": mean_sensitivity_best,
        "std_sensitivity_best_threshold": std_sensitivity_best,
        "mean_specificity_best_threshold": mean_specificity_best,
        "std_specificity_best_threshold": std_specificity_best,
        "selected_gvae_checkpoints": [
            {
                "fold": result.get("fold"),
                "rank_by_gvae_checkpoint_metric": result.get("rank_by_val_auc"),
                "gvae_val_auc": result.get("gvae_val_auc"),
                "ddpm_metrics_threshold_0_5": result.get("threshold_0_5_metrics"),
                "ddpm_metrics_best_threshold": result.get("best_threshold_metrics"),
                "class_distribution": result.get("class_distribution"),
                "prediction_distribution": result.get("prediction_distribution"),
                "confusion_matrix_threshold_0_5": (
                    result.get("confusion_matrix_threshold_0_5")
                ),
                "confusion_matrix_best_threshold": (
                    result.get("confusion_matrix_best_threshold")
                ),
                "evaluation_artifacts": result.get("evaluation_artifacts"),
                "gvae_checkpoint_path": result.get("gvae_checkpoint_path"),
                "latents_path": result.get("latents_path"),
                "selected_checkpoint_path": result.get("selected_checkpoint_path"),
            }
            for result in selected_candidate_results
        ],
        "all_candidate_results": all_candidate_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "DEPRECATED: trains the old loss-based DDPM classifier pipeline. "
            "Use train_conditional_ddpm_augmentation_runner.py instead."
        )
    )
    parser.add_argument("--gvae-run-id", default="gvae_20260615_121426")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--eval-timesteps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--selection-metric", default="pr_auc")
    parser.add_argument("--latent-representation", default="concat_mu")
    parser.add_argument(
        "--allow-deprecated-loss-classifier",
        action="store_true",
        help="Run the legacy DDPM-as-classifier experiment intentionally.",
    )
    args = parser.parse_args()

    if not args.allow_deprecated_loss_classifier:
        raise SystemExit(f"ERROR={DEPRECATED_PIPELINE_NOTICE}")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    output_root = Path("outputs/gvae_ddpm")
    run_id = f"ddpm_from_{args.gvae_run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics_dir = output_root / "metrics" / run_id
    artifact_dir = output_root / "artifacts" / run_id
    metrics_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_root = Path("outputs/gvae/checkpoints") / args.gvae_run_id
    checkpoint_paths = sorted(
        checkpoint_root.glob("*_rank_*.pt"),
        key=checkpoint_sort_key,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No ranked GVAE checkpoints found in {checkpoint_root}")

    checkpoint_paths_by_fold = defaultdict(list)
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_paths_by_fold[int(checkpoint["fold"])].append(path)

    checkpoint_paths = []
    for fold in sorted(checkpoint_paths_by_fold):
        fold_paths = sorted(checkpoint_paths_by_fold[fold], key=checkpoint_sort_key)
        checkpoint_paths.extend(fold_paths[: args.top_k])

    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    model_config = first_checkpoint["model_config"]
    train_config = dict(first_checkpoint.get("train_config", {}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config.update(
        {
            "device": device,
            "run_id": run_id,
            "artifact_dir": str(output_root / "artifacts"),
            "latent_output_dir": "outputs/latent_for_ddpm",
            "ddpm_selection_metric": args.selection_metric,
            "ddpm_threshold_selection_metric": "balanced_accuracy",
            "ddpm_latent_representation": args.latent_representation,
            "latent_sample_seed": seed,
            "classification_threshold_selection_metric": "balanced_accuracy",
            "overwrite_checkpoints": False,
            "allow_deprecated_ddpm_classifier": True,
        }
    )

    ddpm_config = {
        "latent_dim": model_config["d_embed"] * len(model_config["view_configs"]),
        "timesteps": args.timesteps,
        "eval_timesteps": args.eval_timesteps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout_prob": 0.2,
        "dim_mults": [1, 2],
    }

    run_config = {
        "run_id": run_id,
        "source_gvae_run_id": args.gvae_run_id,
        "source_checkpoint_dir": str(checkpoint_root),
        "model_config": json_ready(model_config),
        "train_config": json_ready(train_config),
        "ddpm_config": json_ready(ddpm_config),
        "selected_checkpoint_paths": [str(path) for path in checkpoint_paths],
    }
    with (metrics_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"RUN_ID={run_id}", flush=True)
    print(f"SOURCE_GVAE_RUN_ID={args.gvae_run_id}", flush=True)
    print(f"DEVICE={device}", flush=True)
    print(f"METRICS_DIR={metrics_dir}", flush=True)
    print(f"ARTIFACT_DIR={artifact_dir}", flush=True)
    print(f"LATENT_OUTPUT_DIR={Path('outputs/latent_for_ddpm') / run_id}", flush=True)
    print(f"CANDIDATE_CHECKPOINTS={len(checkpoint_paths)}", flush=True)

    full_data = torch.load("data_ln_pc_ihc_g.pt", map_location="cpu", weights_only=False).to(device)

    selected_candidate_results = []
    all_candidate_results = []

    for fold in sorted(checkpoint_paths_by_fold):
        fold_paths = [
            path for path in checkpoint_paths
            if int(torch.load(path, map_location="cpu", weights_only=False)["fold"]) == fold
        ]
        if not fold_paths:
            continue

        print(f"\n==================== DDPM FOLD {fold} ====================", flush=True)
        candidate_results = []

        for checkpoint_path in fold_paths:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            candidate = checkpoint_to_candidate(checkpoint)
            rank = candidate["rank_by_val_auc"]
            epoch = candidate["epoch"]

            print(
                f"--- [Fold {fold}] GVAE rank {rank}, epoch {epoch}, "
                f"val AUC={candidate['val_auc']:.4f}, PR-AUC={candidate.get('val_pr_auc')} ---",
                flush=True,
            )

            train_indices = torch.as_tensor(checkpoint["train_indices"], dtype=torch.long, device=device)
            val_indices = torch.as_tensor(checkpoint["val_indices"], dtype=torch.long, device=device)

            fold_artifact_dir = artifact_dir / f"fold_{fold}"
            latents_dir = Path("outputs/latent_for_ddpm") / run_id / f"fold_{fold}"
            candidate_gvae_eval_dir = (
                fold_artifact_dir / "gvae_eval" / f"rank_{rank}_epoch_{epoch}"
            )
            eval_dir = fold_artifact_dir / "ddpm_eval" / f"rank_{rank}_epoch_{epoch}"
            latents_path = latents_dir / f"rank_{rank}_epoch_{epoch}_latents_for_ddpm.pt"

            candidate_result = _evaluate_gvae_candidate_with_ddpm(
                full_data=full_data,
                model_config=model_config,
                train_config=train_config,
                ddpm_config=ddpm_config,
                candidate=candidate,
                fold_num=fold,
                train_indices=train_indices,
                val_indices=val_indices,
                candidate_checkpoint_path=checkpoint_path,
                latents_path=latents_path,
                gvae_eval_dir=candidate_gvae_eval_dir,
                eval_dir=eval_dir,
            )
            candidate_results.append(candidate_result)
            all_candidate_results.append(
                {
                    k: v
                    for k, v in candidate_result.items()
                    if k not in {"probs", "labels", "loss_responder", "loss_non_responder"}
                }
            )

            metrics = candidate_result["threshold_0_5_metrics"]
            best_metrics = candidate_result.get("best_threshold_metrics", {})
            print(
                f"  [Fold {fold} Rank {rank}] "
                f"DDPM AUC={metrics['auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, "
                f"F1@0.5={metrics['f1']:.4f}, BalAcc@0.5={metrics['balanced_accuracy']:.4f}, "
                f"BestThr={best_metrics.get('threshold', np.nan):.4f}, "
                f"BestF1={best_metrics.get('f1', np.nan):.4f}, "
                f"BestBalAcc={best_metrics.get('balanced_accuracy', np.nan):.4f}",
                flush=True,
            )

        valid_results = [
            result
            for result in candidate_results
            if result["status"] == "ok"
            and not np.isnan(result["metrics"].get(args.selection_metric, np.nan))
        ]
        if not valid_results:
            print(f"WARNING: No valid DDPM results for fold {fold}.", flush=True)
            continue

        selected = max(
            valid_results,
            key=lambda result: result["metrics"][args.selection_metric],
        )
        selected_checkpoint_path = (
            artifact_dir / f"fold_{fold}" / "checkpoints"
            / f"selected_by_ddpm_{args.selection_metric}.pt"
        )
        selected_checkpoint_path = _prepare_artifact_path(selected_checkpoint_path, False)
        selected_source_checkpoint = Path(selected["gvae_checkpoint_path"])
        selected_source = torch.load(
            selected_source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        selected_ddpm_config = dict(ddpm_config)
        selected_ddpm_config["latent_dim"] = selected["ddpm_latent_dim"]
        torch.save(
            {
                "model_state_dict": selected_source["model_state_dict"],
                "model_config": model_config,
                "train_config": train_config,
                "ddpm_config": selected_ddpm_config,
                "fold": fold,
                "selection_metric": args.selection_metric,
                "selected_candidate": {
                    k: v
                    for k, v in selected.items()
                    if k not in {"probs", "labels", "loss_responder", "loss_non_responder"}
                },
                "source_gvae_checkpoint": str(selected_source_checkpoint),
                "train_indices": selected_source["train_indices"],
                "val_indices": selected_source["val_indices"],
                "selection_stage": "final_gvae_by_downstream_ddpm_validation",
            },
            selected_checkpoint_path,
        )
        selected["selected_checkpoint_path"] = str(selected_checkpoint_path)
        selected_candidate_results.append(selected)

        metrics = selected["threshold_0_5_metrics"]
        best_metrics = selected["best_threshold_metrics"]
        print(
            f"  [Fold {fold}] Selected rank {selected['rank_by_val_auc']} "
            f"by DDPM {args.selection_metric}={metrics[args.selection_metric]:.4f}; "
            f"F1@0.5={metrics['f1']:.4f}, F1@best={best_metrics['f1']:.4f}, "
            f"BalAcc@best={best_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    results = summarize_results(
        selected_candidate_results,
        all_candidate_results,
        args.selection_metric,
        args.latent_representation,
    )
    results["run_id"] = run_id
    results["source_gvae_run_id"] = args.gvae_run_id
    results["source_checkpoint_dir"] = str(checkpoint_root)

    summary_path = metrics_dir / "summary.json"
    comparison_path = metrics_dir / "checkpoint_ddpm_comparison.csv"
    final_path = metrics_dir / "final_gvae_ddpm_results.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(results), f, indent=2)
    write_csv(comparison_path, flatten_candidate_results(results))

    final_payload = {
        "run_id": run_id,
        "source_gvae_run_id": args.gvae_run_id,
        "result_type": "best_ddpm_from_pretrained_gvae_checkpoints",
        "configs": {
            "model_config": json_ready(model_config),
            "train_config": json_ready(train_config),
            "ddpm_config": json_ready(ddpm_config),
        },
        "best_ddpm_by_validation_pr_auc": {
            "selection_metric": args.selection_metric,
            "fold": results.get("final_selected_fold"),
            "rank_by_gvae_checkpoint_metric": (
                results.get("final_selected_rank_by_gvae_checkpoint_metric")
            ),
            "gvae_source_checkpoint": results.get("final_selected_source_checkpoint"),
            "selected_gvae_checkpoint": results.get("final_selected_gvae_checkpoint"),
            "latent_artifact": results.get("final_selected_latents_path"),
            "ddpm_metrics_threshold_0_5": (
                results.get("final_selected_ddpm_metrics_threshold_0_5")
            ),
            "ddpm_metrics_best_threshold": (
                results.get("final_selected_ddpm_metrics_best_threshold")
            ),
        },
        "notes": [
            "GVAE was not retrained in this run.",
            "DDPM input is concat_mu latent embeddings from saved GVAE checkpoints.",
            "Classifier logits/probabilities are not DDPM inputs.",
        ],
    }
    with final_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(final_payload), f, indent=2)

    print("\n==================== DDPM SUMMARY ====================", flush=True)
    for key in [
        "mean_auc",
        "mean_pr_auc",
        "mean_f1",
        "mean_balanced_accuracy",
        "mean_f1_best_threshold",
        "mean_balanced_accuracy_best_threshold",
        "mean_specificity_best_threshold",
        "mean_recall_best_threshold",
    ]:
        if key in results:
            print(f"{key}={results[key]:.4f}", flush=True)
    print(f"SUMMARY_PATH={summary_path}", flush=True)
    print(f"COMPARISON_TABLE_PATH={comparison_path}", flush=True)
    print(f"FINAL_RESULTS_PATH={final_path}", flush=True)
    print("SUMMARY_JSON=" + json.dumps(json_ready(results), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
