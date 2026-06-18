import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.latent_ddpm_augmentation import (  # noqa: E402
    DEFAULT_AUGMENTATION_MODES,
    DEFAULT_AUGMENTATION_RATIOS,
    run_conditional_latent_augmentation_pipeline,
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


def latest_gvae_run_id(checkpoint_root: Path) -> str:
    run_dirs = [path for path in checkpoint_root.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No GVAE checkpoint runs found under {checkpoint_root}")
    return sorted(run_dirs, key=lambda path: path.stat().st_mtime)[-1].name


def fold_from_checkpoint(path: Path) -> int:
    match = re.search(r"_fold_(\d+)_", path.name)
    if match:
        return int(match.group(1))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("fold", 0))


def discover_checkpoints(
    checkpoint_root: Path,
    gvae_run_id: str,
    selector: str,
    rank: int,
    max_folds: int | None,
) -> list[Path]:
    run_dir = checkpoint_root / gvae_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"GVAE checkpoint run not found: {run_dir}")

    if selector == "best":
        checkpoint_paths = sorted(run_dir.glob(f"{gvae_run_id}_fold_*_best.pt"))
    elif selector == "rank":
        checkpoint_paths = sorted(run_dir.glob(f"{gvae_run_id}_fold_*_rank_{rank}_*.pt"))
    else:
        raise ValueError(f"Unsupported checkpoint selector: {selector}")

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found for selector={selector}, rank={rank} in {run_dir}"
        )

    checkpoint_paths = sorted(checkpoint_paths, key=fold_from_checkpoint)
    if max_folds is not None:
        checkpoint_paths = checkpoint_paths[:max_folds]
    return checkpoint_paths


def parse_ratios(raw: str | None) -> list[float]:
    if not raw:
        return list(DEFAULT_AUGMENTATION_RATIOS)
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_modes(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_AUGMENTATION_MODES)
    return [item.strip().replace("-", "_") for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train conditional DDPM p(z|y) on GVAE latents, generate synthetic "
            "latents, and evaluate augmentation with a downstream classifier."
        )
    )
    parser.add_argument("--data-path", default="data_ln_pc_ihc_g.pt")
    parser.add_argument("--gvae-run-id", default=None)
    parser.add_argument("--checkpoint-root", default="outputs/gvae/checkpoints")
    parser.add_argument("--checkpoint-selector", choices=["best", "rank"], default="rank")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/conditional_latent_ddpm")
    parser.add_argument("--latent-key", default="concat_mu")
    parser.add_argument("--ratios", default="0.25,0.50,1.00,2.00")
    parser.add_argument(
        "--augmentation-modes",
        default="both_classes",
        help="Comma-separated: minority_only,responder_only,both_classes",
    )
    parser.add_argument("--filter-synthetic", action="store_true")
    parser.add_argument("--filter-quantile", type=float, default=0.95)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-prob", type=float, default=0.2)
    parser.add_argument("--cond-drop-prob", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--beta-schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--classifier", choices=["logistic_regression", "random_forest"], default="logistic_regression")
    parser.add_argument("--overwrite-latents", action="store_true")
    args = parser.parse_args()

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint_root = Path(args.checkpoint_root)
    gvae_run_id = args.gvae_run_id or latest_gvae_run_id(checkpoint_root)
    checkpoint_paths = discover_checkpoints(
        checkpoint_root=checkpoint_root,
        gvae_run_id=gvae_run_id,
        selector=args.checkpoint_selector,
        rank=args.rank,
        max_folds=args.max_folds,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    full_data = torch.load(args.data_path, map_location="cpu", weights_only=False)

    output_run_id = (
        f"conditional_latent_ddpm_from_{gvae_run_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_root) / output_run_id

    ddpm_config = {
        "epochs": args.epochs,
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "sample_batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout_prob": args.dropout_prob,
        "cond_drop_prob": args.cond_drop_prob,
        "guidance_scale": args.guidance_scale,
        "beta_schedule": args.beta_schedule,
        "dim_mults": [1, 2],
        "grad_clip_norm": 1.0,
    }
    classifier_config = {
        "type": args.classifier,
        "class_weight": "balanced",
        "random_state": seed,
    }
    run_config = {
        "run_id": output_run_id,
        "objective": "conditional_latent_space_augmentation",
        "ddpm_is_classifier": False,
        "gvae_run_id": gvae_run_id,
        "checkpoint_selector": args.checkpoint_selector,
        "checkpoint_rank": args.rank if args.checkpoint_selector == "rank" else None,
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "latent_key": args.latent_key,
        "augmentation_ratios": parse_ratios(args.ratios),
        "augmentation_modes": parse_modes(args.augmentation_modes),
        "filter_synthetic": args.filter_synthetic,
        "filter_quantile": args.filter_quantile,
        "pca_components": args.pca_components,
        "ddpm_config": ddpm_config,
        "classifier_config": classifier_config,
        "device": device,
        "seed": seed,
        "data_path": args.data_path,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(run_config), f, indent=2)

    print(f"RUN_ID={output_run_id}", flush=True)
    print(f"OBJECTIVE=conditional_latent_space_augmentation", flush=True)
    print(f"DDPM_IS_CLASSIFIER=False", flush=True)
    print(f"GVAE_RUN_ID={gvae_run_id}", flush=True)
    print(f"CHECKPOINTS={len(checkpoint_paths)}", flush=True)
    print(f"OUTPUT_DIR={output_dir}", flush=True)

    summary = run_conditional_latent_augmentation_pipeline(
        checkpoint_paths=checkpoint_paths,
        full_data=full_data,
        output_dir=output_dir,
        ratios=parse_ratios(args.ratios),
        augmentation_modes=parse_modes(args.augmentation_modes),
        latent_key=args.latent_key,
        pca_components=args.pca_components,
        filter_config={
            "enabled": args.filter_synthetic,
            "quantile": args.filter_quantile,
        },
        ddpm_config=ddpm_config,
        classifier_config=classifier_config,
        device=device,
        overwrite_latents=args.overwrite_latents,
        sample_seed=seed,
    )

    summary_path = output_dir / "summary.json"
    comparison_path = output_dir / "augmentation_comparison.csv"
    print(f"SUMMARY_PATH={summary_path}", flush=True)
    print(f"COMPARISON_PATH={comparison_path}", flush=True)
    print("SUMMARY_JSON=" + json.dumps(json_ready(summary.get("aggregate", {})), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
