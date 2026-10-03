"""A4 — TSTR control: downstream trained on synthetic-only, tested on val real.

Loads saved A2 generated latents (per fold/mode/ratio), trains the same
logistic-regression downstream on synthetic latents ONLY (no real train
samples), and evaluates on the real validation split. Compared against the
real-only baseline (train on real train -> test real val).

Question: how much of the classification signal do the synthetic latents carry?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.latent_ddpm_augmentation import train_downstream_classifier  # noqa: E402

A2_RUN = "conditional_latent_ddpm_from_gvae_bestparam_ranked_20260807_095233_20260810_153849"
OUTPUT_ROOT = Path("outputs/conditional_latent_ddpm")


def parse_args(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=A2_RUN,
                        help="DDPM run dir under outputs/conditional_latent_ddpm (default: historical A2 run)")
    args = parser.parse_args(argv)
    run_root = OUTPUT_ROOT / args.run_id
    if not run_root.is_dir():
        parser.error(f"run dir not found: {run_root}")
    return run_root


def rank1_dir(fold_dir: Path) -> Path:
    rank_dirs = sorted(fold_dir.glob("rank_*"))
    rank_dirs = [p for p in rank_dirs if p.name.startswith("rank_1_")] or rank_dirs
    assert len(rank_dirs) == 1, f"expected 1 rank dir, got {len(rank_dirs)} in {fold_dir}"
    return rank_dirs[0]


# TSTR is only meaningful for both_classes: minority_only / nonresponder_only
# generate a single class, which a binary downstream cannot be trained on.
MODES = ["both_classes"]
RATIOS = ["25", "50", "100", "200"]
CLASSIFIER_CONFIG = {"type": "logistic_regression", "class_weight": "balanced", "random_state": 42}


def load_real(base: Path):
    real = torch.load(base / "real_latents_for_ddpm.pt", map_location="cpu", weights_only=False)
    splits = real["splits"]
    return (
        splits["train"]["concat_mu"].numpy(),
        splits["train"]["labels"].numpy().reshape(-1),
        splits["val"]["concat_mu"].numpy(),
        splits["val"]["labels"].numpy().reshape(-1),
    )


def main(argv=None):
    run_root = parse_args(argv)
    rows = []
    for fold_dir in sorted(run_root.glob("fold_*")):
        base = rank1_dir(fold_dir)
        real_tr, real_tr_y, real_val, real_val_y = load_real(base)
        # real-only baseline (for reference, per fold)
        baseline = train_downstream_classifier(real_tr, real_tr_y, real_val, real_val_y, CLASSIFIER_CONFIG)
        rows.append({
            "fold": fold_dir.name,
            "mode": "REAL_ONLY",
            "ratio": "-",
            "n_train": int(real_tr.shape[0]),
            "roc_auc": baseline["threshold_0_5"]["roc_auc"],
            "pr_auc": baseline["threshold_0_5"]["pr_auc"],
            "balanced_accuracy": baseline["threshold_0_5"]["balanced_accuracy"],
        })
        for mode in MODES:
            for ratio in RATIOS:
                gen_path = base / "generated" / mode / f"ratio_{ratio}" / "generated_latents.pt"
                if not gen_path.exists():
                    continue
                gen = torch.load(gen_path, map_location="cpu", weights_only=False)
                gen_x = gen["latents"].numpy()
                gen_y = gen["labels"].numpy().reshape(-1)
                # synthetic-only training
                metrics = train_downstream_classifier(gen_x, gen_y, real_val, real_val_y, CLASSIFIER_CONFIG)
                rows.append({
                    "fold": fold_dir.name,
                    "mode": mode,
                    "ratio": ratio,
                    "n_train": int(gen_x.shape[0]),
                    "roc_auc": metrics["threshold_0_5"]["roc_auc"],
                    "pr_auc": metrics["threshold_0_5"]["pr_auc"],
                    "balanced_accuracy": metrics["threshold_0_5"]["balanced_accuracy"],
                })
                print(
                    f"{fold_dir.name} {mode:18s} r{ratio:>4s} n_train={int(gen_x.shape[0]):>4} "
                    f"ROC={metrics['threshold_0_5']['roc_auc']:.4f} "
                    f"PR={metrics['threshold_0_5']['pr_auc']:.4f}",
                    flush=True,
                )

    out_path = run_root / "a4_tstr_control.json"
    out_path.write_text(json.dumps(rows, indent=2, default=float))
    print(f"A4 results -> {out_path}")

    print("\n=== TSTR AGGREGATE (mean over folds) ===")
    print(f"{'mode':<18} {'ratio':>5} {'n_train':>8} {'ROC':>7} {'PR':>7} {'BA':>7}")
    keys = sorted({(r["mode"], r["ratio"]) for r in rows})
    for mode, ratio in keys:
        sub = [r for r in rows if r["mode"] == mode and r["ratio"] == ratio]
        n = np.mean([r["n_train"] for r in sub])
        roc = np.mean([r["roc_auc"] for r in sub])
        pr = np.mean([r["pr_auc"] for r in sub])
        ba = np.mean([r["balanced_accuracy"] for r in sub])
        print(f"{mode:<18} {ratio:>5} {n:>8.0f} {roc:>7.4f} {pr:>7.4f} {ba:>7.4f}")


if __name__ == "__main__":
    main()
