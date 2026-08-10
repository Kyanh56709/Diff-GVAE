"""A3 — Retune filter_quantile post-hoc on saved A2 latents.

Loads the saved real/generated latents from the A2 full rerun
(conditional_latent_ddpm_from_gvae_bestparam_ranked_20260807_095233_20260810_153849)
and re-applies the same-class train-kNN filter at quantiles
[0.7, 0.8, 0.9, 0.95], then evaluates the downstream classifier
(real train + kept synthetic -> val real) exactly like the pipeline.

No DDPM retraining, no modification of pipeline code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.latent_ddpm_augmentation import (  # noqa: E402
    filter_synthetic_latents_by_knn,
    train_downstream_classifier,
)

A2_RUN = "conditional_latent_ddpm_from_gvae_bestparam_ranked_20260807_095233_20260810_153849"
RUN_ROOT = Path("outputs/conditional_latent_ddpm") / A2_RUN
QUANTILES = [0.70, 0.80, 0.90, 0.95, 0.97, 0.99]
MODES = ["both_classes", "minority_only", "nonresponder_only"]
RATIOS = ["25", "50", "100", "200"]  # pipeline ratio dir names (percent)
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


def main():
    rows = []
    for fold_dir in sorted(RUN_ROOT.glob("fold_*")):
        rank_dirs = sorted(fold_dir.glob("rank_*"))
        assert len(rank_dirs) == 1, f"expected 1 rank dir, got {len(rank_dirs)} in {fold_dir}"
        base = rank_dirs[0]
        real_tr, real_tr_y, real_val, real_val_y = load_real(base)
        for mode in MODES:
            for ratio in RATIOS:
                gen_path = base / "generated" / mode / f"ratio_{ratio}" / "generated_latents.pt"
                if not gen_path.exists():
                    continue
                gen = torch.load(gen_path, map_location="cpu", weights_only=False)
                gen_x = gen["latents"].numpy()
                gen_y = gen["labels"].numpy().reshape(-1)
                for q in QUANTILES:
                    filtered = filter_synthetic_latents_by_knn(
                        real_tr, real_tr_y, gen_x, gen_y, quantile=q
                    )
                    kept_x = filtered["latents"]
                    kept_y = filtered["labels"]
                    # Mirror the pipeline: always evaluate the downstream even when
                    # 0 samples are kept (degenerates to the real-only baseline),
                    # so kept=0 folds are NOT dropped from quantile means.
                    metrics = train_downstream_classifier(
                        np.concatenate([real_tr, kept_x], axis=0),
                        np.concatenate([real_tr_y, kept_y], axis=0),
                        real_val,
                        real_val_y,
                        CLASSIFIER_CONFIG,
                    )
                    roc = metrics["threshold_0_5"]["roc_auc"]
                    pr = metrics["threshold_0_5"]["pr_auc"]
                    ba = metrics["threshold_0_5"]["balanced_accuracy"]
                    rows.append(
                        {
                            "fold": fold_dir.name,
                            "mode": mode,
                            "ratio": ratio,
                            "quantile": q,
                            "generated": int(gen_x.shape[0]),
                            "kept": int(kept_x.shape[0]),
                            "kept_frac": filtered["kept_fraction"],
                            "roc_auc": roc,
                            "pr_auc": pr,
                            "balanced_accuracy": ba,
                        }
                    )
                    print(
                        f"{fold_dir.name} {mode:18s} r{ratio:>5s} q{q:.2f} "
                        f"kept {int(kept_x.shape[0]):>3}/{int(gen_x.shape[0])} "
                        f"ROC={roc if roc is None else round(roc, 4)}",
                        flush=True,
                    )

    out_path = RUN_ROOT / "a3_filter_quantile_retune.json"
    out_path.write_text(json.dumps(rows, indent=2, default=float))
    print(f"A3 results -> {out_path}")

    # Aggregate: mean kept / mean ROC across folds per (mode, ratio, quantile)
    print("\n=== AGGREGATE (mean over folds) ===")
    print(f"{'mode':<18} {'ratio':>5} {'q':>4} {'kept_mean':>9} {'roc_mean':>9}")
    keys = sorted({(r["mode"], r["ratio"], r["quantile"]) for r in rows})
    for mode, ratio, q in keys:
        sub = [r for r in rows if (r["mode"], r["ratio"], r["quantile"]) == (mode, ratio, q)]
        kept = np.mean([r["kept"] for r in sub])
        rocs = [r["roc_auc"] for r in sub if r["roc_auc"] is not None]
        roc = np.mean(rocs) if rocs else None
        n_folds = len([r for r in sub if r["roc_auc"] is not None])
        print(f"{mode:<18} {ratio:>5} {q:>4.2f} {kept:>9.1f} {roc if roc is None else round(roc,4):>9} ({n_folds}/5 folds)")


if __name__ == "__main__":
    main()
