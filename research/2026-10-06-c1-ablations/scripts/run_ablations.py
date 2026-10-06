"""C1 — GVAE ablation arms (config-only) with pooled-OOF + DeLong vs full.

Arms expressible by configuration alone (no core change):
  full, no_contrastive (cross_cl=0), clinical_only, pathology_only,
  radiology_only, clinical_pathology, clinical_radiology, pathology_radiology.

Each arm is evaluated with the canonical pooled-OOF protocol
(`training.train_gvae.kfold_evaluate_gvae_classifier`) on the canonical r32
graph, seeds 42-46, so scores are directly comparable to the canonical run and
to each other (same splits). Paired DeLong tests (with Holm correction across
arms) compare every arm's fusion head to `full`.

The graph / no-GNN (MLP) and lesion-pooling (attention vs mean/max) arms need
opt-in model flags and are NOT covered here.

Usage:
    .venv/bin/python research/2026-10-06-c1-ablations/scripts/run_ablations.py \
        --data data_ln_pc_ihc_g_r32.pt --seeds 42,43,44,45,46
    # fast validation of every arm (tiny epochs):
    .venv/bin/python research/2026-10-06-c1-ablations/scripts/run_ablations.py --smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from review_fixes_2026_07.bootstrap_ci import bootstrap_ci  # noqa: E402
from review_fixes_2026_07.delong import delong_paired_test  # noqa: E402
from training.train_gvae import kfold_evaluate_gvae_classifier  # noqa: E402

OUT_ROOT = ROOT / "research/2026-10-06-c1-ablations/output"

# arm -> views (None = all views, unchanged)
ARM_VIEWS = {
    "clinical_only": ("clinical",),
    "pathology_only": ("pathology",),
    "radiology_only": ("radiology",),
    "clinical_pathology": ("clinical", "pathology"),
    "clinical_radiology": ("clinical", "radiology"),
    "pathology_radiology": ("pathology", "radiology"),
}
ARMS = ["full", "no_contrastive"] + list(ARM_VIEWS)


def base_configs(data):
    """Canonical pooled-OOF config, verbatim from run_oof.py."""
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
        "vectorized_contrastive": True,
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


def arm_configs(data, arm):
    model_config, train_config = base_configs(data)
    if arm == "full":
        return model_config, train_config
    if arm == "no_contrastive":
        train_config["loss_weights"]["cross_cl"] = 0.0
        train_config["annealing"]["cross_cl"] = {
            "start_weight": 0.0, "end_weight": 0.0, "start_epoch": 0, "end_epoch": 0}
        return model_config, train_config
    views = ARM_VIEWS[arm]
    model_config["view_configs"] = {v: model_config["view_configs"][v] for v in views}
    if "radiology" not in views:
        model_config["radiology_aggregator_config"] = {}
    return model_config, train_config


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_ln_pc_ihc_g_r32.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--tag", default="ablation")
    ap.add_argument("--smoke", action="store_true", help="tiny epochs to validate all arms")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    data = torch.load(ROOT / args.data, weights_only=False)
    assert data["lesion"].x.shape[1] == 32, f"expected r32 graph, got {data['lesion'].x.shape[1]}"
    y = data["patient"]["binary_label"].numpy().astype(int)

    out_root = OUT_ROOT / args.tag if not args.smoke else OUT_ROOT / "smoke"
    out_root.mkdir(parents=True, exist_ok=True)

    scores = {}  # (arm, seed) -> head OOF probs
    for arm in ARMS:
        model_config, train_config = arm_configs(data, arm)
        if args.smoke:
            train_config["epochs"] = 4
            train_config["pretrain_epochs"] = 4
            train_config["select_after_epoch"] = 1
            train_config["n_bootstrap"] = 50
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            mc = copy.deepcopy(model_config)
            tc = copy.deepcopy(train_config)
            tc["random_seed"] = seed
            summary, _df, oof = kfold_evaluate_gvae_classifier(data, mc, tc)
            assert (oof["y_true"] == y).all(), f"{arm} seed {seed}: label order changed"
            scores[(arm, seed)] = oof["head_probs"]
            d = out_root / f"{arm}_seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "oof_arrays.npz", y_true=oof["y_true"],
                                head_probs=oof["head_probs"], probe_probs=oof["probe_probs"])
            print(f"[{arm:20s} seed {seed}] head AUC {summary['head_auc']:.4f} "
                  f"AUPRC {summary['head_auprc']:.4f} | probe AUC {summary['probe_auc']:.4f}", flush=True)

    # Per-(arm,seed) metrics with bootstrap CIs.
    rows = []
    for arm in ARMS:
        for seed in seeds:
            s = scores[(arm, seed)]
            auc = bootstrap_ci(y, s, roc_auc_score, n_boot=2000, seed=4200)
            pr = bootstrap_ci(y, s, average_precision_score, n_boot=2000, seed=4200)
            rows.append({"arm": arm, "seed": seed, "n": int(len(y)),
                         "roc_auc": auc[0], "roc_auc_lo": auc[1], "roc_auc_hi": auc[2],
                         "pr_auc": pr[0], "pr_auc_lo": pr[1], "pr_auc_hi": pr[2]})
    metrics = pd.DataFrame(rows)

    # DeLong: each non-full arm vs full, per seed, Holm within seed; plus seed-mean.
    tests = []
    for seed in list(seeds) + ["seed_mean"]:
        res = []
        for arm in ARMS:
            if arm == "full":
                continue
            if seed == "seed_mean":
                a = np.mean([scores[("full", s)] for s in seeds], axis=0)
                c = np.mean([scores[(arm, s)] for s in seeds], axis=0)
            else:
                a, c = scores[("full", seed)], scores[(arm, seed)]
            res.append({"reference": "full", "arm": arm, "seed": seed}
                       | delong_paired_test(y, a, c))
        for r, padj in zip(res, holm([r["p_value"] for r in res])):
            r["p_holm"] = float(padj)
        tests.extend(res)
    tests_df = pd.DataFrame(tests)

    summary_rows = []
    for arm in ARMS:
        m = metrics[metrics.arm == arm]
        row = {"arm": arm,
               "roc_auc_mean": m.roc_auc.mean(), "roc_auc_sd": m.roc_auc.std(ddof=1),
               "pr_auc_mean": m.pr_auc.mean(), "pr_auc_sd": m.pr_auc.std(ddof=1),
               "full_minus_arm_mean": 0.0, "seeds_full_better": len(seeds),
               "seed_mean_delta": 0.0, "seed_mean_p_holm": float("nan")}
        if arm != "full":  # `full` is only the DeLong reference, so it has no tests_df rows
            t = tests_df[(tests_df.arm == arm) & (tests_df.seed != "seed_mean")]
            sm = tests_df[(tests_df.arm == arm) & (tests_df.seed == "seed_mean")].iloc[0]
            row.update({
                "full_minus_arm_mean": t.delta_auc.mean(),
                "seeds_full_better": int((t.delta_auc > 0).sum()),
                "seed_mean_delta": sm.delta_auc,
                "seed_mean_p_holm": sm.p_holm,
            })
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    metrics.to_csv(out_root / "metrics_per_seed.csv", index=False)
    tests_df.to_csv(out_root / "delong_tests.csv", index=False)
    summary.to_csv(out_root / "summary.csv", index=False)
    json.dump({"data": args.data, "arms": ARMS, "seeds": seeds},
              open(out_root / "run_meta.json", "w"), indent=2)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\n" + summary.round(4).to_string(index=False))
    print("ABLATION_DONE", args.tag, seeds)


if __name__ == "__main__":
    main()
