"""C3 — hyperparameter sensitivity on the canonical r32 GVAE.

One-axis perturbations around the canonical pooled-OOF config, each evaluated with
the same protocol (`kfold_evaluate_gvae_classifier`), seeds 42-46, with paired
DeLong vs the canonical config (Holm-corrected).

Axes (per `PUBLICATION_CHECKLIST` C3): cross-cl temperature (tau), embedding dim,
attention heads, hidden width, contrastive weight. The graph-threshold 0.7 vs 0.8
axis is handled separately (it needs a rebuilt graph).

Usage:
    .venv/bin/python research/2026-10-06-c3-sensitivity/scripts/run_sensitivity.py \
        --data data_ln_pc_ihc_g_r32.pt --seeds 42,43,44,45,46
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

OUT_ROOT = ROOT / "research/2026-10-06-c3-sensitivity/output"
VARIANTS = ["canonical", "tau02", "d16", "d64", "heads4", "hidden64", "cl02"]


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


def apply_variant(model_config, train_config, name):
    mc, tc = copy.deepcopy(model_config), copy.deepcopy(train_config)
    if name == "canonical":
        pass
    elif name == "tau02":
        tc["cross_cl_temp"] = 0.2
    elif name == "d16":
        mc["d_embed"] = 16
        mc["projection_head_config"]["output_dim"] = 16
    elif name == "d64":
        mc["d_embed"] = 64
        mc["projection_head_config"]["output_dim"] = 64
    elif name == "heads4":
        for v in mc["view_configs"].values():
            v["heads"] = 4
    elif name == "hidden64":
        for v in mc["view_configs"].values():
            v["hidden_channels_vae"] = 64
    elif name == "cl02":
        tc["loss_weights"]["cross_cl"] = 0.2
    else:
        raise ValueError(name)
    return mc, tc


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
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--tag", default="c3")
    ap.add_argument("--smoke", action="store_true", help="tiny epochs to validate all variants")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    data = torch.load(ROOT / args.data, weights_only=False)
    y = data["patient"]["binary_label"].numpy().astype(int)
    out_root = OUT_ROOT / (args.tag if not args.smoke else "smoke")
    out_root.mkdir(parents=True, exist_ok=True)
    model_config, train_config = base_configs(data)
    if args.smoke:
        train_config["epochs"] = 4
        train_config["pretrain_epochs"] = 4
        train_config["select_after_epoch"] = 1
        train_config["n_bootstrap"] = 50

    scores = {}
    for name in VARIANTS:
        mc0, tc0 = apply_variant(model_config, train_config, name)
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            mc, tc = copy.deepcopy(mc0), copy.deepcopy(tc0)
            tc["random_seed"] = seed
            summary, _df, oof = kfold_evaluate_gvae_classifier(data, mc, tc)
            assert (oof["y_true"] == y).all()
            scores[(name, seed)] = oof["head_probs"]
            print(f"[{name:10s} seed {seed}] head AUC {summary['head_auc']:.4f} AUPRC {summary['head_auprc']:.4f}",
                  flush=True)

    rows = []
    for name in VARIANTS:
        for seed in seeds:
            s = scores[(name, seed)]
            auc = bootstrap_ci(y, s, roc_auc_score, n_boot=2000, seed=4200)
            pr = bootstrap_ci(y, s, average_precision_score, n_boot=2000, seed=4200)
            rows.append({"config": name, "seed": seed, "roc_auc": auc[0],
                         "roc_auc_lo": auc[1], "roc_auc_hi": auc[2],
                         "pr_auc": pr[0], "pr_auc_lo": pr[1], "pr_auc_hi": pr[2]})
    metrics = pd.DataFrame(rows)

    tests = []
    for seed in list(seeds) + ["seed_mean"]:
        res = []
        for name in VARIANTS:
            if name == "canonical":
                continue
            if seed == "seed_mean":
                a = np.mean([scores[("canonical", s)] for s in seeds], axis=0)
                c = np.mean([scores[(name, s)] for s in seeds], axis=0)
            else:
                a, c = scores[("canonical", seed)], scores[(name, seed)]
            res.append({"reference": "canonical", "config": name, "seed": seed}
                       | delong_paired_test(y, a, c))
        for r, padj in zip(res, holm([r["p_value"] for r in res])):
            r["p_holm"] = float(padj)
        tests.extend(res)
    tests_df = pd.DataFrame(tests)

    summary_rows = []
    for name in VARIANTS:
        m = metrics[metrics.config == name]
        row = {"config": name, "roc_auc_mean": m.roc_auc.mean(), "roc_auc_sd": m.roc_auc.std(ddof=1),
               "pr_auc_mean": m.pr_auc.mean(), "pr_auc_sd": m.pr_auc.std(ddof=1),
               "delta_vs_canonical": 0.0, "p_holm": float("nan")}
        if name != "canonical":
            t = tests_df[(tests_df.config == name) & (tests_df.seed != "seed_mean")]
            sm = tests_df[(tests_df.config == name) & (tests_df.seed == "seed_mean")].iloc[0]
            row["delta_vs_canonical"] = t.delta_auc.mean()
            row["p_holm"] = sm.p_holm
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    metrics.to_csv(out_root / "metrics_per_seed.csv", index=False)
    tests_df.to_csv(out_root / "delong_tests.csv", index=False)
    summary.to_csv(out_root / "summary.csv", index=False)
    json.dump({"data": args.data, "variants": VARIANTS, "seeds": seeds},
              open(out_root / "run_meta.json", "w"), indent=2)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n" + summary.round(4).to_string(index=False))
    print("SENSITIVITY_DONE", args.tag, seeds)


if __name__ == "__main__":
    main()
