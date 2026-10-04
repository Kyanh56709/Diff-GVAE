"""C2 baselines + B5 DeLong tests against the canonical r32 GVAE pooled-OOF.

Every baseline uses the GVAE pooled-OOF splits exactly
(training.train_gvae.kfold_evaluate_gvae_classifier):
  outer : StratifiedKFold(5, shuffle=True, random_state=seed), seeds 42-46
  inner : train_test_split(outer_train, test_size=0.2, stratify, random_state=4200)
Hyperparameters are picked by inner-val ROC-AUC (fit on inner-train). The chosen
config then scores the outer-test fold after being refit on
  --refit outer_train : the whole outer-train fold (standard; baselines see ~25%
                        more data than the GVAE, which trains on inner-train only)
  --refit inner_train : inner-train only (data parity with the GVAE)
Scalers live inside the sklearn Pipeline, so they are fit on train only.

Patient features (no graph):
  clinical  : x_clinical (22)
  pathology : x_pathology (15, zeros when absent) + presence flag
  radiology : mean and max of lesion.x over the patient's lesions (2 x 32,
              zeros when absent) + presence flag
  all       : concatenation of the three (103) = late-fusion input

GVAE scores are read from the frozen pooled-OOF runs
research/2026-10-03-radiology-artifact-ablation/output/drop_both32_seed{seed}/.

Usage:
    .venv/bin/python research/2026-10-04-baselines-delong/scripts/run_baselines.py --refit outer_train
    .venv/bin/python research/2026-10-04-baselines-delong/scripts/run_baselines.py --refit inner_train
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from review_fixes_2026_07.bootstrap_ci import bootstrap_ci  # noqa: E402
from review_fixes_2026_07.delong import delong_paired_test  # noqa: E402

OUT_ROOT = ROOT / "research/2026-10-04-baselines-delong/output"
GVAE_OOF = ROOT / "research/2026-10-03-radiology-artifact-ablation/output/drop_both32_seed{seed}/oof_arrays.npz"
SEEDS = (42, 43, 44, 45, 46)
N_SPLITS, INNER_VAL, EVAL_SEED = 5, 0.2, 4200


def build_patient_features(data):
    """Per-patient feature blocks (numpy, row i = patient node i) + labels."""
    p = data["patient"]
    n = p.num_nodes
    clinical = p.x_clinical.numpy().astype(np.float64)
    path_mask = p.pathology_mask.numpy().astype(np.float64)[:, None]
    pathology = np.hstack([p.x_pathology.numpy().astype(np.float64) * path_mask, path_mask])

    lesion_x = data["lesion"].x.numpy().astype(np.float64)
    owner, lesion = data["patient", "has_lesion", "lesion"].edge_index.numpy()
    d = lesion_x.shape[1]
    rad_mean = np.zeros((n, d))
    rad_max = np.zeros((n, d))
    has = np.zeros((n, 1))
    for i in np.unique(owner):
        xs = lesion_x[lesion[owner == i]]
        rad_mean[i], rad_max[i], has[i] = xs.mean(axis=0), xs.max(axis=0), 1.0
    radiology = np.hstack([rad_mean, rad_max, has])

    blocks = {"clinical": clinical, "pathology": pathology, "radiology": radiology}
    blocks["all"] = np.hstack([clinical, pathology, radiology])
    return blocks, p.binary_label.numpy().astype(int)


def model_specs(seed):
    """name -> (feature block, estimator factory(params), param grid)."""
    return {
        "logreg_clinical": ("clinical",
                            lambda prm: LogisticRegression(class_weight="balanced", max_iter=5000, **prm),
                            {"C": [0.01, 0.1, 1.0, 10.0]}),
        "logreg_concat": ("all",
                          lambda prm: LogisticRegression(class_weight="balanced", max_iter=5000, **prm),
                          {"C": [0.01, 0.1, 1.0, 10.0]}),
        "svm_rbf_concat": ("all",
                           lambda prm: SVC(kernel="rbf", gamma="scale", class_weight="balanced", **prm),
                           {"C": [0.1, 1.0, 10.0]}),
        "gbdt_concat": ("all",
                        lambda prm: HistGradientBoostingClassifier(
                            max_iter=200, class_weight="balanced", random_state=seed, **prm),
                        {"max_depth": [2, 3], "learning_rate": [0.05, 0.1]}),
        "mlp_late_fusion": ("all",
                            lambda prm: MLPClassifier(max_iter=1000, random_state=seed, **prm),
                            {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-3, 1e-2]}),
    }


def _score(pipe, X):
    est = pipe[-1]
    if hasattr(est, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def oof_scores(X, y, factory, grid, seed, refit="outer_train"):
    """Pooled-OOF scores on the GVAE splits; returns (scores, chosen params per fold)."""
    oof = np.full(len(y), np.nan)
    chosen = []
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(np.arange(len(y)), y):
        inner_tr, inner_val = train_test_split(
            train_idx, test_size=INNER_VAL, stratify=y[train_idx], random_state=EVAL_SEED)
        best, best_auc = None, -np.inf
        for prm in ParameterGrid(grid):
            pipe = make_pipeline(StandardScaler(), factory(prm)).fit(X[inner_tr], y[inner_tr])
            auc = roc_auc_score(y[inner_val], _score(pipe, X[inner_val]))
            if auc > best_auc:
                best, best_auc = prm, auc
        fit_idx = train_idx if refit == "outer_train" else inner_tr
        pipe = make_pipeline(StandardScaler(), factory(best)).fit(X[fit_idx], y[fit_idx])
        oof[test_idx] = _score(pipe, X[test_idx])
        chosen.append({k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()}
                      | {"inner_val_auc": float(best_auc)})
    assert not np.isnan(oof).any()
    return oof, chosen


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_ln_pc_ihc_g_r32.pt")
    parser.add_argument("--refit", choices=("outer_train", "inner_train"), default="outer_train")
    args = parser.parse_args()
    out_root = OUT_ROOT / args.refit
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    data = torch.load(ROOT / args.data, weights_only=False)
    blocks, y = build_patient_features(data)
    out_root.mkdir(parents=True, exist_ok=True)

    scores = {}   # (model, seed) -> oof scores
    chosen = {}
    for seed in SEEDS:
        g = np.load(str(GVAE_OOF).format(seed=seed))
        if not (g["y_true"] == y).all():
            raise ValueError(f"GVAE OOF seed {seed}: label order differs from {args.data}")
        scores[("gvae_head", seed)] = g["head_probs"]
        scores[("gvae_probe", seed)] = g["probe_probs"]
        for name, (block, factory, grid) in model_specs(seed).items():
            s, c = oof_scores(blocks[block], y, factory, grid, seed, refit=args.refit)
            scores[(name, seed)], chosen[f"{name}_seed{seed}"] = s, c
            out = out_root / f"{name}_seed{seed}"
            out.mkdir(exist_ok=True)
            np.savez_compressed(out / "oof_arrays.npz", y_true=y, scores=s)
            print(f"seed {seed} {name:16s} OOF AUC {roc_auc_score(y, s):.4f}", flush=True)

    models = ["gvae_head", "gvae_probe"] + list(model_specs(0))
    baselines = [m for m in models if not m.startswith("gvae")]

    # Per-seed metrics with bootstrap CIs.
    metric_rows = []
    for m in models:
        for seed in SEEDS:
            s = scores[(m, seed)]
            auc = bootstrap_ci(y, s, roc_auc_score, n_boot=2000, seed=EVAL_SEED)
            pr = bootstrap_ci(y, s, average_precision_score, n_boot=2000, seed=EVAL_SEED)
            metric_rows.append({"model": m, "seed": seed,
                                "roc_auc": auc[0], "roc_auc_lo": auc[1], "roc_auc_hi": auc[2],
                                "pr_auc": pr[0], "pr_auc_lo": pr[1], "pr_auc_hi": pr[2]})
    metrics = pd.DataFrame(metric_rows)

    # DeLong: each GVAE score vs each baseline, per seed (Holm over baselines within seed),
    # plus one test on seed-averaged scores.
    test_rows = []
    for ref in ("gvae_head", "gvae_probe"):
        for seed in list(SEEDS) + ["seed_mean"]:
            res = []
            for b in baselines:
                if seed == "seed_mean":
                    a = np.mean([scores[(ref, s)] for s in SEEDS], axis=0)
                    c = np.mean([scores[(b, s)] for s in SEEDS], axis=0)
                else:
                    a, c = scores[(ref, seed)], scores[(b, seed)]
                res.append({"reference": ref, "baseline": b, "seed": seed} | delong_paired_test(y, a, c))
            for r, padj in zip(res, holm([r["p_value"] for r in res])):
                r["p_holm"] = float(padj)
            test_rows.extend(res)
    tests = pd.DataFrame(test_rows)

    # Summary: mean +- sd over seeds, and the head-vs-baseline DeLong outcome.
    summary_rows = []
    for m in models:
        mm = metrics[metrics.model == m]
        row = {"model": m,
               "roc_auc_mean": mm.roc_auc.mean(), "roc_auc_sd": mm.roc_auc.std(ddof=1),
               "pr_auc_mean": mm.pr_auc.mean(), "pr_auc_sd": mm.pr_auc.std(ddof=1)}
        if m in baselines:
            t = tests[(tests.reference == "gvae_head") & (tests.baseline == m)]
            per_seed = t[t.seed != "seed_mean"]
            sm = t[t.seed == "seed_mean"].iloc[0]
            row |= {"head_minus_baseline_mean": per_seed.delta_auc.mean(),
                    "head_better_seeds": int((per_seed.delta_auc > 0).sum()),
                    "seeds_p_lt_0.05": int((per_seed.p_value < 0.05).sum()),
                    "seed_mean_delta": sm.delta_auc, "seed_mean_p": sm.p_value,
                    "seed_mean_p_holm": sm.p_holm}
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    metrics.to_csv(out_root / "metrics_per_seed.csv", index=False)
    tests.to_csv(out_root / "delong_tests.csv", index=False)
    summary.to_csv(out_root / "summary.csv", index=False)
    json.dump({"data": args.data, "refit": args.refit, "seeds": list(SEEDS), "chosen_params": chosen},
              open(out_root / "chosen_params.json", "w"), indent=2)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
