"""C2 — DyAM baseline (deep-multipit / Vanguri et al. 2022) on the GVAE pooled-OOF splits.

Faithful re-implementation of the DyAM late-fusion model from
`sysbio-curie/deep-multipit`:
  * `LateAttentionFusion`: each modality -> a linear predictor with a tanh output;
    missing modalities are zeroed via a mask; the final logit is the attention-weighted
    sum of the unimodal logits.
  * `MSKCCAttention`: per-modality `Linear(d,1)(x)/d`, softplus, mask, L1-normalise.

Our 3 views map to 3 modalities (same per-patient feature blocks as the other C2
baselines in `research/2026-10-04-baselines-delong/scripts/run_baselines.py`):
  clinical (22) | pathology (15 + presence) | radiology (mean+max lesion + presence).
Mask = view presence; absent views are zeroed and excluded from the softmax.

Protocol = the GVAE pooled-OOF protocol (outer StratifiedKFold(5, seed), seeds 42-46;
inner train_test_split(0.2, eval_seed) selects lr/weight-decay by inner-val AUC).
A per-fold model trained on inner-train (data parity with the GVAE) scores the outer
test fold. Bootstrap CIs + DeLong vs the canonical GVAE head/probe.

Usage:
    .venv/bin/python research/2026-10-06-c2-dyam/scripts/run_dyam.py --seeds 42,43,44,45,46
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/2026-10-04-baselines-delong/scripts"))

from review_fixes_2026_07.bootstrap_ci import bootstrap_ci  # noqa: E402
from review_fixes_2026_07.delong import delong_paired_test  # noqa: E402
from run_baselines import build_patient_features, holm  # noqa: E402

GVAE_OOF = ROOT / "research/2026-10-03-radiology-artifact-ablation/output/drop_both32_seed{seed}/oof_arrays.npz"
OUT_ROOT = ROOT / "research/2026-10-06-c2-dyam/output"
SEEDS = (42, 43, 44, 45, 46)
N_SPLITS, INNER_VAL, EVAL_SEED = 5, 0.2, 4200


class DyAM(nn.Module):
    """Late attention fusion (deep-multipit `LateAttentionFusion` + `MSKCCAttention`)."""

    def __init__(self, dims):
        super().__init__()
        self.dims = list(dims)
        self.emb = nn.ModuleList([nn.Sequential(nn.Linear(d, 1), nn.Tanh()) for d in dims])
        self.att = nn.ModuleList([nn.Linear(d, 1) for d in dims])
        self.softplus = nn.Softplus()
        for layer in self.att:
            nn.init.zeros_(layer.bias)

    def forward(self, xs, mask):
        logits = torch.cat([emb(x) for x, emb in zip(xs, self.emb)], dim=-1)  # (B, M)
        logits = torch.where(mask, logits, torch.zeros_like(logits))
        att = torch.stack([self.att[i](xs[i]) / self.dims[i] for i in range(len(xs))], dim=1).squeeze(-1)
        att = torch.where(mask, self.softplus(att), torch.zeros_like(att))
        attn = F.normalize(att, p=1, dim=-1)
        return (attn * logits).sum(1), attn


def _train_eval(xs_tr, m_tr, y_tr, xs_val, m_val, y_val, dims, lr, wd,
                epochs=200, patience=30, batch=64, warmup=10):
    torch.manual_seed(0)
    model = DyAM(dims)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    n = xs_tr[0].shape[0]
    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits, _ = model([x[idx] for x in xs_tr], m_tr[idx])
            loss = lossf(logits, y_tr[idx])
            loss.backward(); opt.step()
        if epoch >= warmup:
            model.eval()
            with torch.no_grad():
                vl, _ = model(xs_val, m_val)
                auc = roc_auc_score(y_val, torch.sigmoid(vl).numpy()) if len(np.unique(y_val)) > 1 else -1.0
            if auc > best_auc:
                best_auc, best_state, no_improve = auc, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_auc


def _predict(model, xs, mask):
    with torch.no_grad():
        logits, _ = model(xs, mask)
    return torch.sigmoid(logits).numpy()


def dyam_oof(xs_list, masks, y, seed, grid, epochs=200, patience=30, warmup=10):
    """Pooled-OOF scores; grid = list of (lr, wd) selected per fold on inner-val AUC."""
    oof = np.full(len(y), np.nan)
    chosen = []
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(len(y)), y):
        itr, iva = train_test_split(tr, test_size=INNER_VAL, stratify=y[tr], random_state=EVAL_SEED)
        sc = [StandardScaler().fit(x[itr]) for x in xs_list]
        xtr = [torch.tensor(sc[i].transform(xs_list[i][itr]), dtype=torch.float32) for i in range(len(xs_list))]
        xva = [torch.tensor(sc[i].transform(xs_list[i][iva]), dtype=torch.float32) for i in range(len(xs_list))]
        xte = [torch.tensor(sc[i].transform(xs_list[i][te]), dtype=torch.float32) for i in range(len(xs_list))]
        dims = [x.shape[1] for x in xtr]
        ytr = torch.tensor(y[itr], dtype=torch.float32)
        best, best_auc, best_model = None, -np.inf, None
        for lr, wd in grid:
            model, auc = _train_eval(xtr, masks[itr], ytr, xva, masks[iva], y[iva], dims, lr, wd,
                                     epochs=epochs, patience=patience, warmup=warmup)
            if auc > best_auc:
                best, best_auc, best_model = (lr, wd), auc, model
        oof[te] = _predict(best_model, xte, masks[te])
        chosen.append({"fold_params": best, "inner_val_auc": float(best_auc)})
    assert not np.isnan(oof).any()
    return oof, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_ln_pc_ihc_g_r32.pt")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--tag", default="dyam")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out_root = OUT_ROOT / (args.tag if not args.smoke else "smoke")
    out_root.mkdir(parents=True, exist_ok=True)

    data = torch.load(ROOT / args.data, weights_only=False)
    blocks, y = build_patient_features(data)
    dims_order = ["clinical", "pathology", "radiology"]
    xs_list = [blocks[k].astype(np.float32) for k in dims_order]
    masks = torch.tensor(np.stack([
        np.ones(len(y), bool),
        data["patient"].pathology_mask.numpy().astype(bool),
        data["patient"].radiology_mask.numpy().astype(bool),
    ], axis=1))
    grid = [(1e-3, 1e-4)] if args.smoke else [(lr, wd) for lr in (1e-3, 3e-3) for wd in (1e-4, 1e-3)]

    scores = {}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        s, chosen = dyam_oof(xs_list, masks, y, seed, grid,
                             epochs=5 if args.smoke else 200, patience=5 if args.smoke else 30,
                             warmup=1 if args.smoke else 10)
        scores[seed] = s
        d = out_root / f"dyam_seed{seed}"; d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "oof_arrays.npz", y_true=y, scores=s)
        print(f"[dyam seed {seed}] OOF AUC {roc_auc_score(y, s):.4f} AUPRC {average_precision_score(y, s):.4f} "
              f"| folds={[c['fold_params'] for c in chosen]}", flush=True)

    rows = []
    for seed in seeds:
        s = scores[seed]
        auc = bootstrap_ci(y, s, roc_auc_score, n_boot=2000, seed=EVAL_SEED)
        pr = bootstrap_ci(y, s, average_precision_score, n_boot=2000, seed=EVAL_SEED)
        rows.append({"seed": seed, "roc_auc": auc[0], "roc_auc_lo": auc[1], "roc_auc_hi": auc[2],
                     "pr_auc": pr[0], "pr_auc_lo": pr[1], "pr_auc_hi": pr[2]})
    metrics = pd.DataFrame(rows)

    # DeLong vs canonical GVAE head/probe on identical splits.
    tests = []
    for seed in list(seeds) + ["seed_mean"]:
        if seed == "seed_mean":
            gs = {r: np.mean([np.load(str(GVAE_OOF).format(seed=s))[r] for s in seeds], axis=0)
                  for r in ("head_probs", "probe_probs")}
            for s in seeds:
                assert (np.load(str(GVAE_OOF).format(seed=s))["y_true"] == y).all(), \
                    f"GVAE OOF seed {s} label order differs from {args.data}"
            ds = np.mean([scores[s] for s in seeds], axis=0)
        for ref in ("head", "probe"):
            if seed == "seed_mean":
                a, c = gs[f"{ref}_probs"], ds
            else:
                g = np.load(str(GVAE_OOF).format(seed=seed))
                assert (g["y_true"] == y).all(), f"GVAE OOF seed {seed} label order differs from {args.data}"
                a, c = g[f"{ref}_probs"], scores[seed]
            tests.append({"reference": f"gvae_{ref}", "seed": seed} | delong_paired_test(y, a, c))
    tests_df = pd.DataFrame(tests)
    tests_df["p_holm"] = np.nan
    for seed in list(seeds) + ["seed_mean"]:
        m = tests_df.seed == seed
        tests_df.loc[m, "p_holm"] = holm(tests_df.loc[m, "p_value"].values)

    metrics.to_csv(out_root / "metrics_per_seed.csv", index=False)
    tests_df.to_csv(out_root / "delong_vs_gvae.csv", index=False)
    summary = {"roc_auc_mean": float(metrics.roc_auc.mean()), "roc_auc_sd": float(metrics.roc_auc.std(ddof=1)),
               "pr_auc_mean": float(metrics.pr_auc.mean()), "pr_auc_sd": float(metrics.pr_auc.std(ddof=1))}
    json.dump({"data": args.data, "seeds": seeds, "grid": grid, "summary": summary},
              open(out_root / "dyam_summary.json", "w"), indent=2)
    print("\nDYAM", json.dumps(summary, indent=2))
    print("DYAM_DONE")


if __name__ == "__main__":
    main()
