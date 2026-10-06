"""F1/F2/F3 — interpretability on the canonical r32 GVAE.

F3  latent-space visualization: leakage-free per-fold VAL latents (`concat_mu`),
    PCA + t-SNE colored by response and by driver-gene status.
F1  feature attribution: Integrated Gradients (torch, no extra deps) of the fusion
    classifier logit w.r.t. each view's mu input to the fusion head.
F2  lesion attention: per-lesion attention weights from the radiology aggregator
    (concentration stats; no lesion-level ground truth exists — see report).

Uses the canonical rank-1 checkpoints of `gvae_bestparam_ranked_r32_20261003_124706`
(each patient appears in exactly one fold's val split, so the latents are OOF).

Run:
    .venv/bin/python research/2026-10-06-f-interpretability/scripts/run_interpretability.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.gvae_model import GVAE, get_separate_view_mus  # noqa: E402

CKPT = ROOT / "outputs/gvae/checkpoints/gvae_bestparam_ranked_r32_20261003_124706"
GENES = ["EGFR", "ERBB2", "BRAF", "MET", "STK11", "ARID1A"]  # x_clinical[:, 5:11]
OUT = ROOT / "research/2026-10-06-f-interpretability/output"


def load_data():
    return torch.load(ROOT / "data_ln_pc_ihc_g_r32.pt", weights_only=False)


def fold_checkpoints():
    # Restrict to the canonical run-id-prefixed files (the dir also holds identical
    # `seed_42_*` copies; avoid double counting).
    cks = sorted(CKPT.glob("gvae_bestparam_ranked_r32_20261003_124706_fold_*_rank_1_*.pt"))
    assert len(cks) == 5, f"expected 5 rank-1 fold checkpoints, found {len(cks)}"
    return cks


def integrated_gradients(fusion_fn, x, baseline, steps=32):
    """IG of the per-sample logit w.r.t. input `x` [B,V,d]; returns attr [B,V,d]."""
    total = torch.zeros_like(x)
    for a in torch.linspace(0.0, 1.0, steps):
        xi = (baseline + a * (x - baseline)).clone().requires_grad_(True)
        logits, _ = fusion_fn(xi)
        logits.sum().backward()
        total = total + xi.grad.detach()
    avg = total / steps
    return (x - baseline) * avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=32, help="IG interpolation steps")
    ap.add_argument("--seed", type=int, default=4200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    labels_all = data["patient"]["binary_label"].numpy().astype(int)
    genes_all = data["patient"].x_clinical.numpy()[:, 5:11]
    n = data["patient"].num_nodes

    concat_mu = None          # [n, V*d], filled lazily on the first fold
    ig_view = None            # [num_views] accumulator of mean |attr|
    ig_dim = None
    attn_rows = []
    covered = np.zeros(n, dtype=bool)

    for ck_path in fold_checkpoints():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = GVAE(**ck["model_config"]).eval()
        model.load_state_dict(ck["model_state_dict"])
        view_names = list(model.views)
        v = len(view_names)
        d = model.d_embed
        val_idx = torch.as_tensor(ck["val_indices"], dtype=torch.long)

        # --- per-view mu (missing views filled with the model's missing embedding) ---
        mus = get_separate_view_mus(model, data, val_idx)
        stacked = torch.stack([mus[vn] for vn in view_names], dim=1)  # [B, V, d]
        flat = stacked.reshape(stacked.shape[0], -1).numpy()
        if concat_mu is None:
            concat_mu = np.full((n, flat.shape[1]), np.nan, dtype=np.float64)
        concat_mu[val_idx.numpy()] = flat
        covered[val_idx.numpy()] = True

        # --- F1: Integrated Gradients on the fusion classifier ---
        attr = integrated_gradients(model.fusion_and_classifier_head, stacked,
                                    torch.zeros_like(stacked), steps=args.steps)
        per_view = attr.abs().mean(dim=(0, 2)).numpy()          # [V]
        per_dim = attr.abs().mean(dim=(0, 1)).numpy()           # [d]
        ig_view = per_view if ig_view is None else ig_view + per_view
        ig_dim = per_dim if ig_dim is None else ig_dim + per_dim

        # --- F2: lesion attention on val patients with radiology ---
        rad_mask = data["patient"]["radiology_mask"].bool()
        val_rad = val_idx[rad_mask[val_idx]]
        if val_rad.numel() > 0:
            local = {int(g): i for i, g in enumerate(val_rad)}
            edges = data["patient", "has_lesion", "lesion"].edge_index
            keep = torch.tensor([int(s) in local for s in edges[0]])
            src = torch.tensor([local[int(s)] for s in edges[0][keep]], dtype=torch.long)
            lesion_global = edges[1][keep]
            uniq, inv = torch.unique(lesion_global, return_inverse=True)
            feats = data["lesion"].x[uniq]
            agg = model.radiology_lesion_aggregator
            alpha = agg.attention_weights(feats, torch.stack([src, inv]), val_rad.numel()).numpy()
            for pid_local in range(val_rad.numel()):
                w = alpha[src.numpy() == pid_local]
                if w.size == 0:
                    continue
                w = np.sort(w)[::-1]
                ent = float(-(w * np.log(w + 1e-12)).sum())
                attn_rows.append({"global_idx": int(val_rad[pid_local]), "n_lesions": int(w.size),
                                  "max_weight": float(w[0]), "entropy": ent,
                                  "gini": float(1 - (2 * np.cumsum(w[::-1]) - w[::-1]).sum() / w.size / w.sum())})

    assert covered.all(), f"{int((~covered).sum())} patients uncovered by any fold"
    assert not np.isnan(concat_mu).any()

    # ---------------- F3: projections ----------------
    pca2 = PCA(n_components=2, random_state=args.seed).fit_transform(concat_mu)
    perplexity = min(30, max(2, (n - 1) // 3))
    tsne2 = TSNE(n_components=2, perplexity=perplexity, init="pca",
                 learning_rate="auto", random_state=args.seed).fit_transform(concat_mu)
    assert np.isfinite(pca2).all() and np.isfinite(tsne2).all(), "non-finite projection"
    np.savez_compressed(OUT / "latent_projections.npz", pca=pca2, tsne=tsne2,
                        labels=labels_all, genes=genes_all,
                        gene_names=np.array(GENES))
    _plot(pca2, labels_all, "PCA", "response", OUT / "f3_pca_response.png")
    _plot(tsne2, labels_all, "t-SNE", "response", OUT / "f3_tsne_response.png")
    for j, gname in enumerate(GENES):
        if genes_all[:, j].sum() >= 5:  # skip near-constant genes
            _plot(tsne2, genes_all[:, j].astype(int), "t-SNE", f"{gname}_driver",
                  OUT / f"f3_tsne_gene_{gname}.png")

    # ---------------- F1/F2 aggregates ----------------
    ig_view = ig_view / len(fold_checkpoints())
    ig_dim = ig_dim / len(fold_checkpoints())

    # F3 quantitative: 5-fold CV linear probe on the OOF latents (response + genes).
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def _probe_auc(X, y):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            return None
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=5000,
                                                solver="liblinear"))
        p = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=args.seed),
                              method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))

    f3_probe = {"response": _probe_auc(concat_mu, labels_all)}
    for j, g in enumerate(GENES):
        f3_probe[f"gene_{g}"] = _probe_auc(concat_mu, genes_all[:, j])

    result = {
        "n_patients": int(n),
        "views": view_names,
        "f1_ig_mean_abs_per_view": {vn: float(ig_view[i]) for i, vn in enumerate(view_names)},
        "f3_label_counts": {int(k): int(v) for k, v in zip(*np.unique(labels_all, return_counts=True))},
        "gene_prevalence": {g: int(genes_all[:, j].sum()) for j, g in enumerate(GENES)},
        "f3_latent_probe_auc": f3_probe,
        "f2_attention": {
            "n_patients_scored": len(attn_rows),
            "median_lesions": float(np.median([r["n_lesions"] for r in attn_rows])),
            "mean_max_weight": float(np.mean([r["max_weight"] for r in attn_rows])),
            "mean_entropy_nats": float(np.mean([r["entropy"] for r in attn_rows])),
            "uniform_entropy_ref": float(np.mean([np.log(r["n_lesions"]) for r in attn_rows])),
        },
    }
    json.dump(result, open(OUT / "interpretability_summary.json", "w"), indent=2)
    print(json.dumps(result, indent=2))
    print("F_INTERPRET_DONE")


def _plot(coords, color, title_prefix, color_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=color, cmap="coolwarm", s=18, alpha=0.8,
                    vmin=min(color), vmax=max(color))
    ax.set_title(f"{title_prefix} — colored by {color_name}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
