"""B2 — nested-CV hyperparameter selection for the GVAE response classifier.

Why this exists
---------------
`kfold_evaluate_gvae_classifier` already keeps the outer test fold report-only and
drives early stopping + checkpoint + threshold selection from an inner-val split.
The one selection step still fixed *outside* the CV is the **hyperparameters**:
the canonical config (`research/2026-10-03-radiology-artifact-ablation/scripts/run_oof.py`,
copied from `pipeline_phase4a_oof.sh`) and `best_params.json` were chosen on the
coarser 34-dim graph, i.e. by looking at the same 247 patients.

This script closes that gap with a proper nested CV: inside every outer fold it
tries a small, pre-registered candidate set, picks the candidate with the best
**inner-val** AUC (never the outer test), reuses that fold's early-stopped model,
and only then scores the outer test. Pooled OOF metrics + bootstrap CIs are then
directly comparable to the canonical pooled-OOF run.

It is a standalone research harness on purpose: `training/` is read-only here
(only public/underscored helpers are imported), so the canonical artifacts and
the core pipeline are untouched.

Usage
-----
Smoke (fast, tiny epochs — proves the whole path runs):
    .venv/bin/python research/2026-10-06-nested-cv-hparam/scripts/nested_cv_gvae.py \
        --data data_ln_pc_ihc_g_r32.pt --seed 42 --smoke

Pilot (one seed, canonical epochs, 3 candidates):
    .venv/bin/python research/2026-10-06-nested-cv-hparam/scripts/nested_cv_gvae.py \
        --data data_ln_pc_ihc_g_r32.pt --seed 42

Full (seeds 42-46):
    .venv/bin/python research/2026-10-06-nested-cv-hparam/scripts/nested_cv_gvae.py \
        --data data_ln_pc_ihc_g_r32.pt --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.gvae_model import GVAE, get_separate_view_mus  # noqa: E402
from training.train_gvae import (  # noqa: E402
    _bootstrap_ci,
    _gvae_loss_and_logits,
    _resolve_clinical_indices,
    _threshold_max_f1,
    pretrain_radiology_aggregator,
)
from utils.loss_utils import (  # noqa: E402
    calculate_contrastive_loss,
    calculate_contrastive_loss_vectorized,
)
from utils.training_utils import linear_anneal  # noqa: E402

OUT_ROOT = ROOT / "research/2026-10-06-nested-cv-hparam/output"

# Pre-registered candidate set. Keep it small and interpretable: one canonical
# config plus single-axis perturbations of the two knobs the June sweep varied.
# `None` = keep the canonical value.
PARAM_GRID: "list[dict]" = [
    {"name": "canonical", "d_embed": None, "hidden_channels_vae": None},
    {"name": "d_embed16", "d_embed": 16, "hidden_channels_vae": None},
    {"name": "hidden64", "d_embed": None, "hidden_channels_vae": 64},
]


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


def apply_candidate(model_config, train_config, cand):
    """Deep-copy the canonical configs and apply one candidate's overrides."""
    mc, tc = copy.deepcopy(model_config), copy.deepcopy(train_config)
    if cand.get("d_embed") is not None:
        mc["d_embed"] = int(cand["d_embed"])
        mc["projection_head_config"]["output_dim"] = int(cand["d_embed"])
    if cand.get("hidden_channels_vae") is not None:
        h = int(cand["hidden_channels_vae"])
        for vcfg in mc["view_configs"].values():
            vcfg["hidden_channels_vae"] = h
    return mc, tc


def build_ctx(tc, clinical_cont_idx, clinical_bin_idx, device):
    lw = tc["loss_weights"]
    use_vec = tc.get("vectorized_contrastive", False)
    return {
        "criterion_mse": nn.MSELoss(),
        "clinical_cont_idx": clinical_cont_idx, "clinical_bin_idx": clinical_bin_idx,
        "w_rec_attr_config": lw["rec_attr"], "w_rec_struct_config": lw["rec_struct"],
        "base_w_class": lw["class"],
        "contrastive_fn": calculate_contrastive_loss_vectorized if use_vec else calculate_contrastive_loss,
        "cross_cl_temp": tc["cross_cl_temp"],
    }


def train_one_candidate(model_config_c, train_config_c, ctx, fold_data, inner_tr, inner_val,
                        device, base_w_kl, base_w_cross_cl, kl_start_w, cl_start_w,
                        kl_end_e, cl_end_e):
    """Train a single (fold, candidate) model; select checkpoint on inner-val AUC.

    Returns (best_state, best_epoch, inner_val_auc, inner_val_threshold).
    Mirrors the selection logic of `kfold_evaluate_gvae_classifier` exactly.
    """
    select_after_epoch = train_config_c.get("select_after_epoch", 20)
    epochs = train_config_c["epochs"]
    patience = train_config_c.get("patience_early_stopping", 30)
    batch_size = train_config_c.get("batch_size", None)

    # Train-only pos weights for the binary-reconstruction + main BCE terms.
    tr_clin_bin = fold_data["patient"].x_clinical[inner_tr][:, ctx["clinical_bin_idx"]]
    tr_n_pos = tr_clin_bin.sum(dim=0)
    tr_n_neg = tr_clin_bin.shape[0] - tr_n_pos
    ctx["clinical_bin_pos_weight"] = (tr_n_neg / (tr_n_pos + 1e-6)).to(device)
    tr_labels = fold_data["patient"].binary_label[inner_tr].cpu().numpy()
    pw_value = np.sum(tr_labels == 0) / (np.sum(tr_labels == 1) + 1e-6)
    ctx["criterion_main_bce"] = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw_value], device=device))

    radiology_state = None
    if "radiology" in model_config_c["view_configs"]:
        radiology_state = pretrain_radiology_aggregator(
            fold_data, inner_tr, model_config_c["radiology_aggregator_config"], device,
            epochs=train_config_c.get("pretrain_epochs", 400),
            use_pos_weight=train_config_c.get("pretrain_use_pos_weight", False),
            pretrain_val_split=train_config_c.get("pretrain_val_split", 0.0),
            patience=train_config_c.get("pretrain_patience", 30),
            seed=train_config_c.get("pretrain_seed", 42))

    model = GVAE(**model_config_c).to(device)
    if radiology_state is not None:
        model.radiology_lesion_aggregator.load_state_dict(radiology_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config_c["lr"],
                                  weight_decay=train_config_c["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=15)

    def make_batches(indices):
        if batch_size and 0 < batch_size < len(indices):
            return torch.split(indices, batch_size)
        return [indices]

    best_sel, best_state, best_epoch = -1.0, None, 0
    best_loss_state, best_loss = None, float("inf")
    no_improve = 0
    for epoch in range(1, epochs + 1):
        w_kl = linear_anneal(epoch, 0, kl_end_e, kl_start_w, base_w_kl)
        w_cl = linear_anneal(epoch, 0, cl_end_e, cl_start_w, base_w_cross_cl)
        model.train()
        for b in make_batches(inner_tr):
            loss, _, _ = _gvae_loss_and_logits(model, fold_data, b, ctx, w_cl, w_kl)
            if not torch.isfinite(loss):
                optimizer.zero_grad(); continue
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config_c.get("grad_clip_norm", 1.0))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            v_loss, v_logits, _ = _gvae_loss_and_logits(model, fold_data, inner_val, ctx, base_w_cross_cl, base_w_kl)
        v_loss = v_loss.item()
        v_probs = torch.sigmoid(v_logits.squeeze()).cpu().numpy()
        v_true = fold_data["patient"].binary_label[inner_val].cpu().numpy()
        v_auc = roc_auc_score(v_true, v_probs) if len(np.unique(v_true)) > 1 else -1.0
        scheduler.step(v_loss)
        if v_loss < best_loss:
            best_loss, best_loss_state = v_loss, copy.deepcopy(model.state_dict())
        if epoch >= select_after_epoch:
            if v_auc > best_sel:
                best_sel, best_state, best_epoch, no_improve = v_auc, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
    if best_state is None:
        best_state, best_epoch = best_loss_state, -1
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, iv_logits, iv_labels = _gvae_loss_and_logits(model, fold_data, inner_val, ctx, base_w_cross_cl, base_w_kl)
    thr = _threshold_max_f1(iv_labels.cpu().numpy(), torch.sigmoid(iv_logits.squeeze()).cpu().numpy())
    return model, best_epoch, float(best_sel), thr


def run_one_seed(data, seed, grid, model_config, train_config, n_bootstrap, eval_seed):
    device = train_config["device"]
    tc0 = copy.deepcopy(train_config)
    tc0["random_seed"] = seed
    assert tc0.get("selection_metric", "inner_val_auc") == "inner_val_auc", \
        "this harness only implements selection_metric='inner_val_auc'"
    full_cpu = data.clone().cpu()

    clinical_dim = full_cpu["patient"].x_clinical.shape[1]
    clinical_cont_idx, clinical_bin_idx = _resolve_clinical_indices(tc0, clinical_dim, device)
    lw = tc0["loss_weights"]
    base_w_kl = lw["kl"]
    base_w_cross_cl = lw.get("cross_cl", 0.0)
    kl_params = tc0["annealing"]["kl"]
    cl_params = tc0["annealing"]["cross_cl"]
    kl_start_w = kl_params.get("start_weight", base_w_kl)
    cl_start_w = cl_params.get("start_weight", base_w_cross_cl)
    kl_end_e = kl_params["end_epoch"]
    cl_end_e = cl_params["end_epoch"]

    N = full_cpu["patient"].num_nodes
    y_all = full_cpu["patient"]["binary_label"].cpu().numpy().astype(int)
    oof_head = np.full(N, np.nan)
    oof_probe = np.full(N, np.nan)
    all_idx = np.arange(N)
    kf = StratifiedKFold(n_splits=tc0["n_splits"], shuffle=True, random_state=seed)
    inner_val_split = tc0.get("inner_val_split", 0.2)

    per_fold, selections = [], []
    for fold, (train_idx_np, test_idx_np) in enumerate(kf.split(all_idx, y_all)):
        inner_tr_np, inner_val_np = train_test_split(
            train_idx_np, test_size=inner_val_split, stratify=y_all[train_idx_np],
            random_state=eval_seed)
        fold_data = full_cpu.to(device)
        inner_tr = torch.tensor(inner_tr_np, device=device)
        inner_val = torch.tensor(inner_val_np, device=device)
        test_idx = torch.tensor(test_idx_np, device=device)

        fold_candidates = []
        for cand in grid:
            mc, tc = apply_candidate(model_config, tc0, cand)
            ctx = build_ctx(tc, clinical_cont_idx, clinical_bin_idx, device)
            t0 = time.time()
            model, best_epoch, inner_auc, thr = train_one_candidate(
                mc, tc, ctx, fold_data, inner_tr, inner_val, device,
                base_w_kl, base_w_cross_cl, kl_start_w, cl_start_w, kl_end_e, cl_end_e)
            fold_candidates.append({
                "candidate": cand["name"], "inner_val_auc": inner_auc,
                "selected_epoch": best_epoch, "threshold": thr,
                "model": model, "train_s": round(time.time() - t0, 1),
            })
            print(f"  [fold {fold+1}] cand={cand['name']:10s} inner_auc={inner_auc:.4f} "
                  f"ep={best_epoch} ({fold_candidates[-1]['train_s']}s)", flush=True)

        chosen = max(fold_candidates, key=lambda c: c["inner_val_auc"])
        selections.append({"fold": fold + 1, "chosen": chosen["candidate"],
                           "inner_val_auc": chosen["inner_val_auc"],
                           "all": {c["candidate"]: round(c["inner_val_auc"], 4) for c in fold_candidates}})
        model = chosen["model"]
        model.eval()
        with torch.no_grad():
            t_logits, _, _, _ = model(fold_data, test_idx)
        head_probs = torch.sigmoid(t_logits.squeeze()).cpu().numpy()
        oof_head[test_idx_np] = head_probs

        def concat_mu(idx_tensor):
            mus = get_separate_view_mus(model, fold_data, idx_tensor)
            z = torch.cat([mus[v] for v in model.views], dim=1).cpu().numpy()
            return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        Z_tr, Z_te = concat_mu(inner_tr), concat_mu(test_idx)
        y_tr = full_cpu["patient"].binary_label[inner_tr].cpu().numpy()
        scaler = StandardScaler().fit(Z_tr)
        Z_tr_s = np.clip(scaler.transform(Z_tr), -10.0, 10.0)
        Z_te_s = np.clip(scaler.transform(Z_te), -10.0, 10.0)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            probe = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")
            probe.fit(Z_tr_s, y_tr)
            probe_probs = probe.predict_proba(Z_te_s)[:, 1]
        oof_probe[test_idx_np] = np.nan_to_num(probe_probs, nan=0.5, posinf=1.0, neginf=0.0)

        y_te = y_all[test_idx_np]
        per_fold.append({
            "fold": fold + 1, "chosen": chosen["candidate"],
            "head_auc": roc_auc_score(y_te, head_probs) if len(np.unique(y_te)) > 1 else float("nan"),
            "probe_auc": roc_auc_score(y_te, probe_probs) if len(np.unique(y_te)) > 1 else float("nan"),
            "head_f1": f1_score(y_te, (head_probs > chosen["threshold"]).astype(int), zero_division=0),
        })
        print(f"  [fold {fold+1}] CHOSEN {chosen['candidate']} | test head_auc={per_fold[-1]['head_auc']:.4f} "
              f"probe_auc={per_fold[-1]['probe_auc']:.4f}", flush=True)

        del model, fold_data
        gc.collect()

    mask = ~np.isnan(oof_head)
    yt, head, probe = y_all[mask], oof_head[mask], oof_probe[mask]
    h_auc = _bootstrap_ci(yt, head, roc_auc_score, n_bootstrap, eval_seed)
    h_ap = _bootstrap_ci(yt, head, average_precision_score, n_bootstrap, eval_seed)
    p_auc = _bootstrap_ci(yt, probe, roc_auc_score, n_bootstrap, eval_seed)
    summary = {
        "seed": seed, "n_pooled": int(mask.sum()), "prevalence": float(yt.mean()),
        "head_auc": h_auc[0], "head_auc_lo": h_auc[1], "head_auc_hi": h_auc[2],
        "head_auprc": h_ap[0], "head_auprc_lo": h_ap[1], "head_auprc_hi": h_ap[2],
        "probe_auc": p_auc[0], "probe_auc_lo": p_auc[1], "probe_auc_hi": p_auc[2],
        "chosen_by_fold": [s["chosen"] for s in selections],
    }
    oof_arrays = {"y_true": y_all, "head_probs": oof_head, "probe_probs": oof_probe}
    return summary, per_fold, selections, oof_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None, help="comma list, overrides --seed")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny epochs/pretrain to validate the path quickly")
    ap.add_argument("--candidates", type=str, default=None,
                    help="comma list of PARAM_GRID names to include (default: all)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    grid = PARAM_GRID
    if args.candidates:
        keep = {s.strip() for s in args.candidates.split(",")}
        grid = [c for c in PARAM_GRID if c["name"] in keep]
        missing = keep - {c["name"] for c in grid}
        assert not missing, f"unknown candidate name(s): {missing}"

    torch.manual_seed(seeds[0]); np.random.seed(seeds[0])
    data = torch.load(ROOT / args.data, weights_only=False)
    assert data["lesion"].x.shape[1] == 32, f"expected r32 graph, got {data['lesion'].x.shape[1]}"
    model_config, train_config = base_configs(data)
    if args.smoke:
        train_config["epochs"] = 5
        train_config["pretrain_epochs"] = 5
        train_config["select_after_epoch"] = 1
    tag = args.tag or ("nestedcv_smoke" if args.smoke else "nestedcv")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        # Reseed per seed so each seed is an independent run, matching run_oof.py
        # (seeding once before the loop chained RNG state across seeds).
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n===== nested CV seed {seed} | candidates={[c['name'] for c in grid]} "
              f"| epochs={train_config['epochs']} pretrain={train_config['pretrain_epochs']} =====",
              flush=True)
        summary, per_fold, selections, oof = run_one_seed(
            data, seed, grid, model_config, train_config,
            train_config.get("n_bootstrap", 2000), train_config.get("eval_seed", 4200))
        rows.append(summary)
        out = OUT_ROOT / f"{tag}_seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "oof_arrays.npz", **oof)
        json.dump({"summary": summary, "per_fold": per_fold, "selections": selections},
                  open(out / "nested_cv_result.json", "w"), indent=2, default=float)
        print(f"NESTED_CV seed {seed} | HEAD AUC {summary['head_auc']:.4f} "
              f"[{summary['head_auc_lo']:.4f}-{summary['head_auc_hi']:.4f}] | "
              f"AUPRC {summary['head_auprc']:.4f} | PROBE AUC {summary['probe_auc']:.4f}", flush=True)

    json.dump({"data": args.data, "grid": grid, "rows": rows},
              open(OUT_ROOT / f"{tag}_summary.json", "w"), indent=2, default=float)
    print("\nNESTED_CV_DONE", json.dumps({"tag": tag, "seeds": seeds,
          "head_auc_mean": float(np.mean([r["head_auc"] for r in rows])),
          "head_auc_sd": float(np.std([r["head_auc"] for r in rows]))}, default=float))


if __name__ == "__main__":
    main()
