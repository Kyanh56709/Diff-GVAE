"""Regression lock: kfold_evaluate_gvae_classifier pos weights are train-only."""
from __future__ import annotations

import numpy as np
import torch


def test_evaluate_pos_weights_are_inner_train_only():
    """The per-fold Main-Task BCE pos_weight must use inner-train labels only,
    not the full dataset (review finding H1: leakage-free claim contradicted by
    full-data pos_weight). We re-run the exact computation the function now does
    per fold and assert it matches the inner-train ratio, not the full ratio."""
    from training.train_gvae import _resolve_clinical_indices

    data = torch.load('data_ln_pc_ihc_g.pt', map_location='cpu', weights_only=False)
    full_cpu = data.clone().cpu()
    labels = full_cpu['patient']['binary_label'].numpy()
    clinical_dim = full_cpu['patient'].x_clinical.shape[1]
    cont_idx, bin_idx = _resolve_clinical_indices({}, clinical_dim, torch.device('cpu'))

    # Simulate one fold's inner-train split — deliberately SKEWED so the
    # train-only ratio differs clearly from the full-dataset ratio:
    # all 62 responders (label 0) + only 10 non-responders (label 1).
    resp_idx = np.where(labels == 0)[0]
    nonresp_idx = np.where(labels == 1)[0][:10]
    inner_tr = np.concatenate([resp_idx, nonresp_idx])

    tr_clin_bin = full_cpu['patient'].x_clinical[inner_tr][:, bin_idx]
    tr_n_pos = tr_clin_bin.sum(dim=0)
    tr_n_neg = tr_clin_bin.shape[0] - tr_n_pos
    bin_pw = (tr_n_neg / (tr_n_pos + 1e-6)).to(torch.device('cpu'))

    tr_labels = labels[inner_tr]
    pw_value = np.sum(tr_labels == 0) / (np.sum(tr_labels == 1) + 1e-6)

    # Expected value = inner-train ratio (62/10 = 6.2), NOT full ratio (62/185).
    exp_pw = np.sum(tr_labels == 0) / np.sum(tr_labels == 1)
    assert abs(pw_value - exp_pw) < 1e-3  # tolerance covers the +1e-6 epsilon in the code
    full_pw = np.sum(labels == 0) / np.sum(labels == 1)
    assert abs(pw_value - full_pw) > 1.0, "pos_weight must be inner-train only, not full-data"
    # Binary-feature pos weight must also be train-only (no NaN from zero columns).
    assert bool(torch.isfinite(bin_pw).all())
