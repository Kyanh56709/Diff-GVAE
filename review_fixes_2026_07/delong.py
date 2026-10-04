"""DeLong test for paired ROC-AUC comparisons on the same patients.

Fast O(n log n) DeLong (Sun & Xu 2014, IEEE SPL 21:1389) with midranks for
ties; equal to the O(n_pos * n_neg) structural-component estimator of
DeLong et al. (1988). Feed it pooled out-of-fold scores of two models that
were evaluated on the same patients in the same order, e.g.
kfold_evaluate_gvae_classifier's `oof_arrays['head_probs']` against a
baseline's OOF scores built on identical splits.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties replaced by their average rank."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _fast_delong(y_true, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """AUCs and their DeLong covariance for k score vectors (shape (k, n))."""
    y_true = np.asarray(y_true).astype(int)
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("DeLong needs both classes (labels 0 and 1) present")
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    pos, neg = scores[:, y_true == 1], scores[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]

    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n          # structural components of positives
    v10 = 1.0 - (tz[:, m:] - ty) / m    # structural components of negatives
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_auc_variance(y_true, y_score) -> Tuple[float, float]:
    """(AUC, DeLong variance of the AUC) for one score vector."""
    aucs, cov = _fast_delong(y_true, np.asarray(y_score)[None, :])
    return float(aucs[0]), float(cov[0, 0])


def delong_paired_test(y_true, score_a, score_b, alpha: float = 0.05) -> Dict[str, float]:
    """Two-sided DeLong test of AUC(a) == AUC(b) on the same patients.

    Returns auc_a, auc_b, delta_auc (a - b), var_diff, z, p_value and a
    normal-approximation (1 - alpha) CI for delta_auc.
    """
    aucs, cov = _fast_delong(y_true, np.vstack([score_a, score_b]))
    delta = float(aucs[0] - aucs[1])
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var_diff <= 0.0:  # identical (or perfectly rank-equivalent) scores
        z, p = 0.0, 1.0
        half = 0.0
    else:
        se = np.sqrt(var_diff)
        z = delta / se
        p = float(2.0 * stats.norm.sf(abs(z)))
        half = float(stats.norm.ppf(1.0 - alpha / 2.0) * se)
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc": delta,
        "var_diff": var_diff,
        "z": float(z),
        "p_value": p,
        "ci_low": delta - half,
        "ci_high": delta + half,
    }
