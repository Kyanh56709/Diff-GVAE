"""Pooled patient-level bootstrap confidence intervals for small-sample
classification metrics.

Resamples pooled out-of-fold (label, score) pairs with replacement — the
same convention as training.train_gvae._bootstrap_ci — so CIs are comparable
with the maintained pipeline. Feed it kfold_evaluate_gvae_classifier's
`oof_arrays['y_true'], oof_arrays['head_probs']`.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def bootstrap_ci(
    y_true,
    y_score,
    metric_fn: Callable,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Point estimate on the full data + percentile bootstrap CI.

    Resamples patients with replacement; skips resamples that collapse to a
    single class (metric undefined). Returns (point, ci_low, ci_high).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    try:
        point = float(metric_fn(y_true, y_score))
    except Exception:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            stats.append(float(metric_fn(y_true[idx], y_score[idx])))
        except Exception:
            continue
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return point, float(lo), float(hi)


def bootstrap_metric_cis(
    y_true,
    y_score,
    *,
    threshold: float = 0.5,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap CIs for the small-sample metric panel.

    ROC-AUC and PR-AUC use raw scores. Balanced-accuracy and F1 use a FIXED
    threshold (default 0.5) — they are NOT threshold-optimised, to avoid the
    same-split optimism flagged in PROJECT_REVIEW.
    """

    def _bacc(yt, ys):
        return balanced_accuracy_score(yt, (np.asarray(ys) >= threshold).astype(int))

    def _f1(yt, ys):
        return f1_score(yt, (np.asarray(ys) >= threshold).astype(int), zero_division=0)

    panel = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "balanced_accuracy": _bacc,
        "f1": _f1,
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, fn in panel.items():
        point, lo, hi = bootstrap_ci(y_true, y_score, fn, n_boot=n_boot, seed=seed, alpha=alpha)
        out[name] = {
            "point": point,
            "ci_low": lo,
            "ci_high": hi,
            "threshold": threshold if name in ("balanced_accuracy", "f1") else None,
        }
    return out
