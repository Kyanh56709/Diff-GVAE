"""Run kfold_train_gvae across seeds and report the across-seed distribution.

Seed sweeps report how stable a metric is under reseeding; they do NOT
bootstrap (use review_fixes_2026_07.bootstrap_ci for patient-level CIs).
Shrink n_splits/epochs in train_config for a smoke-scale sweep.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


def aggregate_seed_results(
    summaries: Sequence[Dict[str, float]],
    metric_prefix: str = "mean_",
) -> Dict[str, Dict[str, Any]]:
    """For every '<metric_prefix><name>' key, report across-seed
    mean/std/min/max/n_seeds and the raw per-seed values."""
    if not summaries:
        return {}
    keys = [k for k in summaries[0] if k.startswith(metric_prefix)]
    out: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        vals: List[float] = []
        for s in summaries:
            v = s.get(k)
            if v is None:
                continue
            fv = float(v)
            if np.isnan(fv):
                continue
            vals.append(fv)
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": int(arr.size),
            "values": vals,
        }
    return out


def run_seed_sweep(
    full_data,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    train_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run kfold_train_gvae once per seed and aggregate the summaries.

    `train_fn` defaults to the real kfold_train_gvae; override it in tests.
    """
    if train_fn is None:
        from training.train_gvae import kfold_train_gvae as train_fn  # noqa: N806
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        cfg = copy.deepcopy(train_config)
        cfg["random_seed"] = int(seed)
        summary, _df, _roc = train_fn(full_data, model_config, cfg)
        row = dict(summary)
        row["seed"] = int(seed)
        per_seed.append(row)
    return {
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregate": aggregate_seed_results(per_seed),
    }
