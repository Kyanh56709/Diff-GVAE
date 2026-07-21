import numpy as np
from sklearn.metrics import roc_auc_score
from review_fixes_2026_07.bootstrap_ci import bootstrap_ci, bootstrap_metric_cis


def test_perfect_separation_point_is_one_and_ci_brackets_it():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    point, lo, hi = bootstrap_ci(y, s, roc_auc_score, n_boot=200, seed=0)
    assert point == 1.0
    assert lo <= point <= hi


def test_metric_panel_keys_and_bounds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    s = rng.random(60)
    res = bootstrap_metric_cis(y, s, n_boot=200, seed=1)
    assert set(res) == {"roc_auc", "pr_auc", "balanced_accuracy", "f1"}
    for name, d in res.items():
        assert d["ci_low"] <= d["point"] <= d["ci_high"]
