import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from review_fixes_2026_07.delong import delong_auc_variance, delong_paired_test


def _naive_delong(y, scores):
    """O(n_pos * n_neg) DeLong straight from the definition (DeLong et al. 1988)."""
    y = np.asarray(y)
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    k, m, n = scores.shape[0], pos.shape[1], neg.shape[1]
    psi = (pos[:, :, None] > neg[:, None, :]) + 0.5 * (pos[:, :, None] == neg[:, None, :])
    aucs = psi.mean(axis=(1, 2))
    v10 = psi.mean(axis=2)  # (k, m)
    v01 = psi.mean(axis=1)  # (k, n)
    cov = np.cov(v10) / m + np.cov(v01) / n
    return aucs, np.atleast_2d(cov)


def _data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a = y + rng.normal(0, 1.0, n)
    b = y + rng.normal(0, 1.5, n)
    return y, a, b


def test_auc_matches_sklearn_and_variance_matches_naive():
    y, a, b = _data()
    auc, var = delong_auc_variance(y, a)
    assert auc == pytest.approx(roc_auc_score(y, a))
    naive_aucs, naive_cov = _naive_delong(y, np.vstack([a]))
    assert var == pytest.approx(naive_cov[0, 0])


def test_paired_test_matches_naive_reference():
    y, a, b = _data(seed=1)
    res = delong_paired_test(y, a, b)
    aucs, cov = _naive_delong(y, np.vstack([a, b]))
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    assert res["auc_a"] == pytest.approx(aucs[0])
    assert res["auc_b"] == pytest.approx(aucs[1])
    assert res["z"] == pytest.approx(z)
    assert res["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z)))


def test_ties_are_handled_like_naive():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 80)
    a = np.round(y + rng.normal(0, 1, 80), 0)  # heavy ties
    b = np.round(y + rng.normal(0, 2, 80), 0)
    res = delong_paired_test(y, a, b)
    aucs, cov = _naive_delong(y, np.vstack([a, b]))
    assert res["auc_a"] == pytest.approx(aucs[0])
    assert res["var_diff"] == pytest.approx(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])


def test_identical_scores_give_p_one():
    y, a, _ = _data(seed=3)
    res = delong_paired_test(y, a, a.copy())
    assert res["delta_auc"] == 0.0
    assert res["p_value"] == 1.0


def test_ci_of_difference_brackets_delta():
    y, a, b = _data(seed=4)
    res = delong_paired_test(y, a, b)
    assert res["ci_low"] < res["delta_auc"] < res["ci_high"]


def test_rejects_single_class():
    with pytest.raises(ValueError):
        delong_paired_test(np.ones(10), np.arange(10.0), np.arange(10.0))
