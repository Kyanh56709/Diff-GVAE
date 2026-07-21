import numpy as np
from utils.classification_eval import build_binary_classification_diagnostics


def test_diagnostics_include_threshold_caveat():
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.2, 0.3, 0.7, 0.8, 0.4, 0.6])
    diag = build_binary_classification_diagnostics(labels, scores)
    assert "threshold_metric_caveat" in diag
    assert "same" in diag["threshold_metric_caveat"].lower()  # "selected on the same split"
