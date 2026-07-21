import numpy as np
from utils.classification_eval import metrics_at_threshold


def test_score_equal_to_threshold_counts_as_positive():
    # patient 0 is a true positive whose score sits exactly on the threshold
    labels = np.array([1, 0])
    scores = np.array([0.5, 0.2])
    metrics, cm = metrics_at_threshold(labels, scores, 0.5)
    assert cm["tp"] == 1        # 0.5 >= 0.5 -> positive
    assert cm["fn"] == 0
    assert metrics["recall"] == 1.0
