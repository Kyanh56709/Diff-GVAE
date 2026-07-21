import torch
from review_fixes_2026_07.data_validation import validate_hetero_graph


def test_valid_graph_passes(synthetic_data):
    # synthetic_data has clinical dim 22 and 3-vs-3 labels; skip strict count check
    r = validate_hetero_graph(synthetic_data, expected_class_counts=None)
    assert r.ok, r.problems
    assert r.info["clinical_dim"] == 22


def test_wrong_clinical_dim_fails(synthetic_data):
    d = synthetic_data.clone()
    n = d["patient"].x_clinical.shape[0]
    d["patient"].x_clinical = torch.zeros(n, 64)   # data_247-style dimensionality
    r = validate_hetero_graph(d, expected_class_counts=None)
    assert not r.ok
    assert any("clinical_dim" in p for p in r.problems)


def test_single_class_labels_fail(synthetic_data):
    d = synthetic_data.clone()
    n = d["patient"].x_clinical.shape[0]
    d["patient"].binary_label = torch.zeros(n, dtype=torch.long)
    d["patient"].y = torch.zeros(n, dtype=torch.long)
    r = validate_hetero_graph(d, expected_class_counts=None)
    assert not r.ok
    assert any("single-class" in p for p in r.problems)
