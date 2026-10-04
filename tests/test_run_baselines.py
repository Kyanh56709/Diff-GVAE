import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_baselines", ROOT / "research/2026-10-04-baselines-delong/scripts/run_baselines.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _toy():
    d = HeteroData()
    d["patient"].num_nodes = 3
    d["patient"].x_clinical = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    d["patient"].x_pathology = torch.tensor([[5.0], [9.0], [7.0]])
    d["patient"].pathology_mask = torch.tensor([True, False, True])
    d["patient"].binary_label = torch.tensor([1, 0, 1])
    # patient 0 has lesions 0,1; patient 2 has lesion 2; patient 1 has none
    d["lesion"].x = torch.tensor([[1.0, 4.0], [3.0, 2.0], [6.0, 6.0]])
    d["patient", "has_lesion", "lesion"].edge_index = torch.tensor([[0, 0, 2], [0, 1, 2]])
    return d


def test_patient_feature_blocks():
    blocks, y = mod.build_patient_features(_toy())
    assert y.tolist() == [1, 0, 1]
    # absent pathology is zeroed even if the tensor held a value, flag = 0
    assert blocks["pathology"].tolist() == [[5.0, 1.0], [0.0, 0.0], [7.0, 1.0]]
    # radiology = [mean(2), max(2), has_lesion]
    np.testing.assert_allclose(blocks["radiology"][0], [2.0, 3.0, 3.0, 4.0, 1.0])
    np.testing.assert_allclose(blocks["radiology"][1], [0.0] * 5)
    np.testing.assert_allclose(blocks["radiology"][2], [6.0, 6.0, 6.0, 6.0, 1.0])
    assert blocks["all"].shape == (3, 2 + 2 + 5)


def test_holm_matches_manual():
    adj = mod.holm([0.01, 0.04, 0.03, 0.5])
    # sorted p: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08->0.09 (monotone), 0.5*1=0.5
    np.testing.assert_allclose(adj, [0.04, 0.09, 0.09, 0.5])


def test_oof_scores_cover_every_patient():
    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 60)
    X = rng.normal(size=(90, 4)) + y[:, None]
    specs = mod.model_specs(seed=42)
    _, factory, grid = specs["logreg_concat"]
    scores, chosen = mod.oof_scores(X, y, factory, grid, seed=42)
    assert scores.shape == (90,) and not np.isnan(scores).any()
    assert len(chosen) == mod.N_SPLITS
    assert all(c["C"] in grid["C"] for c in chosen)


def test_inner_train_refit_differs_from_outer_train():
    rng = np.random.default_rng(1)
    y = np.array([0] * 30 + [1] * 60)
    X = rng.normal(size=(90, 4)) + y[:, None]
    _, factory, grid = mod.model_specs(seed=42)["logreg_concat"]
    outer, _ = mod.oof_scores(X, y, factory, grid, seed=42, refit="outer_train")
    inner, _ = mod.oof_scores(X, y, factory, grid, seed=42, refit="inner_train")
    assert not np.isnan(inner).any()
    assert not np.allclose(outer, inner)
