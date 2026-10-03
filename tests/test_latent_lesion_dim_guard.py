# tests/test_latent_lesion_dim_guard.py
"""A GVAE checkpoint must not be applied to a graph with a different lesion dim
(e.g. a 34-slot checkpoint on the canonical 32-slot r32 graph)."""
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from utils.latent_extraction import (
    _check_lesion_dim,
    extract_latents_for_ddpm,
    extract_recommended_latents_for_ddpm,
)


def _graph(lesion_dim):
    data = HeteroData()
    data["patient"].num_nodes = 2
    data["lesion"].x = torch.zeros(3, lesion_dim)
    return data


def _ckpt(lesion_dim):
    return {
        "model_state_dict": {},
        "model_config": {"radiology_aggregator_config": {"lesion_feature_dim": lesion_dim}},
    }


def test_matching_dims_pass():
    _check_lesion_dim(_ckpt(32), _graph(32), Path("x.pt"))


def test_mismatch_raises_with_both_dims():
    with pytest.raises(ValueError, match=r"lesion_feature_dim=34.*lesion\.x dim 32"):
        _check_lesion_dim(_ckpt(34), _graph(32), Path("x.pt"))


def test_no_aggregator_config_is_skipped():
    _check_lesion_dim({"model_state_dict": {}, "model_config": {}}, _graph(32), Path("x.pt"))


@pytest.mark.parametrize("fn", [extract_latents_for_ddpm, extract_recommended_latents_for_ddpm])
def test_extractors_refuse_mismatch_before_building_model(tmp_path, fn):
    path = tmp_path / "ckpt.pt"
    torch.save(_ckpt(34), path)
    with pytest.raises(ValueError, match="lesion_feature_dim"):
        fn(checkpoint_path=path, full_data=_graph(32), output_dir=tmp_path / "out",
           split_indices={"train": [0], "val": [1]}, device="cpu")
