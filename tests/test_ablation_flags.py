"""Opt-in ablation flags: graph-agnostic (MLP) encoder + lesion pooling.

Covers C1 (no-GNN) and C5 (aggregator attention vs mean/max) flags. Defaults must
stay 'gat' / 'attention' so existing behaviour is unchanged (backward compat).
"""
import copy

import pytest
import torch
from torch_scatter import scatter_mean

from models.gvae_components import RadiologyLesionAttentionAggregator, ViewEncoder
from models.gvae_model import GVAE


# --- ViewEncoder: encoder_type flag -------------------------------------------------

def test_default_encoder_type_is_gat():
    enc = ViewEncoder(8, 16, 4, heads=2, num_gnn_layers=2, edge_dim=-1)
    assert enc.encoder_type == "gat"
    assert hasattr(enc, "conv1") and not hasattr(enc, "mlp1")


def test_mlp_encoder_shapes_and_ignores_edges():
    enc = ViewEncoder(8, 16, 4, heads=2, num_gnn_layers=2, edge_dim=-1,
                      encoder_type="mlp").eval()
    x = torch.randn(5, 8)
    ei_a = torch.tensor([[0, 1], [1, 2]])
    ei_b = torch.tensor([[4, 3], [0, 2]])  # different graph, same nodes
    mu_a, lv_a = enc(x, ei_a, None)
    mu_b, lv_b = enc(x, ei_b, None)
    assert mu_a.shape == (5, 4) and lv_a.shape == (5, 4)
    # A graph-agnostic encoder must not depend on edge_index.
    assert torch.allclose(mu_a, mu_b) and torch.allclose(lv_a, lv_b)


def test_invalid_encoder_type_raises():
    with pytest.raises(ValueError):
        ViewEncoder(8, 16, 4, encoder_type="transformer")


# --- Aggregator: pooling flag -------------------------------------------------------

def test_default_pooling_is_attention():
    agg = RadiologyLesionAttentionAggregator(15, 32, attention_hidden_dim=32, dropout=0.0)
    assert agg.pooling == "attention"


def test_mean_pooling_matches_manual_and_zeroes_absent():
    torch.manual_seed(0)
    agg = RadiologyLesionAttentionAggregator(15, 32, attention_hidden_dim=32,
                                             dropout=0.0, pooling="mean").eval()
    lesion_x = torch.randn(7, 15)
    # patient 2 has no lesions (3 patients, edges only for 0 and 1)
    edge = torch.tensor([[0, 0, 1], [0, 1, 2]])
    out = agg(lesion_x, edge, 3)
    assert out.shape == (3, 32) and torch.isfinite(out).all()
    # Patient 2 has no lesions: its row is tail(0) (not -inf/NaN and not a lesion).
    zero_row = agg.norm_layer(agg.output_projection(torch.zeros(1, 15)))[0]
    assert torch.allclose(out[2], zero_row, atol=1e-6)
    # Single-lesion patient 1: row equals norm(proj(lesion_norm(lesion2))).
    one = agg.lesion_norm(lesion_x[2:3])
    expected = agg.norm_layer(agg.output_projection(one))
    assert torch.allclose(out[1], expected[0], atol=1e-6)


def test_max_pooling_is_finite_with_absent_lesions():
    torch.manual_seed(0)
    agg = RadiologyLesionAttentionAggregator(15, 32, attention_hidden_dim=32,
                                             dropout=0.0, pooling="max").eval()
    lesion_x = torch.randn(7, 15)
    edge = torch.tensor([[0, 0, 1], [0, 1, 2]])  # patient 2 absent
    out = agg(lesion_x, edge, 3)
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()  # NOT -inf for the no-lesion patient
    zero_row = agg.norm_layer(agg.output_projection(torch.zeros(1, 15)))[0]
    assert torch.allclose(out[2], zero_row, atol=1e-6)


def test_invalid_pooling_raises():
    with pytest.raises(ValueError):
        RadiologyLesionAttentionAggregator(15, 32, pooling="median")


def test_attention_weights_normalize_per_patient():
    torch.manual_seed(0)
    agg = RadiologyLesionAttentionAggregator(15, 32, attention_hidden_dim=32,
                                             dropout=0.0).eval()
    lesion_x = torch.randn(7, 15)
    edge = torch.tensor([[0, 0, 0, 1, 2, 2, 2], [0, 1, 2, 3, 4, 5, 6]])
    alpha = agg.attention_weights(lesion_x, edge, 3)
    assert alpha.shape == (7,) and torch.isfinite(alpha).all()
    for p in range(3):
        assert abs(float(alpha[edge[0] == p].sum()) - 1.0) < 1e-5


def test_attention_weights_requires_attention_pooling():
    agg = RadiologyLesionAttentionAggregator(15, 32, pooling="mean")
    with pytest.raises(RuntimeError):
        agg.attention_weights(torch.randn(1, 15), torch.tensor([[0], [0]]), 1)


# --- GVAE end-to-end with both flags ------------------------------------------------

def _set_all_encoders(model_config, encoder_type):
    for v in model_config["view_configs"].values():
        v["encoder_type"] = encoder_type


def test_gvae_forward_with_mlp_encoder_and_mean_pooling(synthetic_data, legacy_model_config):
    cfg = copy.deepcopy(legacy_model_config)
    _set_all_encoders(cfg, "mlp")
    cfg["radiology_aggregator_config"]["pooling"] = "mean"
    model = GVAE(**cfg).eval()
    idx = torch.arange(synthetic_data["patient"].num_nodes)
    logits, vae_out, cl_out, _ = model(synthetic_data, idx)
    assert logits.shape[0] == idx.numel()
    assert torch.isfinite(logits).all()


def test_gvae_defaults_unchanged(synthetic_data, legacy_model_config):
    """Configs without the new keys must still build the GAT + attention model."""
    model = GVAE(**copy.deepcopy(legacy_model_config)).eval()
    assert all(enc.encoder_type == "gat" for enc in model.vae_encoders.values())
    assert model.radiology_lesion_aggregator.pooling == "attention"
