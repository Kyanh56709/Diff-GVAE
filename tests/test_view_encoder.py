import torch
from models.gvae_components import ViewEncoder


def test_logvar_clamped_within_bounds():
    enc = ViewEncoder(8, 16, 4, heads=2, num_gnn_layers=1, edge_dim=-1,
                      logvar_clamp=(-6.0, 2.0)).eval()
    x = torch.randn(5, 8) * 100
    ei = torch.tensor([[0, 1], [1, 2]])
    _, logvar = enc(x, ei, None)
    assert logvar.max().item() <= 2.0 + 1e-5
    assert logvar.min().item() >= -6.0 - 1e-5


def test_logvar_unclamped_by_default():
    enc = ViewEncoder(8, 16, 4, heads=2, num_gnn_layers=1, edge_dim=-1).eval()
    x = torch.randn(5, 8)
    ei = torch.tensor([[0, 1], [1, 2]])
    mu, logvar = enc(x, ei, None)
    assert mu.shape == (5, 4) and logvar.shape == (5, 4)
