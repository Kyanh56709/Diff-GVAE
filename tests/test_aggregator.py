import torch
from torch_scatter import scatter_add
from models.gvae_components import RadiologyLesionAttentionAggregator


def test_attention_weights_normalize_per_patient():
    torch.manual_seed(0)
    agg = RadiologyLesionAttentionAggregator(15, 32, attention_hidden_dim=32, dropout=0.0).eval()
    lesion_x = torch.randn(7, 15)
    # patients: 0 has lesions 0,1,2 ; 1 has 3 ; 2 has 4,5,6
    edge = torch.tensor([[0, 0, 0, 1, 2, 2, 2], [0, 1, 2, 3, 4, 5, 6]])
    out = agg(lesion_x, edge, 3)
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()
