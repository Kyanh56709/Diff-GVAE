import torch
from utils.loss_utils import calculate_contrastive_loss, calculate_contrastive_loss_vectorized


def _two_view_batch():
    torch.manual_seed(1)
    idx = torch.tensor([10, 11, 12, 13])
    emb_a = torch.randn(4, 8)
    emb_b = torch.randn(4, 8)
    return {'clinical': (emb_a, idx), 'pathology': (emb_b, idx)}


def test_vectorized_matches_loop():
    batch = _two_view_batch()
    loop = calculate_contrastive_loss(batch, temperature=0.1)
    vec = calculate_contrastive_loss_vectorized(batch, temperature=0.1)
    assert torch.allclose(loop, vec, atol=1e-5), f"{loop.item()} vs {vec.item()}"


def test_vectorized_handles_empty():
    out = calculate_contrastive_loss_vectorized({}, temperature=0.1)
    assert out.item() == 0.0
