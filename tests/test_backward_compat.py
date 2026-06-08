import torch
from models.gvae_model import GVAE


def test_gvae_builds_with_legacy_config(legacy_model_config):
    """The user's exact cell_4.py config (no new flags) must still build."""
    model = GVAE(**legacy_model_config)
    assert model is not None
    assert set(model.views) == {'clinical', 'pathology', 'radiology'}


def test_gvae_forward_runs_on_synthetic(legacy_model_config, synthetic_data):
    model = GVAE(**legacy_model_config).eval()
    idx = torch.arange(6)
    logits, vae_out, cl_out, _ = model(synthetic_data, idx)
    assert logits.shape[0] == 6
    assert torch.isfinite(logits).all()
