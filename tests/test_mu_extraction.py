import torch
from models.gvae_model import GVAE, get_separate_view_mus


def test_get_separate_view_mus_matches_full_forward(legacy_model_config, synthetic_data):
    model = GVAE(**legacy_model_config).eval()
    idx = torch.arange(6)

    _, vae_out, _, _ = model(synthetic_data, idx)

    mus = get_separate_view_mus(model, synthetic_data, idx)
    assert set(mus.keys()) == {'clinical', 'pathology', 'radiology'}
    for v in mus:
        assert mus[v].shape == (6, legacy_model_config['d_embed'])
        assert torch.isfinite(mus[v]).all()


def test_forward_skips_structure_when_disabled(legacy_model_config, synthetic_data):
    model = GVAE(**legacy_model_config).eval()
    idx = torch.arange(6)
    _, vae_out, _, _ = model(synthetic_data, idx, compute_structure=False)
    for v, vo in vae_out.items():
        if vo and vo.get('mu') is not None:
            assert vo['rec_adj_logits'] is None
            assert vo['original_adj_subset'] is None
