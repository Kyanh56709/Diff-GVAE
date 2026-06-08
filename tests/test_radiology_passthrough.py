import torch
from models.gvae_model import GVAE


def test_zero_lesion_patient_passthrough(legacy_model_config, synthetic_data):
    # Patient 2 has radiology_mask=True but zero lesions; batch = [2] alone
    # => zero total lesion edges in the batch.
    idx = torch.tensor([2])

    off = GVAE(**legacy_model_config).eval()
    _, vae_off, _, _ = off(synthetic_data, idx)
    assert vae_off['radiology'].get('mu') is None  # current behavior: skipped

    cfg = dict(legacy_model_config)
    cfg['radiology_zero_lesion_passthrough'] = True
    on = GVAE(**cfg).eval()
    _, vae_on, _, _ = on(synthetic_data, idx)
    assert vae_on['radiology'].get('mu') is not None  # now encoded from zeros
    assert torch.isfinite(vae_on['radiology']['mu']).all()
