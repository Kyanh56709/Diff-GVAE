"""Clinical feature indices must be config-driven.

The clinical view uses a canonical layout (continuous columns 0..4, binary
columns 5..21) that was previously hardcoded in several places.  The
indices must be overridable via train_config['clinical_cont_indices'] /
train_config['clinical_bin_indices'], with the canonical layout as the
fallback so that 22-column clinical data behaves exactly as before.
"""

import torch

from training import train_gvae as tg
from training.train_gvae import _resolve_clinical_indices

CPU = torch.device('cpu')


def test_default_indices_match_historical_hardcode():
    cont, bin_ = _resolve_clinical_indices({}, clinical_dim=22, device=CPU)
    assert torch.equal(cont, torch.arange(0, 5, device=CPU))
    assert torch.equal(bin_, torch.arange(5, 22, device=CPU))


def test_default_indices_clamped_to_clinical_dim():
    cont, bin_ = _resolve_clinical_indices({}, clinical_dim=10, device=CPU)
    assert cont.tolist() == [0, 1, 2, 3, 4]
    assert bin_.tolist() == [5, 6, 7, 8, 9]


def test_config_indices_override_defaults():
    cfg = {'clinical_cont_indices': [0, 1], 'clinical_bin_indices': [6, 7, 8]}
    cont, bin_ = _resolve_clinical_indices(cfg, clinical_dim=22, device=CPU)
    assert cont.tolist() == [0, 1]
    assert bin_.tolist() == [6, 7, 8]


def test_single_key_override_leaves_other_default():
    # each key is independently optional
    cont, bin_ = _resolve_clinical_indices(
        {'clinical_bin_indices': [7]}, clinical_dim=22, device=CPU
    )
    assert cont.tolist() == [0, 1, 2, 3, 4]
    assert bin_.tolist() == [7]


def test_kfold_training_wires_config_into_clinical_indices(
    legacy_model_config, legacy_train_config, synthetic_data, monkeypatch
):
    calls = []
    original = tg._resolve_clinical_indices

    def spy(train_config, clinical_dim, device):
        calls.append((dict(train_config), clinical_dim, device))
        return original(train_config, clinical_dim, device)

    monkeypatch.setattr(tg, '_resolve_clinical_indices', spy)

    tc = dict(legacy_train_config)
    tc['clinical_bin_indices'] = [6, 7]
    tc['clinical_cont_indices'] = [0, 1]
    data = synthetic_data.to(CPU)
    summary, df, roc = tg.kfold_train_gvae(data, dict(legacy_model_config), tc)

    assert calls, "kfold_train_gvae must resolve clinical indices via helper"
    # the caller's train_config (with overrides) must reach the helper
    for cfg, _, _ in calls:
        assert cfg['clinical_bin_indices'] == [6, 7]
        assert cfg['clinical_cont_indices'] == [0, 1]
    assert isinstance(summary, dict)
    for value in summary.values():
        assert value == value  # not NaN
