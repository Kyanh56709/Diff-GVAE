import torch
from training.train_gvae import kfold_train_gvae


def test_kfold_runs_with_all_flags_on(legacy_model_config, legacy_train_config, synthetic_data):
    mc = dict(legacy_model_config)
    mc['logvar_clamp'] = (-6.0, 2.0)
    mc['radiology_zero_lesion_passthrough'] = True

    tc = dict(legacy_train_config)
    tc['pretrain_use_pos_weight'] = True
    tc['pretrain_val_split'] = 0.3
    tc['vectorized_contrastive'] = True

    data = synthetic_data.to(torch.device('cpu'))
    summary, df, roc = kfold_train_gvae(data, mc, tc)
    assert isinstance(summary, dict)
    for v in summary.values():
        assert v == v  # not NaN
