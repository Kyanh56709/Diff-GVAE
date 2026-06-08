import torch
from training.train_gvae import kfold_train_gvae


def test_kfold_runs_end_to_end(legacy_model_config, legacy_train_config, synthetic_data):
    data = synthetic_data.to(torch.device('cpu'))
    summary, df, roc = kfold_train_gvae(data, legacy_model_config, legacy_train_config)
    # 2 folds requested; each should produce a metrics row
    assert isinstance(summary, dict)
    assert len(df) >= 1
    for v in summary.values():
        assert v == v  # not NaN
