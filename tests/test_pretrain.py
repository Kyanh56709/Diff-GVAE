import torch
from training.train_gvae import pretrain_radiology_aggregator


def _agg_config():
    return {'lesion_feature_dim': 15, 'aggregated_output_dim': 32,
            'attention_hidden_dim': 32, 'dropout': 0.3}


def test_pretrain_runs_with_pos_weight(synthetic_data):
    data = synthetic_data.to(torch.device('cpu'))
    train_idx = torch.tensor([0, 2, 3, 4, 5])
    sd = pretrain_radiology_aggregator(
        data, train_idx, _agg_config(), torch.device('cpu'),
        epochs=5, use_pos_weight=True
    )
    assert sd is not None
    assert any('attention_mlp' in k for k in sd.keys())


def test_pretrain_runs_without_pos_weight_default(synthetic_data):
    data = synthetic_data.to(torch.device('cpu'))
    train_idx = torch.tensor([0, 2, 3, 4, 5])
    sd = pretrain_radiology_aggregator(data, train_idx, _agg_config(), torch.device('cpu'), epochs=5)
    assert sd is not None


def test_pretrain_with_val_split(synthetic_data):
    data = synthetic_data.to(torch.device('cpu'))
    train_idx = torch.tensor([0, 2, 3, 4, 5])
    sd = pretrain_radiology_aggregator(
        data, train_idx, _agg_config(), torch.device('cpu'),
        epochs=10, pretrain_val_split=0.4
    )
    assert sd is not None
