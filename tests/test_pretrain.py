import torch
from training.train_gvae import pretrain_radiology_aggregator, sweep_pretrain_recipes


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
        epochs=10, pretrain_val_split=0.4, patience=5, seed=0
    )
    assert sd is not None


def test_pretrain_zero_epochs_baseline(synthetic_data):
    """epochs=0 is the 'no pretraining' baseline recipe: returns a (random-init)
    aggregator state_dict without running any training step."""
    data = synthetic_data.to(torch.device('cpu'))
    train_idx = torch.tensor([0, 2, 3, 4, 5])
    sd = pretrain_radiology_aggregator(
        data, train_idx, _agg_config(), torch.device('cpu'), epochs=0
    )
    assert sd is not None
    assert any('attention_mlp' in k for k in sd.keys())


def test_stratified_val_split_keeps_both_classes(synthetic_data, monkeypatch):
    """The stratified split must put both labels in val so val-AUC is defined.
    We capture the val_true arrays passed to roc_auc_score during pretraining."""
    import training.train_gvae as tg
    seen = {}
    real = tg.roc_auc_score

    def spy(y_true, y_score, *a, **k):
        import numpy as np
        seen['classes'] = set(np.unique(y_true).tolist())
        return real(y_true, y_score, *a, **k)

    monkeypatch.setattr(tg, 'roc_auc_score', spy)
    data = synthetic_data.to(torch.device('cpu'))
    train_idx = torch.tensor([0, 2, 3, 4, 5])  # labels among these: [0,0,1,0,1] -> 2 pos, 3 neg
    pretrain_radiology_aggregator(
        data, train_idx, _agg_config(), torch.device('cpu'),
        epochs=50, pretrain_val_split=0.4, patience=100, seed=0
    )
    assert seen.get('classes') == {0, 1}


def test_sweep_pretrain_recipes_smoke(legacy_model_config, legacy_train_config, synthetic_data):
    data = synthetic_data.to(torch.device('cpu'))
    recipes = {
        '(a) no pretrain': {'pretrain_epochs': 0},
        '(b) 20 ep': {'pretrain_epochs': 20, 'pretrain_val_split': 0.0},
    }
    results = sweep_pretrain_recipes(data, legacy_model_config, legacy_train_config, recipes=recipes)
    assert set(results.keys()) == set(recipes.keys())
    for mean_auc, std_auc in results.values():
        assert mean_auc == mean_auc  # not NaN (synthetic kfold produces a number)
