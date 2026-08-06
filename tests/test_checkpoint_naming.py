"""Checkpoint filenames must reflect the actual checkpoint metric.

The top-k checkpoint filename previously hardcoded an ``_auc_`` suffix even
when ranking by ``latent_quality`` (or loss / f1 / pr_auc).  The suffix must
name and carry the metric actually used for ranking.
"""

import torch

from training.train_gvae import (
    _checkpoint_metric_label_and_value,
    kfold_train_gvae,
)

CPU = torch.device('cpu')


def _run_and_collect_checkpoint_names(
    legacy_model_config,
    legacy_train_config,
    synthetic_data,
    tmp_path,
    metric,
):
    mc = dict(legacy_model_config)
    tc = dict(legacy_train_config)
    tc['checkpoint_metric'] = metric
    tc['top_k_gvae_checkpoints'] = 1
    tc['save_best_fold_model'] = True
    tc['checkpoint_dir'] = str(tmp_path)
    tc['overwrite_checkpoints'] = True
    tc['compute_latent_quality_metrics'] = True
    data = synthetic_data.to(CPU)
    torch.manual_seed(0)
    kfold_train_gvae(data, mc, tc)
    # top-k rank checkpoints only; the *_best.pt artifact is not metric-named
    return sorted(
        path.name for path in tmp_path.glob('*.pt') if '_rank_' in path.name
    )


def test_auc_metric_keeps_auc_suffix(
    legacy_model_config, legacy_train_config, synthetic_data, tmp_path
):
    names = _run_and_collect_checkpoint_names(
        legacy_model_config, legacy_train_config, synthetic_data, tmp_path, 'auc'
    )
    assert names, 'expected top-k checkpoints to be written'
    for name in names:
        assert '_auc_' in name, name


def test_latent_quality_metric_uses_latent_quality_suffix(
    legacy_model_config, legacy_train_config, synthetic_data, tmp_path
):
    names = _run_and_collect_checkpoint_names(
        legacy_model_config,
        legacy_train_config,
        synthetic_data,
        tmp_path,
        'latent_quality',
    )
    assert names, 'expected top-k checkpoints to be written'
    for name in names:
        assert '_latent_quality_' in name, name
        assert '_auc_' not in name, name


def test_label_and_value_helper_matches_metric_cases():
    candidate = {
        'val_auc': 0.8,
        'val_pr_auc': 0.7,
        'val_balanced_accuracy': 0.6,
        'val_f1': 0.5,
        'val_loss': 1.2,
        'latent_quality_metrics': {
            'latent_quality_score': 0.42,
            'linear_probe_auc': 0.9,
        },
    }
    assert _checkpoint_metric_label_and_value(candidate, 'auc') == ('auc', 0.8)
    assert _checkpoint_metric_label_and_value(candidate, 'roc_auc') == ('auc', 0.8)
    assert _checkpoint_metric_label_and_value(candidate, 'pr_auc') == ('pr_auc', 0.7)
    assert _checkpoint_metric_label_and_value(candidate, 'balanced_accuracy') == (
        'balanced_accuracy',
        0.6,
    )
    assert _checkpoint_metric_label_and_value(candidate, 'f1') == ('f1', 0.5)
    assert _checkpoint_metric_label_and_value(candidate, 'loss') == ('loss', 1.2)
    label, value = _checkpoint_metric_label_and_value(candidate, 'latent_quality')
    assert label == 'latent_quality'
    assert abs(value - 0.42) < 1e-9
    label, value = _checkpoint_metric_label_and_value(
        candidate, 'latent_quality_score'
    )
    assert label == 'latent_quality'
    assert abs(value - 0.42) < 1e-9
    assert _checkpoint_metric_label_and_value(candidate, 'linear_probe_auc') == (
        'latent_linear_probe_auc',
        0.9,
    )


def test_label_and_value_helper_rejects_unknown_metric():
    import pytest

    with pytest.raises(ValueError):
        _checkpoint_metric_label_and_value({}, 'bogus_metric')
