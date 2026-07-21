from pathlib import Path


def test_run_gvae_defaults_to_latent_quality():
    src = Path("run_gvae.py").read_text()
    assert '"checkpoint_metric": "latent_quality"' in src
    assert '"early_stopping_metric": "latent_quality"' in src
    assert "auc_pr_balanced_accuracy" not in src
    assert "train_gvae_runner.py" in src  # header points at the maintained runner
