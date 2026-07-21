from __future__ import annotations

import importlib
from pathlib import Path


def test_legacy_files_moved():
    for old in [
        "data/data_247.pt",
        "training/train_pipeline.py",
        "outputs/gvae/train_gvae_ddpm_runner.py",
        "outputs/gvae/train_ddpm_from_gvae_checkpoints_runner.py",
    ]:
        assert not Path(old).exists(), f"{old} should have been moved"
    for new in [
        "deprecated/__init__.py",
        "deprecated/README.md",
        "deprecated/data_247.pt",
        "deprecated/train_pipeline.py",
        "deprecated/train_gvae_ddpm_runner.py",
        "deprecated/train_ddpm_from_gvae_checkpoints_runner.py",
    ]:
        assert Path(new).exists(), f"{new} should exist"


def test_moved_runners_import_from_deprecated():
    for runner in [
        "deprecated/train_gvae_ddpm_runner.py",
        "deprecated/train_ddpm_from_gvae_checkpoints_runner.py",
    ]:
        src = Path(runner).read_text()
        assert "from training.train_pipeline import" not in src
        assert "from deprecated.train_pipeline import" in src


def test_deprecated_train_pipeline_importable():
    mod = importlib.import_module("deprecated.train_pipeline")
    assert hasattr(mod, "kfold_gvae_ddpm_generative_classifier")
