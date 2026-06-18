import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from training.latent_ddpm_augmentation import (
    apply_train_only_pca,
    filter_synthetic_latents_by_knn,
    generate_synthetic_latents,
)


class DummyConditionalDDPM:
    latent_dim = 2

    def eval(self):
        return self

    def sample(self, y, guidance_scale=2.0):
        return torch.zeros((y.shape[0], self.latent_dim), device=y.device)


def test_generate_synthetic_latents_respects_augmentation_modes():
    train_latents = np.array(
        [[0.0, 0.0], [0.1, 0.2], [0.2, 0.1], [1.0, 1.0]],
        dtype=np.float32,
    )
    labels = torch.tensor([0, 0, 0, 1])
    scaler = StandardScaler().fit(train_latents)
    ddpm = DummyConditionalDDPM()

    minority = generate_synthetic_latents(
        ddpm,
        scaler,
        labels,
        ratio=1.0,
        device=torch.device("cpu"),
        augmentation_mode="minority_only",
        batch_size=8,
    )
    responder = generate_synthetic_latents(
        ddpm,
        scaler,
        labels,
        ratio=1.0,
        device=torch.device("cpu"),
        augmentation_mode="responder_only",
        batch_size=8,
    )
    both = generate_synthetic_latents(
        ddpm,
        scaler,
        labels,
        ratio=1.0,
        device=torch.device("cpu"),
        augmentation_mode="both_classes",
        batch_size=8,
    )

    assert minority["per_class_counts"] == {"0": 0, "1": 1}
    assert responder["per_class_counts"] == {"0": 0, "1": 1}
    assert both["per_class_counts"] == {"0": 3, "1": 1}


def test_filter_synthetic_latents_uses_same_class_train_knn_threshold():
    real = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]], dtype=np.float32)
    real_labels = np.array([0, 0, 0, 1, 1, 1])
    generated = np.array([[0.5], [100.0], [10.5], [-100.0]], dtype=np.float32)
    generated_labels = np.array([0, 0, 1, 1])

    filtered = filter_synthetic_latents_by_knn(
        real,
        real_labels,
        generated,
        generated_labels,
        quantile=0.95,
    )

    assert filtered["kept_count"] == 2
    assert filtered["removed_count"] == 2
    assert filtered["labels"].tolist() == [0, 1]
    assert filtered["filter"]["fit_split"] == "train"


def test_apply_train_only_pca_transforms_train_and_validation():
    train = torch.randn(5, 4)
    val = torch.randn(3, 4)

    train_pca, val_pca, metadata = apply_train_only_pca(
        train,
        val,
        n_components=2,
        seed=7,
    )

    assert train_pca.shape == (5, 2)
    assert val_pca.shape == (3, 2)
    assert metadata["enabled"] is True
    assert metadata["fit_split"] == "train"
    assert metadata["n_components"] == 2
