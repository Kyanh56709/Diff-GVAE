import pytest
import torch

from models.gvae_model import GVAE
from utils.latent_extraction import (
    extract_latents_for_ddpm,
    extract_recommended_latents_for_ddpm,
)


def test_extract_latents_for_ddpm_saves_encoder_latents_only(
    tmp_path,
    legacy_model_config,
    synthetic_data,
):
    model = GVAE(**legacy_model_config).eval()
    checkpoint_path = tmp_path / "fold_1_rank_1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": legacy_model_config,
            "train_config": {"run_id": "latent_test"},
            "fold": 1,
            "rank_by_val_auc": 1,
            "epoch": 3,
            "validation_auc": 0.75,
            "validation_loss": 1.25,
            "train_indices": torch.tensor([0, 1, 2]),
            "val_indices": torch.tensor([3, 4, 5]),
        },
        checkpoint_path,
    )

    artifact_path = extract_latents_for_ddpm(
        checkpoint_path=checkpoint_path,
        full_data=synthetic_data,
        output_dir=tmp_path / "latent_for_ddpm",
        device="cpu",
        sample_seed=123,
    )

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert artifact["classifier_head_used_as_ddpm_input"] is False
    assert artifact["primary_recommended_ddpm_input"] == "concat_mu"

    train_split = artifact["splits"]["train"]
    num_views = len(legacy_model_config["view_configs"])
    d_embed = legacy_model_config["d_embed"]

    assert train_split["labels"].tolist() == [0, 1, 0]
    assert train_split["patient_ids"].tolist() == [0, 1, 2]
    assert train_split["view_mask"].shape == (3, num_views)
    assert train_split["stacked_mu"].shape == (3, num_views, d_embed)
    assert train_split["stacked_z"].shape == (3, num_views, d_embed)
    assert train_split["concat_mu"].shape == (3, num_views * d_embed)
    assert train_split["fused_cls_mu"].shape == (3, d_embed)
    assert "gvae_classifier_logits_from_mu" not in train_split

    with pytest.raises(FileExistsError):
        extract_latents_for_ddpm(
            checkpoint_path=checkpoint_path,
            full_data=synthetic_data,
            output_dir=tmp_path / "latent_for_ddpm",
            device="cpu",
            sample_seed=123,
        )


def test_extract_recommended_latents_for_ddpm_writes_split_files(
    tmp_path,
    legacy_model_config,
    synthetic_data,
):
    model = GVAE(**legacy_model_config).eval()
    checkpoint_path = tmp_path / "fold_2_rank_1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": legacy_model_config,
            "train_config": {"run_id": "file_layout_test", "device": torch.device("cpu")},
            "fold": 2,
            "rank_by_val_auc": 1,
            "epoch": 7,
            "validation_auc": 0.81,
            "validation_loss": 0.42,
            "train_indices": torch.tensor([0, 1]),
            "val_indices": torch.tensor([2, 3]),
            "test_indices": torch.tensor([4, 5]),
        },
        checkpoint_path,
    )

    output_dir = extract_recommended_latents_for_ddpm(
        checkpoint_path=checkpoint_path,
        full_data=synthetic_data,
        output_dir=tmp_path / "latent_for_ddpm",
        device="cpu",
        sample_seed=123,
    )

    assert output_dir.name == "fold_2_rank_1_concat_mu"
    for split_name, labels in {
        "train": [0, 1],
        "val": [0, 1],
        "test": [0, 1],
    }.items():
        split_dir = output_dir / split_name
        embeddings = torch.load(split_dir / "embeddings.pt", map_location="cpu", weights_only=False)
        saved_labels = torch.load(split_dir / "labels.pt", map_location="cpu", weights_only=False)
        patient_ids = torch.load(split_dir / "patient_ids.pt", map_location="cpu", weights_only=False)
        indices = torch.load(split_dir / "indices.pt", map_location="cpu", weights_only=False)

        assert embeddings.shape == (2, 3 * legacy_model_config["d_embed"])
        assert saved_labels.tolist() == labels
        assert patient_ids.tolist() == indices.tolist()

    for metadata_file in [
        "checkpoint_metadata.json",
        "extraction_config.json",
        "fold_info.json",
        "manifest.json",
        "model_config.json",
        "train_config.json",
    ]:
        assert (output_dir / metadata_file).exists()

    with pytest.raises(FileExistsError):
        extract_recommended_latents_for_ddpm(
            checkpoint_path=checkpoint_path,
            full_data=synthetic_data,
            output_dir=tmp_path / "latent_for_ddpm",
            device="cpu",
            sample_seed=123,
        )


def test_extract_latents_handles_split_with_missing_radiology_view(
    tmp_path,
    legacy_model_config,
    synthetic_data,
):
    model = GVAE(**legacy_model_config).eval()
    checkpoint_path = tmp_path / "missing_view.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": legacy_model_config,
            "train_config": {"run_id": "missing_view_test"},
            "fold": 1,
            "train_indices": torch.tensor([1]),
            "val_indices": torch.tensor([0, 2]),
        },
        checkpoint_path,
    )

    artifact_path = extract_latents_for_ddpm(
        checkpoint_path=checkpoint_path,
        full_data=synthetic_data,
        output_dir=tmp_path / "latent_for_ddpm",
        device="cpu",
        sample_seed=123,
    )
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    train_split = artifact["splits"]["train"]

    assert train_split["concat_mu"].shape == (1, 3 * legacy_model_config["d_embed"])
    assert train_split["view_mask"].shape == (1, 3)
    assert train_split["view_mask"][0, 2].item() is False
