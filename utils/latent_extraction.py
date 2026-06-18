from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import json

import torch
from torch_geometric.data import HeteroData

from models.gvae_model import GVAE, get_separate_view_latent_params


def _prepare_output_path(path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Latent artifact already exists: {path}. "
            "Pass overwrite=True or use a different output directory/run id."
        )
    return path


def _prepare_output_dir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Latent output directory already exists and is not empty: {path}. "
            "Pass overwrite=True or use a different output directory/run id."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _detach_to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_to_cpu(v) for v in value)
    return value


def _json_ready(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except TypeError:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2)


def _patient_ids(full_data: HeteroData, indices: torch.Tensor):
    patient_store = full_data['patient']
    if hasattr(patient_store, 'main_index'):
        main_index = patient_store.main_index
        if torch.is_tensor(main_index):
            return main_index[indices].detach().cpu()
        index_list = indices.detach().cpu().tolist()
        return [main_index[int(i)] for i in index_list]
    return indices.detach().cpu()


def _split_indices_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    split_indices = {}
    if 'train_indices' in checkpoint:
        split_indices['train'] = checkpoint['train_indices']
    if 'val_indices' in checkpoint:
        split_indices['val'] = checkpoint['val_indices']
    if 'test_indices' in checkpoint:
        split_indices['test'] = checkpoint['test_indices']
    return split_indices


def _checkpoint_metadata(checkpoint: Dict[str, Any], checkpoint_path: Path) -> Dict[str, Any]:
    keys = [
        'fold',
        'rank_by_val_auc',
        'epoch',
        'validation_auc',
        'validation_loss',
        'validation_metrics',
        'best_val_auc',
        'best_val_loss',
        'selected_threshold',
        'threshold_strategy',
        'checkpoint_metric',
        'selection_stage',
        'random_seed',
    ]
    metadata = {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_name': checkpoint_path.name,
    }
    metadata.update({key: checkpoint.get(key) for key in keys if key in checkpoint})
    return metadata


def _default_extraction_dir(checkpoint: Dict[str, Any], checkpoint_path: Path,
                            output_dir: Path, latent_key: str) -> Path:
    run_id = checkpoint.get('train_config', {}).get('run_id', checkpoint_path.parent.name)
    fold = checkpoint.get('fold')
    fold_dir = f"fold_{fold}" if fold is not None else "fold_unknown"
    return output_dir / str(run_id) / fold_dir / f"{checkpoint_path.stem}_{latent_key}"


@torch.no_grad()
def _extract_split_latents(
    gvae_model: GVAE,
    full_data: HeteroData,
    indices: torch.Tensor,
    sample_seed: int,
) -> Dict[str, Any]:
    views = list(gvae_model.views)
    labels = full_data['patient']['binary_label'].to(indices.device)[indices]
    params = get_separate_view_latent_params(gvae_model, full_data, indices)

    stacked_mu = torch.stack([params[view]['mu'] for view in views], dim=1)
    stacked_logvar = torch.stack([params[view]['logvar'] for view in views], dim=1)
    view_mask = torch.stack([params[view]['mask'] for view in views], dim=1)

    generator = torch.Generator(device=stacked_mu.device)
    generator.manual_seed(sample_seed)
    eps = torch.randn(
        stacked_mu.shape,
        device=stacked_mu.device,
        dtype=stacked_mu.dtype,
        generator=generator,
    )
    stacked_z = stacked_mu + torch.exp(0.5 * stacked_logvar) * eps

    fused_cls_mu = gvae_model.fusion_and_classifier_head.fuse(stacked_mu)

    return {
        'indices': indices.detach().cpu(),
        'patient_ids': _patient_ids(full_data, indices),
        'labels': labels.detach().cpu(),
        'view_names': views,
        'view_mask': view_mask.detach().cpu(),
        'per_view_mu': {
            view: params[view]['mu'].detach().cpu()
            for view in views
        },
        'per_view_logvar': {
            view: params[view]['logvar'].detach().cpu()
            for view in views
        },
        'stacked_mu': stacked_mu.detach().cpu(),
        'stacked_z': stacked_z.detach().cpu(),
        'concat_mu': stacked_mu.reshape(stacked_mu.shape[0], -1).detach().cpu(),
        'concat_z': stacked_z.reshape(stacked_z.shape[0], -1).detach().cpu(),
        'fused_cls_mu': fused_cls_mu.detach().cpu(),
    }


def extract_latents_for_ddpm(
    checkpoint_path: str | Path,
    full_data: HeteroData,
    output_dir: str | Path = 'outputs/latent_for_ddpm',
    split_indices: Optional[Dict[str, Iterable[int]]] = None,
    device: Optional[torch.device | str] = None,
    sample_seed: int = 0,
    overwrite: bool = False,
    artifact_path: Optional[str | Path] = None,
    artifact_name: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    split_extras: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """Extract GVAE encoder latents for downstream DDPM experiments.

    The saved artifact includes deterministic encoder means (`mu`), one
    reproducible sampled latent (`z`), concatenated per-view latents, stacked
    multi-view latents, and the fused CLS representation before the classifier
    head. Classifier logits/probabilities are intentionally not saved as DDPM
    inputs.
    """
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in checkpoint or 'model_config' not in checkpoint:
        raise ValueError(f"Checkpoint is missing model_state_dict/model_config: {checkpoint_path}")

    run_id = checkpoint.get('train_config', {}).get('run_id', checkpoint_path.parent.name)
    if artifact_path is None:
        fold = checkpoint.get('fold')
        fold_dir = f"fold_{fold}" if fold is not None else "fold_unknown"
        name = artifact_name or f"{checkpoint_path.stem}_latents_for_ddpm.pt"
        artifact_path = Path(output_dir) / str(run_id) / fold_dir / name
    else:
        artifact_path = Path(artifact_path)
    artifact_path = _prepare_output_path(artifact_path, overwrite)

    data_on_device = full_data.clone().to(device)
    model = GVAE(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if split_indices is None:
        split_indices = {}
        if 'train_indices' in checkpoint:
            split_indices['train'] = checkpoint['train_indices']
        if 'val_indices' in checkpoint:
            split_indices['val'] = checkpoint['val_indices']
        if 'test_indices' in checkpoint:
            split_indices['test'] = checkpoint['test_indices']
    if not split_indices:
        raise ValueError(
            "No split indices were provided and checkpoint does not contain "
            "train_indices/val_indices/test_indices."
        )

    splits: Dict[str, Any] = {}
    for split_offset, (split_name, raw_indices) in enumerate(split_indices.items()):
        indices = torch.as_tensor(raw_indices, dtype=torch.long, device=device)
        splits[split_name] = _extract_split_latents(
            model,
            data_on_device,
            indices,
            sample_seed=sample_seed + split_offset,
        )
        if split_extras and split_name in split_extras:
            splits[split_name].update(_detach_to_cpu(split_extras[split_name]))

    payload = {
        'artifact_schema_version': 1,
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_name': checkpoint_path.name,
        'fold': checkpoint.get('fold'),
        'rank_by_val_auc': checkpoint.get('rank_by_val_auc'),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_validation_metrics': checkpoint.get('validation_metrics'),
        'checkpoint_validation_auc': checkpoint.get('validation_auc'),
        'checkpoint_validation_loss': checkpoint.get('validation_loss'),
        'model_config': checkpoint.get('model_config'),
        'train_config': checkpoint.get('train_config'),
        'views': list(model.views),
        'd_embed': model.d_embed,
        'sample_seed': sample_seed,
        'recommended_ddpm_inputs': [
            'concat_mu',
            'stacked_mu',
            'fused_cls_mu',
            'concat_z',
            'stacked_z',
        ],
        'primary_recommended_ddpm_input': 'concat_mu',
        'classifier_head_used_as_ddpm_input': False,
        'note': (
            "Use encoder latents for DDPM. "
            "Classifier logits/probabilities are not DDPM inputs."
        ),
        'splits': splits,
    }
    if extra_metadata:
        payload['extra_metadata'] = _detach_to_cpu(extra_metadata)

    torch.save(payload, artifact_path)

    return artifact_path


def extract_recommended_latents_for_ddpm(
    checkpoint_path: str | Path,
    full_data: HeteroData,
    output_dir: str | Path = 'outputs/latent_for_ddpm',
    split_indices: Optional[Dict[str, Iterable[int]]] = None,
    device: Optional[torch.device | str] = None,
    sample_seed: int = 0,
    overwrite: bool = False,
    latent_key: str = 'concat_mu',
    extraction_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export the recommended DDPM latent representation as split files.

    The primary representation is `concat_mu`: deterministic encoder means from
    each modality concatenated into one vector per patient. Labels and patient
    IDs are written as separate files so DDPM training code can consume the
    latent matrix without accidentally depending on classifier outputs.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in checkpoint or 'model_config' not in checkpoint:
        raise ValueError(f"Checkpoint is missing model_state_dict/model_config: {checkpoint_path}")

    if split_indices is None:
        split_indices = _split_indices_from_checkpoint(checkpoint)
    if not split_indices:
        raise ValueError(
            "No split indices were provided and checkpoint does not contain "
            "train_indices/val_indices/test_indices."
        )

    data_on_device = full_data.clone().to(device)
    model = GVAE(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    available_latent_keys = {
        'concat_mu',
        'stacked_mu',
        'fused_cls_mu',
        'concat_z',
        'stacked_z',
    }
    if latent_key not in available_latent_keys:
        raise ValueError(
            f"Unknown latent_key '{latent_key}'. "
            f"Choose one of {sorted(available_latent_keys)}."
        )

    extraction_dir = _default_extraction_dir(
        checkpoint,
        checkpoint_path,
        output_dir,
        latent_key,
    )
    extraction_dir = _prepare_output_dir(extraction_dir, overwrite)

    manifest: Dict[str, Any] = {
        'artifact_schema_version': 1,
        'latent_key': latent_key,
        'primary_recommended_ddpm_input': 'concat_mu',
        'classifier_head_used_as_ddpm_input': False,
        'output_dir': str(extraction_dir),
        'splits': {},
        'files': {
            'checkpoint_metadata': 'checkpoint_metadata.json',
            'extraction_config': 'extraction_config.json',
            'fold_info': 'fold_info.json',
            'model_config': 'model_config.json',
            'train_config': 'train_config.json',
        },
    }
    fold_info: Dict[str, Any] = {
        'fold': checkpoint.get('fold'),
        'split_names': list(split_indices.keys()),
        'splits': {},
    }

    for split_offset, (split_name, raw_indices) in enumerate(split_indices.items()):
        indices = torch.as_tensor(raw_indices, dtype=torch.long, device=device)
        split_payload = _extract_split_latents(
            model,
            data_on_device,
            indices,
            sample_seed=sample_seed + split_offset,
        )
        split_dir = extraction_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        embeddings = split_payload[latent_key]
        torch.save(embeddings, split_dir / 'embeddings.pt')
        torch.save(split_payload['labels'], split_dir / 'labels.pt')
        torch.save(split_payload['patient_ids'], split_dir / 'patient_ids.pt')
        torch.save(split_payload['indices'], split_dir / 'indices.pt')
        torch.save(split_payload['view_mask'], split_dir / 'view_mask.pt')

        manifest['splits'][split_name] = {
            'num_patients': int(indices.numel()),
            'embedding_shape': list(embeddings.shape),
            'files': {
                'embeddings': f"{split_name}/embeddings.pt",
                'labels': f"{split_name}/labels.pt",
                'patient_ids': f"{split_name}/patient_ids.pt",
                'indices': f"{split_name}/indices.pt",
                'view_mask': f"{split_name}/view_mask.pt",
            },
        }
        fold_info['splits'][split_name] = {
            'num_patients': int(indices.numel()),
            'indices': split_payload['indices'],
        }

    config_payload = {
        'latent_key': latent_key,
        'recommended_representation': 'concat_mu',
        'sample_seed': sample_seed,
        'device': str(device),
        'output_dir': str(output_dir),
        'views': list(model.views),
        'd_embed': model.d_embed,
        'user_config': extraction_config or {},
    }

    _write_json(extraction_dir / 'checkpoint_metadata.json',
                _checkpoint_metadata(checkpoint, checkpoint_path))
    _write_json(extraction_dir / 'extraction_config.json', config_payload)
    _write_json(extraction_dir / 'fold_info.json', fold_info)
    _write_json(extraction_dir / 'model_config.json', checkpoint.get('model_config', {}))
    _write_json(extraction_dir / 'train_config.json', checkpoint.get('train_config', {}))
    _write_json(extraction_dir / 'manifest.json', manifest)

    return extraction_dir
