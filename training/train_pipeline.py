import numpy as np
import copy
import csv
import json
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.utils.data
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from typing import Dict, Any, List, Tuple
from models.gvae_model import GVAE, get_separate_view_mus
from utils.data_utils import preprocess_fold_data_with_pca
# from models.ddpm import UnconditionalDenoisingNetwork, UnconditionalDDPM
from training.train_gvae import train_gvae_single_fold
from training.train_ddpm import train_single_unconditional_ddpm
from models.gvae_components import MuFusionTransformer
from utils.data_utils import get_view_subgraph_and_features
from utils.latent_extraction import extract_latents_for_ddpm
from utils.classification_eval import (
    build_binary_classification_diagnostics,
    save_binary_classification_artifacts,
)
from tqdm import tqdm
#from utils.data_utils import preprocess_data_with_pca


def _prepare_artifact_path(path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Artifact already exists: {path}. "
            "Set overwrite_checkpoints=True or choose a new run_id/artifact_dir."
        )
    return path


def _patient_ids_for_indices(full_data, indices) -> List[Any] | None:
    if 'main_index' not in full_data['patient']:
        return None
    main_index = full_data['patient']['main_index']
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    return [main_index[int(idx)] for idx in np.asarray(indices).tolist()]


def _json_ready(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(_json_ready(payload), f, indent=2)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else np.nan


def _safe_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else np.nan


def _safe_brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return brier_score_loss(labels, probs)
    except ValueError:
        return np.nan


def _class_distribution(labels: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    total = int(labels.size)
    return {
        'n': total,
        'positive': positives,
        'negative': negatives,
        'positive_rate': float(positives / total) if total else np.nan,
        'negative_rate': float(negatives / total) if total else np.nan,
    }


def _confusion_matrix_dict(labels: np.ndarray, preds: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        'labels': ['negative', 'positive'],
        'matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def _metrics_at_threshold(labels: np.ndarray, scores: np.ndarray,
                          threshold: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores > threshold).astype(int)
    cm = _confusion_matrix_dict(labels, preds)
    tn, fp, fn, tp = cm['tn'], cm['fp'], cm['fn'], cm['tp']
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    metrics = {
        'threshold': float(threshold),
        'auc': _safe_binary_auc(labels, scores),
        'pr_auc': _safe_pr_auc(labels, scores),
        'f1': f1_score(labels, preds, zero_division=0),
        'accuracy': accuracy_score(labels, preds),
        'balanced_accuracy': balanced_accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'brier_score': _safe_brier_score(labels, scores),
    }
    return metrics, cm


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.array([0.5], dtype=float)
    unique_scores = np.unique(scores)
    candidates = [0.5, 0.0, 1.0]
    eps = 1e-12
    candidates.append(max(0.0, float(unique_scores[0]) - eps))
    candidates.append(min(1.0, float(unique_scores[-1]) + eps))
    if unique_scores.size > 1:
        mids = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates.extend(float(x) for x in mids)
    candidates.extend(float(x) for x in unique_scores)
    return np.array(sorted(set(candidates)), dtype=float)


def _best_threshold_metrics(labels: np.ndarray, scores: np.ndarray,
                            selection_metric: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    best_metrics = None
    best_cm = None
    best_key = None
    for threshold in _threshold_candidates(scores):
        metrics, cm = _metrics_at_threshold(labels, scores, threshold)
        metric_value = metrics.get(selection_metric)
        if metric_value is None or np.isnan(metric_value):
            continue
        key = (
            metric_value,
            metrics.get('balanced_accuracy', np.nan),
            metrics.get('f1', np.nan),
            metrics.get('specificity', np.nan),
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics
            best_cm = cm
    if best_metrics is None:
        best_metrics, best_cm = _metrics_at_threshold(labels, scores, 0.5)
    best_metrics['threshold_selection_metric'] = selection_metric
    return best_metrics, best_cm


def _score_distribution(scores: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return {}
    return {
        'count': int(scores.size),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'median': float(np.median(scores)),
        'q25': float(np.quantile(scores, 0.25)),
        'q75': float(np.quantile(scores, 0.75)),
        'num_above_0_5': int(np.sum(scores > 0.5)),
        'num_at_or_below_0_5': int(np.sum(scores <= 0.5)),
        'all_above_0_5': bool(np.all(scores > 0.5)),
    }


def _curve_data(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) <= 1:
        return {
            'roc': {'fpr': [], 'tpr': [], 'thresholds': []},
            'pr': {'precision': [], 'recall': [], 'thresholds': []},
        }
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    return {
        'roc': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': roc_thresholds,
        },
        'pr': {
            'precision': precision,
            'recall': recall,
            'thresholds': pr_thresholds,
        },
    }


def _save_curve_csvs(eval_dir: Path, curves: Dict[str, Any]) -> Dict[str, str]:
    roc_rows = [
        {
            'fpr': float(fpr),
            'tpr': float(tpr),
            'threshold': float(threshold),
        }
        for fpr, tpr, threshold in zip(
            curves['roc']['fpr'],
            curves['roc']['tpr'],
            curves['roc']['thresholds'],
        )
    ]
    _write_csv(eval_dir / 'roc_curve.csv', roc_rows, ['fpr', 'tpr', 'threshold'])

    pr_thresholds = list(curves['pr']['thresholds'])
    pr_rows = []
    for idx, (precision, recall) in enumerate(
        zip(curves['pr']['precision'], curves['pr']['recall'])
    ):
        threshold = pr_thresholds[idx] if idx < len(pr_thresholds) else np.nan
        pr_rows.append({
            'precision': float(precision),
            'recall': float(recall),
            'threshold': float(threshold) if not np.isnan(threshold) else '',
        })
    _write_csv(eval_dir / 'pr_curve.csv', pr_rows, ['precision', 'recall', 'threshold'])
    return {
        'roc_curve_csv': str(eval_dir / 'roc_curve.csv'),
        'pr_curve_csv': str(eval_dir / 'pr_curve.csv'),
    }


def _save_prediction_csv(eval_dir: Path, labels: np.ndarray, probs: np.ndarray,
                         loss_responder: np.ndarray, loss_non_responder: np.ndarray,
                         indices: torch.Tensor, threshold_0_5: float,
                         best_threshold: float) -> str:
    score_loss_margin = loss_non_responder - loss_responder
    rows = []
    for idx, label, prob, loss_resp, loss_non_resp, margin in zip(
        indices.detach().cpu().numpy().tolist(),
        labels.tolist(),
        probs.tolist(),
        loss_responder.tolist(),
        loss_non_responder.tolist(),
        score_loss_margin.tolist(),
    ):
        rows.append({
            'patient_index': int(idx),
            'label': int(label),
            'ddpm_probability': float(prob),
            'ddpm_score_loss_margin': float(margin),
            'loss_responder': float(loss_resp),
            'loss_non_responder': float(loss_non_resp),
            'pred_threshold_0_5': int(prob > threshold_0_5),
            'pred_best_threshold': int(prob > best_threshold),
        })
    path = eval_dir / 'predicted_probabilities.csv'
    _write_csv(
        path,
        rows,
        [
            'patient_index',
            'label',
            'ddpm_probability',
            'ddpm_score_loss_margin',
            'loss_responder',
            'loss_non_responder',
            'pred_threshold_0_5',
            'pred_best_threshold',
        ],
    )
    return str(path)


def _save_ddpm_evaluation_artifacts(
    eval_dir: Path,
    labels: np.ndarray,
    probs: np.ndarray,
    loss_responder: np.ndarray,
    loss_non_responder: np.ndarray,
    train_labels: np.ndarray,
    val_indices: torch.Tensor,
    threshold_selection_metric: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=True)

    threshold_0_5_metrics, threshold_0_5_cm = _metrics_at_threshold(labels, probs, 0.5)
    best_metrics, best_cm = _best_threshold_metrics(
        labels,
        probs,
        threshold_selection_metric,
    )
    best_f1_metrics, best_f1_cm = _best_threshold_metrics(labels, probs, 'f1')
    best_balanced_accuracy_metrics, best_balanced_accuracy_cm = _best_threshold_metrics(
        labels,
        probs,
        'balanced_accuracy',
    )
    curves = _curve_data(labels, probs)

    class_distribution = {
        'train': _class_distribution(train_labels),
        'val': _class_distribution(labels),
    }
    prediction_distribution = _score_distribution(probs)
    prevalence = class_distribution['val']['positive_rate']
    pr_auc = threshold_0_5_metrics['pr_auc']
    normalized_ap = (
        (pr_auc - prevalence) / (1.0 - prevalence)
        if not np.isnan(pr_auc) and prevalence < 1.0 else np.nan
    )

    _write_json(eval_dir / 'class_distribution.json', class_distribution)
    _write_json(eval_dir / 'confusion_matrix_threshold_0_5.json', threshold_0_5_cm)
    _write_json(eval_dir / 'confusion_matrix_best_threshold.json', best_cm)
    curve_paths = _save_curve_csvs(eval_dir, curves)
    prediction_path = _save_prediction_csv(
        eval_dir,
        labels,
        probs,
        loss_responder,
        loss_non_responder,
        val_indices,
        threshold_0_5=0.5,
        best_threshold=best_metrics['threshold'],
    )

    diagnostics = {
        'metadata': metadata,
        'class_distribution': class_distribution,
        'prediction_distribution': prediction_distribution,
        'threshold_0_5_metrics': threshold_0_5_metrics,
        'best_threshold_metrics': best_metrics,
        'best_f1_threshold_metrics': best_f1_metrics,
        'best_balanced_accuracy_threshold_metrics': best_balanced_accuracy_metrics,
        'confusion_matrix_threshold_0_5': threshold_0_5_cm,
        'confusion_matrix_best_threshold': best_cm,
        'confusion_matrix_best_f1_threshold': best_f1_cm,
        'confusion_matrix_best_balanced_accuracy_threshold': (
            best_balanced_accuracy_cm
        ),
        'pr_auc_baseline_positive_prevalence': prevalence,
        'normalized_average_precision': normalized_ap,
        'probability_calibration_note': (
            "DDPM probability is an inverse-loss normalized score, not a "
            "calibrated posterior probability."
        ),
        'files': {
            'metrics_json': str(eval_dir / 'metrics.json'),
            'class_distribution_json': str(eval_dir / 'class_distribution.json'),
            'confusion_matrix_threshold_0_5_json': str(
                eval_dir / 'confusion_matrix_threshold_0_5.json'
            ),
            'confusion_matrix_best_threshold_json': str(
                eval_dir / 'confusion_matrix_best_threshold.json'
            ),
            'predicted_probabilities_csv': prediction_path,
            **curve_paths,
        },
    }
    _write_json(eval_dir / 'metrics.json', diagnostics)
    return diagnostics


def _classification_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    metrics, _ = _metrics_at_threshold(labels, probs, 0.5)
    return {
        key: metrics[key]
        for key in [
            'auc',
            'pr_auc',
            'f1',
            'accuracy',
            'balanced_accuracy',
            'precision',
            'recall',
            'sensitivity',
            'specificity',
            'brier_score',
        ]
    }

@torch.no_grad()
def get_all_view_mus_from_gvae(
    gvae_model: GVAE,
    full_data: torch.utils.data.Dataset,
    indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts mu vectors for each view for the given patient indices and prepares them as a stacked tensor.
    Missing views are replaced with the GVAE model's learnable missing embeddings.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - A stacked tensor of mu vectors. Shape: [num_indices, num_views, d_embed].
            - The corresponding labels for these indices.
    """
    gvae_model.eval()
    device = indices.device

    mus_by_view = get_separate_view_mus(gvae_model, full_data, indices)
    labels = full_data['patient']['binary_label'].to(device)[indices]
    stacked_mus = torch.stack([mus_by_view[view] for view in gvae_model.views], dim=1)

    return stacked_mus, labels


def _save_gvae_candidate_checkpoint(
    candidate: Dict[str, Any],
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    fold_num: int,
    train_idx_np: np.ndarray,
    val_idx_np: np.ndarray,
    checkpoint_path: Path,
    overwrite: bool,
) -> Path:
    checkpoint_path = _prepare_artifact_path(checkpoint_path, overwrite)
    torch.save({
        'model_state_dict': candidate['model_state_dict'],
        'model_config': model_config,
        'train_config': train_config,
        'fold': fold_num,
        'rank_by_val_auc': candidate['rank_by_val_auc'],
        'rank_by_gvae_checkpoint_metric': candidate.get(
            'rank_by_gvae_checkpoint_metric',
            candidate['rank_by_val_auc'],
        ),
        'rank_by_latent_quality': candidate.get('rank_by_latent_quality'),
        'epoch': candidate['epoch'],
        'validation_auc': candidate['val_auc'],
        'validation_pr_auc': candidate.get('val_pr_auc'),
        'validation_balanced_accuracy': candidate.get('val_balanced_accuracy'),
        'validation_f1': candidate.get('val_f1'),
        'validation_loss': candidate['val_loss'],
        'checkpoint_metric': candidate.get(
            'checkpoint_metric',
            train_config.get('checkpoint_metric'),
        ),
        'validation_diagnostics': candidate.get('validation_diagnostics'),
        'latent_quality_metrics': candidate.get('latent_quality_metrics'),
        'train_indices': train_idx_np,
        'val_indices': val_idx_np,
        'selection_stage': 'top_k_by_gvae_validation_metric',
    }, checkpoint_path)
    return checkpoint_path


def _train_supervised_mu_fusion(
    stacked_mus_train: torch.Tensor,
    stacked_mus_val: torch.Tensor,
    train_labels: torch.Tensor,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    fold_num: int,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    device = train_config['device']
    fusion_transformer = MuFusionTransformer(
        d_embed=model_config['d_embed'],
        n_heads=train_config.get('mu_fusion_heads', 8),
        dim_feedforward=model_config['d_embed'] * train_config.get('mu_fusion_ffn_multiplier', 4),
        dropout=train_config.get('mu_fusion_dropout', 0.1),
    ).to(device)
    temp_classifier = torch.nn.Linear(model_config['d_embed'], 1).to(device)
    fusion_optimizer = torch.optim.AdamW(
        list(fusion_transformer.parameters()) + list(temp_classifier.parameters()),
        lr=train_config.get('mu_fusion_lr', 1e-3),
        weight_decay=train_config.get('mu_fusion_wd', 0.0),
    )
    bce_loss = torch.nn.BCEWithLogitsLoss()

    for _ in tqdm(
        range(train_config.get('mu_fusion_epochs', 200)),
        desc=f"F{fold_num} R{rank} Fusion",
        leave=False,
    ):
        fusion_transformer.train()
        fusion_optimizer.zero_grad()
        mu_fused = fusion_transformer(stacked_mus_train)
        logits = temp_classifier(mu_fused).squeeze(-1)
        loss = bce_loss(logits, train_labels.float())
        loss.backward()
        fusion_optimizer.step()

    fusion_transformer.eval()
    with torch.no_grad():
        train_latents = fusion_transformer(stacked_mus_train)
        val_latents = fusion_transformer(stacked_mus_val)

    metadata = {
        'fusion_transformer_state_dict': {
            k: v.detach().cpu()
            for k, v in fusion_transformer.state_dict().items()
        },
        'fusion_classifier_state_dict': {
            k: v.detach().cpu()
            for k, v in temp_classifier.state_dict().items()
        },
    }
    return train_latents, val_latents, metadata


def _select_ddpm_latents(
    gvae_model: GVAE,
    stacked_mus_train: torch.Tensor,
    stacked_mus_val: torch.Tensor,
    train_labels: torch.Tensor,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    fold_num: int,
    rank: int,
) -> Tuple[str, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    latent_name = train_config.get('ddpm_latent_representation', 'concat_mu')

    if latent_name == 'concat_mu':
        train_latents = stacked_mus_train.reshape(stacked_mus_train.shape[0], -1)
        val_latents = stacked_mus_val.reshape(stacked_mus_val.shape[0], -1)
        return latent_name, train_latents, val_latents, {}

    if latent_name == 'fused_cls_mu':
        with torch.no_grad():
            train_latents = gvae_model.fusion_and_classifier_head.fuse(stacked_mus_train)
            val_latents = gvae_model.fusion_and_classifier_head.fuse(stacked_mus_val)
        return latent_name, train_latents, val_latents, {}

    if latent_name == 'pipeline_supervised_fused_mu':
        train_latents, val_latents, metadata = _train_supervised_mu_fusion(
            stacked_mus_train,
            stacked_mus_val,
            train_labels,
            model_config,
            train_config,
            fold_num,
            rank,
        )
        return latent_name, train_latents, val_latents, metadata

    raise ValueError(
        f"Unknown ddpm_latent_representation: {latent_name}. "
        "Choose one of {'concat_mu', 'fused_cls_mu', 'pipeline_supervised_fused_mu'}."
    )


def _evaluate_gvae_candidate_with_ddpm(
    full_data: torch.utils.data.Dataset,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    ddpm_config: Dict[str, Any],
    candidate: Dict[str, Any],
    fold_num: int,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    candidate_checkpoint_path: Path,
    latents_path: Path,
    gvae_eval_dir: Path,
    eval_dir: Path,
) -> Dict[str, Any]:
    if not train_config.get('allow_deprecated_ddpm_classifier', False):
        raise RuntimeError(
            "_evaluate_gvae_candidate_with_ddpm treats DDPM loss as a "
            "classifier score and is disabled for this project objective. "
            "Use training.latent_ddpm_augmentation, which trains DDPM only for "
            "conditional latent-space augmentation and reports downstream "
            "classifier performance."
        )
    warnings.warn(
        "_evaluate_gvae_candidate_with_ddpm is deprecated for the paper objective. "
        "It treats DDPM as a loss-based classifier. Use "
        "training.latent_ddpm_augmentation for conditional latent-space "
        "augmentation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    device = train_config['device']
    overwrite = train_config.get('overwrite_checkpoints', False)

    gvae_model = GVAE(**model_config).to(device)
    gvae_model.load_state_dict(candidate['model_state_dict'])
    gvae_model.eval()

    stacked_mus_train, train_labels = get_all_view_mus_from_gvae(
        gvae_model, full_data, train_indices
    )
    stacked_mus_val, val_labels = get_all_view_mus_from_gvae(
        gvae_model, full_data, val_indices
    )

    gvae_threshold_selection_metric = train_config.get(
        'classification_threshold_selection_metric',
        'balanced_accuracy',
    )
    if gvae_eval_dir.exists() and any(gvae_eval_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"GVAE classifier evaluation artifact directory already exists: "
            f"{gvae_eval_dir}. Set overwrite_checkpoints=True or choose a new run_id."
        )
    with torch.no_grad():
        gvae_val_logits, _, _, _ = gvae_model(
            full_data,
            val_indices,
            compute_structure=False,
        )
        gvae_val_probs = torch.sigmoid(gvae_val_logits.squeeze(-1)).detach().cpu().numpy()
    gvae_val_labels = val_labels.detach().cpu().numpy()
    gvae_diagnostics = build_binary_classification_diagnostics(
        gvae_val_labels,
        gvae_val_probs,
        train_labels=train_labels.detach().cpu().numpy(),
        threshold_selection_metric=gvae_threshold_selection_metric,
        metadata={
            'model': 'GVAE',
            'fold': fold_num,
            'rank_by_val_auc': candidate['rank_by_val_auc'],
            'epoch': candidate['epoch'],
            'gvae_val_auc': candidate['val_auc'],
            'gvae_val_loss': candidate['val_loss'],
            'score_source': 'sigmoid(classifier_logit)',
            'bce_loss': 'BCEWithLogitsLoss',
            'sigmoid_applied_during_training': False,
            'sigmoid_applied_for_evaluation': True,
            'checkpoint_path': str(candidate_checkpoint_path),
        },
        probability_note=(
            'GVAE classifier probability is sigmoid(logit). The classifier head '
            'outputs logits and training uses BCEWithLogitsLoss.'
        ),
    )
    gvae_eval_files = save_binary_classification_artifacts(
        gvae_eval_dir,
        gvae_diagnostics,
        gvae_val_labels,
        gvae_val_probs,
        indices=val_indices,
        patient_ids=_patient_ids_for_indices(full_data, val_indices),
        score_column='gvae_probability',
    )
    gvae_diagnostics['files'] = gvae_eval_files

    latent_name, final_train_mus, final_val_mus, latent_metadata = _select_ddpm_latents(
        gvae_model,
        stacked_mus_train,
        stacked_mus_val,
        train_labels,
        model_config,
        train_config,
        fold_num,
        candidate['rank_by_val_auc'],
    )
    ddpm_config_for_candidate = copy.deepcopy(ddpm_config)
    ddpm_config_for_candidate['latent_dim'] = int(final_train_mus.shape[1])

    shared_latent_scaler = MinMaxScaler(feature_range=(-1, 1))
    shared_latent_scaler.fit(final_train_mus.detach().cpu().numpy())

    extra_metadata = {
        'ddpm_config': ddpm_config_for_candidate,
        'gvae_validation_auc': candidate['val_auc'],
        'gvae_validation_loss': candidate['val_loss'],
        'pipeline_ddpm_input_name': latent_name,
        'pipeline_ddpm_input_shape': list(final_train_mus.shape),
        'pipeline_ddpm_input_note': (
            "DDPM input is selected by train_config['ddpm_latent_representation']. "
            "Classifier logits/probabilities are not used as DDPM inputs."
        ),
        'shared_latent_scaler': shared_latent_scaler,
    }
    extra_metadata.update(latent_metadata)

    latents_path = extract_latents_for_ddpm(
        checkpoint_path=candidate_checkpoint_path,
        full_data=full_data,
        split_indices={
            'train': train_indices,
            'val': val_indices,
        },
        device=device,
        sample_seed=train_config.get('latent_sample_seed', 0),
        overwrite=overwrite,
        artifact_path=latents_path,
        extra_metadata=extra_metadata,
        split_extras={
            'train': {
                latent_name: final_train_mus,
                'pipeline_ddpm_input': final_train_mus,
            },
            'val': {
                latent_name: final_val_mus,
                'pipeline_ddpm_input': final_val_mus,
            },
        },
    )

    mus_responder = final_train_mus[train_labels == 1]
    mus_non_responder = final_train_mus[train_labels == 0]

    ddpm_responder, _ = train_single_unconditional_ddpm(
        mus_responder, ddpm_config_for_candidate, device, scaler=shared_latent_scaler
    )
    ddpm_non_responder, _ = train_single_unconditional_ddpm(
        mus_non_responder, ddpm_config_for_candidate, device, scaler=shared_latent_scaler
    )

    if ddpm_responder is None or ddpm_non_responder is None:
        nan_metrics = {
            m: np.nan
            for m in [
                'auc',
                'pr_auc',
                'f1',
                'accuracy',
                'balanced_accuracy',
                'precision',
                'recall',
                'sensitivity',
                'specificity',
                'brier_score',
            ]
        }
        return {
            'fold': fold_num,
            'rank_by_val_auc': candidate['rank_by_val_auc'],
            'epoch': candidate['epoch'],
            'gvae_val_auc': candidate['val_auc'],
            'gvae_val_loss': candidate['val_loss'],
            'ddpm_latent_representation': latent_name,
            'ddpm_latent_dim': ddpm_config_for_candidate['latent_dim'],
            'gvae_checkpoint_path': str(candidate_checkpoint_path),
            'latents_path': str(latents_path),
            'eval_dir': str(eval_dir),
            'gvae_classifier_eval_dir': str(gvae_eval_dir),
            'status': 'ddpm_skipped',
            'metrics': nan_metrics,
            'threshold_0_5_metrics': nan_metrics,
            'best_threshold_metrics': nan_metrics,
            'best_f1_threshold_metrics': nan_metrics,
            'best_balanced_accuracy_threshold_metrics': nan_metrics,
            'gvae_classifier_metrics_threshold_0_5': (
                gvae_diagnostics['threshold_0_5_metrics']
            ),
            'gvae_classifier_metrics_best_threshold': (
                gvae_diagnostics['best_threshold_metrics']
            ),
            'gvae_classifier_metrics_best_f1_threshold': (
                gvae_diagnostics['best_f1_threshold_metrics']
            ),
            'gvae_classifier_metrics_best_balanced_accuracy_threshold': (
                gvae_diagnostics['best_balanced_accuracy_threshold_metrics']
            ),
            'gvae_classifier_confusion_matrix_threshold_0_5': (
                gvae_diagnostics['confusion_matrix_threshold_0_5']
            ),
            'gvae_classifier_confusion_matrix_best_threshold': (
                gvae_diagnostics['confusion_matrix_best_threshold']
            ),
            'gvae_classifier_class_distribution': gvae_diagnostics['class_distribution'],
            'gvae_classifier_prediction_distribution': (
                gvae_diagnostics['prediction_distribution']
            ),
            'gvae_classifier_evaluation_artifacts': gvae_diagnostics['files'],
            'class_distribution': {
                'train': _class_distribution(train_labels.detach().cpu().numpy()),
                'val': _class_distribution(val_labels.detach().cpu().numpy()),
            },
            'evaluation_artifacts': {},
            'probs': [],
            'labels': val_labels.detach().cpu().numpy(),
        }

    val_mus_scaled = torch.tensor(
        shared_latent_scaler.transform(final_val_mus.detach().cpu().numpy()),
        dtype=torch.float32,
    ).to(device)

    fold_probs = []
    fold_loss_responder = []
    fold_loss_non_responder = []
    timesteps_to_eval = torch.linspace(
        0, ddpm_config_for_candidate['timesteps'] - 1,
        ddpm_config_for_candidate.get('eval_timesteps', 50),
        dtype=torch.long,
    ).to(device)

    with torch.no_grad():
        for i in tqdm(
            range(len(final_val_mus)),
            desc=f"F{fold_num} R{candidate['rank_by_val_auc']} DDPM Eval",
            leave=False,
        ):
            loss_resp = ddpm_responder.evaluation_loss(
                val_mus_scaled[i].unsqueeze(0), timesteps_to_eval
            )
            loss_non_resp = ddpm_non_responder.evaluation_loss(
                val_mus_scaled[i].unsqueeze(0), timesteps_to_eval
            )

            likelihood_resp = 1 / (loss_resp + 1e-9)
            likelihood_non_resp = 1 / (loss_non_resp + 1e-9)
            prob_is_responder = likelihood_resp / (likelihood_resp + likelihood_non_resp)
            fold_probs.append(prob_is_responder.item())
            fold_loss_responder.append(loss_resp.item())
            fold_loss_non_responder.append(loss_non_resp.item())

    fold_labels = val_labels.detach().cpu().numpy()
    fold_probs_np = np.array(fold_probs)
    fold_loss_responder_np = np.array(fold_loss_responder)
    fold_loss_non_responder_np = np.array(fold_loss_non_responder)
    threshold_selection_metric = train_config.get(
        'ddpm_threshold_selection_metric',
        'balanced_accuracy',
    )
    if eval_dir.exists() and any(eval_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"DDPM evaluation artifact directory already exists: {eval_dir}. "
            "Set overwrite_checkpoints=True or choose a new run_id/artifact_dir."
        )
    diagnostics = _save_ddpm_evaluation_artifacts(
        eval_dir=eval_dir,
        labels=fold_labels,
        probs=fold_probs_np,
        loss_responder=fold_loss_responder_np,
        loss_non_responder=fold_loss_non_responder_np,
        train_labels=train_labels.detach().cpu().numpy(),
        val_indices=val_indices,
        threshold_selection_metric=threshold_selection_metric,
        metadata={
            'fold': fold_num,
            'rank_by_val_auc': candidate['rank_by_val_auc'],
            'epoch': candidate['epoch'],
            'gvae_val_auc': candidate['val_auc'],
            'gvae_val_loss': candidate['val_loss'],
            'ddpm_latent_representation': latent_name,
            'ddpm_latent_dim': ddpm_config_for_candidate['latent_dim'],
            'gvae_checkpoint_path': str(candidate_checkpoint_path),
            'latents_path': str(latents_path),
        },
    )
    metrics = diagnostics['threshold_0_5_metrics']

    return {
        'fold': fold_num,
        'rank_by_val_auc': candidate['rank_by_val_auc'],
        'epoch': candidate['epoch'],
        'gvae_val_auc': candidate['val_auc'],
        'gvae_val_loss': candidate['val_loss'],
        'ddpm_latent_representation': latent_name,
        'ddpm_latent_dim': ddpm_config_for_candidate['latent_dim'],
        'gvae_checkpoint_path': str(candidate_checkpoint_path),
        'latents_path': str(latents_path),
        'eval_dir': str(eval_dir),
        'gvae_classifier_eval_dir': str(gvae_eval_dir),
        'status': 'ok',
        'metrics': metrics,
        'threshold_0_5_metrics': diagnostics['threshold_0_5_metrics'],
        'best_threshold_metrics': diagnostics['best_threshold_metrics'],
        'best_f1_threshold_metrics': diagnostics['best_f1_threshold_metrics'],
        'best_balanced_accuracy_threshold_metrics': (
            diagnostics['best_balanced_accuracy_threshold_metrics']
        ),
        'gvae_classifier_metrics_threshold_0_5': (
            gvae_diagnostics['threshold_0_5_metrics']
        ),
        'gvae_classifier_metrics_best_threshold': (
            gvae_diagnostics['best_threshold_metrics']
        ),
        'gvae_classifier_metrics_best_f1_threshold': (
            gvae_diagnostics['best_f1_threshold_metrics']
        ),
        'gvae_classifier_metrics_best_balanced_accuracy_threshold': (
            gvae_diagnostics['best_balanced_accuracy_threshold_metrics']
        ),
        'gvae_classifier_confusion_matrix_threshold_0_5': (
            gvae_diagnostics['confusion_matrix_threshold_0_5']
        ),
        'gvae_classifier_confusion_matrix_best_threshold': (
            gvae_diagnostics['confusion_matrix_best_threshold']
        ),
        'gvae_classifier_class_distribution': gvae_diagnostics['class_distribution'],
        'gvae_classifier_prediction_distribution': (
            gvae_diagnostics['prediction_distribution']
        ),
        'gvae_classifier_evaluation_artifacts': gvae_diagnostics['files'],
        'class_distribution': diagnostics['class_distribution'],
        'prediction_distribution': diagnostics['prediction_distribution'],
        'confusion_matrix_threshold_0_5': diagnostics['confusion_matrix_threshold_0_5'],
        'confusion_matrix_best_threshold': diagnostics['confusion_matrix_best_threshold'],
        'pr_auc_baseline_positive_prevalence': diagnostics[
            'pr_auc_baseline_positive_prevalence'
        ],
        'normalized_average_precision': diagnostics['normalized_average_precision'],
        'evaluation_artifacts': diagnostics['files'],
        'probs': fold_probs,
        'labels': fold_labels,
        'loss_responder': fold_loss_responder,
        'loss_non_responder': fold_loss_non_responder,
    }




def kfold_gvae_ddpm_generative_classifier(
    full_data: torch.utils.data.Dataset,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    ddpm_config: Dict[str, Any],
    pca_config: Dict[str, int]
) -> Dict[str, float]:
    if not train_config.get('allow_deprecated_ddpm_classifier', False):
        raise RuntimeError(
            "kfold_gvae_ddpm_generative_classifier is disabled because it "
            "reports DDPM inverse-loss classification metrics. DDPM must be "
            "used only as p(concat_mu | class) latent-space augmentation. "
            "Use training.latent_ddpm_augmentation.run_conditional_latent_augmentation_pipeline."
        )
    device = train_config['device']
    print(f"Using device: {device}")

    # # === Step 0: Preprocess data with PCA on CPU ===
    # data_cpu = full_data.cpu()
    # data_pca = preprocess_data_with_pca(data_cpu, pca_config)
        
    # Tạo một bản sao của dữ liệu trên CPU để thực hiện PCA
    full_data_cpu = full_data.clone().cpu()

    # # Cập nhật lại model_config với kích thước mới sau PCA
    # model_config['view_configs']['clinical']['in_channels'] = pca_config['clinical']
    # model_config['view_configs']['pathology']['in_channels'] = pca_config['pathology']
    # model_config['radiology_aggregator_config']['lesion_feature_dim'] = pca_config['radiology_lesion']
    
    y_for_stratification = full_data_cpu['patient'].binary_label.cpu().numpy()
    kf = StratifiedKFold(
        n_splits=train_config['n_splits'],
        shuffle=True,
        random_state=train_config.get('random_seed', 42)
    )
    
    all_true_labels, all_pred_probs = [], []
    fold_aucs, fold_pr_aucs, fold_f1s = [], [], []
    fold_accuracies, fold_balanced_accuracies = [], []
    fold_precisions, fold_recalls, fold_brier_scores = [], [], []
    fold_sensitivities, fold_specificities = [], []
    fold_f1s_best, fold_accuracies_best, fold_balanced_accuracies_best = [], [], []
    fold_precisions_best, fold_recalls_best = [], []
    fold_sensitivities_best, fold_specificities_best = [], []
    selected_candidate_results = []
    all_candidate_results = []
    selection_metric = train_config.get('ddpm_selection_metric', 'auc')

    for fold, (train_idx_np, val_idx_np) in enumerate(kf.split(np.arange(full_data_cpu['patient'].num_nodes), y_for_stratification)):
        fold_num = fold + 1
        print(f"\n{'='*20} FOLD {fold_num}/{train_config['n_splits']} {'='*20}")
        train_indices = torch.from_numpy(train_idx_np).to(device)
        val_indices = torch.from_numpy(val_idx_np).to(device)

        # === Step 1: Train GVAE and keep top-k checkpoints by GVAE validation AUC ===
        print(f"--- [Fold {fold_num}] Training GVAE model and collecting top candidates ---")
        gvae_train_config = copy.deepcopy(train_config)
        gvae_train_config['return_top_k_checkpoints'] = True
        gvae_train_config['top_k_gvae_checkpoints'] = train_config.get('top_k_gvae_checkpoints', 5)
        top_gvae_candidates = train_gvae_single_fold(
            full_data, train_indices, val_indices, model_config, gvae_train_config, fold_num
        )

        if not top_gvae_candidates:
            print(f"WARNING: GVAE training produced no top-k candidates for fold {fold_num}. Skipping.")
            continue

        artifact_dir = Path(train_config.get('artifact_dir', 'artifacts/gvae_ddpm_selection'))
        run_id = train_config.get('run_id', f"seed_{train_config.get('random_seed', 42)}")
        overwrite = train_config.get('overwrite_checkpoints', False)
        fold_artifact_dir = artifact_dir / run_id / f"fold_{fold_num}"
        checkpoint_dir = fold_artifact_dir / "checkpoints"
        gvae_eval_dir = fold_artifact_dir / "gvae_eval"
        ddpm_eval_dir = fold_artifact_dir / "ddpm_eval"
        latents_dir = (
            Path(train_config.get('latent_output_dir', 'outputs/latent_for_ddpm'))
            / run_id
            / f"fold_{fold_num}"
        )

        candidate_results = []
        for candidate in top_gvae_candidates:
            rank = candidate['rank_by_val_auc']
            print(
                f"--- [Fold {fold_num}] Candidate rank {rank}: "
                f"GVAE val AUC={candidate['val_auc']:.4f}, epoch={candidate['epoch']} ---"
            )

            checkpoint_path = checkpoint_dir / f"rank_{rank}_epoch_{candidate['epoch']}_gvae.pt"
            checkpoint_path = _save_gvae_candidate_checkpoint(
                candidate,
                model_config,
                train_config,
                fold_num,
                train_idx_np,
                val_idx_np,
                checkpoint_path,
                overwrite,
            )

            latents_path = latents_dir / f"rank_{rank}_epoch_{candidate['epoch']}_latents_for_ddpm.pt"
            candidate_gvae_eval_dir = gvae_eval_dir / f"rank_{rank}_epoch_{candidate['epoch']}"
            eval_dir = ddpm_eval_dir / f"rank_{rank}_epoch_{candidate['epoch']}"
            candidate_result = _evaluate_gvae_candidate_with_ddpm(
                full_data=full_data,
                model_config=model_config,
                train_config=train_config,
                ddpm_config=ddpm_config,
                candidate=candidate,
                fold_num=fold_num,
                train_indices=train_indices,
                val_indices=val_indices,
                candidate_checkpoint_path=checkpoint_path,
                latents_path=latents_path,
                gvae_eval_dir=candidate_gvae_eval_dir,
                eval_dir=eval_dir,
            )
            candidate_results.append(candidate_result)
            all_candidate_results.append({
                k: v for k, v in candidate_result.items()
                if k not in {'probs', 'labels', 'loss_responder', 'loss_non_responder'}
            })

            metrics = candidate_result['metrics']
            best_metrics = candidate_result.get('best_threshold_metrics', {})
            print(
                f"  [Fold {fold_num} Rank {rank}] "
                f"Latent={candidate_result.get('ddpm_latent_representation')} "
                f"Dim={candidate_result.get('ddpm_latent_dim')} "
                f"DDPM AUC={metrics['auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, "
                f"F1@0.5={metrics['f1']:.4f}, "
                f"BestThr={best_metrics.get('threshold', np.nan):.4f}, "
                f"BestF1={best_metrics.get('f1', np.nan):.4f}, "
                f"BestBalAcc={best_metrics.get('balanced_accuracy', np.nan):.4f}"
            )

        valid_results = [
            r for r in candidate_results
            if r['status'] == 'ok' and not np.isnan(r['metrics'].get(selection_metric, np.nan))
        ]
        if not valid_results:
            print(f"WARNING: No valid DDPM candidate results for fold {fold_num}. Skipping.")
            continue

        selected = max(valid_results, key=lambda r: r['metrics'][selection_metric])
        selected_candidate = next(
            c for c in top_gvae_candidates
            if c['rank_by_val_auc'] == selected['rank_by_val_auc']
        )

        selected_checkpoint_path = checkpoint_dir / f"selected_by_ddpm_{selection_metric}.pt"
        selected_checkpoint_path = _prepare_artifact_path(selected_checkpoint_path, overwrite)
        selected_ddpm_config = copy.deepcopy(ddpm_config)
        selected_ddpm_config['latent_dim'] = selected['ddpm_latent_dim']
        torch.save({
            'model_state_dict': selected_candidate['model_state_dict'],
            'model_config': model_config,
            'train_config': train_config,
            'ddpm_config': selected_ddpm_config,
            'fold': fold_num,
            'selection_metric': selection_metric,
            'selected_candidate': {
                k: v for k, v in selected.items()
                if k not in {'probs', 'labels', 'loss_responder', 'loss_non_responder'}
            },
            'all_candidate_results': [
                {
                    k: v for k, v in result.items()
                    if k not in {'probs', 'labels', 'loss_responder', 'loss_non_responder'}
                }
                for result in candidate_results
            ],
            'train_indices': train_idx_np,
            'val_indices': val_idx_np,
            'selection_stage': 'final_gvae_by_downstream_ddpm_validation',
        }, selected_checkpoint_path)

        selected['selected_checkpoint_path'] = str(selected_checkpoint_path)
        selected_candidate_results.append(selected)

        fold_labels = np.asarray(selected['labels'])
        fold_probs = np.asarray(selected['probs'])
        metrics = selected['threshold_0_5_metrics']
        best_metrics = selected['best_threshold_metrics']

        fold_aucs.append(metrics['auc'])
        fold_pr_aucs.append(metrics['pr_auc'])
        fold_f1s.append(metrics['f1'])
        fold_accuracies.append(metrics['accuracy'])
        fold_balanced_accuracies.append(metrics['balanced_accuracy'])
        fold_precisions.append(metrics['precision'])
        fold_recalls.append(metrics['recall'])
        fold_sensitivities.append(metrics['sensitivity'])
        fold_specificities.append(metrics['specificity'])
        fold_brier_scores.append(metrics['brier_score'])
        fold_f1s_best.append(best_metrics['f1'])
        fold_accuracies_best.append(best_metrics['accuracy'])
        fold_balanced_accuracies_best.append(best_metrics['balanced_accuracy'])
        fold_precisions_best.append(best_metrics['precision'])
        fold_recalls_best.append(best_metrics['recall'])
        fold_sensitivities_best.append(best_metrics['sensitivity'])
        fold_specificities_best.append(best_metrics['specificity'])

        print(
            f"  [Fold {fold_num}] Selected GVAE rank {selected['rank_by_val_auc']} "
            f"by DDPM {selection_metric}={metrics[selection_metric]:.4f}; "
            f"F1@0.5={metrics['f1']:.4f}, "
            f"F1@best={best_metrics['f1']:.4f}, "
            f"BalAcc@best={best_metrics['balanced_accuracy']:.4f}"
        )

        all_true_labels.extend(fold_labels)
        all_pred_probs.extend(fold_probs)

    # === Step 5: Aggregate and Report Final Metrics ===
    print(f"\n{'='*20} FINAL PIPELINE RESULTS {'='*20}")
    if not all_true_labels:
        print("No results to report.")
        return {}

    final_selected = None
    if selected_candidate_results:
        final_selected = max(
            selected_candidate_results,
            key=lambda r: r['metrics'][selection_metric],
        )

    results = {
        'ddpm_selection_metric': selection_metric,
        'final_selected_gvae_checkpoint': (
            final_selected['selected_checkpoint_path'] if final_selected else None
        ),
        'final_selected_source_checkpoint': (
            final_selected['gvae_checkpoint_path'] if final_selected else None
        ),
        'final_selected_latents_path': (
            final_selected['latents_path'] if final_selected else None
        ),
        'final_selected_fold': (
            final_selected['fold'] if final_selected else None
        ),
        'final_selected_rank_by_gvae_val_auc': (
            final_selected['rank_by_val_auc'] if final_selected else None
        ),
        'final_selected_ddpm_metrics': (
            final_selected['metrics'] if final_selected else None
        ),
        'final_selected_ddpm_metrics_threshold_0_5': (
            final_selected.get('threshold_0_5_metrics') if final_selected else None
        ),
        'final_selected_ddpm_metrics_best_threshold': (
            final_selected.get('best_threshold_metrics') if final_selected else None
        ),
        'final_selected_ddpm_metrics_best_f1_threshold': (
            final_selected.get('best_f1_threshold_metrics') if final_selected else None
        ),
        'final_selected_ddpm_metrics_best_balanced_accuracy_threshold': (
            final_selected.get('best_balanced_accuracy_threshold_metrics')
            if final_selected else None
        ),
        'final_selected_class_distribution': (
            final_selected.get('class_distribution') if final_selected else None
        ),
        'final_selected_prediction_distribution': (
            final_selected.get('prediction_distribution') if final_selected else None
        ),
        'final_selected_confusion_matrix_threshold_0_5': (
            final_selected.get('confusion_matrix_threshold_0_5') if final_selected else None
        ),
        'final_selected_confusion_matrix_best_threshold': (
            final_selected.get('confusion_matrix_best_threshold') if final_selected else None
        ),
        'final_selected_evaluation_artifacts': (
            final_selected.get('evaluation_artifacts') if final_selected else None
        ),
        'final_selected_gvae_classifier_metrics_threshold_0_5': (
            final_selected.get('gvae_classifier_metrics_threshold_0_5')
            if final_selected else None
        ),
        'final_selected_gvae_classifier_metrics_best_threshold': (
            final_selected.get('gvae_classifier_metrics_best_threshold')
            if final_selected else None
        ),
        'final_selected_gvae_classifier_class_distribution': (
            final_selected.get('gvae_classifier_class_distribution')
            if final_selected else None
        ),
        'final_selected_gvae_classifier_confusion_matrix_threshold_0_5': (
            final_selected.get('gvae_classifier_confusion_matrix_threshold_0_5')
            if final_selected else None
        ),
        'final_selected_gvae_classifier_confusion_matrix_best_threshold': (
            final_selected.get('gvae_classifier_confusion_matrix_best_threshold')
            if final_selected else None
        ),
        'final_selected_gvae_classifier_evaluation_artifacts': (
            final_selected.get('gvae_classifier_evaluation_artifacts')
            if final_selected else None
        ),
        'ddpm_latent_representation': train_config.get('ddpm_latent_representation', 'concat_mu'),
        'mean_auc': np.nanmean(fold_aucs), 'std_auc': np.nanstd(fold_aucs),
        'mean_pr_auc': np.nanmean(fold_pr_aucs), 'std_pr_auc': np.nanstd(fold_pr_aucs),
        'mean_f1': np.nanmean(fold_f1s), 'std_f1': np.nanstd(fold_f1s),
        'mean_accuracy': np.nanmean(fold_accuracies), 'std_accuracy': np.nanstd(fold_accuracies),
        'mean_balanced_accuracy': np.nanmean(fold_balanced_accuracies),
        'std_balanced_accuracy': np.nanstd(fold_balanced_accuracies),
        'mean_precision': np.nanmean(fold_precisions), 'std_precision': np.nanstd(fold_precisions),
        'mean_recall': np.nanmean(fold_recalls), 'std_recall': np.nanstd(fold_recalls),
        'mean_sensitivity': np.nanmean(fold_sensitivities),
        'std_sensitivity': np.nanstd(fold_sensitivities),
        'mean_specificity': np.nanmean(fold_specificities),
        'std_specificity': np.nanstd(fold_specificities),
        'mean_f1_best_threshold': np.nanmean(fold_f1s_best),
        'std_f1_best_threshold': np.nanstd(fold_f1s_best),
        'mean_accuracy_best_threshold': np.nanmean(fold_accuracies_best),
        'std_accuracy_best_threshold': np.nanstd(fold_accuracies_best),
        'mean_balanced_accuracy_best_threshold': np.nanmean(
            fold_balanced_accuracies_best
        ),
        'std_balanced_accuracy_best_threshold': np.nanstd(
            fold_balanced_accuracies_best
        ),
        'mean_precision_best_threshold': np.nanmean(fold_precisions_best),
        'std_precision_best_threshold': np.nanstd(fold_precisions_best),
        'mean_recall_best_threshold': np.nanmean(fold_recalls_best),
        'std_recall_best_threshold': np.nanstd(fold_recalls_best),
        'mean_sensitivity_best_threshold': np.nanmean(fold_sensitivities_best),
        'std_sensitivity_best_threshold': np.nanstd(fold_sensitivities_best),
        'mean_specificity_best_threshold': np.nanmean(fold_specificities_best),
        'std_specificity_best_threshold': np.nanstd(fold_specificities_best),
        'mean_brier_score': np.nanmean(fold_brier_scores),
        'std_brier_score': np.nanstd(fold_brier_scores),
        'selected_gvae_checkpoints': [
            {
                'fold': result.get('fold', idx + 1),
                'rank_by_val_auc': result['rank_by_val_auc'],
                'gvae_val_auc': result['gvae_val_auc'],
                'ddpm_metrics': result['metrics'],
                'ddpm_metrics_threshold_0_5': result.get('threshold_0_5_metrics'),
                'ddpm_metrics_best_threshold': result.get('best_threshold_metrics'),
                'ddpm_metrics_best_f1_threshold': result.get(
                    'best_f1_threshold_metrics'
                ),
                'ddpm_metrics_best_balanced_accuracy_threshold': result.get(
                    'best_balanced_accuracy_threshold_metrics'
                ),
                'class_distribution': result.get('class_distribution'),
                'prediction_distribution': result.get('prediction_distribution'),
                'confusion_matrix_threshold_0_5': result.get(
                    'confusion_matrix_threshold_0_5'
                ),
                'confusion_matrix_best_threshold': result.get(
                    'confusion_matrix_best_threshold'
                ),
                'evaluation_artifacts': result.get('evaluation_artifacts'),
                'gvae_classifier_metrics_threshold_0_5': result.get(
                    'gvae_classifier_metrics_threshold_0_5'
                ),
                'gvae_classifier_metrics_best_threshold': result.get(
                    'gvae_classifier_metrics_best_threshold'
                ),
                'gvae_classifier_class_distribution': result.get(
                    'gvae_classifier_class_distribution'
                ),
                'gvae_classifier_prediction_distribution': result.get(
                    'gvae_classifier_prediction_distribution'
                ),
                'gvae_classifier_confusion_matrix_threshold_0_5': result.get(
                    'gvae_classifier_confusion_matrix_threshold_0_5'
                ),
                'gvae_classifier_confusion_matrix_best_threshold': result.get(
                    'gvae_classifier_confusion_matrix_best_threshold'
                ),
                'gvae_classifier_evaluation_artifacts': result.get(
                    'gvae_classifier_evaluation_artifacts'
                ),
                'gvae_checkpoint_path': result['gvae_checkpoint_path'],
                'latents_path': result['latents_path'],
                'selected_checkpoint_path': result['selected_checkpoint_path'],
            }
            for idx, result in enumerate(selected_candidate_results)
        ],
        'all_candidate_results': all_candidate_results,
    }

    print("--- Cross-Validation Summary (PCA-GVAE + Generative DDPM Classifier) ---")
    for key, value in results.items():
        if isinstance(value, (int, float, np.floating)):
            print(f"  {key.replace('_', ' ').capitalize()}: {value:.4f}")

    return results
