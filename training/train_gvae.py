import torch
import numpy as np
import pandas as pd
import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from torch_geometric.data import HeteroData
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    silhouette_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm as tqdm_notebook
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any
from torch_geometric.loader import NeighborLoader
from models.gvae_model import GVAE, get_separate_view_latent_params
from utils.training_utils import linear_anneal
from utils.loss_utils import calculate_contrastive_loss, calculate_contrastive_loss_vectorized
from utils.classification_eval import (
    build_binary_classification_diagnostics,
    class_distribution,
    save_binary_classification_artifacts,
)
from tqdm.notebook import tqdm
import torch.optim as optim
from models.gvae_components import StandaloneRadiologyMIL
import copy
from sklearn.metrics import precision_recall_curve


def _make_epoch_results(
    val_labels_np: np.ndarray,
    val_probs_np: np.ndarray,
    val_auc: float,
    val_loss: torch.Tensor,
    epoch: int,
) -> Dict[str, Any]:
    fpr, tpr = np.array([np.nan]), np.array([np.nan])
    if val_auc != -1.0:
        fpr, tpr, _ = roc_curve(val_labels_np, val_probs_np)
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': val_auc,
        'val_loss': float(val_loss.detach().cpu()) if torch.is_tensor(val_loss) else float(val_loss),
        'epoch': epoch,
        'y_true': val_labels_np,
        'y_pred_probs': val_probs_np,
    }


def _safe_metric_value(value: Any, default: float = float('-inf')) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


@torch.no_grad()
def _extract_concat_mu_for_quality(
    model: GVAE,
    full_data: HeteroData,
    indices: torch.Tensor,
) -> torch.Tensor:
    params = get_separate_view_latent_params(model, full_data, indices)
    stacked_mu = torch.stack([params[view]['mu'] for view in model.views], dim=1)
    return stacked_mu.reshape(stacked_mu.shape[0], -1).detach().cpu()


def _binary_fisher_ratio(x: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    if x.shape[0] < 2 or classes.size != 2:
        return float('nan')
    x0 = x[y == classes[0]]
    x1 = x[y == classes[1]]
    if x0.shape[0] == 0 or x1.shape[0] == 0:
        return float('nan')
    mean_distance = np.sum((x1.mean(axis=0) - x0.mean(axis=0)) ** 2)
    within_variance = np.sum(x0.var(axis=0)) + np.sum(x1.var(axis=0))
    return float(mean_distance / (within_variance + 1e-12))


def _safe_silhouette(x: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if x.shape[0] < 3 or np.unique(y).size < 2 or np.unique(y).size >= x.shape[0]:
        return float('nan')
    try:
        return float(silhouette_score(x, y, metric='euclidean'))
    except Exception:
        return float('nan')


def _latent_variance_stability(train_x: np.ndarray, val_x: np.ndarray) -> float:
    if train_x.shape[0] < 2 or val_x.shape[0] < 2:
        return float('nan')
    train_var = np.var(train_x, axis=0)
    val_var = np.var(val_x, axis=0)
    log_ratio = np.abs(np.log((val_var + 1e-8) / (train_var + 1e-8)))
    return float(1.0 / (1.0 + np.mean(log_ratio)))


def _compute_latent_quality_metrics(
    model: GVAE,
    full_data: HeteroData,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    train_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Score concat_mu quality without fitting anything on validation data."""
    train_mu = _extract_concat_mu_for_quality(model, full_data, train_indices).numpy()
    val_mu = _extract_concat_mu_for_quality(model, full_data, val_indices).numpy()
    train_labels = (
        full_data['patient']['binary_label'].to(train_indices.device)[train_indices]
        .detach().cpu().numpy().astype(int)
    )
    val_labels = (
        full_data['patient']['binary_label'].to(val_indices.device)[val_indices]
        .detach().cpu().numpy().astype(int)
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_mu)
    val_scaled = scaler.transform(val_mu)

    linear_probe_auc = float('nan')
    linear_probe_train_auc = float('nan')
    if np.unique(train_labels).size == 2 and np.unique(val_labels).size == 2:
        try:
            probe = LogisticRegression(
                max_iter=int(train_config.get('latent_probe_max_iter', 5000)),
                class_weight=train_config.get('latent_probe_class_weight', 'balanced'),
                solver=train_config.get('latent_probe_solver', 'liblinear'),
                random_state=int(train_config.get('random_seed', 42)),
            )
            probe.fit(train_scaled, train_labels)
            val_scores = probe.predict_proba(val_scaled)[:, 1]
            train_scores = probe.predict_proba(train_scaled)[:, 1]
            linear_probe_auc = float(roc_auc_score(val_labels, val_scores))
            linear_probe_train_auc = float(roc_auc_score(train_labels, train_scores))
        except Exception:
            linear_probe_auc = float('nan')
            linear_probe_train_auc = float('nan')

    train_silhouette = _safe_silhouette(train_scaled, train_labels)
    val_silhouette = _safe_silhouette(val_scaled, val_labels)
    train_fisher = _binary_fisher_ratio(train_scaled, train_labels)
    val_fisher = _binary_fisher_ratio(val_scaled, val_labels)
    variance_stability = _latent_variance_stability(train_scaled, val_scaled)

    weights = train_config.get('latent_quality_weights', {})
    w_probe = float(weights.get('linear_probe_auc', 0.5))
    w_silhouette = float(weights.get('silhouette', 0.2))
    w_fisher = float(weights.get('fisher_ratio', 0.2))
    w_variance = float(weights.get('variance_stability', 0.1))
    weight_sum = max(w_probe + w_silhouette + w_fisher + w_variance, 1e-12)

    silhouette_component = (_safe_metric_value(val_silhouette, 0.0) + 1.0) / 2.0
    fisher_component = _safe_metric_value(val_fisher, 0.0)
    fisher_component = fisher_component / (1.0 + fisher_component)
    variance_component = _safe_metric_value(variance_stability, 0.0)
    probe_component = _safe_metric_value(linear_probe_auc, 0.0)
    quality_score = (
        w_probe * probe_component
        + w_silhouette * silhouette_component
        + w_fisher * fisher_component
        + w_variance * variance_component
    ) / weight_sum

    return {
        'latent_representation': 'concat_mu',
        'scaler_fit_split': 'train',
        'linear_probe_auc': linear_probe_auc,
        'linear_probe_train_auc': linear_probe_train_auc,
        'silhouette_train': train_silhouette,
        'silhouette_val': val_silhouette,
        'fisher_ratio_train': train_fisher,
        'fisher_ratio_val': val_fisher,
        'variance_stability': variance_stability,
        'latent_variance_train_mean': float(np.var(train_scaled, axis=0).mean()),
        'latent_variance_val_mean': float(np.var(val_scaled, axis=0).mean()),
        'latent_quality_score': float(quality_score),
        'weights': {
            'linear_probe_auc': w_probe,
            'silhouette': w_silhouette,
            'fisher_ratio': w_fisher,
            'variance_stability': w_variance,
        },
    }


def _rank_checkpoint_candidates(candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Keep checkpoint candidates ordered by the configured GVAE validation metric."""
    ranked = sorted(
        candidates,
        key=lambda c: _checkpoint_sort_key(
            c,
            c.get('checkpoint_metric', 'auc'),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def _checkpoint_sort_key(candidate: Dict[str, Any], metric: str) -> Tuple[float, ...]:
    diagnostics = candidate.get('validation_diagnostics') or {}
    metrics_0_5 = diagnostics.get('threshold_0_5_metrics') or {}
    best_metrics = diagnostics.get('best_threshold_metrics') or {}
    latent_quality = candidate.get('latent_quality_metrics') or {}
    val_auc = candidate.get('val_auc', metrics_0_5.get('auc', float('-inf')))
    val_pr_auc = candidate.get('val_pr_auc', metrics_0_5.get('pr_auc', float('-inf')))
    val_balanced_accuracy = candidate.get(
        'val_balanced_accuracy',
        best_metrics.get('balanced_accuracy', float('-inf')),
    )
    val_f1 = candidate.get('val_f1', best_metrics.get('f1', float('-inf')))
    val_loss = candidate.get('val_loss', float('inf'))
    epoch = candidate.get('epoch', 0)
    latent_quality_score = _safe_metric_value(
        latent_quality.get('latent_quality_score'),
    )
    latent_probe_auc = _safe_metric_value(
        latent_quality.get('linear_probe_auc'),
    )
    latent_silhouette = _safe_metric_value(
        latent_quality.get('silhouette_val'),
    )
    latent_fisher = _safe_metric_value(
        latent_quality.get('fisher_ratio_val'),
    )
    latent_variance_stability = _safe_metric_value(
        latent_quality.get('variance_stability'),
    )

    if metric in {'latent_quality', 'latent_quality_score'}:
        return (
            latent_quality_score,
            latent_probe_auc,
            latent_silhouette,
            latent_fisher,
            latent_variance_stability,
            val_auc,
            val_pr_auc,
            -val_loss,
            epoch,
        )
    if metric in {'latent_linear_probe_auc', 'linear_probe_auc'}:
        return (
            latent_probe_auc,
            latent_quality_score,
            latent_silhouette,
            latent_fisher,
            latent_variance_stability,
            val_auc,
            -val_loss,
            epoch,
        )

    if metric in {'auc', 'roc_auc'}:
        return (val_auc, val_pr_auc, val_balanced_accuracy, val_f1, -val_loss, epoch)
    if metric == 'pr_auc':
        return (val_pr_auc, val_auc, val_balanced_accuracy, val_f1, -val_loss, epoch)
    if metric == 'balanced_accuracy':
        return (val_balanced_accuracy, val_auc, val_pr_auc, val_f1, -val_loss, epoch)
    if metric == 'f1':
        return (val_f1, val_auc, val_pr_auc, val_balanced_accuracy, -val_loss, epoch)
    if metric == 'loss':
        return (-val_loss, val_auc, val_pr_auc, val_balanced_accuracy, val_f1, epoch)
    if metric in {'auc_pr_balanced_accuracy', 'auc_pr_balanced'}:
        return (val_auc, val_pr_auc, val_balanced_accuracy, val_f1, -val_loss, epoch)
    raise ValueError(f"Unknown checkpoint metric: {metric}")


def _compute_main_pos_weight(
    train_labels_np: np.ndarray,
    train_config: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    n_negative = int(np.sum(train_labels_np == 0))
    n_positive = int(np.sum(train_labels_np == 1))
    use_balanced_batches = bool(train_config.get('balanced_batch_sampling', False))
    strategy = train_config.get('pos_weight_strategy', 'auto')
    resolved_strategy = strategy

    if strategy == 'auto':
        resolved_strategy = 'none' if use_balanced_batches else 'train_distribution'

    if resolved_strategy in {'none', 'off', False}:
        pos_weight = 1.0
    elif resolved_strategy in {'train_distribution', 'balanced'}:
        pos_weight = n_negative / (n_positive + 1e-6)
    else:
        raise ValueError(
            "Unknown pos_weight_strategy. Use one of "
            "{'auto', 'train_distribution', 'balanced', 'none'}."
        )

    metadata = {
        'pos_weight_strategy': strategy,
        'resolved_pos_weight_strategy': resolved_strategy,
        'pos_weight_formula': (
            'n_negative / n_positive'
            if resolved_strategy in {'train_distribution', 'balanced'}
            else '1.0'
        ),
        'pos_weight': float(pos_weight),
        'train_negative': n_negative,
        'train_positive': n_positive,
        'balanced_batch_sampling': use_balanced_batches,
    }
    return float(pos_weight), metadata


def _make_train_batches(
    train_indices: torch.Tensor,
    full_labels: torch.Tensor,
    train_config: Dict[str, Any],
    epoch: int,
    fold_num: int,
) -> List[torch.Tensor]:
    batch_size = train_config.get('batch_size', None)
    if batch_size is None or batch_size <= 0 or batch_size >= len(train_indices):
        return [train_indices]

    if not train_config.get('balanced_batch_sampling', False):
        return list(torch.split(train_indices, batch_size))

    labels = full_labels[train_indices].detach()
    pos_indices = train_indices[labels == 1]
    neg_indices = train_indices[labels == 0]
    if pos_indices.numel() == 0 or neg_indices.numel() == 0:
        return list(torch.split(train_indices, batch_size))

    device = train_indices.device
    half_batch = max(1, batch_size // 2)
    other_half = max(1, batch_size - half_batch)
    n_batches = int(np.ceil(len(train_indices) / batch_size))
    generator = torch.Generator(device=device)
    generator.manual_seed(
        int(train_config.get('random_seed', 42)) + fold_num * 10000 + epoch
    )

    batches = []
    for _ in range(n_batches):
        pos_sample = pos_indices[
            torch.randint(pos_indices.numel(), (half_batch,), device=device, generator=generator)
        ]
        neg_sample = neg_indices[
            torch.randint(neg_indices.numel(), (other_half,), device=device, generator=generator)
        ]
        batch = torch.cat([pos_sample, neg_sample])
        perm = torch.randperm(batch.numel(), device=device, generator=generator)
        batches.append(batch[perm])
    return batches


def _mean_component_records(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0].keys()
    }


def _classification_eval_root(train_config: Dict[str, Any]) -> Path:
    if train_config.get('classification_eval_dir'):
        return Path(train_config['classification_eval_dir'])
    if train_config.get('metrics_dir'):
        return Path(train_config['metrics_dir']) / 'classification_eval'
    return Path(train_config.get('checkpoint_dir', 'checkpoints/gvae')) / 'classification_eval'


def _patient_ids_for_indices(data: HeteroData, indices: np.ndarray) -> Optional[List[Any]]:
    if 'main_index' not in data['patient']:
        return None
    main_index = data['patient']['main_index']
    return [main_index[int(idx)] for idx in indices]


def _compact_classification_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'threshold_0_5_metrics': diagnostics['threshold_0_5_metrics'],
        'best_threshold_metrics': diagnostics['best_threshold_metrics'],
        'best_f1_threshold_metrics': diagnostics['best_f1_threshold_metrics'],
        'best_balanced_accuracy_threshold_metrics': (
            diagnostics['best_balanced_accuracy_threshold_metrics']
        ),
        'confusion_matrix_threshold_0_5': diagnostics['confusion_matrix_threshold_0_5'],
        'confusion_matrix_best_threshold': diagnostics['confusion_matrix_best_threshold'],
        'class_distribution': diagnostics['class_distribution'],
        'prediction_distribution': diagnostics['prediction_distribution'],
        'pr_auc_baseline_positive_prevalence': (
            diagnostics['pr_auc_baseline_positive_prevalence']
        ),
        'normalized_average_precision': diagnostics['normalized_average_precision'],
        'files': diagnostics.get('files', {}),
    }


def _flat_classification_record(
    diagnostics: Dict[str, Any],
    *,
    best_epoch: Optional[int],
    best_val_loss: Optional[float],
    checkpoint_metric: str,
) -> Dict[str, Any]:
    m05 = diagnostics['threshold_0_5_metrics']
    mbest = diagnostics['best_threshold_metrics']
    mf1 = diagnostics['best_f1_threshold_metrics']
    mbal = diagnostics['best_balanced_accuracy_threshold_metrics']
    val_dist = diagnostics['class_distribution']['val']
    train_dist = diagnostics['class_distribution'].get('train', {})
    files = diagnostics.get('files', {})
    return {
        'auc': m05['auc'],
        'roc_auc': m05['roc_auc'],
        'pr_auc': m05['pr_auc'],
        'f1': mbest['f1'],
        'accuracy': mbest['accuracy'],
        'balanced_accuracy': mbest['balanced_accuracy'],
        'precision': mbest['precision'],
        'recall': mbest['recall'],
        'sensitivity': mbest['sensitivity'],
        'specificity': mbest['specificity'],
        'brier_score': m05['brier_score'],
        'f1_threshold_0_5': m05['f1'],
        'accuracy_threshold_0_5': m05['accuracy'],
        'balanced_accuracy_threshold_0_5': m05['balanced_accuracy'],
        'precision_threshold_0_5': m05['precision'],
        'recall_threshold_0_5': m05['recall'],
        'sensitivity_threshold_0_5': m05['sensitivity'],
        'specificity_threshold_0_5': m05['specificity'],
        'f1_best_threshold': mbest['f1'],
        'accuracy_best_threshold': mbest['accuracy'],
        'balanced_accuracy_best_threshold': mbest['balanced_accuracy'],
        'precision_best_threshold': mbest['precision'],
        'recall_best_threshold': mbest['recall'],
        'sensitivity_best_threshold': mbest['sensitivity'],
        'specificity_best_threshold': mbest['specificity'],
        'best_threshold': mbest['threshold'],
        'best_threshold_selection_metric': mbest['threshold_selection_metric'],
        'best_f1_threshold': mf1['threshold'],
        'best_f1_at_best_f1_threshold': mf1['f1'],
        'best_balanced_accuracy_threshold': mbal['threshold'],
        'best_balanced_accuracy_at_best_balanced_accuracy_threshold': (
            mbal['balanced_accuracy']
        ),
        'train_n': train_dist.get('n'),
        'train_positive': train_dist.get('positive'),
        'train_negative': train_dist.get('negative'),
        'train_positive_rate': train_dist.get('positive_rate'),
        'val_n': val_dist.get('n'),
        'val_positive': val_dist.get('positive'),
        'val_negative': val_dist.get('negative'),
        'val_positive_rate': val_dist.get('positive_rate'),
        'best_epoch': best_epoch,
        'best_val_loss_at_selected_epoch': best_val_loss,
        'checkpoint_metric': checkpoint_metric,
        'metrics_json': files.get('metrics_json'),
        'predicted_probabilities_csv': files.get('predicted_probabilities_csv'),
        'confusion_matrix_threshold_0_5_json': files.get(
            'confusion_matrix_threshold_0_5_json'
        ),
        'confusion_matrix_best_threshold_json': files.get(
            'confusion_matrix_best_threshold_json'
        ),
        'roc_curve_csv': files.get('roc_curve_csv'),
        'pr_curve_csv': files.get('pr_curve_csv'),
    }


def kfold_train_gvae(
    full_multi_view_data: HeteroData,
    model_config_original: Dict,
    train_config: Dict,
) -> Tuple[Dict[str, float], pd.DataFrame, List[Dict]]:
    """
    K-Fold GVAE training with MIXED RECONSTRUCTION LOSS and PRE-TRAINED RADIOLOGY AGGREGATOR.
    Includes fix for Contrastive Loss masking.
    """
    device = train_config['device']
    full_multi_view_data_cpu = full_multi_view_data.clone().cpu()

    # --- Define Feature Indices for Clinical View ---
    clinical_cont_idx = torch.arange(0, 5, device=device)
    clinical_bin_idx = torch.arange(5, 22, device=device) 
    
    criterion_mse = nn.MSELoss()
    
    # --- Config Unpacking ---
    loss_weights_config = train_config['loss_weights']
    anneal_config = train_config.get('annealing', {})
    class_loss_multiplier = train_config.get('classification_loss_multiplier', 1.0)
    base_w_class = loss_weights_config['class'] * class_loss_multiplier
    base_w_cross_cl = loss_weights_config.get('cross_cl', 0.0)
    base_w_kl = loss_weights_config['kl']
    w_rec_attr_config = loss_weights_config['rec_attr']
    w_rec_struct_config = loss_weights_config['rec_struct']
    
    kl_params = anneal_config.get('kl', {})
    cl_params = anneal_config.get('cross_cl', {})
    kl_start_w, kl_end_e = kl_params.get('start_weight', base_w_kl), kl_params.get('end_epoch', 0)
    cl_start_w, cl_end_e = cl_params.get('start_weight', base_w_cross_cl), cl_params.get('end_epoch', 0)

    use_vec_cl = train_config.get('vectorized_contrastive', False)
    contrastive_fn = (calculate_contrastive_loss_vectorized
                      if use_vec_cl else calculate_contrastive_loss)

    # --- Data Splitting ---
    all_indices_np = np.arange(full_multi_view_data['patient'].num_nodes)
    y_for_stratification = full_multi_view_data['patient'].binary_label.cpu().numpy()
    kf = StratifiedKFold(n_splits=train_config['n_splits'], shuffle=True, random_state=train_config.get('random_seed', 4200))

    fold_metrics_list = []
    all_roc_data = []

    for fold, (train_global_idx_np, val_global_idx_np) in enumerate(kf.split(all_indices_np, y_for_stratification)):
        print(f"\n===== Fold {fold+1}/{train_config['n_splits']} =====")
        
        model_config = copy.deepcopy(model_config_original)
        fold_data = full_multi_view_data_cpu.to(device)
        
        train_indices = torch.tensor(train_global_idx_np, device=device)
        val_indices = torch.tensor(val_global_idx_np, device=device)

        # Fit loss weights on this fold's training patients only to avoid
        # leaking validation label/feature distributions into optimization.
        train_labels_np = y_for_stratification[train_global_idx_np]
        pos_weight_value, pos_weight_metadata = _compute_main_pos_weight(
            train_labels_np,
            train_config,
        )
        print(
            f"INFO: Fold {fold+1} train-only pos_weight for Main Task BCE loss: "
            f"{pos_weight_value:.2f} "
            f"(negative={pos_weight_metadata['train_negative']}, "
            f"positive={pos_weight_metadata['train_positive']}, "
            f"strategy={pos_weight_metadata['resolved_pos_weight_strategy']})"
        )
        criterion_main_bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], device=device)
        )

        clinical_bin_feats_train = fold_data['patient'].x_clinical[train_indices][:, clinical_bin_idx]
        n_pos = clinical_bin_feats_train.sum(dim=0)
        n_neg = clinical_bin_feats_train.shape[0] - n_pos
        clinical_bin_pos_weight = n_neg / (n_pos + 1e-6)

        # -------------------------------------------------------
        # STEP 1: PRE-TRAIN RADIOLOGY AGGREGATOR
        # -------------------------------------------------------
        radiology_state_dict = None
        if 'radiology' in model_config['view_configs']:
             radiology_state_dict = pretrain_radiology_aggregator(
                 fold_data, 
                 train_indices, 
                 model_config['radiology_aggregator_config'], 
                 device,
                 epochs=train_config.get('pretrain_epochs', 400),
                 use_pos_weight=train_config.get('pretrain_use_pos_weight', False),
                 pretrain_val_split=train_config.get('pretrain_val_split', 0.0),
                 patience=train_config.get('pretrain_patience', 30),
                 seed=train_config.get('pretrain_seed', 42)
             )

        # -------------------------------------------------------
        # STEP 2: INITIALIZE GVAE & LOAD PRE-TRAINED WEIGHTS
        # -------------------------------------------------------
        model = GVAE(**model_config).to(device)
        
        if radiology_state_dict is not None:
            print("   [Main Loop] Loading pre-trained radiology aggregator weights...")
            model.radiology_lesion_aggregator.load_state_dict(radiology_state_dict)

        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)

        best_val_loss = float('inf')
        best_val_auc = -1.0
        epochs_no_improve = 0
        best_model_state = None
        best_epoch_results = {}
        checkpoint_metric = train_config.get('checkpoint_metric', 'latent_quality')
        early_stopping_metric = train_config.get('early_stopping_metric', checkpoint_metric)
        valid_checkpoint_metrics = {
            'auc',
            'roc_auc',
            'pr_auc',
            'balanced_accuracy',
            'f1',
            'loss',
            'auc_pr_balanced_accuracy',
            'auc_pr_balanced',
            'latent_quality',
            'latent_quality_score',
            'latent_linear_probe_auc',
            'linear_probe_auc',
        }
        if checkpoint_metric not in valid_checkpoint_metrics:
            raise ValueError(f"Unknown checkpoint_metric: {checkpoint_metric}")
        if early_stopping_metric not in valid_checkpoint_metrics:
            raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}")
        top_k = int(train_config.get('top_k_gvae_checkpoints', 0))
        top_checkpoint_candidates: List[Dict[str, Any]] = []
        best_checkpoint_key = None
        best_checkpoint_candidate = None
        best_early_stopping_key = None

        for epoch in range(1, train_config['epochs'] + 1):
            # =================== TRAINING PHASE ===================
            model.train()
            w_kl = linear_anneal(epoch, 0, kl_end_e, kl_start_w, base_w_kl)
            w_cl = linear_anneal(epoch, 0, cl_end_e, cl_start_w, base_w_cross_cl)
            
            train_batches = _make_train_batches(
                train_indices,
                fold_data['patient'].binary_label.to(device),
                train_config,
                epoch,
                fold + 1,
            )
            batch_train_losses = []
            batch_component_records = []
            
            for batch_idx in train_batches:
                logits, vae_out, cl_out, _ = model(fold_data, batch_idx)
                labels = fold_data['patient'].binary_label[batch_idx]
                
                # 1. Main Classification Loss
                loss_class = criterion_main_bce(logits.squeeze(-1), labels.float())
                
                # 2. Contrastive Loss
                if len(cl_out) > 0:
                    loss_cl = contrastive_fn(cl_out, train_config['cross_cl_temp'])
                else:
                    loss_cl = torch.tensor(0.0, device=device)
                
                # 3. Reconstruction Losses (Mixed Strategy)
                rec_attr, rec_struct, kl_div, active_views_train = 0.0, 0.0, 0.0, 0
                
                for view, vo in vae_out.items():
                    if vo and vo.get('mu') is not None:
                        active_views_train += 1
                        
                        target_x = vo['original_x_subset']
                        recon_x = vo['rec_x']
                        
                        view_weight = w_rec_attr_config.get(view, 1.0)
                        
                        if view == 'clinical':
                            loss_cont = criterion_mse(recon_x[:, clinical_cont_idx], target_x[:, clinical_cont_idx])
                            loss_bin = F.binary_cross_entropy_with_logits(
                                recon_x[:, clinical_bin_idx], 
                                target_x[:, clinical_bin_idx],
                                pos_weight=clinical_bin_pos_weight
                            )
                            rec_attr += view_weight * (loss_cont + loss_bin)
                        else:
                            rec_attr += view_weight * criterion_mse(recon_x, target_x)

                        # Structure Loss
                        rec_struct += w_rec_struct_config * F.binary_cross_entropy_with_logits(
                             vo['rec_adj_logits'].flatten(), 
                             vo['original_adj_subset'].flatten()
                        )
                        
                        # KL Divergence
                        kl_term = 1 + vo['logvar'] - vo['mu'].pow(2) - vo['logvar'].exp()
                        kl_div += -0.5 * kl_term.sum(dim=1).mean()

                avg_rec_attr = rec_attr / active_views_train if active_views_train > 0 else 0.0
                avg_rec_struct = rec_struct / active_views_train if active_views_train > 0 else 0.0
                avg_kl = kl_div / active_views_train if active_views_train > 0 else 0.0
                
                total_train_loss = (base_w_class * loss_class + w_cl * loss_cl + avg_rec_attr + avg_rec_struct + w_kl * avg_kl)

                # Skip the step on a non-finite loss instead of corrupting all params via backward.
                if not torch.isfinite(total_train_loss):
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                total_train_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.get('grad_clip_norm', 1.0))
                optimizer.step()

                batch_train_losses.append(total_train_loss.item())
                batch_component_records.append({
                    'class': float(loss_class.detach().cpu()),
                    'class_weighted': float((base_w_class * loss_class).detach().cpu()),
                    'contrastive': float(loss_cl.detach().cpu()),
                    'contrastive_weighted': float((w_cl * loss_cl).detach().cpu()),
                    'rec_attr': float(avg_rec_attr.detach().cpu()),
                    'rec_struct': float(avg_rec_struct.detach().cpu()),
                    'kl': float(avg_kl.detach().cpu()),
                    'kl_weighted': float((w_kl * avg_kl).detach().cpu()),
                    'total': float(total_train_loss.detach().cpu()),
                })

            # Guard against an all-skipped epoch (np.mean([]) -> nan + warning).
            if batch_train_losses:
                total_train_loss = torch.tensor(np.mean(batch_train_losses), device=device)
            else:
                total_train_loss = torch.tensor(float('inf'), device=device)
            epoch_loss_components = _mean_component_records(batch_component_records)

            # =================== VALIDATION PHASE ===================
            model.eval()
            with torch.no_grad():
                val_logits, val_vae_out, val_cl_out, _ = model(fold_data, val_indices)
                val_labels = fold_data['patient'].binary_label[val_indices]
                
                # 1. Classification Loss
                val_loss_class = criterion_main_bce(val_logits.squeeze(-1), val_labels.float())
                
                # 2. Contrastive Loss
                if len(val_cl_out) > 0:
                    val_loss_cl = contrastive_fn(val_cl_out, train_config['cross_cl_temp'])
                else:
                    val_loss_cl = torch.tensor(0.0, device=device)

                val_rec_attr, val_rec_struct, val_kl_div, val_active_views = 0.0, 0.0, 0.0, 0
                
                for view, vo in val_vae_out.items():
                    if vo and vo.get('mu') is not None:
                        val_active_views += 1
                        
                        target_x_val = vo['original_x_subset']
                        recon_x_val = vo['rec_x']
                        view_weight = w_rec_attr_config.get(view, 1.0)

                        if view == 'clinical':
                            loss_cont_val = criterion_mse(recon_x_val[:, clinical_cont_idx], target_x_val[:, clinical_cont_idx])
                            loss_bin_val = F.binary_cross_entropy_with_logits(
                                recon_x_val[:, clinical_bin_idx], 
                                target_x_val[:, clinical_bin_idx],
                                pos_weight=clinical_bin_pos_weight
                            )
                            val_rec_attr += view_weight * (loss_cont_val + loss_bin_val)
                        else:
                            val_rec_attr += view_weight * criterion_mse(recon_x_val, target_x_val)

                        val_rec_struct += w_rec_struct_config * F.binary_cross_entropy_with_logits(
                            vo['rec_adj_logits'].flatten(), 
                            vo['original_adj_subset'].flatten()
                        )
                        
                        kl_term_val = 1 + vo['logvar'] - vo['mu'].pow(2) - vo['logvar'].exp()
                        val_kl_div += -0.5 * kl_term_val.sum(dim=1).mean()

                val_avg_rec_attr = val_rec_attr / val_active_views if val_active_views > 0 else 0.0
                val_avg_rec_struct = val_rec_struct / val_active_views if val_active_views > 0 else 0.0
                val_avg_kl = val_kl_div / val_active_views if val_active_views > 0 else 0.0
                
                # Calculate Total Val Loss
                total_val_loss = (base_w_class * val_loss_class + base_w_cross_cl * val_loss_cl + val_avg_rec_attr + val_avg_rec_struct + base_w_kl * val_avg_kl)
                
                current_val_auc = -1.0
                val_probs_np = torch.sigmoid(val_logits.squeeze(-1)).cpu().numpy()
                val_labels_np = val_labels.cpu().numpy()
                if len(np.unique(val_labels_np)) > 1:
                    current_val_auc = roc_auc_score(val_labels_np, val_probs_np)
            
            # --- Checkpointing & Early Stopping ---
            scheduler.step(total_val_loss)

            improved_loss = total_val_loss < best_val_loss if torch.isfinite(total_val_loss) else False
            if improved_loss:
                best_val_loss = total_val_loss

            current_candidate = None
            if current_val_auc != -1.0 and torch.isfinite(total_val_loss):
                threshold_selection_metric = train_config.get(
                    'classification_threshold_selection_metric',
                    'balanced_accuracy',
                )
                candidate_diagnostics = build_binary_classification_diagnostics(
                    val_labels_np,
                    val_probs_np,
                    train_labels=train_labels_np,
                    threshold_selection_metric=threshold_selection_metric,
                    metadata={
                        'model': 'GVAE',
                        'fold': fold + 1,
                        'epoch': epoch,
                        'score_source': 'sigmoid(classifier_logit)',
                        'bce_loss': 'BCEWithLogitsLoss',
                        **pos_weight_metadata,
                        'classification_loss_weight': float(base_w_class),
                        'classification_loss_multiplier': float(class_loss_multiplier),
                        'loss_components': epoch_loss_components,
                    },
                    probability_note=(
                        'GVAE classifier probability is sigmoid(logit). '
                        'The model is trained with BCEWithLogitsLoss, so sigmoid is '
                        'applied only for evaluation.'
                    ),
                )
                latent_quality_metrics = {}
                should_compute_latent_quality = bool(
                    train_config.get(
                        'compute_latent_quality_metrics',
                        checkpoint_metric in {
                            'latent_quality',
                            'latent_quality_score',
                            'latent_linear_probe_auc',
                            'linear_probe_auc',
                        } or top_k > 0,
                    )
                )
                if should_compute_latent_quality:
                    latent_quality_metrics = _compute_latent_quality_metrics(
                        model,
                        fold_data,
                        train_indices,
                        val_indices,
                        train_config,
                    )

                current_candidate = {
                    'epoch': epoch,
                    'val_auc': float(current_val_auc),
                    'val_pr_auc': float(
                        candidate_diagnostics['threshold_0_5_metrics']['pr_auc']
                    ),
                    'val_balanced_accuracy': float(
                        candidate_diagnostics['best_threshold_metrics'][
                            'balanced_accuracy'
                        ]
                    ),
                    'val_f1': float(candidate_diagnostics['best_threshold_metrics']['f1']),
                    'val_loss': float(total_val_loss.detach().cpu()),
                    'checkpoint_metric': checkpoint_metric,
                    'validation_diagnostics': _compact_classification_diagnostics(
                        candidate_diagnostics
                    ),
                    'latent_quality_metrics': latent_quality_metrics,
                    'model_state_dict': {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                }
            if top_k > 0 and current_candidate is not None:
                top_checkpoint_candidates.append(current_candidate)
                top_checkpoint_candidates = _rank_checkpoint_candidates(
                    top_checkpoint_candidates, top_k
                )

            improved_auc = current_val_auc > best_val_auc
            improved_checkpoint = False
            if current_candidate is not None:
                current_checkpoint_key = _checkpoint_sort_key(
                    current_candidate,
                    checkpoint_metric,
                )
                improved_checkpoint = (
                    best_checkpoint_key is None
                    or current_checkpoint_key > best_checkpoint_key
                )
            if improved_checkpoint:
                best_checkpoint_key = current_checkpoint_key
                best_checkpoint_candidate = current_candidate
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch_results = _make_epoch_results(
                    val_labels_np,
                    val_probs_np,
                    current_val_auc,
                    total_val_loss,
                    epoch,
                )

            if improved_auc:
                best_val_auc = current_val_auc

            improved_early_stopping = False
            if current_candidate is not None:
                current_early_stopping_key = _checkpoint_sort_key(
                    current_candidate,
                    early_stopping_metric,
                )
                improved_early_stopping = (
                    best_early_stopping_key is None
                    or current_early_stopping_key > best_early_stopping_key
                )
            if improved_early_stopping:
                best_early_stopping_key = current_early_stopping_key
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= train_config.get('patience_early_stopping', 30):
                print(f"  Early stopping at epoch {epoch}.")
                break
            
            if epoch % 10 == 0:
                 print(
                     f"  Epoch {epoch:03d} | TLoss: {total_train_loss.item():.4f} "
                     f"| VLoss: {total_val_loss.item():.4f} "
                     f"| Class:{epoch_loss_components.get('class', np.nan):.4f} "
                     f"(w:{epoch_loss_components.get('class_weighted', np.nan):.4f}) "
                     f"| RecA:{epoch_loss_components.get('rec_attr', np.nan):.4f} "
                     f"| RecS:{epoch_loss_components.get('rec_struct', np.nan):.4f} "
                     f"| KL:{epoch_loss_components.get('kl', np.nan):.4f} "
                     f"| VAUC: {current_val_auc:.4f} | BestAUC: {best_val_auc:.4f} "
                     f"| CkptMetric:{checkpoint_metric}"
                 )
        
        # --- Save Results ---
        if best_epoch_results:
            all_roc_data.append(best_epoch_results)
            y_true = best_epoch_results['y_true']
            y_pred_probs = best_epoch_results['y_pred_probs']

            threshold_selection_metric = train_config.get(
                'classification_threshold_selection_metric',
                'balanced_accuracy',
            )
            diagnostics = build_binary_classification_diagnostics(
                y_true,
                y_pred_probs,
                train_labels=train_labels_np,
                threshold_selection_metric=threshold_selection_metric,
                metadata={
                    'model': 'GVAE',
                    'fold': fold + 1,
                    'epoch': best_epoch_results.get('epoch'),
                    'checkpoint_metric': checkpoint_metric,
                    'score_source': 'sigmoid(classifier_logit)',
                    'bce_loss': 'BCEWithLogitsLoss',
                    'sigmoid_applied_during_training': False,
                    'sigmoid_applied_for_evaluation': True,
                    **pos_weight_metadata,
                    'classification_loss_weight': float(base_w_class),
                    'classification_loss_multiplier': float(class_loss_multiplier),
                    'stratified_split': True,
                },
                probability_note=(
                    'GVAE classifier probability is sigmoid(logit). '
                    'The classifier head outputs logits and training uses '
                    'BCEWithLogitsLoss.'
                ),
            )
            eval_dir = _classification_eval_root(train_config) / f"fold_{fold+1}"
            files = save_binary_classification_artifacts(
                eval_dir,
                diagnostics,
                y_true,
                y_pred_probs,
                indices=val_global_idx_np,
                patient_ids=_patient_ids_for_indices(
                    full_multi_view_data_cpu,
                    val_global_idx_np,
                ),
                score_column='gvae_probability',
            )
            diagnostics['files'] = files

            fold_record = _flat_classification_record(
                diagnostics,
                best_epoch=best_epoch_results.get('epoch'),
                best_val_loss=best_epoch_results.get('val_loss'),
                checkpoint_metric=checkpoint_metric,
            )
            fold_metrics_list.append(fold_record)
            
            if train_config.get('save_best_fold_model', True):
                checkpoint_dir = Path(train_config.get('checkpoint_dir', 'checkpoints/gvae'))
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                run_id = train_config.get('run_id', f"seed_{train_config.get('random_seed', 4200)}")
                checkpoint_path = checkpoint_dir / f"{run_id}_fold_{fold+1}_best.pt"
                if checkpoint_path.exists() and not train_config.get('overwrite_checkpoints', False):
                    raise FileExistsError(
                        f"Checkpoint already exists: {checkpoint_path}. "
                        "Set overwrite_checkpoints=True or choose a new run_id/checkpoint_dir."
                    )
                torch.save({
                    'model_state_dict': best_model_state,
                    'model_config': model_config,
                    'train_config': train_config,
                    'fold': fold + 1,
                    'random_seed': train_config.get('random_seed', 4200),
                    'train_indices': train_global_idx_np,
                    'val_indices': val_global_idx_np,
                    'validation_metrics': fold_record,
                    'latent_quality_metrics': (
                        best_checkpoint_candidate.get('latent_quality_metrics')
                        if best_checkpoint_candidate else None
                    ),
                    'classification_diagnostics': _compact_classification_diagnostics(
                        diagnostics
                    ),
                    'selected_threshold': fold_record['best_threshold'],
                    'threshold_strategy': 'best_validation_threshold',
                    'threshold_selection_metric': threshold_selection_metric,
                    'checkpoint_metric': checkpoint_metric,
                    'best_val_auc': best_epoch_results['auc'],
                    'selected_val_loss': best_epoch_results['val_loss'],
                    'minimum_val_loss_seen': float(best_val_loss.detach().cpu()) if torch.is_tensor(best_val_loss) else float(best_val_loss),
                    'best_epoch_results': best_epoch_results,
                }, checkpoint_path)

                if top_checkpoint_candidates:
                    metric_name_for_path = checkpoint_metric.replace('/', '_')
                    for rank, candidate in enumerate(top_checkpoint_candidates, start=1):
                        top_path = checkpoint_dir / (
                            f"{run_id}_fold_{fold+1}_rank_{rank}_"
                            f"{metric_name_for_path}_auc_{candidate['val_auc']:.4f}.pt"
                        )
                        if top_path.exists() and not train_config.get('overwrite_checkpoints', False):
                            raise FileExistsError(
                                f"Checkpoint already exists: {top_path}. "
                                "Set overwrite_checkpoints=True or choose a new run_id/checkpoint_dir."
                            )
                        torch.save({
                            'model_state_dict': candidate['model_state_dict'],
                            'model_config': model_config,
                            'train_config': train_config,
                            'fold': fold + 1,
                            'rank_by_val_auc': rank,
                            'rank_by_gvae_checkpoint_metric': rank,
                            'rank_by_latent_quality': (
                                rank if candidate.get('checkpoint_metric') in {
                                    'latent_quality',
                                    'latent_quality_score',
                                    'latent_linear_probe_auc',
                                    'linear_probe_auc',
                                } else None
                            ),
                            'epoch': candidate['epoch'],
                            'validation_auc': candidate['val_auc'],
                            'validation_pr_auc': candidate.get('val_pr_auc'),
                            'validation_balanced_accuracy': candidate.get(
                                'val_balanced_accuracy'
                            ),
                            'validation_f1': candidate.get('val_f1'),
                            'validation_loss': candidate['val_loss'],
                            'checkpoint_metric': candidate.get(
                                'checkpoint_metric',
                                checkpoint_metric,
                            ),
                            'validation_diagnostics': candidate.get(
                                'validation_diagnostics'
                            ),
                            'latent_quality_metrics': candidate.get(
                                'latent_quality_metrics'
                            ),
                            'train_indices': train_global_idx_np,
                            'val_indices': val_global_idx_np,
                            'selection_stage': 'top_k_by_gvae_validation_metric',
                        }, top_path)

        del model, optimizer, scheduler, fold_data
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Summary ---
    if not fold_metrics_list: return {}, pd.DataFrame(), []
    df_fold_metrics = pd.DataFrame(fold_metrics_list)
    metric_cols = [
        'auc',
        'roc_auc',
        'pr_auc',
        'f1',
        'accuracy',
        'balanced_accuracy',
        'precision',
        'recall',
        'sensitivity',
        'specificity',
        'brier_score',
        'f1_threshold_0_5',
        'accuracy_threshold_0_5',
        'balanced_accuracy_threshold_0_5',
        'precision_threshold_0_5',
        'recall_threshold_0_5',
        'sensitivity_threshold_0_5',
        'specificity_threshold_0_5',
        'f1_best_threshold',
        'accuracy_best_threshold',
        'balanced_accuracy_best_threshold',
        'precision_best_threshold',
        'recall_best_threshold',
        'sensitivity_best_threshold',
        'specificity_best_threshold',
        'best_threshold',
        'best_f1_threshold',
        'best_balanced_accuracy_threshold',
    ]
    results_summary = {
        f"{agg}_{m}": getattr(df_fold_metrics[m], agg)()
        for m in metric_cols
        for agg in ['mean', 'std']
    }
    print("\n--- Cross-Validation Summary ---")
    for key, value in results_summary.items(): print(f"  {key:<20}: {value:.4f}")
    return results_summary, df_fold_metrics, all_roc_data


def train_gvae_single_fold(
    full_multi_view_data: HeteroData,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    fold_num: int
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Trains and validates a GVAE model for a single fold of cross-validation.

    This function handles the complete training loop, including loss calculation,
    backpropagation, validation, learning rate scheduling, and early stopping.
    It saves and returns the state dictionary of the model that achieves the
    highest validation AUC during training.

    Args:
        full_multi_view_data (HeteroData): The complete dataset, must be on the correct device.
        train_indices (torch.Tensor): Tensor of global indices for the training set of this fold.
        val_indices (torch.Tensor): Tensor of global indices for the validation set of this fold.
        model_config (Dict[str, Any]): Configuration dictionary for the GVAE model architecture.
        train_config (Dict[str, Any]): Configuration dictionary for the training process.
        fold_num (int): The current fold number (e.g., 1, 2, ...) for logging purposes.

    Returns:
        Optional[Dict[str, torch.Tensor]]: The state dictionary of the best performing model on the validation
                                            set (based on AUC). Returns None if training fails or no best
                                            model is found.
    """
    device = train_config['device']

    # --- 0. Pre-train Radiology Aggregator (if applicable) ---
    radiology_state_dict = None
    if 'radiology' in model_config['view_configs'] and model_config.get('radiology_aggregator_config'):
        radiology_state_dict = pretrain_radiology_aggregator(
            full_multi_view_data, 
            train_indices, 
            model_config['radiology_aggregator_config'], 
            device,
            epochs=train_config.get('pretrain_epochs', 400),
            use_pos_weight=train_config.get('pretrain_use_pos_weight', False),
            pretrain_val_split=train_config.get('pretrain_val_split', 0.0),
            patience=train_config.get('pretrain_patience', 30),
            seed=train_config.get('pretrain_seed', train_config.get('random_seed', 42)),
        )

    # --- 1. Model, Optimizer, and Loss Initialization ---
    model = GVAE(**model_config).to(device)

    if radiology_state_dict is not None:
        print(f"   [Fold {fold_num}] Loading pre-trained radiology aggregator weights...")
        model.radiology_lesion_aggregator.load_state_dict(radiology_state_dict)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=train_config.get('patience', 10),
    )

    train_labels_for_weight = full_multi_view_data['patient']['binary_label'].to(device)[train_indices]
    train_labels_np_for_weight = train_labels_for_weight.detach().cpu().numpy()
    pos_weight_value, pos_weight_metadata = _compute_main_pos_weight(
        train_labels_np_for_weight,
        train_config,
    )
    print(
        f"INFO: Fold {fold_num} train-only pos_weight for Main Task BCE loss: "
        f"{pos_weight_value:.2f} "
        f"(negative={pos_weight_metadata['train_negative']}, "
        f"positive={pos_weight_metadata['train_positive']}, "
        f"strategy={pos_weight_metadata['resolved_pos_weight_strategy']})"
    )
    criterion_main_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=device)
    )
    criterion_bce_logits = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    clinical_dim = model_config.get('view_configs', {}).get('clinical', {}).get(
        'in_channels',
        0,
    )
    clinical_cont_idx = torch.arange(0, min(5, clinical_dim), device=device)
    clinical_bin_idx = torch.arange(5, min(22, clinical_dim), device=device)
    if clinical_bin_idx.numel() > 0 and 'x_clinical' in full_multi_view_data['patient']:
        clinical_bin_feats_train = (
            full_multi_view_data['patient'].x_clinical.to(device)[train_indices][
                :, clinical_bin_idx
            ]
        )
        n_pos = clinical_bin_feats_train.sum(dim=0)
        n_neg = clinical_bin_feats_train.shape[0] - n_pos
        clinical_bin_pos_weight = n_neg / (n_pos + 1e-6)
    else:
        clinical_bin_pos_weight = None

    # Lấy các trọng số loss từ config
    loss_weights = train_config['loss_weights']
    anneal_config = train_config.get('annealing', {})

    class_loss_multiplier = train_config.get('classification_loss_multiplier', 1.0)
    base_w_class = loss_weights['class'] * class_loss_multiplier
    base_w_cross_cl = loss_weights.get('cross_cl', 0.0)
    base_w_kl = loss_weights['kl']
    w_rec_attr_config = loss_weights['rec_attr']
    w_rec_struct_config = loss_weights['rec_struct']

    kl_params = anneal_config.get('kl', {})
    cl_params = anneal_config.get('cross_cl', {})

    kl_start_w = kl_params.get('start_weight', base_w_kl)
    kl_start_e, kl_end_e = kl_params.get(
        'start_epoch', 0), kl_params.get('end_epoch', 0)

    cl_start_w = cl_params.get('start_weight', base_w_cross_cl)
    cl_start_e, cl_end_e = cl_params.get(
        'start_epoch', 0), cl_params.get('end_epoch', 0)

    use_vec_cl_sf = train_config.get('vectorized_contrastive', False)
    contrastive_fn_sf = (calculate_contrastive_loss_vectorized
                         if use_vec_cl_sf else calculate_contrastive_loss)

    # --- 2. Training Loop Initialization ---
    best_val_auc = -1.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    checkpoint_metric = train_config.get('checkpoint_metric', 'latent_quality')
    early_stopping_metric = train_config.get('early_stopping_metric', checkpoint_metric)
    valid_checkpoint_metrics = {
        'auc',
        'roc_auc',
        'pr_auc',
        'balanced_accuracy',
        'f1',
        'loss',
        'auc_pr_balanced_accuracy',
        'auc_pr_balanced',
        'latent_quality',
        'latent_quality_score',
        'latent_linear_probe_auc',
        'linear_probe_auc',
    }
    if checkpoint_metric not in valid_checkpoint_metrics:
        raise ValueError(f"Unknown checkpoint_metric: {checkpoint_metric}")
    if early_stopping_metric not in valid_checkpoint_metrics:
        raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}")
    best_checkpoint_key = None
    best_early_stopping_key = None
    top_k = int(train_config.get('top_k_gvae_checkpoints', 1))
    return_top_k = bool(train_config.get('return_top_k_checkpoints', False))
    top_checkpoint_candidates: List[Dict[str, Any]] = []

    for epoch in range(1, train_config['epochs'] + 1):
        # --- Training Phase ---
        model.train()

        w_kl = linear_anneal(epoch, kl_start_e, kl_end_e,
                             kl_start_w, base_w_kl)
        w_cl = linear_anneal(epoch, cl_start_e, cl_end_e,
                             cl_start_w, base_w_cross_cl)

        train_batches = _make_train_batches(
            train_indices,
            full_multi_view_data['patient']['binary_label'].to(device),
            train_config,
            epoch,
            fold_num,
        )
        batch_train_losses = []
        batch_component_records = []

        for batch_idx in train_batches:
            logits, vae_out, cl_out, _ = model(full_multi_view_data, batch_idx)
            labels = full_multi_view_data['patient']['binary_label'].to(device)[batch_idx]

            # Calculate all loss components for training
            loss_class = criterion_main_bce(logits.squeeze(-1), labels.float())
            if len(cl_out) > 0:
                loss_cl = contrastive_fn_sf(
                    cl_out, train_config['cross_cl_temp'])
            else:
                loss_cl = torch.tensor(0.0, device=device)

            rec_attr, rec_struct, kl_div, active_views = 0.0, 0.0, 0.0, 0
            for view, vo in vae_out.items():
                if vo and vo.get('mu') is not None:
                    active_views += 1
                    w_attr = w_rec_attr_config.get(view, 1.0) if isinstance(
                        w_rec_attr_config, dict) else w_rec_attr_config
                    if (
                        view == 'clinical'
                        and clinical_cont_idx.numel() > 0
                        and clinical_bin_idx.numel() > 0
                    ):
                        loss_cont = criterion_mse(
                            vo['rec_x'][:, clinical_cont_idx],
                            vo['original_x_subset'][:, clinical_cont_idx],
                        )
                        loss_bin = F.binary_cross_entropy_with_logits(
                            vo['rec_x'][:, clinical_bin_idx],
                            vo['original_x_subset'][:, clinical_bin_idx],
                            pos_weight=clinical_bin_pos_weight,
                        )
                        rec_attr += w_attr * (loss_cont + loss_bin)
                    else:
                        rec_attr += w_attr * criterion_mse(
                            vo['rec_x'],
                            vo['original_x_subset'],
                        )

                    w_struct = w_rec_struct_config.get(view, 1.0) if isinstance(
                        w_rec_struct_config, dict) else w_rec_struct_config
                    rec_struct += w_struct * \
                        criterion_bce_logits(
                            vo['rec_adj_logits'].flatten(), vo['original_adj_subset'].flatten())

                    kl_div += -0.5 * \
                        torch.sum(1 + vo['logvar'] - vo['mu'].pow(2) -
                                  vo['logvar'].exp(), dim=1).mean()

            avg_rec_attr = rec_attr / \
                active_views if active_views > 0 else torch.tensor(
                    0.0, device=device)
            avg_rec_struct = rec_struct / \
                active_views if active_views > 0 else torch.tensor(
                    0.0, device=device)
            avg_kl = kl_div / \
                active_views if active_views > 0 else torch.tensor(
                    0.0, device=device)

            total_train_loss = (base_w_class * loss_class +
                                w_cl * loss_cl +
                                avg_rec_attr +
                                avg_rec_struct +
                                w_kl * avg_kl)

            if not torch.isfinite(total_train_loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            total_train_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), train_config['grad_clip_norm'])
            optimizer.step()
            
            batch_train_losses.append(total_train_loss.item())
            batch_component_records.append({
                'class': float(loss_class.detach().cpu()),
                'class_weighted': float((base_w_class * loss_class).detach().cpu()),
                'contrastive': float(loss_cl.detach().cpu()),
                'contrastive_weighted': float((w_cl * loss_cl).detach().cpu()),
                'rec_attr': float(avg_rec_attr.detach().cpu()),
                'rec_struct': float(avg_rec_struct.detach().cpu()),
                'kl': float(avg_kl.detach().cpu()),
                'kl_weighted': float((w_kl * avg_kl).detach().cpu()),
                'total': float(total_train_loss.detach().cpu()),
            })
        
        total_train_loss = (
            torch.tensor(np.mean(batch_train_losses), device=device)
            if batch_train_losses
            else torch.tensor(float('inf'), device=device)
        )
        epoch_loss_components = _mean_component_records(batch_component_records)

        # --- Validation Phase ---
        model.eval()
        current_val_auc = -1.0
        current_candidate = None
        total_val_loss_val = torch.tensor(float('inf'), device=device)

        if val_indices.numel() > 0:
            with torch.no_grad():
                val_logits, val_vae_out, val_cl_out, _ = model(
                    full_multi_view_data, val_indices)
                val_labels = full_multi_view_data['patient']['binary_label'].to(device)[val_indices]

                # Calculate validation loss for scheduler/early stopping
                # Using final annealed weights for a stable target
                val_loss_class = criterion_main_bce(
                    val_logits.squeeze(-1), val_labels.float())
                if len(val_cl_out) > 0:
                    val_loss_cl = contrastive_fn_sf(
                        val_cl_out, train_config['cross_cl_temp'])
                else:
                    val_loss_cl = torch.tensor(0.0, device=device)

                val_rec_attr, val_rec_struct, val_kl, val_active_views = 0.0, 0.0, 0.0, 0
                for view, vo in val_vae_out.items():
                    if vo and vo.get('mu') is not None:
                        val_active_views += 1
                        w_attr = w_rec_attr_config.get(view, 1.0) if isinstance(
                            w_rec_attr_config, dict) else w_rec_attr_config
                        if (
                            view == 'clinical'
                            and clinical_cont_idx.numel() > 0
                            and clinical_bin_idx.numel() > 0
                        ):
                            val_loss_cont = criterion_mse(
                                vo['rec_x'][:, clinical_cont_idx],
                                vo['original_x_subset'][:, clinical_cont_idx],
                            )
                            val_loss_bin = F.binary_cross_entropy_with_logits(
                                vo['rec_x'][:, clinical_bin_idx],
                                vo['original_x_subset'][:, clinical_bin_idx],
                                pos_weight=clinical_bin_pos_weight,
                            )
                            val_rec_attr += w_attr * (val_loss_cont + val_loss_bin)
                        else:
                            val_rec_attr += w_attr * criterion_mse(
                                vo['rec_x'],
                                vo['original_x_subset'],
                            )
                        w_struct = w_rec_struct_config.get(view, 1.0) if isinstance(
                            w_rec_struct_config, dict) else w_rec_struct_config
                        val_rec_struct += w_struct * \
                            criterion_bce_logits(
                                vo['rec_adj_logits'].flatten(), vo['original_adj_subset'].flatten())
                        val_kl += -0.5 * \
                            torch.sum(
                                1 + vo['logvar'] - vo['mu'].pow(2) - vo['logvar'].exp(), dim=1).mean()

                avg_val_rec_attr = val_rec_attr / \
                    val_active_views if val_active_views > 0 else torch.tensor(
                        0.0, device=device)
                avg_val_rec_struct = val_rec_struct / \
                    val_active_views if val_active_views > 0 else torch.tensor(
                        0.0, device=device)
                avg_val_kl = val_kl / \
                    val_active_views if val_active_views > 0 else torch.tensor(
                        0.0, device=device)

                total_val_loss_val = (base_w_class * val_loss_class + base_w_cross_cl * val_loss_cl +
                                      avg_val_rec_attr + avg_val_rec_struct + base_w_kl * avg_val_kl)

                # Calculate validation AUC for model saving
                if not torch.isnan(val_logits).any() and len(torch.unique(val_labels)) > 1:
                    val_probs = torch.sigmoid(
                        val_logits.squeeze(-1)).cpu().numpy()
                    val_labels_np = val_labels.cpu().numpy()
                    current_val_auc = roc_auc_score(
                        val_labels_np, val_probs)
                    threshold_selection_metric = train_config.get(
                        'classification_threshold_selection_metric',
                        'balanced_accuracy',
                    )
                    candidate_diagnostics = build_binary_classification_diagnostics(
                        val_labels_np,
                        val_probs,
                        train_labels=train_labels_np_for_weight,
                        threshold_selection_metric=threshold_selection_metric,
                        metadata={
                            'model': 'GVAE',
                            'fold': fold_num,
                            'epoch': epoch,
                            'score_source': 'sigmoid(classifier_logit)',
                            'bce_loss': 'BCEWithLogitsLoss',
                            **pos_weight_metadata,
                            'classification_loss_weight': float(base_w_class),
                            'classification_loss_multiplier': float(class_loss_multiplier),
                            'loss_components': epoch_loss_components,
                            'stratified_split': True,
                        },
                        probability_note=(
                            'GVAE classifier probability is sigmoid(logit). '
                            'The classifier head outputs logits and training uses '
                            'BCEWithLogitsLoss.'
                        ),
                    )
                    latent_quality_metrics = {}
                    should_compute_latent_quality = bool(
                        train_config.get(
                            'compute_latent_quality_metrics',
                            checkpoint_metric in {
                                'latent_quality',
                                'latent_quality_score',
                                'latent_linear_probe_auc',
                                'linear_probe_auc',
                            } or top_k > 0 or return_top_k,
                        )
                    )
                    if should_compute_latent_quality:
                        latent_quality_metrics = _compute_latent_quality_metrics(
                            model,
                            full_multi_view_data,
                            train_indices,
                            val_indices,
                            train_config,
                        )

                    current_candidate = {
                        'epoch': epoch,
                        'val_auc': float(current_val_auc),
                        'val_pr_auc': float(
                            candidate_diagnostics['threshold_0_5_metrics']['pr_auc']
                        ),
                        'val_balanced_accuracy': float(
                            candidate_diagnostics['best_threshold_metrics'][
                                'balanced_accuracy'
                            ]
                        ),
                        'val_f1': float(
                            candidate_diagnostics['best_threshold_metrics']['f1']
                        ),
                        'val_loss': float(total_val_loss_val.detach().cpu()),
                        'checkpoint_metric': checkpoint_metric,
                        'validation_diagnostics': _compact_classification_diagnostics(
                            candidate_diagnostics
                        ),
                        'latent_quality_metrics': latent_quality_metrics,
                        'model_state_dict': {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        },
                    }
                    if top_k > 0:
                        top_checkpoint_candidates.append(current_candidate)
                        top_checkpoint_candidates = _rank_checkpoint_candidates(
                            top_checkpoint_candidates, top_k
                        )

        # --- Logging and Checkpointing ---
        if epoch % train_config.get('print_every_k_epochs', 10) == 0:
            print(
                f"  F{fold_num} Ep{epoch:03d} "
                f"| TrainLoss:{total_train_loss.item():.4f} "
                f"| ValLoss:{total_val_loss_val.item():.4f} "
                f"| Class:{epoch_loss_components.get('class', np.nan):.4f} "
                f"(w:{epoch_loss_components.get('class_weighted', np.nan):.4f}) "
                f"| RecA:{epoch_loss_components.get('rec_attr', np.nan):.4f} "
                f"| RecS:{epoch_loss_components.get('rec_struct', np.nan):.4f} "
                f"| KL:{epoch_loss_components.get('kl', np.nan):.4f} "
                f"| ValAUC:{current_val_auc:.4f} (BestAUC:{best_val_auc:.4f}) "
                f"| CkptMetric:{checkpoint_metric}"
            )

        # Scheduler still follows total validation loss; checkpoint/early stop can
        # use rank metrics such as AUC -> PR-AUC -> balanced accuracy.
        scheduler.step(total_val_loss_val)
        if total_val_loss_val < best_val_loss:
            best_val_loss = total_val_loss_val

        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc

        if current_candidate is not None:
            current_checkpoint_key = _checkpoint_sort_key(
                current_candidate,
                checkpoint_metric,
            )
            if best_checkpoint_key is None or current_checkpoint_key > best_checkpoint_key:
                best_checkpoint_key = current_checkpoint_key
                best_model_state = copy.deepcopy(model.state_dict())
            current_early_stopping_key = _checkpoint_sort_key(
                current_candidate,
                early_stopping_metric,
            )
            if (
                best_early_stopping_key is None
                or current_early_stopping_key > best_early_stopping_key
            ):
                best_early_stopping_key = current_early_stopping_key
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_config.get('patience_early_stopping', 20):
            print(
                f"  Early stopping triggered at epoch {epoch} for fold {fold_num}.")
            break

    # --- Final Cleanup for the Fold ---
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if best_model_state is None:
        print(
            f"Warning: No best model was saved for fold {fold_num} based on "
            f"{checkpoint_metric} improvement.")

    if return_top_k:
        for rank, candidate in enumerate(top_checkpoint_candidates, start=1):
            candidate['rank_by_val_auc'] = rank
            candidate['rank_by_gvae_checkpoint_metric'] = rank
            candidate['rank_by_latent_quality'] = (
                rank if candidate.get('checkpoint_metric') in {
                    'latent_quality',
                    'latent_quality_score',
                    'latent_linear_probe_auc',
                    'linear_probe_auc',
                } else None
            )
        return top_checkpoint_candidates

    return best_model_state

def pretrain_radiology_aggregator(
    full_data: HeteroData, 
    train_indices: torch.Tensor, 
    config: Dict[str, Any], 
    device: torch.device,
    epochs: int = 400,
    use_pos_weight: bool = False,
    pretrain_val_split: float = 0.0,
    patience: int = 30,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Pre-trains the radiology aggregator to classify response using only lesion data.
    Returns the state_dict of the trained aggregator.
    """
    print(f"   [Pre-training] Warming up Radiology Aggregator for {epochs} epochs...")
    
    # 1. Filter patients who actually have radiology data
    rad_mask = full_data['patient']['radiology_mask'].bool()
    # Intersection of training indices and patients with radiology
    train_indices_mask = torch.zeros(full_data['patient'].num_nodes, dtype=torch.bool, device=device)
    train_indices_mask[train_indices] = True
    
    # Final mask: In Training Set AND Has Radiology
    active_mask = train_indices_mask & rad_mask.to(device)
    active_patient_indices = torch.where(active_mask)[0]
    
    if active_patient_indices.numel() == 0:
        print("   [Pre-training] Warning: No patients with radiology data in this fold. Skipping.")
        return None

    # 2. Prepare Data for Aggregator
    # We need to map global indices to local batch indices (0 to N-1)
    patient_lesion_edges = full_data['patient', 'has_lesion', 'lesion'].edge_index.to(device)
    lesion_features = full_data['lesion'].x.to(device)
    labels = full_data['patient']['binary_label'].float().to(device)

    # Create a map: Global_Index -> Local_Index
    global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(active_patient_indices)}
    
    # Filter edges: Keep only edges belonging to active patients
    edge_mask = torch.isin(patient_lesion_edges[0], active_patient_indices)
    filtered_edges = patient_lesion_edges[:, edge_mask]
    
    # Remap source (patient) indices to local range [0, batch_size]
    local_src = torch.tensor([global_to_local[idx.item()] for idx in filtered_edges[0]], device=device)
    # Keep dst (lesion) indices as is, to index into lesion_features
    global_dst = filtered_edges[1]
    
    # Create the edge_index format required by the Aggregator
    # [0] = local patient index, [1] = local lesion index (relative to the subset of lesions used)
    # However, the aggregator usually takes raw features. 
    # Optimization: Select only relevant lesions to save memory
    unique_lesion_indices, inverse_lesion_indices = torch.unique(global_dst, return_inverse=True)
    batch_lesion_features = lesion_features[unique_lesion_indices]
    
    aggregator_edge_index = torch.stack([local_src, inverse_lesion_indices])
    batch_labels = labels[active_patient_indices]

    # 3. Initialize Model
    mil_model = StandaloneRadiologyMIL(config).to(device)
    optimizer = optim.Adam(mil_model.parameters(), lr=1e-3, weight_decay=1e-4)
    if use_pos_weight:
        n_pos = batch_labels.sum()
        n_neg = batch_labels.numel() - n_pos
        pw = (n_neg / (n_pos + 1e-6)).clamp(min=1e-3).to(device)
        criterion_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')
    else:
        criterion_none = nn.BCEWithLogitsLoss(reduction='none')

    # 4. Optional STRATIFIED held-out split (patient-level) for validation + early stopping.
    #    Stratifying by label keeps both classes in val (AUC is undefined otherwise) and
    #    never empties a class from train. On the small radiology subset an unstratified
    #    split easily lands 0 positives in val, making early stopping unreliable.
    n_active = len(active_patient_indices)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    if pretrain_val_split > 0:
        labels_cpu = batch_labels.detach().cpu()
        val_parts = []
        for cls in (0.0, 1.0):
            cls_local = torch.where(labels_cpu == cls)[0]
            if cls_local.numel() == 0:
                continue
            n_cls_val = int(round(pretrain_val_split * cls_local.numel()))
            # Keep at least one of this class in train; allow 0 in val if the class is tiny.
            n_cls_val = max(0, min(n_cls_val, cls_local.numel() - 1))
            if n_cls_val > 0:
                shuffled = cls_local[torch.randperm(cls_local.numel(), generator=gen)]
                val_parts.append(shuffled[:n_cls_val])
        val_local = torch.cat(val_parts).to(device) if val_parts else torch.empty(0, dtype=torch.long, device=device)
        train_mask = torch.ones(n_active, dtype=torch.bool, device=device)
        train_mask[val_local] = False
        train_local = torch.where(train_mask)[0]
    else:
        val_local = torch.empty(0, dtype=torch.long, device=device)
        train_local = torch.arange(n_active, device=device)
    val_count = val_local.numel()

    best_val_auc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        mil_model.train()
        optimizer.zero_grad()
        logits = mil_model(batch_lesion_features, aggregator_edge_index, n_active)
        per_patient = criterion_none(logits.squeeze(-1), batch_labels)
        loss = per_patient[train_local].mean()
        loss.backward()
        optimizer.step()

        if val_count > 0:
            mil_model.eval()
            with torch.no_grad():
                val_logits = mil_model(batch_lesion_features, aggregator_edge_index, n_active)
                val_probs = torch.sigmoid(val_logits.squeeze(-1))[val_local].cpu().numpy()
                val_true = batch_labels[val_local].cpu().numpy()
            # Early stopping tracks val AUC (the quantity we care about). AUC is undefined
            # if val collapses to one class; on those epochs we leave no_improve unchanged.
            val_auc = None
            if len(np.unique(val_true)) > 1:
                try:
                    val_auc = roc_auc_score(val_true, val_probs)
                except Exception:
                    val_auc = None
            if val_auc is not None:
                if val_auc > best_val_auc:
                    best_val_auc, best_state, no_improve = val_auc, copy.deepcopy(mil_model.state_dict()), 0
                else:
                    no_improve += 1
            if epoch % 50 == 0:
                auc_str = f"{val_auc:.4f}" if val_auc is not None else "n/a"
                print(f"   [Pre-training] Ep {epoch}: train_loss={loss.item():.4f}, val_AUC={auc_str} (best={best_val_auc:.4f})")
            if no_improve >= patience:
                print(f"   [Pre-training] Early stop at epoch {epoch} (best val_AUC={best_val_auc:.4f}).")
                break
        elif epoch % 50 == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits.squeeze(-1))
                try:
                    auc = roc_auc_score(batch_labels.cpu().numpy(), probs.cpu().numpy())
                except Exception:
                    auc = 0.5
                print(f"   [Pre-training] Ep {epoch}: train_loss={loss.item():.4f}, TRAIN AUC={auc:.4f} (no val split)")

    if best_state is not None:
        mil_model.load_state_dict(best_state)

    print("   [Pre-training] Complete. Extracting aggregator weights.")
    return mil_model.aggregator.state_dict()


def sweep_pretrain_recipes(
    full_data: HeteroData,
    model_config: Dict,
    train_config: Dict,
    recipes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Compare radiology-pretraining recipes by the metric that actually matters:
    DOWNSTREAM GVAE cross-validated val-AUC — NOT the pretrainer's own AUC.

    Each recipe is a set of train_config overrides; this runs `kfold_train_gvae`
    once per recipe and tabulates mean/std val-AUC.

    Tip: this runs full k-fold per recipe. For a fast sweep, pass a `train_config`
    with a small `n_splits` (e.g. 3) and `epochs` (e.g. 200) — you are comparing
    recipes against each other, not producing final numbers.

    Returns: {recipe_name: (mean_auc, std_auc)}.
    """
    if recipes is None:
        recipes = {
            '(a) no pretrain':           {'pretrain_epochs': 0},
            '(b) 100 ep, no split':      {'pretrain_epochs': 100, 'pretrain_val_split': 0.0},
            '(c) 400 ep, no split':      {'pretrain_epochs': 400, 'pretrain_val_split': 0.0},
            '(d) val-split + earlystop': {'pretrain_epochs': 400, 'pretrain_val_split': 0.2},
        }

    results: Dict[str, Tuple[float, float]] = {}
    for name, overrides in recipes.items():
        print(f"\n{'#'*60}\n# RECIPE {name}: {overrides}\n{'#'*60}")
        tc = copy.deepcopy(train_config)
        tc.update(overrides)
        summary, _, _ = kfold_train_gvae(full_data, model_config, tc)
        results[name] = (summary.get('mean_auc', float('nan')),
                         summary.get('std_auc', float('nan')))

    print(f"\n{'='*60}\n PRETRAIN RECIPE COMPARISON (downstream GVAE val-AUC)\n{'='*60}")
    print(f"  {'recipe':<28}{'mean_auc':>10}{'std_auc':>10}")
    for name, (m, s) in results.items():
        print(f"  {name:<28}{m:>10.4f}{s:>10.4f}")
    best = max(results, key=lambda k: results[k][0] if results[k][0] == results[k][0] else -1)
    print(f"\n  -> Best by downstream val-AUC: {best}")
    return results
