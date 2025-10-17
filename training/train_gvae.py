import torch
import numpy as np
import pandas as pd
import gc
from typing import Dict, Optional, Tuple, List
from torch_geometric.data import HeteroData
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm as tqdm_notebook
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any
from torch_geometric.loader import NeighborLoader
from models.gvae_model import GVAE
from utils.training_utils import linear_anneal
from utils.loss_utils import calculate_contrastive_loss
from tqdm import tqdm

def kfold_train_gvae(
    full_multi_view_data: HeteroData,
    model_config: Dict,
    train_config: Dict,
    wandb_params: Optional[Dict] = None
) -> Tuple[Dict[str, float], pd.DataFrame, List[Dict]]:
    """
    Performs K-fold cross-validation using INTRA-VIEW contrastive loss.
    """
    device = train_config['device']
    full_multi_view_data = full_multi_view_data.to(device)
    full_multi_view_data_cpu = full_multi_view_data.clone().cpu()

    # --- Loss and Annealing Setup ---
    criterion_bce_logits = nn.BCEWithLogitsLoss()
    # criterion_bce_logits = FocalLoss(alpha=0.33, gamma=1.0)
    criterion_mse = nn.MSELoss()
    loss_weights_config = train_config['loss_weights']
    anneal_config = train_config.get('annealing', {})

    base_w_class = loss_weights_config['class']
    base_w_intra_cl = loss_weights_config.get('cross_cl', 0.0)
    base_w_kl = loss_weights_config['kl']
    w_rec_attr_config = loss_weights_config.get('rec_attr', 1.0)
    w_rec_struct_config = loss_weights_config.get('rec_struct', 1.0)
    print(f"Base Intra-View CL Weight: {base_w_intra_cl}")

    kl_params = anneal_config.get('kl', {})
    kl_start_w = kl_params.get('start_weight', base_w_kl)
    kl_end_w = kl_params.get('end_weight', base_w_kl)
    kl_start_e = kl_params.get('start_epoch', 0)
    kl_end_e = kl_params.get('end_epoch', 0)
    cl_params = anneal_config.get('intra_cl', {})
    cl_start_w = cl_params.get('start_weight', base_w_intra_cl)
    cl_end_w = cl_params.get('end_weight', base_w_intra_cl)
    cl_start_e = cl_params.get('start_epoch', 0)
    cl_end_e = cl_params.get('end_epoch', 0)

    # --- Data Splitting ---
    all_patient_indices_np = np.arange(
        full_multi_view_data['patient'].num_nodes)
    # kf = KFold(n_splits=train_config['n_splits'], shuffle=True, random_state=train_config.get('random_seed', 400))
    y_for_stratification = full_multi_view_data['patient']['binary_label'].cpu(
    ).numpy()
    kf = StratifiedKFold(
        n_splits=train_config['n_splits'], shuffle=True, random_state=train_config.get('random_seed', 420))

    fold_metrics_list = []
    roc_data_per_fold = []
    best_model_state_fold = None

    for fold, (train_global_idx_np, val_global_idx_np) in enumerate(kf.split(all_patient_indices_np, y_for_stratification)):
        print(f"\n===== Fold {fold+1}/{train_config['n_splits']} =====")

        # 2. Prepare dictionaries to hold the new, potentially PCA-reduced feature tensors
        new_patient_features = {}
        new_lesion_features = {}

        # 3. Perform Patient-Level PCA
        pca_config = train_config.get('pca_config', {})
        print("  Processing patient-level features...")
        for view in ['clinical', 'pathology']:
            feature_key = f'x_{view}'
            if feature_key in full_multi_view_data_cpu['patient']:
                original_features = full_multi_view_data_cpu['patient'][feature_key].numpy(
                )
                if view in pca_config:
                    n_components = pca_config[view]
                    max_components = min(
                        len(train_global_idx_np), original_features.shape[1])
                    final_n_components = min(n_components, max_components)
                    if final_n_components < original_features.shape[1]:
                        print(
                            f"    - Applying PCA on '{view}': {original_features.shape[1]} -> {final_n_components} features")
                        pca = PCA(n_components=final_n_components,
                                  random_state=train_config.get('random_seed', 42))
                        pca.fit(original_features[train_global_idx_np])
                        reduced_features = pca.transform(original_features)
                        new_patient_features[feature_key] = torch.tensor(
                            reduced_features, dtype=torch.float32)
                    else:
                        new_patient_features[feature_key] = torch.tensor(
                            original_features, dtype=torch.float32)
                else:
                    new_patient_features[feature_key] = torch.tensor(
                        original_features, dtype=torch.float32)

        # 3. Perform Lesion-Level PCA
        # ============================= FIX 1 START =============================
        # The logic here is fixed to correctly detect lesion features and apply PCA robustly.
        lesion_pca_config = train_config.get('lesion_pca_config')
        print("  Processing lesion-level features...")

        # Use a more robust check for lesion features
        is_lesion_present = 'lesion' in full_multi_view_data_cpu.node_types
        has_lesion_x = is_lesion_present and hasattr(
            full_multi_view_data_cpu['lesion'], 'x') and full_multi_view_data_cpu['lesion'].x.numel() > 0

        if has_lesion_x and lesion_pca_config:
            original_lesion_features_np = full_multi_view_data_cpu['lesion'].x.numpy(
            )
            n_components = lesion_pca_config['n_components']

            # Find lesions belonging to training patients to fit PCA correctly
            patient_lesion_edges = full_multi_view_data_cpu['patient',
                                                            'has_lesion', 'lesion'].edge_index
            is_train_patient_edge = torch.isin(
                patient_lesion_edges[0], torch.from_numpy(train_global_idx_np))
            train_lesion_indices = torch.unique(
                patient_lesion_edges[1, is_train_patient_edge]).numpy()

            max_components = min(len(train_lesion_indices),
                                 original_lesion_features_np.shape[1])
            final_n_components = min(n_components, max_components)

            # Apply PCA only if it's possible and beneficial
            if len(train_lesion_indices) > 0 and final_n_components < original_lesion_features_np.shape[1]:
                print(
                    f"    - Applying PCA on 'lesion.x': {original_lesion_features_np.shape[1]} -> {final_n_components} features")
                pca_lesion = PCA(n_components=final_n_components,
                                 random_state=train_config.get('random_seed', 42))
                # Fit ONLY on training data
                pca_lesion.fit(
                    original_lesion_features_np[train_lesion_indices])
                reduced_features = pca_lesion.transform(
                    original_lesion_features_np)  # Transform ALL data
                new_lesion_features['x'] = torch.tensor(
                    reduced_features, dtype=torch.float32)
            else:
                print(
                    "    - Copying 'lesion.x' features without PCA (conditions for reduction not met).")
                new_lesion_features['x'] = full_multi_view_data_cpu['lesion'].x.clone(
                )

        elif has_lesion_x:
            print("    - Copying 'lesion.x' features without PCA (not in config).")
            new_lesion_features['x'] = full_multi_view_data_cpu['lesion'].x.clone(
            )
        else:
            print("    - Skipping: 'lesion' node type or 'x' features not found.")
        # ============================== FIX 1 END ==============================

        # 4. Construct the new HeteroData object for this fold
        print("  Constructing new HeteroData object for the fold...")
        fold_data = HeteroData()

        # Copy patient features and metadata
        for key, value in new_patient_features.items():
            fold_data['patient'][key] = value
        for key in ['pathology_mask', 'radiology_mask', 'y', 'event', 'binary_label']:
            if key in full_multi_view_data_cpu['patient']:
                fold_data['patient'][key] = full_multi_view_data_cpu['patient'][key].clone()

        # Copy lesion features
        if new_lesion_features:
            fold_data['lesion'].x = new_lesion_features['x']

        # Copy graph structure
        for edge_type in full_multi_view_data_cpu.edge_types:
            for key, value in full_multi_view_data_cpu[edge_type].items():
                fold_data[edge_type][key] = value.clone()

        print("  New data object constructed successfully.")

        # fold_data = full_multi_view_data.clone().cpu()
        fold_data = fold_data.to(device)
        train_fold_global_indices = torch.from_numpy(
            train_global_idx_np).to(device)
        val_fold_global_indices = torch.from_numpy(
            val_global_idx_np).to(device)

        # NOTE: Assumes EndToEndMultiViewVAE_CL_AttentionRadiology is defined elsewhere
        # and its forward pass returns the necessary vae_outputs for loss calculation.
        model = GVAE(
            view_configs=model_config['view_configs'],
            projection_head_config=model_config['projection_head_config'],
            radiology_aggregator_config=model_config['radiology_aggregator_config'],
            fusion_config=model_config['fusion_config'],
            classifier_config=model_config['classifier_config'],
            d_embed=model_config['d_embed'],
            missing_strategy=model_config.get('missing_strategy', 'learnable')
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=train_config.get('patience', 10))

        best_val_loss = float('inf')
        best_val_auc_this_fold = -1.0
        epochs_no_improve = 0
        best_model_state_fold = None
        best_fold_roc_data = {}

        for epoch in range(1, train_config['epochs'] + 1):
            # --- Training Phase ---
            model.train()
            current_w_kl = linear_anneal(
                epoch, kl_start_e, kl_end_e, kl_start_w, kl_end_w)
            current_w_cl = linear_anneal(
                epoch, cl_start_e, cl_end_e, cl_start_w, cl_end_w)

            # The model's forward pass should return a dictionary `vae_outputs` that includes
            # 'mu' and 'local_edge_index' for each view.
            logits_train, vae_outputs_loss_train, mus_projected_for_cl_train, _ = model(
                full_multi_view_data, train_fold_global_indices)

            true_labels_train = full_multi_view_data['patient']['binary_label'][train_fold_global_indices]
            raw_loss_class_train = criterion_bce_logits(
                logits_train.squeeze(), true_labels_train.float())

            # --- MODIFIED: Intra-View CL Calculation ---
            total_cl_loss_train = 0.0
            num_active_views_cl_train = 0
            for view_name in model.views:
                vo = vae_outputs_loss_train.get(view_name, {})
                # Check if view has embeddings and a similarity graph for contrastive loss
                if vo and vo.get('mu') is not None and vo.get('local_edge_index') is not None and vo['local_edge_index'].numel() > 0:
                    projected_mu = model.projection_heads[view_name](vo['mu'])
                    total_cl_loss_train += calculate_contrastive_loss(
                        mus_projected_for_cl_train, train_config['cross_cl_temp'])
                    num_active_views_cl_train += 1
            raw_loss_cl_train = total_cl_loss_train / \
                num_active_views_cl_train if num_active_views_cl_train > 0 else torch.tensor(
                    0.0, device=device)
            # --- END MODIFICATION ---

            total_loss_rec_attr_train, total_loss_rec_struct_train, total_loss_kl_train = 0.0, 0.0, 0.0
            num_active_views_vae_train = 0
            for view_name in model.views:
                vo = vae_outputs_loss_train.get(view_name, {})
                if vo and vo.get('mu') is not None:
                    num_active_views_vae_train += 1
                    w_attr = w_rec_attr_config if isinstance(
                        w_rec_attr_config, float) else w_rec_attr_config.get(view_name, 1.0)
                    total_loss_rec_attr_train += w_attr * \
                        criterion_mse(vo['rec_x'], vo['original_x_subset'])
                    w_struct = w_rec_struct_config if isinstance(
                        w_rec_struct_config, float) else w_rec_struct_config.get(view_name, 1.0)
                    total_loss_rec_struct_train += w_struct * \
                        criterion_bce_logits(
                            vo['rec_adj_logits'].reshape(-1), vo['original_adj_subset'].reshape(-1))
                    kl_div = -0.5 * \
                        torch.sum(
                            1 + vo['logvar'] - vo['mu'].pow(2) - vo['logvar'].exp(), dim=1).mean()
                    total_loss_kl_train += kl_div

            avg_loss_rec_attr_train = total_loss_rec_attr_train / \
                num_active_views_vae_train if num_active_views_vae_train > 0 else 0.0
            avg_loss_rec_struct_train = total_loss_rec_struct_train / \
                num_active_views_vae_train if num_active_views_vae_train > 0 else 0.0
            avg_raw_loss_kl_train = total_loss_kl_train / \
                num_active_views_vae_train if num_active_views_vae_train > 0 else 0.0

            total_train_loss = (base_w_class * raw_loss_class_train +
                                current_w_cl * raw_loss_cl_train +
                                avg_loss_rec_attr_train +
                                avg_loss_rec_struct_train +
                                current_w_kl * avg_raw_loss_kl_train)

            if not torch.isnan(total_train_loss):
                optimizer.zero_grad()
                total_train_loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_config.get('grad_clip_norm', 1.0))
                optimizer.step()

            # --- Validation Phase ---
            model.eval()
            total_validation_loss = torch.tensor(float('inf'), device=device)
            current_val_auc = -1.0

            if val_fold_global_indices.numel() > 0:
                with torch.no_grad():
                    val_logits_raw, vae_outputs_loss_val, projected_mu_val, _ = model(
                        full_multi_view_data, val_fold_global_indices)
                    val_labels = full_multi_view_data['patient']['binary_label'][val_fold_global_indices]

                    raw_loss_class_val = criterion_bce_logits(
                        val_logits_raw.squeeze(), val_labels.float())

                    # --- MODIFIED: Intra-View CL Calculation for Validation ---
                    total_cl_loss_val = 0.0
                    num_active_views_cl_val = 0
                    for view_name in model.views:
                        vo_val = vae_outputs_loss_val.get(view_name, {})
                        if vo_val and vo_val.get('mu') is not None and vo_val.get('local_edge_index') is not None and vo_val['local_edge_index'].numel() > 0:
                            projected_mu_val = model.projection_heads[view_name](
                                vo_val['mu'])
                            total_cl_loss_val += calculate_contrastive_loss(
                                projected_mu_val, train_config['cross_cl_temp'])
                            # calculate_contrastive_loss_with_hard_negatives(projected_mu_val, train_config['cross_cl_temp'], train_config['num_hard_negatives'])
                            num_active_views_cl_val += 1
                    raw_loss_cl_val = total_cl_loss_val / \
                        num_active_views_cl_val if num_active_views_cl_val > 0 else torch.tensor(
                            0.0, device=device)
                    # --- END MODIFICATION ---

                    total_loss_rec_attr_val, total_loss_rec_struct_val, total_loss_kl_val = 0.0, 0.0, 0.0
                    num_active_views_vae_val = 0
                    for view_name in model.views:
                        vo_val = vae_outputs_loss_val.get(view_name, {})
                        if vo_val and vo_val.get('mu') is not None:
                            num_active_views_vae_val += 1
                            w_attr = w_rec_attr_config if isinstance(
                                w_rec_attr_config, float) else w_rec_attr_config.get(view_name, 1.0)
                            total_loss_rec_attr_val += w_attr * \
                                criterion_mse(
                                    vo_val['rec_x'], vo_val['original_x_subset'])
                            w_struct = w_rec_struct_config if isinstance(
                                w_rec_struct_config, float) else w_rec_struct_config.get(view_name, 1.0)
                            total_loss_rec_struct_val += w_struct * criterion_bce_logits(
                                vo_val['rec_adj_logits'].reshape(-1), vo_val['original_adj_subset'].reshape(-1))
                            kl_div_val = -0.5 * \
                                torch.sum(
                                    1 + vo_val['logvar'] - vo_val['mu'].pow(2) - vo_val['logvar'].exp(), dim=1).mean()
                            total_loss_kl_val += kl_div_val

                    avg_loss_rec_attr_val = total_loss_rec_attr_val / \
                        num_active_views_vae_val if num_active_views_vae_val > 0 else 0.0
                    avg_loss_rec_struct_val = total_loss_rec_struct_val / \
                        num_active_views_vae_val if num_active_views_vae_val > 0 else 0.0
                    avg_raw_loss_kl_val = total_loss_kl_val / \
                        num_active_views_vae_val if num_active_views_vae_val > 0 else 0.0

                    total_validation_loss = (base_w_class * raw_loss_class_val +
                                             cl_end_w * raw_loss_cl_val +
                                             avg_loss_rec_attr_val +
                                             avg_loss_rec_struct_val +
                                             kl_end_w * avg_raw_loss_kl_val)

                    if not torch.isnan(val_logits_raw).any() and len(np.unique(val_labels.cpu().numpy())) > 1:
                        val_probs_np = torch.sigmoid(
                            val_logits_raw.squeeze()).cpu().numpy()
                        current_val_auc = roc_auc_score(
                            val_labels.cpu().numpy(), val_probs_np)

            if epoch % train_config.get('print_every_k_epochs', 10) == 0:
                print(f"  F{fold+1} Ep{epoch:03d} TLoss:{total_train_loss.item():.4f} | "
                      f"VLoss:{total_validation_loss.item() if not torch.isinf(total_validation_loss) else -1:.4f} (Best VLoss: {best_val_loss:.4f}) | "
                      f"ValAUC:{current_val_auc:.4f} (Best ValAUC: {best_val_auc_this_fold:.4f})")

            scheduler.step(total_validation_loss)
            if total_validation_loss < best_val_loss:
                best_val_loss = total_validation_loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if current_val_auc != -1.0 and current_val_auc > best_val_auc_this_fold:
                best_val_auc_this_fold = current_val_auc
                best_model_state_fold = model.state_dict().copy()
                val_labels_np = val_labels.cpu().numpy()
                fpr, tpr, _ = roc_curve(val_labels_np, val_probs_np)
                best_fold_roc_data = {
                    'fpr': fpr, 'tpr': tpr, 'auc': current_val_auc,
                    'y_true': val_labels_np, 'y_pred': val_probs_np
                }

            if epochs_no_improve >= train_config.get('patience_early_stopping', 20):
                print(
                    f"  Early stopping at epoch {epoch} for fold {fold+1} due to validation loss stagnation.")
                break

        if best_fold_roc_data:
            roc_data_per_fold.append(best_fold_roc_data)
            y_true = best_fold_roc_data['y_true']
            y_pred_probs = best_fold_roc_data['y_pred']
            y_pred_binary = (y_pred_probs > 0.5).astype(int)
            fold_results = {
                'auc': best_fold_roc_data['auc'],
                'f1': f1_score(y_true, y_pred_binary, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true, y_pred_binary, zero_division=0)
            }
            fold_metrics_list.append(fold_results)
        else:
            fold_metrics_list.append(
                {'auc': np.nan, 'f1': np.nan, 'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan})

        del model, optimizer, scheduler, best_model_state_fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    if not fold_metrics_list:
        print("Warning: No metrics were collected during cross-validation.")
        return {}, pd.DataFrame(), []

    df_fold_metrics = pd.DataFrame(fold_metrics_list)
    mean_metrics = df_fold_metrics.mean()
    std_metrics = df_fold_metrics.std()

    results_summary = {}
    for metric in ['auc', 'f1', 'accuracy', 'precision', 'recall']:
        results_summary[f'mean_{metric}'] = mean_metrics.get(metric, np.nan)

        # Save the best model from the best fold
        if best_model_state_fold:
            torch.save(best_model_state_fold, train_config.get(
                'best_model_path', 'best_model.pth'))
            print(
                f"Best model saved to {train_config.get('best_model_path', 'best_model.pth')}")
            results_summary[f'std_{metric}'] = std_metrics.get(metric, np.nan)

    print("\n--- Cross-Validation Summary (based on true best AUC epochs) ---")
    for key, value in results_summary.items():
        print(f"  {key}: {value:.4f}")

    return results_summary, df_fold_metrics, best_model_state_fold


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

    # --- 1. Model, Optimizer, and Loss Initialization ---
    model = GVAE(
        view_configs=model_config['view_configs'],
        radiology_aggregator_config=model_config['radiology_aggregator_config'],
        projection_head_config=model_config['projection_head_config'],
        fusion_config=model_config['fusion_config'],
        classifier_config=model_config['classifier_config'],
        d_embed=model_config['d_embed'],
        missing_strategy=model_config.get('missing_strategy', 'zero')
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=train_config.get('patience', 10),
    )

    criterion_bce_logits = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    # Lấy các trọng số loss từ config
    loss_weights = train_config['loss_weights']
    anneal_config = train_config.get('annealing', {})

    base_w_class = loss_weights['class']
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

    # --- 2. Training Loop Initialization ---
    best_val_auc = -1.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, train_config['epochs'] + 1):
        # --- Training Phase ---
        model.train()

        w_kl = linear_anneal(epoch, kl_start_e, kl_end_e,
                             kl_start_w, base_w_kl)
        w_cl = linear_anneal(epoch, cl_start_e, cl_end_e,
                             cl_start_w, base_w_cross_cl)

        logits, vae_out, cl_out, _ = model(full_multi_view_data, train_indices)
        labels = full_multi_view_data['patient']['binary_label'].to(device)[train_indices]

        # Calculate all loss components for training
        loss_class = criterion_bce_logits(logits.squeeze(), labels.float())
        loss_cl = calculate_contrastive_loss(
            cl_out, train_config['cross_cl_temp'])

        rec_attr, rec_struct, kl_div, active_views = 0.0, 0.0, 0.0, 0
        for view, vo in vae_out.items():
            if vo and vo.get('mu') is not None:
                active_views += 1
                w_attr = w_rec_attr_config.get(view, 1.0) if isinstance(
                    w_rec_attr_config, dict) else w_rec_attr_config
                rec_attr += w_attr * \
                    criterion_mse(vo['rec_x'], vo['original_x_subset'])

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

        if not torch.isnan(total_train_loss):
            optimizer.zero_grad()
            total_train_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), train_config['grad_clip_norm'])
            optimizer.step()

        # --- Validation Phase ---
        model.eval()
        current_val_auc = -1.0
        total_val_loss_val = torch.tensor(float('inf'), device=device)

        if val_indices.numel() > 0:
            with torch.no_grad():
                val_logits, val_vae_out, val_cl_out, _ = model(
                    full_multi_view_data, val_indices)
                val_labels = full_multi_view_data['patient']['binary_label'].to(device)[val_indices]

                # Calculate validation loss for scheduler/early stopping
                # Using final annealed weights for a stable target
                val_loss_class = criterion_bce_logits(
                    val_logits.squeeze(), val_labels.float())
                val_loss_cl = calculate_contrastive_loss(
                    val_cl_out, train_config['cross_cl_temp'])

                val_rec_attr, val_rec_struct, val_kl, val_active_views = 0.0, 0.0, 0.0, 0
                for view, vo in val_vae_out.items():
                    if vo and vo.get('mu') is not None:
                        val_active_views += 1
                        w_attr = w_rec_attr_config.get(view, 1.0) if isinstance(
                            w_rec_attr_config, dict) else w_rec_attr_config
                        val_rec_attr += w_attr * \
                            criterion_mse(vo['rec_x'], vo['original_x_subset'])
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
                        val_logits.squeeze()).cpu().numpy()
                    current_val_auc = roc_auc_score(
                        val_labels.cpu().numpy(), val_probs)

        # --- Logging and Checkpointing ---
        if epoch % train_config.get('print_every_k_epochs', 10) == 0:
            print(f"  F{fold_num} Ep{epoch:03d} | TrainLoss:{total_train_loss.item():.4f} | ValLoss:{total_val_loss_val.item():.4f} | ValAUC:{current_val_auc:.4f} (BestAUC: {best_val_auc:.4f})")

        # 1. Scheduler and Early Stopping (based on validation loss)
        scheduler.step(total_val_loss_val)
        if total_val_loss_val < best_val_loss:
            best_val_loss = total_val_loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 2. Best Model Saving (based on validation AUC)
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_model_state = model.state_dict().copy()
            # print(f"  F{fold_num} Ep{epoch:03d} -> New best model found with ValAUC: {best_val_auc:.4f}")

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
            f"Warning: No best model was saved for fold {fold_num} based on AUC improvement.")

    return best_model_state

def train_gvae_and_save_best_state(
    full_multi_view_data: HeteroData,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    save_path: str = "gvae_final_best.pth"
) -> Optional[str]:
    """
    Trains a final GVAE model on a given training set, validates on a validation set,
    and saves the state of the model with the best validation AUC to a file.
    This version uses Neighbor Sampling for efficient mini-batching.

    Args:
        full_multi_view_data (HeteroData): The complete dataset.
        train_indices (torch.Tensor): Tensor of global indices for the training set.
        val_indices (torch.Tensor): Tensor of global indices for the validation set.
        model_config (Dict[str, Any]): Configuration for the GVAE model architecture.
        train_config (Dict[str, Any]): Configuration for the training process.
        save_path (str): The file path to save the best model state.

    Returns:
        Optional[str]: The path to the saved model file if successful, otherwise None.
    """
    device = train_config['device']

    # --- 1. Model, Optimizer, Loss Initialization ---
    model = GVAE(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=train_config.get('patience', 10), verbose=True
    )

    criterion_bce_logits = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    # --- 2. DataLoaders with Neighbor Sampling ---
    print(f"Creating training loader with {len(train_indices)} seed nodes...")
    train_loader = NeighborLoader(
        full_multi_view_data,
        num_neighbors=[15, 10],
        input_nodes=('patient', train_indices),
        batch_size=train_config.get('gvae_batch_size', 64),
        shuffle=True,
        num_workers=0
    )
    
    print(f"Creating validation loader with {len(val_indices)} seed nodes...")
    val_loader = NeighborLoader(
        full_multi_view_data,
        num_neighbors=[-1, -1], # Use all neighbors for validation
        input_nodes=('patient', val_indices),
        batch_size=train_config.get('gvae_batch_size', 64),
        shuffle=False,
        num_workers=0
    )

    # --- 3. Training Loop ---
    best_val_auc = -1.0
    epochs_no_improve = 0
    
    loss_weights = train_config['loss_weights']
    anneal_config = train_config.get('annealing', {})
    base_w_class, base_w_cross_cl, base_w_kl = loss_weights['class'], loss_weights.get('cross_cl', 0.0), loss_weights['kl']
    kl_params, cl_params = anneal_config.get('kl', {}), anneal_config.get('cross_cl', {})
    kl_start_w, kl_start_e, kl_end_e = kl_params.get('start_weight', base_w_kl), kl_params.get('start_epoch', 0), kl_params.get('end_epoch', 0)
    cl_start_w, cl_start_e, cl_end_e = cl_params.get('start_weight', base_w_cross_cl), cl_params.get('start_epoch', 0), cl_params.get('end_epoch', 0)
    w_rec_attr_config = loss_weights['rec_attr']
    w_rec_struct_config = loss_weights['rec_struct']

    for epoch in range(1, train_config['epochs'] + 1):
        model.train()
        total_train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            seed_patient_indices = batch['patient'].input_id

            w_kl = linear_anneal(epoch, kl_start_e, kl_end_e, kl_start_w, base_w_kl)
            w_cl = linear_anneal(epoch, cl_start_e, cl_end_e, cl_start_w, base_w_cross_cl)

            logits, vae_out, cl_out, _ = model(batch, seed_patient_indices)
            labels = batch['patient'].binary_label[:batch['patient'].batch_size]
            
            loss_class = criterion_bce_logits(logits.squeeze(), labels.float())
            loss_cl = calculate_contrastive_loss(cl_out, train_config['cross_cl_temp'])
            
            rec_attr, rec_struct, kl_div, active_views = 0.0, 0.0, 0.0, 0
            for view, vo in vae_out.items():
                if vo and vo.get('mu') is not None:
                    active_views += 1
                    w_attr = w_rec_attr_config.get(view, 1.0) if isinstance(w_rec_attr_config, dict) else w_rec_attr_config
                    rec_attr += w_attr * criterion_mse(vo['rec_x'], vo['original_x_subset'])
                    w_struct = w_rec_struct_config.get(view, 1.0) if isinstance(w_rec_struct_config, dict) else w_rec_struct_config
                    rec_struct += w_struct * criterion_bce_logits(vo['rec_adj_logits'].flatten(), vo['original_adj_subset'].flatten())
                    kl_div += -0.5 * torch.sum(1 + vo['logvar'] - vo['mu'].pow(2) - vo['logvar'].exp(), dim=1).mean()
            
            avg_rec_attr = rec_attr / active_views if active_views > 0 else 0.0
            avg_rec_struct = rec_struct / active_views if active_views > 0 else 0.0
            avg_kl = kl_div / active_views if active_views > 0 else 0.0

            total_loss = (base_w_class * loss_class + w_cl * loss_cl +
                          avg_rec_attr + avg_rec_struct + w_kl * avg_kl)

            if not torch.isnan(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip_norm'])
                optimizer.step()
                total_train_loss += total_loss.item()
                pbar.set_postfix(loss=total_loss.item())

        # --- Validation Phase ---
        model.eval()
        all_val_logits, all_val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]", leave=False):
                batch = batch.to(device)
                seed_patient_indices = batch['patient'].input_id
                logits, _, _, _ = model(batch, seed_patient_indices)
                labels = batch['patient'].binary_label[:batch['patient'].batch_size]
                all_val_logits.append(logits)
                all_val_labels.append(labels)
        
        val_logits_tensor = torch.cat(all_val_logits)
        val_labels_tensor = torch.cat(all_val_labels)
        
        current_val_auc = -1.0
        if not torch.isnan(val_logits_tensor).any() and len(torch.unique(val_labels_tensor)) > 1:
            val_probs = torch.sigmoid(val_logits_tensor.squeeze()).cpu().numpy()
            current_val_auc = roc_auc_score(val_labels_tensor.cpu().numpy(), val_probs)

        print(f"  Epoch {epoch:03d} | AvgTrainLoss: {(total_train_loss / len(train_loader)):.4f} | ValAUC: {current_val_auc:.4f} (Best: {best_val_auc:.4f})")

        scheduler.step(current_val_auc)

        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            try:
                torch.save(model.state_dict(), save_path)
                print(f"  -> New best model saved to '{save_path}' with ValAUC: {best_val_auc:.4f}")
            except Exception as e:
                print(f"  ERROR: Could not save model. {e}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= train_config.get('patience_early_stopping', 20):
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    del model, optimizer, scheduler, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
    if best_val_auc > -1.0:
        return save_path
    else:
        print("Warning: Training finished, but no best model was saved based on AUC improvement.")
        return None