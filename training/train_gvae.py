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
from utils.loss_utils import calculate_contrastive_loss, calculate_contrastive_loss_vectorized
from tqdm.notebook import tqdm
import torch.optim as optim
from models.gvae_components import StandaloneRadiologyMIL
import copy
from sklearn.metrics import precision_recall_curve

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
    
    # Calculate pos_weight for clinical binary features (imbalance handling)
    clinical_x_cpu = full_multi_view_data_cpu['patient'].x_clinical    
    # Indices 5 to 22 correspond to the binary features as defined above
    clinical_bin_feats = clinical_x_cpu[:, 5:22]
    n_pos = clinical_bin_feats.sum(dim=0)
    n_neg = clinical_bin_feats.shape[0] - n_pos
    # Weight = Neg / Pos
    clinical_bin_pos_weight = n_neg / (n_pos + 1e-6)
    clinical_bin_pos_weight = clinical_bin_pos_weight.to(device)

    # --- Main Task Loss Setup ---
    labels = full_multi_view_data_cpu['patient']['binary_label'].numpy()
    # Add epsilon to avoid division by zero if classes are imbalanced perfectly
    pos_weight_value = np.sum(labels == 0) / (np.sum(labels == 1) + 1e-6)
    print(f"INFO: Using pos_weight for Main Task BCE loss: {pos_weight_value:.2f}")
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    
    criterion_main_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_mse = nn.MSELoss()
    
    # --- Config Unpacking ---
    loss_weights_config = train_config['loss_weights']
    anneal_config = train_config.get('annealing', {})
    base_w_class = loss_weights_config['class']
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
        loss_at_best_auc = float('inf')
        epoch_at_best_auc = 0
        epochs_no_improve_loss = 0
        best_model_state = None
        best_epoch_results = {}

        # Fallback tracking (based on lowest val loss)
        fallback_model_state = None
        fallback_epoch_results = {}
        fallback_val_loss = float('inf')
        fallback_val_auc = -1.0
        fallback_epoch = 0

        # --- Mini-batch setup ---
        batch_size = train_config.get('batch_size', None)
        if batch_size is not None and batch_size > 0 and batch_size < len(train_indices):
            def make_batches(indices):
                return torch.split(indices, batch_size)
        else:
            def make_batches(indices):
                return [indices]

        for epoch in range(1, train_config['epochs'] + 1):
            # =================== TRAINING PHASE ===================
            model.train()
            w_kl = linear_anneal(epoch, 0, kl_end_e, kl_start_w, base_w_kl)
            w_cl = linear_anneal(epoch, 0, cl_end_e, cl_start_w, base_w_cross_cl)
            
            train_batches = make_batches(train_indices)
            batch_train_losses = []
            
            for batch_idx in train_batches:
                logits, vae_out, cl_out, _ = model(fold_data, batch_idx)
                labels = fold_data['patient'].binary_label[batch_idx]
                
                # 1. Main Classification Loss
                loss_class = criterion_main_bce(logits.squeeze(), labels.float())
                
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

            # Guard against an all-skipped epoch (np.mean([]) -> nan + warning).
            if batch_train_losses:
                total_train_loss = torch.tensor(np.mean(batch_train_losses), device=device)
            else:
                total_train_loss = torch.tensor(float('inf'), device=device)

            # =================== VALIDATION PHASE ===================
            model.eval()
            with torch.no_grad():
                val_logits, val_vae_out, val_cl_out, _ = model(fold_data, val_indices)
                val_labels = fold_data['patient'].binary_label[val_indices]
                
                # 1. Classification Loss
                val_loss_class = criterion_main_bce(val_logits.squeeze(), val_labels.float())
                
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
                val_probs_np = torch.sigmoid(val_logits.squeeze()).cpu().numpy()
                val_labels_np = val_labels.cpu().numpy()
                if len(np.unique(val_labels_np)) > 1:
                    current_val_auc = roc_auc_score(val_labels_np, val_probs_np)
            
            # --- Checkpointing & Early Stopping ---
            scheduler.step(total_val_loss)
            
            # Track validation loss for early stopping patience
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                epochs_no_improve_loss = 0
            else:
                epochs_no_improve_loss += 1

            # Track fallback (lowest validation loss)
            if total_val_loss < fallback_val_loss:
                fallback_val_loss = total_val_loss
                fallback_val_auc = current_val_auc
                fallback_epoch = epoch
                fallback_model_state = copy.deepcopy(model.state_dict())
                
                fpr_f, tpr_f = np.array([np.nan]), np.array([np.nan])
                if current_val_auc != -1.0:
                    fpr_f, tpr_f, _ = roc_curve(val_labels_np, val_probs_np)
                fallback_epoch_results = {
                    'fpr': fpr_f, 'tpr': tpr_f, 'auc': current_val_auc,
                    'y_true': val_labels_np, 'y_pred_probs': val_probs_np
                }

            # Track validation AUC for checkpointing (saving the best AUC model)
            if current_val_auc != -1.0 and current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                loss_at_best_auc = total_val_loss.item()
                epoch_at_best_auc = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                
                fpr, tpr = np.array([np.nan]), np.array([np.nan])
                fpr, tpr, _ = roc_curve(val_labels_np, val_probs_np)

                best_epoch_results = {
                    'fpr': fpr, 'tpr': tpr, 'auc': current_val_auc,
                    'y_true': val_labels_np, 'y_pred_probs': val_probs_np
                }
                print(f"  [Fold {fold+1}] Epoch {epoch:03d} -> New best Val AUC: {best_val_auc:.4f} at Val Loss: {loss_at_best_auc:.4f}")

            if epochs_no_improve_loss >= train_config.get('patience_early_stopping', 30):
                print(f"  Early stopping at epoch {epoch}.")
                break
            
            if epoch % 10 == 0:
                 print(f"  Epoch {epoch:03d} | TLoss: {total_train_loss.item():.4f} | VLoss: {total_val_loss.item():.4f} | VAUC: {current_val_auc:.4f} | Best AUC: {best_val_auc:.4f} at Loss: {loss_at_best_auc:.4f}")
        
        # If no model was saved via AUC improvement, use the lowest val loss fallback
        if best_model_state is None or not best_epoch_results:
            print(f"  [Fold {fold+1}] No model saved based on AUC improvement. Falling back to lowest Val Loss model.")
            best_model_state = fallback_model_state
            best_epoch_results = fallback_epoch_results
            best_val_auc = fallback_val_auc
            loss_at_best_auc = fallback_val_loss.item() if isinstance(fallback_val_loss, torch.Tensor) else fallback_val_loss
            epoch_at_best_auc = fallback_epoch

        print(f"  [Fold {fold+1}] Best Val AUC: {best_val_auc:.4f} at Val Loss: {loss_at_best_auc:.4f} (Epoch {epoch_at_best_auc})")

        # --- Save Results ---
        if best_epoch_results:
            all_roc_data.append(best_epoch_results)
            y_true = best_epoch_results['y_true']
            y_pred_probs = best_epoch_results['y_pred_probs']
            
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
            f1_scores = np.divide(2 * recall[:-1] * precision[:-1], recall[:-1] + precision[:-1], out=np.zeros_like(recall[:-1]), where=(recall[:-1] + precision[:-1])!=0)
            best_threshold = thresholds[np.argmax(f1_scores)] if len(f1_scores) > 0 else 0.5
            y_pred_binary_tuned = (y_pred_probs > best_threshold).astype(int)
            
            fold_metrics_list.append({
                'auc': best_epoch_results['auc'],
                'f1': f1_score(y_true, y_pred_binary_tuned, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred_binary_tuned),
                'precision': precision_score(y_true, y_pred_binary_tuned, zero_division=0),
                'recall': recall_score(y_true, y_pred_binary_tuned, zero_division=0),
                'best_threshold': best_threshold,
                'loss_at_best_auc': loss_at_best_auc,
                'epoch_at_best_auc': epoch_at_best_auc
            })
            
            if train_config.get('save_best_fold_model', True):
                torch.save(best_model_state, f"best_model_fold_{fold+1}.pth")

        del model, optimizer, scheduler, fold_data
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Summary ---
    if not fold_metrics_list: return {}, pd.DataFrame(), []
    df_fold_metrics = pd.DataFrame(fold_metrics_list)
    results_summary = {f"{agg}_{m}": getattr(df_fold_metrics[m], agg)() for m in ['auc', 'f1', 'accuracy', 'loss_at_best_auc'] for agg in ['mean', 'std']}
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
            device
        )

    # --- 1. Model, Optimizer, and Loss Initialization ---
    model = GVAE(
        view_configs=model_config['view_configs'],
        radiology_aggregator_config=model_config['radiology_aggregator_config'],
        projection_head_config=model_config['projection_head_config'],
        fusion_config=model_config['fusion_config'],
        classifier_config=model_config['classifier_config'],
        d_embed=model_config['d_embed'],
        missing_strategy=model_config.get('missing_strategy', 'learnable'),
        logvar_clamp=model_config.get('logvar_clamp', None),
        radiology_zero_lesion_passthrough=model_config.get('radiology_zero_lesion_passthrough', False)
    ).to(device)

    if radiology_state_dict is not None:
        print(f"   [Fold {fold_num}] Loading pre-trained radiology aggregator weights...")
        model.radiology_lesion_aggregator.load_state_dict(radiology_state_dict)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=train_config.get('patience', 10),
    )

    labels_all = full_multi_view_data['patient']['binary_label']
    pos_weight_value = (labels_all == 0).sum().float() / ((labels_all == 1).sum().float() + 1e-6)
    criterion_bce_logits = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=device))
    criterion_mse = nn.MSELoss()

    clinical_cont_idx = torch.arange(0, 5, device=device)
    clinical_bin_idx = torch.arange(5, 22, device=device)
    clinical_bin_feats = full_multi_view_data['patient'].x_clinical[:, 5:22]
    n_pos_bin = clinical_bin_feats.sum(dim=0)
    n_neg_bin = clinical_bin_feats.shape[0] - n_pos_bin
    clinical_bin_pos_weight = (n_neg_bin / (n_pos_bin + 1e-6)).to(device)

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

    use_vec_cl_sf = train_config.get('vectorized_contrastive', False)
    contrastive_fn_sf = (calculate_contrastive_loss_vectorized
                         if use_vec_cl_sf else calculate_contrastive_loss)

    # --- 2. Training Loop Initialization ---
    best_val_auc = -1.0
    loss_at_best_auc = float('inf')
    epoch_at_best_auc = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # Fallback tracking (based on lowest val loss)
    fallback_model_state = None
    fallback_val_loss = float('inf')
    fallback_val_auc = -1.0
    fallback_epoch = 0

    # --- Mini-batch setup ---
    batch_size = train_config.get('batch_size', None)
    if batch_size is not None and batch_size > 0 and batch_size < len(train_indices):
        def make_batches(indices):
            return torch.split(indices, batch_size)
    else:
        def make_batches(indices):
            return [indices]

    for epoch in range(1, train_config['epochs'] + 1):
        # --- Training Phase ---
        model.train()

        w_kl = linear_anneal(epoch, kl_start_e, kl_end_e,
                             kl_start_w, base_w_kl)
        w_cl = linear_anneal(epoch, cl_start_e, cl_end_e,
                             cl_start_w, base_w_cross_cl)

        train_batches = make_batches(train_indices)
        batch_train_losses = []

        for batch_idx in train_batches:
            logits, vae_out, cl_out, _ = model(full_multi_view_data, batch_idx)
            labels = full_multi_view_data['patient']['binary_label'].to(device)[batch_idx]

            # Calculate all loss components for training
            loss_class = criterion_bce_logits(logits.squeeze(), labels.float())
            loss_cl = contrastive_fn_sf(
                cl_out, train_config['cross_cl_temp'])

            rec_attr, rec_struct, kl_div, active_views = 0.0, 0.0, 0.0, 0
            for view, vo in vae_out.items():
                if vo and vo.get('mu') is not None:
                    active_views += 1
                    w_attr = w_rec_attr_config.get(view, 1.0) if isinstance(
                        w_rec_attr_config, dict) else w_rec_attr_config
                    if view == 'clinical':
                        loss_cont = criterion_mse(
                            vo['rec_x'][:, clinical_cont_idx],
                            vo['original_x_subset'][:, clinical_cont_idx])
                        loss_bin = F.binary_cross_entropy_with_logits(
                            vo['rec_x'][:, clinical_bin_idx],
                            vo['original_x_subset'][:, clinical_bin_idx],
                            pos_weight=clinical_bin_pos_weight)
                        rec_attr += w_attr * (loss_cont + loss_bin)
                    else:
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
            
            batch_train_losses.append(total_train_loss.item())
        
        total_train_loss = torch.tensor(np.mean(batch_train_losses), device=device)

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
                val_loss_cl = contrastive_fn_sf(
                    val_cl_out, train_config['cross_cl_temp'])

                val_rec_attr, val_rec_struct, val_kl, val_active_views = 0.0, 0.0, 0.0, 0
                for view, vo in val_vae_out.items():
                    if vo and vo.get('mu') is not None:
                        val_active_views += 1
                        w_attr = w_rec_attr_config.get(view, 1.0) if isinstance(
                            w_rec_attr_config, dict) else w_rec_attr_config
                        if view == 'clinical':
                            val_loss_cont = criterion_mse(
                                vo['rec_x'][:, clinical_cont_idx],
                                vo['original_x_subset'][:, clinical_cont_idx])
                            val_loss_bin = F.binary_cross_entropy_with_logits(
                                vo['rec_x'][:, clinical_bin_idx],
                                vo['original_x_subset'][:, clinical_bin_idx],
                                pos_weight=clinical_bin_pos_weight)
                            val_rec_attr += w_attr * (val_loss_cont + val_loss_bin)
                        else:
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

        # 1. Scheduler and Early Stopping (based on validation loss)
        scheduler.step(total_val_loss_val)
        if total_val_loss_val < best_val_loss:
            best_val_loss = total_val_loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Track fallback (lowest validation loss)
        if total_val_loss_val < fallback_val_loss:
            fallback_val_loss = total_val_loss_val
            fallback_val_auc = current_val_auc
            fallback_epoch = epoch
            fallback_model_state = model.state_dict().copy()

        # 2. Best Model Saving (based on validation AUC)
        if current_val_auc != -1.0 and current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            loss_at_best_auc = total_val_loss_val.item() if isinstance(total_val_loss_val, torch.Tensor) else total_val_loss_val
            epoch_at_best_auc = epoch
            best_model_state = model.state_dict().copy()
            # print(f"  F{fold_num} Ep{epoch:03d} -> New best model found with ValAUC: {best_val_auc:.4f} at ValLoss: {loss_at_best_auc:.4f}")

        # --- Logging and Checkpointing ---
        if epoch % train_config.get('print_every_k_epochs', 10) == 0:
            print(f"  F{fold_num} Ep{epoch:03d} | TrainLoss:{total_train_loss.item():.4f} | ValLoss:{total_val_loss_val.item():.4f} | ValAUC:{current_val_auc:.4f} (BestAUC: {best_val_auc:.4f} at ValLoss: {loss_at_best_auc:.4f})")

        if epochs_no_improve >= train_config.get('patience_early_stopping', 20):
            print(
                f"  Early stopping triggered at epoch {epoch} for fold {fold_num}.")
            break

    # If no model was saved via AUC improvement, fallback to the lowest val loss model
    if best_model_state is None:
        print(f"  F{fold_num} -> No model saved based on AUC improvement. Falling back to lowest Val Loss model.")
        best_model_state = fallback_model_state
        best_val_auc = fallback_val_auc
        loss_at_best_auc = fallback_val_loss.item() if isinstance(fallback_val_loss, torch.Tensor) else fallback_val_loss
        epoch_at_best_auc = fallback_epoch

    print(f"  F{fold_num} -> Training finished. Best ValAUC: {best_val_auc:.4f} at ValLoss: {loss_at_best_auc:.4f} (Epoch {epoch_at_best_auc})")

    # --- Final Cleanup for the Fold ---
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

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


# =============================================================================
# Leakage-free classifier evaluation
# -----------------------------------------------------------------------------
# kfold_evaluate_gvae_classifier() is a stricter alternative to kfold_train_gvae()
# for *reporting* the response-classifier's performance. It fixes three biases
# baked into the kfold_train_gvae reporting path:
#
#   1. Selection != reporting. Each outer fold is held out purely for reporting.
#      Checkpoint selection + early stopping use an INNER validation split carved
#      from that fold's training portion, so the reported number never sees the
#      epoch picker.
#   2. No leakage in selection. The checkpoint is chosen by a single pre-committed
#      rule on INNER-val ('inner_val_auc' by default, or 'inner_val_loss'), after a
#      small warmup (select_after_epoch) that skips the chaotic first epochs. Because
#      selection is on inner-val and reporting is on the untouched outer fold, even an
#      early-peaking checkpoint is judged honestly by the OOF report.
#
#      NOTE on the metric: in this pipeline the val LOSS is dominated by reconstruction
#      and keeps falling long after the latent's DISCRIMINATIVE quality has peaked and
#      started to decay (the two objectives diverge). So 'inner_val_auc' is the default
#      and the right choice when you want a discriminative latent (for the classifier
#      AND for conditioning the DDPM). Use 'inner_val_loss' only if you specifically
#      want the converged/most-regularised latent (e.g. with a strong KL weight + long
#      annealing, in which case set select_after_epoch to the annealing end).
#   3. Pooled out-of-fold (OOF) reporting. Instead of averaging ten noisy
#      ~25-patient fold AUCs, every patient's held-out prediction is pooled into
#      one length-N vector and scored once, with bootstrap CIs.
#
# It reports BOTH the end-to-end fusion head AND a logistic-regression linear
# probe on the frozen mu latents (the representation-quality metric that predicts
# whether downstream DDPM/TSTR can work).
# =============================================================================

def _gvae_loss_and_logits(model, fold_data, indices, ctx, w_cl, w_kl):
    """Single forward pass replicating kfold_train_gvae's composite loss.

    Returns (total_loss, logits, labels). `ctx` carries the precomputed config
    (criteria, clinical indices, loss weights, contrastive fn/temp)."""
    logits, vae_out, cl_out, _ = model(fold_data, indices)
    labels = fold_data['patient'].binary_label[indices]

    loss_class = ctx['criterion_main_bce'](logits.squeeze(), labels.float())

    if len(cl_out) > 0:
        loss_cl = ctx['contrastive_fn'](cl_out, ctx['cross_cl_temp'])
    else:
        loss_cl = torch.tensor(0.0, device=indices.device)

    rec_attr, rec_struct, kl_div, active = 0.0, 0.0, 0.0, 0
    for view, vo in vae_out.items():
        if vo and vo.get('mu') is not None:
            active += 1
            target_x, recon_x = vo['original_x_subset'], vo['rec_x']
            view_weight = ctx['w_rec_attr_config'].get(view, 1.0)
            if view == 'clinical':
                loss_cont = ctx['criterion_mse'](recon_x[:, ctx['clinical_cont_idx']],
                                                 target_x[:, ctx['clinical_cont_idx']])
                loss_bin = F.binary_cross_entropy_with_logits(
                    recon_x[:, ctx['clinical_bin_idx']], target_x[:, ctx['clinical_bin_idx']],
                    pos_weight=ctx['clinical_bin_pos_weight'])
                rec_attr += view_weight * (loss_cont + loss_bin)
            else:
                rec_attr += view_weight * ctx['criterion_mse'](recon_x, target_x)
            rec_struct += ctx['w_rec_struct_config'] * F.binary_cross_entropy_with_logits(
                vo['rec_adj_logits'].flatten(), vo['original_adj_subset'].flatten())
            kl_term = 1 + vo['logvar'] - vo['mu'].pow(2) - vo['logvar'].exp()
            kl_div += -0.5 * kl_term.sum(dim=1).mean()

    avg_rec_attr = rec_attr / active if active > 0 else 0.0
    avg_rec_struct = rec_struct / active if active > 0 else 0.0
    avg_kl = kl_div / active if active > 0 else 0.0

    total = (ctx['base_w_class'] * loss_class + w_cl * loss_cl
             + avg_rec_attr + avg_rec_struct + w_kl * avg_kl)
    return total, logits, labels


def _threshold_max_f1(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Threshold that maximises F1 on (y_true, probs). Chosen on inner-val only."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    denom = recall[:-1] + precision[:-1]
    f1 = np.divide(2 * recall[:-1] * precision[:-1], denom,
                   out=np.zeros_like(denom), where=denom != 0)
    return float(thresholds[np.argmax(f1)]) if len(f1) > 0 else 0.5


def _bootstrap_ci(y_true, y_score, metric_fn, n_boot=2000, seed=0):
    """Point estimate on full data + 95% percentile bootstrap CI.
    Skips resamples that collapse to a single class (metric undefined)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    try:
        point = float(metric_fn(y_true, y_score))
    except Exception:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    n, stats = len(y_true), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            stats.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    if not stats:
        return point, float('nan'), float('nan')
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def kfold_evaluate_gvae_classifier(
    full_multi_view_data: HeteroData,
    model_config_original: Dict,
    train_config: Dict,
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, np.ndarray]]:
    """Leakage-free, pooled-OOF evaluation of the GVAE response classifier.

    Differs from kfold_train_gvae: outer folds are report-only; an inner
    validation split (carved from each fold's train portion) drives early
    stopping + checkpoint selection; the checkpoint is picked by one pre-committed
    rule on inner-val; metrics are pooled across all patients with bootstrap CIs;
    and a linear probe on frozen mu latents is reported alongside the fusion head.

    Extra train_config keys (all optional):
      inner_val_split    : float, fraction of each fold's train set held out for
                           selection/early-stop. Default 0.2 (stratified).
      selection_metric   : 'inner_val_auc' (default) or 'inner_val_loss'. AUC is the
                           default because the composite loss is reconstruction-dominated
                           and diverges from discriminative latent quality (which peaks
                           early). Patience-based early stopping runs on this metric.
      select_after_epoch : warmup floor before a checkpoint becomes selectable.
                           Default min(20, epochs//10). Raise to the annealing end only
                           if you train with a strong KL weight + long annealing.
      n_bootstrap        : bootstrap resamples for CIs. Default 2000.
      eval_seed          : seed for splits + bootstrap. Default 4200.

    Returns (summary_dict, per_fold_df, oof_arrays). oof_arrays has keys
    'y_true', 'head_probs', 'probe_probs' (each length N, ordered by patient id).
    """
    from sklearn.metrics import average_precision_score, brier_score_loss, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from models.gvae_model import get_separate_view_mus

    device = train_config['device']
    full_cpu = full_multi_view_data.clone().cpu()

    # --- Loss context (mirrors kfold_train_gvae setup) ---
    clinical_cont_idx = torch.arange(0, 5, device=device)
    clinical_bin_idx = torch.arange(5, 22, device=device)
    clinical_bin_feats = full_cpu['patient'].x_clinical[:, 5:22]
    n_pos = clinical_bin_feats.sum(dim=0)
    n_neg = clinical_bin_feats.shape[0] - n_pos
    clinical_bin_pos_weight = (n_neg / (n_pos + 1e-6)).to(device)

    labels_all = full_cpu['patient']['binary_label'].numpy()
    pos_weight_value = np.sum(labels_all == 0) / (np.sum(labels_all == 1) + 1e-6)
    print(f"INFO: Using pos_weight for Main Task BCE loss: {pos_weight_value:.2f}")

    loss_weights = train_config['loss_weights']
    anneal_config = train_config.get('annealing', {})
    base_w_class = loss_weights['class']
    base_w_cross_cl = loss_weights.get('cross_cl', 0.0)
    base_w_kl = loss_weights['kl']
    kl_params, cl_params = anneal_config.get('kl', {}), anneal_config.get('cross_cl', {})
    kl_start_w, kl_end_e = kl_params.get('start_weight', base_w_kl), kl_params.get('end_epoch', 0)
    cl_start_w, cl_end_e = cl_params.get('start_weight', base_w_cross_cl), cl_params.get('end_epoch', 0)
    # Warmup floor for checkpoint selection: skip the chaotic first epochs, then pick the
    # best inner-val checkpoint. Defaults to ~10% of epochs (capped at 20). Do NOT default
    # this to the annealing end: with a negligible KL weight the latent's discriminative
    # quality peaks early and decays, so a late floor would freeze the worst checkpoint.
    # Raise it to the annealing end only if you use a strong KL weight + long annealing.
    default_warmup = min(20, max(1, train_config['epochs'] // 10))
    select_after_epoch = train_config.get('select_after_epoch', default_warmup)
    select_after_epoch = max(0, min(select_after_epoch, train_config['epochs'] - 1))

    use_vec_cl = train_config.get('vectorized_contrastive', False)
    ctx = {
        'criterion_main_bce': nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device)),
        'criterion_mse': nn.MSELoss(),
        'clinical_cont_idx': clinical_cont_idx, 'clinical_bin_idx': clinical_bin_idx,
        'clinical_bin_pos_weight': clinical_bin_pos_weight,
        'w_rec_attr_config': loss_weights['rec_attr'], 'w_rec_struct_config': loss_weights['rec_struct'],
        'base_w_class': base_w_class,
        'contrastive_fn': calculate_contrastive_loss_vectorized if use_vec_cl else calculate_contrastive_loss,
        'cross_cl_temp': train_config['cross_cl_temp'],
    }

    # --- Eval config ---
    inner_val_split = train_config.get('inner_val_split', 0.2)
    selection_metric = train_config.get('selection_metric', 'inner_val_auc')
    n_bootstrap = train_config.get('n_bootstrap', 2000)
    eval_seed = train_config.get('eval_seed', 4200)
    batch_size = train_config.get('batch_size', None)

    def make_batches(indices):
        if batch_size and 0 < batch_size < len(indices):
            return torch.split(indices, batch_size)
        return [indices]

    # --- OOF buffers (indexed by global patient id) ---
    N = full_multi_view_data['patient'].num_nodes
    oof_head = np.full(N, np.nan)
    oof_probe = np.full(N, np.nan)
    oof_true = labels_all.astype(int)

    all_idx = np.arange(N)
    y_strat = full_multi_view_data['patient'].binary_label.cpu().numpy()
    kf = StratifiedKFold(n_splits=train_config['n_splits'], shuffle=True,
                         random_state=train_config.get('random_seed', 4200))

    per_fold = []
    direction = "max" if selection_metric != 'inner_val_loss' else "min"
    print(f"\nSelection rule: {direction} '{selection_metric}' on inner-val, after warmup "
          f"epoch {select_after_epoch} (patience early-stop runs on this metric).")

    for fold, (train_idx_np, test_idx_np) in enumerate(kf.split(all_idx, y_strat)):
        print(f"\n===== Outer Fold {fold+1}/{train_config['n_splits']} (report-only test) =====")

        # Inner split: selection/early-stop set carved from THIS fold's train portion.
        inner_tr_np, inner_val_np = train_test_split(
            train_idx_np, test_size=inner_val_split,
            stratify=y_strat[train_idx_np], random_state=eval_seed)

        fold_data = full_cpu.to(device)
        inner_tr = torch.tensor(inner_tr_np, device=device)
        inner_val = torch.tensor(inner_val_np, device=device)
        test_idx = torch.tensor(test_idx_np, device=device)

        # Pretrain radiology aggregator on INNER-TRAIN only (no leakage from val/test).
        model_config = copy.deepcopy(model_config_original)
        radiology_state = None
        if 'radiology' in model_config['view_configs']:
            radiology_state = pretrain_radiology_aggregator(
                fold_data, inner_tr, model_config['radiology_aggregator_config'], device,
                epochs=train_config.get('pretrain_epochs', 400),
                use_pos_weight=train_config.get('pretrain_use_pos_weight', False),
                pretrain_val_split=train_config.get('pretrain_val_split', 0.0),
                patience=train_config.get('pretrain_patience', 30),
                seed=train_config.get('pretrain_seed', 42))

        model = GVAE(**model_config).to(device)
        if radiology_state is not None:
            model.radiology_lesion_aggregator.load_state_dict(radiology_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)

        higher_better = (selection_metric != 'inner_val_loss')
        best_sel = -1.0 if higher_better else float('inf')
        best_state, best_epoch = None, 0
        best_loss_state, best_loss = None, float('inf')  # fallback if window never reached
        no_improve_sel = 0  # patience counter, scoped to the selection window
        patience = train_config.get('patience_early_stopping', 30)

        for epoch in range(1, train_config['epochs'] + 1):
            w_kl = linear_anneal(epoch, 0, kl_end_e, kl_start_w, base_w_kl)
            w_cl = linear_anneal(epoch, 0, cl_end_e, cl_start_w, base_w_cross_cl)

            model.train()
            for b in make_batches(inner_tr):
                loss, _, _ = _gvae_loss_and_logits(model, fold_data, b, ctx, w_cl, w_kl)
                if not torch.isfinite(loss):
                    optimizer.zero_grad(); continue
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.get('grad_clip_norm', 1.0))
                optimizer.step()

            # Inner-val: selection signals computed at FULL target weights (no annealing).
            model.eval()
            with torch.no_grad():
                v_loss, v_logits, v_labels = _gvae_loss_and_logits(
                    model, fold_data, inner_val, ctx, base_w_cross_cl, base_w_kl)
            v_loss = v_loss.item()
            v_probs = torch.sigmoid(v_logits.squeeze()).cpu().numpy()
            v_true = v_labels.cpu().numpy()
            v_auc = roc_auc_score(v_true, v_probs) if len(np.unique(v_true)) > 1 else -1.0

            scheduler.step(v_loss)
            # Track lowest-loss model globally, purely as a fallback if the window is
            # never reached (epochs set too small relative to select_after_epoch).
            if v_loss < best_loss:
                best_loss, best_loss_state = v_loss, copy.deepcopy(model.state_dict())

            # Checkpoint selection + patience are both scoped to the post-ramp window,
            # so early stopping can never fire before we have an eligible checkpoint.
            if epoch >= select_after_epoch:
                cur = v_loss if selection_metric == 'inner_val_loss' else v_auc
                improved = (cur < best_sel) if selection_metric == 'inner_val_loss' else (cur > best_sel)
                if improved:
                    best_sel, best_state, best_epoch = cur, copy.deepcopy(model.state_dict()), epoch
                    no_improve_sel = 0
                else:
                    no_improve_sel += 1

                if no_improve_sel >= patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break
            if epoch % 10 == 0:
                best_str = f"{best_sel:.4f}" if best_state is not None else "pending"
                print(f"  Epoch {epoch:03d} | inner v_loss={v_loss:.4f} v_auc={v_auc:.4f} | "
                      f"best[{selection_metric}]={best_str} @ep{best_epoch} (window opens ep{select_after_epoch})")

        # Fallback: only if the selection window was never reached (e.g. epochs too small).
        if best_state is None:
            print(f"  [Fold {fold+1}] Selection window never reached; using lowest inner-val-loss model.")
            best_state, best_epoch = best_loss_state, -1
        model.load_state_dict(best_state)
        model.eval()

        # === Threshold on INNER-VAL (never on test) ===
        with torch.no_grad():
            _, iv_logits, iv_labels = _gvae_loss_and_logits(
                model, fold_data, inner_val, ctx, base_w_cross_cl, base_w_kl)
        thr = _threshold_max_f1(iv_labels.cpu().numpy(),
                                torch.sigmoid(iv_logits.squeeze()).cpu().numpy())

        # === Fusion head: predict held-out TEST fold (pooled OOF) ===
        with torch.no_grad():
            t_logits, _, _, _ = model(fold_data, test_idx)
        head_probs = torch.sigmoid(t_logits.squeeze()).cpu().numpy()
        oof_head[test_idx_np] = head_probs

        # === Linear probe: fit on INNER-TRAIN mu, predict TEST mu (pooled OOF) ===
        def concat_mu(idx_tensor):
            mus = get_separate_view_mus(model, fold_data, idx_tensor)
            z = torch.cat([mus[v] for v in model.views], dim=1).cpu().numpy()
            # Guard the probe against a degenerate/non-finite latent dim.
            return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        Z_tr, Z_te = concat_mu(inner_tr), concat_mu(test_idx)
        y_tr = full_multi_view_data['patient'].binary_label[inner_tr].cpu().numpy()
        # Standardise, then clip to bound the linear predictor. Under-regularised GVAE
        # latents can have near-zero-variance or extreme dims that overflow lbfgs; clipping
        # standardised features to +/-10 keeps the probe well-conditioned without distorting
        # the bulk of the distribution (standardised normals rarely exceed +/-5).
        probe_scaler = StandardScaler().fit(Z_tr)
        Z_tr_s = np.clip(probe_scaler.transform(Z_tr), -10.0, 10.0)
        Z_te_s = np.clip(probe_scaler.transform(Z_te), -10.0, 10.0)
        # liblinear (coordinate descent) is robust for small binary problems. The errstate
        # guard silences NumPy's spurious matmul FP warnings (the inputs are bounded to
        # +/-10, so the predictions are valid regardless); nan_to_num is a final safety net.
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            probe = LogisticRegression(max_iter=5000, class_weight='balanced', solver='liblinear')
            probe.fit(Z_tr_s, y_tr)
            probe_probs = probe.predict_proba(Z_te_s)[:, 1]
        probe_probs = np.nan_to_num(probe_probs, nan=0.5, posinf=1.0, neginf=0.0)
        oof_probe[test_idx_np] = probe_probs

        # Per-fold (diagnostic only; headline numbers come from pooled OOF below).
        y_te = oof_true[test_idx_np]
        fold_auc = roc_auc_score(y_te, head_probs) if len(np.unique(y_te)) > 1 else float('nan')
        probe_auc = roc_auc_score(y_te, probe_probs) if len(np.unique(y_te)) > 1 else float('nan')
        head_pred = (head_probs > thr).astype(int)
        per_fold.append({
            'fold': fold + 1, 'selected_epoch': best_epoch, 'threshold': thr,
            'head_auc': fold_auc, 'probe_auc': probe_auc,
            'head_f1': f1_score(y_te, head_pred, zero_division=0),
            'head_recall': recall_score(y_te, head_pred, zero_division=0),
        })
        print(f"  [Fold {fold+1}] selected_epoch={best_epoch} | test head_auc={fold_auc:.4f} "
              f"probe_auc={probe_auc:.4f}")

        del model, optimizer, scheduler, fold_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Pooled out-of-fold metrics with bootstrap CIs ---
    mask = ~np.isnan(oof_head)
    yt = oof_true[mask]
    head, probe = oof_head[mask], oof_probe[mask]
    head_thr = _threshold_max_f1(yt, head)  # for reporting F1/recall at a single pooled threshold

    h_auc = _bootstrap_ci(yt, head, roc_auc_score, n_bootstrap, eval_seed)
    h_ap = _bootstrap_ci(yt, head, average_precision_score, n_bootstrap, eval_seed)
    p_auc = _bootstrap_ci(yt, probe, roc_auc_score, n_bootstrap, eval_seed)
    p_ap = _bootstrap_ci(yt, probe, average_precision_score, n_bootstrap, eval_seed)
    head_pred = (head > head_thr).astype(int)
    prevalence = float(yt.mean())

    summary = {
        'n_pooled': int(mask.sum()), 'prevalence': prevalence,
        'head_auc': h_auc[0], 'head_auc_lo': h_auc[1], 'head_auc_hi': h_auc[2],
        'head_auprc': h_ap[0], 'head_auprc_lo': h_ap[1], 'head_auprc_hi': h_ap[2],
        'head_f1': f1_score(yt, head_pred, zero_division=0),
        'head_recall': recall_score(yt, head_pred, zero_division=0),
        'head_brier': brier_score_loss(yt, head),
        'probe_auc': p_auc[0], 'probe_auc_lo': p_auc[1], 'probe_auc_hi': p_auc[2],
        'probe_auprc': p_ap[0], 'probe_auprc_lo': p_ap[1], 'probe_auprc_hi': p_ap[2],
    }

    print(f"\n{'='*64}\n POOLED OUT-OF-FOLD RESULTS (N={summary['n_pooled']}, prevalence={prevalence:.3f})\n{'='*64}")
    print(f"  Fusion head : AUC   {h_auc[0]:.4f}  [95% CI {h_auc[1]:.4f}-{h_auc[2]:.4f}]")
    print(f"              : AUPRC {h_ap[0]:.4f}  [95% CI {h_ap[1]:.4f}-{h_ap[2]:.4f}]  (no-skill={prevalence:.3f})")
    print(f"              : F1 {summary['head_f1']:.4f} | Recall {summary['head_recall']:.4f} | Brier {summary['head_brier']:.4f}")
    print(f"  Linear probe: AUC   {p_auc[0]:.4f}  [95% CI {p_auc[1]:.4f}-{p_auc[2]:.4f}]")
    print(f"              : AUPRC {p_ap[0]:.4f}  [95% CI {p_ap[1]:.4f}-{p_ap[2]:.4f}]")
    print("  (Probe AUC = representation quality; it upper-bounds what TSTR can sensibly reach.)")

    oof_arrays = {'y_true': oof_true, 'head_probs': oof_head, 'probe_probs': oof_probe}
    return summary, pd.DataFrame(per_fold), oof_arrays
    return results