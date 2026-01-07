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

    # --- Main Task Loss Setup ---
    labels = full_multi_view_data_cpu['patient']['binary_label'].numpy()
    # Add epsilon to avoid division by zero if classes are imbalanced perfectly
    pos_weight_value = np.sum(labels == 0) / (np.sum(labels == 1) + 1e-6)
    print(f"INFO: Using pos_weight for Main Task BCE loss: {pos_weight_value:.2f}")
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    
    # Calculate weights for clinical binary features to balance the reconstruction loss
    clinical_bin_features = full_multi_view_data_cpu['patient'].x[:, clinical_bin_idx.cpu()]
    num_pos = torch.sum(clinical_bin_features == 1, dim=0)
    num_neg = torch.sum(clinical_bin_features == 0, dim=0)
    clinical_bin_pos_weights = num_neg / (num_pos + 1e-6)
    clinical_bin_pos_weights = clinical_bin_pos_weights.to(device)
    print(f"INFO: Using pos_weights for Clinical Binary reconstruction (first 5): {clinical_bin_pos_weights[:5].cpu().numpy().tolist()}")

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
        
        train_mask = torch.zeros(fold_data['patient'].num_nodes, dtype=torch.bool, device=device)
        train_mask[train_indices] = True
        val_mask = torch.zeros(fold_data['patient'].num_nodes, dtype=torch.bool, device=device)
        val_mask[val_indices] = True

        # -------------------------------------------------------
        # STEP 1: PRE-TRAIN RADIOLOGY AGGREGATOR
        # -------------------------------------------------------
        radiology_state_dict = None
        if 'radiology' in model_config['view_configs']:
             radiology_state_dict = pretrain_radiology_aggregator(
                 fold_data, 
                 train_indices, 
                 model_config['radiology_aggregator_config'], 
                 device
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
        epochs_no_improve_loss = 0
        best_model_state = None
        best_epoch_results = {}

        # We pass all nodes to GVAE to get the graph structure, but we must mask losses later
        all_node_indices = torch.arange(fold_data['patient'].num_nodes, device=device)

        for epoch in range(1, train_config['epochs'] + 1):
            # =================== TRAINING PHASE ===================
            model.train()
            w_kl = linear_anneal(epoch, 0, kl_end_e, kl_start_w, base_w_kl)
            w_cl = linear_anneal(epoch, 0, cl_end_e, cl_start_w, base_w_cross_cl)
            
            # Forward pass on ALL data
            logits, vae_out, cl_out, _ = model(fold_data, all_node_indices)
            
            # 1. Main Classification Loss (Masked)
            loss_class = criterion_main_bce(logits[train_mask].squeeze(), fold_data['patient'].binary_label[train_mask].float())
            
            # 2. Contrastive Loss
            cl_out_train = {}
            for view, (emb, idx) in cl_out.items():
                # idx are global indices. Check which ones are in train_mask.
                mask_for_view = train_mask[idx]
                if mask_for_view.any():
                    cl_out_train[view] = (emb[mask_for_view], idx[mask_for_view])
            
            if len(cl_out_train) > 0:
                loss_cl = calculate_contrastive_loss(cl_out_train, train_config['cross_cl_temp'])
            else:
                loss_cl = torch.tensor(0.0, device=device)
            
            # 3. Reconstruction Losses (Mixed Strategy)
            rec_attr, rec_struct, kl_div, active_views_train = 0.0, 0.0, 0.0, 0
            
            for view, vo in vae_out.items():
                if vo and vo.get('mu') is not None:
                    # We rely on cl_out to get the indices corresponding to vae_out outputs
                    # (Assuming order is preserved, which it is in your GVAE implementation)
                    view_global_indices = cl_out[view][1]
                    view_train_mask = train_mask[view_global_indices]
                    
                    if view_train_mask.any():
                        active_views_train += 1
                        
                        # Filter reconstruction outputs using the mask
                        target_x = vo['original_x_subset'][view_train_mask]
                        recon_x = vo['rec_x'][view_train_mask]
                        
                        view_weight = w_rec_attr_config.get(view, 1.0)
                        
                        if view == 'clinical':
                            loss_cont = criterion_mse(recon_x[:, clinical_cont_idx], target_x[:, clinical_cont_idx])
                            loss_bin = F.binary_cross_entropy_with_logits(
                                recon_x[:, clinical_bin_idx], 
                                target_x[:, clinical_bin_idx],
                                pos_weight=clinical_bin_pos_weights
                            )
                            rec_attr += view_weight * (loss_cont + loss_bin)
                        else:
                            rec_attr += view_weight * criterion_mse(recon_x, target_x)

                        # Structure Loss (Masked)
                        adj_train_mask = view_train_mask.unsqueeze(1) & view_train_mask.unsqueeze(0)
                        if adj_train_mask.any():
                             rec_struct += w_rec_struct_config * F.binary_cross_entropy_with_logits(
                                 vo['rec_adj_logits'][adj_train_mask], 
                                 vo['original_adj_subset'][adj_train_mask]
                             )
                        
                        # KL Divergence (Masked)
                        kl_term = 1 + vo['logvar'][view_train_mask] - vo['mu'][view_train_mask].pow(2) - vo['logvar'][view_train_mask].exp()
                        kl_div += -0.5 * kl_term.sum(dim=1).mean()

            avg_rec_attr = rec_attr / active_views_train if active_views_train > 0 else 0.0
            avg_rec_struct = rec_struct / active_views_train if active_views_train > 0 else 0.0
            avg_kl = kl_div / active_views_train if active_views_train > 0 else 0.0
            
            total_train_loss = (base_w_class * loss_class + w_cl * loss_cl + avg_rec_attr + avg_rec_struct + w_kl * avg_kl)
            
            optimizer.zero_grad()
            total_train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.get('grad_clip_norm', 1.0))
            optimizer.step()

            # =================== VALIDATION PHASE ===================
            model.eval()
            with torch.no_grad():
                # Mask Classification
                val_logits = logits[val_mask]
                val_labels = fold_data['patient'].binary_label[val_mask]
                val_loss_class = criterion_main_bce(val_logits.squeeze(), val_labels.float())
                
                # Mask Contrastive Loss for Validation
                cl_out_val = {}
                for view, (emb, idx) in cl_out.items():
                    mask_for_view = val_mask[idx]
                    if mask_for_view.any():
                        cl_out_val[view] = (emb[mask_for_view], idx[mask_for_view])
                
                if len(cl_out_val) > 0:
                    val_loss_cl = calculate_contrastive_loss(cl_out_val, train_config['cross_cl_temp'])
                else:
                    val_loss_cl = torch.tensor(0.0, device=device)

                val_rec_attr, val_rec_struct, val_kl_div, val_active_views = 0.0, 0.0, 0.0, 0
                
                for view, vo in vae_out.items():
                    if vo and vo.get('mu') is not None:
                        view_global_indices = cl_out[view][1]
                        view_val_mask = val_mask[view_global_indices]
                        
                        if view_val_mask.any():
                            val_active_views += 1
                            
                            target_x_val = vo['original_x_subset'][view_val_mask]
                            recon_x_val = vo['rec_x'][view_val_mask]
                            view_weight = w_rec_attr_config.get(view, 1.0)

                            if view == 'clinical':
                                loss_cont_val = criterion_mse(recon_x_val[:, clinical_cont_idx], target_x_val[:, clinical_cont_idx])
                                loss_bin_val = F.binary_cross_entropy_with_logits(
                                    recon_x_val[:, clinical_bin_idx], 
                                    target_x_val[:, clinical_bin_idx],
                                    pos_weight=clinical_bin_pos_weights
                                )
                                val_rec_attr += view_weight * (loss_cont_val + loss_bin_val)
                            else:
                                val_rec_attr += view_weight * criterion_mse(recon_x_val, target_x_val)

                            adj_val_mask = view_val_mask.unsqueeze(1) & view_val_mask.unsqueeze(0)
                            if adj_val_mask.any():
                                val_rec_struct += w_rec_struct_config * F.binary_cross_entropy_with_logits(
                                    vo['rec_adj_logits'][adj_val_mask], 
                                    vo['original_adj_subset'][adj_val_mask]
                                )
                            
                            kl_term_val = 1 + vo['logvar'][view_val_mask] - vo['mu'][view_val_mask].pow(2) - vo['logvar'][view_val_mask].exp()
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
            
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                epochs_no_improve_loss = 0
                best_model_state = copy.deepcopy(model.state_dict())
                
                fpr, tpr = np.array([np.nan]), np.array([np.nan])
                if current_val_auc != -1.0:
                    fpr, tpr, _ = roc_curve(val_labels_np, val_probs_np)

                best_epoch_results = {
                    'fpr': fpr, 'tpr': tpr, 'auc': current_val_auc,
                    'y_true': val_labels_np, 'y_pred_probs': val_probs_np
                }
            else:
                epochs_no_improve_loss += 1

            if epochs_no_improve_loss >= train_config.get('patience_early_stopping', 30):
                print(f"  Early stopping at epoch {epoch}.")
                break
            
            if epoch % 10 == 0:
                 print(f"  Epoch {epoch:03d} | TLoss: {total_train_loss:.4f} | VLoss: {total_val_loss:.4f} | VAUC: {current_val_auc:.4f}")
        
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
                'best_threshold': best_threshold
            })
            
            if train_config.get('save_best_fold_model', True):
                torch.save(best_model_state, f"best_model_fold_{fold+1}.pth")

        del model, optimizer, scheduler, fold_data
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Summary ---
    if not fold_metrics_list: return {}, pd.DataFrame(), []
    df_fold_metrics = pd.DataFrame(fold_metrics_list)
    results_summary = {f"{agg}_{m}": getattr(df_fold_metrics[m], agg)() for m in ['auc', 'f1', 'accuracy'] for agg in ['mean', 'std']}
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

    # --- Feature Indices and Weights for Clinical View ---
    clinical_cont_idx = torch.arange(0, 5, device=device)
    clinical_bin_idx = torch.arange(5, 22, device=device)

    # Calculate Weights for Main Task and Clinical Binary Features
    # (Using CPU for initial calculation to avoid device issues if needed)
    labels_cpu = full_multi_view_data['patient']['binary_label'].cpu()
    main_pos_weight = torch.tensor([torch.sum(labels_cpu == 0) / (torch.sum(labels_cpu == 1) + 1e-6)], device=device)
    
    clinical_x_cpu = full_multi_view_data['patient'].x.cpu()
    bin_feats_cpu = clinical_x_cpu[:, 5:22]
    clinical_bin_pos_weights = (torch.sum(bin_feats_cpu == 0, dim=0) / (torch.sum(bin_feats_cpu == 1, dim=0) + 1e-6)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=train_config.get('patience', 10),
    )

    criterion_bce_logits = nn.BCEWithLogitsLoss(pos_weight=main_pos_weight)
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
                
                if view == 'clinical':
                    loss_cont = criterion_mse(vo['rec_x'][:, clinical_cont_idx], vo['original_x_subset'][:, clinical_cont_idx])
                    loss_bin = F.binary_cross_entropy_with_logits(
                        vo['rec_x'][:, clinical_bin_idx], 
                        vo['original_x_subset'][:, clinical_bin_idx],
                        pos_weight=clinical_bin_pos_weights
                    )
                    rec_attr += w_attr * (loss_cont + loss_bin)
                else:
                    rec_attr += w_attr * criterion_mse(vo['rec_x'], vo['original_x_subset'])

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
                        
                        if view == 'clinical':
                            loss_cont_v = criterion_mse(vo['rec_x'][:, clinical_cont_idx], vo['original_x_subset'][:, clinical_cont_idx])
                            loss_bin_v = F.binary_cross_entropy_with_logits(
                                vo['rec_x'][:, clinical_bin_idx], 
                                vo['original_x_subset'][:, clinical_bin_idx],
                                pos_weight=clinical_bin_pos_weights
                            )
                            val_rec_attr += w_attr * (loss_cont_v + loss_bin_v)
                        else:
                            val_rec_attr += w_attr * criterion_mse(vo['rec_x'], vo['original_x_subset'])
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

def pretrain_radiology_aggregator(
    full_data: HeteroData, 
    train_indices: torch.Tensor, 
    config: Dict[str, Any], 
    device: torch.device,
    epochs: int = 400
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
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        mil_model.train()
        optimizer.zero_grad()
        
        logits = mil_model(batch_lesion_features, aggregator_edge_index, len(active_patient_indices))
        loss = criterion(logits.squeeze(), batch_labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits.squeeze())
                try:
                    auc = roc_auc_score(batch_labels.cpu().numpy(), probs.cpu().numpy())
                except:
                    auc = 0.5
                print(f"   [Pre-training] Ep {epoch}: Loss={loss.item():.4f}, Train AUC={auc:.4f}")

    print("   [Pre-training] Complete. Extracting aggregator weights.")
    return mil_model.aggregator.state_dict()