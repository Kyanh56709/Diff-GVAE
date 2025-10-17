import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.utils.data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from typing import Dict, Any, List
from models.gvae_model import GVAE
from utils.data_utils import preprocess_fold_data_with_pca
# from models.ddpm import UnconditionalDenoisingNetwork, UnconditionalDDPM
from training.train_gvae import train_gvae_single_fold
from training.train_ddpm import train_single_unconditional_ddpm
from models.gvae_components import MuFusionTransformer
from utils.data_utils import get_view_subgraph_and_features
from typing import Tuple
from tqdm import tqdm

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

    # Lấy output của GVAE cho các indices được yêu cầu
    _, vae_outputs, _, _ = gvae_model(full_data, indices)
    labels = full_data['patient']['binary_label'].to(device)[indices]

    # Chuẩn bị một dictionary để lưu trữ mu vectors, key là global index
    # Khởi tạo với missing embeddings để xử lý các bệnh nhân không có view nào
    num_indices = len(indices)
    patient_mus = {idx.item(): {view: gvae_model.missing_embeddings_params[view].squeeze(0)
                                for view in gvae_model.views}
                   for idx in indices}

    # Cập nhật với mu vectors thực tế từ các view có sẵn
    for view in gvae_model.views:
        # Lấy các global indices có view này trong batch hiện tại
        _, _, _, global_indices_with_view = get_view_subgraph_and_features(full_data, view, indices)
        
        if global_indices_with_view.numel() > 0 and vae_outputs[view].get('mu') is not None:
            mus_for_view = vae_outputs[view]['mu']
            # Cập nhật mu cho từng bệnh nhân có view này
            for i, global_idx in enumerate(global_indices_with_view):
                patient_mus[global_idx.item()][view] = mus_for_view[i]

    # Sắp xếp lại thành một tensor [batch, num_views, d_embed] theo đúng thứ tự của `indices`
    stacked_mus_list = []
    for idx in indices:
        mus_for_patient = [patient_mus[idx.item()][view] for view in gvae_model.views]
        stacked_mus_list.append(torch.stack(mus_for_patient, dim=0))

    return torch.stack(stacked_mus_list, dim=0), labels




def kfold_gvae_ddpm_generative_classifier(
    full_data: torch.utils.data.Dataset,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    ddpm_config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Performs K-Fold CV for the entire GVAE + DDPM generative classifier pipeline.

    For each fold, it:
    1. Trains a GVAE model on the training data.
    2. Extracts latent vectors (mu) from the trained GVAE.
    3. Trains two separate unconditional DDPMs: one for responders, one for non-responders.
    4. Evaluates on the validation set by comparing the likelihood of a patient's latent
       vector under both DDPMs.
    5. Aggregates metrics across all folds for a final robust evaluation.

    Returns:
        Dict[str, float]: A dictionary containing the final aggregated performance metrics.
    """
    device = train_config['device']
    print(f"Using device: {device}")
    kf = KFold(n_splits=train_config['n_splits'], shuffle=True, random_state=train_config.get('random_seed', 42))
    
    # Tạo một bản sao của dữ liệu trên CPU để thực hiện PCA
    full_data_cpu = full_data.clone().cpu()
    
    all_true_labels, all_pred_probs = [], []
    fold_aucs: List[float] = []
    fold_f1s: List[float] = []
    fold_accuracies: List[float] = []
    fold_precisions: List[float] = []
    fold_recalls: List[float] = []

    for fold, (train_idx_np, val_idx_np) in enumerate(kf.split(np.arange(full_data_cpu['patient'].num_nodes))):
        fold_num = fold + 1
        print(f"\n{'='*20} FOLD {fold_num}/{train_config['n_splits']} {'='*20}")
        train_indices = torch.from_numpy(train_idx_np).to(device)
        val_indices = torch.from_numpy(val_idx_np).to(device)

        # === Step 1: Train GVAE for this Fold ===
        print(f"--- [Fold {fold_num}] Training GVAE model ---")
        best_gvae_state = train_gvae_single_fold(full_data, train_indices, val_indices, model_config, train_config, fold_num)

        if best_gvae_state is None:
            print(f"WARNING: GVAE training failed for fold {fold_num}. Skipping this fold.")
            continue

        # === Step 2: Load Best GVAE and Extract View-Specific Mus ===
        gvae_fold_model = GVAE(**model_config).to(device)
        gvae_fold_model.load_state_dict(best_gvae_state)
        
        print(f"--- [Fold {fold_num}] Extracting view-specific mu vectors from training set ---")
        stacked_mus_train, train_labels = get_all_view_mus_from_gvae(gvae_fold_model, full_data, train_indices)

        # === Step 2.5: Train the MuFusionTransformer ===
        print(f"--- [Fold {fold_num}] Training MuFusionTransformer ---")
        fusion_transformer = MuFusionTransformer(
            d_embed=model_config['d_embed'],
            n_heads=8,
            dim_feedforward=model_config['d_embed'] * 4
        ).to(device)
        
        # Một classifier tạm thời để hướng dẫn việc học của fusion layer
        temp_classifier = torch.nn.Linear(model_config['d_embed'], 1).to(device)
        fusion_params = list(fusion_transformer.parameters()) + list(temp_classifier.parameters())
        fusion_optimizer = torch.optim.AdamW(fusion_params, lr=1e-3, weight_decay=1e-4)
        bce_loss = torch.nn.BCEWithLogitsLoss()

        for _ in tqdm(range(200), desc="Training Fusion Layer"): # Huấn luyện 100 epochs
            fusion_transformer.train()
            fusion_optimizer.zero_grad()
            mu_fused = fusion_transformer(stacked_mus_train)
            logits = temp_classifier(mu_fused).squeeze()
            loss = bce_loss(logits, train_labels.float())
            loss.backward()
            fusion_optimizer.step()
        
        # === Step 3: Generate Intelligently Fused Latents and Train DDPMs ===
        print(f"--- [Fold {fold_num}] Generating fused latents and training DDPMs ---")
        fusion_transformer.eval()
        with torch.no_grad():
            final_train_mus = fusion_transformer(stacked_mus_train)

        mus_responder = final_train_mus[train_labels == 1]
        mus_non_responder = final_train_mus[train_labels == 0]

        ddpm_responder, scaler_resp = train_single_unconditional_ddpm(mus_responder, ddpm_config, device)
        ddpm_non_responder, scaler_non_resp = train_single_unconditional_ddpm(mus_non_responder, ddpm_config, device)

        if ddpm_responder is None or ddpm_non_responder is None:
            print(f"WARNING: DDPM training failed for fold {fold_num}. Skipping this fold.")
            continue
            
        # === Step 4: Evaluate on the Validation Set ===
        print(f"--- [Fold {fold_num}] Evaluating on validation set ({len(val_indices)} samples) ---")
        stacked_mus_val, val_labels = get_all_view_mus_from_gvae(gvae_fold_model, full_data, val_indices)
        
        with torch.no_grad():
            final_val_mus = fusion_transformer(stacked_mus_val)
            
        # Scale validation latents using the scalers fitted on the respective training subsets
        val_mus_scaled_for_resp = torch.tensor(scaler_resp.transform(final_val_mus.cpu().numpy()), dtype=torch.float32).to(device)
        val_mus_scaled_for_non_resp = torch.tensor(scaler_non_resp.transform(final_val_mus.cpu().numpy()), dtype=torch.float32).to(device)
        
        fold_probs = []
        ddpm_responder.eval()
        ddpm_non_responder.eval()
        with torch.no_grad():
            for i in range(len(final_val_mus)):
                loss_resp = ddpm_responder.loss(val_mus_scaled_for_resp[i].unsqueeze(0))
                loss_non_resp = ddpm_non_responder.loss(val_mus_scaled_for_non_resp[i].unsqueeze(0))

                epsilon = 1e-9
                likelihood_resp = 1 / (loss_resp + epsilon)
                likelihood_non_resp = 1 / (loss_non_resp + epsilon)
                
                prob_is_responder = likelihood_resp / (likelihood_resp + likelihood_non_resp)
                fold_probs.append(prob_is_responder.item())
        
        all_true_labels.extend(val_labels.cpu().numpy())
        all_pred_probs.extend(fold_probs)

        fold_labels = val_labels.cpu().numpy()
        fold_auc = roc_auc_score(fold_labels, fold_probs)
        fold_aucs.append(fold_auc)
        print(f"  [Fold {fold_num}] AUC: {fold_auc:.4f}")
        fold_preds_binary = (np.array(fold_probs) > 0.5).astype(int)
        fold_f1 = f1_score(fold_labels, fold_preds_binary, zero_division=0)
        fold_f1s.append(fold_f1)
        print(f"  [Fold {fold_num}] F1 Score: {fold_f1:.4f}")
        fold_accuracy = accuracy_score(fold_labels, fold_preds_binary)
        fold_accuracies.append(fold_accuracy)
        print(f"  [Fold {fold_num}] Accuracy: {fold_accuracy:.4f}")
        fold_precision = precision_score(fold_labels, fold_preds_binary, zero_division=0)
        fold_precisions.append(fold_precision)
        fold_recall = recall_score(fold_labels, fold_preds_binary, zero_division=0)
        fold_recalls.append(fold_recall)
        print(f"  [Fold {fold_num}] Precision: {fold_precision:.4f}")
        print(f"  [Fold {fold_num}] Recall: {fold_recall:.4f}")
        

        all_true_labels.extend(fold_labels)
        all_pred_probs.extend(fold_probs)

    # === Step 5: Aggregate and Report Final Metrics ===
    # (Không có thay đổi ở đây)
    print(f"\n{'='*20} FINAL PIPELINE RESULTS {'='*20}")
    if not all_true_labels:
        print("No results to report. Training may have failed.")
        return {}

    final_auc = roc_auc_score(all_true_labels, all_pred_probs)
    final_preds_binary = (np.array(all_pred_probs) > 0.5).astype(int)
    auc_std = np.std(fold_aucs)
    final_f1 = f1_score(all_true_labels, final_preds_binary, zero_division=0)
    final_accuracy = accuracy_score(all_true_labels, final_preds_binary)
    final_precision = precision_score(all_true_labels, final_preds_binary, zero_division=0)
    final_recall = recall_score(all_true_labels, final_preds_binary, zero_division=0)
    f1_std = np.std(fold_f1s)
    accuracy_std = np.std(fold_accuracies)
    precision_std = np.std(fold_precisions)
    recall_std = np.std(fold_recalls)

    results = {
        'auc': final_auc,
        'auc_std': auc_std,
        'f1': final_f1,
        'f1_std': f1_std,
        'accuracy': final_accuracy,
        'accuracy_std': accuracy_std,
        'precision': final_precision,
        'precision_std': precision_std,
        'recall': final_recall,
        'recall_std': recall_std
    }

    print("--- Cross-Validation Summary (PCA-GVAE + Generative DDPM Classifier) ---")
    for key, value in results.items():
        print(f"  Overall {key.capitalize()}: {value:.4f}")

    return results
