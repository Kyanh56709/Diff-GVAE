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
#from utils.data_utils import preprocess_data_with_pca

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
    ddpm_config: Dict[str, Any],
    pca_config: Dict[str, int]
) -> Dict[str, float]:
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
    
    kf = KFold(n_splits=train_config['n_splits'], shuffle=True, random_state=train_config.get('random_seed', 42))
    
    all_true_labels, all_pred_probs = [], []
    fold_aucs, fold_f1s, fold_accuracies, fold_precisions, fold_recalls = [], [], [], [], []

    for fold, (train_idx_np, val_idx_np) in enumerate(kf.split(np.arange(full_data_cpu['patient'].num_nodes))):
        fold_num = fold + 1
        print(f"\n{'='*20} FOLD {fold_num}/{train_config['n_splits']} {'='*20}")
        train_indices = torch.from_numpy(train_idx_np).to(device)
        val_indices = torch.from_numpy(val_idx_np).to(device)

        # === Step 1: Train GVAE for this Fold on PCA data ===
        print(f"--- [Fold {fold_num}] Training GVAE model ---")
        best_gvae_state = train_gvae_single_fold(full_data, train_indices, val_indices, model_config, train_config, fold_num)

        if best_gvae_state is None:
            print(f"WARNING: GVAE training failed for fold {fold_num}. Skipping.")
            continue

        # === Step 2: Load Best GVAE and Train Fusion Layer ===
        gvae_fold_model = GVAE(**model_config).to(device)
        gvae_fold_model.load_state_dict(best_gvae_state)
        
        print(f"--- [Fold {fold_num}] Extracting mu vectors & Training Fusion Layer ---")
        stacked_mus_train, train_labels = get_all_view_mus_from_gvae(gvae_fold_model, full_data, train_indices)

        fusion_transformer = MuFusionTransformer(d_embed=model_config['d_embed'], n_heads=8, dim_feedforward=model_config['d_embed'] * 4).to(device)
        temp_classifier = torch.nn.Linear(model_config['d_embed'], 1).to(device)
        fusion_optimizer = torch.optim.AdamW(list(fusion_transformer.parameters()) + list(temp_classifier.parameters()), lr=1e-3)
        bce_loss = torch.nn.BCEWithLogitsLoss()

        
        for _ in tqdm(range(200), desc="Training Fusion Layer", leave=False):
            fusion_transformer.train()
            fusion_optimizer.zero_grad()
            mu_fused = fusion_transformer(stacked_mus_train)
            logits = temp_classifier(mu_fused).squeeze()
            loss = bce_loss(logits, train_labels.float())
            loss.backward()
            fusion_optimizer.step()
        
        # === Step 3: Generate Fused Latents and Train DDPMs ===
        print(f"--- [Fold {fold_num}] Generating fused latents and training DDPMs ---")
        fusion_transformer.eval()
        with torch.no_grad():
            final_train_mus = fusion_transformer(stacked_mus_train)

        mus_responder = final_train_mus[train_labels == 1]
        mus_non_responder = final_train_mus[train_labels == 0]

        ddpm_responder, scaler_resp = train_single_unconditional_ddpm(mus_responder, ddpm_config, device)
        ddpm_non_responder, scaler_non_resp = train_single_unconditional_ddpm(mus_non_responder, ddpm_config, device)

        if ddpm_responder is None or ddpm_non_responder is None:
            print(f"WARNING: DDPM training failed. Skipping fold.")
            continue
            
        # === Step 4: Evaluate on the Validation Set (STABLE EVALUATION) ===
        print(f"--- [Fold {fold_num}] Evaluating on validation set ---")
        stacked_mus_val, val_labels = get_all_view_mus_from_gvae(gvae_fold_model, full_data, val_indices)
        
        with torch.no_grad():
            final_val_mus = fusion_transformer(stacked_mus_val)
            
        val_mus_scaled_for_resp = torch.tensor(scaler_resp.transform(final_val_mus.cpu().numpy()), dtype=torch.float32).to(device)
        val_mus_scaled_for_non_resp = torch.tensor(scaler_non_resp.transform(final_val_mus.cpu().numpy()), dtype=torch.float32).to(device)
        
        fold_probs = []
        timesteps_to_eval = torch.linspace(0, ddpm_config['timesteps'] - 1, 50, dtype=torch.long).to(device)
        
        with torch.no_grad():
            for i in tqdm(range(len(final_val_mus)), desc="Evaluating", leave=False):
                loss_resp = ddpm_responder.evaluation_loss(val_mus_scaled_for_resp[i].unsqueeze(0), timesteps_to_eval)
                loss_non_resp = ddpm_non_responder.evaluation_loss(val_mus_scaled_for_non_resp[i].unsqueeze(0), timesteps_to_eval)
                
                likelihood_resp = 1 / (loss_resp + 1e-9)
                likelihood_non_resp = 1 / (loss_non_resp + 1e-9)
                prob_is_responder = likelihood_resp / (likelihood_resp + likelihood_non_resp)
                fold_probs.append(prob_is_responder.item())
        
        # --- Store results for this fold ---
        fold_labels = val_labels.cpu().numpy()
        fold_preds_binary = (np.array(fold_probs) > 0.5).astype(int)
        
        fold_aucs.append(roc_auc_score(fold_labels, fold_probs))
        fold_f1s.append(f1_score(fold_labels, fold_preds_binary, zero_division=0))
        fold_accuracies.append(accuracy_score(fold_labels, fold_preds_binary))
        fold_precisions.append(precision_score(fold_labels, fold_preds_binary, zero_division=0))
        fold_recalls.append(recall_score(fold_labels, fold_preds_binary, zero_division=0))
        
        print(f"  [Fold {fold_num}] AUC: {fold_aucs[-1]:.4f}, F1: {fold_f1s[-1]:.4f}")

        all_true_labels.extend(fold_labels)
        all_pred_probs.extend(fold_probs)

    # === Step 5: Aggregate and Report Final Metrics ===
    print(f"\n{'='*20} FINAL PIPELINE RESULTS {'='*20}")
    if not all_true_labels:
        print("No results to report.")
        return {}

    results = {
        'mean_auc': np.mean(fold_aucs), 'std_auc': np.std(fold_aucs),
        'mean_f1': np.mean(fold_f1s), 'std_f1': np.std(fold_f1s),
        'mean_accuracy': np.mean(fold_accuracies), 'std_accuracy': np.std(fold_accuracies),
        'mean_precision': np.mean(fold_precisions), 'std_precision': np.std(fold_precisions),
        'mean_recall': np.mean(fold_recalls), 'std_recall': np.std(fold_recalls),
    }

    print("--- Cross-Validation Summary (PCA-GVAE + Generative DDPM Classifier) ---")
    for key, value in results.items():
        print(f"  {key.replace('_', ' ').capitalize()}: {value:.4f}")

    return results
