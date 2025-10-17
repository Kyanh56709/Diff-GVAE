import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def get_view_subgraph_and_features(
    full_data: HeteroData,
    view_name: str,
    batch_patient_global_indices: torch.Tensor
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extracts features and a local subgraph for a specific view.
    Ensures that all tensor operations are performed on the correct device.
    """
    # Lấy device từ tensor đầu vào để đảm bảo tính nhất quán
    device = batch_patient_global_indices.device
    
    x_view_feature_key = f'x_{view_name}'
    edge_type_sim = ('patient', f'similar_to_{view_name}', 'patient')

    x_view_subset_batch: Optional[torch.Tensor] = None
    global_indices_of_subset_in_batch: torch.Tensor = batch_patient_global_indices

    # --- Logic để xác định các bệnh nhân có view này trong batch ---
    if view_name == 'radiology':
        mask_feature_key = f'{view_name}_mask'
        if mask_feature_key not in full_data['patient']:
            return None, torch.empty((2,0), dtype=torch.long, device=device), None, torch.empty(0, dtype=torch.long, device=device)

        view_presence_mask_all_patients = full_data['patient'][mask_feature_key].to(device)
        view_presence_in_batch = view_presence_mask_all_patients[batch_patient_global_indices]
        global_indices_of_subset_in_batch = batch_patient_global_indices[view_presence_in_batch]
        
        if global_indices_of_subset_in_batch.numel() == 0:
            return None, torch.empty((2,0), dtype=torch.long, device=device), None, global_indices_of_subset_in_batch

    elif view_name == 'clinical':
        if x_view_feature_key not in full_data['patient']:
             return None, torch.empty((2,0), dtype=torch.long, device=device), None, torch.empty(0, dtype=torch.long, device=device)
        
        # Với clinical, tất cả bệnh nhân trong batch đều được giả định có
        global_indices_of_subset_in_batch = batch_patient_global_indices
        
        # LẤY DỮ LIỆU FEATURE VÀ CHUYỂN LÊN ĐÚNG DEVICE
        x_view_data = full_data['patient'][x_view_feature_key].to(device)
        x_view_subset_batch = x_view_data[global_indices_of_subset_in_batch]

    else: # Các view khác như 'pathology'
        mask_feature_key = f'{view_name}_mask'
        if mask_feature_key not in full_data['patient'] or x_view_feature_key not in full_data['patient']:
            return None, torch.empty((2,0), dtype=torch.long, device=device), None, torch.empty(0, dtype=torch.long, device=device)

        view_presence_mask_all_patients = full_data['patient'][mask_feature_key].to(device)
        view_presence_in_batch = view_presence_mask_all_patients[batch_patient_global_indices]
        global_indices_of_subset_in_batch = batch_patient_global_indices[view_presence_in_batch]

        if global_indices_of_subset_in_batch.numel() == 0:
            return None, torch.empty((2,0), dtype=torch.long, device=device), None, global_indices_of_subset_in_batch
        
        # LẤY DỮ LIỆU FEATURE VÀ CHUYỂN LÊN ĐÚNG DEVICE
        x_view_data = full_data['patient'][x_view_feature_key].to(device)
        x_view_subset_batch = x_view_data[global_indices_of_subset_in_batch]

    # --- Logic trích xuất đồ thị con (subgraph) ---
    if edge_type_sim not in full_data.edge_types:
        return x_view_subset_batch, torch.empty((2,0), dtype=torch.long, device=device), None, global_indices_of_subset_in_batch

    # Đảm bảo edge_index và edge_attr cũng được chuyển lên đúng device
    view_full_edge_index = full_data[edge_type_sim].edge_index.to(device)
    view_full_edge_attr = getattr(full_data[edge_type_sim], 'edge_attr', None)
    if view_full_edge_attr is not None:
        view_full_edge_attr = view_full_edge_attr.to(device)

    if global_indices_of_subset_in_batch.numel() == 0:
        return x_view_subset_batch, torch.empty((2,0), dtype=torch.long, device=device), None, global_indices_of_subset_in_batch

    # Mapping từ global index sang local index (0, 1, 2, ...)
    # Tạo map trên CPU để tăng tốc độ
    global_to_local_idx_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(global_indices_of_subset_in_batch.cpu())}

    src_nodes_global = view_full_edge_index[0]
    dst_nodes_global = view_full_edge_index[1]

    # torch.isin hoạt động tốt nhất khi cả hai tensor trên cùng một device
    mask_src_in_subset = torch.isin(src_nodes_global, global_indices_of_subset_in_batch)
    mask_dst_in_subset = torch.isin(dst_nodes_global, global_indices_of_subset_in_batch)
    edge_selection_mask = mask_src_in_subset & mask_dst_in_subset

    if not edge_selection_mask.any():
        empty_edge_index = torch.empty((2,0), dtype=torch.long, device=device)
        empty_edge_attr = None
        if view_full_edge_attr is not None:
            empty_edge_attr = torch.empty((0, view_full_edge_attr.shape[1]), dtype=view_full_edge_attr.dtype, device=device)
        return x_view_subset_batch, empty_edge_index, empty_edge_attr, global_indices_of_subset_in_batch

    selected_edges_global_src = src_nodes_global[edge_selection_mask]
    selected_edges_global_dst = dst_nodes_global[edge_selection_mask]
    
    # Chuyển đổi từ global sang local indices
    # Thực hiện trên CPU rồi chuyển lại GPU có thể nhanh hơn với dict lookup
    local_edge_src_list = [global_to_local_idx_map[idx.item()] for idx in selected_edges_global_src.cpu()]
    local_edge_dst_list = [global_to_local_idx_map[idx.item()] for idx in selected_edges_global_dst.cpu()]

    local_edge_index_batch = torch.tensor([local_edge_src_list, local_edge_dst_list], dtype=torch.long, device=device)

    local_edge_attr_batch = None
    if view_full_edge_attr is not None:
        local_edge_attr_batch = view_full_edge_attr[edge_selection_mask]

    return x_view_subset_batch, local_edge_index_batch, local_edge_attr_batch, global_indices_of_subset_in_batch

def get_dense_adj_for_reconstruction(local_edge_index: Optional[torch.Tensor], num_nodes_in_subset: int, device: torch.device) -> torch.Tensor:
    """Creates a dense adjacency matrix from local_edge_index for reconstruction loss."""
    adj = torch.zeros((num_nodes_in_subset, num_nodes_in_subset), device=device)
    if local_edge_index is not None and local_edge_index.numel() > 0:
        adj[local_edge_index[0], local_edge_index[1]] = 1
        #make symmetric
        adj = torch.max(adj, adj.t()) 
    return adj


def preprocess_fold_data_with_pca(
    full_data_cpu: HeteroData,
    train_indices_np: np.ndarray,
    config: Dict[str, Any]
) -> HeteroData:
    """
    Applies PCA to high-dimensional features for a single fold to speed up training.

    PCA is fitted ONLY on the training data of the fold to prevent data leakage,
    and then used to transform the entire dataset.

    Args:
        full_data_cpu (HeteroData): The complete dataset, must be on the CPU.
        train_indices_np (np.ndarray): NumPy array of global indices for the training set.
        config (Dict[str, Any]): The training configuration dictionary, which should
                                 contain 'pca_config' and 'lesion_pca_config'.

    Returns:
        HeteroData: A new HeteroData object with reduced-dimension features, on the CPU.
    """
    print("  Preprocessing data for the fold with PCA...")
    
    # Chuẩn bị dữ liệu mới cho fold này
    fold_data = HeteroData()
    pca_config = config.get('pca_config', {})
    random_seed = config.get('random_seed', 42)

    # 1. Xử lý PCA cho các feature cấp độ bệnh nhân (patient-level)
    for view in ['clinical', 'pathology']:
        feature_key = f'x_{view}'
        if feature_key in full_data_cpu['patient']:
            original_features = full_data_cpu['patient'][feature_key].numpy()
            if view in pca_config and pca_config[view] < original_features.shape[1]:
                n_components = pca_config[view]
                # Đảm bảo n_components không lớn hơn số mẫu hoặc số feature
                max_components = min(len(train_indices_np), original_features.shape[1])
                final_n_components = min(n_components, max_components)
                
                print(f"    - Applying PCA on '{view}': {original_features.shape[1]} -> {final_n_components} features")
                pca = PCA(n_components=final_n_components, random_state=random_seed)
                pca.fit(original_features[train_indices_np]) # Fit CHỈ trên tập train
                reduced_features = pca.transform(original_features) # Transform toàn bộ
                fold_data['patient'][feature_key] = torch.tensor(reduced_features, dtype=torch.float32)
            else:
                # Sao chép nếu không áp dụng PCA
                fold_data['patient'][feature_key] = full_data_cpu['patient'][feature_key].clone()

    # 2. Xử lý PCA cho các feature cấp độ lesion
    lesion_pca_config = config.get('lesion_pca_config')
    if 'lesion' in full_data_cpu.node_types and hasattr(full_data_cpu['lesion'], 'x') and lesion_pca_config:
        original_lesion_features = full_data_cpu['lesion'].x.numpy()
        n_components = lesion_pca_config['n_components']

        # Tìm các lesion thuộc về bệnh nhân trong tập train để fit PCA
        edges = full_data_cpu['patient', 'has_lesion', 'lesion'].edge_index
        is_train_edge = torch.isin(edges[0], torch.from_numpy(train_indices_np))
        train_lesion_indices = torch.unique(edges[1, is_train_edge]).numpy()

        if len(train_lesion_indices) > 0:
            max_components = min(len(train_lesion_indices), original_lesion_features.shape[1])
            final_n_components = min(n_components, max_components)

            if final_n_components < original_lesion_features.shape[1]:
                print(f"    - Applying PCA on 'lesion.x': {original_lesion_features.shape[1]} -> {final_n_components} features")
                pca_lesion = PCA(n_components=final_n_components, random_state=random_seed)
                pca_lesion.fit(original_lesion_features[train_lesion_indices])
                reduced_lesion_features = pca_lesion.transform(original_lesion_features)
                fold_data['lesion'].x = torch.tensor(reduced_lesion_features, dtype=torch.float32)
            else:
                fold_data['lesion'].x = full_data_cpu['lesion'].x.clone()
        else:
            fold_data['lesion'].x = full_data_cpu['lesion'].x.clone()
    elif 'lesion' in full_data_cpu.node_types and hasattr(full_data_cpu['lesion'], 'x'):
        fold_data['lesion'].x = full_data_cpu['lesion'].x.clone()
        
    # 3. Sao chép các thuộc tính còn lại (metadata, labels, graph structure)
    for key in ['pathology_mask', 'radiology_mask', 'y', 'event', 'binary_label']:
        if key in full_data_cpu['patient']:
            fold_data['patient'][key] = full_data_cpu['patient'][key].clone()

    for edge_type in full_data_cpu.edge_types:
        for key, value in full_data_cpu[edge_type].items():
            fold_data[edge_type][key] = value.clone()

    print("  ✅ Preprocessing for fold complete.")
    return fold_data