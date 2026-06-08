import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, Any

from .gvae_components import (
    ViewEncoder, StructureDecoder, AttributeDecoder, FusionAndClassifierHead,
    ProjectionHead, RadiologyLesionAttentionAggregator
)

from utils.data_utils import get_view_subgraph_and_features, get_dense_adj_for_reconstruction


class GVAE (nn.Module):
    def __init__(self,
                 view_configs: Dict[str, Any],
                 radiology_aggregator_config: Dict[str, Any],
                 projection_head_config: Dict[str, Any],
                 fusion_config: Dict[str, Any],
                 classifier_config: Dict[str, Any],
                 d_embed: int,
                 missing_strategy: str = 'zero',
                 logvar_clamp=None,
                 radiology_zero_lesion_passthrough: bool = False):
        super().__init__()
        self.views = list(view_configs.keys())
        self.d_embed = d_embed
        self.missing_strategy = missing_strategy
        self.logvar_clamp = logvar_clamp
        self.radiology_zero_lesion_passthrough = radiology_zero_lesion_passthrough

        self.radiology_lesion_aggregator = None
        if 'radiology' in self.views and radiology_aggregator_config:
            self.radiology_lesion_aggregator = RadiologyLesionAttentionAggregator(
                lesion_feature_dim=radiology_aggregator_config['lesion_feature_dim'],
                patient_embed_dim=radiology_aggregator_config['aggregated_output_dim'],
                attention_hidden_dim=radiology_aggregator_config.get(
                    'attention_hidden_dim'),
                dropout=radiology_aggregator_config.get('dropout', 0.1)
            )
            if view_configs['radiology']['in_channels'] != radiology_aggregator_config['aggregated_output_dim']:
                raise ValueError(
                    "Radiology VAE in_channels must match aggregator output_dim")

        # --- VAE Components ---
        self.vae_encoders = nn.ModuleDict()
        self.structure_decoders = nn.ModuleDict()
        self.attribute_decoders = nn.ModuleDict()
        # if self.missing_strategy == 'learnable':
        #     self.missing_embeddings_params = nn.ParameterDict()
        if self.missing_strategy in ('learnable', 'zero'):
            self.missing_embeddings_params = nn.ParameterDict()
            if self.missing_strategy == 'zero':
                # create a non-learnable zero embedding per view (will be overwritten for 'learnable')
                for v in self.views:
                    self.missing_embeddings_params[v] = nn.Parameter(torch.zeros(1, d_embed), requires_grad=False)

        self.projection_heads = nn.ModuleDict()

        for view, config in view_configs.items():
            self.vae_encoders[view] = ViewEncoder(
                config['in_channels'], config['hidden_channels_vae'], d_embed,
                config.get('heads', 4), config.get('dropout', 0.3),
                config.get('num_gnn_layers_vae', config.get('num_gnn_layers', 2)),
                config.get('edge_dim', -1),
                logvar_clamp=logvar_clamp
            )
            self.structure_decoders[view] = StructureDecoder()
            self.attribute_decoders[view] = AttributeDecoder(
                d_embed, config['in_channels'])

            self.projection_heads[view] = ProjectionHead(
                input_dim=d_embed,
                hidden_dim=projection_head_config.get('hidden_dim', d_embed),
                output_dim=projection_head_config.get('output_dim', d_embed),
                dropout=projection_head_config.get('dropout', 0.1)
            )
            if self.missing_strategy == 'learnable':
                self.missing_embeddings_params[view] = nn.Parameter(
                    torch.randn(1, d_embed))
                

        self.fusion_and_classifier_head = FusionAndClassifierHead(
            embed_dim=d_embed,
            num_heads=fusion_config.get('num_fusion_heads', 4),
            ffn_dim_multiplier=fusion_config.get('fusion_ffn_multiplier', 2),
            fusion_dropout=fusion_config.get('fusion_dropout', 0.1),
            classifier_hidden_dim=classifier_config['classifier_hidden_dim'],
            classifier_dropout=classifier_config.get('classifier_dropout', 0.5)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, full_data: HeteroData, batch_patient_global_indices: torch.Tensor):
        device = batch_patient_global_indices.device

        vae_outputs_for_loss = {view: {} for view in self.views}
        all_patient_zs_for_fusion = {}
        all_patient_mus_projected_for_cl = {}

        for view in self.views:
            x_patient_level_subset, local_patient_sim_edge_idx, local_patient_sim_edge_attr, global_indices_subset_patients = \
                get_view_subgraph_and_features(full_data, view, batch_patient_global_indices)
            
            if global_indices_subset_patients.numel() == 0:
                vae_outputs_for_loss[view] = {
                    'mu': None, 'logvar': None, 'z_sampled_for_dec': None,
                    'rec_adj_logits': None, 'rec_x': None,
                    'original_x_subset': None, 'original_adj_subset': None
                }
                continue

            num_active_patients_for_view = global_indices_subset_patients.shape[0]
            x_for_vae_encoder, original_x_for_vae_reconstruction = None, None
            
            if view == 'radiology' and self.radiology_lesion_aggregator:

                patient_lesion_edges_all = full_data['patient', 'has_lesion', 'lesion'].edge_index.to(device)
                all_lesion_features_all = full_data['lesion'].x.to(device)
                
                # Vectorized edge filtering: keep only edges from active patients
                edge_mask = torch.isin(patient_lesion_edges_all[0], global_indices_subset_patients)
                filtered_edges = patient_lesion_edges_all[:, edge_mask]
                
                if filtered_edges.numel() > 0:
                    # Map global patient indices to local batch indices (0, 1, 2, ...)
                    max_patient_idx = patient_lesion_edges_all[0].max().item() + 1
                    global_to_local = torch.full((max_patient_idx,), -1, dtype=torch.long, device=device)
                    global_to_local[global_indices_subset_patients] = torch.arange(num_active_patients_for_view, device=device)
                    batch_lesion_src_patient_local_idx_t = global_to_local[filtered_edges[0]]
                    
                    batch_lesion_node_global_idx_t = filtered_edges[1]
                    
                    # Deduplicate lesions: unique features + remapped edge indices
                    unique_lesion_global_indices, inverse_local_indices = torch.unique(batch_lesion_node_global_idx_t, return_inverse=True)
                    lesion_features_for_batch_agg = all_lesion_features_all[unique_lesion_global_indices]
                    
                    patient_to_batch_lesion_edge_index = torch.stack([
                        batch_lesion_src_patient_local_idx_t,
                        inverse_local_indices
                    ], dim=0)

                    x_for_vae_encoder = self.radiology_lesion_aggregator(
                        lesion_features_for_batch_agg,
                        patient_to_batch_lesion_edge_index,
                        num_active_patients_for_view 
                    )
                    original_x_for_vae_reconstruction = x_for_vae_encoder
                
            elif x_patient_level_subset is not None and x_patient_level_subset.numel() > 0:
                x_for_vae_encoder = x_patient_level_subset
                original_x_for_vae_reconstruction = x_for_vae_encoder

            if x_for_vae_encoder is not None and x_for_vae_encoder.numel() > 0:
                num_nodes_for_vae = x_for_vae_encoder.shape[0]
                mu, logvar = self.vae_encoders[view](x_for_vae_encoder, local_patient_sim_edge_idx, local_patient_sim_edge_attr)
                z_sampled = self.reparameterize(mu, logvar)
                mu_projected = self.projection_heads[view](mu)

                vae_outputs_for_loss[view] = {
                    'mu': mu, 'logvar': logvar, 'z_sampled_for_dec': z_sampled,
                    'original_x_subset': original_x_for_vae_reconstruction,
                    'original_adj_subset': get_dense_adj_for_reconstruction(local_patient_sim_edge_idx, num_nodes_for_vae, device),
                    'rec_adj_logits': self.structure_decoders[view](z_sampled),
                    'rec_x': self.attribute_decoders[view](z_sampled)
                }
                
                for i, global_idx_tensor in enumerate(global_indices_subset_patients):
                    global_idx_item = global_idx_tensor.item()
                    if global_idx_item not in all_patient_zs_for_fusion:
                        all_patient_zs_for_fusion[global_idx_item] = {}
                    all_patient_zs_for_fusion[global_idx_item][view] = z_sampled[i]
                    if global_idx_item not in all_patient_mus_projected_for_cl:
                        all_patient_mus_projected_for_cl[global_idx_item] = {}
                    all_patient_mus_projected_for_cl[global_idx_item][view] = mu_projected[i]

        # --- FUSION AND CLASSIFICATION ---
        batch_fusion_input_list = []
        for global_idx_tensor in batch_patient_global_indices:
            global_idx_item = global_idx_tensor.item()
            patient_embs = []
            patient_z_data = all_patient_zs_for_fusion.get(global_idx_item, {})
            for view in self.views:
                emb = patient_z_data.get(view)
                if emb is None:
                    if self.missing_strategy == 'learnable':
                        emb = self.missing_embeddings_params[view].squeeze(0)
                    else: # zero
                        emb = torch.zeros(self.d_embed, device=device)
                patient_embs.append(emb)
            batch_fusion_input_list.append(torch.stack(patient_embs))

        if not batch_fusion_input_list:
             # Xử lý trường hợp không có bệnh nhân nào trong batch
            return torch.empty(0, 1, device=device), vae_outputs_for_loss, {}, None

        batch_fusion_input_tensor = torch.stack(batch_fusion_input_list)
        logits, fusion_attention = self.fusion_and_classifier_head(batch_fusion_input_tensor)

        # --- FORMAT OUTPUT FOR CONTRASTIVE LOSS ---
        mus_projected_for_cl_formatted = {}
        for patient_idx, view_data in all_patient_mus_projected_for_cl.items():
            for view, emb in view_data.items():
                if view not in mus_projected_for_cl_formatted:
                    mus_projected_for_cl_formatted[view] = {'embeddings': [], 'indices': []}
                mus_projected_for_cl_formatted[view]['embeddings'].append(emb)
                mus_projected_for_cl_formatted[view]['indices'].append(patient_idx)

        final_mus_projected_for_cl = {
            view: (torch.stack(data['embeddings']), torch.tensor(data['indices'], device=device, dtype=torch.long))
            for view, data in mus_projected_for_cl_formatted.items()
        }

        return logits, vae_outputs_for_loss, final_mus_projected_for_cl, fusion_attention
    

@torch.no_grad()
def get_separate_view_mus(
    gvae_model: GVAE,
    full_data: HeteroData,
    indices: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Extracts separate mu vectors for each modality for a given batch of patients.

    For patients missing a modality, the corresponding learnable 'missing embedding'
    from the GVAE model is used as a placeholder.

    Args:
        gvae_model (GVAE): The pre-trained and frozen GVAE model.
        full_data (HeteroData): The complete graph dataset.
        indices (torch.Tensor): A tensor of global patient indices for the batch.

    Returns:
        Dict[str, torch.Tensor]: A dictionary where keys are view names (e.g., 'clinical')
                                 and values are tensors of mu vectors of shape [batch_size, d_embed].
    """
    gvae_model.eval()
    device = indices.device
    num_patients = len(indices)
    d_embed = gvae_model.d_embed
    
    # --- 1. Perform a single forward pass to get all available mu vectors ---
    # The vae_outputs dictionary contains the computed mu vectors, but only for
    # patients who actually have each view.
    _, vae_outputs, _, _ = gvae_model(full_data, indices)

    # --- 2. Create a mapping from global index to its position in the batch ---
    # This is crucial for correctly placing the mu vectors.
    global_to_batch_idx_map = {global_idx.item(): batch_idx 
                               for batch_idx, global_idx in enumerate(indices)}

    # --- 3. Initialize output dictionary and process each view ---
    all_mus = {}
    for view in gvae_model.views:
        # a. Create a default tensor filled with the 'missing' embedding for this view.
        # This will be the template we fill in.
        missing_embedding = gvae_model.missing_embeddings_params[view].expand(num_patients, d_embed)
        view_mus_tensor = missing_embedding.clone()

        # b. Find which patients in this batch actually have this view.
        # We re-use get_view_subgraph_and_features to get this list of indices.
        _, _, _, global_indices_with_view = get_view_subgraph_and_features(full_data, view, indices)
        
        # c. If any patients have this view, get their mu vectors and place them.
        if global_indices_with_view.numel() > 0 and vae_outputs[view].get('mu') is not None:
            mus_for_this_view = vae_outputs[view]['mu']
            
            # Iterate through the patients that have the view
            for i, global_idx in enumerate(global_indices_with_view):
                # Find the patient's position (0, 1, 2...) within this specific batch
                batch_idx = global_to_batch_idx_map[global_idx.item()]
                
                # Overwrite the 'missing' token with the actual computed mu vector
                view_mus_tensor[batch_idx] = mus_for_this_view[i]
        
        all_mus[view] = view_mus_tensor
        
    return all_mus