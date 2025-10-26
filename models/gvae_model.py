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
                 missing_strategy: str = 'zero'):
        super().__init__()
        self.views = list(view_configs.keys())
        self.d_embed = d_embed
        self.missing_strategy = missing_strategy

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
                config.get('num_gnn_layers_vae', 2), config.get('edge_dim', -1)
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
                


        # self.fusion_layer = MHA_CLSToken_FusionLayer(
        #     embed_dim=d_embed,  # The dimension of your view embeddings
        #     num_heads=fusion_config.get('num_fusion_heads', 4),
        #     ffn_dim_multiplier=fusion_config.get('fusion_ffn_multiplier', 2),
        #     dropout=fusion_config.get('dropout_fusion', 0.1),
        #     output_dim=fusion_config.get('fused_dim', d_embed)
        # )

        # self.classifier = ClassifierMLP(
        #     input_dim=fusion_config.get('fused_dim', d_embed),
        #     hidden_dim=classifier_config['hidden_dim_classifier'],
        #     output_dim=1, dropout=classifier_config.get('dropout_class', 0.5)
        # )

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
                

                active_patient_global_to_local_map = {glob_idx.item(): i for i, glob_idx in enumerate(global_indices_subset_patients.cpu())}
                
                batch_lesion_src_patient_local_idx_list = []
                batch_lesion_node_global_idx_list = []

                # Vòng lặp này bây giờ sẽ nhanh hơn vì patient_lesion_edges_all đã ở trên GPU
                for i_edge in range(patient_lesion_edges_all.shape[1]):
                    src_patient_global = patient_lesion_edges_all[0, i_edge].item()
                    dst_lesion_global = patient_lesion_edges_all[1, i_edge].item()
                    if src_patient_global in active_patient_global_to_local_map:
                        batch_lesion_src_patient_local_idx_list.append(active_patient_global_to_local_map[src_patient_global])
                        batch_lesion_node_global_idx_list.append(dst_lesion_global)
                
                if batch_lesion_node_global_idx_list:
                    batch_lesion_src_patient_local_idx_t = torch.tensor(batch_lesion_src_patient_local_idx_list, dtype=torch.long, device=device)
                    batch_lesion_node_global_idx_t = torch.tensor(batch_lesion_node_global_idx_list, dtype=torch.long, device=device)
                    
                    # Thao tác indexing bây giờ sẽ thành công vì cả hai tensor đều trên GPU
                    lesion_features_for_batch_agg = all_lesion_features_all[batch_lesion_node_global_idx_t]
                    
                    unique_lesions_in_batch, inverse_indices = torch.unique(batch_lesion_node_global_idx_t, return_inverse=True)
                    batch_local_lesion_indices_for_agg = torch.arange(lesion_features_for_batch_agg.shape[0], device=device)

                    patient_to_batch_lesion_edge_index = torch.stack([
                        batch_lesion_src_patient_local_idx_t,
                        batch_local_lesion_indices_for_agg
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
    