import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear, LayerNorm, BatchNorm
from torch_scatter import scatter_add, scatter_max
from typing import Tuple, Optional

class ViewEncoder(nn.Module):
    """
    Graph VAE Encoder using GATv2Conv to map node features and graph structure
    of a specific view to parameters of a latent Gaussian distribution (mu, logvar).
    """
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int,
                 heads: int = 4, dropout: float = 0.5, num_gnn_layers: int = 2, edge_dim: int = -1):
        """
        Args:
            in_channels: Dimensionality of input node features for this view.
            hidden_channels: Dimensionality of hidden layers in the GNN.
            latent_dim: Dimensionality of the output latent space (mu and logvar).
            heads: Number of attention heads in GATv2Conv layers.
            dropout: Dropout rate.
            num_gnn_layers: Number of GATv2Conv layers (supports 1 or 2).
            edge_dim: Dimensionality of edge features (-1 if no edge features).
        """
        super().__init__()
        if num_gnn_layers not in [1, 2]:
            raise ValueError("ViewEncoder currently supports 1 or 2 GNN layers.")

        self.num_gnn_layers = num_gnn_layers
        self.dropout_p = dropout
        current_dim = hidden_channels

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True,
                               dropout=dropout, edge_dim=edge_dim, add_self_loops=True)
        self.bn1 = LayerNorm(hidden_channels * heads)

        if num_gnn_layers > 1:
            
            self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True,
                                   dropout=dropout, edge_dim=edge_dim, add_self_loops=True)
            self.bn2 = LayerNorm(hidden_channels * heads)
            current_dim = hidden_channels * heads

        self.fc_mu = Linear(current_dim, latent_dim)
        self.fc_logvar = Linear(current_dim, latent_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        if self.num_gnn_layers > 1:
            self.conv2.reset_parameters()
            self.bn2.reset_parameters()
        self.fc_mu.reset_parameters()
        self.fc_logvar.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node feature matrix [num_nodes, in_channels].
            edge_index: Graph connectivity [2, num_edges].
            edge_attr: Edge feature matrix [num_edges, edge_dim] (optional).

        Returns:
            mu: Latent mean [num_nodes, latent_dim].
            logvar: Latent log variance [num_nodes, latent_dim].
        """
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Layer 2 (if exists)
        if self.num_gnn_layers == 2:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
            x = self.bn2(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Output Projections
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        #logvar = torch.tanh(self.fc_logvar(x)) * 5.0
        #logvar = F.hardtanh(self.fc_logvar(x), min_val=-6.0, max_val=2.0)
        return mu, logvar

# --- 2. Structure Decoder (Adjacency Reconstruction) ---
class StructureDecoder(nn.Module):
    """
    Decodes latent embeddings to reconstruct graph adjacency matrix logits
    using inner product.
    """
    def __init__(self, activation: str = 'none'):
        """
        Args:
            activation: Output activation ('sigmoid' or 'none'). 'none' is suitable
                        for BCEWithLogitsLoss.
        """
        super().__init__()
        if activation not in ['sigmoid', 'none']:
             raise ValueError("Activation must be 'sigmoid' or 'none'")
        self.activation = activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent node embeddings [num_nodes, latent_dim].

        Returns:
            adj_rec_logits or adj_rec_probs: Reconstructed adjacency [num_nodes, num_nodes].
        """
        adj_rec_logits = torch.matmul(z, z.t())
        if self.activation == 'sigmoid':
            return torch.sigmoid(adj_rec_logits)
        return adj_rec_logits

# --- 3. Attribute Decoder (Feature Reconstruction) ---
class AttributeDecoder(nn.Module):
    """
    Decodes latent embeddings back to the original node feature space using an MLP.
    Applies Tanh activation to the output to control scale and prevent explosion.
    """
    def __init__(self, latent_dim: int, original_feature_dim: int, hidden_decoder_dim: Optional[int] = None):
        """
        Args:
            latent_dim: Dimensionality of the latent embeddings.
            original_feature_dim: Dimensionality of the original node features to reconstruct.
            hidden_decoder_dim: Dimensionality of the hidden layer in the MLP decoder.
                                Defaults to latent_dim if None.
        """
        super().__init__()
        if hidden_decoder_dim is None:
            hidden_decoder_dim = latent_dim

        self.mlp = nn.Sequential(
            Linear(latent_dim, hidden_decoder_dim),
            nn.ReLU(),
            #nn.ELU(),
            Linear(hidden_decoder_dim, original_feature_dim),
        )
        self.norm_layer = LayerNorm(original_feature_dim)
        self.reset_parameters()

    def reset_parameters(self):
         for layer in self.mlp:
             if hasattr(layer, 'reset_parameters'):
                 layer.reset_parameters()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.mlp(z)
        return self.norm_layer(x_hat)


# --- 5. Classifier MLP ---
class ClassifierMLP(nn.Module):
    """
    Simple MLP for binary classification based on the fused embedding.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, dropout: float = 0.5):
        """
        Args:
            input_dim: Dimensionality of the fused input embedding.
            hidden_dim: Dimensionality of the hidden layer.
            output_dim: Dimensionality of the output (1 for binary classification logits).
            dropout: Dropout rate.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            Linear(hidden_dim, output_dim) 
        )
        self.reset_parameters()

    def reset_parameters(self):
         for layer in self.mlp:
             if hasattr(layer, 'reset_parameters'):
                 layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fused input embedding [batch_size, input_dim].

        Returns:
            logits: Classification logits [batch_size, output_dim].
        """
        return self.mlp(x)

class MHA_CLSToken_FusionLayer(nn.Module):
    """
    Fuses view embeddings using a learnable [CLS] token and Multi-Head Self-Attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, 
                 ffn_dim_multiplier: int = 2, dropout: float = 0.1, 
                 output_dim: Optional[int] = None):
        """
        Args:
            embed_dim: Dimensionality of the input view embeddings.
            num_heads: Number of attention heads.
            ffn_dim_multiplier: Multiplier for the feed-forward layer's hidden dim.
            dropout: Dropout rate.
            output_dim: Final dimension of the fused embedding. Defaults to embed_dim.
        """
        super().__init__()
        self.output_dim = output_dim if output_dim is not None else embed_dim

        # 1. The learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 2. The Multi-Head Attention layer
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True 
        )
        
        # 3. A standard Feed-Forward Network (part of a Transformer block)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_dim_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_dim_multiplier, embed_dim)
        )

        # 4. Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 5. Optional final projection layer
        self.final_projection = nn.Linear(embed_dim, self.output_dim) if embed_dim != self.output_dim else nn.Identity()
        
    def forward(self, view_embeddings_stacked: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            view_embeddings_stacked: Tensor of view embeddings, shape [batch_size, num_views, embed_dim].
        
        Returns:
            fused_embedding: A single fused vector per patient, shape [batch_size, output_dim].
            attention_weights: None, as extracting them is complex and not the primary goal.
        """
        batch_size = view_embeddings_stacked.shape[0]

        # Prepend the CLS token to the sequence of view embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, view_embeddings_stacked), dim=1) # Shape: [batch_size, num_views + 1, embed_dim]

        # --- First part of Transformer Block: MHA + Residual + Norm ---
        # Self-attention: query, key, and value are all the same
        attn_output, _ = self.mha(x, x, x)
        # Residual connection
        x = x + attn_output
        x = self.norm1(x)

        # --- Second part of Transformer Block: FFN + Residual + Norm ---
        ffn_output = self.ffn(x)
        # Residual connection
        x = x + ffn_output
        x = self.norm2(x)
        
        # The final fused representation is the output of the CLS token (at position 0)
        cls_output = x[:, 0, :] # Shape: [batch_size, embed_dim]

        # Apply final projection
        fused_embedding = self.final_projection(cls_output)

        return fused_embedding, None # Return None for attention weights

class ProjectionHead(nn.Module):
    """
    Projects embeddings (typically mu from VAE) to a new space for contrastive learning.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimensionality of the input embeddings (e.g., d_embed).
            hidden_dim: Dimensionality of the hidden layer.
            output_dim: Dimensionality of the projected embeddings for CL.
            dropout: Dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            Linear(input_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            Linear(hidden_dim, output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, input_dim] or [num_nodes, input_dim].

        Returns:
            Projected and L2-normalized embeddings [batch_size, output_dim].
        """
        projected_x = self.net(x)
        return F.normalize(projected_x, p=2, dim=-1)
    

class FusionAndClassifierHead(nn.Module):
    """
    Fuses view embeddings (z_sampled) using a learnable [CLS] token, MHA,
    and then immediately classifies the resulting fused representation.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 classifier_hidden_dim: int,
                 ffn_dim_multiplier: int = 2,
                 fusion_dropout: float = 0.1,
                 classifier_dropout: float = 0.5):
        """
        Args:
            embed_dim: Dimensionality of the input view embeddings (z_sampled).
            num_heads: Number of attention heads for fusion.
            classifier_hidden_dim: Hidden dimension for the final classifier MLP.
            ffn_dim_multiplier: Multiplier for the feed-forward layer's hidden dim in the transformer block.
            fusion_dropout: Dropout for the fusion part (MHA, FFN).
            classifier_dropout: Dropout for the classifier part.
        """
        super().__init__()

        # --- Phần Fusion (Transformer Block) ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=fusion_dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_dim_multiplier),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(embed_dim * ffn_dim_multiplier, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # --- Phần Classifier (MLP Head) ---
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(embed_dim), # Thêm một LayerNorm để ổn định đầu vào cho classifier
            nn.Linear(embed_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(classifier_hidden_dim, 1) # Output là 1 logit duy nhất
        )

    def forward(self, view_embeddings_stacked: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            view_embeddings_stacked: Tensor of z_sampled from different views,
                                     shape [batch_size, num_views, embed_dim].
        
        Returns:
            logits: The final classification logits, shape [batch_size, 1].
            attention_weights: None.
        """
        batch_size = view_embeddings_stacked.shape[0]

        # --- 1. Fusion using Transformer Block ---
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, view_embeddings_stacked), dim=1)

        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        # Lấy biểu diễn tổng hợp từ [CLS] token
        cls_output = x[:, 0, :] # Shape: [batch_size, embed_dim]

        # --- 2. Classification ---
        logits = self.classifier_head(cls_output)

        return logits, cls_output

class RadiologyLesionAttentionAggregator(nn.Module):
    def __init__(self, lesion_feature_dim: int, patient_embed_dim: int,
                 attention_hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.lesion_feature_dim = lesion_feature_dim
        self.patient_embed_dim = patient_embed_dim
        
        if attention_hidden_dim is None:
            attention_hidden_dim = lesion_feature_dim

        # Attention mechanism: learns to score lesions
        # Takes individual lesion features
        self.attention_mlp = nn.Sequential(
            nn.Linear(lesion_feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1) 
        )

        if lesion_feature_dim != patient_embed_dim:
            self.output_projection = nn.Linear(lesion_feature_dim, patient_embed_dim)
        else:
            self.output_projection = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.norm_layer = nn.LayerNorm(patient_embed_dim) # Normalize the final patient embedding

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.attention_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if isinstance(self.output_projection, nn.Linear):
            self.output_projection.reset_parameters()
        self.norm_layer.reset_parameters()

    def forward(self, lesion_x: torch.Tensor, patient_to_lesion_edge_index: torch.Tensor,
                num_patients_in_batch: int) -> torch.Tensor:
        """
        Aggregates lesion features for patients using attention.

        Args:
            lesion_x: Tensor of lesion features [total_lesions_in_batch, lesion_feature_dim].
            patient_to_lesion_edge_index: Edge index [2, num_edges] connecting
                                           batch-local patient indices to batch-local lesion indices.
                                           edge_index[0] = batch_local_patient_idx
                                           edge_index[1] = batch_local_lesion_idx
            num_patients_in_batch: The number of unique patients in this batch for whom
                                   we need to produce aggregated embeddings.

        Returns:
            patient_radiology_embeddings: Tensor [num_patients_in_batch, patient_embed_dim].
                                          Contains aggregated features for patients who have lesions.
                                          For patients with no lesions, their rows will be zeros.
        """
        if lesion_x.numel() == 0 or patient_to_lesion_edge_index.numel() == 0:
            # No lesions in this batch, return zeros for all patients
            return torch.zeros((num_patients_in_batch, self.patient_embed_dim),
                               device=lesion_x.device, dtype=lesion_x.dtype)

        batch_local_patient_indices = patient_to_lesion_edge_index[0]
        batch_local_lesion_indices = patient_to_lesion_edge_index[1] 

        relevant_lesion_features = lesion_x[batch_local_lesion_indices]

        # 1. Calculate attention scores for each lesion
        attn_scores = self.attention_mlp(relevant_lesion_features)  # [num_batch_edges, 1]

        # 2. Apply softmax grouped by patient to get attention weights

        attn_scores_max_per_patient = scatter_max(attn_scores.squeeze(-1), batch_local_patient_indices, dim=0, dim_size=num_patients_in_batch)[0]
        attn_scores_stabilized = attn_scores.squeeze(-1) - attn_scores_max_per_patient[batch_local_patient_indices]
        
        attn_exp = torch.exp(attn_scores_stabilized)
        attn_exp_sum_per_patient = scatter_add(attn_exp, batch_local_patient_indices, dim=0, dim_size=num_patients_in_batch)
        
        attn_exp_sum_per_patient = attn_exp_sum_per_patient.clamp(min=1e-12) 
        
        alpha = attn_exp / attn_exp_sum_per_patient[batch_local_patient_indices] # [num_batch_edges]
        alpha = alpha.unsqueeze(-1) # [num_batch_edges, 1]

        # 3. Calculate weighted sum of lesion features for each patient
        weighted_lesion_features = relevant_lesion_features * alpha # [num_batch_edges, lesion_feature_dim]
        
        # Aggregate weighted features per patient
        aggregated_patient_features = scatter_add(
            weighted_lesion_features, batch_local_patient_indices, dim=0, dim_size=num_patients_in_batch
        ) # [num_patients_in_batch, lesion_feature_dim]

        # 4. Optional output projection and normalization
        projected_features = self.output_projection(aggregated_patient_features)
        projected_features = self.dropout(projected_features)
        normalized_features = self.norm_layer(projected_features)
        
        return normalized_features
    
class MuFusionTransformer(nn.Module):
    """
    Fuses multiple mu vectors from different modalities into a single, richer latent vector
    using a Transformer Encoder layer and a [CLS] token.
    """
    def __init__(self, d_embed: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        """
        Args:
            d_embed (int): The dimension of the mu vectors (e.g., 64).
            n_heads (int): The number of attention heads in the MHA.
            dim_feedforward (int): The dimension of the feed-forward network in the transformer.
        """
        super().__init__()
        self.d_embed = d_embed

        # The learnable [CLS] token that will act as the aggregator
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_embed))

        # The core Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Optional: Positional embeddings for the views
        # For a fixed small number of views (3), it might not be strictly necessary,
        # but it's good practice.
        self.positional_embedding = nn.Parameter(torch.randn(1, 4, d_embed)) # 1 for CLS + 3 for views

    def forward(self, mu_views: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu_views (torch.Tensor): A tensor containing the stacked mu vectors from different
                                     modalities. Shape: [batch_size, num_views, d_embed].

        Returns:
            torch.Tensor: The fused mu vector. Shape: [batch_size, d_embed].
        """
        batch_size = mu_views.shape[0]

        # 1. Prepend the [CLS] token to the sequence of mu vectors
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, mu_views), dim=1) # Shape: [batch_size, num_views + 1, d_embed]
        
        # 2. Add positional embeddings
        x = x + self.positional_embedding
        
        # 3. Pass through the Transformer Encoder
        transformer_output = self.transformer_encoder(x) # Shape: [batch_size, num_views + 1, d_embed]

        # 4. The output of the [CLS] token (at position 0) is the final fused representation
        mu_fused = transformer_output[:, 0, :] # Shape: [batch_size, d_embed]

        return mu_fused