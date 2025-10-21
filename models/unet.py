# /models/unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple

# --- Helper Modules (không thay đổi) ---

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, dropout_prob: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim is not None else None

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(out_channels, out_channels)
        )

        self.res_conv = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        if self.mlp is not None and t_emb is not None:
            time_out = self.mlp(t_emb)
            h = h + time_out
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, in_channels, n_heads=4, head_dim=32):
        super().__init__()
        self.n_heads = n_heads
        self.scale = head_dim ** -0.5
        hidden_dim = n_heads * head_dim
        self.norm = nn.GroupNorm(8, in_channels)
        self.to_qkv = nn.Linear(in_channels, hidden_dim * 3)
        self.to_out = nn.Linear(hidden_dim, in_channels)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = x.unsqueeze(1)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out).squeeze(1)
        return out + res

# --- The Main U-Net Denoising Network (PHIÊN BẢN ĐÃ SỬA LỖI) ---

class DenoiseUNet(nn.Module):
    """
    A U-Net architecture adapted for 1D vector data (latent vectors), used as the
    denoising function in a Denoising Diffusion Probabilistic Model (DDPM).

    This model can be conditional (on class labels `y`) or unconditional.
    """
    def __init__(self,
                 latent_dim: int,
                 num_classes: Optional[int] = None,
                 dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 dropout_prob: float = 0.5):
        """
        Args:
            latent_dim (int): The dimensionality of the input latent vectors.
            num_classes (Optional[int]): The number of classes for conditional generation.
                                         If None, the model is unconditional. `0` is reserved
                                         for the unconditional token in classifier-free guidance.
            dim_mults (Tuple[int, ...]): Multipliers for the latent_dim to define the
                                         channel dimensions at each U-Net level.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.is_conditional = num_classes is not None
        self.dropout_prob = dropout_prob
        
        # --- Time and Condition Embeddings ---
        time_dim = latent_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(latent_dim),
            nn.Linear(latent_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if self.is_conditional:
            # num_classes should include the unconditional token `0`
            self.class_emb = nn.Embedding(num_classes, time_dim)

        # --- U-Net Architecture ---
        dims = [latent_dim] + [latent_dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.init_conv = nn.Linear(latent_dim, dims[0])
        
        # -- Downsampling Path --
        self.down_blocks = nn.ModuleList([
            nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, dropout_prob=dropout_prob),
                Attention(dim_out)
            ]) for dim_in, dim_out in in_out
        ])

        # -- Bottleneck --
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, dropout_prob=dropout_prob)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, dropout_prob=dropout_prob)

        # --- Up blocks ---
        self.up_blocks = nn.ModuleList([
            nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim, dropout_prob=dropout_prob),
                Attention(dim_in)
            ]) for dim_in, dim_out in reversed(in_out)
        ])
        
        
        # -- Final Block --
        # Takes the output from the last up-block (dims[0]) and the initial skip connection (dims[0])
        self.final_res_block = ResnetBlock(dims[0] * 2, dims[0], time_emb_dim=time_dim, dropout_prob=dropout_prob)
        self.final_linear = nn.Linear(dims[0], latent_dim)
        
        # Optional classifier-free guidance dropout
        self.cond_drop_prob = 0.1
        
    def forward(self, x: torch.Tensor, time: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the denoising U-Net.

        Args:
            x (torch.Tensor): The noisy input latent vector of shape [batch_size, latent_dim].
            time (torch.Tensor): The timestep for each sample in the batch, shape [batch_size].
            y (Optional[torch.Tensor]): The class labels for conditional generation, shape [batch_size].

        Returns:
            torch.Tensor: The predicted noise, of the same shape as x.
        """
        if self.is_conditional and y is None:
            raise ValueError("Class labels `y` must be provided for a conditional model.")

        # 1. Initial convolution and save for final skip connection
        x = self.init_conv(x)
        initial_skip = x.clone()

        # 2. Get time and class embeddings
        t_emb = self.time_mlp(time)
        if self.is_conditional:
            # During training, apply classifier-free guidance dropout
            if self.training and hasattr(self, 'cond_drop_prob') and self.cond_drop_prob > 0:
                 mask = torch.rand(y.shape[0], device=y.device) < self.cond_drop_prob
                 # Set masked labels to 0, the index for the unconditional token
                 y[mask] = 0 
            t_emb = t_emb + self.class_emb(y)

        # 3. Downsampling path
        skip_connections = []
        for res_block, attn_block in self.down_blocks:
            x = res_block(x, t_emb)
            x = attn_block(x)
            skip_connections.append(x)
        
        # 4. Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        # 5. Upsampling path
        for res_block, attn_block in self.up_blocks:
            # Pop the corresponding skip connection from the down path
            skip = skip_connections.pop()
            # Concatenate along the feature dimension
            x = torch.cat((x, skip), dim=1) 
            x = res_block(x, t_emb)
            x = attn_block(x)

        # 6. Final block
        # Concatenate with the very first skip connection
        x = torch.cat((x, initial_skip), dim=1)
        
        # Explicitly call the final ResnetBlock with the time embedding
        x = self.final_res_block(x, t_emb)
        
        # Final projection to the original latent dimension
        return self.final_linear(x)