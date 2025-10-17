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
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels)) if time_emb_dim is not None else None
        self.block1 = nn.Sequential(nn.GroupNorm(8, in_channels), nn.SiLU(), nn.Linear(in_channels, out_channels))
        self.block2 = nn.Sequential(nn.GroupNorm(8, out_channels), nn.SiLU(), nn.Linear(out_channels, out_channels))
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
    def __init__(self,
                 latent_dim: int,
                 num_classes: Optional[int] = None,
                 dim_mults: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.is_conditional = num_classes is not None
        
        # --- Time and Condition Embeddings ---
        time_dim = latent_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(latent_dim),
            nn.Linear(latent_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if self.is_conditional:
            self.class_emb = nn.Embedding(num_classes, time_dim)

        # --- U-Net Architecture ---
        dims = [latent_dim] + [latent_dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.init_conv = nn.Linear(latent_dim, dims[0])
        
        # -- Downsampling Path --
        self.down_blocks = nn.ModuleList([])
        for in_c, out_c in in_out:
            self.down_blocks.append(nn.ModuleList([
                ResnetBlock(in_c, out_c, time_emb_dim=time_dim),
                Attention(out_c)
            ]))

        # -- Bottleneck --
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # -- Upsampling Path --
        self.up_blocks = nn.ModuleList([])
        for in_c, out_c in reversed(in_out):
            # in_c: dimension of the output of this block (e.g., 128)
            # out_c: dimension of the input from below and of the skip connection (e.g., 256)
            self.up_blocks.append(nn.ModuleList([
                # Input is doubled: from below (out_c) + skip connection (out_c)
                ResnetBlock(out_c * 2, in_c, time_emb_dim=time_dim),
                Attention(in_c)
            ]))

        # =================== FIX STARTS HERE (1/2) ===================
        # Tách ResnetBlock ra khỏi nn.Sequential
        self.final_res_block = ResnetBlock(dims[0], dims[0], time_emb_dim=time_dim)
        self.final_linear = nn.Linear(dims[0], latent_dim)
        # ===================  FIX ENDS HERE (1/2)  ===================
        
        # -- Final Block --
        self.final_conv = nn.Sequential(
            ResnetBlock(dims[0], dims[0], time_emb_dim=time_dim),
            nn.Linear(dims[0], latent_dim)
        )
        
    def forward(self, x: torch.Tensor, time: torch.Tensor, y: Optional[torch.Tensor] = None):
        if self.is_conditional and y is None:
            raise ValueError("Class labels `y` must be provided for a conditional model.")

        x = self.init_conv(x)

        t_emb = self.time_mlp(time)
        if self.is_conditional:
            if self.training and hasattr(self, 'cond_drop_prob') and self.cond_drop_prob > 0:
                 mask = torch.rand(y.shape[0], device=y.device) < self.cond_drop_prob
                 y[mask] = 0 # 0 is the index for the null/unconditional token
            t_emb = t_emb + self.class_emb(y)

        # Down path
        skip_connections = []
        for res, attn in self.down_blocks:
            x = res(x, t_emb)
            x = attn(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        # Up path
        for res, attn in self.up_blocks:
            skip = skip_connections.pop()
            x = torch.cat((x, skip), dim=1) # Concatenate along channel dimension
            x = res(x, t_emb)
            x = attn(x)

        # Final
        # =================== FIX STARTS HERE (2/2) ===================
        # Gọi ResnetBlock một cách tường minh và truyền vào t_emb
        x = self.final_res_block(x, t_emb)
        # Sau đó mới đưa qua lớp Linear cuối cùng
        return self.final_linear(x)
        # ===================  FIX ENDS HERE (2/2)  ===================