import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from tqdm.notebook import tqdm as tqdm_notebook
import sys
from models.unet import DenoiseUNet
from typing import Optional

# class SinusoidalPositionEmbeddings(nn.Module):
#     """
#     Module for generating sinusoidal position embeddings for timesteps.
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings

# class Attention(nn.Module):
#     """
#     A simple self-attention mechanism.
#     """
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head**-0.5
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim)

#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, "b (h d) -> b h d", h=self.heads), qkv)
#         dots = torch.einsum("b h i, b h j -> b h i j", q, k) * self.scale
#         attn = dots.softmax(dim=-1)
#         out = torch.einsum("b h i j, b h j -> b h i", attn, v)
#         out = rearrange(out, "b h d -> b (h d)")
#         return self.to_out(out)

# class MLPBlock(nn.Module):
#     """ A single block of MLP with LayerNorm, Attention, and FeedForward """
#     def __init__(self, dim):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = Attention(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.SiLU(),
#             nn.Linear(dim * 4, dim)
#         )

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))
#         return x

# # --- 1. Denoising Network ---
# class DenoisingNetwork(nn.Module):
#     """
#     The core network (epsilon_theta) that predicts noise from a noisy input.
#     This architecture uses MLP with self-attention as requested.
#     """
#     def __init__(self, latent_dim, num_classes, cond_drop_prob=0.1, num_layers=4):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.cond_drop_prob = cond_drop_prob

#         # Embeddings for time and condition
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(latent_dim),
#             nn.Linear(latent_dim, latent_dim * 4),
#             nn.SiLU(),
#             nn.Linear(latent_dim * 4, latent_dim),
#         )
#         self.class_emb = nn.Embedding(num_classes + 1, latent_dim) # +1 for null condition

#         # Main MLP blocks
#         self.input_proj = nn.Linear(latent_dim, latent_dim)
#         self.blocks = nn.ModuleList([MLPBlock(latent_dim) for _ in range(num_layers)])
#         self.output_proj = nn.Linear(latent_dim, latent_dim)

#     def forward(self, x, time, y):
#         # Time embedding
#         t_emb = self.time_mlp(time)

#         # Class embedding with classifier-free guidance dropout
#         if self.training and self.cond_drop_prob > 0:
#             mask = torch.rand(y.shape[0], device=y.device) < self.cond_drop_prob
#             y[mask] = 0 # 0 is the index for the null/unconditional token
        
#         c_emb = self.class_emb(y)

#         # Combine inputs
#         x = self.input_proj(x)
#         x = x + t_emb + c_emb
        
#         # Pass through blocks
#         for block in self.blocks:
#             x = block(x)

#         return self.output_proj(x)


# --- 2. Conditional DDPM Wrapper ---
class ConditionalDDPM(nn.Module):
    def __init__(self, denoise_fn, latent_dim, timesteps=1000, beta_schedule='linear',  cond_drop_prob=0.1):
        super().__init__()
        denoise_fn.cond_drop_prob = cond_drop_prob
        self.denoise_fn = denoise_fn
        self.latent_dim = latent_dim
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def forward_process(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt, noise

    def loss(self, x0, y):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device).long()
        
        xt, noise = self.forward_process(x0, t)
        # Truyền y vào DenoiseUNet
        predicted_noise = self.denoise_fn(xt, t, y)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, y, guidance_scale=3.0):
        batch_size = y.shape[0]
        device = self.betas.device
        
        # Start with random noise
        x = torch.randn((batch_size, self.latent_dim), device=device)
        
        # Unconditional context (null token)
        y_uncond = torch.zeros_like(y)

        for i in tqdm_notebook(reversed(range(self.timesteps)), desc="DDPM Sampling", total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Predict noise with and without condition
            pred_noise_cond = self.denoise_fn(x, t, y)
            pred_noise_uncond = self.denoise_fn(x, t, y_uncond)
            
            # Classifier-Free Guidance
            pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            
            # Denoising step
            alpha_t = (1. - self.betas)[t][:, None]
            alpha_cumprod_t = self.alphas_cumprod[t][:, None]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
            
            pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * pred_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0.clamp_(-1., 1.) # Optional clamping
            
            # Get mean of p(xt-1 | xt, x0)
            posterior_mean = (torch.sqrt(alpha_cumprod_t) * self.betas[t][:, None] * pred_x0 + torch.sqrt(alpha_t) * (1 - alpha_cumprod_t) * x) / (1 - alpha_cumprod_t)
            
            if i == 0:
                x = posterior_mean
            else:
                posterior_variance = self.betas[t][:, None] # Simplified variance
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(posterior_variance) * noise
                
        return x
    

# class UnconditionalDenoisingNetwork(nn.Module):
#     """
#     The core network for an UNCONDITIONAL DDPM.
#     It predicts noise without any class guidance.
#     """
#     def __init__(self, latent_dim: int, num_layers: int = 4):
#         super().__init__()
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(latent_dim),
#             nn.Linear(latent_dim, latent_dim * 4),
#             nn.SiLU(),
#             nn.Linear(latent_dim * 4, latent_dim),
#         )
#         self.input_proj = nn.Linear(latent_dim, latent_dim)
#         self.blocks = nn.ModuleList([MLPBlock(latent_dim) for _ in range(num_layers)])
#         self.output_proj = nn.Linear(latent_dim, latent_dim)

#     def forward(self, x, time):
#         t_emb = self.time_mlp(time)
#         x = self.input_proj(x) + t_emb
#         for block in self.blocks:
#             x = block(x)
#         return self.output_proj(x)


class UnconditionalDDPM(nn.Module):
    """
    A wrapper for an UNCONDITIONAL Denoising Diffusion Probabilistic Model.
    This is simpler as it does not handle any class conditions.
    """
    def __init__(self,
                 denoise_fn: DenoiseUNet,
                 latent_dim: int,
                 timesteps: int = 1000,
                 beta_schedule: str = 'linear'):
        super().__init__()
        
        self.denoise_fn = denoise_fn
        self.latent_dim = latent_dim
        self.timesteps = timesteps

        # --- Setup Beta Schedule (same as conditional) ---
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', F.pad(alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1. - self.alphas_cumprod_prev) / (1. - alphas_cumprod))

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """ Adds noise to an image x0 at timestep t. """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(0, t).reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(-1, 1)
        
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """ Calculates the training loss (MSE). Note: no `y` argument. """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device).long()
        
        xt, noise = self.forward_process(x0, t)
        
        # Predict the noise using the U-Net. Pass `y=None`.
        predicted_noise = self.denoise_fn(xt, t, y=None)
        
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        """ Generates new samples (unconditionally). """
        device = self.betas.device
        
        x = torch.randn((num_samples, self.latent_dim), device=device)

        for i in tqdm_notebook(reversed(range(self.timesteps)), desc="Unconditional Sampling", total=self.timesteps, leave=False):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            # Simple prediction, no guidance
            pred_noise = self.denoise_fn(x, t, y=None)
            
            alpha_t = self.alphas.gather(0, t).reshape(-1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(-1, 1)
            
            pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * pred_noise) / self.sqrt_alphas_cumprod.gather(0, t).reshape(-1, 1)
            
            alpha_cumprod_prev_t = self.alphas_cumprod_prev.gather(0, t).reshape(-1, 1)
            posterior_mean = (
                (torch.sqrt(alpha_cumprod_prev_t) * self.betas.gather(0, t).reshape(-1, 1) * pred_x0) +
                (torch.sqrt(alpha_t) * (1. - alpha_cumprod_prev_t) * x)
            ) / (1. - self.alphas_cumprod.gather(0, t).reshape(-1, 1))

            if i == 0:
                x = posterior_mean
            else:
                posterior_variance_t = self.posterior_variance.gather(0, t).reshape(-1, 1)
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(posterior_variance_t) * noise
                
        return x