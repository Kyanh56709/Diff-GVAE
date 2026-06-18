import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import sys
import math
from models.unet import DenoiseUNet
from typing import Optional
from tqdm import tqdm
from itertools import combinations
from typing import Optional, List, Tuple

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # Clip beta để tránh các điểm kỳ dị toán học ở những timestep cuối cùng
    return torch.clip(betas, 0.0001, 0.999)

# --- 2. Conditional DDPM Wrapper ---
class ConditionalDDPM(nn.Module):
    def __init__(self, denoise_fn, latent_dim, timesteps=1000, beta_schedule='linear',  cond_drop_prob=0.1):
        super().__init__()
        # This part is fine
        if hasattr(denoise_fn, 'cond_drop_prob'):
            denoise_fn.cond_drop_prob = cond_drop_prob
        self.denoise_fn = denoise_fn
        self.latent_dim = latent_dim
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
    
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod + 1e-8)
        self.register_buffer('posterior_variance', posterior_variance)

    def forward_process(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod.gather(0, t).view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).view(-1, 1)

        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt, noise

    def loss(self, x0, y):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device).long()
        xt, noise = self.forward_process(x0, t)
        predicted_noise = self.denoise_fn(xt, t, y)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, y, guidance_scale=3.0):
        # <<< REPLACEMENT START: The Numerically Stable DDPM Sampling Loop >>>
        batch_size = y.shape[0]
        device = self.betas.device
        
        # Start from pure noise
        x = torch.randn((batch_size, self.latent_dim), device=device)
        
        # Unconditional context for guidance (even if scale is 0, we need the placeholder)
        y_uncond = torch.zeros_like(y)

        for i in tqdm(reversed(range(self.timesteps)), desc="DDPM Sampling", total=self.timesteps, leave=False):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # 1. Predict noise with and without condition
            pred_noise_cond = self.denoise_fn(x, t, y)
            pred_noise_uncond = self.denoise_fn(x, t, y_uncond)

            # 2. Classifier-Free Guidance (will do nothing if guidance_scale is 0)
            pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

            # 3. Get variables for this timestep
            alpha_t = self.alphas.gather(0, t).reshape(-1, 1)
            alpha_cumprod_t = self.alphas_cumprod.gather(0, t).reshape(-1, 1)
            alpha_cumprod_prev_t = self.alphas_cumprod_prev.gather(0, t).reshape(-1, 1)
            beta_t = self.betas.gather(0, t).reshape(-1, 1)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(-1, 1)
            
            # 4. Predict the original x_0 (the "denoised" latent)
            # This is a key step in the stable formulation
            pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * pred_noise) / torch.sqrt(alpha_cumprod_t)
            
            # --- Optional: Dynamic Thresholding or Clamping ---
            # For StandardScaler, you don't typically need to clamp pred_x0 here,
            # but if you were using MinMaxScaler, this would be where you do it.
            # pred_x0.clamp_(-1., 1.)

            # 5. Calculate the posterior mean of q(x_{t-1} | x_t, x_0)
            # This formula is derived from the full posterior distribution and is more stable
            posterior_mean_coef1 = (torch.sqrt(alpha_cumprod_prev_t) * beta_t) / (1. - alpha_cumprod_t)
            posterior_mean_coef2 = (torch.sqrt(alpha_t) * (1. - alpha_cumprod_prev_t)) / (1. - alpha_cumprod_t)
            posterior_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x

            # 6. Get the posterior variance
            posterior_variance_t = self.posterior_variance.gather(0, t).reshape(-1, 1)

            # 7. Sample x_{t-1}
            noise = torch.randn_like(x) if i > 0 else 0 # No noise at the final step
            x = posterior_mean + torch.sqrt(posterior_variance_t) * noise
        
        # The final result is x_0, which is the mean at the last step.
        # No final clamp is needed as the loop produces the final denoised output directly.
        return x

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
        self.register_buffer('alphas_cumprod_prev', F.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', betas *
                             (1. - self.alphas_cumprod_prev) / (1. - alphas_cumprod))

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """ Adds noise to an image x0 at timestep t. """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(
            0, t).reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
            0, t).reshape(-1, 1)

        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """ Calculates the training loss (MSE). Note: no `y` argument. """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,),
                          device=x0.device).long()

        xt, noise = self.forward_process(x0, t)

        # Predict the noise using the U-Net. Pass `y=None`.
        predicted_noise = self.denoise_fn(xt, t, y=None)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        """ Generates new samples (unconditionally). """
        device = self.betas.device

        x = torch.randn((num_samples, self.latent_dim), device=device)

        for i in tqdm(reversed(range(self.timesteps)), desc="Unconditional Sampling", total=self.timesteps, leave=False):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            # Simple prediction, no guidance
            pred_noise = self.denoise_fn(x, t, y=None)

            alpha_t = self.alphas.gather(0, t).reshape(-1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
                0, t).reshape(-1, 1)

            pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * pred_noise) / \
                self.sqrt_alphas_cumprod.gather(0, t).reshape(-1, 1)

            alpha_cumprod_prev_t = self.alphas_cumprod_prev.gather(
                0, t).reshape(-1, 1)
            posterior_mean = (
                (torch.sqrt(alpha_cumprod_prev_t) * self.betas.gather(0, t).reshape(-1, 1) * pred_x0) +
                (torch.sqrt(alpha_t) * (1. - alpha_cumprod_prev_t) * x)
            ) / (1. - self.alphas_cumprod.gather(0, t).reshape(-1, 1))

            if i == 0:
                x = posterior_mean
            else:
                posterior_variance_t = self.posterior_variance.gather(
                    0, t).reshape(-1, 1)
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(posterior_variance_t) * noise

        return x

    @torch.no_grad()
    def evaluation_loss(self, x0: torch.Tensor, timesteps_to_eval: torch.Tensor,
                        noise_seed: int = 0) -> torch.Tensor:
        """
        Calculates a stable, deterministic loss for evaluation purposes.
        Averages the MSE loss across a specified set of timesteps.
        """
        total_loss = 0
        for t in timesteps_to_eval:
            t_batch = torch.full(
                (x0.shape[0],), t, device=x0.device, dtype=torch.long)
            generator = torch.Generator(device=x0.device)
            generator.manual_seed(noise_seed + int(t.item()))
            noise = torch.randn(
                x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
            )
            xt, noise = self.forward_process(x0, t_batch, noise=noise)
            predicted_noise = self.denoise_fn(xt, t_batch, y=None)
            total_loss += F.mse_loss(predicted_noise, noise)

        return total_loss / len(timesteps_to_eval)

class AdaGN(nn.Module):
    """
    Adaptive Group Normalization.
    Injects time/class embeddings by scaling and shifting the normalized signal.
    """
    def __init__(self, dim: int, emb_dim: int, num_groups: int = 8):
        super().__init__()
        # Ensure num_groups divides dim; fall back to a valid divisor
        num_groups = _find_valid_num_groups(dim, num_groups)
        self.group_norm = nn.GroupNorm(num_groups, dim)
        self.silu = nn.SiLU()
        # Projects embedding to scale (gamma) and shift (beta)
        self.projection = nn.Linear(emb_dim, dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Dim], emb: [Batch, Emb_Dim]
        
        # 1. Regress scale and shift from time embedding
        emb_out = self.projection(self.silu(emb))
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        
        # 2. Normalize x
        x = self.group_norm(x)
        
        # 3. Modulate
        # (1 + scale) ensures we start close to identity if scale is small
        x = x * (1 + scale) + shift
        return x


def _find_valid_num_groups(dim: int, desired: int = 8) -> int:
    """Return the largest number of groups <= desired that evenly divides dim."""
    for g in range(desired, 0, -1):
        if dim % g == 0:
            return g
    return 1  # Fallback: no grouping


def _round_up_to_lcm(value: int, num_modalities: int, n_heads: int) -> int:
    """
    Round `value` UP to the nearest multiple of lcm(num_modalities, n_heads)
    so that hidden_dim splits evenly across modalities AND each modality-stream
    dimension splits evenly across attention heads.
    """
    import math
    lcm = (num_modalities * n_heads) // math.gcd(num_modalities, n_heads)
    return ((value + lcm - 1) // lcm) * lcm


# --- 1. Positional Embedding (Standard) ---
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

# --- 2. The DCA Block (Robust for 1D vectors) ---
class DCA_Block(nn.Module):
    def __init__(self, d_embed: int, n_heads: int, num_modalities: int, dropout: float = 0.1):
        super().__init__()
        assert d_embed % n_heads == 0, (
            f"DCA_Block: d_embed ({d_embed}) must be divisible by n_heads ({n_heads}). "
            f"This is determined by hidden_dim // num_modalities. Adjust hidden_dim or n_heads."
        )
        self.num_modalities = num_modalities
        
        # Cross-attention layers
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_embed, n_heads, dropout=dropout, batch_first=True)
            for _ in range(num_modalities * (num_modalities - 1))
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(d_embed) for _ in range(num_modalities)])
        
        # FFNs
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_embed, d_embed * 4),
                nn.GELU(),
                nn.Dropout(dropout), # Use the passed dropout
                nn.Linear(d_embed * 4, d_embed),
                nn.Dropout(dropout)
            ) for _ in range(num_modalities)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d_embed) for _ in range(num_modalities)])

    def forward(self, streams: List[torch.Tensor]) -> List[torch.Tensor]:
        # Input: List of [Batch, Dim]
        
        # 1. Prepare Inputs
        # Use a frozen copy for Keys/Values to ensure symmetry
        seq_streams = [s.unsqueeze(1) for s in streams] 
        
        # We will accumulate updates here instead of mutating sequentially
        attention_updates = [torch.zeros_like(s) for s in seq_streams]
        
        attn_idx = 0
        
        # 2. Calculate All-Pairs Attention
        for i, j in combinations(range(self.num_modalities), 2):
            # i attends to j
            out_i, _ = self.cross_attentions[attn_idx](
                query=seq_streams[i], key=seq_streams[j], value=seq_streams[j]
            )
            attention_updates[i] += out_i
            attn_idx += 1

            # j attends to i
            out_j, _ = self.cross_attentions[attn_idx](
                query=seq_streams[j], key=seq_streams[i], value=seq_streams[i]
            )
            attention_updates[j] += out_j
            attn_idx += 1
            
        # 3. Apply Updates & FFN
        final_streams = []
        for i in range(self.num_modalities):
            # Squeeze back to [Batch, Dim]
            update = attention_updates[i].squeeze(1)
            original = streams[i]
            
            # Residual 1 (Attention)
            x = self.norms[i](original + update)
            
            # Residual 2 (FFN)
            x = self.ffn_norms[i](x + self.ffns[i](x))
            
            final_streams.append(x)
            
        return final_streams

# --- 3. Multimodal Linear Block ---
class MultimodalResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, time_emb_dim: int, 
                 num_modalities: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.d_embed_out = out_channels // num_modalities

        assert in_channels % num_modalities == 0, (
            f"in_channels ({in_channels}) must be divisible by num_modalities ({num_modalities})"
        )
        assert out_channels % num_modalities == 0, (
            f"out_channels ({out_channels}) must be divisible by num_modalities ({num_modalities})"
        )
        assert self.d_embed_out % n_heads == 0, (
            f"out_channels // num_modalities ({self.d_embed_out}) must be divisible by "
            f"n_heads ({n_heads}). Adjust hidden_dim, num_modalities, or n_heads so that "
            f"hidden_dim / num_modalities is a multiple of n_heads."
        )
        
        # --- Pre-DCA: AdaGN -> Activation -> Linear ---
        self.adagn1_layers = nn.ModuleList([
            AdaGN(in_channels // num_modalities, time_emb_dim) 
            for _ in range(num_modalities)
        ])
        
        self.block1_linear = nn.ModuleList([
             nn.Sequential(
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels // num_modalities, self.d_embed_out)
            ) for _ in range(num_modalities)
        ])

        # --- Interaction ---
        self.dca = DCA_Block(self.d_embed_out, n_heads, num_modalities, dropout=dropout)
        
        # --- Post-DCA: AdaGN -> Activation -> Linear ---
        self.adagn2_layers = nn.ModuleList([
            AdaGN(self.d_embed_out, time_emb_dim)
            for _ in range(num_modalities)
        ])
        
        self.block2_linear = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_embed_out, self.d_embed_out)
            ) for _ in range(num_modalities)
        ])
        
        # Output projection for residual matching
        self.res_conv = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_channels]
        
        # 1. Split streams
        chunk_size = x.shape[1] // self.num_modalities
        streams = list(torch.split(x, chunk_size, dim=1))
        
        # 2. Block 1: AdaGN + Linear
        processed_streams = []
        for i in range(self.num_modalities):
            # Inject time into the norm
            h = self.adagn1_layers[i](streams[i], t_emb)
            h = self.block1_linear[i](h)
            processed_streams.append(h)
        
        # 3. DCA Fusion
        interacted_streams = self.dca(processed_streams)
        
        # 4. Block 2: AdaGN + Linear
        final_streams = []
        for i in range(self.num_modalities):
            # Inject time again (very important for diffusion!)
            h = self.adagn2_layers[i](interacted_streams[i], t_emb)
            h = self.block2_linear[i](h)
            final_streams.append(h)

        # 5. Recombine
        output = torch.cat(final_streams, dim=1)
        
        return output + self.res_conv(x)
    
class DenoiseDCAMLP(nn.Module):
    def __init__(self, 
                 latent_dim_per_modality: int,
                 num_modalities: int,
                 num_classes: Optional[int] = None,
                 hidden_dim_multiplier: int = 4,
                 num_layers: int = 4,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_modalities = num_modalities
        input_dim = latent_dim_per_modality * num_modalities
        
        # --- FIX: Round hidden_dim to be divisible by BOTH num_modalities and n_heads ---
        # This guarantees hidden_dim / num_modalities is a whole number AND
        # that whole number is divisible by n_heads (required by MultiheadAttention).
        raw_hidden_dim = input_dim * hidden_dim_multiplier
        self.hidden_dim = _round_up_to_lcm(raw_hidden_dim, num_modalities, n_heads)
        
        per_modality_dim = self.hidden_dim // num_modalities
        assert per_modality_dim % n_heads == 0, (
            f"Bug in dimension rounding: per-modality dim {per_modality_dim} "
            f"is not divisible by n_heads {n_heads}"
        )
        
        # Time & Class Embeddings
        time_dim = self.hidden_dim // 2
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(input_dim),
            nn.Linear(input_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.is_conditional = num_classes is not None
        if self.is_conditional:
            # We reserve 0 for the null class (unconditioned)
            # So actual classes start at 1. Total vocab = num_classes + 1
            self.class_emb = nn.Embedding(num_classes + 1, time_dim)
            self.null_class_idx = 0 
            self.cond_drop_prob = 0.1

        # Input Projection
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)

        # Layers
        self.layers = nn.ModuleList([
            MultimodalResnetBlock(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                time_emb_dim=time_dim,
                num_modalities=num_modalities,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final Output Projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embeddings
        t_emb = self.time_mlp(time)
        
        if self.is_conditional and y is not None:
            # Shift y by 1 because 0 is reserved for null
            y_shifted = y + 1
            
            if self.training:
                # Classifier-free guidance dropout
                mask = torch.rand(y.shape[0], device=y.device) < self.cond_drop_prob
                y_shifted[mask] = self.null_class_idx # Set to null token
            
            c_emb = self.class_emb(y_shifted)
            t_emb = t_emb + c_emb

        # Forward
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, t_emb)

        return self.output_proj(x)
