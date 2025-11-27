import torch
import copy
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm.notebook import tqdm
from tqdm import tqdm
# Import các thành phần cần thiết
from models.gvae_model import GVAE
from models.ddpm import ConditionalDDPM, UnconditionalDDPM
from utils.data_utils import get_view_subgraph_and_features
from models.unet import DenoiseUNet
from sklearn.model_selection import train_test_split


def train_single_conditional_ddpm(
    ddpm_model: ConditionalDDPM, 
    latents: torch.Tensor,
    labels: torch.Tensor,
    ddpm_config: dict,
    device: torch.device
) -> tuple['ConditionalDDPM | None', 'MinMaxScaler | None']:

    # --- 1. Split Data ---
    val_split_ratio = ddpm_config.get('val_split_ratio', 0.15)
    try:
        train_latents, val_latents, train_labels, val_labels = train_test_split(
            latents.numpy(), labels.numpy(), test_size=val_split_ratio, 
            stratify=labels.numpy(), random_state=420
        )
    except ValueError:
        train_latents, val_latents, train_labels, val_labels = train_test_split(
            latents.numpy(), labels.numpy(), test_size=val_split_ratio, random_state=42
        )

    # --- 2. Scale Data ---
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_latents_scaled = torch.tensor(scaler.fit_transform(train_latents), dtype=torch.float32)
    val_latents_scaled = torch.tensor(scaler.transform(val_latents), dtype=torch.float32)
    
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    # --- 3. DataLoaders ---
    class LatentDataset(torch.utils.data.Dataset):
        def __init__(self, latents_data, labels_data):
            self.latents = latents_data
            self.labels = labels_data
        def __len__(self): return len(self.latents)
        def __getitem__(self, idx): return self.latents[idx], self.labels[idx]

    train_loader = torch.utils.data.DataLoader(
        LatentDataset(train_latents_scaled, train_labels), 
        batch_size=ddpm_config['batch_size'], shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        LatentDataset(val_latents_scaled, val_labels), 
        batch_size=ddpm_config['batch_size'], shuffle=False
    )
    
    # --- 4. Training Setup ---
    optimizer = torch.optim.AdamW(ddpm_model.parameters(), lr=ddpm_config['lr'])
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    patience = ddpm_config.get('early_stopping_patience', 200)
    p_uncond = ddpm_config.get('p_uncond', 0.15) 

    pbar = tqdm(range(ddpm_config['epochs']), desc="Training DDPM")

    for epoch in pbar:
        # -- Train --
        ddpm_model.train()
        total_train_loss = 0
        for x0, y in train_loader:
            optimizer.zero_grad()
            x0 = x0.to(device)
            
            # SHIFT: 0->1, 1->2 (leaving 0 for unconditional)
            y = y.to(device) + 1 
            
            # DROPOUT: Set some to 0
            if p_uncond > 0:
                mask = torch.rand(y.shape[0], device=device) < p_uncond
                y[mask] = 0 
            
            loss = ddpm_model.loss(x0, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm_model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # -- Validation --
        ddpm_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x0_val, y_val in val_loader:
                x0_val = x0_val.to(device)
                # Validate on specific class performance (Shifted)
                y_val = y_val.to(device) + 1
                
                val_loss = ddpm_model.loss(x0_val, y_val)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

        pbar.set_postfix(t_loss=f"{avg_train_loss:.4f}", v_loss=f"{avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(ddpm_model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

    if best_model_state is None: return ddpm_model, scaler
    ddpm_model.load_state_dict(best_model_state)
    return ddpm_model, scaler


def train_single_unconditional_ddpm(
    latents: torch.Tensor,
    ddpm_config: dict,
    device: torch.device
) -> tuple[UnconditionalDDPM, MinMaxScaler]:
    """
    Trains a single UNCONDITIONAL DDPM on a specific set of latent vectors (one class).
    """
    if latents.shape[0] < ddpm_config['batch_size']:
        print(
            f"Warning: Not enough samples ({latents.shape[0]}) to train DDPM. Skipping.")
        return None, None

    scaler = MinMaxScaler(feature_range=(-1, 1))
    latents_scaled = torch.tensor(scaler.fit_transform(
        latents.cpu().numpy()), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(latents_scaled)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=ddpm_config['batch_size'], shuffle=True, drop_last=True
    )

    denoising_net = DenoiseUNet(
        latent_dim=ddpm_config['latent_dim'],
        num_classes=None  # Báo cho UNet biết đây là model vô điều kiện
    ).to(device)

    ddpm_model = UnconditionalDDPM(
        denoise_fn=denoising_net,
        latent_dim=ddpm_config['latent_dim'],
        timesteps=ddpm_config['timesteps']
    ).to(device)

    optimizer = torch.optim.AdamW(
        ddpm_model.parameters(), lr=ddpm_config['lr'])
    pbar = tqdm(range(ddpm_config['epochs']),
                desc=f"Training Unconditional DDPM")

    for epoch in pbar:
        total_loss = 0
        for (x0,) in dataloader:
            optimizer.zero_grad()
            x0 = x0.to(device)
            loss = ddpm_model.loss(x0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return ddpm_model, scaler
