import torch
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm
# Import các thành phần cần thiết
from models.gvae_model import GVAE
from models.ddpm import ConditionalDDPM, UnconditionalDDPM
from utils.data_utils import get_view_subgraph_and_features
from models.unet import DenoiseUNet

# def train_latent_ddpm(gvae_model: GVAE,
#                         full_data: torch.utils.data.Dataset,
#                         ddpm_config: dict,
#                         device: torch.device):
#     """
#     Trains a Conditional DDPM on the latent space of a pre-trained GVAE model.
#     """
#     print("\n--- Step 1: Extracting latent vectors (μ) from the frozen GVAE ---")
#     gvae_model.eval() # Ensure model is in eval mode

#     all_patient_indices = torch.arange(full_data['patient'].num_nodes).to(device)
#     all_labels = full_data['patient']['binary_label']

#     with torch.no_grad():
#         _, vae_outputs, _, _ = gvae_model(full_data, all_patient_indices)

#     # Fuse mu vectors by averaging available views for each patient
#     fused_mus = torch.zeros(full_data['patient'].num_nodes, ddpm_config['latent_dim']).to(device)
#     counts = torch.zeros(full_data['patient'].num_nodes, 1).to(device)

#     for view in gvae_model.views:
#         _, _, _, global_indices_subset = get_view_subgraph_and_features(full_data, view, all_patient_indices)
#         if global_indices_subset.numel() > 0 and vae_outputs[view].get('mu') is not None:
#             mus_for_view = vae_outputs[view]['mu']
#             fused_mus.index_add_(0, global_indices_subset, mus_for_view)
#             counts.index_add_(0, global_indices_subset, torch.ones_like(global_indices_subset, dtype=torch.float32).unsqueeze(1))

#     counts[counts == 0] = 1
#     fused_mus /= counts
#     print(f"Extracted {fused_mus.shape[0]} fused μ vectors.")

#     # Scale latents to [-1, 1] for DDPM stability
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     fused_mus_scaled = torch.tensor(scaler.fit_transform(fused_mus.cpu().numpy()), dtype=torch.float32)

#     # Create Dataset and DataLoader
#     class LatentDataset(torch.utils.data.Dataset):
#         def __init__(self, latents, labels):
#             self.latents = latents
#             self.labels = labels
#         def __len__(self): return len(self.latents)
#         def __getitem__(self, idx): return self.latents[idx], self.labels[idx] + 1

#     dataset = LatentDataset(fused_mus_scaled, all_labels.cpu())
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=ddpm_config['batch_size'], shuffle=True)

#     print("\n--- Step 2: Training the Conditional Latent DDPM ---")
#     denoising_net = DenoisingNetwork(
#         latent_dim=ddpm_config['latent_dim'],
#         num_classes=ddpm_config['num_classes'] + 1 # +1 for null class
#     ).to(device)

#     ddpm_model = ConditionalDDPM(denoising_net, **ddpm_config).to(device)
#     optimizer = torch.optim.AdamW(ddpm_model.parameters(), lr=ddpm_config['lr'])

#     for epoch in range(ddpm_config['epochs']):
#         total_loss = 0
#         for x0, y in dataloader:
#             optimizer.zero_grad()
#             x0, y = x0.to(device), y.to(device)
#             loss = ddpm_model.loss(x0, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
    
#         if (epoch + 1) % 50 == 0:
#             avg_loss = total_loss / len(dataloader)
#             print(f"Epoch {epoch+1:03d} | DDPM Loss: {avg_loss:.4f}")

#     print("\n--- DDPM Training Complete ---")
#     return ddpm_model, scaler


def train_single_conditional_ddpm(
    latents: torch.Tensor,
    labels: torch.Tensor,
    ddpm_config: dict,
    device: torch.device
) -> tuple[ConditionalDDPM, MinMaxScaler]:
    """
    Trains a single Conditional DDPM on a specific set of latent vectors and their corresponding labels.

    This function is designed to be called within a K-Fold loop, for instance, to train
    a DDPM only on the training data of a specific fold.

    Args:
        latents (torch.Tensor): A tensor of latent vectors (e.g., mu vectors from GVAE)
                                of shape [num_samples, latent_dim]. Should be on the CPU.
        labels (torch.Tensor): A tensor of corresponding labels of shape [num_samples].
                               Should be on the CPU.
        ddpm_config (dict): A dictionary containing configuration for the DDPM model and training.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to train the model on.

    Returns:
        tuple[ConditionalDDPM, MinMaxScaler]: A tuple containing:
            - The trained ConditionalDDPM model.
            - The scaler used to normalize the latent vectors, which is needed for inverse transform or
              normalizing new data.
    """
    if latents.shape[0] == 0:
        print("Warning: Received empty tensor for latents. Skipping DDPM training.")
        return None, None

    # 1. Scale latents to [-1, 1] for DDPM stability
    # The scaler is fitted only on this subset of data and returned for later use.
    scaler = MinMaxScaler(feature_range=(-1, 1))
    latents_scaled = torch.tensor(scaler.fit_transform(latents.cpu().numpy()), dtype=torch.float32)

    # 2. Create PyTorch Dataset and DataLoader
    class LatentDataset(torch.utils.data.Dataset):
        def __init__(self, latents_data, labels_data):
            self.latents = latents_data
            self.labels = labels_data

        def __len__(self):
            return len(self.latents)

        def __getitem__(self, idx):
            # We add +1 to labels because the DDPM's embedding layer reserves index 0
            # for the unconditional (null) class used in classifier-free guidance.
            return self.latents[idx], self.labels[idx] + 1

    dataset = LatentDataset(latents_scaled, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ddpm_config['batch_size'],
        shuffle=True,
        drop_last=True  # Important for stable training if the last batch is too small
    )

    # 3. Instantiate and Train the DDPM
    denoising_net = DenoiseUNet(
        latent_dim=ddpm_config['latent_dim'],
        num_classes=ddpm_config['num_classes'] + 1 
    ).to(device)

    ddpm_model = ConditionalDDPM(
        denoise_fn=denoising_net,
        latent_dim=ddpm_config['latent_dim'],
        timesteps=ddpm_config['timesteps']
    ).to(device)

    optimizer = torch.optim.AdamW(ddpm_model.parameters(), lr=ddpm_config['lr'])
    
    # Progress bar for training epochs
    pbar = tqdm(range(ddpm_config['epochs']), desc=f"Training DDPM on {len(latents)} samples")

    for epoch in pbar:
        total_loss = 0
        for x0, y in dataloader:
            optimizer.zero_grad()
            x0, y = x0.to(device), y.to(device)

            loss = ddpm_model.loss(x0, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm_model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    print(f"Finished DDPM training with final average loss: {avg_loss:.4f}")
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
        print(f"Warning: Not enough samples ({latents.shape[0]}) to train DDPM. Skipping.")
        return None, None

    scaler = MinMaxScaler(feature_range=(-1, 1))
    latents_scaled = torch.tensor(scaler.fit_transform(latents.cpu().numpy()), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(latents_scaled)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=ddpm_config['batch_size'], shuffle=True, drop_last=True
    )

    denoising_net = DenoiseUNet(
        latent_dim=ddpm_config['latent_dim'],
        num_classes=None # Báo cho UNet biết đây là model vô điều kiện
    ).to(device)

    ddpm_model = UnconditionalDDPM(
        denoise_fn=denoising_net,
        latent_dim=ddpm_config['latent_dim'],
        timesteps=ddpm_config['timesteps']
    ).to(device)

    optimizer = torch.optim.AdamW(ddpm_model.parameters(), lr=ddpm_config['lr'])
    pbar = tqdm(range(ddpm_config['epochs']), desc=f"Training Unconditional DDPM")

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