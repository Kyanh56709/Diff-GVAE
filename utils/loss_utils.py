import torch
import torch.nn.functional as F
from typing import Dict, Tuple

def calculate_contrastive_loss(
    sampled_zs_per_view_batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    temperature: float
) -> torch.Tensor:
    """
    Calculates cross-view contrastive loss for patients present in multiple views within the batch.
    Args:
        sampled_zs_per_view_batch: Dict where key is view_name, value is a tuple:
                                   (Tensor of z_embeddings for patients in batch having this view,
                                    Tensor of global_indices for these patients).
        temperature: Temperature for InfoNCE.
    Returns:
        Contrastive loss scalar.
    """
    if sampled_zs_per_view_batch:
        first_emb = list(sampled_zs_per_view_batch.values())[0][0]
        if first_emb is not None:
            device = first_emb.device

    total_contrastive_loss = torch.tensor(0.0, device=device)
    num_contrastive_pairs_total = 0

    # Create a list of (global_patient_idx, view_name, embedding_tensor)
    all_embeddings_flat = []
    for view_name, (embeddings, global_indices) in sampled_zs_per_view_batch.items():
        if embeddings is not None and global_indices is not None and embeddings.numel() > 0: # Check if embeddings exist
            for i in range(embeddings.shape[0]):
                all_embeddings_flat.append((global_indices[i].item(), view_name, embeddings[i]))

    if not all_embeddings_flat:
        return total_contrastive_loss

    # Group embeddings by global_patient_idx
    patient_to_view_embeddings = {}
    for global_idx, view_name, emb in all_embeddings_flat:
        if global_idx not in patient_to_view_embeddings:
            patient_to_view_embeddings[global_idx] = []
        patient_to_view_embeddings[global_idx].append(emb)

    # For each patient with embeddings from multiple views
    for global_idx, view_embs_list in patient_to_view_embeddings.items():
        if len(view_embs_list) < 2: # Need at least two views for this patient
            continue

        # Form positive pairs for this patient
        for i in range(len(view_embs_list)):
            for j in range(i + 1, len(view_embs_list)):
                z_i = view_embs_list[i].unsqueeze(0) # Anchor [1, d_embed]
                z_j = view_embs_list[j].unsqueeze(0) # Positive [1, d_embed]

                # Negative samples: all other embeddings in all_embeddings_flat NOT from this patient
                negatives = torch.stack([
                    other_emb for other_global_idx, _, other_emb in all_embeddings_flat
                    if other_global_idx != global_idx
                ])

                if negatives.numel() == 0: # Only one patient in batch, no negatives
                    continue

                # Cosine similarity
                sim_positive = F.cosine_similarity(z_i, z_j, dim=1) / temperature # Shape [1]
                sim_negatives = F.cosine_similarity(z_i.expand(negatives.shape[0], -1), negatives, dim=1) / temperature # Shape [num_negatives]

                # Concatenate positive score with negative scores for logits
                logits = torch.cat([sim_positive, sim_negatives]) # Shape [1 + num_negatives]
                target_index = torch.tensor([0], device=device, dtype=torch.long) 


                total_contrastive_loss += F.cross_entropy(logits.unsqueeze(0), target_index)
                num_contrastive_pairs_total += 1

    return total_contrastive_loss / num_contrastive_pairs_total if num_contrastive_pairs_total > 0 else torch.tensor(0.0, device=device)