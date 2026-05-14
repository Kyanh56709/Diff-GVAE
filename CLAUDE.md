# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Diff-GVAE is a PyTorch-based multi-view generative classification pipeline. Patients have up to 3 views: **clinical** (mixed continuous/binary features), **pathology** (GLCM texture features), and **radiology** (lesion-level attention-aggregated features). The pipeline trains a multi-view Graph Variational Autoencoder (GVAE) whose latent representations feed a generative classifier built from two class-conditional DDPMs (one per response class).

## Data Format

- Input: `data_ln_pc_ihc_g.pt` — a `torch_geometric.data.HeteroData` object
- Patient node: `x_clinical`, `x_pathology`, `binary_label`/`y`, `pathology_mask`, `radiology_mask`
- Lesion node: `x` (radiology features)
- Edges: `('patient', 'has_lesion', 'lesion')`, `('patient', 'similar_to_clinical', 'patient')`, etc.
- Clinical feature indices 0–4 are continuous; 5–21 are binary (used for pos_weight in reconstruction loss)

## Running Training

The primary entry point is `training.ipynb`. The notebook sets device, loads data, defines configs, then calls:

```python
from training.train_gvae import kfold_train_gvae
summary, df_results, roc_data = kfold_train_gvae(data, model_config, train_config)
```

For the generative classifier pipeline:

```python
from training.train_pipeline import kfold_gvae_ddpm_generative_classifier
results = kfold_gvae_ddpm_generative_classifier(full_data, model_config, train_config, ddpm_config, pca_config)
```

## Key Functions

| File | Function | Purpose |
|------|----------|---------|
| `training/train_gvae.py` | `kfold_train_gvae()` | Full k-fold CV with mixed reconstruction loss, contrastive learning, radiology pretraining |
| `training/train_gvae.py` | `train_gvae_single_fold()` | Single fold GVAE training for the pipeline |
| `training/train_gvae.py` | `pretrain_radiology_aggregator()` | Pre-trains `RadiologyLesionAttentionAggregator` as a MIL classifier on lesion features |
| `training/train_ddpm.py` | `train_single_unconditional_ddpm()` | Trains one DDPM per class on fused mu vectors |
| `training/train_pipeline.py` | `kfold_gvae_ddpm_generative_classifier()` | End-to-end: train GVAE → extract mus → train fusion layer → train DDPMs → evaluate |
| `models/gvae_model.py` | `GVAE.forward()` | Per-view VAE encoding → fusion + classification; handles missing views via `missing_strategy` |
| `models/gvae_model.py` | `get_all_view_mus_from_gvae()` | Extract mu vectors per view for downstream DDPM training |
| `utils/data_utils.py` | `get_view_subgraph_and_features()` | Extracts patient-level features and local subgraph for a given view |
| `utils/loss_utils.py` | `calculate_contrastive_loss()` | Cross-view contrastive loss (InfoNCE) between views for patients with ≥2 views |

## Architecture Notes

- **GVAE**: Each view has a `ViewEncoder` (GATv2Conv → mu/logvar), `AttributeDecoder` (residual MLP → feature reconstruction), and `StructureDecoder` (inner product → adjacency logits). A `RadiologyLesionAttentionAggregator` processes lesion graphs via attention-based MIL before feeding the radiology VAE.
- **Fusion**: `FusionAndClassifierHead` uses a learnable [CLS] token + MHA transformer block to fuse per-view z_sampled vectors, then a MLP classifier.
- **DDPM**: Two `UnconditionalDDPM` wrappers (one per class) trained on fused mu vectors. At inference, likelihood comparison across 50 sampled timesteps classifies validation patients.
- **DCA** (`DenoiseDCA`): Cross-modal attention block used in the DDPM denoising architecture. Requires hidden_dim divisible by both `num_modalities` and `n_heads` (uses `_round_up_to_lcm` for safe padding).
- **Missing views**: Controlled by `missing_strategy` ('learnable' or 'zero'); learnable `missing_embeddings_params` per view replace absent modalities.

## Training Configuration

Key `train_config` keys:
- `n_splits`, `epochs`, `patience_early_stopping`, `lr`, `wd`, `grad_clip_norm`
- `loss_weights`: `class`, `kl`, `cross_cl`, `rec_attr` (per-view), `rec_struct` (per-view)
- `annealing`: `kl` and `cross_cl` weight scheduling via `linear_anneal()`
- `cross_cl_temp`: temperature for contrastive loss
- `print_every_k_epochs`: logging frequency

## Dependencies

```
torch, torch_geometric, torch_scatter, scikit-learn, numpy, pandas, tqdm, matplotlib, wandb, einops
```

torch_scatter and torch_geometric must match your torch version. Install via:
```
pip install torch-geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
```