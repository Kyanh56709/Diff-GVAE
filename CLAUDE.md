# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Diff-GVAE is a PyTorch-based multi-view GVAE pipeline. Patients have up to 3 views: **clinical** (mixed continuous/binary features), **pathology** (GLCM texture features), and **radiology** (lesion-level attention-aggregated features). The GVAE classifier predicts NSCLC immunotherapy response. DDPM is **not** a classifier; it is used only for train-fold latent-space augmentation.

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

For conditional latent augmentation:

```python
from training.latent_ddpm_augmentation import run_conditional_latent_augmentation_pipeline
summary = run_conditional_latent_augmentation_pipeline(
    checkpoint_paths,
    full_data,
    output_dir="outputs/conditional_latent_ddpm",
    latent_key="concat_mu",
    augmentation_modes=("minority_only", "responder_only", "both_classes"),
    ratios=(0.25, 0.5, 1.0, 2.0),
    filter_config={"enabled": True, "quantile": 0.95},
)
```

## Key Functions

| File | Function | Purpose |
|------|----------|---------|
| `training/train_gvae.py` | `kfold_train_gvae()` | Full k-fold CV with mixed reconstruction loss, contrastive learning, radiology pretraining |
| `training/train_gvae.py` | `train_gvae_single_fold()` | Single fold GVAE training for the pipeline |
| `training/train_gvae.py` | `pretrain_radiology_aggregator()` | Pre-trains `RadiologyLesionAttentionAggregator` as a MIL classifier on lesion features |
| `training/train_ddpm.py` | `train_single_unconditional_ddpm()` | Legacy helper; do not use for DDPM-as-classifier results |
| `training/latent_ddpm_augmentation.py` | `run_conditional_latent_augmentation_pipeline()` | End-to-end DDPM augmentation: extract `concat_mu` → train conditional DDPM on train fold → generate synthetic latents → evaluate downstream classifier |
| `training/train_pipeline.py` | `kfold_gvae_ddpm_generative_classifier()` | Deprecated and disabled by default because it treats DDPM loss as a classifier score |
| `models/gvae_model.py` | `GVAE.forward()` | Per-view VAE encoding → fusion + classification; handles missing views via `missing_strategy` |
| `models/gvae_model.py` | `get_all_view_mus_from_gvae()` | Extract mu vectors per view for downstream DDPM training |
| `utils/data_utils.py` | `get_view_subgraph_and_features()` | Extracts patient-level features and local subgraph for a given view |
| `utils/loss_utils.py` | `calculate_contrastive_loss()` | Cross-view contrastive loss (InfoNCE) between views for patients with ≥2 views |

## Architecture Notes

- **GVAE**: Each view has a `ViewEncoder` (GATv2Conv → mu/logvar), `AttributeDecoder` (residual MLP → feature reconstruction), and `StructureDecoder` (inner product → adjacency logits). A `RadiologyLesionAttentionAggregator` processes lesion graphs via attention-based MIL before feeding the radiology VAE.
- **Fusion**: `FusionAndClassifierHead` uses a learnable [CLS] token + MHA transformer block to fuse per-view z_sampled vectors, then a MLP classifier.
- **DDPM**: Conditional latent DDPM learns `p(concat_mu | class)` using `concat_mu = [clinical_mu, pathology_mu, radiology_mu]`. It never consumes GVAE logits/probabilities and should be evaluated only through downstream classifier performance after train-fold augmentation.
- **DCA** (`DenoiseDCA`): Cross-modal attention block used in the DDPM denoising architecture. Requires hidden_dim divisible by both `num_modalities` and `n_heads` (uses `_round_up_to_lcm` for safe padding).
- **Missing views**: Controlled by `missing_strategy` ('learnable' or 'zero'); learnable `missing_embeddings_params` per view replace absent modalities.

## Training Configuration

Key `train_config` keys:
- `n_splits`, `epochs`, `patience_early_stopping`, `lr`, `wd`, `grad_clip_norm`
- `checkpoint_metric`: defaults to `latent_quality`; ranks checkpoints using train-only linear-probe AUC on `concat_mu`, silhouette score, Fisher ratio, and train/validation latent variance stability
- `loss_weights`: `class`, `kl`, `cross_cl`, `rec_attr` (per-view), `rec_struct` (per-view)
- `annealing`: `kl` and `cross_cl` weight scheduling via `linear_anneal()`
- `cross_cl_temp`: temperature for contrastive loss
- `print_every_k_epochs`: logging frequency

## New Feature Flags (v2)

These flags were added in the gvae-lesion-agg-fixes branch. All default to current behavior.

### model_config flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `logvar_clamp` | `Tuple[float,float] \| None` | `None` | Clamp encoder logvar to `(min, max)`. Recommended: `(-6.0, 2.0)`. |
| `radiology_zero_lesion_passthrough` | `bool` | `False` | When True, radiology patients with zero lesions passthrough encoder instead of getting missing embedding. |

### train_config flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `pretrain_use_pos_weight` | `bool` | `False` | Class-weighted BCE loss in radiology pretraining. |
| `pretrain_val_split` | `float` | `0.0` | Fraction held out (STRATIFIED by label) for pretrain validation + early stopping. |
| `pretrain_epochs` | `int` | `400` | Number of radiology-aggregator pretraining epochs. Lower (e.g. 100) acts as a warm-start and curbs overfitting; `0` = no pretraining. |
| `pretrain_patience` | `int` | `30` | Early-stopping patience (epochs without val-AUC improvement) when `pretrain_val_split > 0`. |
| `pretrain_seed` | `int` | `42` | Seed for the stratified pretrain split. |
| `vectorized_contrastive` | `bool` | `False` | Use vectorized InfoNCE implementation for contrastive loss. |

**Recipe sweep:** `training.train_gvae.sweep_pretrain_recipes(full_data, model_config, train_config, recipes=None)` runs `kfold_train_gvae` once per recipe and ranks them by **downstream GVAE val-AUC** (the metric that matters — not the pretrainer's own AUC). Default recipes compare: no pretraining, 100ep, 400ep, and val-split+early-stop. Shrink `n_splits`/`epochs` in `train_config` for a fast sweep.

## Dependencies

```
torch, torch_geometric, torch_scatter, scikit-learn, numpy, pandas, tqdm, matplotlib, wandb, einops
```

torch_scatter and torch_geometric must match your torch version. Install via:
```
pip install torch-geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
```
