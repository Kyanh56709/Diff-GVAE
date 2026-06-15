import torch
import sys
from training.sweep_gvae import run_gvae_sweep

def main():
    # 1. Device selection (supporting CUDA, Apple Silicon MPS, or CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device for sweep: {device}")

    # 2. Load the dataset
    data_path = 'data_ln_pc_ihc_g.pt'
    print(f"Loading data from {data_path}...")
    try:
        data = torch.load(data_path, weights_only=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    DIM_CLINICAL = data['patient'].x_clinical.shape[1]
    DIM_PATHOLOGY = data['patient'].x_pathology.shape[1]
    DIM_RADIOLOGY = data['lesion'].x.shape[1]
    print(f"Clinical Dim: {DIM_CLINICAL}, Path Dim: {DIM_PATHOLOGY}, Rad Dim: {DIM_RADIOLOGY}")

    # 3. Base Configurations (matching training_local.ipynb)
    train_config = {
        'data_path': data_path,
        'device': device,
        'n_splits': 3,
        'epochs': 200,
        'pretrain_epochs': 150,
        'patience': 30,
        'patience_early_stopping': 30,
        'lr': 0.001,
        'wd': 1e-4,
        'batch_size': 32,
        'loss_weights': {
            'class': 1.0,
            'cross_cl': 0.2,
            'rec_attr': {'clinical': 1.0, 'pathology': 1.0, 'radiology': 1.0},
            'rec_struct': 0.1,
            'kl': 0.00001,
        },
        'annealing': {
            'kl': {'start_weight': 0.0, 'end_weight': 0.00001, 'start_epoch': 20, 'end_epoch': 300},
            'cross_cl': {'start_weight': 0.1, 'end_weight': 0.1, 'start_epoch': 20, 'end_epoch': 100},
        },
        'pca_config': {
            'clinical': 16,
            'pathology': 8,
        },
        'lesion_pca_config': {
            'n_components': 15,
        },
        'cross_cl_temp': 0.1,
        'grad_clip_norm': 1.0,
        'print_every_k_epochs': 50, # Print less frequently during sweep
    }

    model_config = {
        'view_configs': {
            'clinical': {'in_channels': DIM_CLINICAL, 'hidden_channels_vae': 64, 'heads': 8, 'dropout': 0.5, 'num_gnn_layers': 2, 'edge_dim': 1},
            'pathology': {'in_channels': DIM_PATHOLOGY, 'hidden_channels_vae': 64, 'heads': 8, 'dropout': 0.5, 'num_gnn_layers': 2, 'edge_dim': 1},
            'radiology': {'in_channels': 32, 'hidden_channels_vae': 64, 'heads': 8, 'dropout': 0.5, 'num_gnn_layers': 2, 'edge_dim': 1},
        },
        'radiology_aggregator_config': {
            'lesion_feature_dim': DIM_RADIOLOGY,
            'aggregated_output_dim': 32,
            'attention_hidden_dim': 64,
            'dropout': 0.3
        },
        'fusion_config': {
            'fused_dim': 32,
            'num_fusion_heads': 8,
            'fusion_ffn_multiplier': 5,
        },
        'classifier_config': {
             'classifier_hidden_dim': 32
        },
        'projection_head_config': {
            'hidden_dim': 32,
            'output_dim': 32,
            'dropout': 0.5
        },
        'd_embed': 32,
        'missing_strategy': 'learnable'
    }

    # 4. Define Parameter Sweep Grid
    # Edit these lists to include whichever parameters and values you want to sweep.
    param_grid = {
        'lr': [1e-3, 5e-4],
        'cross_cl': [0.1, 0.2, 0.5],
        'd_embed': [16, 32, 64],
        'hidden_channels_vae': [32, 64],
        'attention_hidden_dim': [32, 64],
        'heads': [4, 8]
    }

    # 5. Run the Sweep
    print("Starting hyperparameter sweep...")
    run_gvae_sweep(
        data=data,
        base_model_config=model_config,
        base_train_config=train_config,
        param_grid=param_grid,
        n_trials=None, # Set to an integer (e.g. 6) for Random Search; None for full Grid Search
        n_splits=3,    # Fast cross-validation fold size for search
        epochs=200,    # Lower epochs per fold for faster evaluation
        output_csv="gvae_sweep_results.csv",
        random_seed=42
    )

if __name__ == "__main__":
    main()
