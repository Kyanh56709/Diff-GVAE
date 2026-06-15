import torch
import copy
import random
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from torch_geometric.data import HeteroData
from training.train_gvae import kfold_train_gvae

def update_configs_with_params(
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    params: Dict[str, Any]
) -> tuple:
    """
    Safely updates nested dictionaries in model_config and train_config with the current trial parameters.
    """
    mc = copy.deepcopy(model_config)
    tc = copy.deepcopy(train_config)

    # 1. Update basic training parameters
    if 'lr' in params:
        tc['lr'] = params['lr']
    if 'wd' in params:
        tc['wd'] = params['wd']
    if 'batch_size' in params:
        tc['batch_size'] = params['batch_size']
    if 'pretrain_epochs' in params:
        tc['pretrain_epochs'] = params['pretrain_epochs']
    if 'pretrain_val_split' in params:
        tc['pretrain_val_split'] = params['pretrain_val_split']

    # 2. Update latent embedding size (affects multiple keys in config)
    if 'd_embed' in params:
        d_embed = params['d_embed']
        mc['d_embed'] = d_embed
        
        # Projection Head output dim must match d_embed
        if 'projection_head_config' in mc:
            mc['projection_head_config']['output_dim'] = d_embed
            mc['projection_head_config']['hidden_dim'] = max(32, d_embed)
            
        # Fusion Head dim (fused_dim) must match d_embed
        if 'fusion_config' in mc:
            mc['fusion_config']['fused_dim'] = d_embed
            
        # Classifier input/hidden dimensions should adapt to d_embed
        if 'classifier_config' in mc:
            mc['classifier_config']['classifier_hidden_dim'] = max(32, d_embed)

    # 3. Update VAE hidden channels (forces all views to be equal)
    if 'hidden_channels_vae' in params:
        hidden = params['hidden_channels_vae']
        for view in mc['view_configs']:
            mc['view_configs'][view]['hidden_channels_vae'] = hidden
            
    # Update attention heads in VAE encoders (forces all views to be equal)
    if 'heads' in params:
        heads = params['heads']
        for view in mc['view_configs']:
            mc['view_configs'][view]['heads'] = heads

    # Update radiology aggregator attention hidden dimension
    if 'attention_hidden_dim' in params:
        if 'radiology_aggregator_config' in mc:
            mc['radiology_aggregator_config']['attention_hidden_dim'] = params['attention_hidden_dim']

    # 4. Update loss weights
    if 'cross_cl' in params:
        cl_w = params['cross_cl']
        tc['loss_weights']['cross_cl'] = cl_w
        if 'cross_cl' in tc.get('annealing', {}):
            tc['annealing']['cross_cl']['start_weight'] = cl_w
            tc['annealing']['cross_cl']['end_weight'] = cl_w
    elif 'cross_cl_weight' in params:
        cl_w = params['cross_cl_weight']
        tc['loss_weights']['cross_cl'] = cl_w
        if 'cross_cl' in tc.get('annealing', {}):
            tc['annealing']['cross_cl']['start_weight'] = cl_w
            tc['annealing']['cross_cl']['end_weight'] = cl_w

    if 'kl_weight' in params:
        kl_w = params['kl_weight']
        tc['loss_weights']['kl'] = kl_w
        if 'kl' in tc.get('annealing', {}):
            tc['annealing']['kl']['start_weight'] = 0.0
            tc['annealing']['kl']['end_weight'] = kl_w

    if 'rec_struct_weight' in params:
        tc['loss_weights']['rec_struct'] = params['rec_struct_weight']

    return mc, tc

def run_gvae_sweep(
    data: HeteroData,
    base_model_config: Dict[str, Any],
    base_train_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    n_trials: Optional[int] = None,
    n_splits: int = 3,
    epochs: int = 200,
    output_csv: str = "gvae_sweep_results.csv",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Run a Grid Search or Random Search hyperparameter sweep for GVAE.
    
    Args:
        data: The multi-view PyTorch Geometric HeteroData dataset.
        base_model_config: Base configuration dictionary for GVAE architecture.
        base_train_config: Base configuration dictionary for training.
        param_grid: A dictionary mapping hyperparameter names to a list of values to search.
        n_trials: If set, performs Random Search with this many trials. Otherwise performs full Grid Search.
        n_splits: Number of folds for Cross Validation (default 3 for sweep speed).
        epochs: Max epochs per training run (default 200 for sweep speed).
        output_csv: Path to save the intermediate and final trial results to CSV.
        random_seed: Seed for reproducibility of random choices.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 1. Generate all possible combinations
    import itertools
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if n_trials is not None and n_trials < len(all_combinations):
        print(f"INFO: Randomly sampling {n_trials} trials out of {len(all_combinations)} possible combinations.")
        trials = random.sample(all_combinations, n_trials)
    else:
        print(f"INFO: Running full Grid Search with all {len(all_combinations)} combinations.")
        trials = all_combinations

    # Load existing results if any to resume
    results = []
    start_trial = 0
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv)
            print(f"INFO: Found existing CSV with {len(df_existing)} runs. Resuming...")
            results = df_existing.to_dict('records')
            start_trial = len(results)
        except Exception as e:
            print(f"WARNING: Could not load existing CSV: {e}. Starting fresh.")

    # Override training configuration defaults for sweep efficiency
    sweep_train_config = copy.deepcopy(base_train_config)
    sweep_train_config['n_splits'] = n_splits
    sweep_train_config['epochs'] = epochs
    sweep_train_config['save_best_fold_model'] = False  # Avoid cluttering workspace with models during sweep

    for i in range(start_trial, len(trials)):
        trial_params = trials[i]
        print(f"\n=========================================")
        print(f" TRIAL {i + 1}/{len(trials)}: {trial_params}")
        print(f"=========================================")

        # Prepare modified configs for this trial
        mc, tc = update_configs_with_params(base_model_config, sweep_train_config, trial_params)

        try:
            # Run cross-validation GVAE training
            summary, df_results, _ = kfold_train_gvae(data, mc, tc)
            
            # Combine params and results
            trial_record = {**trial_params}
            trial_record['mean_auc'] = summary.get('mean_auc', 0.0)
            trial_record['std_auc'] = summary.get('std_auc', 0.0)
            trial_record['mean_f1'] = summary.get('mean_f1', 0.0)
            trial_record['std_f1'] = summary.get('std_f1', 0.0)
            trial_record['mean_accuracy'] = summary.get('mean_accuracy', 0.0)
            trial_record['mean_loss_at_best_auc'] = summary.get('mean_loss_at_best_auc', 0.0)
            trial_record['std_loss_at_best_auc'] = summary.get('std_loss_at_best_auc', 0.0)
            
            results.append(trial_record)
            
            # Save progress immediately
            df_res = pd.DataFrame(results)
            df_res.to_csv(output_csv, index=False)
            
            print(f"Trial {i + 1} Done -> Mean Val-AUC: {trial_record['mean_auc']:.4f} (std={trial_record['std_auc']:.4f}) | Loss at Best AUC: {trial_record['mean_loss_at_best_auc']:.4f} (std={trial_record['std_loss_at_best_auc']:.4f})")
        except Exception as e:
            print(f"ERROR: Trial {i + 1} failed: {e}")
            # Log the failure
            trial_record = {**trial_params}
            trial_record['mean_auc'] = float('nan')
            trial_record['error'] = str(e)
            results.append(trial_record)
            
            df_res = pd.DataFrame(results)
            df_res.to_csv(output_csv, index=False)

    # Compile and return final sorted results
    df_final = pd.DataFrame(results)
    df_final = df_final.sort_values(by='mean_auc', ascending=False)
    
    print("\n================ SWEEP COMPLETE ==================")
    print("Top 5 Hyperparameter Configurations:")
    print(df_final.head(5).to_string(index=False))
    
    best_config = df_final.iloc[0].to_dict()
    print(f"\nBest Config: {best_config}")
    return df_final
