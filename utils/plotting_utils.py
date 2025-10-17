import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
import gc
import os
import torch
from typing import List, Dict, Optional

def plot_kfold_roc_curves(roc_data_per_fold: List[Dict], title: str = "ROC 10-CV", save_path: Optional[str] = None):
    """
    Generates a K-fold ROC plot with individual, mean, and merged curves.

    Args:
        roc_data_per_fold (List[Dict]): A list where each element is a dictionary
            from a fold containing {'fpr', 'tpr', 'auc', 'y_true', 'y_pred'}.
        title (str): The title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    all_y_true = []
    all_y_pred = []

    # Plot individual fold ROC curves
    for i, data in enumerate(roc_data_per_fold):
        ax.plot(data['fpr'], data['tpr'], lw=1.5, alpha=0.6,
                label=f"Fold {i+1} (AUC = {data['auc']:.2f}), N={len(data['y_true'])} patients")
        
        # For calculating the mean ROC
        interp_tpr = interpolate.interp1d(data['fpr'], data['tpr'], kind='linear', bounds_error=False, fill_value=(0.0, 1.0))(mean_fpr)
        tprs.append(interp_tpr)
        aucs.append(data['auc'])
        

    # Plot chance line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)

    # Calculate and plot MEAN ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='blue',
            label=f'Mean ROC (AUC = {mean_auc:.2f} $\\pm$ {std_auc:.2f}), N={len(roc_data_per_fold)} folds',
            lw=2.5, alpha=0.9)

    # Plot standard deviation area
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,
                    label=r'$\pm$ 1 std. dev.')

    # Final plot settings
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel='False Positive Rate',
           ylabel='True Positive Rate',
           title=title)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.5)

    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Plot saved to: {save_path}")
        
    plt.show()
    plt.close(fig)