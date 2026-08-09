#!/usr/bin/env python
"""Phase 4: figures + reproduction table from verified artifacts (2026-08-09)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

ROOT = 'research/2026-08-06-project-review-audit/output'
FIG = f'{ROOT}/figures'
TAB = f'{ROOT}/tables'
import os
os.makedirs(FIG, exist_ok=True)

# ---------- 1. ROC / PR curves (pooled OOF) ----------
d = np.load(f'{TAB}/oof_arrays.npz')
y, hp, pp = d['y_true'], d['head_probs'], d['probe_probs']

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for probs, name, color in [(hp, 'Fusion head', '#1f77b4'), (pp, 'Linear probe (mu)', '#ff7f0e')]:
    fpr, tpr, _ = roc_curve(y, probs)
    ax[0].plot(fpr, tpr, color=color, lw=1.6, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
    prec, rec, _ = precision_recall_curve(y, probs)
    ax[1].plot(rec, prec, color=color, lw=1.6, label=f'{name} (AP={average_precision_score(y, probs):.3f})')
ax[0].plot([0, 1], [0, 1], 'k--', lw=0.8); ax[0].set(xlabel='FPR', ylabel='TPR', title='Pooled OOF ROC')
ax[1].set(xlabel='Recall', ylabel='Precision', title='Pooled OOF PR (positive = non-responder, label 1)')
ax[1].axhline(y.mean(), color='gray', ls=':', lw=0.8, label=f'prevalence {y.mean():.2f}')
for a in ax: a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIG}/oof_roc_pr.png', dpi=150); plt.close()
print('figure 1 saved: oof_roc_pr.png')

# ---------- 2. Comparison bar (GVAE vs downstream vs augmentation) ----------
aucs = {
    'GVAE direct\n(val-fold mean)': 0.6789,
    'Pooled OOF\nfusion head': 0.6065,
    'Pooled OOF\nlinear probe': 0.6519,
    'Downstream\nreal-only': 0.7104,
    'Downstream +\nmin0.25 both': 0.7115,
}
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(range(len(aucs)), list(aucs.values()), color=['#4c72b0', '#4c72b0', '#4c72b0', '#55a868', '#55a868'])
for i, v in enumerate(aucs.values()):
    ax.text(i, v + 0.004, f'{v:.3f}', ha='center', fontsize=9)
ax.set_xticks(range(len(aucs))); ax.set_xticklabels(list(aucs.keys()), fontsize=8)
ax.set_ylabel('ROC-AUC'); ax.set_ylim(0.5, 0.8); ax.set_title('ROC-AUC comparison (positive = non-responder)')
plt.tight_layout(); plt.savefig(f'{FIG}/auc_comparison.png', dpi=150); plt.close()
print('figure 2 saved: auc_comparison.png')

# ---------- 3. Reproduction table: old FINAL_REPORT SS6.2 vs new numbers ----------
old = {'ROC-AUC': 0.6894, 'PR-AUC': 0.8523, 'Balanced Accuracy': 0.7265, 'F1': 0.8101, 'Accuracy': 0.7409}
ci = json.load(open(f'{TAB}/oof_metrics_with_ci.json'))
new = {'ROC-AUC': ci['ci']['head']['roc_auc']['point'], 'PR-AUC': ci['ci']['head']['pr_auc']['point'],
       'Balanced Accuracy': ci['ci']['head']['balanced_accuracy']['point'], 'F1': ci['ci']['head']['f1']['point']}
rows = []
for k, o in old.items():
    if k in new:
        rows.append({'metric': k, 'old_FINAL_REPORT_SS6.2': round(o, 4), 'new_pooled_OOF_head': round(new[k], 4),
                     'delta': round((new[k] - o) / o * 100, 2)})
pd.DataFrame(rows).to_csv(f'{TAB}/reproduction_table.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('reproduction_table.csv saved')
