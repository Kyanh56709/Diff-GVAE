# DDPM Validation Diagnostic Report

Run ID: `gvae_ddpm_concat_mu_20260615_023242`

No retraining was performed for this diagnostic.

## Executive Summary

The selected DDPM checkpoint has useful ranking signal (`AUC=0.7162`, `PR-AUC=0.9084`), but the fixed threshold `0.5` collapses the classifier to predicting every validation sample as positive.

For the final selected fold:

| Item | Value |
|---|---:|
| Fold | 3 |
| GVAE rank | 3 |
| Epoch | 137 |
| Validation positives | 37 |
| Validation negatives | 12 |
| Positive prevalence | 0.7551 |
| DDPM AUC | 0.7162 |
| DDPM PR-AUC | 0.9084 |
| DDPM F1 at threshold 0.5 | 0.8605 |
| Balanced accuracy at threshold 0.5 | 0.5000 |

The `F1=0.8605` is therefore not evidence of useful thresholded classification. It is mostly the result of the dominant positive class.

## Class Distribution

Overall dataset:

| Split | N | Positive | Negative | Positive rate |
|---|---:|---:|---:|---:|
| All patients | 247 | 185 | 62 | 0.7490 |

Selected checkpoints by fold:

| Fold | Train N | Train + | Train - | Train + rate | Val N | Val + | Val - | Val + rate | Test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 197 | 148 | 49 | 0.7513 | 50 | 37 | 13 | 0.7400 | Not used |
| 2 | 197 | 148 | 49 | 0.7513 | 50 | 37 | 13 | 0.7400 | Not used |
| 3 | 198 | 148 | 50 | 0.7475 | 49 | 37 | 12 | 0.7551 | Not used |
| 4 | 198 | 148 | 50 | 0.7475 | 49 | 37 | 12 | 0.7551 | Not used |
| 5 | 198 | 148 | 50 | 0.7475 | 49 | 37 | 12 | 0.7551 | Not used |

There is no held-out test split in this run. The pipeline used 5-fold cross-validation with train/validation folds.

## Confusion Matrix at Threshold 0.5

For the final selected DDPM checkpoint, the reported metrics imply all validation samples were predicted positive.

Code path: `_classification_metrics()` uses `preds = (probs > 0.5).astype(int)`.

Confusion matrix for fold 3:

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 0 | 12 |
| Positive | 0 | 37 |

Metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.7551 |
| Balanced accuracy | 0.5000 |
| Precision | 0.7551 |
| Recall | 1.0000 |
| F1 | 0.8605 |

This is exactly the behavior of an all-positive classifier on a validation fold with 75.51% positives.

## Predicted Probability Distribution

The exact probability distribution cannot be reconstructed from the saved artifacts.

Reason: during checkpoint saving, the pipeline explicitly removes `probs` and `labels` from `selected_candidate` and `all_candidate_results`. The selected checkpoint also does not contain trained DDPM state dictionaries, so the validation scores cannot be regenerated without retraining the DDPM models.

Available evidence:

- All selected folds have `recall=1.0`.
- All selected folds have `balanced_accuracy=0.5`.
- Precision equals validation positive prevalence in each fold.

Therefore, every selected fold is thresholding to all-positive predictions at `0.5`.

## Are All Probabilities Above 0.5?

Yes, for the selected checkpoints this is implied by the metrics and code:

- `preds = (probs > 0.5).astype(int)`
- Recall is `1.0`
- True negative count is `0`
- Balanced accuracy is `0.5`

Since all labels are predicted positive and the code uses `> 0.5`, the saved metrics imply all validation scores used for these selected checkpoints were strictly above `0.5`.

## Best Threshold and Optimized-Threshold Metrics

Not available for the current run.

The per-sample DDPM scores were not saved, and the trained DDPM models were not saved. Without those scores, we cannot compute:

- best F1 threshold
- best balanced-accuracy threshold
- sensitivity/specificity tradeoff
- ROC curve points
- PR curve points
- calibrated threshold

This should be fixed before using F1 or thresholded accuracy in the paper.

## ROC Curve and PR Curve

Only scalar AUC and PR-AUC were saved. Curve points were not saved.

Current available values for the final selected checkpoint:

| Curve metric | Value |
|---|---:|
| ROC-AUC | 0.7162 |
| PR-AUC / Average Precision | 0.9084 |

To reproduce curves, save per-sample labels and scores, or save the actual `fpr`, `tpr`, `precision`, `recall`, and threshold arrays during evaluation.

## PR-AUC Inflation from Positive-Class Dominance

The positive class is dominant:

| Scope | Positive prevalence |
|---|---:|
| Overall dataset | 0.7490 |
| Final selected validation fold | 0.7551 |

For average precision / PR-AUC, the random baseline is approximately the positive prevalence. Therefore, a PR-AUC of `0.9084` should be compared against a baseline near `0.7551`, not zero.

For the final selected fold:

| Quantity | Value |
|---|---:|
| PR-AUC | 0.9084 |
| Baseline PR-AUC | 0.7551 |
| Absolute lift | 0.1533 |
| Relative lift | 1.2030x |
| Normalized AP | 0.6259 |

Interpretation: PR-AUC is not meaningless, but it looks inflated because the positive class is common. Report prevalence-baseline PR-AUC or normalized AP alongside raw PR-AUC.

## Calibration

The DDPM score is not calibrated.

Current code computes:

```python
likelihood_resp = 1 / (loss_resp + 1e-9)
likelihood_non_resp = 1 / (loss_non_resp + 1e-9)
prob_is_responder = likelihood_resp / (likelihood_resp + likelihood_non_resp)
```

This value is a heuristic score derived from MSE evaluation loss. It is not a statistically calibrated posterior probability.

Evidence of poor threshold calibration:

- all selected checkpoints classify all validation samples as positive at threshold `0.5`
- balanced accuracy is exactly `0.5`
- recall is `1.0`
- precision equals positive prevalence

Brier score was computed on this uncalibrated score, so it should be treated as a rough diagnostic only, not as evidence of calibrated probability quality.

## Evaluation Code Assessment

The DDPM evaluation code is not using classifier-head outputs, which is correct. It uses DDPM reconstruction/noise-prediction losses on latent embeddings.

However, the conversion from DDPM losses to probabilities is scientifically weak:

1. `1 / loss` is not a proper likelihood.
2. Normalizing inverse losses gives a number in `[0, 1]`, but not a calibrated probability.
3. Threshold `0.5` is equivalent to choosing the class with lower DDPM evaluation loss.
4. With imbalanced classes, this threshold can collapse to the majority class.
5. The pipeline currently computes F1/accuracy from a fixed threshold instead of tuning threshold on validation scores.

## Recommended Fixes

High priority:

1. Save per-sample validation outputs for every DDPM candidate:
   - patient id
   - label
   - score
   - `loss_resp`
   - `loss_non_resp`
   - fold
   - checkpoint path
   - latent artifact path

2. Save trained DDPM state dictionaries:
   - responder DDPM
   - non-responder DDPM
   - scaler
   - eval timesteps and noise seed

3. Replace fixed threshold `0.5` reporting with validation-optimized thresholds:
   - max F1 threshold
   - max balanced-accuracy threshold
   - sensitivity/specificity operating point

4. Treat DDPM output as a score, not a calibrated probability:
   - prefer `score = loss_non_resp - loss_resp`
   - or `score = -loss_resp + loss_non_resp`
   - compute AUC/PR-AUC from the score
   - only call it probability after calibration

5. Report prevalence-aware PR-AUC:
   - raw PR-AUC
   - positive prevalence baseline
   - normalized AP
   - optionally negative-class PR-AUC if non-responder detection matters clinically

Medium priority:

6. Add calibration:
   - Platt scaling / logistic calibration
   - isotonic regression
   - temperature scaling of loss-difference logits
   - evaluate ECE, Brier score, reliability curve

7. Save ROC and PR curve arrays:
   - `fpr`, `tpr`, ROC thresholds
   - precision, recall, PR thresholds

8. Add thresholded metrics:
   - specificity
   - sensitivity
   - balanced accuracy
   - MCC
   - confusion matrix

9. Use nested validation if threshold/calibration/model selection are all tuned:
   - inner validation for threshold/calibration
   - outer validation/test for final reporting

## Bottom Line

The current DDPM has ranking signal, especially in PR-AUC, but the thresholded classifier is currently degenerate at threshold `0.5`.

For the paper, use AUC/PR-AUC as ranking metrics only for this run. Do not claim the current F1/accuracy as meaningful until per-sample scores are saved and a validation-tuned threshold or calibrated probability is used.
