# 1. Executive Summary

This project is a multimodal NSCLC immunotherapy response prediction pipeline built around a PyTorch Geometric `HeteroData` graph and a multi-view GVAE. The current intended prediction model is the GVAE: it encodes clinical, pathology, and radiology views, fuses their latent representations, and predicts binary response with a classifier head.

The current safe augmentation path is `training/latent_ddpm_augmentation.py`: it extracts GVAE latent vectors, trains a conditional DDPM on train-fold `concat_mu`, generates synthetic latent vectors by class, and evaluates those synthetic vectors only through a downstream classifier. In this path, DDPM is generator-only and is not used as a response predictor.

The repo still contains legacy DDPM-as-classifier code in `training/train_pipeline.py` and old runners. That path is disabled unless `allow_deprecated_ddpm_classifier=True`, but it is still a design risk because it trains class-specific unconditional DDPMs and converts denoising losses into class probabilities.

The biggest project risks are data/provenance ambiguity, two serialized graph files with opposite label polarity, stale/legacy runners, weak documentation/config centralization, and metric interpretation. Validation/test leakage is mostly controlled in the current conditional augmentation path, but it remains possible in legacy utilities or if the wrong entry point is used.

# 2. Intended Pipeline vs Current Pipeline

| Area | Intended design | Current code behavior | Status |
|---|---|---|---|
| Prediction model | GVAE predicts NSCLC response from multimodal `HeteroData`. | `training/train_gvae.py:kfold_train_gvae` trains GVAE with BCE classifier loss plus reconstruction/KL/contrastive losses. | Matches. |
| DDPM role | DDPM is not a classifier. | Correct path `training/latent_ddpm_augmentation.py` sets `ddpm_is_classifier=False`; legacy classifier path remains disabled by default in `training/train_pipeline.py:603` and `training/train_pipeline.py:952`. | Mostly matches, legacy risk remains. |
| DDPM input | Only `concat_mu = [clinical_mu, pathology_mu, radiology_mu]`. | Correct runner defaults to `--latent-key concat_mu`; however `utils/latent_extraction.py:326` still allows `fused_cls_mu`, `concat_z`, and `stacked_z`. | Partial mismatch. |
| Checkpoint selection | Rank GVAE checkpoints by latent quality, not only validation AUC. | `training/train_gvae.py:117` computes train-only linear probe AUC, silhouette, Fisher ratio, and variance stability; `training/train_gvae.py:219` supports latent-quality sorting. `run_gvae.py:75` is stale and still uses `auc_pr_balanced_accuracy`. | Partially fixed. |
| Scaler/PCA fitting | Fit only on train fold. | Current conditional path fits DDPM scaler on train latent only at `training/latent_ddpm_augmentation.py:321`; downstream scaler at `training/latent_ddpm_augmentation.py:278`; PCA at `training/latent_ddpm_augmentation.py:858`. | Matches in current path. |
| Synthetic data | Generate train-fold latent samples by class, then evaluate downstream. | `generate_synthetic_latents` implements `minority_only`, `responder_only`, `both_classes`, and ratios. | Matches. |
| Metrics | GVAE metrics are direct prediction; DDPM metrics are downstream classifier after augmentation. | Current summary stores `ddpm_is_classifier=False` and ablation labels; old reports/runners may still confuse this. | Mostly matches. |

# 3. Repository Structure

- `models/gvae_model.py`: main GVAE wrapper, per-view encoding, missing-view handling, radiology aggregation, fusion/classifier output, latent extraction helpers.
- `models/gvae_components.py`: GATv2 view encoder, structure/attribute decoders, fusion transformer-style classifier head, lesion attention aggregator, radiology MIL pretraining wrapper.
- `models/ddpm.py`: conditional and unconditional DDPM implementations. Conditional DDPM is appropriate for latent generation; unconditional DDPM is used by the deprecated classifier path.
- `models/unet.py`: conditional denoising network used by the DDPM.
- `training/train_gvae.py`: main GVAE cross-validation training loop, losses, checkpoint ranking, early stopping, diagnostics.
- `training/latent_ddpm_augmentation.py`: current correct latent-space conditional DDPM augmentation and downstream evaluation pipeline.
- `training/train_pipeline.py`: deprecated GVAE+DDPM generative-classifier path, disabled unless explicitly opted in.
- `training/train_ddpm.py`: lower-level DDPM training helpers; conditional helper is generator-style, unconditional helper supports legacy classifier use.
- `utils/data_utils.py`: view-specific subgraph extraction and optional train-fold PCA preprocessing.
- `utils/latent_extraction.py`: checkpoint loading and GVAE latent artifact extraction.
- `utils/classification_eval.py`: binary metrics, curves, threshold diagnostics, saved classification artifacts.
- `outputs/gvae/*.py`: runnable scripts for GVAE training and downstream DDPM augmentation. Some legacy runners remain.
- `data_ln_pc_ihc_g.pt`: main serialized `HeteroData` used by current runners.
- `data/data_247.pt`: second serialized graph with different feature dimensions and opposite label polarity.
- `README.md` and `configs/config.py`: currently too sparse for reproducible experiment documentation.

# 4. Data Flow

Current safe flow:

1. Load serialized graph from `data_ln_pc_ihc_g.pt` in `outputs/gvae/train_gvae_runner.py:69`.
2. Build model config from tensor dimensions in the loaded graph.
3. Split patient nodes with stratified K-fold in `training/train_gvae.py:523`.
4. Train GVAE on train indices and validate on held-out fold.
5. Save top GVAE checkpoints and fold metadata under `outputs/gvae/checkpoints/<run_id>/`.
6. Load each selected checkpoint in `training/latent_ddpm_augmentation.py:931`.
7. Extract train/validation latent artifacts with `utils/latent_extraction.py:174`.
8. Select `concat_mu` from train fold for DDPM training and validation/test fold for downstream evaluation.
9. Fit DDPM scaler only on train latent vectors.
10. Train conditional DDPM to predict noise in latent space conditioned on class label.
11. Generate synthetic latent vectors for configured class modes and ratios.
12. Optionally filter synthetic vectors by same-class train-nearest-neighbor distance.
13. Train a downstream classifier on real train latents or real+synthetic train latents.
14. Evaluate downstream classifier on held-out validation/test latent vectors.

No source script was found that rebuilds `data_ln_pc_ihc_g.pt` from raw clinical/pathology/radiology data. Graph construction is therefore only auditable from the serialized object, not from raw preprocessing code.

# 5. GVAE Pipeline

The GVAE class starts at `models/gvae_model.py:14`. It supports three views: clinical, pathology, and radiology. For each view, `utils/data_utils.py:get_view_subgraph_and_features` selects the relevant patient subset and view-specific edges. Clinical uses all patients in the batch, while pathology and radiology respect `pathology_mask` and `radiology_mask`.

Per-view encoding uses `ViewEncoder` in `models/gvae_components.py:9`, with GATv2 layers, layer normalization, dropout, and separate `mu` and `logvar` projections. The radiology branch first aggregates lesion-level features to patient-level vectors with `RadiologyLesionAttentionAggregator` in `models/gvae_components.py:466`.

The forward pass in `models/gvae_model.py:100` produces:

- `logits`: direct GVAE response logits.
- `vae_outputs_by_view`: per-view latent/reconstruction tensors used for losses.
- `projected_mus`: projected view latents used by contrastive loss.
- `fusion_attn_weights` and `cls_output`: fusion diagnostics.

The classifier head is `FusionAndClassifierHead` in `models/gvae_components.py:386`. It stacks per-view embeddings, prepends a CLS token, applies multi-head attention and feed-forward blocks, then maps the fused CLS representation to one binary logit. Loss uses `BCEWithLogitsLoss` with train-fold `pos_weight` in `training/train_gvae.py:640`.

GVAE losses in `training/train_gvae.py:623` include:

- Classification BCE on response logits.
- Cross-view contrastive loss from `utils/loss_utils.py:5`.
- Attribute reconstruction loss: mixed MSE/BCE for clinical features and MSE for pathology/radiology.
- Structure reconstruction loss via dense adjacency BCE.
- KL divergence.

Checkpoint ranking now supports latent quality. `_compute_latent_quality_metrics` at `training/train_gvae.py:117` fits a `StandardScaler` and logistic linear probe only on train-fold `concat_mu`, transforms validation `concat_mu`, and computes linear probe AUC, silhouette score, Fisher ratio, and variance stability. `_checkpoint_sort_key` at `training/train_gvae.py:219` ranks by these metrics when configured. A stale simple script still uses the older metric at `run_gvae.py:75`.

# 6. Latent Extraction

Latent extraction is implemented in `utils/latent_extraction.py`.

For each split, `_extract_split_latents` at `utils/latent_extraction.py:125` calls `GVAE.get_separate_view_latent_params`, which returns per-view `mu`, `logvar`, and a view mask without using classifier logits or probabilities. It then saves:

- `per_view_mu`: shape `[N, 3, d_embed]`.
- `per_view_logvar`: shape `[N, 3, d_embed]`.
- `view_mask`: true where a modality is present.
- `stacked_mu`: per-view latent tensor.
- `stacked_z`: reparameterized per-view latent tensor.
- `concat_mu`: flattened `[clinical_mu, pathology_mu, radiology_mu]`.
- `concat_z`: flattened sampled latent.
- `fused_cls_mu`: fused latent before classifier head.
- `labels`, `patient_indices`, and metadata.

`extract_latents_for_ddpm` at `utils/latent_extraction.py:174` writes fold-specific latent payloads. It explicitly avoids saving GVAE classifier logits/probabilities as DDPM inputs. However, `extract_recommended_latents_for_ddpm` at `utils/latent_extraction.py:326` still allows non-`concat_mu` latent keys, which conflicts with the stricter intended design.

# 7. DDPM Pipeline

The correct DDPM pipeline is generator-only and lives in `training/latent_ddpm_augmentation.py`.

`train_conditional_ddpm_on_latents` at `training/latent_ddpm_augmentation.py:312` fits a `StandardScaler` on train-fold latent vectors only, trains a conditional DDPM to predict Gaussian noise, and conditions on shifted class tokens where token `0` is reserved for classifier-free guidance. It does not output response probabilities.

`ConditionalDDPM` in `models/ddpm.py:26` implements a standard denoising objective: sample timestep, add noise to `x0`, predict noise, and optimize MSE. Sampling in `models/ddpm.py:74` starts from noise and reverses the diffusion chain conditioned on a class label.

The incorrect historical design is still present in `training/train_pipeline.py`. `_evaluate_gvae_candidate_with_ddpm` at `training/train_pipeline.py:589` is blocked unless `allow_deprecated_ddpm_classifier=True`. If enabled, it trains separate unconditional DDPMs for responder/non-responder classes and turns class-specific denoising losses into probabilities. That path treats DDPM as a classifier and should remain disabled or be removed.

# 8. Synthetic Data Augmentation

Synthetic generation is implemented by `generate_synthetic_latents` at `training/latent_ddpm_augmentation.py:385`.

Supported augmentation modes:

- `minority_only`: generate only for the minority class within the train fold.
- `responder_only`: generate only class `1`.
- `both_classes`: generate each class proportionally.

Supported ratios are configurable and currently include `0.25`, `0.5`, `1.0`, and `2.0`. The generated latent vectors are inverse-transformed back to original latent scale before downstream training.

Quality metrics are computed in `evaluate_generated_latent_quality` at `training/latent_ddpm_augmentation.py:657`:

- MMD with RBF kernel.
- Class-wise mean distance.
- Class-wise covariance distance.
- Nearest-neighbor distance to real train latent vectors of the same class.

Optional filtering is implemented in `filter_synthetic_latents_by_knn` at `training/latent_ddpm_augmentation.py:569`. It removes generated samples whose same-class nearest-neighbor distance is above a train-derived quantile threshold.

# 9. Downstream Evaluation

Downstream evaluation is separate from GVAE direct prediction. `train_downstream_classifier` at `training/latent_ddpm_augmentation.py:237` trains either logistic regression or random forest on latent vectors. For logistic regression, the scaler is fit only on the training latent matrix at `training/latent_ddpm_augmentation.py:278`.

Ablations represented by the current pipeline:

- A: GVAE-only direct prediction metrics from `training/train_gvae.py`.
- B: downstream classifier on real `concat_mu` only.
- C: real `concat_mu` plus DDPM synthetic latent vectors.
- D: real `concat_mu` plus filtered DDPM synthetic latent vectors.
- E: PCA-transformed `concat_mu` plus DDPM synthetic latent vectors when `--pca-components` is provided.

The comparison between GVAE and downstream classifiers is useful but not perfectly fair: GVAE is an end-to-end model trained with graph reconstruction and classification objectives, while the downstream classifier is trained on frozen latent vectors. DDPM experiment metrics should be read as "downstream classifier after augmentation," not as DDPM prediction performance.

# 10. Data Leakage Audit

| Component | Current behavior | Leakage assessment |
|---|---|---|
| K-fold split | `StratifiedKFold` uses labels at `training/train_gvae.py:523`; train/val indices are stored in checkpoint metadata. | Acceptable supervised CV split. |
| GVAE classifier loss | Uses train batches only; validation uses held-out fold. | No obvious leakage. |
| Radiology pretraining | `pretrain_radiology_aggregator` receives only `train_idx` at `training/train_gvae.py:563` and filters held-out nodes. | No obvious leakage. |
| GVAE checkpoint selection | Uses validation metrics and latent-quality validation metrics. | Normal model selection, but not a final independent test. |
| Latent-quality scaler/probe | Scaler/probe fit on train `concat_mu` only at `training/train_gvae.py:136`. | Good. |
| DDPM scaler | Fit on train latent only at `training/latent_ddpm_augmentation.py:321`. | Good. |
| Downstream scaler | Fit on train latent only at `training/latent_ddpm_augmentation.py:278`. | Good. |
| PCA in augmentation | Fit on train latent only at `training/latent_ddpm_augmentation.py:858`. | Good. |
| `utils/data_utils.preprocess_fold_data_with_pca` | PCA fit on train nodes/lesions, then transforms all nodes at `utils/data_utils.py:161` and `utils/data_utils.py:175`. | Fit is train-only, but transformed val/test features exist in memory; safe only if downstream training respects indices. |
| Legacy DDPM classifier | Disabled unless opt-in; if used, it performs model selection through DDPM validation classification metrics. | Design leakage/misinterpretation risk. |
| Thresholded metrics | Best threshold is selected and reported on the same validation split in `utils/classification_eval.py:141`. | ROC/PR AUC OK; F1/balanced accuracy can be optimistic. |
| Data provenance | Serialized graphs exist without raw graph-build script. | Cannot audit whether similarity edges or preprocessing used all labels/folds. |

# 11. Metric Interpretation

- GVAE metrics are direct response prediction performance from the GVAE classifier head.
- DDPM training loss is denoising MSE, not response prediction loss.
- DDPM experiment metrics are downstream classifier metrics after training on real latent vectors plus optional DDPM-generated latent vectors.
- DDPM itself does not predict response in the correct pipeline.
- ROC AUC measures rank separation of predicted scores.
- PR AUC is more informative for imbalanced response labels.
- F1, balanced accuracy, sensitivity, and specificity depend on a decision threshold.
- MMD, mean distance, covariance distance, and kNN distance describe synthetic latent distribution quality; they are not clinical prediction metrics.

Latest runnable results already present in outputs:

| Experiment | Mean ROC AUC | Mean PR AUC | Mean balanced accuracy | Mean F1 |
|---|---:|---:|---:|---:|
| GVAE-only run `gvae_latent_quality_codex_20260615_204355` | 0.6894 | 0.8523 | 0.7265 | 0.8101 |
| Downstream real `concat_mu` only | 0.6705 | 0.8562 | 0.6872 | 0.7293 |
| Best DDPM augmentation by ROC AUC: `both_classes`, ratio `0.5`, unfiltered, no PCA | 0.6805 | 0.8597 | 0.6901 | 0.7142 |
| Best DDPM augmentation by balanced accuracy: `minority_only`, ratio `2.0`, unfiltered, no PCA | 0.6747 | 0.8590 | 0.7052 | 0.7993 |

These DDPM rows are downstream classifier results after augmentation, not DDPM classifier results.

# 12. Run Commands

Use the Windows virtual environment already present in the workspace:

```powershell
.venv-win312\Scripts\python.exe outputs\gvae\train_gvae_runner.py `
  --epochs 200 `
  --pretrain-epochs 400 `
  --checkpoint-metric latent_quality `
  --early-stopping-metric latent_quality `
  --top-k 5
```

Run the conditional latent DDPM augmentation from a selected GVAE run:

```powershell
.venv-win312\Scripts\python.exe outputs\gvae\train_conditional_ddpm_augmentation_runner.py `
  --gvae-run-id <GVAE_RUN_ID> `
  --checkpoint-root outputs\gvae\checkpoints `
  --checkpoint-selector rank `
  --rank 1 `
  --output-root outputs\conditional_latent_ddpm `
  --latent-key concat_mu `
  --augmentation-modes minority_only,responder_only,both_classes `
  --ratios 0.25,0.5,1.0,2.0 `
  --filter-synthetic `
  --filter-quantile 0.95 `
  --epochs 200 `
  --timesteps 100 `
  --batch-size 32 `
  --seed 42
```

Run PCA ablation E:

```powershell
.venv-win312\Scripts\python.exe outputs\gvae\train_conditional_ddpm_augmentation_runner.py `
  --gvae-run-id <GVAE_RUN_ID> `
  --latent-key concat_mu `
  --pca-components 32 `
  --augmentation-modes minority_only,responder_only,both_classes `
  --ratios 0.25,0.5,1.0,2.0 `
  --filter-synthetic
```

Avoid these legacy commands unless explicitly auditing deprecated behavior:

- `outputs/gvae/train_gvae_ddpm_runner.py`
- `outputs/gvae/train_ddpm_from_gvae_checkpoints_runner.py --allow-deprecated-loss-classifier`
- `training.train_pipeline.kfold_gvae_ddpm_generative_classifier`

# 13. Output Artifacts

GVAE artifacts:

- `outputs/gvae/checkpoints/<run_id>/fold_<k>/*.pt`: model checkpoints and split metadata.
- `outputs/gvae/metrics/<run_id>/summary.json`: cross-validation summary.
- `outputs/gvae/metrics/<run_id>/fold_metrics.csv`: fold-level metrics.
- `outputs/gvae/metrics/<run_id>/roc_data.json`: ROC curve data.
- `outputs/gvae/metrics/<run_id>/run_config.json`: run configuration.

Latent/DDPM augmentation artifacts:

- `outputs/conditional_latent_ddpm/<run_id>/summary.json`: aggregate downstream and synthetic-quality metrics.
- `outputs/conditional_latent_ddpm/<run_id>/comparison.csv`: ablation table.
- `outputs/conditional_latent_ddpm/<run_id>/fold_<k>/.../latent_artifacts.pt`: extracted train/validation latent tensors.
- `outputs/conditional_latent_ddpm/<run_id>/fold_<k>/.../ddpm_checkpoint.pt`: conditional DDPM model and train-only scaler statistics.
- `outputs/conditional_latent_ddpm/<run_id>/fold_<k>/.../generated/.../generated_latents.pt`: synthetic latent vectors and labels.
- `outputs/conditional_latent_ddpm/<run_id>/fold_<k>/.../quality_metrics.json`: synthetic quality metrics.
- `outputs/conditional_latent_ddpm/<run_id>/fold_<k>/.../downstream_metrics.json`: downstream classifier metrics.
- `outputs/conditional_latent_ddpm/<run_id>/best_result.json` and `best_result.csv`: best aggregate result from the latest completed run.
- `outputs/best_gvae_ddpm_result.json`: top-level copy of best result.

Known artifact issue: older generated output directories contain long checkpoint stems and can exceed Windows path limits, as shown by `git status` warnings.

# 14. Problems Found

| Severity | File/path | Function/class | What is wrong | Why it matters |
|---|---|---|---|---|
| High | `data_ln_pc_ihc_g.pt`, `data/data_247.pt` | Serialized data | The two graph files have opposite binary label polarity: `data_ln_pc_ihc_g.pt` has `{0:62, 1:185}`, while `data/data_247.pt` has `{0:185, 1:62}`. | Switching data files can invert responder/non-responder meaning and invalidate metrics. |
| High | Repository data preprocessing | Missing graph build script | No raw-to-`HeteroData` graph construction code was found. Similarity edges, masks, and preprocessing cannot be fully audited. | Data leakage in graph construction cannot be ruled out from source alone. |
| High | `training/train_pipeline.py:589` | `_evaluate_gvae_candidate_with_ddpm` | Deprecated path treats DDPM denoising loss as a classifier score when opt-in is enabled. | Violates the intended design that DDPM is not a predictor. |
| Medium | `outputs/gvae/train_gvae_ddpm_runner.py:222` | Runner config | Explicitly enables `allow_deprecated_ddpm_classifier=True`. | Easy to run an obsolete experiment and report DDPM as classifier. |
| Medium | `outputs/gvae/train_ddpm_from_gvae_checkpoints_runner.py:325` | CLI guard | Legacy loss-classifier runner still exists and can be enabled. | Keeps a misleading experimental path alive. |
| Medium | `utils/latent_extraction.py:326` | `extract_recommended_latents_for_ddpm` | Allows DDPM latent keys beyond `concat_mu`, including `fused_cls_mu`, `concat_z`, and `stacked_z`. | Conflicts with strict design: DDPM should consume only `concat_mu`. |
| Medium | `training/train_pipeline.py:548` | `_select_ddpm_latents` | Legacy selector supports `fused_cls_mu` and `pipeline_supervised_fused_mu`. | Can mix supervised fused representations into DDPM path if legacy classifier is enabled. |
| Medium | `run_gvae.py:75` | Manual runner config | Uses stale `auc_pr_balanced_accuracy` checkpoint and early-stopping metrics. | Users may train checkpoints optimized for prediction AUC rather than latent quality. |
| Medium | `training/train_gvae.py:498` | `kfold_train_gvae` | Clinical reconstruction assumes hard-coded continuous indices `0:5` and binary indices `5:22`. | Correct for `data_ln_pc_ihc_g.pt`, wrong for `data/data_247.pt` with 64 clinical features. |
| Medium | `utils/classification_eval.py:141` | `best_threshold_metrics` | Chooses and reports best threshold on the same validation split. | Thresholded metrics may be optimistic; AUC metrics remain safer. |
| Medium | `outputs/gvae/train_conditional_ddpm_augmentation_runner.py:41` | `latest_gvae_run_id` | Defaults to latest run by directory modified time if no run ID is supplied. | Can accidentally train DDPM from an incomplete or unintended GVAE run. |
| Medium | `configs/config.py` | Config | Empty config file; key experiment settings live in scripts. | Reproducibility and experiment tracking are weaker than they should be. |
| Medium | `training/train_ddpm.py:15` | `train_single_conditional_ddpm` | Splits whatever latent array it receives internally. | Safe only if caller passes train-fold latents; dangerous if full-dataset latents are passed. |
| Low | `training/train_gvae.py:1019` | Checkpoint file naming | Checkpoint filenames still include `_auc_` even when ranked by `latent_quality`. | Misleading artifact names. |
| Low | `utils/classification_eval.py:103` | `metrics_at_threshold` | Uses `scores > threshold` rather than `>= threshold`. | Minor convention difference, but can affect exact threshold metrics. |
| Low | `CLAUDE.md` | Documentation | References `training.ipynb`, which was not found in the file list. | Onboarding/run instructions can send users to missing files. |
| Low | `README.md` | Documentation | README only contains a title. | Users cannot reproduce the intended pipeline from README alone. |
| Low | `outputs/conditional_latent_ddpm/...` | Artifacts | Some generated output paths exceed Windows path length limits. | Makes cleanup, Git status, and artifact inspection painful. |

# 15. Recommended Fix Plan

1. Critical correctness bugs

- Decide and document the positive-class meaning for response labels.
- Remove or quarantine `data/data_247.pt` unless its label polarity is corrected and documented.
- Add a data validation script that checks feature dimensions, label polarity, masks, class counts, and edge types before training.

2. Data leakage

- Add or recover the raw graph-building script so similarity edges and feature preprocessing can be audited.
- Ensure graph construction never uses validation/test labels or fold-specific outcomes.
- Keep all scalers, PCA models, DDPMs, and downstream classifiers train-fold only.

3. Pipeline/design mismatch

- Remove legacy DDPM-as-classifier runners, or move them into an explicit `deprecated/` folder.
- Restrict DDPM latent input APIs to `concat_mu` only.
- Update `run_gvae.py` to use `latent_quality`, or replace it with the maintained runner.

4. Metric/reporting issues

- Label every DDPM table as downstream classifier performance after augmentation.
- Report GVAE direct metrics separately from latent downstream metrics.
- Prefer ROC AUC and PR AUC for model selection; report thresholded metrics with clear threshold source.
- Save a single best-result artifact per run, including ranking criterion.

5. Performance improvements

- Tune synthetic filtering because the latest `filter_quantile=0.95` run kept zero synthetic samples in filtered branches.
- Add repeated CV or bootstrap confidence intervals for small-sample stability.
- Add seed sweeps for DDPM and downstream classifiers.

# 16. Final Correct Pipeline

1. Validate `data_ln_pc_ihc_g.pt`: node types, feature dimensions, label polarity, masks, edge types, and class distribution.
2. Split patient nodes with stratified K-fold.
3. For each fold, train radiology MIL pretraining only on train-fold radiology patients.
4. Train GVAE on train fold with classifier BCE, reconstruction losses, KL, and contrastive loss.
5. During validation, compute direct GVAE prediction metrics and train-only latent-quality metrics.
6. Select GVAE checkpoints using latent-quality ranking when the checkpoint will feed DDPM augmentation.
7. Extract latent artifacts from selected GVAE checkpoints, using `concat_mu = [clinical_mu, pathology_mu, radiology_mu]` as the only DDPM input.
8. Fit optional PCA only on train-fold `concat_mu`; transform validation/test with that PCA.
9. Fit DDPM latent scaler only on train-fold `concat_mu`.
10. Train conditional DDPM with denoising MSE, conditioned on class label.
11. Generate synthetic train-fold latent vectors by mode and ratio.
12. Compute synthetic quality metrics: MMD, class-wise mean/covariance distance, and same-class train kNN distance.
13. Optionally filter generated samples using same-class train kNN thresholds.
14. Train downstream classifier only on train real latents or train real+synthetic latents.
15. Evaluate downstream classifier on held-out fold only.
16. Report GVAE direct prediction metrics separately from DDPM augmentation downstream metrics, and never report DDPM itself as a response predictor.
