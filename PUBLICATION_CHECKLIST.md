# Diff-GVAE — Journal Publication Checklist

**Target:** Methods / health-informatics Q1 (Medical Image Analysis, IEEE TMI/JBHI, npj Digital Medicine, Nature Communications).
**Data reality:** Public data only → external validation is **component-wise** (clinical/genomic), not full tri-modal.
**Timeline:** 1–3 months.

**Working assumptions** (change if wrong):
- Full multimodal external validation is **de-scoped** to component-wise and stated as a limitation.
- The **TSTR "denoising effect"** stays a headline **only if** it survives the controls in §E; otherwise the paper's center of gravity shifts to the Stage-1 representation-learning results, which are on firmer ground.

Legend: `[ ]` todo · **(BLOCKING)** = nothing downstream is credible until done.

---

## A. Internal validity & leakage — **do first (BLOCKING)**

- [x] **A1.** Recover or rewrite the raw-data → `data_ln_pc_ihc_g.pt` build script. Without it, preprocessing is unauditable and a reviewer rejects on this alone. — **Done 2026-10-03:** `data/build_ln_pc_ihc_g.py` reproduces the canonical graph exactly (14/14 checks, maxdiff 0.0; evidence `research/2026-10-03-a1-build-script/`).
- [x] **A2.** Prove **imputation** (median/mode) is fit on **train fold only**, then applied to val/test. — *2026-10-03: build-time imputation median/mode is over the frozen cohort, feature columns only (no label reads) per the A1 script; in-pipeline scalers are train-fold-only per `tests/test_train_fold_only_fitting.py`.* **Done 2026-10-06:** Methods scope written with file:line evidence (`research/2026-10-06-internal-validity-methods/reports/internal_validity_methods.md` §1); build-time imputation at `data/build_ln_pc_ihc_g.py:184-190,246-251,296-310` is feature-only, per-fold PCA fit train-only at `utils/data_utils.py:162,187`.
- [x] **A3.** Prove **RobustScaler / feature scaling** is fit on train fold only. — *Same scope note as A2: build-time RobustScaler is part of the frozen dataset (features only); every in-pipeline scaler is train-fold-only per `tests/test_train_fold_only_fitting.py`.* **Done 2026-10-06:** documented with evidence in the same report §1 (build-time RobustScaler `data/build_ln_pc_ihc_g.py:190,251,310`; probe scaler train-only `training/train_gvae.py:2265`).
- [x] **A4.** Prove the **patient-similarity graph** (cosine **> 0.8**, verified by the A1 rebuild 2026-10-03) is constructed **without using val/test outcome information**. Edge weights are feature-based (OK), but confirm no label/target leaks into edge or feature construction. — **Confirmed 2026-10-03:** all three similarity graphs are cosine over feature matrices only; the build script never reads `binary_label`/`y`/`event`; edges verified bit-exact against canonical.
- [x] **A5.** Confirm the message-passing split (train→train-subgraph at `train_gvae.py:642`, val→val-subgraph at `:726`) is preserved in *every* code path used for reported numbers (including latent extraction and DDPM). Document it explicitly in Methods. — **Done 2026-10-06:** `get_view_subgraph_and_features` keeps only edges with **both endpoints inside the index set** (`utils/data_utils.py:82-84`) and is used by `GVAE.forward` (`models/gvae_model.py:109`), encoder-only `get_mus_only` (`:244`), the pooled-OOF evaluation, and per-split DDPM latent extraction (`utils/latent_extraction.py`); covered by `tests/test_subgraph_remap.py`. Methods paragraph + evidence in `research/2026-10-06-internal-validity-methods/reports/internal_validity_methods.md` §2.
- [x] **A6.** Resolve the **two divergent result sets**: `training_local.ipynb` (~0.81–0.87) vs `PROJECT_REVIEW.md` outputs (~0.69). Pick ONE canonical run, re-run end-to-end, archive it. Explain the divergence in your lab notes. — **Done 2026-10-05:** `0.81/0.87` traced to `run_gvae_sweep`→`kfold_train_gvae` **val-fold** selection on the **34-dim** graph (`gvae_sweep_results.csv` row 83: mean_auc 0.8116, mean_f1 0.8787); it is circular (checkpoint and threshold chosen on the same val fold). Not reproducible with HEAD code (re-run with notebook config → 0.6086 with default `latent_quality`, 0.7190 with `checkpoint_metric='auc'`; the Jun-15 sweep code selected by val-AUC and had no `checkpoint_metric`). The notebook's own clean protocol (cell 4/6 `kfold_evaluate_gvae_classifier`) gives head AUC 0.6502, matching canonical. Canonical remains pooled-OOF r32 (head 0.6635±0.016). Evidence: `research/2026-10-05-a6-notebook-reconciliation/reports/a6_reconciliation.md`. **I2 reconciled too:** the draft mixes linear-probe (0.867) and fusion-head (0.806) AUCs from the legacy val-fold protocol; use the r32 package and label each explicitly.
- [x] **A7.** Freeze one **canonical GVAE run** and one **canonical DDPM augmentation run** for all paper numbers/figures. — **Done 2026-10-03** (graph `data_ln_pc_ihc_g_r32.pt`). Which run is behind which number: GVAE headline numbers = pooled-OOF `drop_both32_seed42`–`46` (`research/2026-10-03-radiology-artifact-ablation/output/`); DDPM latent source = `gvae_bestparam_ranked_r32_20261003_124706` (rank 1); DDPM A2 = `conditional_latent_ddpm_from_gvae_bestparam_ranked_r32_20261003_124706_20261003_125733`, A5 = `conditional_latent_ddpm_from_gvae_bestparam_ranked_r32_20261003_124706_20261003_130455`; `canonical_r32_20261003_111618` is a val-fold-mean reference only. All numbers in `research/2026-10-03-canonical-r32/reports/final_results_package_r32.md`.
- [x] **A8.** Fix/quarantine the opposite-label-polarity data file (`data/data_247.pt` vs `data_ln_pc_ihc_g.pt`); document which is authoritative and the positive-class definition (`binary_label=1` = responder?). — **Done:** `data/data_247.pt` quarantined earlier; `configs/data_247.pt` moved to `deprecated/configs_data_247.pt` on 2026-10-03 and documented in `deprecated/README.md`. Authoritative file = `data_ln_pc_ihc_g_r32.pt` since 2026-10-03 (same labels as `data_ln_pc_ihc_g.pt`; 247 patients; `binary_label=1` = non-responder, 185).

## B. Statistical rigor

- [x] **B1.** Replace fold SD (±0.039) with **bootstrap 95% CIs** on all headline metrics. — **Done 2026-10-03:** pooled-OOF bootstrap 95% CIs in `oof_metrics_with_ci.json` per seed (`research/2026-10-03-radiology-artifact-ablation/output/drop_both32_seed4{2..6}/`); headline table in `research/2026-10-03-canonical-r32/reports/final_results_package_r32.md` §2a.
- [x] **B2.** Add a **locked held-out test set** OR **nested CV** (outer test never touched for model selection). Current CV mixes selection and evaluation. — **Done 2026-10-06:** `kfold_evaluate_gvae_classifier` already keeps the outer fold report-only (inner-val drives checkpoint + threshold); the remaining step (hyperparameter choice) is now covered by a nested CV that selects among 3 pre-registered configs on each fold's inner-val, never the outer test. Result: nested-CV head AUC **0.6603 ± 0.0460** vs fixed-config **0.6531 ± 0.0113** (seeds 42–46, per-seed reseeded) — Δ +0.007 is inside the nested sd and seed variance rises ~4×, so selection does **not** manufacture the headline. Harness is bit-exact to `run_oof.py` per seed on the canonical-only path. Evidence: `research/2026-10-06-nested-cv-hparam/reports/nested_cv_report.md`.
- [ ] **B3.** Report **multiple seeds** (≥5) for GVAE and DDPM; report mean ± CI, not a single lucky run. — **Partial 2026-10-03:** GVAE pooled-OOF done for 5 seeds (42–46, mean ± sd in `final_results_package_r32.md` §2a). **Still open:** DDPM augmentation runs (A2/A5) are single-seed.
- [x] **B4.** State **threshold-selection protocol** explicitly. Thresholded metrics (F1, balanced accuracy) are currently tuned on the same validation split → optimistic. Either report threshold-free (AUC/AUPRC) as primary, or select threshold on train/inner-val only. — **Done 2026-10-06:** AUC/PR-AUC are the primary (threshold-free) metrics; F1/BA use a threshold selected on **inner-val only** (`_threshold_max_f1` at `training/train_gvae.py:2243` in `kfold_evaluate_gvae_classifier`), with the optimism caveat documented in `utils/classification_eval.py:271-275`. Protocol written in `research/2026-10-06-internal-validity-methods/reports/internal_validity_methods.md` §3.
- [ ] **B5.** Add **significance tests** for key comparisons (DeLong for AUC differences vs DyAM/XGBoost/SVM; paired test across folds for ablations). — **Partial 2026-10-04:** DeLong paired tests (`review_fixes_2026_07/delong.py`) of GVAE head/probe vs 5 baselines on identical splits, 5 seeds, Holm-adjusted: no head-vs-baseline difference is significant (min p_holm 0.157); `research/2026-10-04-baselines-delong/reports/baselines_delong_report.md`. **Still open:** paired tests for ablations (C1) and DyAM.
- [x] **B6.** Report **AUPRC** prominently alongside AUC (class imbalance; prevalence is skewed). — **Done 2026-10-06:** PR-AUC already computed with bootstrap CIs (`kfold_evaluate_gvae_classifier`) and in `final_results_package_r32.md` §2a — head 0.8063 [0.7421–0.8803] / 5-seed 0.8297 ± 0.0133; probe 0.8221 / 0.8084 ± 0.0234 (prevalence 0.749). Carrying it into the manuscript headline table is tracked under I4.

## C. Core-method experiments & baselines (the science)

- [ ] **C1.** Ablations with CIs (you have these — add stats): no-contrastive, no-GNN (MLP), unimodal, bimodal, full. Confirm each drop is significant. — *Partial 2026-10-06:* 7/8 config arms run (`research/2026-10-06-c1-ablations/`), pooled-OOF seeds 42–46, DeLong vs full + Holm. **No drop is significant after Holm** (min p_holm 0.099, `radiology_only`); `full` (0.6531 ± 0.0113) is **not** the best arm — `pathology_only` 0.6790, `clinical_only` 0.6615 nominally higher, and radiology-only 0.5286 is the weakest view; `no_contrastive` ≥ full (contrastive not helping). Consistent with the C2 baseline result. **Still open:** no-GNN (MLP) arm (needs an `encoder_type`/MLP opt-in flag in `models/`), aggregator-pooling arm (C5). Evidence: `research/2026-10-06-c1-ablations/reports/c1_ablations_report.md`.
- [ ] **C2.** Baselines re-run under **identical CV/splits/seeds**: XGBoost, SVM, DyAM re-implementation, plus a **late-fusion concat + MLP** baseline (the "no alignment" control). — **Partial 2026-10-04:** LR (clinical-only and concat), RBF-SVM, gradient-boosted trees (sklearn HistGradientBoosting in place of XGBoost, to keep the pinned env), and late-fusion MLP on the GVAE pooled-OOF splits, seeds 42–46 (`research/2026-10-04-baselines-delong/reports/baselines_delong_report.md`). Clinical-only LR has the highest mean ROC-AUC (0.707 ± 0.016 vs GVAE head 0.664 ± 0.016); not significant after Holm. **Still open:** DyAM re-implementation.
- [ ] **C3.** Hyperparameter sensitivity (τ=0.2, embedding dim, heads, graph threshold 0.7 vs 0.8) → move from Supplementary claims to a real table with numbers.
- [ ] **C4.** Complete-cohort robustness (N=366, AUC 0.711±0.080) — re-run under canonical pipeline, report with CI.
- [ ] **C5.** Ablate the **lesion attention aggregator** vs mean/max pooling (currently only "likely contributes" — quantify it).

## D. External validation — public, component-wise

- [ ] **D1.** Identify ≥1 **public NSCLC anti-PD-(L)1 cohort with clinical(+genomic) + response** (search cBioPortal MSK IO cohorts, e.g. Rizvi/Hellmann/Samstein-type; GEO; published supplementary tables). **Verify** each has: NSCLC, IO treatment, a response/outcome label. Do NOT assume modality coverage.
- [ ] **D2.** Externally validate the **clinical(+genomic) sub-model**: does the clinical encoder separate responders on the external cohort (report AUC + CI)?
- [ ] **D3.** Externally validate the **spontaneous gene-encoding claim** (EGFR/STK11 linear-probe AUC) on the external cohort — this is your most novel and most testable biological claim.
- [ ] **D4.** (Stretch) Search **TCIA** for an NSCLC-IO imaging cohort with response labels for radiology-view external check. Expect most TCIA NSCLC sets to be staging/survival, not IO — treat as bonus.
- [ ] **D5.** Write external validation up **honestly**: "external validation of the clinical modality and gene-signal claim; full multimodal external validation is infeasible on current public data (Limitation)."
- [ ] **D6.** Address **domain shift / harmonization** in Methods (different institutions, IHC/scanner protocols) — even a paragraph acknowledging it is expected.

## E. Generative-claim validation (TSTR / denoising effect)

- [ ] **E1.** **TSTR controls (make-or-break):** compare DDPM augmentation against (a) SMOTE, (b) Gaussian jitter, (c) no augmentation, across ≥5 seeds with CIs. If DDPM ≈ SMOTE, drop "denoising effect" framing.
- [ ] **E2.** Evaluate TSTR on a **properly held-out real test set**, not the train-fold-adjacent split, to rule out leakage as the cause of TSTR > Real-to-Real.
- [ ] **E3.** Strengthen the **distribution-fidelity** discriminator (you have AUC 0.58): add a two-sample test (MMD permutation test p-value), per-feature KS tests, correlation-matrix Frobenius distance with a null.
- [ ] **E4.** **Mode-coverage / diversity** check: precision-recall for generative models (or nearest-neighbor distance distributions) to show synthetic data isn't collapsed/memorized. Confirm no synthetic sample is a near-copy of a real train patient (privacy + memorization).
- [ ] **E5.** Report the **centroid-shift cosine similarity** number (currently `= ....` placeholder in the draft).
- [ ] **E6.** Reframe **in-silico gene–TMB analysis** explicitly as *hypothesis generation, not clinical discovery* (your own note already flags this). Caveat binary-conditioning imperfection.

## F. Interpretability (expected at methods Q1)

- [ ] **F1.** Feature attribution on the fusion/classifier (SHAP or Integrated Gradients).
- [ ] **F2.** Visualize/analyze **lesion attention weights** — show the aggregator attends to plausible lesions.
- [ ] **F3.** Latent-space visualization (UMAP/t-SNE colored by response and by driver-gene status) to support the "biologically meaningful latent" claim.

## G. Reporting standards & compliance

- [ ] **G1.** Complete a **TRIPOD+AI** checklist (prediction-model reporting standard) — reviewers at these venues expect it; include as supplementary.
- [ ] **G2.** If imaging is emphasized, complete a **CLAIM** checklist (imaging-AI reporting).
- [ ] **G3.** Data-availability statement: MSK-MIND/Vanguri source access terms; your GitHub repo link (currently `clm-gvae` — make it real and public at submission).
- [ ] **G4.** Ethics/IRB statement for the source cohort (cite Vanguri et al.'s approvals; you're using public derived data).
- [ ] **G5.** Compute/runtime disclosure (hardware, training time) — required by several target venues.

## H. Reproducibility & code release

- [x] **H1.** Public, runnable repo: pinned `requirements.txt` + exact torch/torch-geometric/torch-scatter versions, seeds, config files (the empty `configs/config.py` must be populated). — **Done 2026-10-03** (`6add390`): exact pins in `requirements.txt` (Python 3.9.6, torch 2.8.0, torch_geometric 2.6.1, torch_scatter 2.1.2); `configs/config.py` holds canonical paths + build constants (not yet imported by the pipeline). Making the repo public belongs to G3.
- [ ] **H2.** One-command reproduction script for the **canonical** GVAE + DDPM runs.
- [x] **H3.** Remove/quarantine deprecated DDPM-as-classifier path (`train_pipeline.py`) into a `deprecated/` folder so reviewers don't mistake it for the method. — **Done:** `deprecated/train_pipeline.py` + legacy runners, documented in `deprecated/README.md`; guarded by `tests/test_deprecated_layout.py`.
- [x] **H4.** Data dictionary: clinical (22 cols), pathology (GLCM features), radiology (34 radiomics) — define every feature. — **Done:** `research/2026-08-06-data-dictionary/` (`build_dictionary.py` regenerates the CSVs); exact slot→column order in `data/build_ln_pc_ihc_g.py` (`PATHOLOGY_FEATURES`, `RADIOLOGY_FEATURES`; canonical r32 has 32 radiology slots).
- [ ] **H5.** Save per-sample outputs (scores, labels, patient IDs) for audit of ROC/PR/calibration. — **Partial:** GVAE scores + labels in `oof_arrays.npz` (y_true, head_probs, probe_probs; 247 entries, no explicit patient-ID array) + `oof_per_fold.csv` per seed in `research/2026-10-03-radiology-artifact-ablation/output/drop_both32_seed4{2..6}/`. **Still open:** save patient IDs alongside the scores; DDPM runs keep per-fold `summary.json` only.
- [x] **H6.** Run the existing test suite; report pass status. — **Done 2026-10-04:** `.venv/bin/python -m pytest` → 96 passed.

## I. Manuscript completeness (fix before submission)

- [ ] **I1.** Remove ALL Vietnamese placeholder notes ("Chua sua", "Co sua sau", "dien sau", "Ket qua cua em Loan", "Tom Tat", etc.).
- [x] **I2.** Reconcile the **headline classifier AUC** — the draft shows both **0.806** and **0.867** for the full model. Decide which is the fusion-classifier AUC vs the linear-probe AUC and use consistently. — **Done 2026-10-05 (per A6):** both are legacy val-fold numbers. `0.867±0.039` = combined **linear probe** (Table 1) but is also misused as the "full GVAE" AUC in §4.1.1; `0.806` = **fusion head** after contrastive ablation (0.806→0.761) and in the modality ablation (clinical+pathology 0.772). Canonical r32: fusion head **0.6635±0.016** (seed 42 0.6431), linear probe **0.6356±0.024** — use `final_results_package_r32.md` and label head vs probe explicitly. See `research/2026-10-05-a6-notebook-reconciliation/reports/a6_reconciliation.md` §4.
- [ ] **I3.** Fix framework-name inconsistency: **Diff-GVAE** vs **CLM-GVAE** vs **CALM-VAE** appear interchangeably. Pick one.
- [ ] **I4.** Fill all placeholder numbers: centroid cosine sim, Table 2 gene rows (MET/ARID1A cut off), Stage-2 hyperparameters (§3.3.7 "dien sau"), Supplementary Tables S1–S4.
- [ ] **I5.** Render all figures at publication quality (Fig 2 architecture, Fig 3 PCA/centroid, add UMAP, calibration, external-validation ROC). Resolve broken `Figure??` cross-references.
- [ ] **I6.** Rewrite Abstract/Intro/Related Work/Conclusion after results are final (you flagged these as draft).
- [ ] **I7.** Ensure Introduction contributions match final evidence (don't claim external validation you didn't do; don't claim "denoising effect" if E1 kills it).
- [ ] **I8.** Complete author list, affiliations, CRediT statement, references formatting (some refs truncated/incomplete in draft).
- [ ] **I9.** Add a **Limitations** paragraph covering: single source cohort, N=247, component-wise external validation only, synthetic-data hypothesis status, retrospective data.

## J. Submission logistics

- [ ] **J1.** Confirm final target journal + read its author guidelines (length, structure, checklist requirements).
- [ ] **J2.** Prepare cover letter positioning the contribution (generative multimodal framework, not just a classifier).
- [ ] **J3.** Suggest reviewers / declare competing interests.
- [ ] **J4.** Optional but strong: post a **preprint** (arXiv/medRxiv) once internally valid, to establish priority.

---

## Critical path (if you do nothing else, do these in order)

1. **A1–A8** — prove internal validity, pick canonical run. *Everything else is worthless without this.*
2. **B1–B4** — CIs + held-out/nested CV.
3. **E1–E2** — TSTR controls. Decides whether "denoising effect" is a headline or a footnote.
4. **D1–D3** — component-wise external validation of the clinical + gene claims.
5. **I1–I9** — finish the manuscript.
