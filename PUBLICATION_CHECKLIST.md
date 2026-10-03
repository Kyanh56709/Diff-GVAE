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
- [ ] **A2.** Prove **imputation** (median/mode) is fit on **train fold only**, then applied to val/test. Currently unconfirmed. — *2026-10-03: build-time imputation median/mode is over the frozen cohort, feature columns only (no label reads) per the A1 script; in-pipeline scalers are train-fold-only per `tests/test_train_fold_only_fitting.py`. Spell out this scope in Methods.*
- [ ] **A3.** Prove **RobustScaler / feature scaling** is fit on train fold only. — *Same scope note as A2: build-time RobustScaler is part of the frozen dataset (features only); every in-pipeline scaler is train-fold-only per `tests/test_train_fold_only_fitting.py`.*
- [x] **A4.** Prove the **patient-similarity graph** (cosine **> 0.8**, verified by the A1 rebuild 2026-10-03) is constructed **without using val/test outcome information**. Edge weights are feature-based (OK), but confirm no label/target leaks into edge or feature construction. — **Confirmed 2026-10-03:** all three similarity graphs are cosine over feature matrices only; the build script never reads `binary_label`/`y`/`event`; edges verified bit-exact against canonical.
- [ ] **A5.** Confirm the message-passing split you already have (train→train-subgraph at `train_gvae.py:642`, val→val-subgraph at `:726`) is preserved in *every* code path used for reported numbers (including latent extraction and DDPM). Document it explicitly in Methods — it's a strength, use it.
- [ ] **A6.** Resolve the **two divergent result sets**: `training_local.ipynb` (~0.81–0.87) vs `PROJECT_REVIEW.md` outputs (~0.69). Pick ONE canonical run, re-run end-to-end, archive it. Explain the divergence in your lab notes. — **Partial (2026-10-03):** canonical protocol = pooled-OOF (A7); the ~0.69 → 0.6065 drop is explained in `research/2026-08-06-project-review-audit/reports/final_results_package.md` §4 (val-fold mean + tuned threshold vs pooled-OOF). **Still open:** the notebook's ~0.81–0.87 has not been reproduced or explained; the 2026-08-06 audit (`research/2026-08-06-project-review-audit/reports/report.md` §4) only shows the notebook is legacy, uses the non-canonical `data/data_247_scaled_ln_pc_ihc_g.pt` (cells 15–20) and does not run top-to-bottom.
- [x] **A7.** Freeze one **canonical GVAE run** and one **canonical DDPM augmentation run** for all paper numbers/figures. — **Done 2026-10-03** (graph `data_ln_pc_ihc_g_r32.pt`). Which run is behind which number: GVAE headline numbers = pooled-OOF `drop_both32_seed42`–`46` (`research/2026-10-03-radiology-artifact-ablation/output/`); DDPM latent source = `gvae_bestparam_ranked_r32_20261003_124706` (rank 1); DDPM A2 = `conditional_latent_ddpm_from_gvae_bestparam_ranked_r32_20261003_124706_20261003_125733`, A5 = `conditional_latent_ddpm_from_gvae_bestparam_ranked_r32_20261003_124706_20261003_130455`; `canonical_r32_20261003_111618` is a val-fold-mean reference only. All numbers in `research/2026-10-03-canonical-r32/reports/final_results_package_r32.md`.
- [x] **A8.** Fix/quarantine the opposite-label-polarity data file (`data/data_247.pt` vs `data_ln_pc_ihc_g.pt`); document which is authoritative and the positive-class definition (`binary_label=1` = responder?). — **Done:** `data/data_247.pt` quarantined earlier; `configs/data_247.pt` moved to `deprecated/configs_data_247.pt` on 2026-10-03 and documented in `deprecated/README.md`. Authoritative file = `data_ln_pc_ihc_g_r32.pt` since 2026-10-03 (same labels as `data_ln_pc_ihc_g.pt`; 247 patients; `binary_label=1` = non-responder, 185).

## B. Statistical rigor

- [ ] **B1.** Replace fold SD (±0.039) with **bootstrap 95% CIs** on all headline metrics.
- [ ] **B2.** Add a **locked held-out test set** OR **nested CV** (outer test never touched for model selection). Current CV mixes selection and evaluation.
- [ ] **B3.** Report **multiple seeds** (≥5) for GVAE and DDPM; report mean ± CI, not a single lucky run.
- [ ] **B4.** State **threshold-selection protocol** explicitly. Thresholded metrics (F1, balanced accuracy) are currently tuned on the same validation split → optimistic. Either report threshold-free (AUC/AUPRC) as primary, or select threshold on train/inner-val only.
- [ ] **B5.** Add **significance tests** for key comparisons (DeLong for AUC differences vs DyAM/XGBoost/SVM; paired test across folds for ablations).
- [ ] **B6.** Report **AUPRC** prominently alongside AUC (class imbalance; prevalence is skewed).

## C. Core-method experiments & baselines (the science)

- [ ] **C1.** Ablations with CIs (you have these — add stats): no-contrastive, no-GNN (MLP), unimodal, bimodal, full. Confirm each drop is significant.
- [ ] **C2.** Baselines re-run under **identical CV/splits/seeds**: XGBoost, SVM, DyAM re-implementation, plus a **late-fusion concat + MLP** baseline (the "no alignment" control).
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

- [ ] **H1.** Public, runnable repo: pinned `requirements.txt` + exact torch/torch-geometric/torch-scatter versions, seeds, config files (the empty `configs/config.py` must be populated).
- [ ] **H2.** One-command reproduction script for the **canonical** GVAE + DDPM runs.
- [ ] **H3.** Remove/quarantine deprecated DDPM-as-classifier path (`train_pipeline.py`) into a `deprecated/` folder so reviewers don't mistake it for the method.
- [ ] **H4.** Data dictionary: clinical (22 cols), pathology (GLCM features), radiology (34 radiomics) — define every feature.
- [ ] **H5.** Save per-sample outputs (scores, labels, patient IDs) for audit of ROC/PR/calibration.
- [ ] **H6.** Run the existing test suite; report pass status.

## I. Manuscript completeness (fix before submission)

- [ ] **I1.** Remove ALL Vietnamese placeholder notes ("Chua sua", "Co sua sau", "dien sau", "Ket qua cua em Loan", "Tom Tat", etc.).
- [ ] **I2.** Reconcile the **headline classifier AUC** — the draft shows both **0.806** and **0.867** for the full model. Decide which is the fusion-classifier AUC vs the linear-probe AUC and use consistently.
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
