# Diff-GVAE Feature Data Dictionary — 2026-08-06

Built by `research/2026-08-06-data-dictionary/build_dictionary.py` run with the repo venv; source CSVs read from `/Users/admin/Diff-GVAE/data`.

## Source files

| Source CSV | Rows | Raw columns | Feature columns (this dictionary) | Notes |
|---|---:|---:|---:|---|
| `x_clinical_unscaled_ln_pc_ihc_g.csv` | 247 | 23 | 22 | `main_index` dropped |
| `glcm_features.csv` | 157 | 139 | 137 | `main_index`, `dmp_pt_id` dropped |
| `radiology_features.csv` | 431 | 1674 | 1670 | `main_index`, `lesion_index`, `radiology_accession_number`, `dmp_pt_id` dropped |

Radiology raw column count is 1674; after dropping the four identifier columns the exact feature count is **1670** (the brief's estimate of 1671 counted only the three named identifiers — `dmp_pt_id` is also an identifier, not a radiomics feature).

## Clinical view (22 features)

| dtype | features |
|---|---:|
| binary | 17 |
| continuous | 5 |

| confidence | features |
|---|---:|
| High | 19 |
| Medium | 3 |
| Low | 0 |

| name | dtype | confidence | inferred_semantics |
|---|---|---|---|
| albumin | continuous | High | Serum albumin |
| dnlr | continuous | High | Derived neutrophil-to-lymphocyte ratio |
| TMB | continuous | High | Tumor mutational burden (mut/Mb) |
| tumor_burden | continuous | Medium | Tumor burden measure (units/method unknown) |
| clinical_pdl1_score | continuous | Medium | PD-L1 expression score 0-100 (TPS vs CPS unknown) |
| EGFR_driver | binary | High | Driver mutation flag: EGFR |
| ERBB2_driver | binary | High | Driver mutation flag: ERBB2 |
| BRAF_driver | binary | High | Driver mutation flag: BRAF |
| MET_driver | binary | High | Driver mutation flag: MET |
| STK11_driver | binary | High | Driver mutation flag: STK11 |
| ARID1A_driver | binary | High | Driver mutation flag: ARID1A |
| io_drug_Atezolizumab | binary | High | Treatment flag: received Atezolizumab (immunotherapy/agent) |
| io_drug_Durvalumab | binary | High | Treatment flag: received Durvalumab (immunotherapy/agent) |
| io_drug_Ipilimumab | binary | High | Treatment flag: received Ipilimumab (immunotherapy/agent) |
| io_drug_Nivolumab | binary | High | Treatment flag: received Nivolumab (immunotherapy/agent) |
| io_drug_Pembrolizumab | binary | High | Treatment flag: received Pembrolizumab (immunotherapy/agent) |
| io_drug_Resection | binary | Medium | Treatment flag: received tumor resection (surgery, not a drug) |
| io_drug_Tremelimumab | binary | High | Treatment flag: received Tremelimumab (immunotherapy/agent) |
| ecog_0 | binary | High | ECOG performance status one-hot: 0 |
| ecog_1 | binary | High | ECOG performance status one-hot: 1 |
| ecog_2 | binary | High | ECOG performance status one-hot: 2 |
| ecog_3 | binary | High | ECOG performance status one-hot: 3 |

## Pathology view (137 features)

| feature_family | features |
|---|---:|
| first_order | 5 |
| glcm | 132 |

| statistic | features |
|---|---:|
| kurtosis | 25 |
| mean | 25 |
| skewness | 25 |
| variance | 25 |
| lognorm_fit_p2 | 19 |
| lognorm_fit_p0 | 18 |

`original_pixels_channel_1_lognorm_fit_p2` is a lognormal-fit parameter (not a first-order statistic) but follows the `original_pixels_*` naming, so it is classed as first-order.

## Radiology view (1670 features)

| feature_class | features |
|---|---:|
| glcm | 422 |
| firstorder | 323 |
| glrlm | 287 |
| glszm | 287 |
| gldm | 251 |
| ngtdm | 86 |
| shape | 14 |

By filter:

| filter | features |
|---|---:|
| original | 107 |
| exponential | 93 |
| wavelet-HHH | 93 |
| wavelet-LLH | 93 |
| wavelet-LHL | 93 |
| wavelet-LHH | 93 |
| wavelet-HLL | 93 |
| wavelet-HLH | 93 |
| wavelet-HHL | 93 |
| squareroot | 93 |
| gradient | 93 |
| square | 93 |
| logarithm | 93 |
| lbp-3D-m2 | 93 |
| lbp-3D-m1 | 93 |
| lbp-3D-k | 93 |
| wavelet-LLL | 93 |
| lbp-2D | 75 |

## Required caveats

> **Graph dims 22/15/34; raw CSV dims 22/137/1671; reductions upstream (no build script in repo); semantics derived from column names, clinical meaning must be confirmed by cohort owner.**

Correction to the parenthetical raw-CSV figure: the radiology CSV carries **1670** feature columns once its four identifier columns (including `dmp_pt_id`) are excluded; 1671 counts only the three identifiers named in the brief. The 15-dim pathology and 34-dim radiology graph views are a subset of these columns, reduced upstream; no reduction script exists in this repo, so the mapping from raw column to graph slot is not recoverable from the repository alone.

## Data-quality notes

- Clinical: 0 missing values; all 17 binary columns take exactly {0,1}; ECOG is a perfect one-hot (every patient row sums to 1); patients may carry multiple driver mutations (max 2).
- Pathology: all 137 feature columns are NaN-free; the only missing values in the raw CSV are 14 empty `dmp_pt_id` cells (identifier column, dropped).
- Radiology: all 1670 feature columns are NaN-free; the only missing values in the raw CSV are 44 empty `dmp_pt_id` cells (identifier column, dropped).
