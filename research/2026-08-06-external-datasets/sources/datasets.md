# External ICI datasets for Diff-GVAE adaptation (NSCLC + non-NSCLC solid tumors)

Research date: 2026-08-06 (session `dataset-search`). Every URL below was fetched/verified during this session unless explicitly marked otherwise. Raw evidence: `sources/raw/`.

## Criteria update (lead decision, 2026-08-06)

Datasets do **NOT** have to be NSCLC. **NSCLC remains the highest priority**, but the search now also includes **any solid tumor treated with immune checkpoint inhibitors (anti-PD-(L)1 / anti-CTLA-4)** that has **treatment response labels** (RECIST/irRECIST/iRECIST, CR/PR/SD/PD or binary) **and ≥2 modalities** (clinical tabular / imaging or pre-extracted radiomics / pathology / genomics). Every candidate below is explicitly marked **NSCLC: Yes** or **NSCLC: No**.

Network notes (same as previous session): TCIA website/API and Synapse unreachable from this network (timeouts / NIH geo-block NOT-OD-25-083); TCIA facts verified via Wayback captures. Bioconductor's `IMvigor210CoreBiologies` package was **removed** from all release branches (checked 3.17–3.21); its data is available via GitHub mirrors and cBioPortal/iAtlas instead (see entry 5).

## Tier summary

| Tier | # | Dataset | Tumor (NSCLC?) | Modalities | N (ICI) | Response labels | License |
|---|---|---|---|---|---|---|---|
| **S** | 1 | I3LUNG DATASETS (Zenodo 17535424) | NSCLC — **Yes** | clinical + pyradiomics + FM radiology + digital pathology + genomics | 2,075 (tri-modal 391) | BEST RESPONSE (PD/SD/PR/CR), ORR, DCR, CBR, PFS, OS | CC-BY-NC-4.0 |
| **S** | 2 | cBioPortal `lung_msk_mind_2020` (Vanguri 2022) | NSCLC — **Yes** | clinical + MSK-IMPACT mutations/CNA/SV | 247 | BOR (CR/PR/SD/POD), DCR, PFS, OS | cBioPortal public |
| **S/A** | 3 | Synapse `syn26642505` (Vanguri 2022 full release) | NSCLC — **Yes** | "all data" (incl. radiomics + PD-L1 IHC texture) | 247 | RECIST binarized | Synapse (geo-blocked) |
| **A** | 4 | cBioPortal `nsclc_mskcc_2018` (Hellmann/CheckMate 012) | NSCLC — **Yes** | clinical + WES | 75 | BEST_OVERALL_RESPONSE, DCB, PFS | cBioPortal public |
| **A** | 5 | **IMvigor210** — GitHub mirror of Bioconductor `IMvigor210CoreBiologies` | **Urothelial (bladder) — No** | clinical + RNA-seq counts + FMOne targeted mutations + TMB | 348 (RNA-seq 298) | **RECIST v1.1 4-class (CR/PR/SD/PD) + binaryResponse** + OS | Genentech license (package LICENSE; raw EGA controlled) |
| **A** | 6 | **IMvigor210** — cBioPortal `blca_iatlas_imvigor210_2017` (iAtlas) | **Urothelial (bladder) — No** | clinical + WES mutations + RNA-seq + gene signatures | 347 | RESPONDER (mRECIST CR/PR) 68/230, CB, PROGRESSION, OS, TMB, neoantigens | cBioPortal/iAtlas public |
| **A** | 7 | **IMmotion150** — cBioPortal `rcc_iatlas_immotion150_2018` | **RCC — No** | clinical + WES mutations + RNA-seq + gene signatures | 263 (174 atezolizumab) | RESPONDER 72/175, CB, PROGRESSION, PFS | cBioPortal/iAtlas public |
| **A** | 8 | **Gide 2019** — cBioPortal `mel_iatlas_gide_2019` | **Melanoma — No** | clinical + RNA-seq + gene signatures | 91 (75 labeled) | RESPONDER 40/35, CB, PROGRESSION, OS | cBioPortal/iAtlas public |
| **B** | 9 | cBioPortal `tmb_mskcc_2018` (Samstein 2019) | pan-cancer incl. NSCLC — **Yes** (≈315) | clinical + MSK-IMPACT | 1,661 | OS only (no RECIST in portal) | cBioPortal public |
| **B** | 10 | GEO GSE126044 | NSCLC — **Yes** | RNA-seq | 16 | responder/non-responder in metadata | GEO public |
| **B** | 11 | GEO GSE135222 | NSCLC — **Yes** | RNA-seq | 27 | response in paper only | GEO public |
| **B** | 12 | Zenodo 5162861 (Granata 2021) | NSCLC (LUAD) — **Yes** | CT radiomics (IBSI) | 88 | OS/PFS only (no RECIST) | CC-BY-4.0 |
| **B** | 13 | TCIA Anti-PD-1_Lung | NSCLC — **Yes** | CT + PET DICOM | 46 | none | CC BY 3.0 |
| **B** | 14 | **Hugo 2016** — GEO GSE78220 + cBioPortal `mel_ucla_2016` | **Melanoma — No** | RNA-seq + WES + CNA + clinical | 28 (GEO) / 38 (portal) | `anti-pd-1 response` in GEO characteristics; DCB (CR 7/PR 14/PD 17) | GEO + cBioPortal public |
| **B** | 15 | **Liu 2019** — cBioPortal `mel_iatlas_liu_2019` | **Melanoma — No** | clinical + WES mutations + RNA-seq | 122 | RESPONDER 48/74, CB | cBioPortal/iAtlas public |
| **B** | 16 | **Riaz 2017** — cBioPortal `mel_iatlas_riaz_nivolumab_2017` | **Melanoma — No** | clinical + WES mutations + RNA-seq | 107 (64 nivo; 56 labeled) | RESPONDER 11/45 | cBioPortal/iAtlas public |
| **B** | 17 | **Miao 2018** — cBioPortal `mixed_allen_2018` (MSS pan-cancer) | **Mixed (NSCLC, melanoma, HNSCC, RCC, bladder, CRC-MSS) — No (mostly)** | clinical + WES mutations | 249 (HNSCC ≈12) | RECIST_RESPONSE: CB 70 / SD 56 / NCB 123, PFS, OS | cBioPortal public |
| **B** | 18 | **PRINCE trial** — cBioPortal `paad_iatlas_prince_2022` | **Pancreatic — No** | clinical + WES mutations + RNA-seq | 93 | RESPONDER 31/35 | cBioPortal/iAtlas public |
| **C** | 19 | TCIA classic NSCLC radiomics cohorts | NSCLC — **Yes** | CT (+seg, some +RNA-seq) | 89–422 | none | TCIA (CC BY 3.0) |
| **C** | 20 | **Jerby-Arnon 2018** — GEO GSE115821 | **Melanoma — No** | RNA-seq | 37 | R/NR in metadata (3 R / 34 NR) | GEO public |
| **C** | 21 | **Anders 2022 TNBC** — cBioPortal `brca_iatlas_anders_2022` | **TNBC (breast) — No** | clinical + WES mutations + RNA-seq | 31 | RESPONDER 7/24 | cBioPortal/iAtlas public |
| **C** | 22 | **Cloughesy/Prins 2019 GBM** — cBioPortal `gbm_iatlas_prins_2019` | **Glioblastoma — No** | clinical + RNA-seq | 30 | RESPONDER (all FALSE among 28 labeled; RANO-based) | cBioPortal/iAtlas public |
| **C** | 23 | **Choueiri 2016 ccRCC** — cBioPortal `ccrcc_iatlas_choueiri_2016` | **RCC — No** | clinical + WES mutations + RNA-seq | 16 | RESPONDER 3/13 | cBioPortal/iAtlas public |
| **C** | 24 | GEO GSE332566 | **HNSCC — No** | methylation arrays (+ paper genomics/transcriptomics) | 26 | response in paper only | GEO public |
| **C** | 25 | GEO GSE205506 | **dMMR/MSI-H CRC — No** | scRNA-seq (19 patients, pre/on PD-1) | 19 pts | response (pCR) in paper | GEO public |
| **C** | 26 | ENA PRJEB23709 (Gide 2019 raw) | **Melanoma — No** | raw RNA-seq FASTQ | 91 runs | labels in paper supplement only | ENA public |

---

## Tier S — NSCLC (highest priority)

### 1. I3LUNG DATASETS — Zenodo record 17535424 — **NSCLC: Yes**
- **Description**: Feature-level release of the I3LUNG consortium (EU Horizon trial NCT05537922, 6 centers). Pre-extracted multimodal features + immunotherapy outcomes for real-world NSCLC ICI patients. The only **full tri-modal (clinical + radiology + pathology) NSCLC+ICI public dataset** found.
- **Verified URL**: https://zenodo.org/records/17535424 (record JSON in `raw/zenodo_rec_17535424.json`; downloaded and inspected `I3LUNG_DATA.zip`, 26,116,168 B — listing in `raw/i3lung_evidence/zip_listing.txt`).
- **License/access**: open, CC-BY-NC-4.0 (non-commercial).
- **Modalities**: clinical `cb.csv` (2,075×57: age, sex, ECOG, smoking, PD-L1, stage, mets sites, IO line, LDH, NLR, BMI, LIPI, center); `pyradiomics.csv` (877×131); `fmrad.csv` (896×4,099 FM embeddings); `digital_pathology.csv` + `digital_pathology_titan.csv` (846×771 GigaPath/TITAN embeddings); `genomics.csv` (1,705: KRAS/TP53/STK11/DRIVER); `features.json`.
- **N**: 2,075 subjects with outcomes; all NSCLC ICI. Tri-modal (cb+outcomes+pyradiomics+pathology) = 391; cb+radiomics 877; cb+pathology 846; cb+genomics 1,705.
- **Response labels**: BEST RESPONSE 4-class (0=PD 894, 1=SD 532, 2=PR 582, 3=CR 60; consistency-checked), ORR/DCR/CBR binary, PFS/OS.
- **Download**: `curl -L -o I3LUNG_DATA.zip https://zenodo.org/api/records/17535424/files/I3LUNG_DATA.zip/content` (26 MB; `results.zip` 2.08 GB optional).
- **Fit tier: S** (clinical view maps nearly 1:1; pyradiomics ≈ project's radiology view; pathology = FM embeddings, needs extractor swap).
- **Blockers**: non-commercial license; features only (no raw CT/WSI); radiology per-patient not per-lesion; pathology WSI-level embeddings not GLCM.
- **Verification**: record API + full zip download + column/row inspection + label-consistency check, this session. Related: protocol PMID 36959048; validation preprint DOI 10.64898/2026.01.16.25342913 (HTTP 200).

### 2. cBioPortal `lung_msk_mind_2020` — Vanguri et al. 2022 cohort — **NSCLC: Yes**
- **Description**: Genomic/clinical component of the project's own source cohort (MSK MIND; 247 advanced NSCLC on PD-(L)1; Nature Cancer 2022).
- **Verified URL**: https://www.cbioportal.org/study/summary?id=lung_msk_mind_2020 (API data in `raw/cbio_lung_msk_mind_2020*.json`, `raw/cbio_mind_clinical_*.json`).
- **License/access**: public via portal/API; no formal license (cBioPortal terms; MSK data policy).
- **Modalities**: clinical (AGE, SEX, ECOG, ALBUMIN, DNLR, PACK_YEARS, SMOKING, PDL1 246, JS_PDL1 201, TMB, FGA, driver flags, IO_DRUG/LINE/MONO_COMBO, CT_SCAN_TYPE, MANUAL_ANNOTATION 193, HALO_TUMOR_QUALITY 164, MSI, PFS/OS) + MSK-IMPACT mutations/CNA/SV.
- **N**: 247 (246 with BOR). All NSCLC ICI.
- **Response labels**: BOR (RECIST): CR 6, PR 55, SD 48, POD 92, POD/death 29, POD/brain 9, POD/clinical 4, POD/bone 3; DCR; PFS/OS.
- **Download**: `curl "https://www.cbioportal.org/api/studies/lung_msk_mind_2020/clinical-data?clinicalDataType=PATIENT&projection=DETAILED&pageSize=10000"` (+ `/molecular-data/mutations`).
- **Fit tier: S** for clinical+genomics (matches the 22-dim clinical view nearly 1:1).
- **Blockers**: no radiology/pathology features in the portal (those live on Synapse); no imaging.
- **Verification**: study record, attribute list, BOR distribution, molecular profiles — fetched this session.

### 3. Synapse `syn26642505` — Vanguri et al. 2022 full data release — **NSCLC: Yes**
- **Description**: The paper's official release ("All data are publicly available at synapse … syn26642505", verified verbatim from PMC9586871; also cited by KatherLab Cancer Res Commun 2024 and LORIS Nat Cancer 2024).
- **Verified URL**: https://www.synapse.org/#!Synapse:syn26642505 — existence verified via 3 independent data-availability statements; **direct access blocked from this region** (NOT-OD-25-083).
- **License/access**: Synapse project; download requires account + ToU acceptance.
- **Modalities**: expected cohort clinical tables, response labels, radiomics features, PD-L1 IHC texture, genomic features; images not released.
- **N**: 247; binarized CR/PR (62, 25%) vs SD/PD (185).
- **Response labels**: RECIST v1.1 best response, binarized.
- **Download**: web UI only (geo-blocked from this region).
- **Fit tier: S/A** — the exact cohort the project was built for.
- **Blockers**: geo-block; account required; file inventory unverified.
- **Verification**: PMC full text DAS (this session) + 2 papers' DAS.

---

## Tier A

### 4. cBioPortal `nsclc_mskcc_2018` — Hellmann et al. Cancer Cell 2018 (CheckMate 012 WES) — **NSCLC: Yes**
- **Description**: WES of 75 tumor/normal NSCLC pairs, PD-1 + CTLA-4 blockade (nivolumab±ipilimumab).
- **Verified URL**: https://www.cbioportal.org/study/summary?id=nsclc_mskcc_2018 (PMID 29657128; DOI 10.1016/j.ccell.2018.03.018 via OpenAlex).
- **License/access**: public portal/API.
- **Modalities**: clinical (AGE, SEX, ECOG, SMOKING, HISTOLOGY, PDL1_EXP, BOR, DCB, PFS, TMB, neoantigen, HLA) + WES mutations.
- **N**: 75. **Response labels**: BEST_OVERALL_RESPONSE, DURABLE_CLINICAL_BENEFIT, PFS.
- **Download**: portal API (pattern as #2).
- **Fit tier: A** (clinical+genomics; small N). **Blockers**: no imaging; N=75; combination ICI.
- **Verification**: study record + attributes fetched this session.

### 5. IMvigor210 — Bioconductor `IMvigor210CoreBiologies` via GitHub mirror — **NSCLC: No (urothelial/bladder)** ★ lead-named target
- **Description**: The complete data package for the Mariathasan et al. Nature 2018 IMvigor210 cohort (atezolizumab, anti-PD-L1, metastatic urothelial carcinoma; NCT02108652/NCT02951767) — the canonical public ICI cohort with **clinical + RNA-seq + mutation + response**. The Bioconductor package was **removed from all release branches** (3.17–3.21 all 404 this session) and the original Genentech host (`research-pub.gene.com`) is down, but two **GitHub mirrors of the original package** were verified this session.
- **Verified URLs** (both checked this session; identical file trees, same DESCRIPTION v2.0.0 "Data and software R package enabling the reproduction of results presented in the manuscript Mariathasan et al. …"):
  - https://github.com/BioInfoCloud/IMvigor210CoreBiologies (branch `main`; `data/cds.RData` 15,825,165 B)
  - https://github.com/SiYangming/IMvigor210CoreBiologies (branch `master`; `data/cds.RData` 51,233,046 B)
  - Evidence: `raw/imvigor210_description.txt`, `raw/imvigor210_data_doc.txt` (verbatim `R/data.R` docs).
- **License/access**: package LICENSE is a PDF (142,111 B) from Genentech — free for research use but **not OSI open source**; raw sequencing reads are EGA-controlled (EGAS00001002556). GitHub mirrors are publicly downloadable.
- **Modalities** (from `R/data.R` docs + file inspection):
  - `data/cds.RData` (validated: RDX3 RData containing `binaryResponse`, `ANONPT_ID`, `os`, `censOS` — verified by decompression): countDataSet with **raw RNA-seq counts for all genes** + per-sample annotations: **Best Confirmed Overall Response (RECIST v1.1, independent radiology review: PD/SD/PR/CR)**, `binaryResponse` (PD/SD vs CR/PR), PD-L1 IC/TC levels, `Baseline ECOG Score`, Sex, Race, tobacco history, mets status (LN only/visceral/liver), Lund + TCGA subtypes, **immune phenotype (inflamed/excluded/desert)**, OS + censoring, FMOne TMB.
  - `data/dat19.RData`, `dat25.RData`, `dat57.RData`: **FMOne targeted-panel mutation calls** (channels: amplifications, deletions, gains, known_short, likely_short) + response + IC/TC + TMB per sample.
  - `data/fmone.RData`: FMOne mutation burden.
- **N**: 348 trial participants (package contains "the majority of participants"); RNA-seq cohort ≈298; mutation cohort ≈310 (paper numbers).
- **Response labels**: **4-class RECIST v1.1 (CR/PR/SD/PD) + binary** per sample, plus OS. Best available response label granularity of any non-NSCLC ICI cohort found.
- **Download**: `git clone https://github.com/BioInfoCloud/IMvigor210CoreBiologies` or raw-file links (`https://raw.githubusercontent.com/BioInfoCloud/IMvigor210CoreBiologies/main/data/cds.RData`, ~16 MB).
- **Fit tier: A** — clinical view (ECOG, PD-L1 IC/TC, mets, sex, race, TMB) + genomics (RNA-seq + targeted mutations) = 2–3 modalities with crisp response labels. No imaging (would need external CT, none public).
- **Blockers**: (1) RData format — requires R or pyreadr to load; (2) license PDF is Genentech terms, not OSI open source (check with lead before any public redistribution); (3) raw reads controlled (processed package is public); (4) no imaging/pathology view.
- **Verification**: package DESCRIPTION + `data.R` docs fetched from GitHub; `cds.RData` downloaded (15.8 MB) and byte-verified (gzip → RDX3 RData containing `binaryResponse`/`ANONPT_ID`/`os`/`censOS`).

### 6. IMvigor210 — cBioPortal `blca_iatlas_imvigor210_2017` (iAtlas harmonized) — **NSCLC: No (urothelial/bladder)** ★ lead-named target
- **Description**: Same IMvigor210 trial, reprocessed and harmonized by iAtlas (WES + RNA-seq). Study record: "Metastatic Bladder Urothelial Carcinoma (IMvigor210 Phase II Trial, ESMO Open. 2024) - iAtlas Harmonized".
- **Verified URL**: https://www.cbioportal.org/study/summary?id=blca_iatlas_imvigor210_2017 (record + attributes + clinical data + profiles fetched this session; evidence in `raw/cbio_iatlas_ici_studies.json`).
- **License/access**: public portal/API (iAtlas harmonization).
- **Modalities**: clinical (SEX, ICI_RX=atezolizumab ×347, ICI_TARGET=PD-L1, CLINICAL_STAGE, SAMPLE_TREATMENT, METASTASIZED, TMB_NONSYNONYMOUS, neoantigen counts SNV/INDEL/FUSION/ERV/viral, OS_MONTHS/STATUS) + molecular profiles: **mutations (MAF)**, **RNA-seq expression (continuous)**, gene signatures.
- **N**: 347 samples (298 with response).
- **Response labels**: **RESPONDER (mRECIST CR/PR): TRUE 68 / FALSE 230**; CLINICAL_BENEFIT (298); PROGRESSION (298); OS. (iAtlas response is mRECIST-based binary; the 4-class RECIST lives in entry 5's package.)
- **Download**: `curl "https://www.cbioportal.org/api/studies/blca_iatlas_imvigor210_2017/clinical-data?clinicalDataType=PATIENT&projection=DETAILED&pageSize=10000"`; mutations via `/molecular-data`; RNA-seq via `/molecular-data/fetch` with `rcc_iatlas_imvigor210_2017_rna_seq_mrna`.
- **Fit tier: A** — clinical + WES + RNA-seq, binary response; complements entry 5 (iAtlas WES/RNA-seq + harmonized labels vs package's FMOne panel + 4-class RECIST).
- **Blockers**: same cohort as #5 (not independent); binary response only in portal; no imaging.
- **Verification**: study record, attr list, RESPONDER/ICI_RX distributions, molecular profiles — all fetched this session.

### 7. IMmotion150 — cBioPortal `rcc_iatlas_immotion150_2018` — **NSCLC: No (RCC)** ★ lead-named target
- **Description**: Randomized phase 2, atezolizumab (anti-PD-L1) ± bevacizumab vs sunitinib, 263 treatment-naïve metastatic RCC (McDermott et al. Nat Med 2018), iAtlas-harmonized (WES + RNA-seq).
- **Verified URL**: https://www.cbioportal.org/study/summary?id=rcc_iatlas_immotion150_2018 (fetched this session).
- **License/access**: public portal/API.
- **Modalities**: clinical (ICI_RX: atezolizumab 174 / None 89 [sunitinib arm], ICI_TARGET PD-L1, RESPONDER/CB/PROGRESSION, PFS_MONTHS/STATUS, OS) + **mutations (MAF) + RNA-seq + gene signatures**.
- **N**: 263 (247 with response).
- **Response labels**: RESPONDER **TRUE 72 / FALSE 175**; CLINICAL_BENEFIT; PROGRESSION; PFS; OS.
- **Download**: portal API (pattern as #6).
- **Fit tier: A** — clinical + WES + RNA-seq with binary response; largest public RCC ICI cohort with response.
- **Blockers**: note 89 patients are the sunitinib (non-ICI) control arm — filter to atezolizumab arm for ICI response modeling; binary response only; no imaging.
- **Verification**: study record, attributes, RESPONDER/ICI_RX distributions, profiles — fetched this session.

### 8. Gide et al. 2019 — cBioPortal `mel_iatlas_gide_2019` — **NSCLC: No (melanoma)** ★ lead-named target
- **Description**: Australian melanoma cohort (Cancer Cell 2019, PMID 30753825): anti-PD-1 monotherapy and anti-PD-1+anti-CTLA-4 combination, iAtlas-harmonized (RNA-seq).
- **Verified URL**: https://www.cbioportal.org/study/summary?id=mel_iatlas_gide_2019 (fetched this session).
- **License/access**: public portal/API.
- **Modalities**: clinical (SEX, ICI_RX: pembrolizumab 32 / nivolumab 9 / ipi+pembro 26 / ipi+nivo 8; ICI_TARGET: PD1 41 / CTLA4+PD1 34; OS) + **RNA-seq expression + gene signatures** (no mutation profile in portal).
- **N**: 91 (75 with response).
- **Response labels**: RESPONDER **TRUE 40 / FALSE 35**; CLINICAL_BENEFIT; PROGRESSION; OS.
- **Download**: portal API (pattern as #6). Raw FASTQ: ENA PRJEB23709 (see entry 26).
- **Fit tier: A** — clinical + RNA-seq (2 modalities), binary response; moderate N.
- **Blockers**: no mutations in portal (WES not released publicly — ENA is RNA-seq only); response labels binary; no imaging.
- **Verification**: study record, attributes, RESPONDER/ICI_RX/OS distributions, profiles — fetched this session.

---

## Tier B

### 9. cBioPortal `tmb_mskcc_2018` — Samstein et al. Nat Genet 2019 — **NSCLC: Yes (subset)** (unchanged from previous session)
- 1,661 ICI-treated across 10+ cancer types; NSCLC ≈315 samples; TMB + OS + drugs; **no RECIST response in portal release**. Fit tier B. URL https://www.cbioportal.org/study/summary?id=tmb_mskcc_2018 (PMID 30643254).

### 10. GEO GSE126044 — anti-PD-1 NSCLC RNA-seq (n=16) — **NSCLC: Yes** (unchanged)
- 16 pre-treatment NSCLC (nivolumab), responder/non-responder **in sample characteristics** (verified). URL https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126044. Fit B; blockers: N=16, no clinical table, no imaging.

### 11. GEO GSE135222 — anti-PD-1/PD-L1 NSCLC RNA-seq (n=27) — **NSCLC: Yes** (unchanged)
- RNA-seq for 27 advanced NSCLC (anti-PD-1/PD-L1); response in paper only (not GEO metadata). Fit B.

### 12. Zenodo 5162861 — Granata et al. 2021 CT radiomics — **NSCLC: Yes** (unchanged)
- 88 LUAD ICI patients, IBSI-style radiomics xlsx (downloaded/inspected); **OS/PFS only, no RECIST**. CC-BY-4.0. Fit B.

### 13. TCIA Anti-PD-1_Lung — **NSCLC: Yes** (unchanged)
- 46 NSCLC cases, CT + PET DICOM, **no response labels** (verified via Wayback). Fit B (imaging only).

### 14. Hugo et al. 2016 — GEO GSE78220 + cBioPortal `mel_ucla_2016` — **NSCLC: No (melanoma)** ★ lead-named target
- **Description**: Pre-treatment melanoma biopsies from 27–38 patients on anti-PD-1 (pembrolizumab) — the original "transcriptome + exome" response cohort (Cell 2016). Two public access routes.
- **Verified URLs** (both fetched this session):
  - GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78220 — "mRNA expressions in pre-treatment melanomas undergoing anti-PD-1 checkpoint inhibition therapy", 28 samples, RNA-seq; `raw/geo_gse78220_characteristics.txt`.
  - cBioPortal: https://www.cbioportal.org/study/summary?id=mel_ucla_2016 — 38 samples, molecular profiles: mutations + CNA + RNA-seq.
- **License/access**: GEO public; cBioPortal public.
- **Modalities**: RNA-seq (GEO) + WES/CNA/RNA-seq (portal); clinical in GEO characteristics: age, sex, M stage, OS, vital status, treatment, biopsy time, mutation status (NRAS etc.).
- **N**: 28 (GEO) / 38 (portal).
- **Response labels**: GEO `!Sample_characteristics_ch1 = anti-pd-1 response: Progressive Disease / Partial Response` (verified in matrix); portal DURABLE_CLINICAL_BENEFIT: **CR 7 / PR 14 / PD 17** (n=38); TREATMENT_RESPONSE attr.
- **Download**: GEO: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/GSE78220/` (matrix + supplement); cBioPortal: portal API (pattern as #2).
- **Fit tier: B** (clinical + RNA-seq + WES = 3 modalities, but small N).
- **Blockers**: small N; partial overlap between GEO and portal sample sets (28 vs 38); melanoma biology differs from NSCLC (schema transfer needed).
- **Verification**: GEO esummary + full characteristics file; cBioPortal study + DCB distribution — this session.

### 15. Liu et al. 2019 — cBioPortal `mel_iatlas_liu_2019` — **NSCLC: No (melanoma)**
- **Description**: 122 metastatic melanoma patients on anti-PD-1 (pembrolizumab 71 / nivolumab 51), iAtlas harmonized (WES + RNA-seq). (The companion `mel_dfci_2019` study has no response attrs in the portal — use the iAtlas version.)
- **Verified URL**: https://www.cbioportal.org/study/summary?id=mel_iatlas_liu_2019 (fetched this session).
- **License/access**: public portal/API. **Modalities**: clinical (ICI_RX, ICI_TARGET=PD1) + mutations (MAF) + RNA-seq + gene signatures.
- **N**: 122. **Response labels**: RESPONDER **TRUE 48 / FALSE 74**; CLINICAL_BENEFIT; PROGRESSION; OS.
- **Download**: portal API (pattern as #6). **Fit tier: B** (3 modalities, moderate N; no imaging).
- **Blockers**: binary response; WES/RNA-seq not raw. **Verification**: fetched this session.

### 16. Riaz et al. 2017 — cBioPortal `mel_iatlas_riaz_nivolumab_2017` — **NSCLC: No (melanoma)**
- **Description**: 107 melanoma patients on nivolumab (MSK; Cell 2017), pre- and on-treatment biopsies, iAtlas harmonized. (Previously listed as Tier C; upgraded to B — 3 modalities + response.)
- **Verified URL**: https://www.cbioportal.org/study/summary?id=mel_iatlas_riaz_nivolumab_2017 (fetched this session).
- **License/access**: public portal/API. **Modalities**: clinical + mutations (MAF) + RNA-seq + gene signatures.
- **N**: 107 (64 with ICI_RX; 56 with response). **Response labels**: RESPONDER **TRUE 11 / FALSE 45**; CLINICAL_BENEFIT; PROGRESSION.
- **Download**: portal API (pattern as #6). **Fit tier: B**; **Blockers**: response only for subset; binary.
- **Verification**: profiles + RESPONDER distribution fetched this session.

### 17. Miao et al. 2018 — cBioPortal `mixed_allen_2018` (MSS pan-cancer) — **NSCLC: No (mostly; mixed solid tumors)** ★ covers the head & neck + CRC-MSS targets
- **Description**: "Genomic correlates of response to immune checkpoint blockade in microsatellite-stable solid tumors" (Nat Genet 2018): 249 MSS patients across melanoma (~135), NSCLC (~49), HNSCC (≈12: tonsil 6, oropharynx 2, larynx 1, base of tongue 1, oral tongue 1, nasopharynx 1), RCC, bladder, CRC-MSS, uveal, leiomyosarcoma.
- **Verified URL**: https://www.cbioportal.org/study/summary?id=mixed_allen_2018 (fetched this session).
- **License/access**: public portal/API.
- **Modalities**: clinical (AGE_START_IO, SEX, DRUG_TYPE, OTHER_CONCURRENT_THERAPY, SMOKER, TMB_NONSYNONYMOUS, MUTATION_COUNT, FRACTION_GENOME_ALTERED, PFS/OS) + **WES mutations** (only profile).
- **N**: 249. **Response labels**: **RECIST_RESPONSE: clinical benefit 70 / stable disease 56 / no clinical benefit 123**; raw RECIST; ROH_RESPONSE, VA_RESPONSE; PFS; OS.
- **Download**: portal API (pattern as #6).
- **Fit tier: B** — the only public WES+RECIST ICI cohort containing an HNSCC subset; also covers MSS-CRC. **Blockers**: no RNA-seq/imaging; HNSCC subset small (≈12); MSS only (not MSI-H).
- **Verification**: study record, histology counts (incl. HNSCC sites), RECIST_RESPONSE distribution — fetched this session.

### 18. PRINCE trial — cBioPortal `paad_iatlas_prince_2022` — **NSCLC: No (pancreatic)**
- **Description**: Metastatic pancreatic adenocarcinoma randomized trial with nivolumab-containing arms (Nat Med 2022), iAtlas harmonized (WES + RNA-seq).
- **Verified URL**: https://www.cbioportal.org/study/summary?id=paad_iatlas_prince_2022 (fetched this session).
- **License/access**: public portal/API. **Modalities**: clinical (ICI_RX: nivolumab 50) + mutations + RNA-seq + gene signatures.
- **N**: 93. **Response labels**: RESPONDER **TRUE 31 / FALSE 35**; CLINICAL_BENEFIT; PROGRESSION.
- **Download**: portal API. **Fit tier: B** (3 modalities, moderate N, chemo-backbone confound).
- **Blockers**: pancreatic ICI response is weak (low ORR) — hard label distribution; binary labels. **Verification**: fetched this session.

---

## Tier C

### 19. TCIA classic NSCLC radiomics cohorts — **NSCLC: Yes** (unchanged)
- NSCLC-Radiomics (Aerts 2014, ~422), NSCLC-Radiogenomics (Bakr 2018, ~211, +RNA-seq), NSCLC-Radiomics-Interobserver1, NSCLC-RADIOMICS-GENOMICS (89), Lung-PET-CT-Dx, QIN-Lung-CT, 4D-Lung. None ICI-treated; no RECIST-ICI labels. Fit C (pretraining only).

### 20. Jerby-Arnon et al. 2018 — GEO GSE115821 — **NSCLC: No (melanoma)**
- **Description**: "A Cancer Cell Program Promotes T Cell Exclusion and Resistance to PD-1 Blockade" (Cell 2018); melanoma biopsies pre-ICI: anti-PD-1 27, anti-CTLA-4 6, combination 4 (of 37 samples).
- **Verified URL**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115821 (characteristics in `raw/geo_gse115821_characteristics.txt`).
- **License/access**: GEO public. **Modalities**: RNA-seq (37 samples) + clinical in characteristics (age, batch, timepoint, tumor type).
- **N**: 37 (34 with response). **Response labels**: `response: R / NR` in characteristics (**R 3 / NR 34** — heavily imbalanced in this series; paper's full cohort n≈53).
- **Download**: GEO FTP. **Fit tier: C** — imbalanced labels in GEO; paper supplement has the balanced cohort.
- **Blockers**: label imbalance as deposited; no WES; small. **Verification**: esummary + characteristics fetched this session.

### 21. Anders et al. 2022 — cBioPortal `brca_iatlas_anders_2022` — **NSCLC: No (TNBC)**
- UNC phase II, metastatic TNBC on pembrolizumab (J Immunother Cancer 2022), iAtlas harmonized. N=31; RESPONDER TRUE 7 / FALSE 24; mutations + RNA-seq. URL https://www.cbioportal.org/study/summary?id=brca_iatlas_anders_2022. Fit C (small N). Verified this session.

### 22. Cloughesy/Prins et al. 2019 — cBioPortal `gbm_iatlas_prins_2019` — **NSCLC: No (glioblastoma)**
- Neoadjuvant pembrolizumab GBM RCT (Nat Med 2019), iAtlas harmonized. N=30; RNA-seq only; RESPONDER all FALSE among 28 labeled (RANO-based — GBM has no RECIST); response semantics differ. URL https://www.cbioportal.org/study/summary?id=gbm_iatlas_prins_2019. Fit C. Verified this session.

### 23. Choueiri et al. 2016 — cBioPortal `ccrcc_iatlas_choueiri_2016` — **NSCLC: No (ccRCC)**
- Nivolumab phase-1 biomarker cohort, ccRCC (Clin Cancer Res 2016), iAtlas harmonized. N=16; RESPONDER TRUE 3 / FALSE 13; mutations + RNA-seq. URL https://www.cbioportal.org/study/summary?id=ccrcc_iatlas_choueiri_2016. Fit C (tiny N). Verified this session.

### 24. GEO GSE332566 — HNSCC on PD-1/PD-L1 — **NSCLC: No (head & neck)** ★ HNSCC target
- **Description**: "predictive value of genomic, transcriptomic and epigenetic biomarkers in patients with recurrent and/or metastatic HNSCC treated with PD-1/PD-L1 inhibitors" (n=26).
- **Verified URL**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE332566 (characteristics in `raw/geo_gse332566_characteristics.txt`).
- **License/access**: GEO public. **Modalities**: methylation arrays (GEO series); paper additionally reports genomic/transcriptomic biomarkers (not in GEO).
- **N**: 26 (all tissue: HNSC). **Response labels**: **not in GEO characteristics** (paper-defined; response in paper only).
- **Download**: GEO FTP. **Fit tier: C** — methylation-only as deposited; labels not machine-readable.
- **Blockers**: 1 modality in GEO; labels in paper; small N. **Verification**: series summary + characteristics fetched this session.

### 25. GEO GSE205506 — dMMR/MSI-H colorectal on PD-1 blockade — **NSCLC: No (MSI-H CRC)** ★ CRC MSI-H target
- **Description**: "Remodeling of the Immune and Stromal Cell Compartment by PD-1 Blockade in Mismatch Repair-Deficient Colorectal Cancer": **single-cell** RNA-seq of 19 dMMR/MSI-H CRC patients, pre- and on-treatment, response (incl. pCR) per paper.
- **Verified URL**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205506 (esummary verified this session).
- **License/access**: GEO public. **Modalities**: scRNA-seq (needs pseudobulk aggregation for the project's bulk schema).
- **N**: 19 patients (40 samples). **Response labels**: in paper (pCR/response); not in GEO characteristics.
- **Download**: GEO FTP. **Fit tier: C** — single-cell, no clinical table, labels in paper.
- **Blockers**: modality mismatch (scRNA vs bulk/clinical); MSI-H ICI bulk cohorts with response are not public (see UNAVAILABLE). **Verification**: esummary this session.

### 26. ENA PRJEB23709 — Gide et al. 2019 raw RNA-seq — **NSCLC: No (melanoma)**
- **Description**: Raw FASTQ for the Gide 2019 cohort (see entry 8): 91 RNA-seq runs, sample titles `PD1_*_PRE` (41), `ipiPD1_*_PRE` (32), `ipiPD1_*_EDT` (early on-treatment, 10) — verified via ENA API (`raw/ena_prjeb23709_runs.tsv`).
- **Verified URL**: https://www.ebi.ac.uk/ena/browser/view/PRJEB23709 (study record via ENA portal API search).
- **License/access**: ENA public (raw reads). **Modalities**: raw RNA-seq FASTQ only.
- **N**: 91 runs. **Response labels**: in the paper's supplement, not in ENA metadata.
- **Download**: ENA (FASTQ via https://www.ebi.ac.uk/ena/portal/api/fetch?result=read_run&query=study_accession="PRJEB23709").
- **Fit tier: C** (raw RNA only; use the iAtlas harmonized version in entry 8 instead).
- **Blockers**: mapping runs to response requires paper Table S1; heavy downloads. **Verification**: ENA API search (91 runs, titles + fastq sizes) this session.

---

## UNAVAILABLE / NOT FOUND

| Item | Reason |
|---|---|
| Rizvi et al. 2018 Nat Med WES cohort (n=240, pembrolizumab, NSCLC) | dbGaP controlled access; accession unverified (eutils `db=gap` unsupported). |
| GSE176307 (urothelial) | GEO states raw data unavailable (privacy). |
| Zenodo 17723682, 14936801 (radiomics PDFs) | Records contain only paper PDFs — no data files. |
| Zenodo 8345959 (BAMF segmentations) | Broken/repurposed record; unverified. |
| HuggingFace Hub (nsclc/immunotherapy/radiomics) | No ICI-response dataset found; only TCIA mirrors. |
| TCIA REST API + website (direct) | Connection timeouts; verified via Wayback captures instead. |
| Synapse API/website (direct) | NIH geo-restriction (NOT-OD-25-083). |
| "Any public TCIA collection with NSCLC+ICI response labels" | Only Anti-PD-1_Lung exists and it is imaging-only. |
| Trebeschi et al. 2019 radiomics-ICI data (NSCLC n=52 + urothelial n=104) | No public release found. |
| Bioconductor `IMvigor210CoreBiologies` (all releases) | **Removed** from Bioconductor 3.17–3.21 (404 verified); original host research-pub.gene.com unreachable. Use GitHub mirrors (entry 5). |
| **MSI-H CRC ICI bulk cohort with response (e.g., CheckMate 142)** | No public bulk RNA-seq/WES release with response found. Closest: GSE205506 (scRNA, entry 25) and Miao MSS CRC subset (entry 17). |
| `pancan_mimsi_msk_2024` (MiMSI pan-cancer MSI, n=5,033) | MSK-IMPACT MSI cohort — **no ICI treatment or response data** in portal (only ETHNICITY attr). Not usable for response modeling. |
| `ucec_ccr_msk_2022` (endometrial MSI) | MSI endometrial cohort — no treatment/response attrs in portal. |
| Braun et al. 2020 (nivolumab ccRCC, n=64 WES) | Data on EGA/controlled access; no public GEO/cBioPortal release found. |
| Dedicated HNSCC ICI cohort with clinical + imaging | Not found. Best HNSCC options: Miao HNSCC subset (entry 17, WES+RECIST) and GSE332566 (entry 24, methylation, labels in paper). |

(End of file)
