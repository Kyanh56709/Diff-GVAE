# Papers — NSCLC/ICI + non-NSCLC ICI datasets and data-availability notes

All DOIs/PMIDs below were resolved or verified during session 2026-08-06 unless noted. Raw evidence: `sources/raw/`. Criteria update (lead, 2026-08-06): datasets no longer must be NSCLC — any solid tumor on anti-PD-(L)1/anti-CTLA-4 with response labels + ≥2 modalities qualifies; NSCLC remains highest priority. Tumor type is marked per row (**NSCLC** vs **non-NSCLC**).

## Primary source cohort

| Paper | DOI / PMID | Data availability (verified) |
|---|---|---|
| Vanguri R, et al. Multimodal integration of radiology, pathology and genomics for prediction of response to PD-(L)1 blockade in patients with non-small cell lung cancer. **Nature Cancer** 2022;3:1151–1164 (**NSCLC**) | 10.1038/s43018-022-00416-8 / PMC9586871, PMID 36038778 | "All data are publicly available at synapse: https://www.synapse.org/#!Synapse:syn26642505. The genomic features of the cohort can be explored on cbioportal: https://www.cbioportal.org/study/summary?id=lung_msk_mind_2020. Source data are provided with this paper." (verified verbatim from PMC full text this session). Cohort: n=247 advanced NSCLC, PD-(L)1 blockade 2014–2019, best response RECIST v1.1 binarized CR/PR (62, 25%) vs SD/PD (185). Sub-cohorts: radiology n=50, pathology n=52. |

## I3LUNG

| Paper | DOI / PMID | Data availability (verified) |
|---|---|---|
| The EU-funded I3LUNG Project: Integrative Science, Intelligent Data Platform for Individualized LUNG Cancer Care With Immunotherapy. Clin Lung Cancer 2023 | 10.1016/j.cllc.2023.02.005 / PMID 36959048 | Protocol; trial NCT05537922 (verified PMID→title/DOI mapping via PubMed esummary). |
| I3LUNG: Clinical Validation of a Multimodal AI Tool to Support Immunotherapy Decisions in NSCLC. medRxiv preprint 2026 (DOI = Zenodo record 17535424's own DOI, per zenodo_rec_17535424.json) | 10.64898/2026.01.16.25342913 | Abstract (from record): 2,365 patients enrolled, 6 centers, RWD clinical + CT + digital pathology + genomics, MLEF/DLIF fusion, AUC≈0.74 (test) / 0.82 first-line. NOTE (review 2026-08-06): the "HTTP 200" claim had no saved fetch artifact — reclassified as unverified; the DOI resolves to the Zenodo record itself. |
| Dataset release: **I3LUNG DATASETS**, Zenodo record 17535424 | https://zenodo.org/records/17535424 | I3LUNG_DATA.zip (26 MB) + results.zip (2.08 GB), CC-BY-NC-4.0. Contents downloaded and inspected this session (see datasets.md #1). |

## cBioPortal-hosted ICI cohorts

| Paper | DOI / PMID | Notes (verified) |
|---|---|---|
| Samstein RM, et al. Tumor mutational load predicts survival after immunotherapy across multiple cancer types. Nat Genet 2019 (**pan-cancer incl. NSCLC**) | PMID 30643254 (portal citation); DOI 10.1038/s41588-018-0314-8 **failed to resolve this session — use PMID** | cBioPortal `tmb_mskcc_2018`: 1,661 patients; NSCLC ≈315 samples; TMB + OS + drug type; **no RECIST response in portal release**. |
| Hellmann MD, et al. Genomic Features of Response to Combination Immunotherapy in Patients with Advanced Non-Small-Cell Lung Cancer. Cancer Cell 2018 (**NSCLC**) | 10.1016/j.ccell.2018.03.018 (verified via OpenAlex) / PMID 29657128 | cBioPortal `nsclc_mskcc_2018`: WES of 75 tumor/normal pairs, PD-1+CTLA-4 (CheckMate 012); BEST_OVERALL_RESPONSE, DCB, PDL1_EXP, TMB, neoantigen, HLA. |
| Liu D, et al. Integrative molecular and clinical modeling of clinical outcomes to PD1 blockade in patients with metastatic melanoma. Nat Med 2019 (**non-NSCLC: melanoma**) | PMID 30833747 (portal citation `mel_dfci_2019`) | 144 melanoma, WES + response. iAtlas version `mel_iatlas_liu_2019` (n=122) has RESPONDER 48/74 + RNA-seq (verified this session). |
| Riaz N, et al. Tumor and Microenvironment Evolution during Immunotherapy with Nivolumab. Cell 2017 (**non-NSCLC: melanoma**) | portal `mel_iatlas_riaz_nivolumab_2017` | 107 melanoma, WES + RNA-seq + response (iAtlas; RESPONDER 11/45, verified this session). |
| Hugo W, et al. Genomic and Transcriptomic Features of Response to Anti-PD-1 Therapy in Metastatic Melanoma. Cell 2016 (**non-NSCLC: melanoma**) | portal `mel_ucla_2016`; GEO GSE78220 | 27–38 melanoma. cBioPortal: WES+CNA+RNA-seq, DCB CR 7/PR 14/PD 17. GEO GSE78220 (28 RNA-seq samples): `anti-pd-1 response` in characteristics (both verified this session). |
| Mariathasan S, et al. TGFβ attenuates tumour response to PD-L1 blockade (IMvigor210). Nature 2018 (**non-NSCLC: urothelial**) | 10.1038/nature25501 / PMID 29670289 | **Bioconductor `IMvigor210CoreBiologies` removed from all releases** (3.17–3.21 404 this session); original host research-pub.gene.com unreachable. Full package (counts + RECIST v1.1 4-class + binary + FMOne mutations + TMB + ECOG) via GitHub mirrors: BioInfoCloud/IMvigor210CoreBiologies (main), SiYangming/IMvigor210CoreBiologies (master) — DESCRIPTION v2.0.0 + data files verified this session. Raw reads EGA EGAS00001002556 (controlled). |
| McDermott DF, et al. (IMmotion150). Nat Med 2018 (**non-NSCLC: RCC**) | portal `rcc_iatlas_immotion150_2018` | 263 RCC, atezolizumab±bevacizumab vs sunitinib. iAtlas: mutations + RNA-seq; RESPONDER 72/175; 174 atezolizumab (verified this session). |
| Miao D, et al. Genomic correlates of response to immune checkpoint blockade in microsatellite-stable solid tumors. Nat Genet 2018 (**non-NSCLC: MSS pan-cancer incl. HNSCC, CRC-MSS, NSCLC, melanoma, RCC, bladder**) | portal `mixed_allen_2018` | 249 MSS patients, WES; RECIST_RESPONSE: clinical benefit 70 / SD 56 / no clinical benefit 123; PFS/OS (verified this session). Contains the only public HNSCC ICI subset with RECIST (≈12: tonsil 6, oropharynx 2, larynx 1, base of tongue 1, oral tongue 1, nasopharynx 1). |
| Gide TN, et al. Distinct immune cell populations define response to anti-PD-1 monotherapy and anti-PD-1+anti-CTLA-4 combination therapy. Cancer Cell 2019 (**non-NSCLC: melanoma**) | 10.1016/j.ccell.2019.01.003 / PMID 30753825 | RNA-seq deposited at **ENA PRJEB23709** (91 runs, verified this session — data-availability statement from paper text). iAtlas `mel_iatlas_gide_2019` (n=91; RESPONDER 40/35; pembro 32 / nivo 9 / ipi+pembro 26 / ipi+nivo 8). |
| Jerby-Arnon L, et al. A Cancer Cell Program Promotes T Cell Exclusion and Resistance to PD-1 Blockade. Cell 2018 (**non-NSCLC: melanoma**) | 10.1016/j.cell.2018.09.006 | GEO **GSE115821**: 37 RNA-seq samples (anti-PD-1 27 / anti-CTLA-4 6 / combo 4); `response: R/NR` in characteristics (verified this session). |
| O'Reilly EM, et al. Randomized phase 2 PRINCE trial. Nat Med 2022 (**non-NSCLC: pancreatic**) | portal `paad_iatlas_prince_2022` | 93; nivolumab arms; iAtlas mutations + RNA-seq; RESPONDER 31/35 (verified this session). |
| Anders CK, et al. UNC phase II, metastatic TNBC + pembrolizumab. J Immunother Cancer 2022 (**non-NSCLC: TNBC**) | portal `brca_iatlas_anders_2022` | 31; iAtlas mutations + RNA-seq; RESPONDER 7/24 (verified this session). |
| Cloughesy TF, et al. Neoadjuvant anti-PD-1 immunotherapy (GBM). Nat Med 2019 (**non-NSCLC: glioblastoma**) | portal `gbm_iatlas_prins_2019` | 30; iAtlas RNA-seq; RESPONDER labels present (28 FALSE; RANO-based semantics) (verified this session). |
| Choueiri TK, et al. Immunomodulatory activity of nivolumab in metastatic RCC. Clin Cancer Res 2016 (**non-NSCLC: ccRCC**) | portal `ccrcc_iatlas_choueiri_2016` | 16; iAtlas mutations + RNA-seq; RESPONDER 3/13 (verified this session). |

## GEO/ENA-hosted ICI expression cohorts (NSCLC + non-NSCLC)

| Series | Paper / DOI / PMID | Response labels | Verified |
|---|---|---|---|
| **GSE126044** (n=16 RNA-seq, nivolumab NSCLC, pre-treatment) | Cho JW, et al. Exp Mol Med 2020. 10.1038/s12276-020-00493-8 / PMID 32879421 (**NSCLC**) | Yes — responder/non-responder in `!Sample_characteristics_ch1` | matrix + counts file (559,926 B) HEAD/downloaded |
| **GSE135222** (n=27 RNA-seq, anti-PD-1/PD-L1 NSCLC) | PMID 31537801; 32762727 (**NSCLC**) | In paper only (not in GEO metadata) | matrix + suppl (1,663,896 B) |
| **GSE176307** | Urothelial, NOT NSCLC; "raw data unavailable due to patient privacy concerns" (GEO statement) | — | verified — exclude |
| **GSE78220** (n=28 RNA-seq, pre-treatment melanoma, anti-PD-1) | Hugo W, et al. Cell 2016. 10.1016/j.cell.2016.02.065 (**non-NSCLC: melanoma**) | Yes — `anti-pd-1 response: Progressive Disease / Partial Response` in characteristics (+ treatment, OS, mutations) | esummary + full characteristics file downloaded |
| **GSE115821** (n=37 RNA-seq, melanoma pre-ICI: anti-PD-1 27 / anti-CTLA-4 6 / combo 4) | Jerby-Arnon L, et al. Cell 2018. 10.1016/j.cell.2018.09.006 (**non-NSCLC: melanoma**) | Yes — `response: R/NR` in characteristics (R 3 / NR 34 as deposited) | esummary + full characteristics file downloaded |
| **GSE332566** (n=26 methylation, recurrent/metastatic HNSCC on PD-1/PD-L1) | paper-defined (**non-NSCLC: HNSCC**) | In paper only (not in GEO characteristics) | series summary + characteristics downloaded |
| **GSE205506** (n=19 pts, scRNA-seq, dMMR/MSI-H CRC on PD-1 blockade) | paper-defined (**non-NSCLC: MSI-H CRC**) | In paper only (pCR/response); not in GEO characteristics | esummary verified |
| GSE206127 (n=213, circulating microRNA, nivolumab NSCLC exceptional responders) | — (**NSCLC**) | paper-defined | esummary only |
| GSE225620 (n=43, blood-based genomics, neoadjuvant PD-1 NSCLC) | — (**NSCLC**) | paper-defined | esummary only |
| **ENA PRJEB23709** (91 RNA-seq runs, Gide 2019 melanoma: PD1 pre 41, ipiPD1 pre 32 + on-treatment 10) | Gide TN, et al. Cancer Cell 2019. 10.1016/j.ccell.2019.01.003 (**non-NSCLC: melanoma**) | In paper supplement only (not in ENA metadata) | ENA API: study + 91 runs (titles, fastq sizes) |

## ICI radiomics papers — data-release status

| Paper | DOI | Data release (verified) |
|---|---|---|
| Mu Y, et al. Predicting benefit from immune checkpoint inhibitors in patients with non-small-cell lung cancer by CT-based ensemble deep learning model. Lancet Digit Health 2023 | 10.1016/s2589-7500(23)00082-1 (OpenAlex) | No public data release found. |
| Real-World and Clinical Trial Validation of a Deep Learning Radiomic Biomarker for PD-(L)1 ICI. JCO Clin Cancer Inform 2024 | 10.1200/cci.24.00133 (OpenAlex) | No public data release found. |
| Radiomics and Delta-Radiomics Signatures to Predict Response and Survival in NSCLC. Cancers 2023 | 10.3390/cancers15071968 (OpenAlex) | No public data release found. |
| Moiseenko F, et al. Baseline Radiomics as a Prognostic Tool for Clinical Benefit from ICI in Inoperable NSCLC. Cancers 2025 | 10.3390/cancers17111790 (Zenodo record 17723682) | Zenodo record contains only the PDF — **no data files** (verified). |
| Rishi Reddy Kothinti. DL-Based Radiomic Signature for Predicting Treatment Response to Immunotherapy in NSCLC. IJISRT 2025 | Zenodo record 14936801 | PDF only — **no data files** (verified). |
| Granata V, et al. Preliminary Report on CT Radiomics Features as Biomarkers to Immunotherapy Selection in Lung Adenocarcinoma Patients. 2021 | Zenodo record 5162861 (paper DOI unverified) | **Full radiomics xlsx (88 pts) + OS/PFS**, CC-BY-4.0 (downloaded & inspected). |
| Trebeschi S, et al. Predicting response to cancer immunotherapy using noninvasive radiomic biomarkers. Lancet Digit Health 2019 | 10.1016/S2589-7500(19)30136-X (DOI not re-verified) | No public data release found (NSCLC n=52, urothelial n=104). |
| Weakly Supervised Deep Learning Predicts Immunotherapy Response in Solid Tumors Based on PD-L1 Expression. Cancer Res Commun 2024 | PMC10782919 | "All data from NSCLC-MSK cohort publicly available https://www.synapse.org/#!Synapse:syn26642505" — independent confirmation that syn26642505 is usable/downloadable. |
| LORIS robustly predicts patient outcomes on immune checkpoint blockade therapy. Nat Cancer 2024 | 10.1038/s43018-024-00772-7 | Reuses Vanguri et al. cohort from Synapse syn26642505 + cBioPortal lung_msk_mind_2020. |

## Other

| Paper | DOI | Note |
|---|---|---|
| Rizvi NA, et al. Molecular determinants of response to anti-PD-1 blockade in patients with non-small-cell lung cancer. Nat Med 2018 (n=240 WES) | 10.1038/s41591-018-0134-3 (not re-verified) | WES data under dbGaP controlled access; **accession not verifiable this session** (eutils `db=gap` unsupported). Treat as restricted access. |
