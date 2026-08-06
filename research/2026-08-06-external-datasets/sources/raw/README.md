# Raw evidence files (session 2026-08-06)

## Non-NSCLC ICI update (broadened criteria, same session)

- cbio_iatlas_ici_studies.json — 11 iAtlas/cBioPortal ICI studies: sample counts, molecular profiles, RESPONDER / RECIST_RESPONSE / DCB / ICI_RX distributions (IMvigor210, IMmotion150, Gide 2019, Liu 2019, Riaz 2017, Hugo mel_ucla_2016, Miao mixed_allen_2018, PRINCE, Anders TNBC, Prins GBM, Choueiri ccRCC)
- imvigor210_description.txt / imvigor210_data_doc.txt — verbatim DESCRIPTION + R/data.R docs of Bioconductor `IMvigor210CoreBiologies` fetched from GitHub mirror (documents RECIST v1.1 4-class response, binaryResponse, ECOG, FMOne mutations, counts data)
- geo_gse78220_characteristics.txt — GSE78220 (Hugo 2016) sample characteristics incl. `anti-pd-1 response`
- geo_gse115821_characteristics.txt — GSE115821 (Jerby-Arnon 2018) characteristics incl. `response: R/NR`, antibody, treatment state
- geo_gse332566_characteristics.txt — GSE332566 (HNSCC ICI) characteristics (tissue: HNSC; no response in GEO)
- ena_prjeb23709_runs.tsv — ENA PRJEB23709 (Gide 2019): 91 RNA-seq runs with titles/fastq sizes

## NSCLC session (original scope)

- cbioportal_all_studies.json — full cBioPortal studies dump (539 studies)
- cbio_*_attrs.json — clinical attribute lists
- cbio_mind_* / cbio_lung_msk_mind_2020* — Vanguri cohort (lung_msk_mind_2020) records + clinical data (SAMPLE + PATIENT) + molecular profiles
- cbio_nsclc2018_full.json — Hellmann nsclc_mskcc_2018 record
- cbio_samstein_*.json — Samstein tmb_mskcc_2018 patient/sample clinical data
- geo_esummary_3gse.json, geo_search_nsclc_ici.json, geo_esummary_nsclc_ici_60.json — NCBI E-utilities
- GSE126044_matrix.txt.gz / GSE135222_matrix.txt.gz — downloaded series matrices (contain metadata; labels verified in GSE126044 characteristics)
- database_immuno.xlsx — Granata radiomics workbook (Zenodo 5162861)
- hf_nsclc.json / hf_immunotherapy.json / hf_radiomics.json — HuggingFace API
- zenodo_*.json — Zenodo API queries + records (17535424 I3LUNG, 5162861, 17723682, 14936801, 17829724, BAMF attempts)
- openalex_*.json — OpenAlex records (Vanguri, I3LUNG papers, ICI-radiomics 2022-2026 search)
- synapse_syn26642505*.json — Synapse API responses (geo-blocked, NOT-OD-25-083)
- tcia_wiki_collections_wayback.html — archived TCIA wiki Collections page
- antipd1_lung_wiki.html — archived Anti-PD-1_Lung wiki page
- dbgap_phs001493*.json — dbGaP esearch attempts (db name invalid; accession unverified)
- pubmed_36959048.json — I3LUNG protocol PMID
- i3lung_evidence/ — I3LUNG_DATA.zip listing, CSV heads, label counts, features.json
