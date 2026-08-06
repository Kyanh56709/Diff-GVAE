# Plan — Tìm dataset NSCLC đa phương thức công khai cho Diff-GVAE

## Câu hỏi
"Tìm các bộ dataset (NSCLC thì tốt) multimodal tương tự project mà Diff-GVAE có thể áp dụng đơn giản — với dữ liệu canonical data_ln_pc_ihc_g.pt, 1 = non-responder, 0 = responder."

## Bối cảnh (input requirement của project)
- Nguồn cohort: Vanguri et al. 2022 (Nat Cancer, DOI 10.1038/s43018-022-00416-8) — 366 NSCLC PD-(L)1; sub-cohort TMB N=247 = data project.
- 3 views: clinical 22-dim (5 cont: albumin/dnlr/TMB/tumor_burden/PDL1 + 17 binary: driver genes/IO drugs/ECOG), pathology 15-dim GLCM (từ PD-L1 IHC slides), radiology 34-dim radiomics/lesion (pyradiomics) → attention aggregation cấp patient.
- Graph: patient + lesion nodes; similarity edges per view (cosine, prune < 0.7); has_lesion edges.
- Task: binary RECIST v1.1 response.

## Sub-questions
1. Dataset NSCLC + ICI (PD-(L)1) nào công khai có: (a) response label RECIST, (b) clinical tabular (TMB/PD-L1/labs/ECOG), (c) CT imaging/segmentations hoặc radiomics sẵn có, (d) pathology (IHC/slide features)?
2. Nếu không có đủ 3 modality: dataset nào có ≥2 modality mà "simple adaptation" khả thi (tabular sẵn, không cần pipeline segmentation lại)?
3. Có dataset nào tương tự tri-modal ICI-response (kể cả solid tumor khác) phát hành 2020-2026 (TCIA, cBioPortal, GEO, Zenodo, HuggingFace)?
4. License/access path từng dataset (download command), format (CSV/DICOM/slides), size.
5. Dataset nào cho phép dựng HeteroData cùng ý tưởng (cosine similarity edges + lesion-level radiology) mà không cần code lại nhiều?

## Tiêu chí screen (fit matrix) — CẬP NHẬT 2026-08-06 (owner)
- **NSCLC + ICI response: ưu tiên cao nhất (điểm cộng lớn), nhưng KHÔNG bắt buộc.**
- Chấp nhận MỌI solid tumor + ICI (anti-PD-(L)1/anti-CTLA-4) có response label (RECIST/irRECIST/iRECIST, CR/PR/SD/PD hoặc binary) + ≥2 modality (clinical tabular / imaging hoặc radiomics sẵn / pathology / genomics).
- **LOẠI (owner 2026-08-06):** cBioPortal `lung_msk_mind_2020` và Synapse `syn26642505` — đều là chính cohort nguồn của project (Vanguri 2022 / MSK-MIND, N=247) → chỉ giữ làm reference, không phải ứng viên external.
- Ví dụ cần tìm thêm: urothelial/bladder (IMvigor210 atezolizumab — clinical + RNA-seq + response), melanoma (Gide 2019, Hugo 2016), RCC, head & neck, colorectal MSI-H ICI.
- Ghi rõ trong datasets.md: NSCLC vs non-NSCLC + cờ EXCLUDED.
- Các tiêu chí còn lại giữ nguyên (response labels thật, public access + license rõ, N ≥ ~50, effort "simple": tabular/radiomics sẵn > raw DICOM).

## Agent
- Wave 1: `lit-researcher` — tìm + ghi nguồn (OpenAlex/arXiv/HF/Zenodo/TCIA/cBioPortal/GEO), verify URL.
- Screen: tôi (lead) chọn ≤5 candidate cho bảng fit.
- Wave 3: `science-reviewer` — bắt buộc, verify các claim dataset trong report.

## Output
- `sources/datasets.md` (lit-researcher), `reports/report.md` (fit matrix + khuyến nghị), `evidence/provenance.md`.
