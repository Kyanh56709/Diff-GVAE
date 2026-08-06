# Dataset NSCLC+ICI đa phương thức cho Diff-GVAE — Báo cáo tổng hợp (2026-08-06)

## 1. Trả lời

Sau khi loại 2 bộ dữ liệu trùng cohort nguồn của project (Vanguri 2022 / MSK-MIND — owner xác nhận 2026-08-06), **I3LUNG (Zenodo 17535424)** là dataset NSCLC + ICI + tri-modal (clinical + radiology + pathology) công khai **duy nhất** tìm được; các ứng viên ngoài NSCLC tốt nhất là IMvigor210 (urothelial), IMmotion150 (RCC) và Gide 2019 (melanoma), đều dạng clinical + genomics. Toàn bộ dataset tìm được là **feature-level** (không có raw CT/WSI công khai kèm response ICI) — phù hợp với kiến trúc "đơn giản như trong project" (tabular → HeteroData, cosine-similarity edges, radiomics sẵn có), nhưng không dataset nào khớp 100% schema 22/15/34 — mọi ứng viên cần bước mapping feature.

## 2. Bảng fit (sau khi loại EXCLUDED)

| Ưu tiên | Dataset | Tumor | Modalities | N | Response labels | License | Fit với schema project |
|---|---|---|---|---|---|---|---|
| **1 (S)** | **I3LUNG** — Zenodo 17535424 | NSCLC ✅ | clinical (57) + pyradiomics (131) + pathology FM embeddings (846) + genomics (KRAS/TP53/STK11) | 2.075 (tri-modal 391) | BEST RESPONSE 4-class + ORR/DCR/CBR + PFS/OS | CC-BY-NC-4.0 | Clinical ≈ 1:1 (TMB/PD-L1/ECOG/NLR/BM); radiology = pyradiomics **patient-level** (project dùng lesion-level); pathology = FM embeddings → cần thay extractor GLCM; edges cosine tính được |
| **2 (A)** | **IMvigor210** — GitHub mirror Bioconductor / cBioPortal `blca_iatlas_imvigor210_2017` | Urothelial | clinical + RNA-seq + mutations + TMB | 348 (RNA 298) | **RECIST 4-class + binary** | Genentech license (pkg) / cBioPortal | Clinical view: age/sex/ECOG/albumin/dNLR?/TMB ✓; thêm view genomics mới (RNA-seq) hoặc map vào clinical |
| **3 (A)** | **IMmotion150** — cBioPortal `rcc_iatlas_immotion150_2018` | RCC | clinical + WES + RNA-seq | 263 (174 atezo) | RESPONDER 72/175 + CB + PFS | cBioPortal public | Như trên; PFS có nhưng response label chính là RESPONDER |
| **4 (A)** | **Gide 2019** — cBioPortal `mel_iatlas_gide_2019` | Melanoma | clinical + RNA-seq | 91 (75 labeled) | RESPONDER 40/35 + CB + OS | cBioPortal public | Như trên |
| **5 (A/B)** | **Hellmann/CheckMate 012** — cBioPortal `nsclc_mskcc_2018` | NSCLC ✅ | clinical + WES | 75 | BEST_OVERALL_RESPONSE + DCB + PFS | cBioPortal public | NSCLC + clinical → external check cho clinical view; không imaging |
| **6 (B)** | **Samstein 2019** — cBioPortal `tmb_mskcc_2018` | pan-cancer, NSCLC ≈315 | clinical + MSK-IMPACT (TMB/drivers) | 1.661 | **OS only** (portal không có RECIST) | cBioPortal public | TMB/driver phong phú; thiếu response label → chỉ dùng cho TMB/driver so sánh |
| **7 (B)** | GSE126044 / GSE135222 | NSCLC ✅ | RNA-seq | 16 / 27 | responder/non-responder (metadata / paper) | GEO public | Rất nhỏ, chỉ pilot |
| 8 (B) | Granata 2021 (Zenodo 5162861) | NSCLC LUAD | CT radiomics IBSI | 88 | OS/PFS only (không RECIST) | CC-BY-4.0 | Thiếu response → không fit |
| 9 (C) | TCIA Anti-PD-1_Lung | NSCLC | CT+PET DICOM | 46 | none | CC BY 3.0 | Không label → cần bổ sung |

**Nguồn EXCLUDED (cùng cohort project):** cBioPortal `lung_msk_mind_2020` + Synapse `syn26642505` (Vanguri 2022, N=247) — giữ làm reference (Synapse có radiomics + IHC texture, geo-blocked từ vùng này).

## 3. Khuyến nghị cho "simple adaptation"

1. **I3LUNG trước** — dataset ngoài duy nhất có đủ 3 modality + NSCLC + RECIST. Việc cần làm: (a) mapping clinical 57 cột → 22-dim (giữ TMB/PD-L1/ECOG/albumin/NLR + driver flags); (b) quyết định radiology: dùng pyradiomics patient-level làm view trực tiếp (bỏ lesion nodes) hoặc tìm cột lesion ID; (c) pathology: thay GLCM 15-dim bằng PCA từ FM embeddings (846×771) hoặc bỏ view; (d) dựng HeteroData + cosine edges như project; (e) binary hóa response (CR/PR = responder, SD/PD = non-responder) đúng convention 1=non-responder.
2. **IMvigor210** — external check mạnh nhất ngoài NSCLC: 348 BN, RECIST 4-class sẵn, clinical view gần khớp; dùng pipeline 1-view (clinical) hoặc 2-view (clinical + genomics-latent).
3. **Hellmann/Samstein** — NSCLC clinical-only check (nhanh, giá trị tuyên bố "clinical encoder").
4. TCIA/synapse geo-block: cần VPN/account để tải Synapse full release (radiomics+IHC texture) — ghi vào blockers.

## 4. Hạn chế & điều kiện thay đổi kết luận

- **Hạn chế:** không dataset ngoài nào khớp 100% schema (lesion-level radiology + GLCM pathology); I3LUNG license **non-commercial** (CC-BY-NC-4.0 — phù hợp đồ án/paper academic, không phù hợp thương mại); IMvigor210 license Genentech cần đọc kỹ; các URL TCIA/Synapse không fetch trực tiếp được từ vùng mạng này (geo-block NOT-OD-25-083) — xác minh qua Wayback + data-availability statements.
- **Điều gì đổi kết luận:** nếu truy cập được Synapse `syn26642505` (qua VPN/account) → bản full Vanguri vẫn là cohort nguồn (không tính là external); nếu xuất hiện dataset NSCLC ICI mới có raw CT + segmentation + response (2026+) → S-tier mới; nếu owner chấp nhận non-NC license → I3LUNG là lựa chọn hàng đầu rõ ràng.
