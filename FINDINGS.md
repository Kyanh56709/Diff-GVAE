# Findings (owner-side) — multimodal GVAE on the NSCLC immunotherapy cohort

**Phạm vi:** tổng hợp các phát hiện từ phần owner (nhóm A/B/C/F/H). **Chưa gồm** phần diffusion (DDPM/TSTR, nhóm E/D — của Loan). Cohort: Vanguri/DFCI NSCLC PD-(L)1, canonical N=247 (TMB not-null), objective response; seeds 42–46.

**Finding chính:** mô hình **GVAE đa phương thức không vượt baseline đơn giản**. Hồi quy logistic chỉ trên clinical là mạnh nhất; các thành phần cốt lõi của phương pháp (graph message passing, cross-view alignment, contrastive, lesion attention) **không tạo lợi ích đo được** trên cohort này. Đây là một **null result có kiểm soát rò rỉ**, không phải một con số "đẹp".

---

## 1. Mô hình không vượt baseline

Pooled-OOF ROC-AUC (5 seed), cùng split:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| clinical-only logistic regression | **0.7069 ± 0.0157** | 0.8626 |
| late-fusion MLP | 0.6920 ± 0.0305 | 0.8704 |
| SVM-RBF (concat) | 0.6872 ± 0.0461 | 0.8527 |
| DyAM (deep-multipit) | 0.6822 ± 0.0140 | 0.8567 |
| concat logistic regression | 0.6804 ± 0.0440 | 0.8471 |
| gradient-boosted trees | 0.6802 ± 0.0268 | 0.8562 |
| **GVAE head (fusion)** | 0.6635 ± 0.0161 | 0.8297 |
| GVAE linear probe | 0.6356 ± 0.0238 | 0.8084 |

**DeLong (Holm, 5 seed):** không so sánh nào có ý nghĩa — GVAE head vs baselines min p_holm 0.157; vs DyAM p_holm 0.84 (head)/0.65 (probe). Xem `research/2026-10-04-baselines-delong/`, `research/2026-10-06-c2-dyam/`.

## 2. Các thành phần "cốt lõi" đóng góp ~0

| Thành phần | Kết quả | Kết luận |
|---|---|---|
| Graph message passing | bỏ cạnh (encoder MLP) 0.6518 vs full 0.6531 | vô dụng cho head |
| Cross-view alignment / fusion | `pathology_only` 0.6790, `clinical_only` 0.6615 > full 0.6531 | không cải thiện |
| Contrastive loss | `no_contrastive` 0.6633 ≥ full 0.6531 | không giúp |
| Lesion attention vs mean/max | 0.6531 vs 0.6567 / 0.6674 | không hơn pooling đơn giản |
| Radiology view | `radiology_only` 0.5286 | view yếu nhất, không đóng góp |

Không ablation nào có ý nghĩa sau Holm (min p_holm 0.099 ở `radiology_only`). Xem `research/2026-10-06-c1-ablations/`.

## 3. Kết quả bền vững (robustness)

- **Nested CV** (chọn hyperparameter theo inner-val từng fold): 0.6603 ± 0.0460 vs cấu hình cố định 0.6531 ± 0.0113 — chênh +0.007 nằm trong sd, độ tản seed ~4× → **không** có rò rỉ chọn-lựa làm phồng headline. `research/2026-10-06-nested-cv-hparam/`.
- **Hyperparameter-insensitive:** 7 cấu hình một-trục trong 0.641–0.669, mọi p_holm = 1.0. `research/2026-10-06-c3-sensitivity/`.
- **Cohort đầy đủ N=366:** 0.6406 ± 0.0137 (AUPRC 0.8175) — gần bằng N=247 (0.6531); con số legacy "0.711 ± 0.080" **không tái lập**. `research/2026-10-06-c4-366/`.
- **Internal validity sạch (điểm mạnh thật):** imputation/scaler chỉ fit train-fold; message passing tách split (train→train, val→val, test→test) xác minh ở mọi đường, kể cả latent extraction cho DDPM. `research/2026-10-06-internal-validity-methods/`.

## 4. Interpretability nhất quán với kết quả âm

- **Integrated Gradients** gần đều 3 view: clinical 0.0215 · pathology 0.0242 · radiology 0.0226 → không view nào chi phối.
- **Lesion attention** chỉ hơi tập trung: mean max weight 0.778, entropy 0.417 nats (đều = 0.461) → không chọn lọc mạnh; không có ground-truth cấp lesion để xác nhận "hợp lý".
- **Latent probe** cho response chỉ **0.645**; gene probe 0.60–0.73 nhưng gene là **input** nên phản ánh tái tạo, **không** phải "gene encoding" tự phát.
`research/2026-10-06-f-interpretability/`.

## 5. Đóng góp phương pháp/kỹ thuật (dương tính)

- Tái lập **canonical r32** (32 radiomics, bỏ 2 cột index artifact) và hòa giải số giữa notebook và pipeline.
- Bảo đảm **internal validity** có bằng chứng: fit train-only, tách split đồ thị, không rò rỉ (A2/A3/A5).
- Khung đánh giá chặt: **pooled-OOF + nested CV + DeLong/Holm + bootstrap CI**.
- **Script tái lập một lệnh** (`reproduce_canonical.py`, dry-run được).

## 6. Caveat (bắt buộc ghi)

- N nhỏ (247/366), **một cohort**, 5 seed; vô hiệu có thể do cỡ mẫu/tín hiệu yếu — **không** hàm ý phương pháp vô dụng nói chung.
- Các so sánh theo seed (n=5) chỉ mô tả; không đủ lực cho kiểm định mạnh.
- F2 không có ground truth cấp lesion; F3 gene-probe là tái tạo input.
- **Chưa gồm diffusion** (DDPM/TSTR) — tầng finding đó thuộc nhóm của Loan.

---

### Tài liệu bằng chứng (trong `research/`)

`2026-10-04-baselines-delong/` · `2026-10-06-c2-dyam/` · `2026-10-06-c1-ablations/` · `2026-10-06-nested-cv-hparam/` · `2026-10-06-c3-sensitivity/` · `2026-10-06-c4-366/` · `2026-10-06-internal-validity-methods/` · `2026-10-06-f-interpretability/`. Tái lập toàn bộ: `reproduce_canonical.py`.
