# F1–F3 — Interpretability trên GVAE canonical r32

> **Ngày:** 2026-10-06 · **Người làm:** owner-side agent · **Trạng thái:** đóng F1, F2, F3.
> **Nguồn:** latent OOF lấy từ 5 checkpoint rank-1 của `gvae_bestparam_ranked_r32_20261003_124706` (mỗi BN nằm đúng một fold val → latent không rò rỉ).
> **Kết luận một dòng:** attribution gần như chia đều giữa 3 view (F1), attention chỉ hơi tập trung (F2), và latent phân tách response ở mức khiêm tốn (probe AUC 0.645) — không có bằng chứng về một không gian latent "giàu sinh học" mạnh hơn mức input cho phép.

---

## F1 — Feature attribution (Integrated Gradients)

IG (torch thuần, 32 bước) của logit fusion-classifier theo từng view-mu; `mean |attr|` mỗi view (trung bình 5 fold, 247 BN):

| View | mean \|IG\| |
|---|---|
| clinical | 0.0215 |
| pathology | 0.0242 |
| radiology | 0.0226 |

**Đọc:** không view nào chi phối — ba view đóng góp gần bằng nhau (pathology hơi cao nhất). **Nhất quán với C1** (unimodal ≥ full; bỏ message-passing không hại): bộ phân loại không dựa chủ yếu vào một modality, và không có bằng chứng alignment tạo ra trọng số vượt trội.

## F2 — Lesion attention

Trên 187 BN có radiology (median 2 lesion/BN):

| Chỉ số | Giá trị |
|---|---|
| mean max attention weight | 0.778 |
| mean entropy (nats) | 0.417 |
| entropy nếu đều (reference) | 0.461 |

**Đọc:** attention **hơi** tập trung hơn đều (0.417 so với 0.461) nhưng không chọn lọc mạnh; với median 2 lesion, trọng số cao nhất ~0.78 là mức vừa phải. **Giới hạn:** không có nhãn cấp-lesion nên **không thể** xác nhận "attention vào lesion hợp lý" — chỉ định lượng được mức tập trung. Cần nhãn lesion/kiểm chứng lâm sàng để xác nhận ý nghĩa.

## F3 — Không gian latent (PCA + t-SNE + probe)

Hình (PNG, git-ignored, trong `research/2026-10-06-f-interpretability/output/`):
`f3_pca_response.png`, `f3_tsne_response.png`, `f3_tsne_gene_{EGFR,ERBB2,BRAF,MET,STK11,ARID1A}.png`.

Probe tuyến tính 5-fold CV trên `concat_mu` OOF (247 BN):

| Đích | probe ROC-AUC |
|---|---|
| response | 0.645 |
| EGFR | 0.713 |
| MET | 0.727 |
| STK11 | 0.708 |
| BRAF | 0.683 |
| ARID1A | 0.681 |
| ERBB2 | 0.603 |

**Đọc trung thực:**
- Response probe 0.645 khớp canonical pooled-OOF probe (~0.636) — nhất quán, mức khiêm tốn.
- Gene probe cao (0.60–0.73) **nhưng các cờ gene là input của view clinical**, nên việc probe đọc lại được chúng phản ánh **tái tạo input**, không phải "gene encoding tự phát" (đúng caveat trong `Loan_agents.md` §L12). Đây **không** phải bằng chứng sinh học mới.
- Hình PCA/t-SNE là định tính; không tuyên bố tách lớp mạnh. Báo cáo nên đặt cạnh probe AUC.

## Kết luận & hệ quả cho bài

1. **Không có "view quan trọng nhất"** theo IG; phù hợp với chuỗi kết quả âm tính (C1/C2: full không hơn unimodal; contrastive/GNN không giúp).
2. **Attention không chọn lọc mạnh** — nên trình bày như phân tích phụ, không phải điểm mạnh.
3. **Gene signal từ latent không phải phát hiện tự phát** (gene là input) — phải ghi rõ; nếu muốn claim "gene encoding", cần thiết kế sạch kiểu I3LUNG (không đưa genomics vào input — việc của Loan, L12).
4. Đây là các kết quả **âm tính/trung tính hợp lệ**, không phải thất bại; viết trung thực.

## Giới hạn

- Latent OOF lấy từ checkpoint `kfold_train_gvae` (val-fold), không phải pooled-OOF; nhưng không rò rỉ (mỗi BN một fold val).
- IG baseline = 0 (chuẩn); attribution theo view-mu (không đi ngược qua encoder/message-passing tới feature gốc).
- F2 không có ground truth cấp lesion.
- Cảnh báo `RuntimeWarning: ... in matmul` từ numpy/sklearn trên macOS Accelerate là **giả** (đã kiểm chứng trong repo); projection đã assert hữu hạn (`np.isfinite`).

## Tái lập

```bash
.venv/bin/python research/2026-10-06-f-interpretability/scripts/run_interpretability.py
# output (git-ignored): research/2026-10-06-f-interpretability/output/{interpretability_summary.json,latent_projections.npz,f3_*.png}
```

Code hỗ trợ F2: `RadiologyLesionAttentionAggregator.attention_weights(...)` (đọc-only, thêm 2026-10-06; mặc định không đổi hành vi; test `tests/test_ablation_flags.py`).
