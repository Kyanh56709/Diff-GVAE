# C2 — DyAM baseline (deep-multipit) trên split GVAE

> **Ngày:** 2026-10-06 · **Trạng thái:** đóng C2 (DyAM đã cài đặt lại).
> **Nguồn:** `https://github.com/sysbio-curie/deep-multipit` — mô hình MSKCC `LateAttentionFusion` + `MSKCCAttention` (Vanguri et al. 2022; Captier et al. Nat Commun 2025).
> **Kết luận một dòng:** DyAM đạt head ROC-AUC **0.6822 ± 0.0140** (AUPRC 0.8567) — cao hơn GVAE head (0.6635 ± 0.0161) về danh nghĩa nhưng **không** khác biệt có ý nghĩa sau Holm (seed-mean p_holm 0.84 head / 0.65 probe), và vẫn thua clinical LR (0.707).

---

## 1. Mô hình (tái cài đặt trung thực)

Từ `dmultipit/model/model.py::LateAttentionFusion` + `attentions.py::MSKCCAttention` + `embeddings.py::ModalityEmbedding`:

- Mỗi modality → `Linear(d,1)` + `Tanh` (predictor unimodal, `h_sizes=[]`).
- `MSKCCAttention`: `Linear(d,1)(x)/d` mỗi modality → `softplus` → mask view thiếu (0) → `L1-normalize` → trọng số attention.
- Logit cuối = tổng có trọng số attention của các unimodal logits; view thiếu bị zero.

**Ánh xạ sang 3 view của mình** (cùng block feature như các baseline C2 trong `run_baselines.py`):
clinical (22) · pathology (15 + presence-flag) · radiology (mean+max lesion + presence-flag, 65). Mask = view presence; thiếu → zero + loại khỏi softmax.

## 2. Protocol

`research/2026-10-06-c2-dyam/scripts/run_dyam.py` — **đúng split/seeds của GVAE pooled-OOF**: outer `StratifiedKFold(5, seed)` (42–46); inner `train_test_split(0.2, eval_seed=4200)` chọn `lr×wd` (4 tổ hợp) theo inner-val AUC; model train trên inner-train (parity dữ liệu với GVAE), early-stop inner-val, chấm outer test. Bootstrap CI + DeLong vs GVAE head/probe (Holm).

## 3. Kết quả (seeds 42–46)

| Model | ROC-AUC (mean ± sd) | PR-AUC |
|---|---|---|
| **DyAM** | **0.6822 ± 0.0140** | **0.8567 ± 0.0091** |
| GVAE head (reference tested by DeLong) | 0.6635 ± 0.0161 | 0.8297 |
| GVAE probe (reference tested by DeLong) | 0.6356 ± 0.0238 | — |
| clinical LR (baseline C2) | 0.7069 ± 0.0157 | 0.8626 |
| late-fusion MLP | 0.6920 ± 0.0305 | 0.8704 |
| SVM-RBF concat | 0.6872 ± 0.0461 | 0.8527 |

DeLong DyAM vs GVAE (Holm): head seed-mean Δ −0.0077, **p_holm 0.844**; probe Δ −0.0373, **p_holm 0.649**. Per-seed chỉ `seed43/probe` p=0.042 → p_holm 0.084 (không ý nghĩa). Không seed nào khác p<0.05.

## 4. Kết luận trung thực

1. **DyAM ≈ GVAE, không hơn có ý nghĩa.** Danh nghĩa 0.682 vs 0.664 (reference `drop_both32` mà DeLong test) nhưng DeLong ns sau Holm — đồng nhất với phát hiện B5 (không baseline nào khác GVAE có ý nghĩa).
2. **Clinical LR vẫn mạnh nhất** (0.707); DyAM nằm trong nhóm baseline (0.68–0.69), trên GVAE head một chút.
3. Đóng C2: XGBoost→HistGradientBoosting, SVM, LR, late-fusion MLP (2026-10-04) + **DyAM (2026-10-06)** đều đã chạy trên cùng split/seeds; mọi so sánh có DeLong.

## 5. Giới hạn

- Modality radiology của mình là **aggregate mean+max** per patient (32→65), không phải per-site radiomics (PC/LN/PL) như Vanguri — do dữ liệu khác.
- Chỉ 4 tổ hợp lr×wd (chọn inner-val); train trên inner-train (không refit outer-train như một số baseline khác) → ước lượng thận trọng hơn.
- Một cohort; 5 seed.
- Trong `_train_eval` có `torch.manual_seed(0)` mỗi lần train → **init giống nhau giữa các seed**; độ tản seed chỉ phản ánh split outer (khác GVAE). Ghi rõ để không diễn giải quá mức độ tản.

## 6. Tái lập

```bash
.venv/bin/python research/2026-10-06-c2-dyam/scripts/run_dyam.py --seeds 42,43,44,45,46
# output (git-ignored): research/2026-10-06-c2-dyam/output/dyam/{metrics_per_seed,delong_vs_gvae}.csv, dyam_summary.json
```
