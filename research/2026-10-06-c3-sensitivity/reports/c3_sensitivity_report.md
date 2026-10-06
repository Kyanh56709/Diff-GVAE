# C3 — Hyperparameter sensitivity (canonical r32 GVAE)

> **Ngày:** 2026-10-06 · **Trạng thái:** đóng C3 (các trục hyperparameter).
> **Kết luận một dòng:** không biến thể hyperparameter nào khác canonical có ý nghĩa sau Holm (mọi `p_holm = 1.0`); ROC-AUC nằm trong dải hẹp 0.641–0.669 và thứ hạng cấu hình ổn định → kết quả **không nhạy** với các trục đã khảo sát.

---

## 1. Protocol

`research/2026-10-06-c3-sensitivity/scripts/run_sensitivity.py` — pooled-OOF đúng protocol canonical (`kfold_evaluate_gvae_classifier`), seeds 42–46, mỗi biến thể lệch **một trục** so với canonical; DeLong ghép cặp vs canonical + Holm.

## 2. Kết quả (seeds 42–46)

| Cấu hình | Thay đổi | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) | Δ vs canonical | p_holm |
|---|---|---|---|---|---|
| **canonical** | — | **0.6531 ± 0.0113** | 0.8287 ± 0.0034 | — | — |
| `tau02` | cross_cl_temp 0.2 | 0.6578 ± 0.0191 | 0.8251 ± 0.0103 | −0.0047 | 1.0 |
| `d16` | d_embed 16 | 0.6409 ± 0.0386 | 0.8286 ± 0.0183 | +0.0121 | 1.0 |
| `d64` | d_embed 64 | 0.6425 ± 0.0278 | 0.8220 ± 0.0126 | +0.0106 | 1.0 |
| `heads4` | attention heads 4 | 0.6693 ± 0.0243 | **0.8385** ± 0.0174 | −0.0163 | 1.0 |
| `hidden64` | hidden width 64 | 0.6626 ± 0.0183 | 0.8233 ± 0.0164 | −0.0095 | 1.0 |
| `cl02` | cross_cl weight 0.2 | 0.6536 ± 0.0332 | 0.8274 ± 0.0207 | −0.0005 | 1.0 |

DeLong (seed-mean, Holm): mọi `p_holm = 1.0`.

## 3. Kết luận trung thực

- **Không trục nào có ý nghĩa.** Chênh lớn nhất là `heads4` (−0.016 ⚠️ *theo hướng tốt hơn* canonical: 0.669 vs 0.653) nhưng p_holm 1.0.
- **Ổn định (mức trung bình):** ROC-AUC toàn bộ 7 cấu hình trong 0.641–0.669; PR-AUC 0.822–0.839. Chênh lệch **per-seed** đổi dấu ở một số trục (vd `heads4`), nên chỉ kết luận ở mức trung bình: không cấu hình nào bị chi phối/áp đảo bởi hyperparameter.
- PR-AUC cao nhất ở `heads4` (0.8385) — cùng mức canonical (0.829).

## 4. Trục "graph threshold 0.7 vs 0.8"

- Canonical dùng `SIMILARITY_THRESHOLD = 0.8` (`data/build_ln_pc_ihc_g.py:50`); ghi chú code nói con số "0.7" trong tài liệu repo là **cũ/sai**.
- Trục này **được bao hàm gián tiếp**: C1 `no_gnn` (MLP, bỏ toàn bộ cạnh) cho 0.6518 vs full 0.6531 → cạnh similarity **không đóng góp đo được**, nên việc chọn ngưỡng 0.7 hay 0.8 gần như không thể đổi head. Chạy trực tiếp 0.7 cần thêm cờ `--similarity-threshold` cho build script (chưa làm) — ghi là tuỳ chọn.

## 5. Giới hạn

- Một cohort (N=247), 5 seed; so sánh theo seed n=5 chỉ mô tả.
- Chưa chạy trực tiếp graph threshold 0.7 (lý do ở §4).

## 6. Tái lập

```bash
.venv/bin/python research/2026-10-06-c3-sensitivity/scripts/run_sensitivity.py \
    --data data_ln_pc_ihc_g_r32.pt --seeds 42,43,44,45,46 --tag c3
# output (git-ignored): research/2026-10-06-c3-sensitivity/output/c3/{summary,metrics_per_seed,delong_tests}.csv
```
