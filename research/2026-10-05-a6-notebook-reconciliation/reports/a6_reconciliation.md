# A6 — Hòa giải bộ số notebook `~0.81–0.87` vs canonical pooled-OOF

> **Ngày:** 2026-10-05 · **Người làm:** owner-side agent · **Trạng thái:** đóng A6 (giải thích được divergence; không còn số nào chưa hòa giải).
> **Kết luận một dòng:** `~0.81–0.87` là **artifact legacy của protocol chọn-trên-chính-tập-báo-cáo** (sweep `kfold_train_gvae` val-fold, chọn checkpoint theo val-AUC, ngưỡng tune trên val) trên **graph 34 chiều** — không phải số canonical, **không tái lập được** bằng code hiện tại, và đã bị thay bởi pooled-OOF r32.

---

## 1. Câu hỏi

`PUBLICATION_CHECKLIST` A6: hai bộ số phân kỳ — `training_local.ipynb` (~0.81–0.87) vs `PROJECT_REVIEW.md` outputs (~0.69). Bộ ~0.69 → 0.6065 đã giải thích (val-fold mean + tuned threshold vs pooled-OOF). **Còn mở:** 0.81–0.87 đến từ đâu và có tái lập được không.

## 2. Protocol điều tra

1. Trích xuất mọi output số trong `training_local.ipynb` (48 cell) và truy vết 0.81/0.87.
2. Đọc code tạo số: `training/sweep_gvae.py::run_gvae_sweep` → `training/train_gvae.py::kfold_train_gvae`.
3. Chạy lại **code HEAD hiện tại** với **đúng config notebook** (cell 1) và **đúng hyperparam thắng sweep** trên **đúng graph legacy** `data_ln_pc_ihc_g.pt`, output cách ly vào `research/2026-10-05-a6-notebook-reconciliation/output/`.
4. So sánh với bản code lịch sử tạo số gốc (`git show 0392718:training/train_gvae.py`, commit 2026-06-15 "implement hyperparameter sweep pipeline").
5. Đối chiếu claim trong manuscript `EJC_Research_Paper.pdf` (31 trang).

## 3. Kết quả

### 3.1 Nguồn của `~0.81–0.87` trong notebook

`training_local.ipynb` cell 2 chạy `run_gvae_sweep(...)` (grid 108 tổ hợp). Output in ra:

| Trial thắng | mean_auc | std_auc | mean_f1 | std_f1 | mean_accuracy | mean_loss_at_best_auc |
|---|---|---|---|---|---|---|
| `cross_cl=0.5, d_embed=16, hidden=128, attn=64, heads=8` | **0.8116** | 0.0557 | **0.8787** | 0.0260 | 0.8100 | 4.66 |

→ `0.8116` và `0.8787` **chính là** khoảng "~0.81–0.87" mà checklist nêu. Không có chuỗi `0.806`/`0.867` literal trong notebook; hai số đó nằm trong **manuscript**.

### 3.2 Chính notebook đó cũng chứa số canonical

Cell 4 và cell 6 gọi `kfold_evaluate_gvae_classifier` (protocol pooled-OOF hiện đại) và in:

```
POOLED OUT-OF-FOLD RESULTS (N=247, prevalence=0.749)
  Fusion head : AUC   0.6502  [95% CI 0.5693-0.7325]
              : AUPRC 0.8161
  Linear probe: AUC   0.5842
```

→ Trên **cùng graph legacy**, protocol canonical của notebook đã cho ~0.65, khớp canonical r32 (head 0.6635 ± 0.016). Nghĩa là notebook chứa **cả hai** protocol và số sạch của nó đồng ý với canonical.

### 3.3 Chạy lại code HEAD với config notebook (seed 42, 200 epoch, graph legacy)

| Biến thể | `checkpoint_metric` | mean_auc | mean_f1 | pooled val AUC |
|---|---|---|---|---|
| HEAD hôm nay | `latent_quality` (mặc định) | **0.6086** | 0.5685 | 0.6168 |
| HEAD hôm nay | `auc` | **0.7190** | 0.8299 | 0.6693 |
| Sweep gốc (2026-06) | val-AUC trực tiếp | **0.8116** | 0.8787 | — |

Không biến thể nào của code HEAD đạt 0.81. `mean_loss_at_best_auc` cũng lệch bậc so với bản gốc (4.66 gốc vs 5 746 hiện tại vs 57 774 bản v2 2026-08) → **code đã đổi**.

### 3.4 Khác biệt code quyết định

`git show 0392718:training/train_gvae.py` (bản tạo số 0.8116) **không có khái niệm `checkpoint_metric`/`latent_quality`**: nó lưu model tại `current_val_auc > best_val_auc` (dòng 310–311), và trong `fold_metrics_list` ghi `auc = best_epoch_results['auc']` — tức **AUC đo trên chính val fold dùng để chọn checkpoint**. `f1` lấy ngưỡng `argmax` F1 **trên cùng val fold** (dòng 351). Từ HEAD, `checkpoint_metric` mặc định `latent_quality` (ổn định variance train/val + linear-probe + silhouette/fisher), không tối ưu trực tiếp AUC.

## 4. Kết luận trung thực

**Cơ chế của 0.81–0.87 (3 tầng, cộng dồn):**

1. **Protocol chọn-trên-tập-báo-cáo (chính):** sweep ghi AUC/F1 của checkpoint được chọn *vì* AUC cao nhất trên val, rồi báo cáo trên đúng val đó; ngưỡng F1 cũng tune trên val. Đây là circular selection → lạc quan hệ thống, đúng như B4 cảnh báo. Pooled-OOF (inner-val chọn, outer-test báo cáo) loại bỏ vòng này.
2. **Graph 34 chiều:** sweep chạy trên `data_ln_pc_ihc_g.pt` (34 = 32 radiomics + 2 index artifact). Ablation r32 (`research/2026-10-03-radiology-artifact-ablation/`) chứng minh 2 slot index tương quan nhãn → 34-dim lạc quan hơn r32.
3. **Code drift:** bản 2026-06 chọn checkpoint theo val-AUC; HEAD mặc định `latent_quality`. Chạy HEAD với config notebook cho 0.6086 (`latent_quality`) hoặc 0.7190 (`auc`) — **không đạt 0.81**, nên phần dư còn lại đến từ code pre-Jun15 và/hoặc seed split.

**Quyết định:** `0.81/0.87` là **số legacy đã bị thay**, không dùng cho bất kỳ bảng/hình nào. Số canonical duy nhất = pooled-OOF trên r32: **fusion head 0.6635 ± 0.016 (5 seed), seed 42 = 0.6431 [0.5628–0.7272]; probe 0.6356 ± 0.024** (`research/2026-10-03-canonical-r32/reports/final_results_package_r32.md`).

**Đồng thời hòa giải I2** (draft có 0.806 và 0.867): theo manuscript, **0.867** là "combined linear probe" *và* bị dùng lẫn cho "full GVAE" (§4.1.1 và Table 1 — hai đại lượng khác nhau bị gán cùng số); **0.806** là số fusion sau ablation contrastive (`0.806 → 0.761`) và modality-ablation. Cả hai đều thuộc protocol val-fold (một nguồn ghi "10 folds", repo chạy 5 folds) → là họ số legacy, không phải canonical. Khuyến nghị manuscript: chỉ dùng `final_results_package_r32.md`, tách rõ fusion-head vs linear-probe, và ghi số canonical (0.6431 seed42 / 0.6635±0.016).

## 4b. Core-code issue phát hiện kèm (ngoài scope A6, chưa sửa)

Audit error-handling độc lập (`silent-failure-hunter`, 2026-10-05) tìm thấy các vấn đề **có sẵn trong code lõi** (không phải do script A6). Ghi lại để owner quyết định; **không sửa tự ý vì đổi hành vi sẽ ảnh hưởng kết quả canonical**:

1. **`training/train_gvae.py` — chọn checkpoint có thể chọn fold đã diverge.** Non-finite batch bị `continue` không đếm/log; guard duy nhất là `torch.isfinite(total_val_loss)`, nên loss **explode nhưng vẫn hữu hạn** lọt qua. Bằng chứng trong chính artifact A6: fold 4 của **cả hai** run chọn checkpoint có `val_loss` 28 714 (`latent_quality`) và 163 147 (`auc`), vẫn được tính vào mean. Script A6 giờ **đánh dấu** các fold này (`diverged_folds`, `diverged_fold_count`) thay vì che.
2. **`training/train_gvae.py::_compute_latent_quality_metrics`** — `except Exception` rộng biến lỗi linear-probe thành `NaN`; NaN được map thành `-inf` trong sort key nhưng candidate vẫn được chọn, và fold record **không lưu** metric latent-quality → không truy vết được probe có chạy không.
3. **`training/sweep_gvae.py`** — trial crash được ghi như một row "thành công" (`mean_auc=NaN` + cột `error` chung schema); `summary.get('mean_auc', 0.0)` biến key thiếu thành `0.0` (một giá trị AUC hợp lệ); CSV hỏng khi resume chỉ `WARNING` rồi bắt đầu lại.
   *Đối với A6:* premise này **không bị ảnh hưởng** — row sweep nguồn (row 83) có `mean_auc=0.8116` hữu hạn, CSV không có row NaN/`error` nào (đã kiểm 2026-10-05).

Các điểm trên nằm trong đường code lõi mà `Loan_agents.md` §1.3 xếp là "của owner"; nếu muốn sửa nên làm thành PR flag opt-in kèm test (quy ước "New Feature Flags v2" của `CLAUDE.md`) và chạy lại canonical sau.

## 5. Giới hạn

- Không tái lập bit-exact 0.8116: cần code pre-Jun15 (git có thể check out `0392718`) **cộng** seed split gốc không được lưu. Đã chứng minh đủ để giải thích divergence mà không cần dựng lại môi trường cũ.
- Chạy lại chỉ seed 42/200 epoch (khớp sweep); sweep gốc gọi `run_gvae_sweep(random_seed=420)` nhưng `kfold_train_gvae` khi đó không nhận `random_seed` → split không seed, thêm phương sai.
- Số sweep dùng CPU (notebook gốc có thể MPS/CUDA); backend có thể đổi thứ tự float, không đổi kết luận.
- Chưa chạy 10-fold (manuscript ghi 10 folds) — repo chuẩn là 5-fold.

## 6. Tái lập

```bash
# Biến thể HEAD mặc định (latent_quality) — kỳ vọng mean_auc ~0.609
.venv/bin/python research/2026-10-05-a6-notebook-reconciliation/scripts/run_legacy_kfold_repro.py \
    --epochs 200 --seed 42

# Biến thể mô phỏng chọn checkpoint theo val-AUC — kỳ vọng mean_auc ~0.719
.venv/bin/python research/2026-10-05-a6-notebook-reconciliation/scripts/run_legacy_kfold_repro.py \
    --epochs 200 --seed 42 --checkpoint-metric auc

# Số canonical để so (đã freeze, không tái tạo lại)
# research/2026-10-03-canonical-r32/reports/final_results_package_r32.md §2a
```

Output: `research/2026-10-05-a6-notebook-reconciliation/output/legacy_repro_seed42_ep200_{latent_quality,auc}.json`.
