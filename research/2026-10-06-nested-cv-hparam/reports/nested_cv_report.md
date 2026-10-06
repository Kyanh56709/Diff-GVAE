# B2 — Nested CV (chọn hyperparameter bên trong vòng lặp CV)

> **Ngày:** 2026-10-06 · **Người làm:** owner-side agent · **Trạng thái:** đóng B2.
> **Kết luận một dòng:** nested CV chọn hyperparameter theo từng outer fold cho head AUC **0.6603 ± 0.0460**, cấu hình cố định **0.6531 ± 0.0113** — chênh **+0.007** nằm trong sd của nested và độ tản seed tăng ~4×, tức nested CV **không** thổi phồng headline và outer test không bị rò rỉ qua việc chọn hyperparameter.

---

## 1. Câu hỏi

`PUBLICATION_CHECKLIST` B2: "Add a locked held-out test set OR nested CV (outer test never touched for model selection). Current CV mixes selection and evaluation."

Protocol `kfold_evaluate_gvae_classifier` (`training/train_gvae.py:2034`) **đã** giữ outer fold report-only và chọn checkpoint + threshold trên inner-val (train-only). Bước chọn còn nằm *ngoài* CV là **hyperparameter**: config canonical (copy từ `pipeline_phase4a_oof.sh`) và `best_params.json` được chốt trên graph 34-chiều, tức có nhìn vào cùng 247 bệnh nhân. Task này bổ sung **nested CV** để đóng khoảng trống đó.

## 2. Protocol

Harness độc lập (không sửa `training/`, chỉ import helper):
`research/2026-10-06-nested-cv-hparam/scripts/nested_cv_gvae.py`.

- Split giống canonical: `StratifiedKFold(5, shuffle=True, random_state=seed)` (outer, report-only); `train_test_split(0.2, stratify, random_state=4200)` (inner) — tách từ **train của mỗi fold**.
- Bộ candidate **chốt trước** (pre-registered), 3 cấu hình một-trục:
  | Tên | Thay đổi so với canonical |
  |---|---|
  | `canonical` | — |
  | `d_embed16` | `d_embed = 16` |
  | `hidden64` | `hidden_channels_vae = 64` |
- Mỗi (fold × candidate): pretrain radiology (100 ep) → train GVAE (80 ep, early-stop theo inner-val AUC) — đúng như `kfold_evaluate_gvae_classifier`.
- **Chọn candidate có inner-val AUC cao nhất**, rồi dùng chính model đã early-stop đó để dự đoán **outer test**; linear probe fit trên inner-train µ → outer test. Gom OOF toàn bộ 247 BN; bootstrap 95% CI. Seed 42–46.
- Comparator chạy **cùng harness, cùng code, chỉ `canonical`** → bit-exact với `run_oof.py`.

### 2.1 Kiểm tra harness trung thực

Harness `canonical`-only phải khớp `run_oof.py`. Lần đầu khớp *không* đúng (0.6230 vs 0.6602) vì harness hardcode `cross_cl` anneal bắt đầu từ `0.0` thay vì `start_weight=0.1` trong config. Sau khi sửa, harness khớp **bit-exact**:

| Đường chạy (seed 42) | HEAD AUC | AUPRC | PROBE AUC |
|---|---|---|---|
| `run_oof.py` (canonical, hiện tại) | 0.6602 | 0.8234 | 0.6622 |
| harness `canonical`-only | 0.6602 | 0.8234 | 0.6622 |

Cả hai đường đều **tất định** (chạy lại 2 lần ra y hệt), nên đây là đối chứng hợp lệ chứ không phải nhiễu.

**Ghi chú sửa lỗi (review gate, 2026-10-06):** bản đầu của harness chỉ `torch.manual_seed`/`np.random.seed` **một lần trước vòng lặp seed**, nên các seed 43–46 chạy nối tiếp trạng thái RNG và *không* khớp `run_oof.py` (vốn seed mỗi process). Đã sửa: reseed **trong** vòng lặp, mỗi seed độc lập. Sau khi sửa, harness `canonical`-only khớp `run_oof.py` theo **từng seed** (seed 42 = 0.6602; seed 43 = 0.6579), không chỉ seed đầu. Toàn bộ số dưới đây là bản đã sửa.

## 3. Kết quả

Head AUC theo seed (outer test, OOF) — mỗi seed độc lập:

| Seed | Canonical cố định | Nested CV | Candidate được chọn theo fold |
|---|---|---|---|
| 42 | 0.6602 [0.5813–0.7373] | 0.5994 [0.5177–0.6881] | d16, canon, canon, d16, canon |
| 43 | 0.6579 [0.5792–0.7401] | 0.6524 [0.5733–0.7301] | d16, d16, canon, canon, h64 |
| 44 | 0.6377 [0.5550–0.7146] | 0.7218 [0.6447–0.7929] | canon, h64, h64, h64, canon |
| 45 | 0.6450 [0.5659–0.7274] | 0.6432 [0.5652–0.7256] | canon, h64, h64, d16, canon |
| 46 | 0.6647 [0.5839–0.7458] | 0.6848 [0.6050–0.7616] | canon, canon, canon, canon, d16 |
| **Mean ± sd (mẫu)** | **0.6531 ± 0.0113** | **0.6603 ± 0.0460** | — |

| Chỉ số (mean ± sd mẫu, 5 seed) | Canonical cố định | Nested CV |
|---|---|---|
| Head ROC-AUC | 0.6531 ± 0.0113 | 0.6603 ± 0.0460 |
| Head AUPRC | 0.8287 ± 0.0034 | 0.8306 ± 0.0242 |
| Linear-probe AUC | 0.6435 ± 0.0285 | 0.6373 ± 0.0211 |

## 4. Kết luận trung thực

- **Nested CV không thổi phồng kết quả.** Head AUC trung bình chênh **+0.007** (0.6603 vs 0.6531) — nhỏ hơn nhiều so với chính độ tản của nested (sd 0.046), và **độ tản seed tăng ~4 lần** (0.046 vs 0.011). AUPRC và probe AUC gần như không đổi.
- Diễn giải: với N = 247, inner-val mỗi fold chỉ ~40 BN, chọn hyperparameter theo inner-val **ồn** — có fold chọn `d_embed16`/`hidden64` rồi thắng, có fold thua `canonical` (seed 42 tụt còn 0.5994). Đây là hành vi đúng của nested CV ở cỡ mẫu nhỏ, không phải lỗi.
- Hệ quả cho bài: headline dùng **cấu hình cố định** là hợp lệ; kết quả không phải sản phẩm của việc nhìn trước outer test khi chọn hyperparameter. Bộ số canonical (`final_results_package_r32.md`, head 0.6635 ± 0.016) và bộ chạy lại hiện tại (0.6531 ± 0.011) khác nhau ~0.010 do trạng thái code/thread — nằm trong biên độ tái lập đã ghi của repo, không đổi kết luận.
- **B2 đóng:** outer test không được dùng cho bất kỳ bước chọn nào (checkpoint, threshold, hyperparameter).

## 5. Giới hạn

- Chỉ 3 candidate một-trục; lưới đầy đủ (108–216 tổ hợp) bất khả thi trên CPU.
- Một cohort, N = 247; nested CV không giải quyết được "locked held-out test độc lập" — vẫn là hạn chế dữ liệu, đã ghi ở Limitations.
- So sánh cặp theo seed (n=5) chỉ mô tả; không đủ lực cho kiểm định.

## 6. Tái lập

```bash
# comparator cấu hình cố định (bit-exact run_oof)
.venv/bin/python research/2026-10-06-nested-cv-hparam/scripts/nested_cv_gvae.py \
  --data data_ln_pc_ihc_g_r32.pt --candidates canonical --seeds 42,43,44,45,46 --tag canonical_fixed
# nested CV 3 candidate
.venv/bin/python research/2026-10-06-nested-cv-hparam/scripts/nested_cv_gvae.py \
  --data data_ln_pc_ihc_g_r32.pt --seeds 42,43,44,45,46
```

Output (git-ignored): `research/2026-10-06-nested-cv-hparam/output/{nestedcv,canonical_fixed}_summary.json`
và `..._{tag}_seed{42..46}/nested_cv_result.json` (kèm candidate được chọn mỗi fold).
