# Loan_agents.md — Phân công cho Loan: Diffusion + Dataset ngoài

> **Người đọc:** Loan và các AI coding agent Loan dùng (Claude Code, Codex, Kilo…). Agent: đọc toàn bộ file này **và** `CLAUDE.md` trước khi làm bất cứ việc gì.
> **Chủ repo / first author:** Kyanh56709 (gọi tắt "owner" bên dưới).
> **Ngày lập:** 2026-10-04. Mọi số liệu dưới đây được đối chiếu trực tiếp trên repo ngày này (commit `5effd33`).

---

## 0. Tóm tắt một trang

**Loan phụ trách hai mảng, chạy song song với owner và không phải chờ owner:**

1. **Diffusion (DDPM latent augmentation)** — đóng nhóm E của `PUBLICATION_CHECKLIST.md` (E1–E6), phần DDPM của B3 (multi-seed), và **hòa giải phần "Stage 2" của manuscript** (§4.1.4, §4.2, §4.3 — có ghi chú "Ket qua cua em Loan, se check lai sau") với bộ số canonical r32.
2. **Dataset ngoài (external validation)** — đóng nhóm D (D2, D3, D5, D6; D4 là tùy chọn) với I3LUNG làm dataset chính.

**Ba sự thật Loan cần biết ngay từ đầu (đã kiểm chứng, không phải giả định):**

- **Claim TSTR trong manuscript không còn đứng được trên dữ liệu canonical.** Manuscript ghi TSTR AUC 0.860 > Real-to-Real 0.774 ("denoising hypothesis"). Trên canonical r32 (A4), mọi nhánh train-chỉ-trên-synthetic đều **thấp hơn** real-only: tốt nhất 0.632 so với 0.698. Toàn bộ nhánh augmentation (A2, A5, A3) cũng không nhánh nào vượt real-only quá 1 sd.
- **Không thể áp thẳng GVAE đã train lên I3LUNG.** `cb.csv` của I3LUNG không có TMB và albumin (báo cáo 2026-08-06 ghi "≈ 1:1" là sai); chỉ **3/32** tên feature radiomics của project trùng với 128 cột pyradiomics của I3LUNG; pathology của I3LUNG là embedding foundation-model, không phải GLCM. External validation phải thiết kế lại (xem §5).
- **Baseline đơn giản đang ngang hoặc hơn GVAE** (C2, 2026-10-04): logistic regression chỉ dùng clinical đạt ROC-AUC 0.707 ± 0.016 so với GVAE head 0.664 ± 0.016 trên cùng split; không có ý nghĩa thống kê sau Holm. Mọi kết quả Loan báo cáo nên đặt cạnh baseline tương ứng.

**Ước lượng:** phần của Loan còn khoảng **24–35 ngày công** (5–7 tuần nếu làm toàn thời gian). Toàn dự án hiện đạt khoảng **25–35% theo khối lượng công việc**; chi tiết ở §8.

---

## 1. Nguyên tắc làm việc để không bị block

### 1.1 Git

- Clone `https://github.com/Kyanh56709/Diff-GVAE` (owner thêm Loan làm collaborator). **Repo đang public** — mọi thứ push lên đều công khai.
- Mỗi task một nhánh từ `main`: `loan/<task-id>-<mô-tả-ngắn>`, ví dụ `loan/L2-smote-jitter-controls`.
- Mở Pull Request vào `main`; owner review và merge. **Không push thẳng `main`, không force-push nhánh đã chia sẻ, không viết lại lịch sử.**
- Commit nhỏ, message theo kiểu repo: `feat: ...`, `fix: ...`, `docs: ...`.

### 1.2 Thư mục làm việc

- Mỗi task một thư mục: `research/YYYY-MM-DD-loan-<chủ-đề>/{scripts,output,reports}` (theo quy ước các thư mục `research/2026-10-0x-*` hiện có).
- `output/` **không vào git**: repo ignore `*.csv`, `*.json`, `logs/`, `research/**/oof_arrays.npz`. Số liệu cuối cùng phải nằm trong `reports/*.md` (đây là quy ước của repo).
- Dữ liệu ngoài đặt ở `data/external/<tên-dataset>/` — thư mục này đã được thêm vào `.gitignore`. **Tuyệt đối không commit dữ liệu thô của I3LUNG** (license CC-BY-NC-4.0, repo public) hay bất kỳ file per-patient nào (nhãn, điểm dự đoán).

### 1.3 Code lõi — chỉ đọc, trừ khi qua PR riêng

Các đường dẫn sau là "lõi" của owner: `training/`, `models/`, `utils/`, `review_fixes_2026_07/`, `data/build_ln_pc_ihc_g.py`, `outputs/gvae/*.py`.

- Cần hành vi mới trong lõi? **Trước tiên chép logic vào script trong `research/...loan.../scripts/`** để không phải chờ ai; sau đó (nếu muốn) mở PR riêng thêm **flag opt-in, mặc định giữ nguyên hành vi cũ, kèm test** — đúng quy ước "New Feature Flags (v2)" trong `CLAUDE.md`.
- Không sửa hay xóa test hiện có để cho pass.

### 1.4 Artifact đã chốt — chỉ đọc

| Artifact | Đường dẫn | Ghi chú |
|---|---|---|
| Graph canonical | `data_ln_pc_ihc_g_r32.pt` | sha256 `2d5c7719368e9a489fce6eb58b56f6eb097d5f40b742922db6ca063223358c02`; 247 BN, lesion 32 chiều |
| GVAE nguồn latent cho DDPM | `outputs/gvae/checkpoints/gvae_bestparam_ranked_r32_20261003_124706/` (5 file `*_fold_{1..5}_rank_1_*.pt`, ~27 MB/file) | sha256 ở §2.2 |
| DDPM A2 (canonical) | `outputs/conditional_latent_ddpm/conditional_latent_ddpm_from_gvae_bestparam_ranked_r32_20261003_124706_20261003_125733/` | 49 MB; chứa `real_latents_for_ddpm.pt` và `generated/<mode>/ratio_<r>/*.pt` theo fold |
| DDPM A5 (PCA 32) | `..._20261003_130455/` | 26 MB |
| GVAE pooled-OOF 5 seed | `research/2026-10-03-radiology-artifact-ablation/output/drop_both32_seed{42..46}/oof_arrays.npz` | không trong git; owner gửi nếu cần |

**Quy tắc:** không ghi đè các thư mục trên. Run mới luôn có run id mới. Khi gọi runner DDPM **luôn truyền `--gvae-run-id gvae_bestparam_ranked_r32_20261003_124706` tường minh** — run `gvae_bestparam_ranked_r32_20261003_133903` mới hơn nhưng chỉ dùng để kiểm tra tên file, không dùng cho số liệu (pipeline GVAE không deterministic hoàn toàn).

### 1.5 Chuẩn báo cáo (bắt buộc cho mọi task)

- Viết bằng tiếng Việt, có các mục: Câu hỏi → Protocol → Kết quả (bảng) → **Kết luận trung thực** → Giới hạn → Tái lập (lệnh chạy chính xác).
- Ghi run id, seed, commit cho mọi con số.
- **Chỉ so sánh trong cùng một run / cùng split.** Không đặt số DDPM (val-fold mean, downstream LR trên `concat_mu`) cạnh số GVAE pooled-OOF để kết luận mô hình nào tốt hơn.
- **DDPM không phải classifier** — không bao giờ dùng loss/score của DDPM làm xác suất dự đoán (`CLAUDE.md`; đường code cũ đã cách ly vào `deprecated/`).
- Không tuyên bố "có cải thiện" nếu chênh lệch nhỏ hơn 1 sd giữa các fold hoặc CI chứa 0; nói rõ "không phân biệt được".
- `.venv/bin/python -m pytest` phải pass trước khi mở PR (hiện tại: 106 passed).

### 1.6 Khi cần owner quyết định

Mỗi điểm cần quyết định trong file này đều có **mặc định**. Loan hỏi owner qua comment trong PR hoặc issue; **nếu sau 2 ngày làm việc chưa có trả lời thì làm theo mặc định** và ghi rõ trong báo cáo là đã dùng mặc định. Danh sách đầy đủ ở §7.

---

## 2. Bước 0 — Bàn giao và cài đặt (L0, ~0.5 ngày)

### 2.1 Owner gửi cho Loan (qua kênh riêng, không qua GitHub)

| File | Kích thước | Bắt buộc? |
|---|---|---|
| `data_ln_pc_ihc_g_r32.pt` | 158 KB | Bắt buộc (hoặc 3 CSV thô bên dưới để tự build) |
| `data/clinical_features_tmb.csv`, `data/glcm_features.csv`, `data/radiology_features.csv` | — | Chỉ khi muốn tự build lại graph |
| 5 checkpoint rank-1 của `gvae_bestparam_ranked_r32_20261003_124706` | ~135 MB | Bắt buộc cho Track Diffusion |
| 2 thư mục DDPM A2, A5 | 75 MB | Tùy chọn (tái tạo được trong ~7 phút/run) |
| `drop_both32_seed{42..46}/oof_arrays.npz` | nhỏ | Chỉ khi cần so với GVAE pooled-OOF |

### 2.2 Kiểm tra checksum sau khi nhận

```bash
shasum -a 256 data_ln_pc_ihc_g_r32.pt
# 2d5c7719368e9a489fce6eb58b56f6eb097d5f40b742922db6ca063223358c02
cd outputs/gvae/checkpoints/gvae_bestparam_ranked_r32_20261003_124706
shasum -a 256 gvae_bestparam_ranked_r32_20261003_124706_fold_*_rank_1_*.pt
# 5cec5e2dcfa21180aafae182e9c70851b4fbe869b7c3343ab8313d15f87dd69f  ..._fold_1_rank_1_latent_quality_0.5370.pt
# f66397b516406571f007bce45e6dc0303e966284c52418f20dbfedc9c87290e7  ..._fold_2_rank_1_latent_quality_0.5989.pt
# b11723639626c1d56e2ffead97a6debc6b662d8e2551eee8dc907787e2180345  ..._fold_3_rank_1_latent_quality_0.5509.pt
# 4da288155bc527e6f72f23e9ee1295c73f85970b88a636d3b4019bf16c09e920  ..._fold_4_rank_1_latent_quality_0.5638.pt
# c0d40494d0b3298c0d8fca0252583323d4b793a072f1e18872a290c071a3b239  ..._fold_5_rank_1_latent_quality_0.5584.pt
```

Nếu tự build graph thay vì nhận file:

```bash
.venv/bin/python data/build_ln_pc_ihc_g.py --out data_ln_pc_ihc_g_r32.pt --drop-radiology-artifacts both
.venv/bin/python review_fixes_2026_07/data_validation.py data_ln_pc_ihc_g_r32.pt   # phải in RESULT: PASS
```

### 2.3 Môi trường

- Python 3.9.6; cài theo `requirements.txt` (đã pin: torch 2.8.0, torch_geometric 2.6.1, torch_scatter 2.1.2 — torch_scatter cài từ wheel index của PyG như ghi trong file).
- Trên macOS, numpy 2.0.2 + Accelerate có thể in warning giả `overflow/invalid value encountered in matmul`; đã kiểm chứng là không làm sai kết quả. Có thể chạy với `-W ignore::RuntimeWarning`, nhưng script phải assert không có NaN.
- Không thêm dependency mới vào `requirements.txt` mà không hỏi owner (ví dụ: không có `xgboost`, `imbalanced-learn` — tự cài đặt SMOTE bằng numpy, xem L2).

**Tiêu chí xong L0:** checksum khớp; `pytest` pass; chạy lại được A4 trên run A2 và ra đúng `REAL_ONLY ROC 0.6982`:

```bash
.venv/bin/python research/2026-08-06-project-review-audit/scripts/a4_tstr_control.py \
  --run-id conditional_latent_ddpm_from_gvae_bestparam_ranked_r32_20261003_124706_20261003_125733
```

---

## 3. Hiện trạng mảng Diffusion (để Loan nắm, không phải làm lại)

**Pipeline đúng:** GVAE → trích `concat_mu = [clinical_mu, pathology_mu, radiology_mu]` theo từng fold → DDPM có điều kiện theo nhãn học `p(concat_mu | class)` **chỉ trên train fold** → sinh latent tổng hợp → train downstream logistic regression trên real (+ synthetic) → đánh giá trên val fold real. Entry point: `training/latent_ddpm_augmentation.py::run_conditional_latent_augmentation_pipeline`; runner CLI: `outputs/gvae/train_conditional_ddpm_augmentation_runner.py`; chuỗi canonical: `research/2026-10-03-canonical-r32/scripts/run_ddpm_r32.sh`.

**Config chốt:** 600 epoch, 250 timesteps, guidance 1.0, filter kNN quantile 0.95, modes `minority_only` / `nonresponder_only` / `both_classes`, ratio 0.25/0.5/1.0/2.0, seed 42, CPU (~7 phút/run).

**Kết quả canonical r32** (nguồn: `research/2026-10-03-canonical-r32/reports/final_results_package_r32.md`):

| Thí nghiệm | real_only | Nhánh tốt nhất | Kết luận |
|---|---|---|---|
| A2 (full) | 0.6982 ± 0.0359 | `both_classes` r1 filtered 0.7055 ± 0.0461 | Δ +0.007 < 1 sd |
| A5 (PCA 32) | 0.7075 ± 0.0271 | `minority_only` r1 filtered 0.7113 ± 0.0312 | Δ +0.004 < 1 sd |
| A3 (retune filter) | 0.6982 | q0.97 `both_classes` r1 0.7064 | Δ +0.008 < 1 sd |
| A4 (TSTR) | 0.6982 | `both_classes` r0.5 0.6324 | Synthetic-only **thấp hơn** real |

**Đã có:** MMD và coverage cho từng nhánh; near-duplicate = 0.0 (bản 34 chiều); code UMAP/t-SNE/PCA (`training/latent_ddpm_augmentation.py` khoảng dòng 812–837).

**Một lỗ hổng phương pháp Loan cần biết (lý do có task E2):** "val fold" mà downstream DDPM được đánh giá chính là val fold mà `kfold_train_gvae` đã dùng khi chọn checkpoint GVAE (`checkpoint_metric = latent_quality` có dùng độ ổn định variance train/val). Nghĩa là tập đánh giá DDPM **không hoàn toàn held-out**. Protocol pooled-OOF của GVAE (`kfold_evaluate_gvae_classifier`, `training/train_gvae.py:2034`) thì sạch: outer fold chỉ để báo cáo, inner-val dùng để chọn.

**Claim trong manuscript cần hòa giải** (trích từ `EJC_Research_Paper.pdf`):

| Mục manuscript | Claim | Trạng thái so với canonical |
|---|---|---|
| §4.1.4 Reconstruction Fidelity ("Ket qua cua em Loan, se check lai sau") | binary acc 93.11%, F1 80.76%, MSE 0.066/0.103, Frobenius 3.156 (pathology), radiology per-view AUC 0.785, MMD 0.035 | Chưa có script/run id trong repo → chưa kiểm chứng được |
| §4.2.1 Distribution Fidelity | discriminator XGBoost real vs synthetic AUC 0.580 | Chưa có trên canonical |
| §4.2.2 Biological Consistency | centroid shift cosine "= ...." (để trống); Table 2 tần suất binary đã decode (STK11 0.027/0.224, EGFR 0.001/0.120…) | Placeholder + chưa có trên canonical |
| §4.2.3 Functional Utility | TSTR 0.8603 > R2R 0.7744; TRTS 0.7097 | **Mâu thuẫn** với A4 canonical (TSTR 0.632 < 0.698) |
| §4.3 In-silico Gene–TMB | sinh dữ liệu có điều kiện theo gene | DDPM hiện tại chỉ có điều kiện theo **nhãn response**, không theo gene → chưa có code tương ứng trong pipeline chính |

---

## 4. Track A — Diffusion: danh sách task

Thứ tự đề xuất: L1 → L2 → (L3, L4, L5 song song) → L6 → L7 → L8. L1–L5 chỉ cần artifact đã bàn giao; L6 tự chứa (không chờ owner).

### L1. DDPM multi-seed (phần DDPM của B3) — 1–1.5 ngày

- **Mục tiêu:** biết kết quả A2/A5 có ổn định theo seed của DDPM không.
- **Cách làm:** chạy lại runner với `--seed 43 44 45 46` (giữ nguyên `COMMON` trong `run_ddpm_r32.sh`, giữ nguyên GVAE run id), cho cả A2 và A5. Tổng 8 run × ~7 phút.
- **Output:** bảng mỗi branch: ROC-AUC mean ± sd qua 5 seed DDPM; ΔAUC so với real_only **trong cùng seed**; số seed mà branch thắng real_only.
- **Xong khi:** báo cáo `research/2026-10-xx-loan-ddpm-multiseed/reports/*.md` có đủ 5 seed × 2 cấu hình, kèm danh sách run id. Lưu ý real_only không phụ thuộc seed DDPM — nếu nó thay đổi theo seed thì đó là bug cần báo.

### L2. E1 — Đối chứng SMOTE / Gaussian jitter / không augmentation (make-or-break) — 2–3 ngày

- **Mục tiêu:** trả lời "DDPM có tốt hơn các cách augmentation rẻ tiền không?". Nếu không, bỏ khung "denoising effect" khỏi bài.
- **Cách làm:**
  1. Đọc `real_latents_for_ddpm.pt` của từng fold (`splits['train'|'val']['concat_mu'|'labels']` — xem `load_real` trong `research/2026-08-06-project-review-audit/scripts/a4_tstr_control.py`).
  2. Cài đặt (trong script của Loan, numpy thuần, có test):
     - **SMOTE:** với mỗi mẫu sinh, chọn ngẫu nhiên một mẫu train cùng lớp, chọn 1 trong k=5 láng giềng gần nhất cùng lớp, nội suy tuyến tính với hệ số U(0,1).
     - **Gaussian jitter:** mẫu train cùng lớp + nhiễu N(0, (α·σ_feature)²), σ tính trên train fold, α ∈ {0.1, 0.25, 0.5} (chọn α trên inner split của train fold, không dùng val).
  3. Dùng **đúng** modes/ratios như DDPM (số mẫu sinh bằng số mẫu DDPM sinh ở cùng ô), **đúng** downstream (`train_downstream_classifier` từ `training/latent_ddpm_augmentation.py`, config như A4: logistic regression, `class_weight='balanced'`, `random_state=42`), đúng fold.
  4. 5 seed cho bộ sinh (42–46). So DDPM (từ L1) với SMOTE và jitter theo từng ô (mode × ratio).
- **Quy tắc quyết định (ghi trước khi chạy):** nếu DDPM không hơn SMOTE **và** jitter quá 1 sd ở ô tốt nhất của DDPM → khuyến nghị bỏ claim "denoising effect" và chuyển DDPM thành phân tích phụ (xem Q2 ở §7).
- **Xong khi:** bảng so sánh 3 phương pháp + no-aug, mean ± sd 5 seed, kèm kết luận theo quy tắc trên.

### L3. E3 — Độ trung thực phân phối — 1.5–2 ngày

- **Mục tiêu:** thay số "discriminator AUC 0.580" của manuscript bằng bộ kiểm định có p-value trên canonical.
- **Cách làm (theo fold, trên latent train real vs latent sinh cùng fold):**
  - MMD (RBF, bandwidth theo median heuristic) + **permutation test** 1000 lần → p-value.
  - KS từng feature + hiệu chỉnh Benjamini–Hochberg → số feature khác biệt có ý nghĩa.
  - Khoảng cách Frobenius giữa ma trận tương quan real và synthetic, kèm **null** từ việc chia đôi ngẫu nhiên tập real (bootstrap 200 lần).
  - **C2ST:** logistic regression và gradient boosting (sklearn `HistGradientBoostingClassifier`; không có xgboost trong venv) phân biệt real vs synthetic với 5-fold CV → AUC ± sd.
- **Xong khi:** bảng theo branch tốt nhất của A2 và A5 (ít nhất `both_classes` r1 và các nhánh filtered), kèm đối chiếu với số 0.580 cũ.

### L4. E4 — Đa dạng / ghi nhớ (memorization) — 1–1.5 ngày

- Improved precision/recall (Kynkäänniemi et al. 2019, k = 3) giữa latent real train và synthetic.
- Distance-to-closest-record: phân phối khoảng cách synthetic → real train **so với** real val → real train. Nếu synthetic gần train hơn hẳn val gần train → dấu hiệu ghi nhớ.
- Đếm near-copy (khoảng cách < quantile 1% của khoảng cách real–real) trên canonical r32.
- **Xong khi:** bảng + histogram (lưu PNG trong `output/`, chèn vào báo cáo dạng mô tả số).

### L5. E5 — Centroid-shift cosine (điền "= ...." trong §4.2.2) — 0.5–1 ngày

- Cosine giữa vector (centroid non-responder → centroid responder) của latent real và của latent synthetic, theo fold; bootstrap CI; **null** bằng cách hoán vị nhãn của synthetic (1000 lần).
- **Xong khi:** có một con số + CI + p-value thay cho placeholder, kèm câu diễn giải trung thực (nếu cosine cao nhưng đó là điều hiển nhiên do DDPM có điều kiện theo nhãn, phải nói rõ).

### L6. E2 — TSTR / augmentation trên tập held-out thật — 4–6 ngày (task lớn nhất, tự chứa)

- **Mục tiêu:** loại bỏ khả năng "val fold đã chạm vào chọn checkpoint GVAE" làm sai kết quả DDPM (§3).
- **Thiết kế:** dùng đúng công thức split của pooled-OOF (`StratifiedKFold(5, shuffle=True, random_state=seed)` outer; `train_test_split(0.2, stratify, random_state=4200)` inner; seed 42–46):
  1. Với mỗi outer fold: train GVAE trên inner-train, chọn checkpoint theo inner-val AUC (giống `kfold_evaluate_gvae_classifier`).
  2. Trích `concat_mu` cho inner-train, inner-val, outer-test bằng GVAE đó (message passing chỉ trong subgraph tương ứng — xem A5 trong checklist; dùng `utils/latent_extraction.py`).
  3. Train DDPM trên latent inner-train; sinh; train downstream trên inner-train (+ synthetic); **đánh giá trên outer-test**.
  4. Gộp OOF trên 247 BN như pooled-OOF; bootstrap CI; DeLong so augmentation vs real-only bằng `review_fixes_2026_07/delong.py` (hàm `delong_paired_test`).
- **Không bị block:** `kfold_evaluate_gvae_classifier` hiện không trả về model theo fold. Loan **chép vòng lặp đó vào script của mình** và giữ model lại. Nếu muốn đưa vào lõi, mở PR riêng thêm flag opt-in (ví dụ `save_fold_models_dir`, mặc định `None`) + test.
- **Chi phí tính toán:** GVAE pooled-OOF ~13 phút/seed trên CPU + DDPM ~7 phút cho mỗi lần chạy theo fold → ~1–2 giờ cho 5 seed (chạy nền).
- **Xong khi:** bảng real-only vs best augmentation vs TSTR trên outer-test, 5 seed, CI + DeLong.

### L7. Hòa giải manuscript "Stage 2" — 2–3 ngày (sau L3–L6)

- **Output bắt buộc:** một bảng trong `reports/`:

  | Mục manuscript | Claim cũ | Giá trị canonical r32 | Script + run id | Trạng thái (giữ / sửa số / bỏ) |
  |---|---|---|---|---|

- Phủ hết các dòng trong bảng claim ở §3, gồm cả §4.1.4. Với §4.1.4: nếu còn code/notebook cũ đã tạo ra các số đó, đưa vào `research/...loan.../scripts/` và chạy lại trên canonical; nếu không còn, chạy lại reconstruction metrics trên checkpoint canonical (`gvae_bestparam_ranked_r32_20261003_124706`, rank 1) và thay số.
- **Không tự sửa PDF/manuscript.** Đề xuất câu chữ thay thế trong báo cáo; owner (first author) quyết định và sửa.

### L8. E6 — Phân tích in-silico gene–TMB (§4.3) — 1 ngày

- DDPM hiện tại có điều kiện theo nhãn response, không theo gene. Sinh "có điều kiện theo STK11" đòi hỏi một mô hình điều kiện khác — **không khuyến nghị** làm lại trong phạm vi này.
- **Mặc định (Q3):** nếu không tái lập được trên canonical bằng code có trong repo → đề xuất bỏ §4.3 khỏi bài chính; nếu tái lập được → chuyển sang Supplementary, ghi rõ "hypothesis generation, không phải phát hiện y sinh", kèm các caveat đã có trong ghi chú tiếng Việt của §4.3.

---

## 5. Track B — Dataset ngoài: danh sách task

### 5.0 Hiện trạng (đã kiểm chứng 2026-10-04)

- Báo cáo tìm dataset: `research/2026-08-06-external-datasets/reports/report.md`. **I3LUNG (Zenodo 17535424)** là dataset công khai duy nhất có NSCLC + ICI + đủ 3 modality.
- File trong `I3LUNG_DATA.zip` (59 MB): `cb.csv` (clinical), `genomics.csv`, `pyradiomics.csv`, `fmrad.csv`, `digital_pathology.csv`, `digital_pathology_titan.csv`, `outcomes.csv`, `features.json`.
- N = 2,075 (tri-modal 391). Nhãn: `BEST RESPONSE` (mã 0–3: 894 / 532 / 582 / 60; 7 trống), `ORR` (0: 1,426; 1: 642), kèm DCR, CBR, PFS, OS.
- Clinical (`cb.csv`): có `ECOG PS`, `NLR` (và `NEUTROPHYL`, `LYMPHOCYTES`, `MONOCYTE`), `PDL1 CATEGORY`, smoking, histology, các vị trí di căn, `LDH`, `BMI`, `CENTER`, `SET`. **Không có TMB, không có albumin.**
- Genomics: `DRIVER`, `KRAS`, `P53`, `STK11`.
- Radiomics: 128 feature pyradiomics **cấp bệnh nhân** (tiền tố `vanilla_`); project dùng 32 feature **cấp lesion** — chỉ 3 tên trùng (`logarithm_firstorder_Range`, `..._RootMeanSquared`, `..._Mean`).
- Pathology: embedding foundation-model (846 × 771), không phải GLCM 15 chiều.
- License **CC-BY-NC-4.0**: dùng được cho paper học thuật, không dùng thương mại, không phân phối lại dữ liệu thô qua repo public.
- Bản sao local của các file evidence (5 dòng đầu mỗi CSV) nằm ở `research/2026-08-06-external-datasets/sources/raw/i3lung_evidence/` trên máy owner (không có trong git).

**Hệ quả thiết kế:** không có "external validation của GVAE đã train" theo nghĩa chặt. Có hai câu hỏi khác nhau, làm cả hai:

- **D2a — Transportability của tín hiệu clinical (external thật):** train trên MSK (247) chỉ với feature **chung** giữa hai dataset, test trên I3LUNG.
- **D2b — Replication phương pháp:** dựng lại toàn bộ pipeline trên I3LUNG với feature của chính I3LUNG, đánh giá bằng pooled-OOF nội bộ I3LUNG. Đây không phải external validation, chỉ là "phương pháp có lặp lại được trên cohort khác không".

### L9. D1 refresh — Tải I3LUNG + data card — 1–1.5 ngày

- Tải từ Zenodo record 17535424 vào `data/external/i3lung/`; ghi sha256 từng file vào báo cáo.
- Đọc tài liệu của dataset để chốt **ý nghĩa mã `BEST RESPONSE` 0–3** và cột `SET` (dataset có chia train/test sẵn không?), `CENTER` (bao nhiêu trung tâm).
- **Mapping nhãn (mặc định Q5):** CR/PR → responder → `binary_label = 0`; SD/PD → non-responder → `binary_label = 1` (đúng quy ước project: 1 = non-responder, positive class). Nếu mã không phân biệt được, dùng `ORR` (1 = có đáp ứng → label 0).
- **Data card:** N theo modality, phân bố nhãn, tỷ lệ thiếu từng cột, phân bố theo `CENTER`/`SET`.

### L10. D2a — External validation của mô hình clinical hài hòa — 2–3 ngày

- Lập bảng mapping feature MSK ↔ I3LUNG. Ứng viên:
  - ECOG (one-hot 0–3) ↔ `ECOG PS`.
  - PD-L1: MSK là điểm liên tục (`clinical_pdl1_score`, đã log1p + RobustScaler lúc build) ↔ `PDL1 CATEGORY` → đưa cả hai về cùng bậc (<1%, 1–49%, ≥50%). Cần biết thang gốc của MSK: ngược biến đổi bằng `data/clinical_scaler.pkl`, hoặc lấy từ `data/x_clinical_unscaled_ln_pc_ihc_g.csv`.
  - STK11 ↔ `STK11`; EGFR ↔ có thể suy từ `DRIVER` (cần đọc tài liệu).
  - dNLR (MSK) vs NLR (I3LUNG): **không cùng định nghĩa**; I3LUNG thiếu WBC nên không tính đúng dNLR. Mặc định: bỏ cặp này, hoặc chạy thêm một biến thể có NLR≈dNLR và ghi rõ là xấp xỉ.
- Mô hình: logistic regression (cùng họ với baseline mạnh nhất của C2) train trên toàn bộ 247 BN MSK, test trên I3LUNG. Chuẩn hóa fit trên MSK. Báo cáo AUC, PR-AUC, bootstrap CI, calibration (Brier + reliability), và kết quả theo từng `CENTER`.
- Tùy chọn: so với cùng mô hình train bằng CV nội bộ trên I3LUNG (để thấy mức suy giảm do domain shift).

### L11. D2b — Dựng lại pipeline trên I3LUNG — 5–7 ngày

- **Adapter** `research/...loan-i3lung.../scripts/build_i3lung_graph.py` (không sửa `data/build_ln_pc_ihc_g.py`): xuất `HeteroData` cùng schema với canonical:
  - `patient.x_clinical`: clinical đã mã hóa số; imputation median/mode chỉ trên feature (không đọc nhãn), giống A2 của checklist. **Không đưa cột genomics vào input** (để dành cho D3).
  - Radiology: mỗi bệnh nhân **một node lesion** mang 128 feature pyradiomics → aggregator hiện có chạy được mà không phải đổi code.
  - Pathology: embedding FM để nguyên, cho pipeline tự giảm chiều qua `pca_config` (PCA fit trên train fold) — **không PCA lúc build** (sẽ leak).
  - Cạnh similarity: cosine > 0.8 chỉ trên feature, giống canonical; ghi mật độ cạnh vào báo cáo (ngưỡng có thể cần chỉnh vì số chiều khác — xem Q6).
  - Mask modality, `binary_label`, `y` (PFS), `event`.
- Kiểm tra bằng `review_fixes_2026_07/data_validation.py` (nếu validator cứng nhắc theo N = 247 thì viết validator riêng trong script của Loan, không sửa file lõi).
- Chạy pooled-OOF GVAE 5 seed (cấu hình của `research/2026-10-03-radiology-artifact-ablation/scripts/run_oof.py`, chỉ đổi `in_channels`) + baseline (tái sử dụng logic của `research/2026-10-04-baselines-delong/scripts/run_baselines.py` với bộ dựng feature mới) + DeLong.
- Tùy chọn: chạy DDPM augmentation trên I3LUNG (protocol L6 nếu L6 đã xong).
- **Tính toán:** N = 391 (tri-modal) hoặc 2,075 (cho phép thiếu modality) → GVAE chậm hơn canonical; ước lượng 30–90 phút/seed trên CPU. Bắt đầu với tập tri-modal.

### L12. D3 — Claim "gene encoding tự phát" — 1.5–2 ngày

- **Caveat quan trọng trên MSK:** các cờ gene (EGFR, STK11…) là **input** của view clinical, nên probe được gene từ `mu` gần như chắc chắn chỉ phản ánh tái tạo input, không phải "tự phát". Phải ghi rõ điều này trong báo cáo.
- Trên I3LUNG (L11 không đưa genomics vào input): linear probe `STK11`, `KRAS` (và EGFR nếu suy được) từ `mu` của GVAE theo fold, AUC + CI, so với probe trên feature thô và với chance.
- **Xong khi:** có bảng probe AUC cho MSK (kèm caveat) và I3LUNG (thiết kế sạch).

### L13. D5 + D6 — Viết phần external validation và domain shift — 1–1.5 ngày

- Một đoạn Methods + Results, nói trung thực: "external validation cho tín hiệu clinical (D2a) và replication phương pháp (D2b); external validation đầy đủ đa phương thức của mô hình đã train là không khả thi với dữ liệu công khai hiện có".
- Domain shift: khác trung tâm (`CENTER`), khác bộ feature radiomics (lesion vs patient-level, 3/32 trùng), pathology GLCM vs FM embedding, khác tỷ lệ đáp ứng.

### L14. (Tùy chọn) Kiểm tra clinical-only trên cohort NSCLC khác — 1.5–2.5 ngày

- Hellmann / CheckMate 012 (`nsclc_mskcc_2018` trên cBioPortal, N = 75; có BEST_OVERALL_RESPONSE + DCB). D4 (TCIA): chỉ ghi lại kết quả tìm kiếm (0.5 ngày) — báo cáo 2026-08-06 cho thấy không có bộ TCIA NSCLC-ICI nào có nhãn response.

---

## 6. Ranh giới với phần của owner

| Owner làm (Loan không đụng) | Loan làm | Điểm giao |
|---|---|---|
| A2/A3/A5 (viết Methods), A6 (số notebook 0.81–0.87), B2 nested CV cho GVAE, B4, B6 | E1–E6, DDPM của B3, D2–D6 | Loan dùng split recipe pooled-OOF; owner có thể dùng code L6 cho B2 |
| C1 ablations, C2 DyAM, C3–C5 | Baseline trên I3LUNG trong L11 | Script baseline dùng chung (`run_baselines.py`) |
| F1–F3 interpretability | — | Hình UMAP của latent synthetic (L4) có thể dùng lại cho F3 |
| G (TRIPOD+AI, CLAIM, ethics…), H2, I (manuscript), J (nộp bài) | Bảng hòa giải L7, đoạn văn L13 | Owner quyết định câu chữ cuối trong manuscript |

---

## 7. Điểm cần owner quyết định (có mặc định — quá 2 ngày làm việc không trả lời thì dùng mặc định)

| ID | Câu hỏi | Mặc định |
|---|---|---|
| Q1 | Dùng I3LUNG (CC-BY-NC-4.0) cho paper? | Có (học thuật; không phân phối lại dữ liệu thô) |
| Q2 | Nếu E1 cho thấy DDPM không hơn SMOTE/jitter: bỏ khung "denoising effect"? | Có — chuyển DDPM thành phân tích phụ, giữ kết quả âm tính trung thực |
| Q3 | §4.3 gene–TMB: bỏ hay đưa vào Supplementary? | Bỏ nếu không tái lập được trên canonical; nếu tái lập được thì Supplementary kèm caveat |
| Q4 | Có thêm flag lõi `save_fold_models_dir` cho L6 không? | Chưa — Loan làm bằng script riêng; PR flag sau |
| Q5 | Mapping nhãn I3LUNG | CR/PR = responder (label 0); SD/PD = non-responder (label 1) |
| Q6 | Ngưỡng cosine cho cạnh I3LUNG | Giữ 0.8; báo mật độ cạnh; nếu graph quá thưa (<1 cạnh/BN) thì thử thêm 0.7 như phân tích nhạy |
| Q7 | Tên framework dùng trong báo cáo | "Diff-GVAE" (tên repo) cho tới khi owner chốt I3 |

---

## 8. Ước lượng: đã làm được bao nhiêu, còn bao nhiêu

### 8.1 Theo checklist (58 mục, `PUBLICATION_CHECKLIST.md`, ngày 2026-10-04)

| Nhóm | Xong | Một phần | Mở | Ghi chú |
|---|---|---|---|---|
| A. Internal validity | 4 (A1, A4, A7, A8) | 1 (A6) | 3 | A2/A3 chỉ còn viết Methods; A5 cần tài liệu hóa |
| B. Statistical rigor | 1 (B1) | 2 (B3, B5) | 3 | B2 trên thực tế đã có protocol pooled-OOF (một phần) |
| C. Experiments & baselines | 0 | 1 (C2) | 4 | C3 có sweep 10/216 tổ hợp |
| D. External validation | 0 | 0 | 6 | D1 thực chất đã xong (báo cáo 2026-08-06) nhưng chưa tick |
| E. Generative claim | 0 | 0 | 6 | E3/E4 thực chất một phần (MMD, near-dup) |
| F. Interpretability | 0 | 0 | 3 | F3 có code, chưa có hình chất lượng xuất bản |
| G. Reporting | 0 | 0 | 5 | |
| H. Reproducibility | 4 (H1, H3, H4, H6) | 1 (H5) | 1 (H2) | |
| I. Manuscript | 0 | 0 | 9 | |
| J. Submission | 0 | 0 | 4 | |
| **Tổng** | **9 (16%)** | **5 (9%)** | **44 (76%)** | Tính một phần = ½ → **~20% theo số mục** |

### 8.2 Theo khối lượng công việc (ngày công, ước lượng)

Số mục không phản ánh đúng tiến độ vì phần đã xong là phần nặng và rủi ro nhất (tái tạo dữ liệu, chống leakage, protocol đánh giá).

| Hạng mục | Đã làm (ước lượng) | Còn lại | Ai |
|---|---|---|---|
| Dữ liệu + internal validity (A1 rebuild, ablation radiology, r32, A4, A7, A8) | ~13 ngày | 2–4 ngày (Methods, A6) | Owner |
| Thống kê + baseline (B1, B3-GVAE, pooled-OOF, B5/C2) | ~7 ngày | 7–12 ngày (B2, B4, B6, C1, DyAM, C3–C5) | Owner |
| Diffusion (DDPM diagnosis A1–A5, filter, PCA, TSTR) | ~5 ngày | **14–20 ngày** (L1–L8) | **Loan** |
| External validation (tìm dataset D1) | ~2 ngày | **10–15 ngày** (L9–L13) | **Loan** |
| Interpretability (F) | ~0.5 ngày | 4–5 ngày | Owner |
| Reproducibility (H) | ~3 ngày | 1.5–2.5 ngày | Owner |
| Reporting + manuscript + nộp (G, I, J) | 0 | 13–19 ngày | Owner |
| Review PR + điều phối | — | 3–4 ngày | Owner |
| **Tổng** | **~30 ngày** | **Loan 24–35 + Owner 31–47 ≈ 55–82 ngày** | |

→ **Đã xong khoảng 25–35% theo khối lượng công việc.**

### 8.3 Chi tiết phần của Loan

| Task | Ước lượng | Phụ thuộc |
|---|---|---|
| L0 bàn giao + cài đặt | 0.5 ngày | Owner gửi file |
| L1 DDPM multi-seed | 1–1.5 | L0 |
| L2 E1 SMOTE/jitter | 2–3 | L1 (để có DDPM 5 seed) |
| L3 E3 fidelity | 1.5–2 | L0 |
| L4 E4 diversity | 1–1.5 | L0 |
| L5 E5 centroid | 0.5–1 | L0 |
| L6 E2 held-out | 4–6 | L0 (tự chứa) |
| L7 hòa giải Stage 2 | 2–3 | L3–L6 |
| L8 E6 in-silico | 1 | L7 |
| L9 I3LUNG + data card | 1–1.5 | — (song song với Track A) |
| L10 D2a harmonized external | 2–3 | L9 |
| L11 D2b pipeline trên I3LUNG | 5–7 | L9 |
| L12 D3 gene probe | 1.5–2 | L11 |
| L13 D5/D6 viết | 1–1.5 | L10–L12 |
| **Tổng bắt buộc** | **24–35 ngày** | |
| L14 tùy chọn (Hellmann, TCIA) | +1.5–2.5 | |

### 8.4 Lịch dự kiến

- **Toàn thời gian cả hai người, bắt đầu 2026-10-05:** phần của Loan xong sau khoảng 5–7 tuần (giữa đến cuối tháng 11/2026). Đường găng là phần owner (8–11 tuần, gồm viết manuscript sau khi Loan bàn giao số) → **bản sẵn sàng nộp khoảng đầu đến cuối tháng 12/2026**.
- **Bán thời gian (~50%):** nhân đôi → khoảng tháng 2–3/2027.
- Cột mốc gợi ý cho Loan:
  - **M1 (tuần 2):** L1 + L2 xong → owner chốt Q2 (giữ hay bỏ "denoising effect"). Đây là quyết định lớn nhất về câu chuyện của bài.
  - **M2 (tuần 4):** L3–L6 + L9–L10 xong.
  - **M3 (tuần 6–7):** L7, L8, L11–L13 xong; bàn giao toàn bộ bảng số + đoạn văn cho owner.

### 8.5 Rủi ro ảnh hưởng ước lượng

- Mã `BEST RESPONSE` hoặc cột `SET` của I3LUNG khác giả định → L9–L11 có thể thêm 1–2 ngày.
- GVAE trên I3LUNG (N lớn hơn, feature khác) có thể cần chỉnh siêu tham số → L11 có thể lên 8–10 ngày; giữ cấu hình canonical làm mặc định, chỉ chỉnh khi không hội tụ.
- Code tạo các số trong §4.1.4/§4.2 của manuscript đã mất → L7 phải chạy lại từ đầu (đã tính trong ước lượng cao).
- Kết quả âm tính (DDPM không hơn SMOTE; external AUC thấp) **không phải là rủi ro tiến độ** — là kết quả hợp lệ, báo cáo trung thực.

---

## 9. Hướng dẫn riêng cho AI agent của Loan

1. Đọc `CLAUDE.md`, file này, và báo cáo `research/2026-10-03-canonical-r32/reports/final_results_package_r32.md` trước khi viết code.
2. Chỉ làm trong nhánh `loan/...` và thư mục `research/YYYY-MM-DD-loan-*`. Không sửa file lõi (§1.3) trừ khi task nói rõ và đi kèm flag opt-in + test.
3. Không commit: dữ liệu thô, `*.pt` checkpoint, per-patient arrays, file trong `data/external/`. Kiểm tra `git status` trước mỗi commit; không dùng `git add -A`/`git add .`.
4. Trước khi báo "xong": chạy `.venv/bin/python -m pytest`, đối chiếu mọi con số trong báo cáo với file output thật, ghi lệnh tái lập.
5. Không suy diễn kết quả tốt hơn dữ liệu cho phép; không xóa hay làm mờ kết quả âm tính.
6. Gặp điểm quyết định không có trong §7 → dừng, ghi câu hỏi vào PR/issue cho owner, chuyển sang task khác không phụ thuộc (Track A và Track B độc lập với nhau).
