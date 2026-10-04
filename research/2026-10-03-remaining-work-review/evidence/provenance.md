# Evidence log — remaining-work review (2026-10-03)

Mọi lệnh chạy tại `/Users/admin/Diff-GVAE`, Python: `.venv/bin/python` (3.9.6, torch 2.8, tg 2.6.1). Output đầy đủ lưu tại phiên chạy; đây là trích yếu.

| # | Check | Command | Kết quả |
|---|---|---|---|
| E1 | Test suite | `.venv/bin/python -m pytest tests/ -rN` | `72 passed in 15.26s` |
| E2 | Không có build script | `wc -c utils/preprocessing.py` → 0; `git log --all --oneline -- utils/preprocessing.py` → chỉ initial commit | Không tồn tại script raw→graph |
| E3 | Graph canonical | `torch.load('data_ln_pc_ihc_g.pt')` | 247 patients, 333 lesions; x_clinical (247,22), x_pathology (247,15), lesion.x (333,34); masks 105/187; edges 1152/1016/652/333 |
| E4 | Clinical reconstruction | log1p(5 cột) → `clinical_scaler.pkl['scaler'].transform` vs graph | maxdiff: albumin 5.4e-08, dnlr 2.2e-07, TMB 1.1e-07, tumor_burden 5.9e-08, clinical_pdl1_score 4.0e-08 |
| E5 | Labels từ CSV | `label`↔binary_label, `pfs`↔y, `pfs_censor`↔event | Khớp tuyệt đối (boolean True toàn bộ) |
| E6 | Cohort | `clinical_features_tmb.csv`: cohort giá trị | discovery 246 + NaN 1 = 247 (trùng canonical main_index) |
| E7 | Genes | ALK/ROS1/RET_driver trong cohort | Toàn `False` → 6 gene còn lại |
| E8 | Pathology match | grid search 137 cột glcm, fit affine trên raw/log1p, 105 BN mask | 9/15 dims exact (residual ≤ 1e-7); 6 dims (0,2,3,4,6,12) không khớp (best R² 0.44–0.98) |
| E9 | Radiology match | 89 BN 1-lesion; grid search 1670 cột, raw/log1p affine | 32/34 dims maxres ≤ 1e-6; 2 dims (0,1) không khớp |
| E10 | Lesion/mask structure | `radiology_features.csv` lọc cohort | 333 lesion/187 BN; phân bố {1:89,2:60,3:28,4:10} = canonical |
| E11 | Edge threshold | cosine trên x_clinical/x_pathology cho cạnh stored | min clinical = 0.8000282; min pathology = 0.8000482; 576/508 cặp; 0 self-loop |
| E12 | A8/H1 | `ls configs/`; `wc -c configs/config.py`; `cat requirements.txt` | data_247.pt còn; config.py 0 byte; requirements không pin |
| E13 | Artifacts missing | `rg`/`find` cho smote/jitter, i3lung adapter, nested CV, shap/umap, ablation | Không có (xem report §3) |
| E14 | Seed sweep tool | `find . -name "seed_sweep*"` | `review_fixes_2026_07/seed_sweep.py` (chưa có output số cuối) |
| E15 | Manuscript placeholder | `pdftotext EJC_Research_Paper.pdf` + rg | "(Chua Sua)"×1 + "Chua sua"×1 (case-insensitive ×2), "Check lai sau"×1, "dien sau"×1, "Ket qua cua em Loan"×1; "0.806"×4, "0.867"×4 |
| E16 | OOF + runs frozen | `.npz` keys; `outputs/gvae/*` | y_true/head_probs/probe_probs (247,); 3 run dirs canonical/bestparam/ranked |
| E17 | DDPM artifacts | run `..._20260810_153849/` | summary.json + a3_filter_quantile_retune.json + a4_tstr_control.json |
| E18 | Session trước | `kilo_local_recall` read `ses_001b67fdeffevG5Umu4BnUzzAn` | Kết thúc bằng câu hỏi "làm gì tiếp"; chưa thực thi rewrite |
| E19 | Held-out-OOF code | `rg -n "outer folds|inner validation|report-only" training/train_gvae.py` | `kfold_evaluate_gvae_classifier` (dòng 2034–2058): outer folds report-only, inner-val chọn checkpoint; dùng bởi `pipeline_phase4a_oof.sh` |
| E20 | F3 code + artifacts | `rg -n "TSNE|UMAP" training/latent_ddpm_augmentation.py`; `find outputs -iname "*tsne*"` | Code dòng 812–837; artifact `projections/tsne_projection.png|csv`, `umap`, `pca` trong DDPM runs |
| E21 | Xác minh độc lập | claim-verifier (subagent `ses_f006d748affeX4x0QdYUcgy92K`) chạy lại 10 claim | 8 verified / 0 refuted / 2 partial; verdict **PARTIAL** (hiệu chỉnh B2, F3, đếm "(Chua Sua)") |
