# deprecated/

Quarantined legacy code from the DDPM-as-classifier era. **Nothing here is
part of the maintained pipeline.** These files are preserved for reference
and are not expected to run.

| File | Why it is here |
|------|----------------|
| `train_pipeline.py` | `kfold_gvae_ddpm_generative_classifier` treats per-class DDPM denoising loss as a classifier score — the design PROJECT_REVIEW flags as invalid. |
| `train_gvae_ddpm_runner.py` | Runner that sets `allow_deprecated_ddpm_classifier=True`. |
| `train_ddpm_from_gvae_checkpoints_runner.py` | Legacy loss-classifier runner, gated behind `--allow-deprecated-loss-classifier`. |
| `data_247.pt` | Non-canonical graph: 64 clinical columns and **inverted** `binary_label` vs. the canonical `data_ln_pc_ihc_g.pt`. Quarantined until its polarity is confirmed/corrected by the project owner. |

The maintained augmentation path is `training/latent_ddpm_augmentation.py`
(conditional latent DDPM, evaluated only through a downstream classifier).
