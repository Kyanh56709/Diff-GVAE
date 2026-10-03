"""Best-params ranked GVAE rerun on the canonical r32 graph (source of concat_mu for DDPM).

Config copied verbatim from research/2026-08-06-project-review-audit/scripts/pipeline_phase3_ddpm.sh
step 1; only the input graph (data_ln_pc_ihc_g_r32.pt) and the run-id prefix differ.
best_params.json (tuned on the 34-slot graph) is reused on purpose (owner decision 2026-10-03).
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from training.sweep_gvae import update_configs_with_params  # noqa: E402
from training.train_gvae import kfold_train_gvae  # noqa: E402

DATA = "data_ln_pc_ihc_g_r32.pt"
BEST = ROOT / "research/2026-08-06-project-review-audit/output/raw/best_params.json"
RUN_ID_FILE = ROOT / "research/2026-10-03-canonical-r32/output/bestparam_ranked_r32_run_id.txt"


def main():
    data = torch.load(ROOT / DATA, weights_only=False)
    dim_clinical = data["patient"].x_clinical.shape[1]
    dim_pathology = data["patient"].x_pathology.shape[1]
    dim_radiology = data["lesion"].x.shape[1]
    assert dim_radiology == 32, f"expected the 32-slot r32 graph, got lesion dim {dim_radiology}"

    best = json.load(open(BEST))
    for k in ("d_embed", "hidden_channels_vae", "attention_hidden_dim", "heads"):
        best[k] = int(best[k])  # JSON round-trip -> floats -> nn.Linear TypeError

    train_config = {
        "data_path": DATA,
        "device": torch.device("cpu"),
        "n_splits": 5, "epochs": 80, "pretrain_epochs": 100,
        "patience": 30, "patience_early_stopping": 30,
        "lr": 0.001, "wd": 1e-4, "batch_size": 64,
        "loss_weights": {"class": 1.0, "cross_cl": 0.2,
                         "rec_attr": {"clinical": 1.0, "pathology": 1.0, "radiology": 1.0},
                         "rec_struct": 0.1, "kl": 0.00001},
        "annealing": {"kl": {"start_weight": 0.0, "end_weight": 0.00001, "start_epoch": 20, "end_epoch": 300},
                      "cross_cl": {"start_weight": 0.1, "end_weight": 0.1, "start_epoch": 20, "end_epoch": 100}},
        "pca_config": {"clinical": 16, "pathology": 8},
        "lesion_pca_config": {"n_components": 15},
        "cross_cl_temp": 0.1, "grad_clip_norm": 1.0, "print_every_k_epochs": 10,
        "random_seed": 42, "vectorized_contrastive": True,
        "checkpoint_metric": "latent_quality", "early_stopping_metric": "latent_quality",
        "top_k_gvae_checkpoints": 3,
        "save_best_fold_model": True,
    }
    model_config = {
        "view_configs": {
            "clinical": {"in_channels": dim_clinical, "hidden_channels_vae": 64, "heads": 8, "dropout": 0.3, "num_gnn_layers": 2, "edge_dim": 1},
            "pathology": {"in_channels": dim_pathology, "hidden_channels_vae": 64, "heads": 8, "dropout": 0.3, "num_gnn_layers": 2, "edge_dim": 1},
            "radiology": {"in_channels": 32, "hidden_channels_vae": 64, "heads": 8, "dropout": 0.3, "num_gnn_layers": 2, "edge_dim": 1},
        },
        "radiology_aggregator_config": {"lesion_feature_dim": dim_radiology, "aggregated_output_dim": 32,
                                        "attention_hidden_dim": 64, "dropout": 0.3},
        "fusion_config": {"fused_dim": 32, "num_fusion_heads": 8, "fusion_ffn_multiplier": 5},
        "classifier_config": {"classifier_hidden_dim": 32},
        "projection_head_config": {"hidden_dim": 32, "output_dim": 32, "dropout": 0.3},
        "d_embed": 32, "missing_strategy": "learnable",
        "logvar_clamp": (-4.0, 2.0), "radiology_zero_lesion_passthrough": True,
    }
    mc, tc = update_configs_with_params(model_config, train_config, best)
    run_id = "gvae_bestparam_ranked_r32_" + time.strftime("%Y%m%d_%H%M%S")
    tc["run_id"] = run_id
    tc["checkpoint_dir"] = f"outputs/gvae/checkpoints/{run_id}"
    tc["metrics_dir"] = f"outputs/gvae/metrics/{run_id}"
    print("FINAL_CONFIG:", json.dumps({"params": best}, indent=2), flush=True)
    summary, _df, _roc = kfold_train_gvae(data, mc, tc)
    print("SUMMARY:", json.dumps(summary, indent=2, default=float), flush=True)
    json.dump({"run_id": run_id, "data_path": DATA, "best_params": best, "summary": summary},
              open(f"outputs/gvae/metrics/{run_id}/bestparam_meta.json", "w"), indent=2, default=float)
    RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUN_ID_FILE.write_text(run_id)
    print("RUN_ID:", run_id, flush=True)


if __name__ == "__main__":
    main()
