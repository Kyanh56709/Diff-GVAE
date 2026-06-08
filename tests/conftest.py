import torch
import torch_geometric
from torch_geometric.data import HeteroData
import pytest

NUM_PATIENTS = 6
CLINICAL_DIM = 22
PATHOLOGY_DIM = 10
LESION_DIM = 15
AGG_OUT_DIM = 32
D_EMBED = 64


@pytest.fixture(scope="module")
def synthetic_data():
    torch.manual_seed(0)

    data = HeteroData()

    # ── Labels: balanced [0, 1, 0, 1, 0, 1] ──
    y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    # ── Clinical features (NUM_PATIENTS x CLINICAL_DIM) ──
    x_clinical = torch.zeros(NUM_PATIENTS, CLINICAL_DIM)
    x_clinical[:, :5] = torch.randn(NUM_PATIENTS, 5)          # continuous cols 0-4
    x_clinical[:, 5:22] = (torch.rand(NUM_PATIENTS, 17) > 0.5).float()  # binary cols 5-21

    # ── Pathology features ──
    x_pathology = torch.randn(NUM_PATIENTS, PATHOLOGY_DIM)

    # ── Masks ──
    pathology_mask = torch.ones(NUM_PATIENTS, dtype=torch.bool)
    pathology_mask[0] = False  # patient 0: missing pathology

    radiology_mask = torch.ones(NUM_PATIENTS, dtype=torch.bool)
    radiology_mask[1] = False  # patient 1: missing radiology

    # ── Lesion features ──
    # Patient 0: 2 lesions, Patient 1: 1 lesion, Patient 2: ZERO lesions,
    # Patient 3: 1 lesion, Patient 4: 2 lesions, Patient 5: 1 lesion
    # (even-indexed get 2, odd-indexed get 1, except p2 which is edge case)
    lesion_patient_map = [0, 0, 1, 3, 4, 4, 5]
    n_lesions = len(lesion_patient_map)  # = 6
    x_lesion = torch.randn(n_lesions, LESION_DIM)

    # ── Patient-level node storage ──
    data['patient'].x_clinical = x_clinical
    data['patient'].x_pathology = x_pathology
    data['patient'].y = y
    data['patient'].binary_label = y
    data['patient'].pathology_mask = pathology_mask
    data['patient'].radiology_mask = radiology_mask

    # ── Lesion node storage ──
    data['lesion'].x = x_lesion
    data['lesion'].num_nodes = n_lesions

    # ── has_lesion edges (patient <-> lesion) ──
    has_lesion_edge_index = torch.stack([
        torch.tensor(lesion_patient_map, dtype=torch.long),
        torch.arange(n_lesions, dtype=torch.long)
    ], dim=0)
    data['patient', 'has_lesion', 'lesion'].edge_index = has_lesion_edge_index

    # ── similar_to_clinical edges between patients ──
    # ring: 0-1-2-3-4-5-0
    sc_edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 0]
    ], dtype=torch.long)
    sc_edge_attr = torch.randn(6, 1)
    data['patient', 'similar_to_clinical', 'patient'].edge_index = sc_edge_index
    data['patient', 'similar_to_clinical', 'patient'].edge_attr = sc_edge_attr

    # ── other relation types (minimal) ──
    # similar_to_pathology
    sp_edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 0]
    ], dtype=torch.long)
    sp_edge_attr = torch.randn(6, 1)
    data['patient', 'similar_to_pathology', 'patient'].edge_index = sp_edge_index
    data['patient', 'similar_to_pathology', 'patient'].edge_attr = sp_edge_attr

    # similar_to_radiology
    sr_edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 0]
    ], dtype=torch.long)
    sr_edge_attr = torch.randn(6, 1)
    data['patient', 'similar_to_radiology', 'patient'].edge_index = sr_edge_index
    data['patient', 'similar_to_radiology', 'patient'].edge_attr = sr_edge_attr

    return data


@pytest.fixture(scope="module")
def legacy_model_config():
    return {
        'view_configs': {
            'clinical': {'in_channels': CLINICAL_DIM, 'hidden_channels_vae': 16, 'heads': 4, 'dropout': 0.5, 'num_gnn_layers': 2, 'edge_dim': 1},
            'pathology': {'in_channels': PATHOLOGY_DIM, 'hidden_channels_vae': 16, 'heads': 4, 'dropout': 0.5, 'num_gnn_layers': 2, 'edge_dim': 1},
            'radiology': {'in_channels': AGG_OUT_DIM, 'hidden_channels_vae': 16, 'heads': 4, 'dropout': 0.5, 'num_gnn_layers': 2, 'edge_dim': 1},
        },
        'radiology_aggregator_config': {
            'lesion_feature_dim': LESION_DIM,
            'aggregated_output_dim': AGG_OUT_DIM,
            'attention_hidden_dim': 32,
            'dropout': 0.3,
        },
        'fusion_config': {'fused_dim': D_EMBED, 'num_fusion_heads': 4, 'fusion_ffn_multiplier': 4},
        'classifier_config': {'classifier_hidden_dim': 32},
        'projection_head_config': {'hidden_dim': 32, 'output_dim': D_EMBED, 'dropout': 0.5},
        'd_embed': D_EMBED,
        'missing_strategy': 'learnable',
    }


@pytest.fixture(scope="module")
def legacy_train_config():
    return {
        'device': torch.device('cpu'),
        'n_splits': 2,
        'epochs': 3,
        'patience': 30,
        'patience_early_stopping': 30,
        'lr': 1e-3,
        'wd': 1e-4,
        'loss_weights': {
            'class': 1.0, 'cross_cl': 0.2,
            'rec_attr': {'clinical': 1.0, 'pathology': 1.0, 'radiology': 1.0},
            'rec_struct': 0.1, 'kl': 1e-7,
        },
        'annealing': {
            'kl': {'start_weight': 0.0, 'end_weight': 1e-7, 'start_epoch': 1, 'end_epoch': 3},
            'cross_cl': {'start_weight': 0.1, 'end_weight': 0.1, 'start_epoch': 1, 'end_epoch': 2},
        },
        'cross_cl_temp': 0.1,
        'grad_clip_norm': 1.0,
        'print_every_k_epochs': 10,
        'random_seed': 0,
        'save_best_fold_model': False,
    }
