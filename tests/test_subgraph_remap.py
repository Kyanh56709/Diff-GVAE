import torch
from utils.data_utils import get_view_subgraph_and_features


def _reference_local_edges(global_subset, sel_src, sel_dst):
    m = {g.item(): i for i, g in enumerate(global_subset)}
    return (torch.tensor([m[s.item()] for s in sel_src]),
            torch.tensor([m[d.item()] for d in sel_dst]))


def test_remap_matches_reference(synthetic_data):
    idx = torch.arange(6)
    x, ei, ea, gsub = get_view_subgraph_and_features(synthetic_data, 'clinical', idx)
    # Rebuild what the selected global edges were, then compare local mapping.
    full = synthetic_data['patient', 'similar_to_clinical', 'patient'].edge_index
    mask_s = torch.isin(full[0], gsub)
    mask_d = torch.isin(full[1], gsub)
    sel = mask_s & mask_d
    ref_src, ref_dst = _reference_local_edges(gsub, full[0][sel], full[1][sel])
    assert torch.equal(ei[0].cpu().sort().values, ref_src.sort().values)
    assert torch.equal(ei[1].cpu().sort().values, ref_dst.sort().values)
    # Edge count preserved
    assert ei.shape[1] == int(sel.sum())
