"""Smoke tests for data/build_ln_pc_ihc_g.py (checklist A1).

The raw CSVs under data/ are git-ignored, so these tests skip on a fresh clone.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data" / "build_ln_pc_ihc_g.py"
CANONICAL = ROOT / "data_ln_pc_ihc_g.pt"
RAW = [ROOT / "data" / f for f in ("clinical_features_tmb.csv", "glcm_features.csv", "radiology_features.csv")]

pytestmark = pytest.mark.skipif(
    not CANONICAL.exists() or not all(p.exists() for p in RAW),
    reason="raw CSVs / canonical graph not present (data/ is git-ignored)",
)


def _build(out, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), *extra],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_rebuild_matches_canonical(tmp_path):
    proc = _build(tmp_path / "rebuilt.pt", "--verify", str(CANONICAL))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERDICT: ALL CHECKS PASS" in proc.stdout


@pytest.mark.parametrize("mode, dropped", [("file_rank", [0]), ("both", [0, 1])])
def test_artifact_ablation_only_drops_index_slots(tmp_path, mode, dropped):
    proc = _build(tmp_path / f"{mode}.pt", "--drop-radiology-artifacts", mode)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ablated = torch.load(tmp_path / f"{mode}.pt", weights_only=False)
    canon = torch.load(CANONICAL, weights_only=False)
    keep = [i for i in range(canon["lesion"].x.shape[1]) if i not in dropped]
    # RobustScaler is per-column, so the remaining slots are unchanged.
    assert torch.equal(ablated["lesion"].x, canon["lesion"].x[:, keep])
    assert torch.equal(ablated["patient"].x_clinical, canon["patient"].x_clinical)
    assert torch.equal(ablated["patient"].binary_label, canon["patient"].binary_label)


def test_verify_refuses_ablation_mode(tmp_path):
    proc = _build(tmp_path / "x.pt", "--drop-radiology-artifacts", "both", "--verify", str(CANONICAL))
    assert proc.returncode != 0
