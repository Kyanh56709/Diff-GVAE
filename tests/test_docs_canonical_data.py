from __future__ import annotations

from pathlib import Path


def test_readme_documents_canonical_data():
    txt = Path("README.md").read_text()
    assert "`data_ln_pc_ihc_g_r32.pt` is the sole canonical data file" in txt
    assert "--drop-radiology-artifacts both" in txt
    assert "canonical" in txt.lower()
    assert "non-responder (185/247)" in txt
    assert "confirmed 2026-08-06" in txt
    assert "OWNER TO CONFIRM" not in txt   # positive-class meaning confirmed
    assert "deprecated/data_247.pt" in txt


def test_final_report_uses_r32_numbers():
    txt = Path("FINAL_REPORT.md").read_text()
    assert "`(333, 32)`" in txt
    assert "`(333, 34)`" not in txt
    assert "data_ln_pc_ihc_g_r32.pt" in txt
    assert "0.6894" not in txt  # stale protocol number (lost artifact) must be gone


def test_final_report_has_no_stale_34_slot_results():
    txt = Path("FINAL_REPORT.md").read_text()
    # results from the lost 2026-06 artifacts must not be presented as current
    assert "GVAE run hiện hành: `gvae_latent_quality_codex_20260615_204355`" not in txt
    assert "Dữ liệu chính: `data_ln_pc_ihc_g.pt`" not in txt
    assert "outputs/best_gvae_ddpm_result.json" not in txt
    # DDPM downstream (val-fold mean) is not comparable to GVAE pooled-OOF
    assert "chưa vượt GVAE direct prediction" not in txt
    assert "không vượt GVAE direct prediction" not in txt
