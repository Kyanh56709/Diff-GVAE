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
