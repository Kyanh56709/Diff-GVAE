from __future__ import annotations

from pathlib import Path


def test_readme_documents_canonical_data():
    txt = Path("README.md").read_text()
    assert "data_ln_pc_ihc_g.pt" in txt
    assert "canonical" in txt.lower()
    assert "non-responder (185/247)" in txt
    assert "confirmed 2026-08-06" in txt
    assert "OWNER TO CONFIRM" not in txt   # positive-class meaning confirmed
    assert "deprecated/data_247.pt" in txt
