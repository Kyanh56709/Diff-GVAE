from __future__ import annotations

from pathlib import Path


def test_readme_documents_canonical_data():
    txt = Path("README.md").read_text()
    assert "data_ln_pc_ihc_g.pt" in txt
    assert "canonical" in txt.lower()
    assert "OWNER TO CONFIRM" in txt   # positive-class meaning flagged as pending
    assert "deprecated/data_247.pt" in txt
