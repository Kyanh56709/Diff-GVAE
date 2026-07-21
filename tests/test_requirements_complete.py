from __future__ import annotations

from pathlib import Path


def test_requirements_list_notebook_libs():
    reqs = Path("requirements.txt").read_text().lower()
    assert "scipy" in reqs
    assert "umap-learn" in reqs
