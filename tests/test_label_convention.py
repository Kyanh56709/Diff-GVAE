from __future__ import annotations

from pathlib import Path


def test_readme_documents_label_convention():
    """README must document that label 1 = non-responder and label 0 = responder."""
    txt = Path("README.md").read_text()
    assert "binary_label = 1" in txt
    assert "non-responder (185/247)" in txt
    assert "binary_label = 0" in txt
    assert "responder (62/247)" in txt


def test_project_review_has_no_pending_label_confirmation():
    """The positive-class meaning is confirmed; no 'OWNER TO CONFIRM' may remain."""
    txt = Path("PROJECT_REVIEW.md").read_text()
    assert "OWNER TO CONFIRM" not in txt
