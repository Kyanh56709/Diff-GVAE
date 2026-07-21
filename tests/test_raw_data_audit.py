import pandas as pd
import torch
from review_fixes_2026_07.raw_data_audit import (
    AuditReport,
    audit_index_alignment,
    audit_row_count,
)


def test_row_count_mismatch_recorded():
    r = AuditReport()
    n = audit_row_count("clinical", pd.DataFrame({"a": [1, 2, 3]}), 5, r)
    assert n == 3
    assert any("!= expected 5" in m for m in r.mismatches)


def test_index_alignment_flags_missing_patients():
    r = AuditReport()
    df = pd.DataFrame({"main_index": [1, 2]})
    audit_index_alignment("clinical", df, torch.tensor([1, 2, 3]), r)
    assert any("absent from CSV" in m for m in r.mismatches)


def test_missing_index_column_is_opaque_not_mismatch():
    r = AuditReport()
    audit_index_alignment("glcm", pd.DataFrame({"x": [1]}), torch.tensor([1]), r)
    assert r.mismatches == []
    assert any("unauditable" in o for o in r.opaque)
