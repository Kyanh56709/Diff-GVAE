"""Cross-check the near-raw CSVs against the canonical graph.

Audit-and-document only: it reports what CAN be reconciled (row counts,
main_index alignment, value ranges) and what remains OPAQUE (similarity-edge
thresholds, GLCM internals, in-model radiology aggregation). It does not
reconstruct or prove the full raw-to-graph pipeline.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


@dataclass
class AuditReport:
    auditable: Dict[str, Any] = field(default_factory=dict)
    opaque: List[str] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)


def audit_row_count(name: str, df: pd.DataFrame, expected: Optional[int], report: AuditReport) -> int:
    n = len(df)
    report.auditable[f"{name}_rows"] = n
    if expected is not None and n != expected:
        report.mismatches.append(f"{name}: {n} rows != expected {expected}")
    return n


def audit_index_alignment(name, df, graph_index, report, index_col: str = "main_index") -> None:
    if index_col not in df.columns:
        report.opaque.append(f"{name}: no '{index_col}' column; alignment unauditable")
        return
    if graph_index is None:
        report.opaque.append(f"{name}: graph has no patient '{index_col}'; alignment unauditable")
        return
    csv_idx = set(df[index_col].tolist())
    g_idx = set(graph_index.tolist() if hasattr(graph_index, "tolist") else graph_index)
    report.auditable[f"{name}_index_overlap"] = len(g_idx & csv_idx)
    missing = g_idx - csv_idx
    if missing:
        report.mismatches.append(f"{name}: {len(missing)} graph patient indices absent from CSV")


def audit_value_ranges(name: str, df: pd.DataFrame, report: AuditReport, max_cols: int = 8) -> None:
    num = df.select_dtypes(include=[np.number])
    ranges: Dict[str, Dict[str, float]] = {}
    for c in list(num.columns)[:max_cols]:
        col = num[c].to_numpy(dtype=float)
        ranges[c] = {
            "min": float(np.nanmin(col)),
            "max": float(np.nanmax(col)),
            "mean": float(np.nanmean(col)),
        }
    report.auditable[f"{name}_value_ranges"] = ranges


def run_raw_data_audit(
    graph_path: str,
    csv_paths: Dict[str, str],
    report: Optional[AuditReport] = None,
) -> AuditReport:
    report = report or AuditReport()
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    p = data["patient"]
    npat = int(p.x_clinical.shape[0])
    g_index = p["main_index"] if "main_index" in p else None
    report.auditable["graph_num_patients"] = npat
    report.auditable["graph_num_lesions"] = (
        int(data["lesion"].x.shape[0]) if "lesion" in data.node_types else None
    )

    expected = {"clinical_unscaled": npat}   # only the per-patient clinical CSV should match 1:1
    for name, path in csv_paths.items():
        df = pd.read_csv(path)
        audit_row_count(name, df, expected.get(name), report)
        audit_index_alignment(name, df, g_index, report)
        audit_value_ranges(name, df, report)

    report.opaque += [
        "similarity-graph edge thresholds (similar_to_* edges) are not derivable from the CSVs",
        "GLCM texture computation internals are upstream of glcm_features.csv and unauditable here",
        "radiology lesion->patient attention aggregation happens inside the model, not the CSVs",
    ]
    return report


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    graph = argv[0] if argv else "data_ln_pc_ihc_g.pt"
    csvs = {
        "clinical_unscaled": "data/x_clinical_unscaled_ln_pc_ihc_g.csv",
        "glcm": "data/glcm_features.csv",
        "radiology": "data/radiology_features.csv",
        "clinical_tmb": "data/clinical_features_tmb.csv",
    }
    rep = run_raw_data_audit(graph, csvs)
    print("=== AUDITABLE ===")
    for k, v in rep.auditable.items():
        print(f"  {k}: {v}")
    print("=== STILL OPAQUE ===")
    for o in rep.opaque:
        print(f"  - {o}")
    print("=== MISMATCHES ===")
    if rep.mismatches:
        for m in rep.mismatches:
            print(f"  - {m}")
    else:
        print("  (none)")
    return 0 if not rep.mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
