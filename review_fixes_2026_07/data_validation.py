"""Validate a Diff-GVAE HeteroData graph before training.

Checks node/edge types, clinical feature dimensionality, continuous/binary
column layout, label polarity & class counts, and mask consistency. Run as a
script (`python data_validation.py <graph.pt>`; exit 0 pass / 1 fail) or
import `validate_hetero_graph`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

CANONICAL_CLINICAL_DIM = 22
CANONICAL_BIN_SLICE = (5, 22)          # cols 5-21 are binary; 0-4 continuous
CANONICAL_CLASS_COUNTS = {0: 62, 1: 185}
REQUIRED_EDGE_TYPES = [
    ("patient", "has_lesion", "lesion"),
    ("patient", "similar_to_clinical", "patient"),
    ("patient", "similar_to_pathology", "patient"),
    ("patient", "similar_to_radiology", "patient"),
]


@dataclass
class ValidationReport:
    ok: bool = True
    problems: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.problems.append(msg)

    def note(self, key: str, value: Any) -> None:
        self.info[key] = value


def validate_hetero_graph(
    data,
    *,
    expected_clinical_dim: int = CANONICAL_CLINICAL_DIM,
    expected_class_counts: Optional[Dict[int, int]] = CANONICAL_CLASS_COUNTS,
) -> ValidationReport:
    r = ValidationReport()

    for nt in ("patient", "lesion"):
        if nt not in data.node_types:
            r.fail(f"missing node type '{nt}'")
    if "patient" not in data.node_types:
        return r
    p = data["patient"]

    for attr in ("x_clinical", "x_pathology", "pathology_mask", "radiology_mask"):
        if attr not in p:
            r.fail(f"patient missing attribute '{attr}'")

    label = p["binary_label"] if "binary_label" in p else (p["y"] if "y" in p else None)
    if label is None:
        r.fail("patient missing label ('binary_label' or 'y')")

    npat = None
    if "x_clinical" in p:
        npat, dim = int(p.x_clinical.shape[0]), int(p.x_clinical.shape[1])
        r.note("num_patients", npat)
        r.note("clinical_dim", dim)
        if dim != expected_clinical_dim:
            r.fail(
                f"clinical_dim={dim} != expected {expected_clinical_dim} "
                f"(64 indicates the non-canonical data_247 file)"
            )
        else:
            lo, hi = CANONICAL_BIN_SLICE
            bincols = p.x_clinical[:, lo:hi]
            if not bool(torch.all((bincols == 0) | (bincols == 1))):
                sample = torch.unique(bincols)[:6].tolist()
                r.fail(f"binary clinical cols {lo}:{hi} contain non-binary values (sample {sample})")

    if label is not None:
        vals = label.long().tolist()
        counts = {int(k): int(sum(1 for v in vals if v == k)) for k in sorted(set(vals))}
        r.note("class_counts", counts)
        if len(counts) < 2:
            r.fail(f"labels are single-class: {counts}")
        if expected_class_counts is not None and counts != expected_class_counts:
            r.fail(f"class_counts {counts} != expected {expected_class_counts} (possible label-polarity inversion)")

    for m in ("pathology_mask", "radiology_mask"):
        if m in p:
            mask = p[m]
            if mask.dtype != torch.bool:
                r.fail(f"{m} dtype is {mask.dtype}, expected bool")
            if npat is not None and mask.numel() != npat:
                r.fail(f"{m} length {mask.numel()} != num_patients {npat}")
            r.note(f"{m}_true", int(mask.sum()))

    present = set(data.edge_types)
    for et in REQUIRED_EDGE_TYPES:
        if et not in present:
            r.fail(f"missing edge type {et}")

    if ("patient", "has_lesion", "lesion") in present and "lesion" in data.node_types:
        ei = data[("patient", "has_lesion", "lesion")].edge_index
        n_les = int(data["lesion"].x.shape[0]) if "x" in data["lesion"] else None
        r.note("num_lesions", n_les)
        if ei.numel():
            if npat is not None and int(ei[0].max()) >= npat:
                r.fail("has_lesion edge references patient index out of range")
            if n_les is not None and int(ei[1].max()) >= n_les:
                r.fail("has_lesion edge references lesion index out of range")

    return r


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python data_validation.py <graph.pt>")
        return 2
    data = torch.load(argv[0], map_location="cpu", weights_only=False)
    r = validate_hetero_graph(data)
    print("=== data_validation ===")
    for k, v in r.info.items():
        print(f"  {k}: {v}")
    if r.problems:
        print("PROBLEMS:")
        for msg in r.problems:
            print(f"  - {msg}")
    print("RESULT:", "PASS" if r.ok else "FAIL")
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
