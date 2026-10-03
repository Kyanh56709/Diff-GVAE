#!/usr/bin/env python
"""Rebuild data_ln_pc_ihc_g.pt (PyG HeteroData) from the raw CSVs in data/.

Portable rewrite (checklist item A1) of the original Windows data-creation
pipeline.  It reads the three near-raw exports kept in the repo, applies the
preprocessing the canonical graph was built with, and writes a new HeteroData
object.  With --verify it also compares the rebuilt graph against the canonical
`data_ln_pc_ihc_g.pt` field by field and prints a delta table.

Feature mappings were reverse-engineered numerically against the canonical
graph (see research/2026-10-03-a1-build-script/); every source column listed
below reproduces its canonical slot with max |delta| <= ~1e-6 in scaled units.

Known artifact (documented, reproduced on purpose): the canonical radiology
view has 34 slots = 32 radiomics features + 2 index columns that leaked into
the feature matrix.  Slot 0 is a RobustScaler of the file-order rank of the
lesion row among the cohort-selected rows (1..333); slot 1 is a RobustScaler
of `lesion_index`.  Both are reproduced so the rebuild stays faithful;
downstream users should treat them as identifiers, not radiomics.

Usage:
    python data/build_ln_pc_ihc_g.py \
        --out data/ln_pc_ihc_g_rebuilt.pt \
        --verify data_ln_pc_ihc_g.pt
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy import log1p
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, RobustScaler

try:
    from torch_geometric.data import HeteroData
except ImportError as exc:  # pragma: no cover
    sys.exit(f"torch_geometric is required: {exc}")

# --------------------------------------------------------------------------
# Pipeline constants (verified against the canonical graph; do not change
# without re-running --verify).
# --------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.8  # cosine > threshold; the repo docs' "0.7" is stale

CLINICAL_CONTINUOUS = ["albumin", "dnlr", "TMB", "tumor_burden", "clinical_pdl1_score"]
CLINICAL_LOG1P = CLINICAL_CONTINUOUS  # all five are log1p(clip>=0) before scaling
CLINICAL_GENES = [
    "EGFR_driver",
    "ERBB2_driver",
    "BRAF_driver",
    "MET_driver",
    "STK11_driver",
    "ARID1A_driver",
]
# ALK_driver / ROS1_driver / RET_driver are constant in the cohort and absent
# from the canonical vector (those slots are all-False pre-processing).
CLINICAL_ECOG_LEVELS = [0, 1, 2, 3]

PATHOLOGY_FEATURES = [
    "original_glcm_ClusterShade_channel_1_skewness",
    "original_glcm_ClusterProminence_channel_1_kurtosis",
    "original_glcm_JointAverage_channel_1_kurtosis",
    "original_glcm_SumAverage_channel_1_kurtosis",
    "original_glcm_Idm_channel_1_kurtosis",
    "original_glcm_ClusterProminence_channel_1_variance",
    "original_glcm_Autocorrelation_channel_1_kurtosis",
    "original_glcm_ClusterShade_channel_1_variance",
    "original_glcm_JointAverage_channel_1_skewness",
    "original_glcm_SumAverage_channel_1_skewness",
    "original_glcm_Imc2_channel_1_variance",
    "original_glcm_DifferenceVariance_channel_1_skewness",
    "original_glcm_MCC_channel_1_kurtosis",
    "original_glcm_Autocorrelation_channel_1_lognorm_fit_p2",
    "original_glcm_Contrast_channel_1_kurtosis",
]

# 32 radiomics slots of the canonical 34-dim radiology view, in slot order
# (slot 0 = lesion file-rank artifact, slot 1 = lesion_index artifact).
RADIOLOGY_FEATURES = [
    "logarithm_firstorder_Range",
    "exponential_firstorder_RobustMeanAbsoluteDeviation",
    "lbp-3D-k_glcm_Autocorrelation",
    "lbp-3D-k_glcm_JointAverage",
    "lbp-3D-k_glcm_SumAverage",
    "exponential_firstorder_90Percentile",
    "exponential_firstorder_MeanAbsoluteDeviation",
    "exponential_firstorder_InterquartileRange",
    "lbp-3D-k_glcm_ClusterTendency",
    "wavelet-HLL_gldm_DependenceNonUniformityNormalized",
    "lbp-3D-m2_firstorder_90Percentile",
    "lbp-3D-k_glcm_MaximumProbability",
    "lbp-3D-k_glcm_SumSquares",
    "lbp-3D-m2_gldm_LowGrayLevelEmphasis",
    "lbp-3D-m2_firstorder_Uniformity",
    "wavelet-HLL_gldm_DependenceVariance",
    "lbp-3D-m2_ngtdm_Strength",
    "lbp-3D-k_glszm_GrayLevelNonUniformityNormalized",
    "lbp-3D-k_glszm_GrayLevelVariance",
    "lbp-3D-k_glszm_SizeZoneNonUniformityNormalized",
    "logarithm_firstorder_RootMeanSquared",
    "wavelet-HHL_glcm_InverseVariance",
    "lbp-3D-m2_gldm_HighGrayLevelEmphasis",
    "lbp-3D-k_glcm_ClusterProminence",
    "wavelet-HLH_glcm_SumEntropy",
    "logarithm_firstorder_Mean",
    "lbp-3D-m2_glcm_Imc2",
    "lbp-3D-m1_glszm_HighGrayLevelZoneEmphasis",
    "wavelet-HLH_gldm_LargeDependenceLowGrayLevelEmphasis",
    "wavelet-HLH_glcm_Imc1",
    "lbp-3D-m1_glszm_LowGrayLevelZoneEmphasis",
    "wavelet-HLH_glcm_Imc2",
]
# These six are NOT log1p-transformed in the canonical build (affine raw match).
RADIOLOGY_NO_LOG1P = {
    "wavelet-HLL_gldm_DependenceVariance",
    "logarithm_firstorder_RootMeanSquared",
    "logarithm_firstorder_Mean",
    "wavelet-HLH_gldm_LargeDependenceLowGrayLevelEmphasis",
    "wavelet-HLH_glcm_Imc1",
    "wavelet-HLH_glcm_Imc2",
}

RADIOLOGY_ARTIFACT_MODES = ("none", "file_rank", "both")

EDGE_TYPES = [
    ("patient", "similar_to_clinical", "patient"),
    ("patient", "similar_to_pathology", "patient"),
    ("patient", "has_lesion", "lesion"),
    ("patient", "similar_to_radiology", "patient"),
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _split_drugs(value):
    """Split the comma-separated io_drug string, mirroring the original code."""
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    return [drug.strip() for drug in s.split(",")]


def _sanitize(name: str) -> str:
    return name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")


def _similarity_edges(matrix: np.ndarray, threshold: float):
    """Cosine-similarity edges (both directions), no self loops, strict > t."""
    if not np.isfinite(matrix).all():
        raise ValueError("similarity matrix contains non-finite values")
    with warnings.catch_warnings():
        # macOS Accelerate BLAS emits spurious "encountered in matmul"
        # RuntimeWarnings for finite subnormal products (numpy#22487).
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, -np.inf)
    src, dst = np.where(sim > threshold)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(sim[src, dst], dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


# --------------------------------------------------------------------------
# Build steps
# --------------------------------------------------------------------------


def build_clinical(df_clin: pd.DataFrame, patient_order: list):
    """Return (matrix float64, columns, similarity edges) for the cohort."""
    cohort = df_clin.loc[patient_order]

    # 5 continuous: median impute -> clip(>=0) -> log1p -> RobustScaler
    num = cohort[CLINICAL_CONTINUOUS].astype(float).copy()
    for col in CLINICAL_CONTINUOUS:
        if num[col].isna().any():
            num[col] = num[col].fillna(num[col].median())
    num = log1p(num.clip(lower=0))
    num_scaled = RobustScaler().fit_transform(num)

    # 6 driver-gene flags (TRUE/FALSE -> 1/0, NaN -> 0)
    def _gene_flag(value):
        if pd.isna(value):
            return 0
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return int(value)
        text = str(value).strip().upper()
        if text in {"TRUE", "FALSE"}:
            return 1 if text == "TRUE" else 0
        raise ValueError(f"unrecognized driver-gene encoding: {value!r}")

    genes = cohort[CLINICAL_GENES].apply(lambda col: col.map(_gene_flag))

    # io_drug multi-label one-hot
    labels = cohort["io_drug"].apply(_split_drugs)
    mlb = MultiLabelBinarizer(sparse_output=False)
    io_values = mlb.fit_transform(labels)
    io_columns = [f"io_drug_{_sanitize(cls)}" for cls in mlb.classes_]

    # ecog one-hot
    if cohort["ecog"].isna().any():
        raise ValueError("ecog has missing values; imputation rule not reproduced")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ecog_values = encoder.fit_transform(cohort["ecog"].astype(str).to_numpy().reshape(-1, 1))
    ecog_columns = list(encoder.get_feature_names_out(["ecog"]))
    expected_ecog = [f"ecog_{level}" for level in CLINICAL_ECOG_LEVELS]
    if ecog_columns != expected_ecog:
        raise ValueError(f"unexpected ecog one-hot schema: {ecog_columns}")

    matrix = np.hstack(
        [
            num_scaled,
            genes.to_numpy(dtype=float),
            io_values.astype(float),
            ecog_values.astype(float),
        ]
    )
    columns = list(CLINICAL_CONTINUOUS) + list(CLINICAL_GENES) + io_columns + ecog_columns
    edge_index, edge_attr = _similarity_edges(matrix, SIMILARITY_THRESHOLD)
    return matrix, columns, (edge_index, edge_attr)


def build_pathology(df_glcm: pd.DataFrame, patient_order: list, num_patients: int):
    """Return (X (N,15) fp32, mask, similarity edges)."""
    glcm = df_glcm.drop(columns=["dmp_pt_id"], errors="ignore")
    # Keep original CSV row order (it defines the edge row order).
    subset = glcm[glcm.index.isin(patient_order)]
    missing = [c for c in PATHOLOGY_FEATURES if c not in subset.columns]
    if missing:
        raise KeyError(f"pathology columns missing from CSV: {missing}")
    subset = subset[PATHOLOGY_FEATURES].astype(float)

    for col in PATHOLOGY_FEATURES:  # median imputation (no-op on current data)
        if subset[col].isna().any():
            subset[col] = subset[col].fillna(subset[col].median())

    values = log1p(subset.clip(lower=0))
    scaled = RobustScaler().fit_transform(values)

    patient_to_global = {pid: i for i, pid in enumerate(patient_order)}
    X = np.zeros((num_patients, len(PATHOLOGY_FEATURES)), dtype=np.float32)
    mask = torch.zeros(num_patients, dtype=torch.bool)
    for row, pid in enumerate(subset.index):
        g = patient_to_global[pid]
        mask[g] = True
        X[g] = scaled[row]

    edge_index, edge_attr = _similarity_edges(scaled, SIMILARITY_THRESHOLD)
    local_to_global = np.array([patient_to_global[pid] for pid in subset.index], dtype=np.int64)
    edge_index = torch.tensor(local_to_global[edge_index.numpy()], dtype=torch.long)
    return X, mask, (edge_index, edge_attr)


def build_radiology(df_rad: pd.DataFrame, patient_order: list, drop_artifacts: str = "none"):
    """Return (lesion_x, radiology_mask, has_lesion edges, similarity edges).

    drop_artifacts: "none" (canonical 34 slots), "file_rank" (drop slot 0 ->
    33 slots) or "both" (drop slots 0 and 1 -> 32 radiomics slots).  Only for
    ablations; the dropped slots also leave the radiology similarity signature.
    """
    if drop_artifacts not in RADIOLOGY_ARTIFACT_MODES:
        raise ValueError(f"drop_artifacts must be one of {RADIOLOGY_ARTIFACT_MODES}")
    rad = pd.read_csv(df_rad) if isinstance(df_rad, (str, Path)) else df_rad.copy()
    rad["_file_pos"] = np.arange(len(rad))

    missing = [c for c in RADIOLOGY_FEATURES if c not in rad.columns]
    if missing:
        raise KeyError(f"radiology columns missing from CSV: {missing}")

    groups = [rad[rad["main_index"] == pid] for pid in patient_order]
    groups = [g for g in groups if len(g)]
    selected = pd.concat(groups)
    n_lesions = len(selected)
    if n_lesions == 0:
        raise ValueError("no radiology lesion rows found for the cohort")

    # metadata slots: file-order rank (1..N) and lesion_index
    file_pos = selected["_file_pos"].to_numpy()
    counter = np.empty(n_lesions, dtype=float)
    counter[np.argsort(file_pos, kind="stable")] = np.arange(1, n_lesions + 1, dtype=float)
    lesion_index = selected["lesion_index"].astype(float).to_numpy()

    # 32 radiomics features: median impute + per-column transform
    feats = selected[RADIOLOGY_FEATURES].astype(float).copy()
    for col in RADIOLOGY_FEATURES:
        if feats[col].isna().any():
            feats[col] = feats[col].fillna(feats[col].median())
    transformed = []
    for col in RADIOLOGY_FEATURES:
        v = feats[col].to_numpy()
        if col not in RADIOLOGY_NO_LOG1P:
            v = log1p(np.clip(v, 0, None))
        transformed.append(v)
    artifacts = {"none": [counter, lesion_index], "file_rank": [lesion_index], "both": []}[drop_artifacts]
    matrix = np.column_stack(artifacts + transformed)

    lesion_x = RobustScaler().fit_transform(matrix)

    # patient-level aggregation signature (mean, std, min, max) for similarity
    pid_array = selected["main_index"].to_numpy()
    patient_to_global = {pid: i for i, pid in enumerate(patient_order)}
    n_patients = len(patient_order)
    rad_mask = torch.zeros(n_patients, dtype=torch.bool)
    lesion_edge_src, lesion_edge_dst = [], []
    agg_pids, agg_rows = [], []
    for pid in patient_order:
        positions = np.where(pid_array == pid)[0]
        if positions.size == 0:
            continue
        g = patient_to_global[pid]
        rad_mask[g] = True
        block = lesion_x[positions]
        lesion_edge_src.extend([g] * positions.size)
        lesion_edge_dst.extend(positions.tolist())
        mean = block.mean(axis=0)
        std = block.std(axis=0, ddof=1) if positions.size > 1 else np.zeros(block.shape[1])
        agg_pids.append(pid)
        agg_rows.append(np.concatenate([mean, std, block.min(axis=0), block.max(axis=0)]))

    has_edge_index = torch.tensor([lesion_edge_src, lesion_edge_dst], dtype=torch.long)
    edge_index, edge_attr = _similarity_edges(np.vstack(agg_rows), SIMILARITY_THRESHOLD)
    local_to_global = np.array([patient_to_global[pid] for pid in agg_pids], dtype=np.int64)
    edge_index = torch.tensor(local_to_global[edge_index.numpy()], dtype=torch.long)

    return (
        lesion_x.astype(np.float32),
        rad_mask,
        has_edge_index,
        (edge_index, edge_attr),
        n_lesions,
    )


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def _edge_key(idx: torch.Tensor):
    return set(map(tuple, idx.t().tolist()))


def verify(rebuilt: HeteroData, canonical_path: str, tol: float = 1e-5) -> bool:
    canon = torch.load(canonical_path, weights_only=False)
    rows = []

    def check(name, ok, detail=""):
        rows.append((name, "PASS" if ok else "FAIL", detail))
        return ok

    check("main_index", list(rebuilt["patient"].main_index) == list(canon["patient"].main_index))
    check(
        "x_clinical_columns",
        list(rebuilt["patient"].x_clinical_columns) == list(canon["patient"].x_clinical_columns),
    )

    for node_key in ["x_clinical", "x_pathology"]:
        a = rebuilt["patient"][node_key]
        b = canon["patient"][node_key]
        ok = a.shape == b.shape and torch.allclose(a, b, atol=tol, rtol=0)
        detail = ""
        if a.shape == b.shape:
            detail = f"maxdiff={float((a - b).abs().max()):.3e}"
        check(f"patient.{node_key} {tuple(a.shape)}", ok, detail)

    a, b = rebuilt["lesion"].x, canon["lesion"].x
    ok = a.shape == b.shape and torch.allclose(a, b, atol=tol, rtol=0)
    check(
        f"lesion.x {tuple(a.shape)}",
        ok,
        f"maxdiff={float((a - b).abs().max()):.3e}" if a.shape == b.shape else "",
    )

    for k in ["pathology_mask", "radiology_mask"]:
        check(k, bool(torch.equal(rebuilt["patient"][k], canon["patient"][k])), f"sum={int(rebuilt['patient'][k].sum())}")

    for k in ["binary_label", "event"]:
        check(k, bool(torch.equal(rebuilt["patient"][k], canon["patient"][k])))
    ya, yb = rebuilt["patient"].y, canon["patient"].y
    ok = ya.shape == yb.shape and bool(torch.isclose(ya, yb, atol=tol, rtol=0, equal_nan=True).all())
    check("y", ok, f"nan={int(torch.isnan(ya).sum())}")

    for et in EDGE_TYPES:
        a_idx, b_idx = rebuilt[et].edge_index, canon[et].edge_index
        exact = a_idx.shape == b_idx.shape and torch.equal(a_idx, b_idx)
        set_equal = _edge_key(a_idx) == _edge_key(b_idx)
        attr_ok, attr_detail = False, ""
        if exact:
            aa, ba = rebuilt[et].get("edge_attr"), canon[et].get("edge_attr")
            if aa is None and ba is None:
                attr_ok = True
            elif aa is not None and ba is not None:
                attr_ok = torch.allclose(aa, ba, atol=tol, rtol=0)
                attr_detail = f"attr maxdiff={float((aa - ba).abs().max()):.3e}"
        detail = f"edges={a_idx.shape[1]}"
        if not exact and set_equal:
            detail += ", set-equal (order differs)"
        elif not set_equal:
            detail += f", set-diff={len(_edge_key(a_idx) ^ _edge_key(b_idx))}"
        if attr_detail:
            detail += f", {attr_detail}"
        check(f"edge {et[1]}", exact and attr_ok, detail)

    width = max(len(r[0]) for r in rows)
    print("\n--- verification vs", canonical_path, "---")
    for name, status, detail in rows:
        print(f"  {name:<{width}}  {status:<4}  {detail}")
    ok = all(r[1] == "PASS" for r in rows)
    print("VERDICT:", "ALL CHECKS PASS" if ok else "MISMATCHES PRESENT")
    return ok


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Rebuild data_ln_pc_ihc_g.pt from raw CSVs")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--clinical", default=str(root / "data/clinical_features_tmb.csv"))
    parser.add_argument("--pathology", default=str(root / "data/glcm_features.csv"))
    parser.add_argument("--radiology", default=str(root / "data/radiology_features.csv"))
    parser.add_argument("--out", default=str(root / "data/ln_pc_ihc_g_rebuilt.pt"))
    parser.add_argument("--verify", default=None, help="canonical .pt to compare against")
    parser.add_argument(
        "--drop-radiology-artifacts",
        choices=RADIOLOGY_ARTIFACT_MODES,
        default="none",
        help="ablation only: drop the lesion file-rank slot ('file_rank') or both index slots ('both')",
    )
    args = parser.parse_args()
    if args.verify and args.drop_radiology_artifacts != "none":
        parser.error("--verify compares against the canonical 34-slot graph; use it only with --drop-radiology-artifacts none")

    if args.verify and Path(args.out).resolve() == Path(args.verify).resolve():
        parser.error("--out and --verify must resolve to different files (refusing to clobber the canonical graph)")

    # ---- cohort ----
    df_clin = pd.read_csv(args.clinical, index_col="main_index")
    df_clin = df_clin[df_clin["TMB"].notna()].copy()
    patient_order = df_clin.index.tolist()
    n_patients = len(patient_order)
    print(f"cohort: {n_patients} patients (TMB notna)")

    # ---- clinical ----
    clin_matrix, clf_columns, clin_edges = build_clinical(df_clin, patient_order)
    print(f"clinical: {clin_matrix.shape[0]} x {clin_matrix.shape[1]}; edges {clin_edges[0].shape[1]}")

    # ---- pathology ----
    df_glcm = pd.read_csv(args.pathology, index_col="main_index")
    X_path, path_mask, path_edges = build_pathology(df_glcm, patient_order, n_patients)
    print(f"pathology: {X_path.shape} (mask {int(path_mask.sum())}); edges {path_edges[0].shape[1]}")

    # ---- radiology ----
    df_rad = pd.read_csv(args.radiology)
    lesion_x, rad_mask, has_edges, rad_edges, n_lesions = build_radiology(
        df_rad, patient_order, drop_artifacts=args.drop_radiology_artifacts
    )
    print(
        f"radiology: {n_lesions} lesions x {lesion_x.shape[1]} ({int(rad_mask.sum())} patients); "
        f"edges {rad_edges[0].shape[1]}"
    )

    # ---- labels ----
    cohort = df_clin.loc[patient_order]
    for col in ["label", "pfs_censor"]:  # never invent a class/event for a missing value
        if cohort[col].isna().any():
            raise ValueError(f"{col} has {int(cohort[col].isna().sum())} missing values in the cohort")
    y = torch.tensor(cohort["pfs"].to_numpy(dtype=np.float32), dtype=torch.float32)
    event = torch.tensor(cohort["pfs_censor"].to_numpy(dtype=np.int64), dtype=torch.long)
    binary_label = torch.tensor(cohort["label"].to_numpy(dtype=np.int64), dtype=torch.long)

    # ---- assemble ----
    data = HeteroData()
    data["patient"].num_nodes = n_patients
    data["patient"].main_index = patient_order
    data["patient"].x_clinical = torch.tensor(clin_matrix, dtype=torch.float32)
    data["patient"].x_clinical_columns = clf_columns
    data["patient", "similar_to_clinical", "patient"].edge_index = clin_edges[0]
    data["patient", "similar_to_clinical", "patient"].edge_attr = clin_edges[1]
    data["patient"].x_pathology = torch.tensor(X_path, dtype=torch.float32)
    data["patient"].pathology_mask = path_mask
    data["patient", "similar_to_pathology", "patient"].edge_index = path_edges[0]
    data["patient", "similar_to_pathology", "patient"].edge_attr = path_edges[1]
    data["lesion"].x = torch.tensor(lesion_x, dtype=torch.float32)
    data["lesion"].num_nodes = n_lesions
    data["patient", "has_lesion", "lesion"].edge_index = has_edges
    data["patient"].radiology_mask = rad_mask
    data["patient", "similar_to_radiology", "patient"].edge_index = rad_edges[0]
    data["patient", "similar_to_radiology", "patient"].edge_attr = rad_edges[1]
    data["patient"].y = y
    data["patient"].event = event
    data["patient"].binary_label = binary_label

    data.validate(raise_on_error=True)
    torch.save(data, args.out)
    print(f"saved: {args.out}")

    ok = True
    if args.verify:
        ok = verify(data, args.verify)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
