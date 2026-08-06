#!/usr/bin/env python
"""Build feature data dictionaries for the Diff-GVAE dataset.

Reads the three source CSVs (canonical location: <repo>/data/) and emits, into
the directory this script lives in:

  data_dictionary_clinical.csv   - 22 features of the clinical view
  data_dictionary_pathology.csv  - 137 GLCM/first-order columns (pathology view)
  data_dictionary_radiology.csv  - radiomics columns (radiology view)
  data_dictionary_summary.md     - counts, family distributions, caveats

Usage:
  /Users/admin/Diff-GVAE/.venv/bin/python build_dictionary.py [--data-dir PATH]

Notes on provenance (see summary md):
  - The graph uses 22/15/34 dims; the raw CSVs carry 22/137/1670 feature
    columns. The reductions (15-dim pathology, 34-dim radiology) happened
    upstream; no build script for them exists in the repo.
  - radiology_features.csv has 1674 columns; main_index, lesion_index,
    radiology_accession_number AND dmp_pt_id are identifiers and are dropped,
    leaving 1670 radiomics features (the task brief estimated 1671 counting
    only the first three id columns; dmp_pt_id is a patient id and is not a
    radiomics feature, so it is dropped too).
  - Semantics for clinical features are inferred from column names only;
    clinical meaning must be confirmed by the cohort owner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# --- Paths -------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/Users/admin/Diff-GVAE/data")

CLINICAL_CSV = "x_clinical_unscaled_ln_pc_ihc_g.csv"
PATHOLOGY_CSV = "glcm_features.csv"
RADIOLOGY_CSV = "radiology_features.csv"

CLINICAL_ID_COLS = ["main_index"]
PATHOLOGY_ID_COLS = ["main_index", "dmp_pt_id"]
RADIOLOGY_ID_COLS = ["main_index", "lesion_index", "radiology_accession_number", "dmp_pt_id"]

RADIOLOGY_CLASSES = {"shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"}

# --- Clinical semantics ------------------------------------------------------

SEMANTIC_RULES = [
    ("TMB", "Tumor mutational burden (mut/Mb)"),
    ("dnlr", "Derived neutrophil-to-lymphocyte ratio"),
    ("albumin", "Serum albumin"),
    ("tumor_burden", "Tumor burden measure (units/method unknown)"),
    ("clinical_pdl1_score", "PD-L1 expression score 0-100 (TPS vs CPS unknown)"),
]

HIGH_CONFIDENCE_NAMES = {
    "albumin", "TMB", "dnlr",
    "ecog_0", "ecog_1", "ecog_2", "ecog_3",
    "EGFR_driver", "ERBB2_driver", "BRAF_driver",
    "MET_driver", "STK11_driver", "ARID1A_driver",
    "io_drug_Atezolizumab", "io_drug_Durvalumab", "io_drug_Ipilimumab",
    "io_drug_Nivolumab", "io_drug_Pembrolizumab", "io_drug_Tremelimumab",
}
MEDIUM_CONFIDENCE_NAMES = {"tumor_burden", "clinical_pdl1_score", "io_drug_Resection"}


def infer_clinical_semantics(name: str) -> str:
    """Map a clinical column name to a human-readable semantics string."""
    for key, desc in SEMANTIC_RULES:
        if name == key:
            return desc
    if name.startswith("ecog_"):
        return f"ECOG performance status one-hot: {name.removeprefix('ecog_')}"
    if name.startswith("io_drug_"):
        drug = name.removeprefix("io_drug_")
        if drug == "Resection":
            return "Treatment flag: received tumor resection (surgery, not a drug)"
        return f"Treatment flag: received {drug} (immunotherapy/agent)"
    if name.endswith("_driver"):
        return f"Driver mutation flag: {name.removesuffix('_driver')}"
    return "Unknown"


def clinical_note(name: str) -> str:
    if name.startswith("ecog_"):
        return "one-hot; exactly one of ecog_0..3 is 1 per patient (verified: all 247 row sums == 1)"
    if name.endswith("_driver"):
        return "gene-level driver mutation presence; patients may carry >1 driver (max observed: 2)"
    if name == "io_drug_Resection":
        return "coded under io_drug_ prefix but denotes surgery, not a drug"
    if name == "clinical_pdl1_score":
        return "0-100 range consistent with PD-L1 TPS%; scoring method not documented in repo"
    return ""


# --- Helpers -----------------------------------------------------------------


def describe_feature(series: pd.Series) -> dict:
    """min/max/mean with NaN skipped (imaging CSVs contain sparse NaNs)."""
    return {
        "min": round(float(series.min()), 6),
        "max": round(float(series.max()), 6),
        "mean": round(float(series.mean()), 6),
    }


def nan_report(df: pd.DataFrame, label: str) -> None:
    ncols = int(df.isna().sum().gt(0).sum())
    ncells = int(df.isna().sum().sum())
    print(f"  {label}: {ncols} column(s) with NaN, {ncells} NaN cells total")
    if ncols:
        cols = df.columns[df.isna().sum() > 0].tolist()
        print(f"    NaN columns: {cols}")


# --- Clinical ----------------------------------------------------------------


def build_clinical(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / CLINICAL_CSV)
    feats = df.drop(columns=CLINICAL_ID_COLS)

    rows = []
    for name in feats.columns:
        s = feats[name]
        non_null = int(s.notna().sum())
        unique = set(s.dropna().unique())
        dtype = "binary" if unique <= {0, 1} else "continuous"
        stats = describe_feature(s)
        conf = (
            "High" if name in HIGH_CONFIDENCE_NAMES
            else "Medium" if name in MEDIUM_CONFIDENCE_NAMES
            else "Low"
        )
        rows.append({
            "name": name,
            "dtype": dtype,
            **stats,
            "non_null": non_null,
            "inferred_semantics": infer_clinical_semantics(name),
            "confidence": conf,
            "note": clinical_note(name),
        })
    out = pd.DataFrame(rows, columns=[
        "name", "dtype", "min", "max", "mean", "non_null",
        "inferred_semantics", "confidence", "note",
    ])
    assert len(out) == 22, f"clinical feature rows == {len(out)}, expected 22"
    assert len(CLINICAL_ID_COLS) == 1 and out["name"].is_unique
    return out


# --- Pathology ---------------------------------------------------------------


def build_pathology(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / PATHOLOGY_CSV)
    feats = df.drop(columns=PATHOLOGY_ID_COLS)

    rows = []
    for name in feats.columns:
        # family: original_pixels_* -> first-order; everything else is GLCM
        if name.startswith("original_pixels_"):
            family = "first_order"
        elif name.startswith("original_glcm_"):
            family = "glcm"
        else:
            raise AssertionError(f"unparseable pathology column: {name}")
        # statistic: lognormal-fit parameter columns end in lognorm_fit_pN
        if name.endswith("lognorm_fit_p0") or name.endswith("lognorm_fit_p2"):
            statistic = "_".join(name.rsplit("_", 3)[-3:])  # e.g. lognorm_fit_p0
        else:
            statistic = name.split("_")[-1]
        rows.append({"name": name, "feature_family": family, "statistic": statistic,
                     **describe_feature(feats[name]), "note": ""})

    n_features = len(rows)
    assert n_features == 137, f"pathology feature rows == {n_features}, expected 137"
    assert {r["feature_family"] for r in rows} == {"first_order", "glcm"}

    out = pd.DataFrame(rows, columns=["name", "feature_family", "statistic",
                                      "min", "max", "mean", "note"])
    summary = pd.DataFrame([{
        "name": "GRAPH_SUMMARY",
        "feature_family": "reduction",
        "statistic": "",
        "min": float("nan"), "max": float("nan"), "mean": float("nan"),
        "note": ("Graph uses a 15-dim pathology view (15 of these 137 columns). "
                 "Reduction was applied upstream; no build script for it exists in the repo."),
    }])
    return pd.concat([out, summary], ignore_index=True)


# --- Radiology ---------------------------------------------------------------


def build_radiology(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / RADIOLOGY_CSV)
    feats = df.drop(columns=RADIOLOGY_ID_COLS)

    rows = []
    for name in feats.columns:
        tokens = name.split("_")
        filt = tokens[0]                      # original | wavelet-LLL | exponential | ...
        fclass = tokens[1]                    # shape | firstorder | glcm | ...
        assert fclass in RADIOLOGY_CLASSES, f"unknown radiology class '{fclass}' in {name}"
        feature = "_".join(tokens[2:])        # e.g. Elongation, Contrast, Mean
        statistic = feature.split("_")[-1]
        rows.append({"name": name, "filter": filt, "feature_class": fclass,
                     "statistic": statistic, **describe_feature(feats[name]),
                     "note": ""})

    n_features = len(rows)
    exact_count = len(feats.columns)
    assert n_features == exact_count, (
        f"radiology feature rows == {n_features}, expected exact CSV feature count {exact_count}"
    )

    out = pd.DataFrame(rows, columns=["name", "filter", "feature_class", "statistic",
                                      "min", "max", "mean", "note"])
    summary = pd.DataFrame([{
        "name": "GRAPH_SUMMARY",
        "filter": "",
        "feature_class": "reduction",
        "statistic": "",
        "min": float("nan"), "max": float("nan"), "mean": float("nan"),
        "note": ("Graph uses a 34-dim radiology view (34 of these 1670 columns). "
                 "Reduction was applied upstream; no build script for it exists in the repo."),
    }])
    return pd.concat([out, summary], ignore_index=True)


# --- Summary markdown --------------------------------------------------------


def write_summary_md(out_dir: Path, clin: pd.DataFrame, patho: pd.DataFrame,
                     radio: pd.DataFrame, data_dir: Path) -> None:
    rfeats = radio[radio["name"] != "GRAPH_SUMMARY"]
    pfeats = patho[patho["name"] != "GRAPH_SUMMARY"]

    class_counts = rfeats.groupby("feature_class")["name"].count().sort_values(ascending=False)
    filter_counts = rfeats.groupby("filter")["name"].count().sort_values(ascending=False)
    fam_counts = pfeats.groupby("feature_family")["name"].count()
    stat_counts = pfeats.groupby("statistic")["name"].count().sort_values(ascending=False)
    conf_counts = clin.groupby("confidence")["name"].count().reindex(
        ["High", "Medium", "Low"], fill_value=0)
    dtype_counts = clin.groupby("dtype")["name"].count()

    def md_table(series: pd.Series, col1: str, col2: str) -> str:
        lines = [f"| {col1} | {col2} |", "|---|---:|"]
        for k, v in series.items():
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines)

    rows = []
    rows.append("# Diff-GVAE Feature Data Dictionary — 2026-08-06\n")
    rows.append(f"Built by `research/2026-08-06-data-dictionary/build_dictionary.py` "
                f"run with the repo venv; source CSVs read from `{data_dir}`.\n")

    rows.append("## Source files\n")
    rows.append("| Source CSV | Rows | Raw columns | Feature columns (this dictionary) | Notes |")
    rows.append("|---|---:|---:|---:|---|")
    rows.append(f"| `{CLINICAL_CSV}` | 247 | 23 | 22 | `main_index` dropped |")
    rows.append(f"| `{PATHOLOGY_CSV}` | 157 | 139 | 137 | `main_index`, `dmp_pt_id` dropped |")
    rows.append(f"| `{RADIOLOGY_CSV}` | 431 | 1674 | 1670 | `main_index`, `lesion_index`, "
                f"`radiology_accession_number`, `dmp_pt_id` dropped |")
    rows.append("")
    rows.append(f"Radiology raw column count is 1674; after dropping the four identifier "
                f"columns the exact feature count is **1670** "
                f"(the brief's estimate of 1671 counted only the three named identifiers — "
                f"`dmp_pt_id` is also an identifier, not a radiomics feature).\n")

    rows.append("## Clinical view (22 features)\n")
    rows.append(md_table(dtype_counts, "dtype", "features"))
    rows.append("")
    rows.append(md_table(conf_counts, "confidence", "features"))
    rows.append("")
    rows.append("| name | dtype | confidence | inferred_semantics |")
    rows.append("|---|---|---|---|")
    for _, r in clin.iterrows():
        rows.append(f"| {r['name']} | {r['dtype']} | {r['confidence']} | {r['inferred_semantics']} |")
    rows.append("")

    rows.append("## Pathology view (137 features)\n")
    rows.append(md_table(fam_counts, "feature_family", "features"))
    rows.append("")
    rows.append(md_table(stat_counts, "statistic", "features"))
    rows.append("")
    rows.append("`original_pixels_channel_1_lognorm_fit_p2` is a lognormal-fit parameter "
                "(not a first-order statistic) but follows the `original_pixels_*` naming, "
                "so it is classed as first-order.\n")

    rows.append("## Radiology view (1670 features)\n")
    rows.append(md_table(class_counts, "feature_class", "features"))
    rows.append("")
    rows.append("By filter:\n")
    rows.append(md_table(filter_counts, "filter", "features"))
    rows.append("")

    rows.append("## Required caveats\n")
    rows.append("> **Graph dims 22/15/34; raw CSV dims 22/137/1671; reductions upstream "
                "(no build script in repo); semantics derived from column names, clinical "
                "meaning must be confirmed by cohort owner.**")
    rows.append("")
    rows.append("Correction to the parenthetical raw-CSV figure: the radiology CSV carries "
                "**1670** feature columns once its four identifier columns (including "
                "`dmp_pt_id`) are excluded; 1671 counts only the three identifiers named in "
                "the brief. The 15-dim pathology and 34-dim radiology graph views are a "
                "subset of these columns, reduced upstream; no reduction script exists in "
                "this repo, so the mapping from raw column to graph slot is not recoverable "
                "from the repository alone.\n")
    rows.append("## Data-quality notes\n")
    rows.append("- Clinical: 0 missing values; all 17 binary columns take exactly {0,1}; "
                "ECOG is a perfect one-hot (every patient row sums to 1); patients may carry "
                "multiple driver mutations (max 2).")
    rows.append("- Pathology: all 137 feature columns are NaN-free; the only missing values "
                "in the raw CSV are 14 empty `dmp_pt_id` cells (identifier column, dropped).")
    rows.append("- Radiology: all 1670 feature columns are NaN-free; the only missing values "
                "in the raw CSV are 44 empty `dmp_pt_id` cells (identifier column, dropped).")
    rows.append("")

    (out_dir / "data_dictionary_summary.md").write_text("\n".join(rows), encoding="utf-8")


# --- Main --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="directory holding the three source CSVs")
    args = ap.parse_args()

    data_dir = args.data_dir
    for f in (CLINICAL_CSV, PATHOLOGY_CSV, RADIOLOGY_CSV):
        assert (data_dir / f).is_file(), f"missing source CSV: {data_dir / f}"

    out_dir = SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building clinical dictionary ...")
    clin = build_clinical(data_dir)
    print("building pathology dictionary ...")
    patho = build_pathology(data_dir)
    print("building radiology dictionary ...")
    radio = build_radiology(data_dir)

    clin.to_csv(out_dir / "data_dictionary_clinical.csv", index=False)
    patho.to_csv(out_dir / "data_dictionary_pathology.csv", index=False)
    radio.to_csv(out_dir / "data_dictionary_radiology.csv", index=False)
    write_summary_md(out_dir, clin, patho, radio, data_dir)

    rfeats = radio[radio["name"] != "GRAPH_SUMMARY"]
    pfeats = patho[patho["name"] != "GRAPH_SUMMARY"]
    low_conf = clin[clin["confidence"] == "Low"]["name"].tolist()

    # --- verification assertions ---------------------------------------------
    assert len(clin) == 22, "clinical rows != 22"
    assert len(pfeats) == 137, "pathology feature rows != 137"
    assert len(rfeats) == 1670, f"radiology feature rows != exact count 1670 ({len(rfeats)})"
    assert len(pfeats) + len(patho[patho["name"] == "GRAPH_SUMMARY"]) == len(patho)
    assert len(rfeats) + len(radio[radio["name"] == "GRAPH_SUMMARY"]) == len(radio)
    for f in ("data_dictionary_clinical.csv", "data_dictionary_pathology.csv",
              "data_dictionary_radiology.csv", "data_dictionary_summary.md"):
        assert (out_dir / f).is_file(), f"output missing: {f}"

    print("\n=== OUTPUTS ===")
    print(f"  {out_dir / 'data_dictionary_clinical.csv'}    ({len(clin)} rows)")
    print(f"  {out_dir / 'data_dictionary_pathology.csv'}   ({len(patho)} rows incl. 1 summary)")
    print(f"  {out_dir / 'data_dictionary_radiology.csv'}   ({len(radio)} rows incl. 1 summary)")
    print(f"  {out_dir / 'data_dictionary_summary.md'}")
    print("\n=== COUNTS ===")
    print(f"  clinical features : {len(clin)}")
    print(f"  pathology features: {len(pfeats)}  families: {dict(pfeats['feature_family'].value_counts())}")
    print(f"  radiology features: {len(rfeats)}  classes: {dict(rfeats['feature_class'].value_counts())}")
    print(f"  clinical confidence: {dict(clin['confidence'].value_counts())}")
    print(f"  Low-confidence clinical columns: {low_conf if low_conf else '(none)'}")
    nan_report(pd.read_csv(data_dir / PATHOLOGY_CSV).drop(columns=PATHOLOGY_ID_COLS), "pathology")
    nan_report(pd.read_csv(data_dir / RADIOLOGY_CSV).drop(columns=RADIOLOGY_ID_COLS), "radiology")
    print("\nAll assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
