"""Canonical defaults for the maintained Diff-GVAE pipeline (H1).

Nothing imports this module yet.  It exists so scripts and reports have one
authoritative place for the canonical paths and the build-time constants
(values verified 2026-10-03 by the A1 rebuild, see
``research/2026-10-03-a1-build-script/``).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Raw inputs consumed by data/build_ln_pc_ihc_g.py.
CLINICAL_CSV = DATA_DIR / "clinical_features_tmb.csv"
PATHOLOGY_CSV = DATA_DIR / "glcm_features.csv"
RADIOLOGY_CSV = DATA_DIR / "radiology_features.csv"

# Canonical graph (sole authoritative training input) and its rebuild script.
# Since 2026-10-03 the canonical graph drops the two radiology index slots
# (lesion file-rank, lesion_index): build with
#   python data/build_ln_pc_ihc_g.py --out data_ln_pc_ihc_g_r32.pt --drop-radiology-artifacts both
# The 34-slot graph stays for reproducing results frozen before that date.
CANONICAL_GRAPH = PROJECT_ROOT / "data_ln_pc_ihc_g_r32.pt"
LEGACY_GRAPH_34 = PROJECT_ROOT / "data_ln_pc_ihc_g.pt"
RADIOLOGY_LESION_DIM = 32
BUILD_SCRIPT = DATA_DIR / "build_ln_pc_ihc_g.py"

# Patient-similarity graphs use strict cosine > 0.8 (docs used to say 0.7).
SIMILARITY_THRESHOLD = 0.8

# Cohort / training defaults.
COHORT_SIZE = 247  # patients with TMB notna in clinical_features_tmb.csv
RANDOM_SEED = 42
