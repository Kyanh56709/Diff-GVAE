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
CANONICAL_GRAPH = PROJECT_ROOT / "data_ln_pc_ihc_g.pt"
BUILD_SCRIPT = DATA_DIR / "build_ln_pc_ihc_g.py"

# Patient-similarity graphs use strict cosine > 0.8 (docs used to say 0.7).
SIMILARITY_THRESHOLD = 0.8

# Cohort / training defaults.
COHORT_SIZE = 247  # patients with TMB notna in clinical_features_tmb.csv
RANDOM_SEED = 42
