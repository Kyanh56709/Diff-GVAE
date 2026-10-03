# tests/test_summarize_ddpm.py
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "summarize_ddpm", ROOT / "research/2026-10-03-canonical-r32/scripts/summarize_ddpm.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

COLS = "fold,augmentation_ratio,augmentation_mode,source,filtered,pca_enabled,pca_components,roc_auc,pr_auc,f1,balanced_accuracy,precision,recall,specificity,mmd,coverage,synthetic_count"


def _csv(tmp_path):
    rows = [
        "1,0.0,none,real_only,False,False,,0.70,0.86,0.8,0.72,0.8,0.8,0.5,,,0",
        "2,0.0,none,real_only,False,False,,0.72,0.88,0.8,0.74,0.8,0.8,0.5,,,0",
        "1,0.25,both_classes,real_plus_generated,False,False,,0.71,0.87,0.8,0.71,0.8,0.8,0.5,0.04,0.03,49",
        "2,0.25,both_classes,real_plus_generated,False,False,,0.73,0.89,0.8,0.73,0.8,0.8,0.5,0.05,0.02,49",
        # filtered branch keeps 0 synthetic samples -> must show synthetic_count 0
        "1,0.25,both_classes,real_plus_filtered_generated,True,False,,0.70,0.86,0.8,0.72,0.8,0.8,0.5,,,0",
        "2,0.25,both_classes,real_plus_filtered_generated,True,False,,0.72,0.88,0.8,0.74,0.8,0.8,0.5,,,0",
    ]
    p = tmp_path / "augmentation_comparison.csv"
    p.write_text(COLS + "\n" + "\n".join(rows) + "\n")
    return p


def test_real_only_first_and_branch_names(tmp_path):
    df = mod.summarize(_csv(tmp_path))
    assert df.iloc[0]["branch"] == "real_only"
    assert set(df["branch"]) == {"real_only", "both_classes r0.25", "both_classes r0.25 (filtered)"}
    assert round(df.iloc[0]["roc_auc"], 4) == 0.71 and df.iloc[0]["n_folds"] == 2


def test_zero_kept_filtered_branch_shows_zero_synthetic(tmp_path):
    df = mod.summarize(_csv(tmp_path)).set_index("branch")
    assert df.loc["both_classes r0.25 (filtered)", "synthetic_count"] == 0
    assert df.loc["both_classes r0.25", "synthetic_count"] == 49


def test_markdown_has_synthetic_column(tmp_path):
    md = mod.to_markdown(mod.summarize(_csv(tmp_path)))
    assert md.splitlines()[0].startswith("| Branch | ROC-AUC")
    assert "synthetic" in md.splitlines()[0]
