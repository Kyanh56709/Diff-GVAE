"""Summarise a conditional-latent DDPM run (augmentation_comparison.csv) as a markdown table.

Usage: .venv/bin/python research/2026-10-03-canonical-r32/scripts/summarize_ddpm.py <run_dir>
"""
import sys
from pathlib import Path

import pandas as pd


def summarize(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    filtered = df["filtered"].astype(str).str.lower() == "true"
    df["branch"] = [
        "real_only" if src == "real_only"
        else f"{mode} r{ratio:g}" + (" (filtered)" if filt else "")
        for src, mode, ratio, filt in zip(df["source"], df["augmentation_mode"], df["augmentation_ratio"], filtered)
    ]
    out = df.groupby("branch").agg(
        roc_auc=("roc_auc", "mean"), roc_sd=("roc_auc", "std"), pr_auc=("pr_auc", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), synthetic_count=("synthetic_count", "mean"),
        coverage=("coverage", "mean"), mmd=("mmd", "mean"), n_folds=("fold", "nunique"),
    ).reset_index()
    real = out[out["branch"] == "real_only"]
    rest = out[out["branch"] != "real_only"].sort_values("roc_auc", ascending=False)
    return pd.concat([real, rest], ignore_index=True)


def _fmt(v, nd=4):
    return "—" if pd.isna(v) else f"{v:.{nd}f}"


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| Branch | ROC-AUC (±sd) | PR-AUC | BA | synthetic TB/fold | coverage | MMD | folds |",
             "|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.branch} | {_fmt(r.roc_auc)} ± {_fmt(r.roc_sd)} | {_fmt(r.pr_auc)} | "
                     f"{_fmt(r.balanced_accuracy)} | {r.synthetic_count:.1f} | {_fmt(r.coverage, 3)} | "
                     f"{_fmt(r.mmd, 3)} | {r.n_folds} |")
    return "\n".join(lines)


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    print(f"<!-- {run_dir.name} -->")
    print(to_markdown(summarize(run_dir / "augmentation_comparison.csv")))
