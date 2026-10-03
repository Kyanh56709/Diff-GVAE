"""Summarise the radiology-artifact ablation: per-run OOF AUCs and paired
bootstrap CIs for the AUC difference vs full34 (same seed => same folds,
same patients, so resampling patients keeps the pairing)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path(__file__).resolve().parents[1] / "output"
VARIANTS = ["full34", "drop_rank33", "drop_both32"]
N_BOOT, RNG_SEED = 2000, 4200


def load(tag, seed):
    path = OUT / f"{tag}_seed{seed}" / "oof_arrays.npz"
    return np.load(path) if path.exists() else None


def paired_delta(y, a, b, rng):
    obs = roc_auc_score(y, b) - roc_auc_score(y, a)
    deltas = []
    n = len(y)
    while len(deltas) < N_BOOT:
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return obs, lo, hi


def main():
    seeds = sorted({int(p.name.rsplit("seed", 1)[1]) for p in OUT.glob("*_seed*") if p.is_dir()})
    rows, deltas = [], []
    for seed in seeds:
        runs = {v: load(v, seed) for v in VARIANTS}
        for v, r in runs.items():
            if r is None:
                continue
            for head in ("head", "probe"):
                rows.append({"seed": seed, "variant": v, "score": head,
                             "auc": roc_auc_score(r["y_true"], r[f"{head}_probs"]),
                             "auprc": average_precision_score(r["y_true"], r[f"{head}_probs"])})
        base = runs["full34"]
        if base is None:
            continue
        for v in VARIANTS[1:]:
            if runs[v] is None:
                continue
            assert np.array_equal(base["y_true"], runs[v]["y_true"]), "OOF order differs between variants"
            for head in ("head", "probe"):
                obs, lo, hi = paired_delta(base["y_true"], base[f"{head}_probs"], runs[v][f"{head}_probs"],
                                           np.random.default_rng(RNG_SEED))
                deltas.append({"seed": seed, "variant": v, "score": head,
                               "delta_auc_vs_full34": obs, "ci_low": lo, "ci_high": hi})
    per_run = pd.DataFrame(rows)
    delta_df = pd.DataFrame(deltas)
    per_run.to_csv(OUT / "per_run_auc.csv", index=False)
    delta_df.to_csv(OUT / "paired_delta_auc.csv", index=False)
    pd.set_option("display.width", 140)
    print(per_run.pivot_table(index=["score", "variant"], columns="seed", values="auc").round(4))
    print("\nmean over seeds:")
    print(per_run.groupby(["score", "variant"])[["auc", "auprc"]].agg(["mean", "std"]).round(4))
    if len(delta_df):
        print("\npaired delta AUC vs full34 (95% bootstrap CI):")
        print(delta_df.round(4).to_string(index=False))
    json.dump({"seeds": seeds}, open(OUT / "compare_meta.json", "w"))


if __name__ == "__main__":
    main()
