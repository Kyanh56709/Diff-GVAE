"""Ad-hoc reverse-engineering of the raw->graph feature mappings.

Compares canonical graph views against repo raw CSVs with affine-invariant
matching under several monotone transforms. Run from repo root with .venv python.
"""
import pickle

import numpy as np
import pandas as pd
import torch
from numpy import log1p

REPO = "."
TOL = 1e-4


def zs(a):
    a = a.astype(float)
    sd = a.std(0)
    out = np.zeros_like(a)
    nz = sd > 0
    out[:, nz] = (a[:, nz] - a[:, nz].mean(0)) / sd[nz]
    return out


def affine_stats(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = float(np.abs(pred - y).max())
    ssres = float(((y - pred) ** 2).sum())
    sstot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ssres / sstot if sstot > 0 else float("nan")
    return r2, resid, coef


def match(Y, X, transforms, topk=8):
    """For each target column of Y, report top affine-matching candidate columns of X."""
    names, mats = [], []
    Xv = X.astype(float).values
    for tname, tf in transforms:
        mats.append(tf(Xv))
        names.extend([(tname, c) for c in X.columns])
    M = np.hstack(mats)
    Zm = zs(M)
    Zy = zs(Y)
    R = (Zm.T @ Zy) / len(Y)
    report = []
    for j in range(Y.shape[1]):
        idx = np.argsort(-np.abs(R[:, j]))[:topk]
        cands = []
        for i in idx:
            r2, resid, coef = affine_stats(M[:, i], Y[:, j])
            cands.append((names[i], round(r2, 8), resid))
        report.append(cands)
    return report


d = torch.load("data_ln_pc_ihc_g.pt", weights_only=False)
p = d["patient"]
order = list(p.main_index)
maskp = p.pathology_mask.numpy()
maskr = p.radiology_mask.numpy()
gcols = list(p.x_clinical_columns)
Xclin = p.x_clinical.numpy()

print("=" * 30, "CLINICAL")
clin = pd.read_csv("data/clinical_features_tmb.csv")
coh = clin[clin["TMB"].notna()].set_index("main_index")
unc = pd.read_csv("data/x_clinical_unscaled_ln_pc_ihc_g.csv", index_col="main_index")
with open("data/clinical_scaler.pkl", "rb") as fh:
    sc = pickle.load(fh)
cols5 = sc["columns"]
print("cohort", coh.shape, "unscaled", unc.shape)
for c in ["age", "pack_years", "smoking_status", "sex", "histo", "pdl1_tiss_site"]:
    if c in coh.columns:
        print(f"  dropped-col {c}: missing={coh[c].isna().sum()} nunique={coh[c].nunique()}")
raw5 = coh.loc[unc.index, cols5]
print("unscaled-vs-raw   maxdiff:", np.abs(unc[cols5] - raw5).max().to_dict())
print("unscaled-vs-log1p maxdiff:", np.abs(unc[cols5] - log1p(raw5.clip(lower=0))).max().to_dict())
scaled = sc["scaler"].transform(unc[cols5].loc[order])
sel = [gcols.index(c) for c in cols5]
print("scaled5-vs-graph maxdiff:", float(np.abs(scaled - Xclin[:, sel]).max()))
print(unc.head(2).to_string())

print("=" * 30, "PATHOLOGY")
glcm = pd.read_csv("data/glcm_features.csv", index_col="main_index").drop(
    columns=["dmp_pt_id"], errors="ignore"
)
pids = [order[i] for i in np.where(maskp)[0]]
print("glcm", glcm.shape, "path patients:", len(pids), "all present:", all(pid in glcm.index for pid in pids))
Xp = glcm.loc[pids]
Yp = p.x_pathology.numpy()[maskp]
transforms_p = [
    ("raw", lambda a: a),
    ("log1p", lambda a: log1p(np.clip(a, 0, None))),
    ("log1p2", lambda a: log1p(log1p(np.clip(a, 0, None)))),
    ("sqrt", lambda a: np.sqrt(np.clip(a, 0, None))),
]
rep = match(Yp, Xp, transforms_p, topk=6)
for j, cands in enumerate(rep):
    exact = [c for c in cands if c[1] > 0.999999 and c[2] < TOL]
    print(f"PATH dim {j:2d} | exact={bool(exact)} | top: {cands}")

print("=" * 30, "RADIOLOGY")
rad = pd.read_csv("data/radiology_features.csv")
exclude = ["main_index", "lesion_index", "radiology_accession_number", "job_tag", "dmp_pt_id"]
feat = [c for c in rad.columns if c not in exclude]
rows = []
for i, pid in enumerate(order):
    if maskr[i]:
        sub = rad[rad["main_index"] == pid]
        if len(sub):
            rows.append(sub)
Xr = pd.concat(rows)
print("lesion rows built:", len(Xr), "expected:", int(maskr.sum()))
Yr = d["lesion"].x.numpy()
print("lesion.x:", tuple(Yr.shape))
if len(Xr) == len(Yr):
    transforms_r = [
        ("raw", lambda a: a),
        ("log1p", lambda a: log1p(np.clip(a, 0, None))),
    ]
    rep = match(Yr, Xr[feat], transforms_r, topk=6)
    for j, cands in enumerate(rep):
        exact = [c for c in cands if c[1] > 0.999999 and c[2] < TOL]
        print(f"RAD dim {j:2d} | exact={bool(exact)} | top: {cands}")
else:
    print("ALIGNMENT MISMATCH - cannot match radiology by patient-order concatenation")
