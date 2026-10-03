"""Third probe: is dim0 an affine function of any column (incl. excluded identifiers)?

Also test whether u=round(y0*166) is a permutation of consecutive ints and
whether it matches row position under some ordering.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

d = torch.load("data_ln_pc_ihc_g.pt", weights_only=False)
p = d["patient"]
order = list(p.main_index)
maskr = p.radiology_mask.numpy()
rad_full = pd.read_csv("data/radiology_features.csv")
rad_full["_file_pos"] = np.arange(len(rad_full))
exclude = ["main_index", "lesion_index", "radiology_accession_number", "job_tag", "dmp_pt_id"]
rows = []
for i, pid in enumerate(order):
    if maskr[i]:
        sub = rad_full[rad_full["main_index"] == pid]
        if len(sub):
            rows.append(sub)
sub = pd.concat(rows)
Y = d["lesion"].x.numpy().astype(float)
y0 = Y[:, 0]

u = np.round(y0 * 166.0).astype(int)
print("max |y0 - u/166| =", float(np.abs(y0 - u / 166.0).max()))
print("u range:", u.min(), u.max(), "unique:", len(np.unique(u)))
print("u == -166..166 permutation:", np.array_equal(np.sort(u), np.arange(-166, 167)))

# residual of y0 vs every candidate column (including excluded), as affine combos
cand_cols = [c for c in rad_full.columns if c not in ("_file_pos",)]
results = []
for c in cand_cols:
    v = sub[c]
    try:
        v = pd.to_numeric(v, errors="coerce").values.astype(float)
    except Exception:
        continue
    if np.isnan(v).any() or v.std() == 0:
        continue
    r = np.corrcoef(v, y0)[0, 1]
    results.append((abs(r), c, r))
results.sort(reverse=True)
print("top raw numeric correlations with dim0 (incl identifiers):")
for ar, c, r in results[:12]:
    print(f"  {c}: r={r:.6f}")

# residual test vs file position and graph position
for name, pos in [("file_pos", sub["_file_pos"].values.astype(float)), ("graph_pos", np.arange(len(sub), dtype=float))]:
    r = np.corrcoef(pos, y0)[0, 1]
    resid = np.abs(y0 - (np.polyval(np.polyfit(pos, y0, 1), pos)))
    print(f"{name}: r={r:.5f} maxlinresid={resid.max():.3e}")

# is u+167 a permutation rank in file order?
ranks_file = rankdata(sub["_file_pos"].values, method="ordinal")
print("u+167 == rank in file order:", np.array_equal(u + 167, ranks_file))
ranks_graph = rankdata(np.arange(len(sub)), method="ordinal")
print("u+167 == rank in graph order:", np.array_equal(u + 167, ranks_graph))

# relation between u and patient order: print u grouped by first appearance order
dfp = pd.DataFrame({"pid": sub["main_index"].values, "u": u, "li": sub["lesion_index"].values, "fpos": sub["_file_pos"].values})
print(dfp.head(25).to_string())
print("u vs lesion_index corr:", np.corrcoef(dfp["u"], dfp["li"])[0, 1])
print("u vs file_pos corr:", np.corrcoef(dfp["u"], dfp["fpos"])[0, 1])
