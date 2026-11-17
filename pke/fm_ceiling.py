"""Noise ceiling for interaction-residual prediction: split cells of every condition into random halves,
compute the exact residual R_ab from each half, and measure reliability on the evaluation genes (top-50 DE per double).
Spearman-Brown gives full-data reliability rho_full = 2r/(1+r); the best achievable R2-vs-additive is ~rho_full."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmcore as F, fmpredict as FP

p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--reps", type=int, default=5)
p.add_argument("--wd", default=os.environ.get("FMTRANSFER_ROOT", ".")); a = p.parse_args()
import anndata as ad, scanpy as sc, scipy.sparse as sp
S = json.load(open(os.path.join(a.run, "summary.json")))
FILES = {"Norman": "NormanWeissman2019_filtered.h5ad", "Wessels": "WesselsSatija2023.h5ad",
         "TianIPSC": "TianKampmann2019_iPSC.h5ad", "TianNeuron": "TianKampmann2019_day7neuron.h5ad"}
Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True); hn = list(Z["genes"]); conds = list(Z["conds"])
expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}
singles, doubles, _ = F.parse_conditions(conds); DE = FP.de_sets(expr, "control", doubles, np.asarray(hn), 50)
idx = np.load(os.path.join(a.run, "subsample.npz"))["idx"]
H = ad.read_h5ad(os.path.join(a.wd, "data", FILES[S["dataset"]]), backed="r"); X = H[idx].to_memory(); H.file.close(); X.var_names_make_unique()
M = sp.csr_matrix(X.X if sp.issparse(X.X) else sp.csr_matrix(X.X), dtype=np.float32); M.data = np.round(M.data); X.X = M
sc.pp.normalize_total(X, target_sum=1e4); sc.pp.log1p(X)
col = {g: i for i, g in enumerate(X.var_names)}; gcols = np.array([col[g] for g in hn])
lab = X.obs["perturbation"].astype(str).values; Xh = sp.csr_matrix(X.X)[:, gcols]
rows = {c: np.flatnonzero(lab == c) for c in conds}
rs = []
for rep in range(a.reps):
    rng = np.random.default_rng(100 + rep); h = [{}, {}]
    for c in conds:
        w = rng.permutation(rows[c]); k = len(w) // 2
        for j, part in enumerate((w[:k], w[k:])): h[j][c] = np.asarray(Xh[part].mean(0)).ravel().astype(np.float64)
    R1 = np.asarray(F.residual_exact(h[0], "control", doubles)); R2 = np.asarray(F.residual_exact(h[1], "control", doubles))
    x = np.concatenate([R1[i, DE[i]] for i in range(len(doubles))]); y = np.concatenate([R2[i, DE[i]] for i in range(len(doubles))])
    rs.append(float(np.corrcoef(x, y)[0, 1]))
r = float(np.mean(rs)); full = 2 * r / (1 + r)
out = dict(dataset=S["dataset"], seed=S["seed"], n_doubles=len(doubles), split_half_r=r, split_half_r_reps=rs, rho_full=full)
json.dump(out, open(os.path.join(a.run, "ceiling.json"), "w"), indent=1)
print(f"CEILING {S['dataset']} s{S['seed']} doubles={len(doubles)} split-half r={r:.3f} -> rho_full={full:.3f}")
