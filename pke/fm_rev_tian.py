"""Noise ceiling for the iPSC/neuron comparison. For each Tian screen (subsample s0), cells of every condition are split into
random halves (5 repetitions). For the doubles measured in both cell types, on shared highly variable genes:
  within   : cosine between the two half-data residuals of the same pair in the same cell type (each half centered by its own
             mean residual over doubles), i.e. how well one replicate predicts the other
  cross    : cosine between half-data residuals of the same pair in iPSC and in neurons (same centering)
Spearman-Brown is not applied to cosines; both quantities use half data and are directly comparable.
Usage: fm_rev_tian.py --wd W"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FILES = {"TianIPSC": "TianKampmann2019_iPSC.h5ad", "TianNeuron": "TianKampmann2019_day7neuron.h5ad"}


def halves(wd, ds, reps):
    import anndata as ad, scanpy as sc, scipy.sparse as sp, fmcore as F
    run = os.path.join(wd, "results", f"{ds}_s0_mc500"); Z = np.load(os.path.join(run, "pb_hvg.npz"), allow_pickle=True)
    hn = list(Z["genes"]); conds = list(Z["conds"]); singles, doubles, _ = F.parse_conditions(conds)
    idx = np.load(os.path.join(run, "subsample.npz"))["idx"]
    H = ad.read_h5ad(os.path.join(wd, "data", FILES[ds]), backed="r"); X = H[idx].to_memory(); H.file.close(); X.var_names_make_unique()
    M = sp.csr_matrix(X.X if sp.issparse(X.X) else sp.csr_matrix(X.X), dtype=np.float32); M.data = np.round(M.data); X.X = M
    sc.pp.normalize_total(X, target_sum=1e4); sc.pp.log1p(X)
    col = {g: i for i, g in enumerate(X.var_names)}; Xh = sp.csr_matrix(X.X)[:, np.array([col[g] for g in hn])]
    lab = X.obs["perturbation"].astype(str).values; rows = {c: np.flatnonzero(lab == c) for c in conds}; out = []
    for rep in range(reps):
        rng = np.random.default_rng(100 + rep); h = [{}, {}]
        for c in conds:
            w = rng.permutation(rows[c]); k = len(w) // 2
            for j, part in enumerate((w[:k], w[k:])): h[j][c] = np.asarray(Xh[part].mean(0)).ravel().astype(np.float64)
        Rs = [np.asarray(F.residual_exact(h[j], "control", doubles)) for j in (0, 1)]
        out.append([r - r.mean(0) for r in Rs])
    return hn, {frozenset((x, y)): i for i, (_, x, y) in enumerate(doubles)}, out


def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--wd", required=True); p.add_argument("--reps", type=int, default=5); a = p.parse_args()
    gA, pA, hA = halves(a.wd, "TianIPSC", a.reps); gB, pB, hB = halves(a.wd, "TianNeuron", a.reps)
    shared = sorted(set(gA) & set(gB)); ia = np.array([gA.index(g) for g in shared]); ib = np.array([gB.index(g) for g in shared])
    common = [k for k in pA if k in pB]; res = dict(n_pairs=len(common), n_genes=len(shared))
    W_A, W_B, X_ = [], [], []
    for rep in range(a.reps):
        for k in common:
            i, j = pA[k], pB[k]
            W_A.append(cos(hA[rep][0][i, ia], hA[rep][1][i, ia])); W_B.append(cos(hB[rep][0][j, ib], hB[rep][1][j, ib]))
            X_.append(np.mean([cos(hA[rep][s][i, ia], hB[rep][t][j, ib]) for s in (0, 1) for t in (0, 1)]))
    res.update(within_ipsc_median=float(np.median(W_A)), within_neuron_median=float(np.median(W_B)), cross_median=float(np.median(X_)),
               within_ipsc_mean=float(np.mean(W_A)), within_neuron_mean=float(np.mean(W_B)), cross_mean=float(np.mean(X_)))
    print("TIAN_CEILING", json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(a.wd, "results", "rev_tian.json"), "w"), indent=1); print("REVTIAN_DONE", flush=True)


if __name__ == "__main__":
    main()
