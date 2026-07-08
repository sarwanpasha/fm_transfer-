"""
Experiment: post-hoc geometric correction (negative result).

Tests whether label-free linear corrections fit only on singles+control
(whitening, rank-restoration) can recover GI-tracking on the failure case
without degrading tracking cells. They cannot -- the lost structure is absent
rather than recoverably distorted.

Usage:
    python analysis/correction.py
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import ROOT, NORMAN_H5AD, WESSELS_H5AD, TOPN_DE
from analysis.common import (load_pb, single_gene, double_genes,
                             expr_pseudobulk, rawpca_pb)

KINDS = ["none", "whiten", "rankrestore"]
PB = {
    "scGPT/norman": "scgpt_pseudobulk.npz", "scGPT/wessels": "wessels_scgpt_pseudobulk.npz",
    "Geneformer/norman": "geneformer_pseudobulk.npz", "Geneformer/wessels": "wessels_geneformer_pseudobulk.npz",
    "scVI/norman": "scvi_norman_pseudobulk.npz", "scVI/wessels": "scvi_wessels_pseudobulk.npz",
}


def fit_correction(E_single, kind):
    d = E_single.shape[1]
    mu = E_single.mean(0)
    X = E_single - mu
    if kind == "none":
        return lambda V: V
    if kind == "whiten":
        C = np.cov(X.T)
        lam = 0.1 * np.trace(C) / d
        C = C + lam * np.eye(d)
        U, S, _ = np.linalg.svd(C)
        W = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
        return lambda V: V @ W
    if kind == "rankrestore":
        k = min(len(E_single) - 1, d, 50)
        pca = PCA(n_components=k).fit(X)
        comps = pca.components_
        sd = np.sqrt(pca.explained_variance_) + 1e-8

        def T(V):
            proj = (V - mu) @ comps.T
            proj_eq = proj / sd * sd.mean()
            return mu + proj_eq @ comps + (V - mu - proj @ comps)
        return T
    raise ValueError(kind)


def gi_gap(space, expr, uniq):
    rawpca = rawpca_pb(expr, uniq)
    g2s = {single_gene(c): c for c in uniq if single_gene(c)}
    ctrl_e = expr["ctrl"]
    gi, rm, rr = [], [], []
    for c in uniq:
        dg = double_genes(c)
        if not dg:
            continue
        a, b = dg
        if a not in g2s or b not in g2s:
            continue
        sa, sb = g2s[a], g2s[b]
        if sa not in space or sb not in space:
            continue
        ea = expr[sa] - ctrl_e
        eb = expr[sb] - ctrl_e
        act = expr[c] - ctrl_e
        de = np.argsort(np.abs(act))[::-1][:TOPN_DE]
        gi.append(np.linalg.norm((act - (ea + eb))[de]))
        rm.append(np.linalg.norm(space[c] - (space[sa] + space[sb] - space["ctrl"])))
        rr.append(np.linalg.norm(rawpca[c] - (rawpca[sa] + rawpca[sb] - rawpca["ctrl"])))
    gi = np.array(gi)
    return float(spearmanr(rm, gi).correlation - spearmanr(rr, gi).correlation)


def main():
    exprcache = {}
    results = {}
    print(f"{'cell':<20}" + "".join(f"{k:>13}" for k in KINDS))
    for cell, pbfile in PB.items():
        path = os.path.join(ROOT, pbfile)
        if not os.path.exists(path):
            print(f"[skip] {cell}: {pbfile} not found")
            continue
        emb_d = np.load(path, allow_pickle=True)
        emb = emb_d["emb"].astype(np.float64)
        cond = emb_d["condition"].astype(str)
        ds = cell.split("/")[1]
        if ds not in exprcache:
            h5 = NORMAN_H5AD if ds == "norman" else WESSELS_H5AD
            exprcache[ds] = expr_pseudobulk(h5, wessels=(ds == "wessels"))
        expr, uniq = exprcache[ds]
        d = {c: emb[i] for i, c in enumerate(cond)}
        ctrl = d["ctrl"]
        Esingle = np.array([d[c] - ctrl for c in cond if single_gene(c) is not None])
        row = {}
        for kind in KINDS:
            T = fit_correction(Esingle, kind)
            corr = {c: T(emb[i:i + 1])[0] for i, c in enumerate(cond)}
            row[kind] = gi_gap(corr, expr, uniq)
        results[cell] = row
        print(f"{cell:<20}" + "".join(f"{row[k]:>13.3f}" for k in KINDS))

    json.dump(results, open(os.path.join(ROOT, "correction.json"), "w"),
              indent=2, default=float)
    print("[correction] wrote correction.json")


if __name__ == "__main__":
    main()
