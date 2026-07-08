"""Shared utilities for the analysis scripts."""
import os
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ALIAS = {"VPRBP": "DCAF1"}


def load_pb(path):
    """Load a pseudobulk npz -> dict[condition] = vector."""
    d = np.load(path, allow_pickle=True)
    emb = d["emb"]
    cond = d["condition"].astype(str)
    return {c: emb[i] for i, c in enumerate(cond)}


def single_gene(c):
    p = c.split("+")
    if "ctrl" in p and len(p) == 2:
        return p[0] if p[1] == "ctrl" else p[1]
    return None


def double_genes(c):
    p = c.split("+")
    if "ctrl" not in p and len(p) == 2:
        return p[0], p[1]
    return None


def norm_cond_wessels(p):
    if p == "control":
        return "ctrl"
    parts = [ALIAS.get(g, g) for g in p.split("_")]
    return f"{parts[0]}+ctrl" if len(parts) == 1 else "+".join(parts)


def expr_pseudobulk(h5ad_path, wessels=False):
    """Log-normalized per-condition pseudobulk expression from an AnnData."""
    ad = sc.read_h5ad(h5ad_path)
    if wessels:
        good = np.array([(":" not in g) and ("_g1" not in g) and ("_g2" not in g)
                         for g in ad.var.index.astype(str)])
        ad = ad[:, good].copy()
        ad.obs["condition"] = [norm_cond_wessels(p)
                               for p in ad.obs["perturbation"].astype(str)]
        raw = sp.csr_matrix(ad.X)
    else:
        raw = ad.layers["counts"] if "counts" in ad.layers else ad.X
        raw = sp.csr_matrix(raw)
    cond = ad.obs["condition"].astype(str).values
    libsize = np.asarray(raw.sum(1)).ravel()
    libsize[libsize == 0] = 1
    uniq = np.unique(cond)
    expr = {}
    for c in uniq:
        rows = np.where(cond == c)[0]
        sub = raw[rows].toarray().astype(np.float32) / libsize[rows][:, None] * 1e4
        expr[c] = np.log1p(sub).mean(0)
    return expr, uniq


def rawpca_pb(expr, uniq, n_components=50):
    M = np.stack([expr[c] for c in uniq])
    Z = PCA(n_components=n_components, random_state=0).fit_transform(
        StandardScaler().fit_transform(M))
    return {c: Z[i] for i, c in enumerate(uniq)}
