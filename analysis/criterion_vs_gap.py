"""
Experiment: effective-rank criterion vs measured GI gap across architectures.

Computes the label-free single-effect effective-rank criterion and the measured
GI-tracking gap for each (model x dataset) cell present on disk, to test whether
a single geometric statistic predicts GI-tracking across architecture types.

Usage:
    python analysis/criterion_vs_gap.py
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import ROOT, NORMAN_H5AD, WESSELS_H5AD, TOPN_DE
from analysis.common import (load_pb, single_gene, double_genes,
                             expr_pseudobulk, rawpca_pb)

NSUB = 28
NREP = 500
rng = np.random.default_rng(0)

MODELS = {
    "scGPT": {"norman": "scgpt_pseudobulk.npz", "wessels": "wessels_scgpt_pseudobulk.npz"},
    "Geneformer": {"norman": "geneformer_pseudobulk.npz", "wessels": "wessels_geneformer_pseudobulk.npz"},
    "scVI": {"norman": "scvi_norman_pseudobulk.npz", "wessels": "scvi_wessels_pseudobulk.npz"},
}


def eff_rank(E):
    Ec = E - E.mean(0)
    s = np.linalg.svd(Ec, compute_uv=False)
    p = s ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def criterion(space):
    ctrl = space["ctrl"]
    E = np.array([space[c] - ctrl for c in space if single_gene(c) is not None])
    n = len(E)
    if n <= NSUB:
        return eff_rank(E)
    return float(np.mean([eff_rank(E[rng.choice(n, NSUB, replace=False)])
                          for _ in range(NREP)]))


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
    return float(spearmanr(rm, gi).correlation - spearmanr(rr, gi).correlation), len(gi)


def main():
    exprcache = {}
    rows = []
    for model, dsmap in MODELS.items():
        for ds, pb in dsmap.items():
            path = os.path.join(ROOT, pb)
            if not os.path.exists(path):
                print(f"[skip] {model}/{ds}: {pb} not found")
                continue
            space = load_pb(path)
            crit = criterion(space)
            if ds not in exprcache:
                h5 = NORMAN_H5AD if ds == "norman" else WESSELS_H5AD
                exprcache[ds] = expr_pseudobulk(h5, wessels=(ds == "wessels"))
            expr, uniq = exprcache[ds]
            gap, n = gi_gap(space, expr, uniq)
            rows.append((f"{model}/{ds}", crit, gap, n))

    print(f"{'cell':<22}{'criterion':>12}{'GI_gap':>10}{'n':>6}")
    for name, crit, gap, n in rows:
        print(f"{name:<22}{crit:>12.2f}{gap:>10.3f}{n:>6}")
    if len(rows) >= 4:
        X = np.array([r[1] for r in rows])
        Y = np.array([r[2] for r in rows])
        sr = spearmanr(X, Y)
        print(f"\nSpearman(criterion, gap) over {len(rows)} cells = "
              f"{sr.correlation:.3f} (p={sr.pvalue:.3g})")
    json.dump([{"cell": n, "criterion": c, "gap": g, "n": nn} for n, c, g, nn in rows],
              open(os.path.join(ROOT, "criterion_vs_gap.json"), "w"), indent=2, default=float)
    print("[criterion] wrote criterion_vs_gap.json")


if __name__ == "__main__":
    main()
