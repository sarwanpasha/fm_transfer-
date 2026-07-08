"""
Experiment: GI-tracking gap with bootstrap confidence intervals.

Computes, per representation space, the Spearman correlation between the
embedding interaction-residual magnitude and an expression-space genetic-
interaction (GI) magnitude (DE-based). Reports the FM-minus-rawPCA gap with a
paired bootstrap CI and one-sided significance proxy.

Usage:
    python analysis/gi_tracking.py            # Norman
    python analysis/gi_tracking.py wessels
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

NBOOT = 5000
rng = np.random.default_rng(0)


def run(name):
    prefix = "" if name == "norman" else f"{name}_"
    h5 = NORMAN_H5AD if name == "norman" else WESSELS_H5AD
    scgpt = load_pb(os.path.join(ROOT, f"{prefix}scgpt_pseudobulk.npz"))
    gf = load_pb(os.path.join(ROOT, f"{prefix}geneformer_pseudobulk.npz"))
    expr, uniq = expr_pseudobulk(h5, wessels=(name == "wessels"))
    rawpca = rawpca_pb(expr, uniq)
    g2s = {single_gene(c): c for c in uniq if single_gene(c)}
    ctrl_e = expr["ctrl"]
    spaces = {"scGPT": scgpt, "Geneformer": gf, "rawPCA": rawpca}

    gi, resid = [], {s: [] for s in spaces}
    for c in uniq:
        dg = double_genes(c)
        if not dg:
            continue
        a, b = dg
        if a not in g2s or b not in g2s:
            continue
        sa, sb = g2s[a], g2s[b]
        if sa not in scgpt or sb not in scgpt:
            continue
        ea = expr[sa] - ctrl_e
        eb = expr[sb] - ctrl_e
        act = expr[c] - ctrl_e
        de = np.argsort(np.abs(act))[::-1][:TOPN_DE]
        gi.append(float(np.linalg.norm((act - (ea + eb))[de])))
        for sn, space in spaces.items():
            r = space[c] - (space[sa] + space[sb] - space["ctrl"])
            resid[sn].append(float(np.linalg.norm(r)))
    gi = np.array(gi)
    resid = {k: np.array(v) for k, v in resid.items()}
    n = len(gi)

    point = {s: float(spearmanr(resid[s], gi).correlation) for s in spaces}
    idx = np.arange(n)
    gap_sc = np.empty(NBOOT)
    gap_gf = np.empty(NBOOT)
    boot = {s: np.empty(NBOOT) for s in spaces}
    for i in range(NBOOT):
        bi = rng.choice(idx, n, replace=True)
        rs = spearmanr(resid["scGPT"][bi], gi[bi]).correlation
        rf = spearmanr(resid["Geneformer"][bi], gi[bi]).correlation
        rr = spearmanr(resid["rawPCA"][bi], gi[bi]).correlation
        boot["scGPT"][i], boot["Geneformer"][i], boot["rawPCA"][i] = rs, rf, rr
        gap_sc[i] = rs - rr
        gap_gf[i] = rf - rr

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    out = {
        "n": n,
        "spearman": point,
        "spearman_ci": {s: ci(boot[s]) for s in spaces},
        "gap_scgpt": dict(point=point["scGPT"] - point["rawPCA"], ci=ci(gap_sc),
                          frac_pos=float(np.mean(gap_sc > 0))),
        "gap_geneformer": dict(point=point["Geneformer"] - point["rawPCA"], ci=ci(gap_gf),
                               frac_pos=float(np.mean(gap_gf > 0))),
    }
    print(f"=== GI-tracking ({name}), DE-based, bootstrap {NBOOT} ===")
    for s in spaces:
        lo, hi = out["spearman_ci"][s]
        print(f"  {s:<12} rho={point[s]:.3f}  95% CI [{lo:.3f},{hi:.3f}]")
    for key, lab in [("gap_scgpt", "scGPT-raw"), ("gap_geneformer", "Geneformer-raw")]:
        g = out[key]
        print(f"  {lab:<16} gap={g['point']:+.3f}  CI [{g['ci'][0]:+.3f},{g['ci'][1]:+.3f}]  "
              f"P(>0)={g['frac_pos']:.3f}")
    json.dump(out, open(os.path.join(ROOT, f"gi_tracking_{name}.json"), "w"),
              indent=2, default=float)
    print(f"[gi] wrote gi_tracking_{name}.json")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "norman")
