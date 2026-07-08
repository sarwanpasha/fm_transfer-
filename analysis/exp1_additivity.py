"""
Experiment 1: compositional additivity of perturbation embeddings.

For each double perturbation A+B, compares its embedding effect to the additive
composition of its single-perturbation effects, and reports the median relative
residual and directional cosine per representation space.

Usage:
    python analysis/exp1_additivity.py            # Norman
    python analysis/exp1_additivity.py wessels
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import ROOT, NORMAN_H5AD, WESSELS_H5AD
from analysis.common import (load_pb, single_gene, double_genes,
                             expr_pseudobulk, rawpca_pb)


def run(name):
    prefix = "" if name == "norman" else f"{name}_"
    h5 = NORMAN_H5AD if name == "norman" else WESSELS_H5AD
    scgpt = load_pb(os.path.join(ROOT, f"{prefix}scgpt_pseudobulk.npz"))
    gf = load_pb(os.path.join(ROOT, f"{prefix}geneformer_pseudobulk.npz"))
    expr, uniq = expr_pseudobulk(h5, wessels=(name == "wessels"))
    rawpca = rawpca_pb(expr, uniq)

    spaces = {"scGPT": scgpt, "Geneformer": gf, "rawPCA": rawpca}
    results = {}
    print(f"=== EXP1 additivity ({name}) ===")
    print(f"{'space':<12}{'n':>5}{'relResid(med)':>15}{'cos(med)':>10}")
    for sn, space in spaces.items():
        g2s = {single_gene(c): c for c in space if single_gene(c)}
        ctrl = space["ctrl"]
        rr, cs = [], []
        for c in space:
            dg = double_genes(c)
            if not dg:
                continue
            a, b = dg
            if a not in g2s or b not in g2s:
                continue
            pred = space[g2s[a]] + space[g2s[b]] - ctrl
            act = space[c]
            eff = act - ctrl
            pe = pred - ctrl
            den = np.linalg.norm(eff)
            if den == 0:
                continue
            rr.append(np.linalg.norm(act - pred) / den)
            cs.append(np.dot(eff, pe) / (np.linalg.norm(eff) * np.linalg.norm(pe) + 1e-12))
        results[sn] = dict(n=len(rr), rel_resid_med=float(np.median(rr)),
                           cos_med=float(np.median(cs)))
        print(f"{sn:<12}{len(rr):>5}{np.median(rr):>15.4f}{np.median(cs):>10.4f}")

    json.dump(results, open(os.path.join(ROOT, f"exp1_{name}.json"), "w"),
              indent=2, default=float)
    print(f"[exp1] wrote exp1_{name}.json")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "norman")
