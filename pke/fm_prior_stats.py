"""Screen templates for training GrnPFN from its public code. The released code reads per-screen statistics from
eval/results/bench/copfn/prior_stats.json, which is not distributed; this script rebuilds them from the raw screens:
 cells_single, cells_pair  5%, 25%, 50%, 75% and 95% quantiles of the number of cells per single and per double
 cells_ctrl                number of control cells
 cell_var_over_scale       per-cell variance (summed over genes, averaged over perturbations) divided by the mean squared
                           norm of the perturbation effects, for singles and doubles; reported over all genes and over the
                           50 most changed genes of each perturbation
 spectrum                  eigenvalues of the centered matrix of single-perturbation effects, in decreasing order
Log-normalized expression on the highly variable genes of the benchmark. Output: results/prior_stats.json.
Usage: fm_prior_stats.py --wd W [--datasets Norman Wessels]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_disjoint import load

Q = (5, 25, 50, 75, 95)


def cell_stats(C, ix):
    X = C[ix]; mu = np.asarray(X.mean(0)).ravel(); m2 = np.asarray(X.multiply(X).mean(0)).ravel()
    return mu, np.maximum(m2 - mu ** 2, 0) * len(ix) / max(len(ix) - 1, 1)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--wd", default="."); p.add_argument("--datasets", nargs="*", default=["Norman", "Wessels"])
    p.add_argument("--min-cells", type=int, default=30); a = p.parse_args(); out = {}
    for ds in a.datasets:
        genes, singles, doubles, C, cells = load(a.wd, ds, a.min_cells)
        mu0, _ = cell_stats(C, cells["control"]); res = {}
        for kind, conds in (("single", singles), ("pair", [d for d, _, _ in doubles])):
            v_all, v_top, e_all, e_top = [], [], [], []
            for c in conds:
                mu, var = cell_stats(C, cells[c]); d = mu - mu0; top = np.argsort(-np.abs(d))[:50]
                v_all.append(var.sum()); e_all.append(d @ d); v_top.append(var[top].sum()); e_top.append(d[top] @ d[top])
            res[kind] = dict(all=float(np.mean(v_all) / np.mean(e_all)), top50=float(np.mean(v_top) / np.mean(e_top)))
        D = np.stack([cell_stats(C, cells[s])[0] - mu0 for s in singles]); D = D - D.mean(0)
        ev = np.linalg.svd(D, compute_uv=False) ** 2
        out[ds] = dict(cells_single=[float(np.percentile([len(cells[s]) for s in singles], q)) for q in Q],
                       cells_pair=[float(np.percentile([len(cells[d]) for d, _, _ in doubles], q)) for q in Q],
                       cells_ctrl=int(len(cells["control"])),
                       cell_var_over_scale=dict(single=res["single"]["top50"], pair=res["pair"]["top50"],
                                                single_allgenes=res["single"]["all"], pair_allgenes=res["pair"]["all"]),
                       spectrum=[float(x) for x in ev], n_single=len(singles), n_pair=len(doubles), n_genes=len(genes))
        o = out[ds]; print(ds, json.dumps({k: v for k, v in o.items() if k != "spectrum"}), "spectrum_top5", [round(x / sum(o["spectrum"]), 3) for x in o["spectrum"][:5]], flush=True)
    json.dump(out, open(os.path.join(a.wd, "results", "prior_stats.json"), "w"), indent=1); print("PRIOR_STATS_DONE")


if __name__ == "__main__":
    main()
