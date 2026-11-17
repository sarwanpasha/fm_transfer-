"""Build the pseudobulk input of all PKE scripts for one (dataset, seed).

Reads an scPerturb .h5ad file from $FMTRANSFER_ROOT/data, subsamples at most --max-cells cells per condition, normalizes
counts to 1e4 per cell, log-transforms, selects highly variable genes and writes
$FMTRANSFER_ROOT/results/<dataset>_s<seed>_mc<max-cells>/{pb_hvg.npz, subsample.npz, summary.json}.

Usage: python prepare.py --dataset Norman --seed 0
"""
import argparse, collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmcore as F

FILES = {"Norman": "NormanWeissman2019_filtered.h5ad", "Wessels": "WesselsSatija2023.h5ad",
         "TianIPSC": "TianKampmann2019_iPSC.h5ad", "TianNeuron": "TianKampmann2019_day7neuron.h5ad"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(FILES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cells", type=int, default=500)
    p.add_argument("--n-hvg", type=int, default=5000)
    p.add_argument("--min-cells-tian", type=int, default=30)
    p.add_argument("--wd", default=os.environ.get("FMTRANSFER_ROOT", "."))
    a = p.parse_args()
    import anndata as ad, scanpy as sc, scipy.sparse as sp
    od = os.path.join(a.wd, "results", f"{a.dataset}_s{a.seed}_mc{a.max_cells}")
    os.makedirs(od, exist_ok=True)
    A = ad.read_h5ad(os.path.join(a.wd, "data", FILES[a.dataset]), backed="r")
    pert = A.obs["perturbation"].astype(str).values
    singles, doubles, _ = F.parse_conditions(pert)
    if a.dataset.startswith("Tian"):  # high-MOI design: keep conditions with enough cells
        cnt = collections.Counter(pert)
        singles = [x for x in singles if cnt[x] >= a.min_cells_tian]
        ss = set(singles)
        doubles = [(d, x, y) for d, x, y in doubles if cnt[d] >= a.min_cells_tian and x in ss and y in ss]
    keep = sorted(set(["control"] + singles + [d for d, _, _ in doubles]))
    rng = np.random.default_rng(a.seed)
    idx = []
    for c in keep:
        w = np.flatnonzero(pert == c)
        if len(w) > a.max_cells:
            w = np.sort(rng.choice(w, a.max_cells, replace=False))
        idx.append(w)
    idx = np.sort(np.concatenate(idx))
    X = A[idx].to_memory()
    A.file.close()
    X.var_names_make_unique()
    M = sp.csr_matrix(X.X if sp.issparse(X.X) else sp.csr_matrix(X.X), dtype=np.float32)
    M.data = np.round(M.data)
    X.X = M
    sc.pp.normalize_total(X, target_sum=1e4)
    sc.pp.log1p(X)
    sc.pp.highly_variable_genes(X, n_top_genes=a.n_hvg, flavor="seurat")
    hvg = np.asarray(X.var["highly_variable"].values)
    lab = X.obs["perturbation"].astype(str).values
    expr = {c: np.asarray(X.X[lab == c][:, hvg].mean(0)).ravel() for c in keep}
    np.savez(os.path.join(od, "subsample.npz"), idx=idx)
    np.savez_compressed(os.path.join(od, "pb_hvg.npz"), conds=np.array(keep), pb=np.stack([expr[c] for c in keep]),
                        genes=np.asarray(X.var_names)[hvg])
    json.dump(dict(dataset=a.dataset, seed=a.seed, max_cells=a.max_cells, n_hvg=int(hvg.sum()),
                   n_singles=len(singles), n_doubles=len(doubles)), open(os.path.join(od, "summary.json"), "w"), indent=1)
    print(f"{a.dataset} seed={a.seed}: {len(singles)} singles, {len(doubles)} doubles -> {od}")


if __name__ == "__main__":
    main()
