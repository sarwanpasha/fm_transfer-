"""Held-out screen never used during development: combinatorial sci-Plex drug screen in A549 cells (Lotfollahi et al. 2023,
CPA tutorial file combo_sciplex_prep_hvg_filtered.h5ad). condition_ID: control CHEMBL504 (DMSO), single drugs, and drug pairs
'A+B'. Drugs play the role of perturbed genes; doses are pooled within a condition. Raw counts (layers['counts']) are processed
exactly as the CRISPR screens (at most 500 cells per condition, counts normalized to 1e4, log1p, up to 5,000 HVGs, pseudobulk).
Writes results/Sciplex_s<seed>_mc500/{pb_hvg.npz, subsample.npz, summary.json}.  Usage: fm_rev_sciplex.py --h5ad F --wd W"""
import argparse, collections, json, os
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument("--h5ad", required=True); p.add_argument("--wd", required=True)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4]); a = p.parse_args()
    import anndata as ad, scanpy as sc, scipy.sparse as sp
    A = ad.read_h5ad(a.h5ad)
    print("shape", A.shape, "obs", list(A.obs.columns), "layers", list(A.layers.keys()), flush=True)
    for c in ("condition_ID", "cell_type", "log_dose", "cov_drug_dose"):
        if c in A.obs: v = A.obs[c].astype(str).value_counts(); print("COL", c, len(v), v.head(40).to_dict(), flush=True)
    lab0 = A.obs["condition_ID"].astype(str).values
    def harm(l):
        g = sorted({x for x in l.split("+") if x})
        g = [x for x in g if x != "CHEMBL504"]
        return "control" if not g else g[0] if len(g) == 1 else "_".join(g) if len(g) == 2 else None
    lab = np.array([harm(l) or "DROP" for l in lab0]); cnt = collections.Counter(lab)
    keep = {c for c, k in cnt.items() if c != "DROP" and k >= 30}
    singles = {c for c in keep if c != "control" and "_" not in c}
    doubles = {c for c in keep if "_" in c and all(g in singles for g in c.split("_"))}
    conds = sorted({"control"} | singles | doubles)
    print(f"control cells={cnt['control']} singles={len(singles)} doubles={len(doubles)}", sorted(doubles), flush=True)
    C = A.layers["counts"] if "counts" in A.layers else A.X
    for sd in a.seeds:
        rng = np.random.default_rng(sd); idx = []
        for c in conds:
            w = np.flatnonzero(lab == c)
            if len(w) > 500: w = np.sort(rng.choice(w, 500, replace=False))
            idx.append(w)
        idx = np.sort(np.concatenate(idx))
        X = ad.AnnData(X=sp.csr_matrix(C[idx], dtype=np.float32)); X.var_names = A.var_names; X.var_names_make_unique()
        X.X.data = np.round(X.X.data)
        sc.pp.normalize_total(X, target_sum=1e4); sc.pp.log1p(X)
        nh = min(5000, X.n_vars); sc.pp.highly_variable_genes(X, n_top_genes=nh, flavor="seurat")
        hvg = np.asarray(X.var["highly_variable"].values); l = lab[idx]
        pb = np.stack([np.asarray(X.X[l == c][:, hvg].mean(0)).ravel() for c in conds])
        od = os.path.join(a.wd, "results", f"Sciplex_s{sd}_mc500"); os.makedirs(od, exist_ok=True)
        np.savez(os.path.join(od, "subsample.npz"), idx=idx)
        np.savez_compressed(os.path.join(od, "pb_hvg.npz"), conds=np.array(conds), pb=pb, genes=np.asarray(X.var_names)[hvg])
        json.dump(dict(dataset="Sciplex", seed=sd, max_cells=500, n_hvg=int(hvg.sum()), n_singles=len(singles), n_doubles=len(doubles)),
                  open(os.path.join(od, "summary.json"), "w"), indent=1)
        print(f"seed {sd}: cells={len(idx)} hvg={int(hvg.sum())}", flush=True)
    print("SCIPLEX_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
