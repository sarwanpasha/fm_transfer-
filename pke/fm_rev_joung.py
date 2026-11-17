"""Held-out screen never used during development: Joung et al. 2023 combinatorial TF overexpression in hESCs (scPerturb).
Labels 'A+B' include GFP as the empty ORF: GFP alone (or GFP+GFP) -> control, GFP+X -> single X, X+Y -> double X_Y; triples are
dropped. Conditions with fewer than 30 cells are dropped (as for the Tian screens). For seeds 0-4, at most 500 cells per condition
are sampled and the standard preprocessing is applied (counts normalized to 1e4, log1p, 5,000 HVGs, pseudobulk).
Writes results/Joung_s<seed>_mc500/{pb_hvg.npz, subsample.npz, summary.json}.  Usage: fm_rev_joung.py --wd W"""
import argparse, collections, json, os
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument("--wd", required=True); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4]); a = p.parse_args()
    import anndata as ad, scanpy as sc, scipy.sparse as sp
    A = ad.read_h5ad(os.path.join(a.wd, "data", "scperturb", "JoungZhang2023_combinatorial.h5ad"), backed="r")
    lab0 = A.obs["perturbation"].astype(str).values
    def harm(l):
        g = sorted({x for x in l.split("+") if x and x.upper() != "GFP"})
        return "control" if len(g) == 0 else g[0] if len(g) == 1 else "_".join(g) if len(g) == 2 else None
    lab = np.array([harm(l) or "DROP" for l in lab0]); cnt = collections.Counter(lab)
    print("ALL LABELS", sorted(collections.Counter(lab0).items(), key=lambda kv: -kv[1]), flush=True)
    keep = {c for c, k in cnt.items() if c != "DROP" and k >= 30}
    singles = {c for c in keep if c != "control" and "_" not in c}
    doubles = {c for c in keep if "_" in c and all(g in singles for g in c.split("_"))}
    pseudo = "control" not in keep
    if pseudo:
        print("no empty-vector control: control profile = mean of the single-TF pseudobulk profiles (pseudo-control)", flush=True)
    conds = sorted(({"control"} if not pseudo else set()) | singles | doubles); print(f"control cells={cnt['control']} singles={len(singles)} doubles={len(doubles)}", flush=True)
    for sd in a.seeds:
        rng = np.random.default_rng(sd); idx = []
        for c in conds:
            w = np.flatnonzero(lab == c)
            if len(w) > 500: w = np.sort(rng.choice(w, 500, replace=False))
            idx.append(w)
        idx = np.sort(np.concatenate(idx)); X = A[idx].to_memory(); X.var_names_make_unique()
        M = sp.csr_matrix(X.X if sp.issparse(X.X) else sp.csr_matrix(X.X), dtype=np.float32)
        vals = M.data[:100000]; print(f"seed {sd}: cells={X.n_obs} integer_counts={bool(np.allclose(vals, np.round(vals)))} max={float(M.data.max()) if M.nnz else 0}", flush=True)
        M.data = np.round(M.data); X.X = M
        sc.pp.normalize_total(X, target_sum=1e4); sc.pp.log1p(X); sc.pp.highly_variable_genes(X, n_top_genes=5000, flavor="seurat")
        hvg = np.asarray(X.var["highly_variable"].values); l = lab[idx]
        pb = np.stack([np.asarray(X.X[l == c][:, hvg].mean(0)).ravel() for c in conds])
        cn = list(conds)
        if pseudo:
            pb = np.vstack([pb, pb[[k for k, c in enumerate(conds) if c in singles]].mean(0)]); cn = cn + ["control"]
        od = os.path.join(a.wd, "results", f"Joung_s{sd}_mc500"); os.makedirs(od, exist_ok=True)
        np.savez(os.path.join(od, "subsample.npz"), idx=idx)
        np.savez_compressed(os.path.join(od, "pb_hvg.npz"), conds=np.array(cn), pb=pb, genes=np.asarray(X.var_names)[hvg])
        json.dump(dict(dataset="Joung", seed=sd, pseudo_control=pseudo, max_cells=500, n_hvg=int(hvg.sum()), n_singles=len(singles), n_doubles=len(doubles)),
                  open(os.path.join(od, "summary.json"), "w"), indent=1)
    A.file.close(); print("JOUNG_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
