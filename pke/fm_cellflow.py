"""CellFlow baseline (Klein et al. 2025) on our runs, with the cells, folds and DE-gene scoring of fm_gears.py. CellFlow is a
flow-matching model that maps control cells to perturbed cells conditioned on the perturbation. Each gene perturbation is
represented by its GenePT embedding (64 principal components; control is the zero vector) and a double by the set of its 2
gene embeddings (mean pooling, as in the CellFlow combination tutorial). Cells are represented by 50 principal components of
log-normalized expression on the 5,000 highly variable genes, computed on the training cells of each fold; predictions are
decoded to genes with the same components and averaged over 200 control cells transported by the model (--pred-cells) with 20 fixed Tsit5 steps (--ode-steps). The predicted residual is
pred(double) - expr[a] - expr[b] + expr[ctrl], scored on the 50 most changed genes of each double.
Usage: fm_cellflow.py --run R --emb GenePT.pickle [--iterations 20000] [--schemes random genegrp] [--max-folds 5]"""
import argparse, functools, json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--emb", required=True)
    p.add_argument("--iterations", type=int, default=20000); p.add_argument("--batch", type=int, default=512); p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--max-cells", type=int, default=100); p.add_argument("--ctrl-cells", type=int, default=1000); p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-folds", type=int, default=5); p.add_argument("--schemes", nargs="*", default=["random", "genegrp"]); p.add_argument("--npc", type=int, default=50); p.add_argument("--pred-cells", type=int, default=200); p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--wd", default=os.environ.get("FMTRANSFER_ROOT", ".")); a = p.parse_args()
    import anndata as ad, pandas as pd, scanpy as sc, scipy.sparse as sp
    import fmcore as F, fmpredict as FP
    from fm_gira import gene_feats, ridge_pairs
    from cellflow.model import CellFlow
    from cellflow.utils import match_linear
    import diffrax
    run = os.path.basename(a.run.rstrip("/")); S = json.load(open(os.path.join(a.run, "summary.json")))
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds)
    R = np.asarray(F.residual_exact(expr, "control", doubles), dtype=np.float64); DE = FP.de_sets(expr, "control", doubles, hn, 50)
    FILES = {"Norman": "NormanWeissman2019_filtered.h5ad", "Wessels": "WesselsSatija2023.h5ad", "TianIPSC": "TianKampmann2019_iPSC.h5ad", "TianNeuron": "TianKampmann2019_day7neuron.h5ad"}
    idx = np.load(os.path.join(a.run, "subsample.npz"))["idx"]
    H = ad.read_h5ad(os.path.join(a.wd, "data", FILES[S["dataset"]]), backed="r"); X = H[idx].to_memory(); H.file.close(); X.var_names_make_unique()
    M = sp.csr_matrix(X.X if sp.issparse(X.X) else sp.csr_matrix(X.X), dtype=np.float32); M.data = np.round(M.data); X.X = M
    sc.pp.normalize_total(X, target_sum=1e4); sc.pp.log1p(X); sc.pp.highly_variable_genes(X, n_top_genes=S["n_hvg"], flavor="seurat")
    hvg = np.asarray(X.var["highly_variable"].values); assert list(np.asarray(X.var_names)[hvg]) == list(hn), "HVG mismatch vs pb_hvg"
    lab = X.obs["perturbation"].astype(str).values; Xd = X[:, hvg].X; Xd = Xd.toarray() if sp.issparse(Xd) else np.asarray(Xd)
    rng = np.random.default_rng(0); keep = []
    for c in conds:
        ii = np.flatnonzero(lab == c); cap = a.ctrl_cells if c == "control" else a.max_cells
        keep.extend(rng.choice(ii, size=min(cap, len(ii)), replace=False).tolist())
    keep = np.sort(np.array(keep)); Xk = Xd[keep].astype(np.float32); lk = lab[keep]
    sd_of = {d: (x, y) for d, x, y in doubles}
    g1 = np.array(["control" if c == "control" else (sd_of[c][0] if c in sd_of else c) for c in lk])
    g2 = np.array(["control" if (c == "control" or c not in sd_of) else sd_of[c][1] for c in lk])
    emb = pickle.load(open(a.emb, "rb")); genes = sorted(set(singles)); E = []
    for g in genes:
        v = emb.get(str(g).upper()); E.append(None if v is None else np.asarray(v, np.float64))
    mean = np.mean([e for e in E if e is not None], 0); E = np.stack([mean if e is None else e for e in E]); E -= E.mean(0)
    U, Sv, _ = np.linalg.svd(E, full_matrices=False); Ze = (U[:, :64] * Sv[:64]); Ze /= np.linalg.norm(Ze, axis=1).mean()
    reps = {g: Ze[i].astype(np.float32) for i, g in enumerate(genes)}; reps["control"] = np.zeros(Ze.shape[1], np.float32)
    f20 = gene_feats({s: expr[s] - expr["control"] for s in singles}, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for _, x, y in doubles])
    out = {}
    for scheme in a.schemes:
        n = len(doubles); acc = {m: dict(err_add=np.zeros(n), err_mod=np.zeros(n), cnt=np.zeros(n)) for m in ("cellflow", "ridge_raw")}; t0 = time.time()
        for sd in a.seeds:
            for fi, (tr, te) in enumerate(FP.splits(doubles, scheme, a.folds, sd)):
                if fi >= a.max_folds: break
                trd = {doubles[i][0] for i in tr}; mask = np.array([(c == "control") or (c in singles) or (c in trd) for c in lk])
                mu = Xk[mask].mean(0); _, _, Vt = np.linalg.svd(Xk[mask] - mu, full_matrices=False); V = Vt[:a.npc]
                pcs = ((Xk - mu) @ V.T).astype(np.float32)
                A = ad.AnnData(X=np.zeros((int(mask.sum()), 1), np.float32), obs=pd.DataFrame({"gene1": g1[mask], "gene2": g2[mask], "condition": lk[mask],
                               "is_control": lk[mask] == "control"}, index=[f"c{i}" for i in range(int(mask.sum()))]))
                A.obs["gene1"] = A.obs["gene1"].astype("category"); A.obs["gene2"] = A.obs["gene2"].astype("category")
                A.obsm["X_pca"] = pcs[mask]; A.uns["genept"] = reps
                cf = CellFlow(A, solver="otfm")
                cf.prepare_data(sample_rep="X_pca", control_key="is_control", perturbation_covariates={"genes": ("gene1", "gene2")},
                                perturbation_covariate_reps={"genes": "genept"}, max_combination_length=2, null_value=0.0)
                cf.prepare_model(condition_mode="deterministic", regularization=0.0, pooling="mean",
                                 layers_before_pool={"genes": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0}},
                                 layers_after_pool={"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0}, condition_embedding_dim=64,
                                 cond_output_dropout=0.5, hidden_dims=[512, 512, 512], conditioning="concatenation", decoder_dims=[1024, 1024, 1024],
                                 probability_path={"constant_noise": 0.5}, match_fn=functools.partial(match_linear, epsilon=1.0, tau_a=1.0, tau_b=1.0),
                                 linear_projection_before_concatenation=True)
                cf.train(num_iterations=a.iterations, batch_size=a.batch)
                cm = lk == "control"; cm[np.flatnonzero(cm)[a.pred_cells:]] = False; Ac = ad.AnnData(X=np.zeros((int(cm.sum()), 1), np.float32), obs=pd.DataFrame({"is_control": True}, index=[f"k{i}" for i in range(int(cm.sum()))])); Ac.obsm["X_pca"] = pcs[cm]; Ac.uns["genept"] = reps
                cov = pd.DataFrame({"gene1": [doubles[i][1] for i in te], "gene2": [doubles[i][2] for i in te], "condition": [doubles[i][0] for i in te],
                                    "is_control": False})
                cov["gene1"] = pd.Categorical(cov["gene1"], categories=A.obs["gene1"].cat.categories.union(cov["gene1"].unique()))
                cov["gene2"] = pd.Categorical(cov["gene2"], categories=A.obs["gene2"].cat.categories.union(cov["gene2"].unique()))
                tp = time.time(); print(f"  {scheme} fold{fi} trained ({time.time() - t0:.0f}s), predicting", flush=True)
                preds = cf.predict(adata=Ac, sample_rep="X_pca", condition_id_key="condition", covariate_data=cov, dt0=1.0 / a.ode_steps, stepsize_controller=diffrax.ConstantStepSize())
                print(f"  prediction {time.time() - tp:.0f}s", flush=True); Pr = ridge_pairs(Fraw[tr], Fraw[te], R[tr])
                for j, i in enumerate(te):
                    d, x, y = doubles[i]; pd_ = np.asarray(preds[d]).mean(0) @ V + mu
                    pg = pd_ - expr[x] - expr[y] + expr["control"]; t = R[i, DE[i]]
                    for mm, pp in (("cellflow", pg[DE[i]]), ("ridge_raw", Pr[j, DE[i]])):
                        A_ = acc[mm]; A_["err_add"][i] += t @ t; A_["err_mod"][i] += (t - pp) @ (t - pp); A_["cnt"][i] += 1
                print(f"  {scheme} seed{sd} fold{fi} done ({time.time() - t0:.0f}s)", flush=True)
        np.savez(os.path.join(a.run, f"cellflow_acc_{scheme}.npz"), **{f"{mm}_{k}": v for mm, A_ in acc.items() for k, v in A_.items()})
        ii = np.flatnonzero(acc["ridge_raw"]["cnt"] > 0); B = np.random.default_rng(0).choice(ii, size=(1000, len(ii)))
        r2 = lambda A_, k: 1 - A_["err_mod"][k].sum(-1) / A_["err_add"][k].sum(-1)
        dd = r2(acc["cellflow"], B) - r2(acc["ridge_raw"], B); vc, vr = float(r2(acc["cellflow"], ii)), float(r2(acc["ridge_raw"], ii))
        out[scheme] = dict(cellflow=vc, ridge_raw=vr, d=vc - vr, lo=float(np.percentile(dd, 2.5)), hi=float(np.percentile(dd, 97.5)), n_test=int(len(ii)))
        print(f"CF {scheme:8s} cellflow R2={vc:+.3f} ridge R2={vr:+.3f} d={vc - vr:+.3f} lo={out[scheme]['lo']:+.3f} hi={out[scheme]['hi']:+.3f}", flush=True)
    json.dump(out, open(os.path.join(a.run, "cellflow.json"), "w"), indent=1); print("CELLFLOW_DONE", flush=True)


if __name__ == "__main__":
    main()
