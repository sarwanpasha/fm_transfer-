"""AttentionPert baseline (Bai et al. 2024) on our runs, with the cells, folds and DE-gene scoring of fm_gears.py.
AttentionPert extends GEARS with gene2vec-initialized perturbation embeddings and attention over perturbation weights. It is
trained per fold on all singles and the training doubles (10% held out for validation) and predicts the held-out doubles;
the predicted residual is pred(double) - expr[a] - expr[b] + expr[ctrl]. The gene ontology graph is built once per run with
the Jaccard rule and threshold of the AttentionPert code (vectorized), and gene2vec embeddings follow its example script
(genes without a vector are drawn from a Gaussian fitted to all vectors).
Usage: fm_attnpert.py --run R --attnpert-src S --gene2go F --gene2vec F [--epochs 20] [--schemes random genegrp]"""
import argparse, json, os, sys, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def go_edges(genes, gene2go, thr=0.1):
    import pandas as pd, scipy.sparse as sp
    gs = [g for g in genes if g in gene2go]; terms = sorted({t for g in gs for t in gene2go[g]}); ti = {t: i for i, t in enumerate(terms)}
    rows, cols = zip(*[(i, ti[t]) for i, g in enumerate(gs) for t in set(gene2go[g])])
    M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(gs), len(terms)))
    inter = (M @ M.T).toarray(); sz = np.asarray(M.sum(1)).ravel(); J = inter / (sz[:, None] + sz[None, :] - inter)
    s, t = np.nonzero(J > thr)
    return pd.DataFrame({"source": [gs[i] for i in s], "target": [gs[j] for j in t], "importance": J[s, t]})


def gene2vec_matrix(path, genes, seed=0):
    vec = {}
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(" "); vec[p[0]] = np.array([float(x) for x in p[1:] if x != ""])
    V = np.stack(list(vec.values())); mu, cov = V.mean(0), np.cov(V.T); rng = np.random.default_rng(seed); miss = 0
    out = np.zeros((len(genes), V.shape[1]))
    for i, g in enumerate(genes):
        if g in vec: out[i] = vec[g]
        else: out[i] = rng.multivariate_normal(mu, cov); miss += 1
    return out, miss


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--attnpert-src", required=True)
    p.add_argument("--gene2go", required=True); p.add_argument("--gene2vec", required=True); p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--max-cells", type=int, default=100); p.add_argument("--ctrl-cells", type=int, default=1000)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--schemes", nargs="*", default=["random", "genegrp"])
    p.add_argument("--wd", default=os.environ.get("FMTRANSFER_ROOT", ".")); a = p.parse_args()
    import anndata as ad, scanpy as sc, scipy.sparse as sp, torch
    import fmcore as F, fmpredict as FP
    from fm_gira import gene_feats, ridge_pairs
    sys.path.insert(0, a.attnpert_src)
    from attnpert import PertData
    from attnpert.attnpert import ATTNPERT_RECORD_TRAIN
    from attnpert.model import PL_PW_non_add_Model
    from attnpert.inference import evaluate
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
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
    lab = X.obs["perturbation"].astype(str).values; X = X[:, hvg].copy()
    gmap = lambda c: "ctrl" if c == "control" else ("+".join(c.split("_")) if "_" in c else c + "+ctrl")
    rng = np.random.default_rng(0); keep = []
    for c in conds:
        ii = np.flatnonzero(lab == c); cap = a.ctrl_cells if c == "control" else a.max_cells
        keep.extend(rng.choice(ii, size=min(cap, len(ii)), replace=False).tolist())
    keep = np.sort(np.array(keep))
    A = ad.AnnData(X=sp.csr_matrix(X.X[keep]), obs={"condition": [gmap(c) for c in lab[keep]], "cell_type": ["cell"] * len(keep)})
    A.var["gene_name"] = hn; A.var_names = hn; A.obs_names = [f"c{i}" for i in range(len(keep))]
    gdir = os.path.join(a.wd, "data", "attnpert"); os.makedirs(gdir, exist_ok=True); name = f"fmattn_{run}".lower()
    pdata = PertData(gdir); pdata.new_data_process(dataset_name=name, adata=A)
    genes = list(pdata.gene_names)
    gene2go = pickle.load(open(a.gene2go, "rb")); go_edges(genes, gene2go).to_csv(os.path.join(gdir, name, "go.csv"), index=False)
    G2V, miss = gene2vec_matrix(a.gene2vec, genes); g2v_path = os.path.join(gdir, name, "gene2vec.npy"); np.save(g2v_path, G2V)
    print(f"run={run} cells={len(keep)} genes={len(genes)} gene2vec_missing={miss}", flush=True)
    avail = set(pdata.adata.obs["condition"].unique()); processed = pdata.dataset_processed
    f20 = gene_feats({s: expr[s] - expr["control"] for s in singles}, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for _, x, y in doubles])
    setting = dict(gene2vec_args={"gene2vec_file": g2v_path}, pert_local_min_weight=0.75, pert_local_conv_K=1, pert_weight_heads=2,
                   pert_weight_head_dim=64, pert_weight_act="softmax", non_add_beta=5e-2, record_pred=False)
    out = {}; cwd = os.getcwd()
    for scheme in a.schemes:
        n = len(doubles); acc = {m: dict(err_add=np.zeros(n), err_mod=np.zeros(n), cnt=np.zeros(n)) for m in ("attnpert", "ridge_raw")}; nfall = 0; t0 = time.time()
        for sd in a.seeds:
            for fi, (tr, te) in enumerate(FP.splits(doubles, scheme, a.folds, sd)):
                r = np.random.default_rng(100 + fi); trd = [gmap(doubles[i][0]) for i in tr if gmap(doubles[i][0]) in avail]; r.shuffle(trd)
                nv = max(1, len(trd) // 10); val, trd = trd[:nv], trd[nv:]
                sing = [gmap(s) for s in singles if gmap(s) in avail]; ted = [gmap(doubles[i][0]) for i in te if gmap(doubles[i][0]) in avail]
                pdata.dataset_processed = processed; pdata.split = "custom"; pdata.seed = fi; pdata.train_gene_set_size = 0.75; pdata.subgroup = None
                pdata.set2conditions = {"train": sing + trd + ["ctrl"], "val": val, "test": ted if ted else val}
                pdata.get_dataloader(batch_size=32, test_batch_size=128)
                os.chdir(gdir); torch.manual_seed(fi)
                m = ATTNPERT_RECORD_TRAIN(pdata, device="cpu"); m.model_initialize(hidden_size=64, model_class=PL_PW_non_add_Model, **setting)
                m.train(epochs=a.epochs); os.chdir(cwd)
                res = evaluate(pdata.dataloader["test_loader"], m.best_model, False, "cpu", real_num_genes=m.real_num_genes)
                pred = {}
                for c in np.unique(res["pert_cat"]): pred[c] = res["pred"][res["pert_cat"] == c].mean(0).astype(np.float64)
                Pr = ridge_pairs(Fraw[tr], Fraw[te], R[tr])
                for j, i in enumerate(te):
                    d, x, y = doubles[i]; key = gmap(d)
                    if key in pred: pg = pred[key] - expr[x] - expr[y] + expr["control"]
                    else: pg = np.zeros(R.shape[1]); nfall += 1
                    t = R[i, DE[i]]
                    for mm, pp in (("attnpert", pg[DE[i]]), ("ridge_raw", Pr[j, DE[i]])):
                        A_ = acc[mm]; A_["err_add"][i] += t @ t; A_["err_mod"][i] += (t - pp) @ (t - pp); A_["cnt"][i] += 1
                print(f"  {scheme} seed{sd} fold{fi} done ({time.time() - t0:.0f}s, fallback so far {nfall})", flush=True)
        np.savez(os.path.join(a.run, f"attnpert_acc_{scheme}.npz"), **{f"{mm}_{k}": v for mm, A_ in acc.items() for k, v in A_.items()}, nfall=nfall)
        ii = np.flatnonzero(acc["ridge_raw"]["cnt"] > 0); B = np.random.default_rng(0).choice(ii, size=(1000, len(ii)))
        r2 = lambda A_, k: 1 - A_["err_mod"][k].sum(-1) / A_["err_add"][k].sum(-1)
        dd = r2(acc["attnpert"], B) - r2(acc["ridge_raw"], B); va, vr = float(r2(acc["attnpert"], ii)), float(r2(acc["ridge_raw"], ii))
        out[scheme] = dict(attnpert=va, ridge_raw=vr, d=va - vr, lo=float(np.percentile(dd, 2.5)), hi=float(np.percentile(dd, 97.5)), n_test=int(len(ii)), n_fallback=nfall)
        print(f"AP {scheme:8s} attnpert R2={va:+.3f} ridge R2={vr:+.3f} d={va - vr:+.3f} lo={out[scheme]['lo']:+.3f} hi={out[scheme]['hi']:+.3f} fallback={nfall}", flush=True)
    json.dump(out, open(os.path.join(a.run, "attnpert.json"), "w"), indent=1); print("ATTNPERT_DONE", flush=True)


if __name__ == "__main__":
    main()
