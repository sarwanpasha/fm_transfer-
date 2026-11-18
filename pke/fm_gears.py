"""GEARS baseline (Roohani et al. 2023) on our runs, same CV folds / DE-gene scoring as fm_pke (fold seed 0 by default).
Cells: same subsample + log1p-CP10k + HVG as fmpredict.py. GEARS is trained per fold on all singles + training doubles
(10% held out for validation), and predicts held-out doubles; predicted residual = pred(double) - expr[a] - expr[b] + expr[ctrl].
Doubles whose genes GEARS cannot place in its graph fall back to the additive prediction (residual 0) and are counted."""
import argparse, json, os, sys, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--epochs", type=int, default=15); p.add_argument("--max-cells", type=int, default=100); p.add_argument("--ctrl-cells", type=int, default=1000)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--max-folds", type=int, default=99); p.add_argument("--schemes", nargs="*", default=["random", "genegrp"])
    p.add_argument("--only-fold", type=int, default=-1)
    p.add_argument("--wd", default=os.environ.get("FMTRANSFER_ROOT", ".")); a = p.parse_args()
    import anndata as ad, scanpy as sc, scipy.sparse as sp, torch
    import fmcore as F, fmpredict as FP
    from fm_gira import gene_feats, ridge_pairs
    from gears import PertData, GEARS
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    run = os.path.basename(a.run.rstrip("/")); S = json.load(open(os.path.join(a.run, "summary.json")))
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds)
    R = np.asarray(F.residual_exact(expr, "control", doubles), dtype=np.float64); DE = FP.de_sets(expr, "control", doubles, hn, 50)
    # --- cells, exactly as fmpredict.py
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
    keep = np.sort(np.array(keep)); A = ad.AnnData(X=sp.csr_matrix(X.X[keep]), obs={"condition": [gmap(c) for c in lab[keep]], "cell_type": ["cell"] * len(keep)})
    A.var["gene_name"] = hn; A.var_names = hn; A.obs_names = [f"c{i}" for i in range(len(keep))]
    gdir = os.path.join(a.wd, "data", "gears"); name = f"fmgears_{run}_{'_'.join(a.schemes)}" + (f"_f{a.only_fold}" if a.only_fold >= 0 else "")
    pdata = PertData(gdir, default_pert_graph=False)
    if os.path.exists(os.path.join(gdir, name)): pdata.load(data_path=os.path.join(gdir, name))
    else: pdata.new_data_process(dataset_name=name, adata=A)
    avail = set(pdata.adata.obs["condition"].unique()); print(f"run={run} cells={len(keep)} conds={len(conds)} gears_conds={len(avail)}", flush=True)
    f20 = gene_feats({s: expr[s] - expr["control"] for s in singles}, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for _, x, y in doubles])
    out = {}
    for scheme in a.schemes:
        n = len(doubles); acc = {m: dict(err_add=np.zeros(n), err_mod=np.zeros(n), cnt=np.zeros(n)) for m in ("gears", "gears_gi", "ridge_raw", "gears_shift", "additive_shift")}
        normr = np.zeros((n, 2)); pd20 = np.full((4, n), np.nan); nfall = 0; t0 = time.time()
        for sd in a.seeds:
            for fi, (tr, te) in enumerate(FP.splits(doubles, scheme, a.folds, sd)):
                if fi >= a.max_folds: break
                if a.only_fold >= 0 and fi != a.only_fold: continue
                r = np.random.default_rng(100 + fi); trd = [gmap(doubles[i][0]) for i in tr if gmap(doubles[i][0]) in avail]; r.shuffle(trd)
                nv = max(1, len(trd) // 10); val, trd = trd[:nv], trd[nv:]
                sing = [gmap(s) for s in singles if gmap(s) in avail]
                ted = [gmap(doubles[i][0]) for i in te if gmap(doubles[i][0]) in avail]
                split = {"train": sing + trd + ["ctrl"], "val": val, "test": ted if ted else val}
                spath = os.path.join(a.run, f"gears_split_{scheme}_{sd}_{fi}.pkl"); pickle.dump(split, open(spath, "wb"))
                pdata.prepare_split(split="custom", split_dict_path=spath); pdata.get_dataloader(batch_size=32, test_batch_size=128)
                torch.manual_seed(fi); g = GEARS(pdata, device="cpu"); g.model_initialize(hidden_size=64); g.train(epochs=a.epochs)
                PL = set(g.pert_list); todo = [i for i in te if gmap(doubles[i][0]) in avail and doubles[i][1] in PL and doubles[i][2] in PL]   # genes outside GEARS graph -> additive fallback (counted)
                sg = sorted({g_ for i in todo for g_ in doubles[i][1:]})
                pred = g.predict([[doubles[i][1], doubles[i][2]] for i in todo] + [[x] for x in sg]) if todo else {}
                Pr = ridge_pairs(Fraw[tr], Fraw[te], R[tr])
                for j, i in enumerate(te):
                    d, x, y = doubles[i]; key = f"{x}_{y}" if f"{x}_{y}" in pred else f"{y}_{x}"
                    if key in pred:
                        pd_ = np.asarray(pred[key], dtype=np.float64)
                        pg = pd_ - expr[x] - expr[y] + expr["control"]                                            # vs observed additive
                        pgi = pd_ - np.asarray(pred[x], np.float64) - np.asarray(pred[y], np.float64) + expr["control"]  # GEARS' own interaction
                    else: pg = np.zeros(R.shape[1]); pgi = pg; pd_ = expr[x] + expr[y] - expr["control"]; nfall += 1
                    t = R[i, DE[i]]
                    ts = (expr[d] - expr["control"])[DE[i]]
                    sh = expr[d] - expr["control"]; top = np.argsort(-np.abs(sh))[:20]; pa = expr[x] + expr[y] - 2 * expr["control"]
                    pd20[:, i] = (np.corrcoef(pd_[top] - expr["control"][top], sh[top])[0, 1], np.corrcoef(pa[top], sh[top])[0, 1],
                                  np.mean((pd_[top] - expr[d][top]) ** 2), np.mean((pa[top] - sh[top]) ** 2))
                    for m, pp in (("gears_shift", (pd_ - expr["control"])[DE[i]]), ("additive_shift", (expr[x] + expr[y] - 2 * expr["control"])[DE[i]])):
                        A_ = acc[m]; A_["err_add"][i] += ts @ ts; A_["err_mod"][i] += (ts - pp) @ (ts - pp); A_["cnt"][i] += 1
                    normr[i] += (np.linalg.norm(pgi[DE[i]]), np.linalg.norm(t))
                    for m, pp in (("gears", pg[DE[i]]), ("gears_gi", pgi[DE[i]]), ("ridge_raw", Pr[j, DE[i]])):
                        A_ = acc[m]; A_["err_add"][i] += t @ t; A_["err_mod"][i] += (t - pp) @ (t - pp); A_["cnt"][i] += 1
                print(f"  {scheme} seed{sd} fold{fi} done ({time.time() - t0:.0f}s, fallback so far {nfall})", flush=True)
        np.savez(os.path.join(a.run, f"gears_acc_{scheme}_s{'-'.join(map(str, a.seeds))}_f{a.only_fold}.npz"),
                 **{f"{m}_{k}": v for m, A_ in acc.items() for k, v in A_.items()}, nfall=nfall, normr=normr)
        np.save(os.path.join(a.run, f"gears_pd20_{scheme}_s{'-'.join(map(str, a.seeds))}_f{a.only_fold}.npy"), pd20)
        ii = np.flatnonzero(acc["ridge_raw"]["cnt"] > 0); B = np.random.default_rng(0).choice(ii, size=(1000, len(ii)))
        r2 = lambda A_, k: 1 - A_["err_mod"][k].sum(-1) / A_["err_add"][k].sum(-1)
        d = r2(acc["gears"], B) - r2(acc["ridge_raw"], B); vg, vr = float(r2(acc["gears"], ii)), float(r2(acc["ridge_raw"], ii))
        out[scheme] = dict(gears=vg, ridge_raw=vr, d=vg - vr, lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), n_test=int(len(ii)), n_fallback=nfall)
        print(f"GR {scheme:8s} gears_gi R2={float(r2(acc['gears_gi'], ii)):+.3f}", flush=True)
        print(f"GR {scheme:8s} gears R2={vg:+.3f} ridge R2={vr:+.3f} d={vg - vr:+.3f} lo={out[scheme]['lo']:+.3f} hi={out[scheme]['hi']:+.3f} fallback={nfall}", flush=True)
    json.dump(out, open(os.path.join(a.run, f"gears_{'_'.join(a.schemes)}_f{a.only_fold}.json"), "w"), indent=1); print("GEARS_DONE")

if __name__ == "__main__":
    main()
