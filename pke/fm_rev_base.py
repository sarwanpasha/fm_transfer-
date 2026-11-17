"""Additional baselines on the same folds and scoring as fm_pke.py (CPU):
 salt        : additive model (r = 0), Gaudelet et al. 2024
 peper       : salt + symmetric MLP correction f2(f1(mu_a) + f1(mu_b)) on single-perturbation mean effects, MSE on training doubles
 slim        : SLIM combination model (Hu et al. 2026): STRING node2vec embeddings averaged over the two genes, PCA basis K=10,
               closed-form W = (G'G+lI)^-1 G'Y P (P'P+lI)^-1, separate single/double biases, lambda=0.1
 pke_string  : PKE with an RBF gene kernel on STRING embeddings added to the main-effect and interaction parts
 noisy singles (--noisy N): single profiles re-estimated from N random cells per single (targets keep full data); PKE, ridge
               and PKE+STRING are compared to test whether prior knowledge helps when single profiles are noisy
Output: <run>/rev_base.json (or rev_noisy<N>.json).  Usage: fm_rev_base.py --run R --emb STRING.h5 [--noisy 10] [--wd W]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE
from fm_gira import gene_feats, ridge_pairs
from fm_gira2 import gk


def load_string(path, genes):
    if path.endswith(".npz"):
        Z = np.load(path, allow_pickle=True); keys = {str(k).upper(): k for k in Z.files}
        return {g: np.asarray(Z[keys[g.upper()]], np.float64) for g in genes if g.upper() in keys}
    import h5py
    with h5py.File(path, "r") as f:
        keys = {k.upper(): k for k in f.keys()}
        out = {g: np.asarray(f[keys[g.upper()]][()], np.float64) for g in genes if g.upper() in keys}
    return out


def slim_fit_predict(Ysing, Psing, Ydbl, Pdbl, Pte, K=10, lam=0.1):
    """Ysing/Ydbl: conditions x genes (mean expression), P*: conditions x D (embeddings). Returns predicted double profiles."""
    bs, bd = Ysing.mean(0), Ydbl.mean(0)
    Yc = np.vstack([Ysing - bs, Ydbl - bd]).T                      # genes x conditions
    U, S, _ = np.linalg.svd(Yc, full_matrices=False); G = U[:, :K]
    P = np.vstack([Psing, Pdbl])
    Z = np.linalg.solve(G.T @ G + lam * np.eye(G.shape[1]), G.T @ Yc)
    W = np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), (Z @ P).T).T
    return np.maximum((G @ W @ Pte.T).T + bd, 0)


def peper_fit_predict(Mtr_a, Mtr_b, Rtr, Mte_a, Mte_b, seed=0, epochs=400):
    import torch
    torch.manual_seed(seed); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    d = Mtr_a.shape[1]; sc = float(np.abs(np.vstack([Mtr_a, Mtr_b])).std() + 1e-8); rs = float(np.abs(Rtr).std() + 1e-8)
    enc = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(512, 256))
    dec = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, Rtr.shape[1]))
    T = lambda x: torch.tensor(x / sc, dtype=torch.float32)
    A, B, Y = T(Mtr_a), T(Mtr_b), torch.tensor(Rtr / rs, dtype=torch.float32)
    n = len(Y); rng = np.random.default_rng(seed); val = rng.choice(n, max(1, n // 10), replace=False); tr = np.setdiff1d(np.arange(n), val)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-4)
    f = lambda a, b: dec(enc(a) + enc(b)); best, state, bad = 1e18, None, 0
    for ep in range(epochs):
        enc.train(); dec.train(); opt.zero_grad(); loss = ((f(A[tr], B[tr]) - Y[tr]) ** 2).mean(); loss.backward(); opt.step()
        enc.eval(); dec.eval()
        with torch.no_grad(): vl = float(((f(A[val], B[val]) - Y[val]) ** 2).mean())
        if vl < best - 1e-6: best, bad, state = vl, 0, ([p.clone() for p in enc.parameters()], [p.clone() for p in dec.parameters()])
        else:
            bad += 1
            if bad > 40: break
    with torch.no_grad():
        for p, q in zip(list(enc.parameters()) + list(dec.parameters()), state[0] + state[1]): p.copy_(q)
        enc.eval(); dec.eval(); return f(T(Mte_a), T(Mte_b)).numpy() * rs


def noisy_singles(wd, run, singles, hn, ncell, seed=0):
    import anndata as ad, scanpy as sc, scipy.sparse as sp
    S = json.load(open(os.path.join(run, "summary.json")))
    FILES = {"Norman": "NormanWeissman2019_filtered.h5ad", "Wessels": "WesselsSatija2023.h5ad",
             "TianIPSC": "TianKampmann2019_iPSC.h5ad", "TianNeuron": "TianKampmann2019_day7neuron.h5ad"}
    idx = np.load(os.path.join(run, "subsample.npz"))["idx"]
    H = ad.read_h5ad(os.path.join(wd, "data", FILES[S["dataset"]]), backed="r"); X = H[idx].to_memory(); H.file.close(); X.var_names_make_unique()
    M = sp.csr_matrix(X.X if sp.issparse(X.X) else sp.csr_matrix(X.X), dtype=np.float32); M.data = np.round(M.data); X.X = M
    sc.pp.normalize_total(X, target_sum=1e4); sc.pp.log1p(X)
    col = {g: i for i, g in enumerate(X.var_names)}; Xh = sp.csr_matrix(X.X)[:, np.array([col[g] for g in hn])]
    lab = X.obs["perturbation"].astype(str).values; rng = np.random.default_rng(seed); out = {}
    for s in singles:
        w = np.flatnonzero(lab == s); w = rng.choice(w, min(ncell, len(w)), replace=False)
        out[s] = np.asarray(Xh[w].mean(0)).ravel().astype(np.float64)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--emb", required=True)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); p.add_argument("--noisy", type=int, default=0)
    p.add_argument("--wd", default=os.environ.get("FMTRANSFER_ROOT", ".")); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds); ctrl = expr["control"]
    R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64); DE = FP.de_sets(expr, "control", doubles, hn, 50); n = len(doubles)
    pairs = [(x, y) for _, x, y in doubles]
    sexpr = dict(expr)
    if a.noisy:
        ns = noisy_singles(a.wd, a.run, singles, hn, a.noisy); sexpr.update(ns)
    raw = {s: sexpr[s] - ctrl for s in singles}
    emb = load_string(a.emb, singles); D = len(next(iter(emb.values()))) if emb else 64
    E = {s: emb.get(s, np.zeros(D)) for s in singles}; cover = float(np.mean([s in emb for s in singles]))
    print(f"run={os.path.basename(a.run.rstrip('/'))} doubles={n} STRING coverage={cover:.2f} noisy={a.noisy}", flush=True)
    f20 = gene_feats(raw, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in pairs])
    base = PKE(raw, singles); H0 = (20, 1.0)
    have = [s for s in singles if s in emb]
    if cover >= 0.8:
        base.G[("string", H0)] = gk(gene_feats(E, 20), singles, 1.0)
    def make(extra):
        m = PKE.__new__(PKE); m.__dict__.update({k: v for k, v in base.__dict__.items() if k in ("singles", "gi", "G")})
        m.sum_m = list(base.sum_m) + extra; m.prod_m = list(base.prod_m) + extra; m.n = len(m.sum_m) + len(m.prod_m); return m
    meths = ["salt", "ridge_raw", "pke"] + ([] if os.environ.get("NO_PEPER") else ["peper"]) + ([] if os.environ.get("NO_SLIM") else ["slim"]) + (["pke_string"] if ("string", H0) in base.G else [])
    if a.noisy: meths = ["salt", "ridge_raw", "pke"] + (["pke_string"] if ("string", H0) in base.G else [])
    out = dict(string_coverage=cover, noisy=a.noisy)
    for scheme in ("random", "genegrp"):
        acc = {m: np.zeros((2, n)) for m in meths}
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                P = {"salt": np.zeros((len(te), R.shape[1])), "ridge_raw": ridge_pairs(Fraw[tr], Fraw[te], R[tr]),
                     "pke": make([]).fit([pairs[i] for i in tr], R[tr]).predict([pairs[i] for i in te])}
                if "pke_string" in meths: P["pke_string"] = make([("string", H0)]).fit([pairs[i] for i in tr], R[tr]).predict([pairs[i] for i in te])
                if "peper" in meths:
                    P["peper"] = peper_fit_predict(np.stack([raw[pairs[i][0]] for i in tr]), np.stack([raw[pairs[i][1]] for i in tr]), R[tr],
                                                   np.stack([raw[pairs[i][0]] for i in te]), np.stack([raw[pairs[i][1]] for i in te]), seed=sd)
                if "slim" in meths:
                    Ys = np.stack([expr[s] for s in singles]); Ps = np.stack([E[s] for s in singles])
                    Yd = np.stack([expr[doubles[i][0]] for i in tr]); Pd = np.stack([(E[pairs[i][0]] + E[pairs[i][1]]) / 2 for i in tr])
                    Pt = np.stack([(E[pairs[i][0]] + E[pairs[i][1]]) / 2 for i in te])
                    Yh = slim_fit_predict(Ys, Ps, Yd, Pd, Pt)
                    P["slim"] = np.stack([Yh[j] - expr[pairs[i][0]] - expr[pairs[i][1]] + ctrl for j, i in enumerate(te)])
                for m in meths:
                    for j, i in enumerate(te):
                        t = R[i, DE[i]]; e = t - P[m][j, DE[i]]; acc[m][:, i] += (e @ e, t @ t)
        ok = acc["salt"][1] > 0; Bt = np.random.default_rng(0).choice(np.flatnonzero(ok), size=(1000, ok.sum()))
        r2 = lambda A, ii: 1 - A[0, ii].sum(-1) / A[1, ii].sum(-1); res = {}
        for m in meths:
            d = r2(acc[m], Bt) - r2(acc["pke"], Bt)
            res[m] = dict(r2=float(r2(acc[m], np.flatnonzero(ok))), vs_pke=float(r2(acc[m], np.flatnonzero(ok)) - r2(acc["pke"], np.flatnonzero(ok))),
                          lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
        out[scheme] = res
        print(f"BASE {scheme:8s} " + " ".join(f"{m}={v['r2']:+.3f}" for m, v in res.items()), flush=True)
    fn = f"rev_noisy{a.noisy}.json" if a.noisy else "rev_base.json"
    json.dump(out, open(os.path.join(a.run, fn), "w"), indent=1); print("BASE_DONE", flush=True)


if __name__ == "__main__":
    main()
