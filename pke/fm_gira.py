"""GIRA pilot: GI-aware readout via symmetric pair-kernel ridge with learned kernel mixing.
Predict the exact interaction residual R_ab = mu_ab - mu_a - mu_b + mu_0 (HVG space) of held-out doubles
from single-perturbation geometry. Same splits/scoring as fmpredict.evaluate (top-50 DE genes per double,
R2 vs additive = 1 - SSE_model/SSE_additive, pooled over doubles and seeds)."""
import argparse, glob, json, os, sys
import numpy as np

def gene_feats(eff, k):
    """eff: dict single -> vector. centred, PCA-k (singles only), whitened. returns dict."""
    names = sorted(eff); E = np.stack([eff[s] for s in names]).astype(np.float64); E -= E.mean(0)
    U, S, Vt = np.linalg.svd(E, full_matrices=False); k = min(k, len(names) - 1)
    Z = U[:, :k] * S[:k]; Z /= np.maximum(Z.std(0, keepdims=True), 1e-12)
    return {s: Z[i] for i, s in enumerate(names)}

def gene_kernel(feat, genes, kind="rbf"):
    Z = np.stack([feat[g] for g in genes])
    if kind == "cos":
        Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12); return Zn @ Zn.T
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1); s2 = np.median(D[D > 0])
    return np.exp(-D / s2)

def pair_kernel(kg, gi, P, Q):
    """symmetric Kronecker pair kernel between pair lists P, Q (index pairs into gene list)."""
    a, b = gi[P[:, 0]], gi[P[:, 1]]; c, d = gi[Q[:, 0]], gi[Q[:, 1]]
    return 0.5 * (kg[np.ix_(a, c)] * kg[np.ix_(b, d)] + kg[np.ix_(a, d)] * kg[np.ix_(b, c)])

def krr_fit_predict(Ktr, Kte, Y, lams=(1e-3, 1e-2, 1e-1, 1, 10)):
    """kernel ridge; lambda by closed-form LOO on the training pairs. returns predictions, best lam, loo mse."""
    w, V = np.linalg.eigh(Ktr); best = None
    for lam in lams:
        Hinv = (V / (w + lam)) @ V.T; A = Hinv @ Y  # LOO residual of KRR = A_i / Hinv_ii
        loo = ((A / np.diag(Hinv)[:, None]) ** 2).mean()
        if best is None or loo < best[2]: best = (lam, A, loo)
    return Kte @ best[1], best[0], best[2]

def ridge_pairs(Ftr, Fte, Y, alphas=np.logspace(-2, 4, 13)):
    from sklearn.linear_model import RidgeCV
    return RidgeCV(alphas=alphas).fit(Ftr, Y).predict(Fte)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--k", type=int, default=20)
    p.add_argument("--emb", nargs="*", default=[]); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = p.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fmcore as F, fmpredict as FP
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds)
    R = np.asarray(F.residual_exact(expr, "control", doubles), dtype=np.float64)
    DE = FP.de_sets(expr, "control", doubles, hn, 50)
    genes = sorted({g for _, x, y in doubles for g in (x, y)}); gi_all = {g: i for i, g in enumerate(genes)}
    PAIRS = np.array([[gi_all[x], gi_all[y]] for _, x, y in doubles]); ident = np.arange(len(genes))
    reps = {"raw": {s: expr[s] - expr["control"] for s in singles}}
    for f in a.emb:
        E = np.load(f, allow_pickle=True); zc = {c: i for i, c in enumerate(E["conds"])}
        nm = os.path.basename(f).replace("_" + os.path.basename(a.run.rstrip("/")) + "_cell.npz", "")
        reps[nm] = {s: E["emb"][zc[s]].astype(np.float64) - E["emb"][zc["control"]] for s in singles}
    feats = {nm: gene_feats(e, a.k) for nm, e in reps.items()}
    KG = {nm: gene_kernel(f, genes) for nm, f in feats.items()}
    methods = {}
    def add(name, fn): methods[name] = fn
    # baseline replicating fmpredict rawExpr ridge
    Fraw = np.stack([FP.pair_features(feats["raw"][x], feats["raw"][y]) for _, x, y in doubles])
    add("ridge_raw", lambda tr, te: ridge_pairs(Fraw[tr], Fraw[te], R[tr]))
    for nm in KG:
        add(f"kkr_{nm}", (lambda nm: lambda tr, te: krr_fit_predict(pair_kernel(KG[nm], ident, PAIRS[tr], PAIRS[tr]),
                                                                    pair_kernel(KG[nm], ident, PAIRS[te], PAIRS[tr]), R[tr])[0])(nm))
    # GIRA: learned convex mixture of gene kernels (raw + each FM) chosen by LOO on training pairs
    def gira(tr, te, pool):
        Ks = {nm: (pair_kernel(KG[nm], ident, PAIRS[tr], PAIRS[tr]), pair_kernel(KG[nm], ident, PAIRS[te], PAIRS[tr])) for nm in pool}
        best = None
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        others = [m for m in pool if m != "raw"]
        cands = [{"raw": 1.0}] + [{"raw": 1 - w, o: w} for o in others for w in grid[1:]]
        for wts in cands:
            Ktr = sum(w * Ks[m][0] for m, w in wts.items()); Kte = sum(w * Ks[m][1] for m, w in wts.items())
            P, lam, loo = krr_fit_predict(Ktr, Kte, R[tr])
            if best is None or loo < best[0]: best = (loo, P, wts)
        return best[1], best[2]
    pools = {"gira_raw+fm": list(KG)}
    out = {}
    for scheme in ("random", "genegrp"):
        n = len(doubles); acc = {}
        chosen = []
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                preds = {nm: fn(tr, te) for nm, fn in methods.items()}
                for pn, pool in pools.items():
                    P_, w_ = gira(tr, te, pool); preds[pn] = P_; chosen.append(w_)
                for nm, P in preds.items():
                    A = acc.setdefault(nm, dict(err_add=np.zeros(n), err_mod=np.zeros(n), cnt=np.zeros(n)))
                    for j, i in enumerate(te):
                        t, pp = R[i, DE[i]], P[j, DE[i]]
                        A["err_add"][i] += t @ t; A["err_mod"][i] += (t - pp) @ (t - pp); A["cnt"][i] += 1
        res = {}
        seen = acc["ridge_raw"]["cnt"] > 0; idx = np.flatnonzero(seen); rng = np.random.default_rng(0)
        B = rng.choice(idx, size=(1000, len(idx)))
        r2 = lambda A, ii: 1 - A["err_mod"][ii].sum(-1) / A["err_add"][ii].sum(-1)
        for nm, A in acc.items():
            d = r2(A, B) - r2(acc["ridge_raw"], B)
            res[nm] = dict(r2=float(r2(A, idx)), d_vs_ridge_raw=float(r2(A, idx) - r2(acc["ridge_raw"], idx)),
                           d_ci=list(np.percentile(d, [2.5, 97.5])), p_better=float((d > 0).mean()))
        from collections import Counter
        res["_gira_weights"] = dict(Counter(json.dumps(w, sort_keys=True) for w in chosen).most_common(5))
        out[scheme] = res
        for nm, v in sorted(res.items(), key=lambda kv: -kv[1]["r2"] if not kv[0].startswith("_") else 9):
            if nm.startswith("_"): print(f"{scheme:8s} {nm} {v}"); continue
            print(f"{scheme:8s} {nm:28s} R2={v['r2']:+.3f}  d_vs_ridge_raw={v['d_vs_ridge_raw']:+.3f} CI=[{v['d_ci'][0]:+.3f},{v['d_ci'][1]:+.3f}] p>0={v['p_better']:.2f}", flush=True)
    json.dump(out, open(os.path.join(a.run, "gira_pilot.json"), "w"), indent=1, default=float)
    print("GIRA_DONE")

if __name__ == "__main__":
    main()
