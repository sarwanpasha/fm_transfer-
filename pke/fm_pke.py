"""PKE: pair-kernel ensemble for predicting genetic-interaction residuals of unseen double perturbations
from single-perturbation profiles (no foundation model). Baselines, the method, ablations, and GI-ranking metrics.
 baselines : mean_resid (training mean residual), ridge_raw (pair features), knn_pair (5-NN in pair kernel), kkr_raw (one RBF pair kernel)
 method    : pke (raw RBF at 9 (k,bw) + M random tanh feature maps), pke_nngp (closed-form infinite-width erf kernels, 4 scales)
 method+  : pke_anova / pke_anova_nngp (ANOVA: additive main-effect kernel + product interaction kernel)
 ablations : kkr_sum (additive pair kernel; cannot express interactions), kkr_nosym (unsymmetrised product), pke_rftM{1,3,16}
 GI ranking: per test double, Spearman(pred ||residual||, true ||residual||) and precision of top-20% strongest GIs (over pooled CV predictions)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_gira import gene_feats, ridge_pairs
from fm_gira2 import gk
from fm_gira3 import krr_loo

def kpca_feats(Kg, names, k):
    n = len(Kg); H = np.eye(n) - 1.0 / n; w, V = np.linalg.eigh(H @ Kg @ H); o = np.argsort(w)[::-1][:min(k, n - 1)]
    Z = V[:, o] * np.sqrt(np.maximum(w[o], 0)); Z /= np.maximum(Z.std(0, keepdims=True), 1e-12); return {s: Z[i] for i, s in enumerate(names)}

def erf_nngp(X, s2):   # Williams (1997): infinite-width one-hidden-layer erf network kernel, weight variance s2
    G = X @ X.T; d = 1 + 2 * s2 * np.diag(G); return (2 / np.pi) * np.arcsin(np.clip(2 * s2 * G / np.sqrt(np.outer(d, d)), -1, 1))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--M", type=int, default=8); p.add_argument("--rf-dim", type=int, default=512)
    p.add_argument("--no-center", action="store_true"); a = p.parse_args(); CENTER = not a.no_center
    import fmcore as F, fmpredict as FP
    from scipy.stats import spearmanr
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds)
    R = np.asarray(F.residual_exact(expr, "control", doubles), dtype=np.float64); DE = FP.de_sets(expr, "control", doubles, hn, 50)
    genes = sorted({g for _, x, y in doubles for g in (x, y)}); gidx = {g: i for i, g in enumerate(genes)}
    PAIRS = np.array([[gidx[x], gidx[y]] for _, x, y in doubles]); run = os.path.basename(a.run.rstrip("/"))
    raw = {s: expr[s] - expr["control"] for s in singles}; X = np.stack([raw[s] for s in singles]); X = X / (X.std() + 1e-12); G = X.shape[1]
    feats = {"raw": raw}
    Mmax = max(a.M, 16)
    for i in range(Mmax):
        W = np.random.default_rng(100 + i).standard_normal((G, a.rf_dim)) / np.sqrt(G); T = np.tanh(2.0 * X @ W)
        feats[f"rft{i}"] = {s: T[j] for j, s in enumerate(singles)}
    NG = {}
    for s2 in (0.5, 1.0, 2.0, 4.0):
        Kn = erf_nngp(X, s2 / 1.0); NG[s2] = Kn
    HP = [(k, bw) for k in (10, 20, 40) for bw in (0.5, 1.0, 2.0)]; H0 = (20, 1.0); GK = {}
    def K(m, h):
        if (m, h) not in GK:
            f = kpca_feats(NG[float(m[4:])], singles, h[0]) if m.startswith("nngp") else gene_feats(feats[m], h[0])
            GK[(m, h)] = gk(f, genes, h[1])
        return GK[(m, h)]
    def PK(kg, P, Q, mode="sym"):
        A, B, C, D = P[:, 0], P[:, 1], Q[:, 0], Q[:, 1]
        if mode == "sum": return 0.5 * (kg[np.ix_(A, C)] + kg[np.ix_(B, D)] + kg[np.ix_(A, D)] + kg[np.ix_(B, C)])
        if mode == "nosym": return kg[np.ix_(A, C)] * kg[np.ix_(B, D)]
        return 0.5 * (kg[np.ix_(A, C)] * kg[np.ix_(B, D)] + kg[np.ix_(A, D)] * kg[np.ix_(B, C)])
    def kern(members, P, Q, mode="sym"):   # members: (name, hp) or (name, hp, mode)
        return sum(PK(K(x[0], x[1]), P, Q, x[2] if len(x) > 2 else mode) for x in members) / len(members)
    def fit(members, mode="sym", center=None):
        def f(tr, te):
            mu = R[tr].mean(0) if (CENTER if center is None else center) else 0.0   # intercept: fit KRR on centred residuals, add training mean back
            _, A_, _ = krr_loo(kern(members, PAIRS[tr], PAIRS[tr], mode), R[tr] - mu); return kern(members, PAIRS[te], PAIRS[tr], mode) @ A_ + mu
        return f
    def fit_anova(prod, alphas=(0.0, 0.1, 0.25, 0.5)):   # K = a*K_main + (1-a)*K_int, a and lambda by closed-form LOO
        def f(tr, te):
            Ks_tr, Kp_tr = kern(prod, PAIRS[tr], PAIRS[tr], "sum"), kern(prod, PAIRS[tr], PAIRS[tr]); best = None
            mu = R[tr].mean(0) if CENTER else 0.0
            for al in alphas:
                loo, A_, _ = krr_loo(al * Ks_tr + (1 - al) * Kp_tr, R[tr] - mu)
                if best is None or loo < best[0]: best = (loo, A_, al)
            _, A_, al = best
            return (al * kern(prod, PAIRS[te], PAIRS[tr], "sum") + (1 - al) * kern(prod, PAIRS[te], PAIRS[tr])) @ A_ + mu
        return f
    def knn(tr, te, k=5):
        S = kern([("raw", H0)], PAIRS[te], PAIRS[tr]); nn = np.argsort(-S, 1)[:, :k]; return R[tr][nn].mean(1)
    f20 = gene_feats(raw, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for _, x, y in doubles])
    RAWHP = [("raw", h) for h in HP]; RFT = lambda M: [(f"rft{i}", H0) for i in range(M)]
    methods = {"mean_resid": lambda tr, te: np.repeat(R[tr].mean(0, keepdims=True), len(te), 0),
               "ridge_raw": lambda tr, te: ridge_pairs(Fraw[tr], Fraw[te], R[tr]), "knn_pair": knn,
               "kkr_raw": fit([("raw", H0)]), "kkr_sum": fit([("raw", H0)], "sum"), "kkr_nosym": fit([("raw", H0)], "nosym"),
               "pke": fit(RAWHP + RFT(a.M)), "pke_nngp": fit(RAWHP + [(f"nngp{s}", H0) for s in (0.5, 1.0, 2.0, 4.0)]),
               "pke_rftM1": fit([("raw", H0)] + RFT(1)), "pke_rftM3": fit([("raw", H0)] + RFT(3)), "pke_rftM16": fit([("raw", H0)] + RFT(16)),
               # ANOVA pair kernel: main effects g(a)+g(b) (sum kernel) + interaction h(a,b) (product kernel), both ensembled
               "pke_anova": fit([(m, h, "sum") for m, h in RAWHP + RFT(a.M)] + RAWHP + RFT(a.M)),
               "pke_anova_loo": fit_anova(RAWHP + RFT(a.M)),
               "pke_nocenter": fit(RAWHP + RFT(a.M), center=False),
               "pke_anova_nngp": fit([(m, h, "sum") for m, h in RAWHP] + RAWHP + [(f"nngp{s}", H0) for s in (0.5, 1.0, 2.0, 4.0)])}
    print(f"run={run} singles={len(singles)} doubles={len(doubles)} genes={len(genes)}", flush=True)
    out = {}
    for scheme in ("random", "genegrp"):
        n = len(doubles); acc = {}
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                for nm, fn in methods.items():
                    P = fn(tr, te); A = acc.setdefault(nm, dict(err_add=np.zeros(n), err_mod=np.zeros(n), pn=np.zeros(n), cnt=np.zeros(n)))
                    for j, i in enumerate(te):
                        t, pp = R[i, DE[i]], P[j, DE[i]]; A["err_add"][i] += t @ t; A["err_mod"][i] += (t - pp) @ (t - pp)
                        A["pn"][i] += np.linalg.norm(pp); A["cnt"][i] += 1
        idx = np.flatnonzero(acc["ridge_raw"]["cnt"] > 0); B = np.random.default_rng(0).choice(idx, size=(1000, len(idx)))
        r2 = lambda A, ii: 1 - A["err_mod"][ii].sum(-1) / A["err_add"][ii].sum(-1)
        tn = np.array([np.linalg.norm(R[i, DE[i]]) for i in range(n)]); top = tn[idx] >= np.quantile(tn[idx], 0.8); res = {}
        for nm, A in acc.items():
            d = r2(A, B) - r2(acc["ridge_raw"], B); v = float(r2(A, idx)); pn = A["pn"][idx] / A["cnt"][idx]
            rho = float(spearmanr(pn, tn[idx])[0]) if np.std(pn) > 0 else 0.0
            prec = float(top[np.argsort(-pn)[:top.sum()]].mean())
            res[nm] = dict(r2=v, d=float(v - r2(acc["ridge_raw"], idx)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), gi_rho=rho, gi_prec20=prec)
            print(f"P5 {scheme:8s} {nm:11s} R2={v:+.3f} d={res[nm]['d']:+.3f} lo={res[nm]['lo']:+.3f} hi={res[nm]['hi']:+.3f} rho={rho:+.3f} p20={prec:.3f}", flush=True)
        out[scheme] = res
    json.dump(out, open(os.path.join(a.run, "pke.json"), "w"), indent=1); print("PKE_DONE")

if __name__ == "__main__":
    main()
