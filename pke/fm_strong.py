"""Stronger kernel baselines on the folds and evaluation genes of fm_rev.py.
 poly_krr   symmetric polynomial kernel ridge on the concatenated single-perturbation features [z_a, z_b] (the kernel baseline of
            GrnPFN): k(u, v) = 1/2 [(c + <[z_a, z_b], [z_c, z_d]>/d)^q + (c + <[z_a, z_b], [z_d, z_c]>/d)^q], with the number of
            whitened principal components d in {10, 20, 40}, degree q in {2, 3}, offset c in {0.5, 1, 2} and penalty all chosen by
            leave-one-out error on the training doubles
 gp_ml      Gaussian process with a symmetric main-effect plus interaction RBF kernel on 20 whitened components, whose length scale,
            2 kernel variances and noise variance are fitted by maximizing the marginal likelihood (shared over output genes)
 mkl_align  multiple kernel learning over the 22 PKE pair kernels with nonnegative weights maximizing centered alignment to the
            training residuals (ALIGNF, as in pairwiseMKL), followed by kernel ridge with leave-one-out penalty
 pke, ridge_raw  recomputed on the same folds for paired comparisons
Output: <run>/strong.json and <run>/strong_pd.npz (same layout as rev_pd.npz).  Usage: fm_strong.py --run R [--seeds 0 1 2]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE
from fm_gira import gene_feats, ridge_pairs
from fm_gira3 import krr_loo
from scipy.optimize import minimize, nnls

LAMS = (1e-3, 1e-2, 1e-1, 1, 10, 100)


def sym_poly(Za, Zb, Zc, Zd, c, q):
    d = Za.shape[1] * 2
    s1 = (Za @ Zc.T + Zb @ Zd.T) / d; s2 = (Za @ Zd.T + Zb @ Zc.T) / d
    return 0.5 * ((c + s1) ** q + (c + s2) ** q)


def krr_fit_predict(Ktr, Kte, Y):
    mu = Y.mean(0); loo, A, lam = krr_loo(Ktr, Y - mu, LAMS); return mu + Kte @ A, loo


def poly_krr(feats, ptr, pte, Y):
    best = None
    for d, Z in feats.items():
        Za, Zb = np.stack([Z[a] for a, _ in ptr]), np.stack([Z[b] for _, b in ptr])
        Ta, Tb = np.stack([Z[a] for a, _ in pte]), np.stack([Z[b] for _, b in pte])
        for q in (2, 3):
            for c in (0.5, 1.0, 2.0):
                Ktr = sym_poly(Za, Zb, Za, Zb, c, q); Kte = sym_poly(Ta, Tb, Za, Zb, c, q)
                P, loo = krr_fit_predict(Ktr, Kte, Y)
                if best is None or loo < best[0]: best = (loo, P, dict(d=d, q=q, c=c))
    return best[1], best[2]


def rbf_parts(Za, Zb, Zc, Zd, ell):
    def g(X, Y): return np.exp(-((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1) / (2 * ell ** 2))
    AC, BD, AD, BC = g(Za, Zc), g(Zb, Zd), g(Za, Zd), g(Zb, Zc)
    return 0.5 * (AC + BD + AD + BC), 0.5 * (AC * BD + AD * BC)


def gp_ml(Z, ptr, pte, Y):
    Za, Zb = np.stack([Z[a] for a, _ in ptr]), np.stack([Z[b] for _, b in ptr])
    Ta, Tb = np.stack([Z[a] for a, _ in pte]), np.stack([Z[b] for _, b in pte])
    mu = Y.mean(0); Yc = Y - mu; n, p = Yc.shape; S = Yc @ Yc.T / p; d = Z[next(iter(Z))].shape[0]

    def nlml(th):
        ell, sm, si, sn = np.exp(th); Ks, Kp = rbf_parts(Za, Zb, Za, Zb, ell)
        K = sm * Ks + si * Kp + (sn + 1e-8) * np.eye(n)
        try: L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError: return 1e12
        Li = np.linalg.solve(L, np.eye(n)); Ki = Li.T @ Li
        return 0.5 * (np.sum(Ki * S) + 2 * np.log(np.diag(L)).sum())
    v0 = float(np.mean(np.diag(S)))
    best = None
    for l0 in (0.5, 1.0, 2.0):
        th0 = np.log([l0 * np.sqrt(d), 0.3 * v0, 0.3 * v0, 0.3 * v0])
        r = minimize(nlml, th0, method="L-BFGS-B", bounds=[(np.log(0.05 * np.sqrt(d)), np.log(20 * np.sqrt(d)))] + [(np.log(1e-6 * v0 + 1e-12), np.log(1e3 * v0 + 1e-12))] * 3)
        if best is None or r.fun < best.fun: best = r
    ell, sm, si, sn = np.exp(best.x)
    Ks, Kp = rbf_parts(Za, Zb, Za, Zb, ell); Kts, Ktp = rbf_parts(Ta, Tb, Za, Zb, ell)
    K = sm * Ks + si * Kp; A = np.linalg.solve(K + sn * np.eye(n), Yc)
    return mu + (sm * Kts + si * Ktp) @ A, dict(ell=float(ell / np.sqrt(d)), main=float(sm), inter=float(si), noise=float(sn))


def pair_kernels(base, P, Q):
    A, B, C, D = P[:, 0], P[:, 1], Q[:, 0], Q[:, 1]; out = []
    for m in base.sum_m:
        g = base.G[m]; out.append(0.5 * (g[np.ix_(A, C)] + g[np.ix_(B, D)] + g[np.ix_(A, D)] + g[np.ix_(B, C)]))
    for m in base.prod_m:
        g = base.G[m]; out.append(0.5 * (g[np.ix_(A, C)] * g[np.ix_(B, D)] + g[np.ix_(A, D)] * g[np.ix_(B, C)]))
    return out


def mkl_align(base, ptr, pte, Y):
    P, Q = base.idx(ptr), base.idx(pte); Ktr, Kte = pair_kernels(base, P, P), pair_kernels(base, Q, P)
    n = len(ptr); H = np.eye(n) - 1.0 / n; mu = Y.mean(0); Yc = Y - mu; Ky = Yc @ Yc.T
    Kc = [H @ K @ H for K in Ktr]; Kc = [K / (np.linalg.norm(K) + 1e-12) for K in Kc]
    M = np.array([[np.sum(a * b) for b in Kc] for a in Kc]); av = np.array([np.sum(a * Ky) for a in Kc]) / (np.linalg.norm(Ky) + 1e-12)
    L = np.linalg.cholesky(M + 1e-8 * np.eye(len(M))); w, _ = nnls(L.T, np.linalg.solve(L, av))
    if w.sum() <= 0: w = np.ones(len(Ktr))
    w = w / w.sum(); scale = [1.0 / (np.linalg.norm(H @ K @ H) + 1e-12) for K in Ktr]
    K = sum(wi * s * Ki for wi, s, Ki in zip(w, scale, Ktr)); Kt = sum(wi * s * Ki for wi, s, Ki in zip(w, scale, Kte))
    P_, _ = krr_fit_predict(K, Kt, Y); return P_, dict(n_nonzero=int((w > 1e-6).sum()), w_max=float(w.max()))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds); ctrl = expr["control"]
    R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64); n, G = R.shape
    raw = {s: expr[s] - ctrl for s in singles}; pairs = [(x, y) for _, x, y in doubles]; pos = {g: i for i, g in enumerate(hn)}
    GS = {"de_obs": FP.de_sets(expr, "control", doubles, hn, 50)}; s50 = []
    for d, x, y in doubles:
        s = np.maximum(np.abs(raw[x]), np.abs(raw[y])).astype(float)
        for g in (x, y):
            if g in pos: s[pos[g]] = -np.inf
        s50.append(np.argpartition(-s, 50)[:50])
    GS["singles50"] = s50; GS["hvg"] = [np.arange(G)] * n
    f20 = gene_feats(raw, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in pairs])
    feats = {d: gene_feats(raw, d) for d in (10, 20, 40)}; base = PKE(raw, singles)
    meth = ["ridge_raw", "pke", "poly_krr", "gp_ml", "mkl_align", "mean_resid"]; out = dict(n_doubles=n, doubles=[d for d, _, _ in doubles], hp={}); pd = {}
    for scheme in ("random", "genegrp"):
        acc = {(m, g, c): np.zeros((3, n)) for m in meth for g in GS for c in ("all", "u1", "u2")}; hps = []
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                trg = {g for i in tr for g in pairs[i]}; ptr, pte = [pairs[i] for i in tr], [pairs[i] for i in te]
                mu = R[tr].mean(0); P = {"mean_resid": np.repeat(mu[None], len(te), 0), "ridge_raw": ridge_pairs(Fraw[tr], Fraw[te], R[tr])}
                m = PKE.__new__(PKE); m.__dict__.update(base.__dict__); P["pke"] = m.fit(ptr, R[tr]).predict(pte)
                P["poly_krr"], h1 = poly_krr(feats, ptr, pte, R[tr]); P["gp_ml"], h2 = gp_ml(feats[20], ptr, pte, R[tr])
                P["mkl_align"], h3 = mkl_align(base, ptr, pte, R[tr]); hps.append(dict(poly=h1, gp=h2, mkl=h3))
                for j, i in enumerate(te):
                    nu = sum(g not in trg for g in pairs[i]); cats = ["all"] + ([f"u{nu}"] if scheme == "genegrp" and nu in (1, 2) else [])
                    for gname, sets in GS.items():
                        ix = sets[i]; t = R[i, ix]; tc = t - mu[ix]
                        for mm in meth:
                            e = t - P[mm][j, ix]
                            for c in cats: acc[(mm, gname, c)][:, i] += (e @ e, t @ t, tc @ tc)
        res = {}
        for (mm, gname, c), A in acc.items():
            ok = A[1] > 0
            if ok.sum() == 0: continue
            res.setdefault(gname, {}).setdefault(c, {})[mm] = dict(r2=float(1 - A[0, ok].sum() / A[1, ok].sum()), r2_cen=float(1 - A[0, ok].sum() / A[2, ok].sum()), n=int(ok.sum()))
            pd[f"{scheme}|{gname}|{c}|{mm}"] = A
        out[scheme] = res; out["hp"][scheme] = hps
        for gname in res:
            for c in res[gname]:
                r = res[gname][c]; print(f"STRONG {scheme:8s} {gname:9s} {c:3s} " + " ".join(f"{k}={v['r2']:+.3f}" for k, v in r.items()), flush=True)
    json.dump(out, open(os.path.join(a.run, "strong.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.run, "strong_pd.npz"), **{k.replace("|", "__"): v for k, v in pd.items()}, doubles=np.array(out["doubles"]))
    print("STRONG_DONE", flush=True)


if __name__ == "__main__":
    main()
