"""Revision analyses for one run (same folds and scoring as fm_pke.py).
 baselines  : additive (r=0), mean_resid (intercept only), knn_pair, ridge_raw, PKE, PKE-fANOVA (zero-mean gene kernels)
 gene sets  : de_obs (top-50 |x_ab-x0|, as before), singles50 (top-50 by max(|d_a|,|d_b|), no test label), hvg (all genes)
 metrics    : R2 vs additive, R2 vs intercept-only (centered), per double SSE saved for pooled bootstrap across subsamples
 genegrp    : held-out doubles split into 1 unseen gene vs 2 unseen genes
 GP         : posterior variance of PKE, Spearman(variance, error) on held-out doubles
 shares     : stability of representer shares (lambda x0.1/x10, 90% bootstraps of training doubles); fANOVA pair-specific shares
 modes      : PKE vs ridge GI-mode metrics with paired bootstrap and Holm correction
Output: <run>/rev.json and <run>/rev_pd.npz
Usage: fm_rev.py --run <run>"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE, mode_metrics
from fm_gira import gene_feats, ridge_pairs


class PKEF(PKE):
    """PKE with zero-mean (functional-ANOVA) gene kernels: k_c = H k H over the perturbed genes. Main-effect functions and
    interaction functions are then orthogonal under the uniform measure on genes, so the main/interaction split is identifiable."""
    def __init__(self, raw, singles):
        super().__init__(raw, singles)
        n = len(self.singles); Hm = np.eye(n) - 1.0 / n
        self.G = {k: Hm @ v @ Hm for k, v in self.G.items()}


def post_var(m, pairs):
    Q = m.idx(pairs); Ks, Kp = m.kparts(Q, Q); Kqq = np.diag(Ks + Kp)
    Kst, Kpt = m.kparts(Q, m.P); Kq = Kst + Kpt; Ks2, Kp2 = m.kparts(m.P, m.P); Ktr = Ks2 + Kp2
    Hinv = np.linalg.inv(Ktr + m.lam * np.eye(len(Ktr)))
    return np.maximum(Kqq - np.einsum("ij,jk,ik->i", Kq, Hinv, Kq), 0)


def shares(m, pair):
    f, fm, fi, K = m.predict([pair], parts=True); d = f[0] - m.mu; den = d @ d + 1e-12
    return K[0] * (m.A @ d) / den, float(fm[0] @ d / den), float(fi[0] @ d / den)


def fit_lam(cls, raw, singles, pairs, Y, lam):
    m = cls(raw, singles); m.P = m.idx(pairs); m.mu = Y.mean(0); Ks, Kp = m.kparts(m.P, m.P)
    m.A = np.linalg.solve(Ks + Kp + lam * np.eye(len(Ks)), Y - m.mu); m.lam = lam; return m


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    from scipy.stats import spearmanr
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds); ctrl = expr["control"]
    R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64); n, G = R.shape
    raw = {s: expr[s] - ctrl for s in singles}; pairs = [(x, y) for _, x, y in doubles]
    pos = {g: i for i, g in enumerate(hn)}
    GS = {"de_obs": FP.de_sets(expr, "control", doubles, hn, 50)}
    s50 = []
    for d, x, y in doubles:
        s = np.maximum(np.abs(raw[x]), np.abs(raw[y])).astype(float)
        for g in (x, y):
            if g in pos: s[pos[g]] = -np.inf
        s50.append(np.argpartition(-s, 50)[:50])
    GS["singles50"] = s50; GS["hvg"] = [np.arange(G)] * n
    f20 = gene_feats(raw, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in pairs])
    genes = sorted({g for pr in pairs for g in pr}); gi = {g: i for i, g in enumerate(genes)}
    base = PKE(raw, singles); K0 = base.G[("raw", (20, 1.0))]; sidx = {s: i for i, s in enumerate(base.singles)}
    def knn(tr, te, k=5):
        S = np.array([[0.5 * (K0[sidx[pairs[i][0]], sidx[pairs[j][0]]] * K0[sidx[pairs[i][1]], sidx[pairs[j][1]]] +
                              K0[sidx[pairs[i][0]], sidx[pairs[j][1]]] * K0[sidx[pairs[i][1]], sidx[pairs[j][0]]]) for j in tr] for i in te])
        nn = np.argsort(-S, 1)[:, :k]; return R[tr][nn].mean(1)
    meth = ["additive", "mean_resid", "knn_pair", "ridge_raw", "pke", "pke_fanova"]
    out = dict(n_doubles=n, doubles=[d for d, _, _ in doubles]); pd = {}
    for scheme in ("random", "genegrp"):
        acc = {(m, g, c): np.zeros((3, n)) for m in meth for g in GS for c in ("all", "u1", "u2")}  # rows: sse, ss_add, ss_cen
        gpv, gpe = [], []
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                trg = {g for i in tr for g in pairs[i]}
                mu = R[tr].mean(0); P = {"additive": np.zeros((len(te), G)), "mean_resid": np.repeat(mu[None], len(te), 0),
                                         "knn_pair": knn(tr, te), "ridge_raw": ridge_pairs(Fraw[tr], Fraw[te], R[tr])}
                m = PKE(raw, singles).fit([pairs[i] for i in tr], R[tr]); P["pke"] = m.predict([pairs[i] for i in te])
                P["pke_fanova"] = PKEF(raw, singles).fit([pairs[i] for i in tr], R[tr]).predict([pairs[i] for i in te])
                if scheme == "random":
                    v = post_var(m, [pairs[i] for i in te]); e = ((R[te] - P["pke"]) ** 2).sum(1)
                    gpv += list(v); gpe += list(e / np.maximum(((R[te] - mu) ** 2).sum(1), 1e-12))
                for j, i in enumerate(te):
                    nu = sum(g not in trg for g in pairs[i]); cats = ["all"] + ([f"u{nu}"] if scheme == "genegrp" and nu in (1, 2) else [])
                    for gname, sets in GS.items():
                        ix = sets[i]; t = R[i, ix]; tc = t - mu[ix]
                        for mm in meth:
                            e = t - P[mm][j, ix]
                            for c in cats:
                                acc[(mm, gname, c)][:, i] += (e @ e, t @ t, tc @ tc)
        res = {}
        for (mm, gname, c), A in acc.items():
            ok = A[1] > 0
            if ok.sum() == 0: continue
            res.setdefault(gname, {}).setdefault(c, {})[mm] = dict(r2=float(1 - A[0, ok].sum() / A[1, ok].sum()),
                                                                  r2_cen=float(1 - A[0, ok].sum() / A[2, ok].sum()), n=int(ok.sum()))
            pd[f"{scheme}|{gname}|{c}|{mm}"] = A
        out[scheme] = res
        if scheme == "random":
            out["gp"] = dict(rho_var_relerr=float(spearmanr(gpv, gpe)[0]),
                             relerr_low_var=float(np.median(np.array(gpe)[np.array(gpv) <= np.median(gpv)])),
                             relerr_high_var=float(np.median(np.array(gpe)[np.array(gpv) > np.median(gpv)])))
        for gname in res:
            for c in res[gname]:
                r = res[gname][c]
                print(f"REV {scheme:8s} {gname:9s} {c:3s} n={r['pke']['n']:4d} " + " ".join(f"{k}={v['r2']:+.3f}/{v['r2_cen']:+.3f}" for k, v in r.items()), flush=True)
    print("GP", out["gp"], flush=True)
    # ---------------- representer-share stability and fANOVA shares (leave-one-double-out explanations)
    rng = np.random.default_rng(0); stab = []; fan = {}
    strength = np.linalg.norm(R - R.mean(0), axis=1); order = np.argsort(-strength)[:20]
    for i in order:
        tr = np.array([j for j in range(n) if j != i]); pr = [pairs[j] for j in tr]
        m = PKE(raw, singles).fit(pr, R[tr]); s0, mf0, if0 = shares(m, pairs[i]); top0 = set(np.argsort(-s0)[:5])
        row = dict(pair=doubles[i][0], int_share=if0)
        for nm, lam in (("lam_x0.1", m.lam * 0.1), ("lam_x10", m.lam * 10)):
            s1, _, _ = shares(fit_lam(PKE, raw, singles, pr, R[tr], lam), pairs[i])
            row[nm] = dict(rho=float(spearmanr(s0, s1)[0]), top5_jacc=len(top0 & set(np.argsort(-s1)[:5])) / len(top0 | set(np.argsort(-s1)[:5])))
        bs = []
        for b in range(20):
            keep = np.sort(rng.choice(len(tr), int(0.9 * len(tr)), replace=False))
            mb = PKE(raw, singles).fit([pr[k] for k in keep], R[tr][keep]); sb, _, _ = shares(mb, pairs[i])
            full = np.full(len(tr), np.nan); full[keep] = sb; ok = ~np.isnan(full)
            t0 = set(np.argsort(-np.where(ok, s0, -np.inf))[:5]); t1 = set(np.argsort(-np.where(ok, full, -np.inf))[:5])
            bs.append((spearmanr(s0[ok], full[ok])[0], len(t0 & t1) / len(t0 | t1)))
        row["boot90"] = dict(rho=float(np.median([x[0] for x in bs])), top5_jacc=float(np.median([x[1] for x in bs])))
        mfa = PKEF(raw, singles).fit(pr, R[tr]); _, mff, iff = shares(mfa, pairs[i]); row["int_share_fanova"] = iff
        stab.append(row)
        print(f"SHARE {doubles[i][0]:20s} int={if0:.2f} int_fanova={iff:.2f} lam0.1 rho={row['lam_x0.1']['rho']:.2f} lam10 rho={row['lam_x10']['rho']:.2f} boot rho={row['boot90']['rho']:.2f} J={row['boot90']['top5_jacc']:.2f}", flush=True)
    out["share_stability"] = stab
    # global variance fractions of main vs interaction part over all pairs of measured singles (fANOVA model, all doubles)
    mfa = PKEF(raw, singles).fit(pairs, R); allp = [(x, y) for k, x in enumerate(singles) for y in singles[k + 1:]]
    _, fm, fi, _ = mfa.predict(allp, parts=True); vm, vi = float((fm ** 2).sum()), float((fi ** 2).sum())
    out["fanova_global"] = dict(main_frac=vm / (vm + vi), int_frac=vi / (vm + vi), cross=float((fm * fi).sum() / (vm + vi)))
    print("FANOVA_GLOBAL", out["fanova_global"], flush=True)
    # ---------------- GI modes: paired bootstrap, Holm
    true = [mode_metrics(raw[x], raw[y], expr[d] - ctrl) for d, x, y in doubles]; OOF = {k: np.zeros_like(R) for k in ("pke", "ridge")}; cnt = np.zeros(n)
    for sd in a.seeds:
        for tr, te in FP.splits(doubles, "random", 5, sd):
            OOF["pke"][te] += PKE(raw, singles).fit([pairs[i] for i in tr], R[tr]).predict([pairs[i] for i in te])
            OOF["ridge"][te] += ridge_pairs(Fraw[tr], Fraw[te], R[tr]); cnt[te] += 1
    from sklearn.metrics import roc_auc_score
    pred = {k: [mode_metrics(raw[x], raw[y], raw[x] + raw[y] + (OOF[k] / cnt[:, None])[i]) for i, (x, y) in enumerate(pairs)] for k in OOF}
    tm = np.array([t["magnitude"] for t in true]); tn = np.array([t["neomorphism"] for t in true]); td = np.array([t["dominance"] for t in true])
    lab = dict(syn=tm > np.quantile(tm, .8), buf=tm < np.quantile(tm, .2), neo=tn > np.quantile(tn, .8))
    def metrics(k, ii):
        pm = np.array([pred[k][i]["magnitude"] for i in ii]); pdm = np.array([pred[k][i]["dominance"] for i in ii]); pn = np.array([pred[k][i]["neomorphism"] for i in ii])
        r = dict(mag=spearmanr(pm, tm[ii])[0], dom=spearmanr(pdm, td[ii])[0])
        for nm, sc in (("syn", pm), ("buf", -pm), ("neo", pn)):
            y = lab[nm][ii]; r[nm] = roc_auc_score(y, sc) if 0 < y.sum() < len(y) else np.nan
        return r
    full = np.arange(n); b0 = {k: metrics(k, full) for k in OOF}; diffs = {m: [] for m in b0["pke"]}
    for b in range(2000):
        ii = rng.choice(n, n); mp, mr = metrics("pke", ii), metrics("ridge", ii)
        for m in diffs: diffs[m].append(mp[m] - mr[m])
    pv = {m: float(min(1, 2 * min(np.nanmean(np.array(v) <= 0), np.nanmean(np.array(v) >= 0)))) for m, v in diffs.items()}
    order_p = sorted(pv, key=pv.get); holm = {}; run_max = 0
    for r_, m in enumerate(order_p):
        run_max = max(run_max, min(1, (len(pv) - r_) * pv[m])); holm[m] = run_max
    out["modes"] = {m: dict(pke=float(b0["pke"][m]), ridge=float(b0["ridge"][m]), diff_lo=float(np.nanpercentile(diffs[m], 2.5)),
                            diff_hi=float(np.nanpercentile(diffs[m], 97.5)), p=pv[m], p_holm=holm[m]) for m in diffs}
    print("MODES", json.dumps(out["modes"]), flush=True)
    json.dump(out, open(os.path.join(a.run, "rev.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(a.run, "rev_pd.npz"), **{k.replace("|", "__"): v for k, v in pd.items()}, doubles=np.array(out["doubles"]))
    print("REV_DONE", flush=True)


if __name__ == "__main__":
    main()
