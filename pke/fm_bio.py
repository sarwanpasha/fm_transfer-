"""Biology use cases + interpretability for PKE (main method = pke_anova_nngp, same construction as fm_pke.py).

Outputs (in <run>/bio/):
  modes_cv.json        GI-mode prediction on held-out doubles (out-of-fold, random + genegrp splits): magnitude / dominance /
                       neomorphism metrics (Norman et al. 2019 style) from predicted vs observed doubles; PKE vs ridge vs additive.
  screen.tsv           in-silico screen of ALL untested pairs among measured singles (fit on all observed doubles):
                       predicted interaction strength, ANOVA split (main-effect vs pairwise part), predicted mode metrics.
  propensity.tsv       gene-level interaction propensity ||g(a)|| from the main-effect (sum-kernel) component, with the empirical
                       mean residual norm of the gene's observed doubles (in-sample) and a leave-gene-out check.
  explain.json         for top predicted pairs and top observed pairs: representer decomposition (training pairs that support the
                       prediction, with shares), ANOVA split, dominant residual programs with top genes.
  programs.tsv         residual programs (PCA of observed interaction residuals): top +/- loading genes.
Usage: fm_bio.py --run results/Norman_s0_mc500 [--seeds 0]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_gira import gene_feats, ridge_pairs
from fm_gira2 import gk
from fm_gira3 import krr_loo
from fm_pke import kpca_feats, erf_nngp

HP = [(k, bw) for k in (10, 20, 40) for bw in (0.5, 1.0, 2.0)]; H0 = (20, 1.0); S2 = (0.5, 1.0, 2.0, 4.0)

def mode_metrics(da, db, dab):
    """Norman-style: fit dab ~ ca*da + cb*db (least squares over genes). Returns magnitude, dominance, neomorphism, fit corr."""
    X = np.stack([da, db], 1); c, *_ = np.linalg.lstsq(X, dab, rcond=None); fit = X @ c
    mag = float(np.linalg.norm(fit) / max(np.linalg.norm(da + db), 1e-12))          # >1 synergy, <1 buffering
    dom = float(abs(np.log10(max(abs(c[0]), 1e-6) / max(abs(c[1]), 1e-6))))          # one gene dominates
    r = float(np.corrcoef(fit, dab)[0, 1]) if np.std(fit) > 0 and np.std(dab) > 0 else 0.0
    return dict(magnitude=mag, dominance=dom, neomorphism=1 - r, ca=float(c[0]), cb=float(c[1]))

class PKE:
    """pke_anova_nngp: equal-weight average of 9 raw-RBF sum (main-effect) kernels + 9 raw-RBF and 4 erf-NNGP product
    (interaction) kernels, symmetric over pair order, centred KRR with closed-form LOO lambda."""
    def __init__(self, raw, singles):
        self.singles = list(singles); self.gi = {g: i for i, g in enumerate(self.singles)}
        X = np.stack([raw[s] for s in self.singles]); X = X / (X.std() + 1e-12)
        self.G = {}
        for h in HP: self.G[("raw", h)] = gk(gene_feats(raw, h[0]), self.singles, h[1])
        for s2 in S2: self.G[(f"nngp{s2}", H0)] = gk(kpca_feats(erf_nngp(X, s2), self.singles, H0[0]), self.singles, H0[1])
        self.sum_m = [("raw", h) for h in HP]; self.prod_m = [("raw", h) for h in HP] + [(f"nngp{s}", H0) for s in S2]
        self.n = len(self.sum_m) + len(self.prod_m)
    def idx(self, pairs): return np.array([[self.gi[a], self.gi[b]] for a, b in pairs])
    def kparts(self, P, Q):
        A, B, C, D = P[:, 0], P[:, 1], Q[:, 0], Q[:, 1]
        Ks = sum(0.5 * (g[np.ix_(A, C)] + g[np.ix_(B, D)] + g[np.ix_(A, D)] + g[np.ix_(B, C)]) for g in (self.G[m] for m in self.sum_m)) / self.n
        Kp = sum(0.5 * (g[np.ix_(A, C)] * g[np.ix_(B, D)] + g[np.ix_(A, D)] * g[np.ix_(B, C)]) for g in (self.G[m] for m in self.prod_m)) / self.n
        return Ks, Kp
    def fit(self, pairs, Y):
        self.P = self.idx(pairs); self.mu = Y.mean(0); Ks, Kp = self.kparts(self.P, self.P)
        _, self.A, self.lam = krr_loo(Ks + Kp, Y - self.mu); return self
    def predict(self, pairs, parts=False):
        Q = self.idx(pairs); Ks, Kp = self.kparts(Q, self.P); fm, fi = Ks @ self.A, Kp @ self.A
        return (self.mu + fm + fi, fm, fi, Ks + Kp) if parts else self.mu + fm + fi
    def propensity(self, genes):
        """main-effect function g(x) = sum_i 0.5*[k(x,a_i)+k(x,b_i)] A_i / n  (so f_main(a,b) = g(a)+g(b))."""
        q = np.array([self.gi[g] for g in genes]); a, b = self.P[:, 0], self.P[:, 1]
        W = sum(0.5 * (self.G[m][np.ix_(q, a)] + self.G[m][np.ix_(q, b)]) for m in self.sum_m) / self.n
        return W @ self.A

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--top", type=int, default=15); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    od = os.path.join(a.run, "bio"); os.makedirs(od, exist_ok=True); run = os.path.basename(a.run.rstrip("/"))
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds); ctrl = expr["control"]
    R = np.asarray(F.residual_exact(expr, "control", doubles), dtype=np.float64); DE = FP.de_sets(expr, "control", doubles, hn, 50)
    raw = {s: expr[s] - ctrl for s in singles}; pairs = [(x, y) for _, x, y in doubles]; n = len(doubles)
    print(f"run={run} singles={len(singles)} doubles={n} genes={len(hn)}", flush=True)
    # ---------- (1) GI-mode prediction on held-out doubles (out-of-fold) ----------
    f20 = gene_feats(raw, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in pairs])
    true = [mode_metrics(raw[x], raw[y], expr[d] - ctrl) for d, x, y in doubles]
    modes = {}
    for scheme in ("random", "genegrp"):
        OOF = {m: np.zeros_like(R) for m in ("pke", "ridge")}; cnt = np.zeros(n)
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                mdl = PKE(raw, singles).fit([pairs[i] for i in tr], R[tr])
                OOF["pke"][te] += mdl.predict([pairs[i] for i in te]); OOF["ridge"][te] += ridge_pairs(Fraw[tr], Fraw[te], R[tr]); cnt[te] += 1
        res = {}
        for m in ("pke", "ridge", "additive"):
            P_ = np.zeros_like(R) if m == "additive" else OOF[m] / np.maximum(cnt, 1)[:, None]
            pred = [mode_metrics(raw[x], raw[y], raw[x] + raw[y] + P_[i]) for i, (x, y) in enumerate(pairs)]
            r = {}
            for k in ("magnitude", "dominance", "neomorphism"):
                t_, p_ = np.array([d[k] for d in true]), np.array([d[k] for d in pred])
                r[f"rho_{k}"] = float(spearmanr(p_, t_)[0]) if np.std(p_) > 0 else 0.0
            tm = np.array([d["magnitude"] for d in true]); pm = np.array([d["magnitude"] for d in pred])
            syn, buf = tm > np.quantile(tm, 0.8), tm < np.quantile(tm, 0.2)          # top/bottom quintile = synergy / buffering
            r["auc_synergy"] = float(roc_auc_score(syn, pm)) if np.std(pm) > 0 else 0.5
            r["auc_buffering"] = float(roc_auc_score(buf, -pm)) if np.std(pm) > 0 else 0.5
            tn = np.array([d["neomorphism"] for d in true]); pn = np.array([d["neomorphism"] for d in pred])
            r["auc_neomorphic"] = float(roc_auc_score(tn > np.quantile(tn, 0.8), pn)) if np.std(pn) > 0 else 0.5
            res[m] = r; print(f"MODE {scheme:8s} {m:8s} " + " ".join(f"{k}={v:+.3f}" for k, v in r.items()), flush=True)
        modes[scheme] = res
    json.dump(dict(modes=modes, true_modes={d: t for (d, _, _), t in zip(doubles, true)}), open(os.path.join(od, "modes_cv.json"), "w"), indent=1)
    # ---------- (2) fit on all observed doubles; screen all untested pairs ----------
    mdl = PKE(raw, singles).fit(pairs, R); seen = {frozenset(pq) for pq in pairs}
    cand = [(x, y) for i, x in enumerate(singles) for y in singles[i + 1:] if frozenset((x, y)) not in seen]
    rows = []
    for j in range(0, len(cand), 2000):
        ch = cand[j:j + 2000]; f, fm, fi, _ = mdl.predict(ch, parts=True)
        for k, (x, y) in enumerate(ch):
            dev = f[k] - mdl.mu; mm = mode_metrics(raw[x], raw[y], raw[x] + raw[y] + f[k])
            rows.append((x, y, float(np.linalg.norm(dev)), float(np.linalg.norm(fm[k])), float(np.linalg.norm(fi[k])),
                         float(np.linalg.norm(f[k])), mm["magnitude"], mm["dominance"], mm["neomorphism"]))
    rows.sort(key=lambda r: -r[2])
    with open(os.path.join(od, "screen.tsv"), "w") as fh:
        fh.write("gene_a\tgene_b\tpred_dev_norm\tmain_part_norm\tint_part_norm\tpred_resid_norm\tpred_magnitude\tpred_dominance\tpred_neomorphism\n")
        for r in rows: fh.write("\t".join([r[0], r[1]] + [f"{v:.5g}" for v in r[2:]]) + "\n")
    print(f"SCREEN candidates={len(cand)} top: " + ", ".join(f"{r[0]}+{r[1]}({r[2]:.2f})" for r in rows[:8]), flush=True)
    # ---------- (3) gene interaction propensity (main-effect component) ----------
    g = mdl.propensity(singles); prop = np.linalg.norm(g, axis=1)
    emp = {s: [] for s in singles}
    for i, (x, y) in enumerate(pairs): emp[x].append(np.linalg.norm(R[i] - R.mean(0))); emp[y].append(np.linalg.norm(R[i] - R.mean(0)))
    # leave-gene-out: refit without any double containing the gene, predict its observed doubles' deviation norm
    lgo = {}
    for s in singles:
        te = [i for i, (x, y) in enumerate(pairs) if s in (x, y)]
        tr = [i for i in range(n) if i not in te]
        if not te or len(tr) < 10: continue
        m2 = PKE.__new__(PKE); m2.__dict__.update({k: v for k, v in mdl.__dict__.items() if k in ("singles", "gi", "G", "sum_m", "prod_m", "n")})
        m2.fit([pairs[i] for i in tr], R[tr]); pr = m2.predict([pairs[i] for i in te])
        lgo[s] = (float(np.mean(np.linalg.norm(pr - m2.mu, axis=1))), float(np.mean([np.linalg.norm(R[i] - R.mean(0)) for i in te])))
    ks = sorted(lgo); rho_lgo = float(spearmanr([lgo[k][0] for k in ks], [lgo[k][1] for k in ks])[0]) if len(ks) > 3 else float("nan")
    with open(os.path.join(od, "propensity.tsv"), "w") as fh:
        fh.write("gene\tpropensity\tn_obs_doubles\temp_mean_dev\tlgo_pred_dev\tlgo_obs_dev\n")
        for i in np.argsort(-prop):
            s = singles[i]; e = np.mean(emp[s]) if emp[s] else float("nan"); l = lgo.get(s, (float("nan"), float("nan")))
            fh.write(f"{s}\t{prop[i]:.5g}\t{len(emp[s])}\t{e:.5g}\t{l[0]:.5g}\t{l[1]:.5g}\n")
    print(f"PROPENSITY top: {', '.join(singles[i] for i in np.argsort(-prop)[:10])} | leave-gene-out rho(pred,obs)={rho_lgo:+.3f} (n={len(ks)})", flush=True)
    # ---------- (4) residual programs + explanations ----------
    Rc = R - R.mean(0); U, Sv, Vt = np.linalg.svd(Rc, full_matrices=False); K_ = min(6, len(Sv)); ev = Sv ** 2 / (Sv ** 2).sum()
    with open(os.path.join(od, "programs.tsv"), "w") as fh:
        fh.write("program\tvar_explained\ttop_pos_genes\ttop_neg_genes\n")
        for k in range(K_):
            o = np.argsort(Vt[k]); fh.write(f"P{k + 1}\t{ev[k]:.4f}\t{','.join(hn[o[::-1][:12]])}\t{','.join(hn[o[:12]])}\n")
    def explain(x, y, f, fm, fi, krow):
        dev = f - mdl.mu; denom = max(dev @ dev, 1e-12)
        share = np.array([krow[i] * (mdl.A[i] @ dev) / denom for i in range(n)])     # representer shares, sum to 1
        o = np.argsort(-share)[:6]; proj = Vt[:K_] @ dev
        return dict(pair=f"{x}+{y}", pred_dev_norm=float(np.sqrt(denom)), main_frac=float((fm @ dev) / denom), int_frac=float((fi @ dev) / denom),
                    support=[dict(pair=f"{pairs[i][0]}+{pairs[i][1]}", share=float(share[i])) for i in o],
                    programs={f"P{k + 1}": float(proj[k]) for k in np.argsort(-np.abs(proj))[:3]},
                    top_up=list(hn[np.argsort(-dev)[:10]]), top_down=list(hn[np.argsort(dev)[:10]]))
    top_c = rows[:a.top]; f, fm, fi, Kq = mdl.predict([(r[0], r[1]) for r in top_c], parts=True)
    ex_screen = [explain(r[0], r[1], f[k], fm[k], fi[k], Kq[k]) for k, r in enumerate(top_c)]
    # observed doubles: explain the out-of-fold (random split) prediction for the strongest observed interactions
    on = np.linalg.norm(Rc, axis=1); top_o = np.argsort(-on)[:a.top]; ex_obs = []
    for i in top_o:
        tr = [j for j in range(n) if j != i]; m2 = PKE.__new__(PKE)
        m2.__dict__.update({k: v for k, v in mdl.__dict__.items() if k in ("singles", "gi", "G", "sum_m", "prod_m", "n")}); m2.fit([pairs[j] for j in tr], R[tr])
        f2, fm2, fi2, K2 = m2.predict([pairs[i]], parts=True); dev = f2[0] - m2.mu; denom = max(dev @ dev, 1e-12)
        share = np.array([K2[0][k] * (m2.A[k] @ dev) / denom for k in range(n - 1)]); o = np.argsort(-share)[:6]
        cos = float(dev @ (R[i] - m2.mu) / (np.linalg.norm(dev) * np.linalg.norm(R[i] - m2.mu) + 1e-12))
        ex_obs.append(dict(pair=f"{pairs[i][0]}+{pairs[i][1]}", obs_dev_norm=float(on[i]), loo_pred_dev_norm=float(np.sqrt(denom)), loo_cosine=cos,
                           main_frac=float((fm2[0] @ dev) / denom), int_frac=float((fi2[0] @ dev) / denom), true_mode=true[i],
                           support=[dict(pair=f"{pairs[tr[k]][0]}+{pairs[tr[k]][1]}", share=float(share[k])) for k in o],
                           obs_top_up=list(hn[np.argsort(-(R[i] - R.mean(0)))[:10]]), obs_top_down=list(hn[np.argsort(R[i] - R.mean(0))[:10]])))
        print(f"OBS {ex_obs[-1]['pair']:18s} dev={on[i]:.2f} LOO cos={cos:+.2f} main={ex_obs[-1]['main_frac']:+.2f} int={ex_obs[-1]['int_frac']:+.2f} "
              f"support={','.join(s['pair'] for s in ex_obs[-1]['support'][:3])}", flush=True)
    json.dump(dict(screen=ex_screen, observed=ex_obs, lambda_=float(mdl.lam), propensity_lgo_rho=rho_lgo), open(os.path.join(od, "explain.json"), "w"), indent=1)
    print("BIO_DONE", flush=True)

if __name__ == "__main__":
    main()
