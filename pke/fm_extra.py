"""Sensitivity and calibration analyses on the folds and evaluation genes of fm_rev.py (random and held-out-gene splits, fold
seeds 0-2, the 50 most changed genes of each double).
 grid        PKE with alternative kernel grids: ranks {5,10,20} or {40,80,120} instead of {10,20,40}; bandwidths x0.5 or x2;
             NNGP variances x0.5 or x2; 20 random halves of the 22 kernels; main-effect and interaction kernels with
             2:1 or 1:2 weights
 calib       calibration of the GP predictive distribution: per output gene j, the scale tau_j^2 is the maximum-likelihood value
             y_j^T (K + lambda I)^{-1} y_j / n on the training doubles, and the predictive variance of a test pair is
             tau_j^2 (v(u) + lambda); reports the coverage of central 50%, 80%, 90% and 95% intervals and the
             mean squared standardized residual
 panel       dependence of the zero-mean (ANOVA) split on the gene panel: 20 times, 20% of the perturbed genes and their doubles
             are removed; PKE-ANOVA is refitted and the pair-specific fraction and the gene propensities are compared with the full fit
Output: <run>/extra.json.  Usage: fm_extra.py --run R [--seeds 0 1 2]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm_bio
from fm_bio import PKE
from fm_rev import PKEF, post_var
from scipy.stats import norm, spearmanr


class GridPKE(PKE):
    def __init__(self, raw, singles, hp=None, s2=None, subset=None, wm=1.0, wi=1.0):
        old = (fm_bio.HP, fm_bio.S2)
        if hp is not None: fm_bio.HP = hp
        if s2 is not None: fm_bio.S2 = s2
        try:
            PKE.__init__(self, raw, singles)
            if hp is not None:
                self.sum_m = [("raw", h) for h in hp]; self.prod_m = [("raw", h) for h in hp] + [m for m in self.prod_m if m[0] != "raw"]
        finally:
            fm_bio.HP, fm_bio.S2 = old
        allk = [("s", m) for m in self.sum_m] + [("p", m) for m in self.prod_m]
        if subset is not None:
            keep = [allk[i] for i in subset]
            self.sum_m = [m for t, m in keep if t == "s"]; self.prod_m = [m for t, m in keep if t == "p"]
        self.n = len(self.sum_m) + len(self.prod_m); self.wm, self.wi = wm, wi

    def kparts(self, P, Q):
        Ks, Kp = PKE.kparts(self, P, Q); return self.wm * Ks, self.wi * Kp


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds)
    R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64); n, G = R.shape
    raw = {s: expr[s] - expr["control"] for s in singles}; pairs = [(x, y) for _, x, y in doubles]
    DE = FP.de_sets(expr, "control", doubles, hn, 50)
    HP0 = [(k, bw) for k in (10, 20, 40) for bw in (0.5, 1.0, 2.0)]
    variants = {"pke": dict(), "ranks_small": dict(hp=[(k, bw) for k in (5, 10, 20) for bw in (0.5, 1.0, 2.0)]),
                "ranks_large": dict(hp=[(k, bw) for k in (40, 80, 120) for bw in (0.5, 1.0, 2.0)]),
                "bw_half": dict(hp=[(k, bw * 0.5) for k, bw in HP0]), "bw_double": dict(hp=[(k, bw * 2) for k, bw in HP0]),
                "nngp_half": dict(s2=(0.25, 0.5, 1.0, 2.0)), "nngp_double": dict(s2=(1.0, 2.0, 4.0, 8.0)),
                "main2_int1": dict(wm=2.0, wi=1.0), "main1_int2": dict(wm=1.0, wi=2.0)}
    rng = np.random.default_rng(0)
    for b in range(20): variants[f"half{b:02d}"] = dict(subset=sorted(rng.choice(22, 11, replace=False).tolist()))
    bases = {}
    for k, kw in variants.items():
        try: bases[k] = GridPKE(raw, singles, **kw)
        except Exception as e: print("SKIP", k, e, flush=True)
    out = dict(n_doubles=n, grid={}, calib={}, panel={})
    levels = (0.5, 0.8, 0.9, 0.95); zq = {l: norm.ppf(0.5 + l / 2) for l in levels}
    for scheme in ("random", "genegrp"):
        acc = {k: np.zeros(2) for k in bases}; cov = {l: [0, 0] for l in levels}; z2 = [0.0, 0]
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                ptr, pte = [pairs[i] for i in tr], [pairs[i] for i in te]
                for k, base in bases.items():
                    m = GridPKE.__new__(GridPKE); m.__dict__.update(base.__dict__); P = m.fit(ptr, R[tr]).predict(pte)
                    for j, i in enumerate(te):
                        e = R[i, DE[i]] - P[j, DE[i]]; acc[k] += (e @ e, R[i, DE[i]] @ R[i, DE[i]])
                    if k == "pke":
                        Ks, Kp = m.kparts(m.P, m.P); K = Ks + Kp; Yc = R[tr] - m.mu
                        tau2 = np.einsum("ij,ij->j", Yc, np.linalg.solve(K + m.lam * np.eye(len(K)), Yc)) / len(tr)
                        v = post_var(m, pte)
                        for j, i in enumerate(te):
                            sd_ = np.sqrt(np.maximum(tau2[DE[i]] * (v[j] + m.lam), 1e-12)); zz = (R[i, DE[i]] - P[j, DE[i]]) / sd_
                            z2[0] += float((zz ** 2).sum()); z2[1] += len(zz)
                            for l in levels: cov[l][0] += int((np.abs(zz) <= zq[l]).sum()); cov[l][1] += len(zz)
        out["grid"][scheme] = {k: float(1 - v[0] / v[1]) for k, v in acc.items()}
        out["calib"][scheme] = dict(coverage={str(l): cov[l][0] / cov[l][1] for l in levels}, mean_z2=z2[0] / z2[1])
        h = [v for k, v in out["grid"][scheme].items() if k.startswith("half")]
        print(f"GRID {scheme} " + " ".join(f"{k}={v:+.3f}" for k, v in out["grid"][scheme].items() if not k.startswith("half")) +
              f" halves median={np.median(h):+.3f} range=[{min(h):+.3f},{max(h):+.3f}]", flush=True)
        print(f"CALIB {scheme} " + json.dumps(out["calib"][scheme]), flush=True)
    # panel dependence of the ANOVA split
    def anova(sing, dbl_idx):
        pr = [pairs[i] for i in dbl_idx]; m = PKEF({s: raw[s] for s in sing}, sing).fit(pr, R[dbl_idx])
        allp = [(x, y) for k, x in enumerate(sing) for y in sing[k + 1:]]; _, fm, fi, _ = m.predict(allp, parts=True)
        vm, vi = float((fm ** 2).sum()), float((fi ** 2).sum()); g = np.linalg.norm(PKE.propensity(m, sing), axis=1)
        return vi / (vm + vi), dict(zip(sing, g))
    f0, g0 = anova(singles, np.arange(n)); rows = []
    for b in range(20):
        drop = set(rng.choice(singles, int(0.2 * len(singles)), replace=False)); sing = [s for s in singles if s not in drop]
        keep = np.array([i for i, (x, y) in enumerate(pairs) if x not in drop and y not in drop])
        if len(keep) < 20: continue
        fb, gb = anova(sing, keep); common = [s for s in sing if s in g0]
        rows.append(dict(int_frac=fb, rho_propensity=float(spearmanr([g0[s] for s in common], [gb[s] for s in common])[0])))
    out["panel"] = dict(int_frac_full=f0, int_frac_drop20=[r["int_frac"] for r in rows], rho_propensity_drop20=[r["rho_propensity"] for r in rows])
    print(f"PANEL full int_frac={f0:.4f} drop20 median={np.median([r['int_frac'] for r in rows]):.4f} "
          f"rho_prop median={np.median([r['rho_propensity'] for r in rows]):.3f}", flush=True)
    json.dump(out, open(os.path.join(a.run, "extra.json"), "w"), indent=1); print("EXTRA_DONE", flush=True)


if __name__ == "__main__":
    main()
