"""GIRA v4 diversity controls: is the kernel-ensemble gain due to FM pretraining or to kernel diversity?
 ridge_raw (baseline) | kkr_raw | gira_uni_pre (raw + pretrained FMs, k20 bw1)
 raw_multi   : raw kernels at all 9 (k,bw) averaged (hyperparameter diversity only)
 raw_rft     : raw + M random tanh feature maps of raw shifts (M = #pretrained FMs) (FM-free diversity)
 raw_rfl     : raw + M random linear projections (JL control; should ~ kkr_raw)
 pre_multi   : raw + pretrained at all 9 (k,bw)
 uni_rand    : raw + random-init FM kernels (when available)"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_gira import gene_feats, pair_kernel, ridge_pairs
from fm_gira2 import gk
from fm_gira3 import krr_loo

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--emb", nargs="*", default=[])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); p.add_argument("--rf-dim", type=int, default=512); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds)
    R = np.asarray(F.residual_exact(expr, "control", doubles), dtype=np.float64); DE = FP.de_sets(expr, "control", doubles, hn, 50)
    genes = sorted({g for _, x, y in doubles for g in (x, y)}); gidx = {g: i for i, g in enumerate(genes)}
    PAIRS = np.array([[gidx[x], gidx[y]] for _, x, y in doubles]); ident = np.arange(len(genes)); run = os.path.basename(a.run.rstrip("/"))
    raw = {s: expr[s] - expr["control"] for s in singles}; reps = {"raw": raw}
    for f in a.emb:
        E = np.load(f, allow_pickle=True); zc = {c: i for i, c in enumerate(E["conds"])}
        nm = os.path.basename(f).replace(f"_{run}_cell.npz", "")
        if all(s in zc for s in singles + ["control"]):
            reps[nm] = {s: E["emb"][zc[s]].astype(np.float64) - E["emb"][zc["control"]] for s in singles}
    pre = [m for m in reps if m != "raw" and "_pre" in m]; rnd = [m for m in reps if "_rand" in m]
    M = max(len(pre), 3); X = np.stack([raw[s] for s in singles]); sc = X.std() + 1e-12; G = X.shape[1]
    for i in range(M):
        W = np.random.default_rng(100 + i).standard_normal((G, a.rf_dim)) / np.sqrt(G)
        reps[f"rft{i}"] = {s: np.tanh(2.0 * (raw[s] / sc) @ W) for s in singles}
        reps[f"rfl{i}"] = {s: (raw[s] / sc) @ W for s in singles}
    HP = [(k, bw) for k in (10, 20, 40) for bw in (0.5, 1.0, 2.0)]; H0 = (20, 1.0)
    GK = {}
    def K(m, h):
        if (m, h) not in GK: GK[(m, h)] = gk(gene_feats(reps[m], h[0]), genes, h[1])
        return GK[(m, h)]
    PK = lambda Gm, P, Q: pair_kernel(Gm, ident, P, Q)
    def fit(members):
        def f(tr, te):
            Ktr = sum(PK(K(m, h), PAIRS[tr], PAIRS[tr]) for m, h in members) / len(members)
            Kte = sum(PK(K(m, h), PAIRS[te], PAIRS[tr]) for m, h in members) / len(members)
            _, A, _ = krr_loo(Ktr, R[tr]); return Kte @ A
        return f
    f20 = gene_feats(raw, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for _, x, y in doubles])
    methods = {"ridge_raw": lambda tr, te: ridge_pairs(Fraw[tr], Fraw[te], R[tr]),
               "kkr_raw": fit([("raw", H0)]),
               "raw_multi": fit([("raw", h) for h in HP]),
               "raw_rft": fit([("raw", H0)] + [(f"rft{i}", H0) for i in range(M)]),
               "raw_rfl": fit([("raw", H0)] + [(f"rfl{i}", H0) for i in range(M)])}
    if pre:
        methods["gira_uni_pre"] = fit([("raw", H0)] + [(m, H0) for m in pre])
        methods["pre_multi"] = fit([(m, h) for m in ["raw"] + pre for h in HP])
    if rnd: methods["uni_rand"] = fit([("raw", H0)] + [(m, H0) for m in rnd])
    print(f"run={run} pre={pre} rand={rnd} M={M}", flush=True)
    out = {}
    for scheme in ("random", "genegrp"):
        n = len(doubles); acc = {}
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                for nm, fn in methods.items():
                    P = fn(tr, te); A = acc.setdefault(nm, dict(err_add=np.zeros(n), err_mod=np.zeros(n)))
                    for j, i in enumerate(te):
                        t, pp = R[i, DE[i]], P[j, DE[i]]; A["err_add"][i] += t @ t; A["err_mod"][i] += (t - pp) @ (t - pp)
        idx = np.flatnonzero(acc["ridge_raw"]["err_add"] > 0); B = np.random.default_rng(0).choice(idx, size=(1000, len(idx)))
        r2 = lambda A, ii: 1 - A["err_mod"][ii].sum(-1) / A["err_add"][ii].sum(-1)
        res = {}
        for nm, A in acc.items():
            d = r2(A, B) - r2(acc["ridge_raw"], B); v = float(r2(A, idx))
            res[nm] = dict(r2=v, d=float(v - r2(acc["ridge_raw"], idx)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
            print(f"G4 {scheme:8s} {nm:13s} R2={v:+.3f} d={res[nm]['d']:+.3f} lo={res[nm]['lo']:+.3f} hi={res[nm]['hi']:+.3f}", flush=True)
        if "gira_uni_pre" in acc:   # paired: pretrained ensemble vs FM-free random-feature ensemble
            d = r2(acc["gira_uni_pre"], B) - r2(acc["raw_rft"], B)
            res["pre_minus_rft"] = dict(d=float(r2(acc["gira_uni_pre"], idx) - r2(acc["raw_rft"], idx)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
            print(f"G4 {scheme:8s} pre-rft       d={res['pre_minus_rft']['d']:+.3f} lo={res['pre_minus_rft']['lo']:+.3f} hi={res['pre_minus_rft']['hi']:+.3f}", flush=True)
        out[scheme] = res
    json.dump(out, open(os.path.join(a.run, "gira4.json"), "w"), indent=1); print("GIRA4_DONE")

if __name__ == "__main__":
    main()
