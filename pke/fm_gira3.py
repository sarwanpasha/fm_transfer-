"""GIRA v3: pair-kernel ridge on an ensemble of gene kernels. Variants:
 uni_pre   : uniform average of raw + all pretrained-FM gene kernels (k=20, bw=1)          <- simplest GIRA
 loo_sel   : kernel set + (k,bw) per kernel + lambda chosen by closed-form LOO on training pairs
 uni_rand  : uniform average of raw + random-init FM kernels (control; seed-0 runs only)
 uni_fm    : uniform average of pretrained FM kernels only (no raw)
Same splits/scoring as fmpredict; also reports R2 as a fraction of the split-half noise ceiling if ceiling.json exists."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_gira import gene_feats, pair_kernel, ridge_pairs
from fm_gira2 import gk

def krr_loo(Ktr, Y, lams=(1e-3, 1e-2, 1e-1, 1, 10, 100)):
    w, V = np.linalg.eigh(Ktr); best = None
    for lam in lams:
        Hinv = (V / (w + lam)) @ V.T; A = Hinv @ Y; loo = ((A / np.diag(Hinv)[:, None]) ** 2).mean()
        if best is None or loo < best[0]: best = (loo, A, lam)
    return best

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--emb", nargs="*", default=[])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds)
    R = np.asarray(F.residual_exact(expr, "control", doubles), dtype=np.float64); DE = FP.de_sets(expr, "control", doubles, hn, 50)
    genes = sorted({g for _, x, y in doubles for g in (x, y)}); gidx = {g: i for i, g in enumerate(genes)}
    PAIRS = np.array([[gidx[x], gidx[y]] for _, x, y in doubles]); ident = np.arange(len(genes)); run = os.path.basename(a.run.rstrip("/"))
    reps = {"raw": {s: expr[s] - expr["control"] for s in singles}}
    for f in a.emb:
        E = np.load(f, allow_pickle=True); zc = {c: i for i, c in enumerate(E["conds"])}
        nm = os.path.basename(f).replace(f"_{run}_cell.npz", "")
        if all(s in zc for s in singles + ["control"]):
            reps[nm] = {s: E["emb"][zc[s]].astype(np.float64) - E["emb"][zc["control"]] for s in singles}
    HP = [(k, bw) for k in (10, 20, 40) for bw in (0.5, 1.0, 2.0)]
    GK = {nm: {h: gk(gene_feats(e, h[0]), genes, h[1]) for h in HP} for nm, e in reps.items()}
    pre = [m for m in reps if m != "raw" and "_pre" in m]; rnd = [m for m in reps if "_rand" in m]
    f20 = gene_feats(reps["raw"], 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for _, x, y in doubles])
    PK = lambda G, P, Q: pair_kernel(G, ident, P, Q)

    def uniform(tr, te, pool, h=(20, 1.0)):
        Ktr = sum(PK(GK[m][h], PAIRS[tr], PAIRS[tr]) for m in pool) / len(pool)
        Kte = sum(PK(GK[m][h], PAIRS[te], PAIRS[tr]) for m in pool) / len(pool)
        _, A, lam = krr_loo(Ktr, R[tr]); return Kte @ A

    def loo_sel(tr, te):
        cands = {}
        for m in reps:   # single kernels with LOO-chosen (k, bw)
            for h in HP: cands[(m, h)] = [(m, h)]
        cands[("uni_pre", None)] = [("raw", (20, 1.0))] + [(m, (20, 1.0)) for m in pre]
        if pre: cands[("uni_fm", None)] = [(m, (20, 1.0)) for m in pre]
        best = None
        for key, members in cands.items():
            Ktr = sum(PK(GK[m][h], PAIRS[tr], PAIRS[tr]) for m, h in members) / len(members)
            loo, A, lam = krr_loo(Ktr, R[tr])
            if best is None or loo < best[0]: best = (loo, A, members, key)
        _, A, members, key = best
        Kte = sum(PK(GK[m][h], PAIRS[te], PAIRS[tr]) for m, h in members) / len(members)
        return Kte @ A, key[0]

    methods = {"ridge_raw": lambda tr, te: (ridge_pairs(Fraw[tr], Fraw[te], R[tr]), None),
               "kkr_raw": lambda tr, te: (uniform(tr, te, ["raw"]), None),
               "gira_uni_pre": lambda tr, te: (uniform(tr, te, ["raw"] + pre), None),
               "gira_loo_sel": loo_sel}
    if pre: methods["gira_uni_fm"] = lambda tr, te: (uniform(tr, te, pre), None)
    if rnd: methods["gira_uni_rand"] = lambda tr, te: (uniform(tr, te, ["raw"] + rnd), None)
    ceil = None
    cf = os.path.join(a.run, "ceiling.json")
    if os.path.exists(cf): ceil = json.load(open(cf))["rho_full"]
    out = {"ceiling_rho_full": ceil}
    for scheme in ("random", "genegrp"):
        n = len(doubles); acc = {}; picks = []
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                for nm, fn in methods.items():
                    P, info = fn(tr, te)
                    if nm == "gira_loo_sel": picks.append(info)
                    A = acc.setdefault(nm, dict(err_add=np.zeros(n), err_mod=np.zeros(n), cnt=np.zeros(n)))
                    for j, i in enumerate(te):
                        t, pp = R[i, DE[i]], P[j, DE[i]]
                        A["err_add"][i] += t @ t; A["err_mod"][i] += (t - pp) @ (t - pp); A["cnt"][i] += 1
        idx = np.flatnonzero(acc["ridge_raw"]["cnt"] > 0); B = np.random.default_rng(0).choice(idx, size=(1000, len(idx)))
        r2 = lambda A, ii: 1 - A["err_mod"][ii].sum(-1) / A["err_add"][ii].sum(-1)
        res = {}
        for nm, A in acc.items():
            d = r2(A, B) - r2(acc["ridge_raw"], B); v = float(r2(A, idx))
            res[nm] = dict(r2=v, d=float(v - r2(acc["ridge_raw"], idx)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                           frac_ceiling=(v / ceil if ceil else None))
            fc = f" ceil%={100 * v / ceil:5.1f}" if ceil else ""
            print(f"{scheme:8s} {nm:16s} R2={v:+.3f} d={res[nm]['d']:+.3f} CI=[{res[nm]['lo']:+.3f},{res[nm]['hi']:+.3f}]{fc}", flush=True)
        from collections import Counter
        res["_loo_picks"] = dict(Counter(picks)); print(f"{scheme:8s} loo_picks {dict(Counter(picks))}")
        out[scheme] = res
    json.dump(out, open(os.path.join(a.run, "gira3.json"), "w"), indent=1); print("GIRA3_DONE")

if __name__ == "__main__":
    main()
