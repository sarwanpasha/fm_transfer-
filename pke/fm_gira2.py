"""GIRA v2: multiple-kernel pair-kernel ridge with kernel-target-alignment weights (Cortes et al. 2012),
per-kernel (k, bandwidth) chosen by training alignment, ridge lambda + output rank chosen by closed-form LOO.
Controls: random-init FM kernels only (where available). Same splits/scoring as fmpredict."""
import argparse, json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_gira import gene_feats, pair_kernel, ridge_pairs

def gk(feat, genes, bw):
    Z = np.stack([feat[g] for g in genes]); D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    return np.exp(-D / (bw * np.median(D[D > 0])))

def center(K):
    n = len(K); H = np.eye(n) - 1.0 / n; return H @ K @ H

def align(K, Ky):
    Kc = center(K); return float((Kc * Ky).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Ky) + 1e-12))

def krr_lowrank(Ktr, Kte, Y, lams=(1e-3, 1e-2, 1e-1, 1, 10), ranks=(5, 10, 20, None)):
    """KRR on Y projected to its top-q principal directions (training only); q, lambda by LOO in full output space."""
    mu = Y.mean(0); Yc = Y - mu; U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    w, V = np.linalg.eigh(Ktr); best = None
    for q in ranks:
        Vq = Vt if q is None else Vt[:min(q, len(S))]
        Yq = Yc @ Vq.T
        for lam in lams:
            Hinv = (V / (w + lam)) @ V.T; A = Hinv @ Yq
            loo_res = (A / np.diag(Hinv)[:, None]) @ Vq            # LOO residual in reduced space mapped back
            loo = ((loo_res + (Yc - Yq @ Vq)) ** 2).mean()          # plus truncation error
            if best is None or loo < best[0]: best = (loo, A, Vq, q, lam)
    _, A, Vq, q, lam = best
    return mu + (Kte @ A) @ Vq, q, lam

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--emb", nargs="*", default=[])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds)
    R = np.asarray(F.residual_exact(expr, "control", doubles), dtype=np.float64)
    DE = FP.de_sets(expr, "control", doubles, hn, 50)
    genes = sorted({g for _, x, y in doubles for g in (x, y)}); gidx = {g: i for i, g in enumerate(genes)}
    PAIRS = np.array([[gidx[x], gidx[y]] for _, x, y in doubles]); ident = np.arange(len(genes))
    run = os.path.basename(a.run.rstrip("/"))
    reps = {"raw": {s: expr[s] - expr["control"] for s in singles}}
    for f in a.emb:
        E = np.load(f, allow_pickle=True); zc = {c: i for i, c in enumerate(E["conds"])}
        nm = os.path.basename(f).replace(f"_{run}_cell.npz", "")
        if not all(s in zc for s in singles + ["control"]): continue
        reps[nm] = {s: E["emb"][zc[s]].astype(np.float64) - E["emb"][zc["control"]] for s in singles}
    # candidate gene kernels per representation: k in {10,20,40} x bw in {0.5,1,2}
    GK = {nm: {(k, bw): gk(gene_feats(e, k), genes, bw) for k in (10, 20, 40) for bw in (0.5, 1.0, 2.0)} for nm, e in reps.items()}
    pre = [m for m in reps if m == "raw" or "_pre" in m or m.startswith("scVI")]
    rnd = [m for m in reps if "_rand" in m]
    Fraw = np.stack([FP.pair_features(gene_feats(reps["raw"], 20)[x], gene_feats(reps["raw"], 20)[y]) for _, x, y in doubles])

    def mkl(tr, te, pool, lowrank):
        Ytr = R[tr]; Ky = center(Ytr @ Ytr.T); Ktr = 0; Kte = 0; wts = {}
        for m in pool:   # per-representation hyperparameters by training alignment
            sc = {h: align(pair_kernel(G, ident, PAIRS[tr], PAIRS[tr]), Ky) for h, G in GK[m].items()}
            h = max(sc, key=sc.get); wts[m] = max(sc[h], 0.0)
            wts[m] = (wts[m], h)
        tot = sum(v[0] for v in wts.values()) or 1.0
        for m, (wt, h) in wts.items():
            if wt <= 0: continue
            G = GK[m][h]; Ktr = Ktr + wt / tot * pair_kernel(G, ident, PAIRS[tr], PAIRS[tr]); Kte = Kte + wt / tot * pair_kernel(G, ident, PAIRS[te], PAIRS[tr])
        if lowrank: P, q, lam = krr_lowrank(Ktr, Kte, Ytr)
        else: P, q, lam = krr_lowrank(Ktr, Kte, Ytr, ranks=(None,))
        return P, {m: round(v[0] / tot, 3) for m, v in wts.items()}

    methods = {"ridge_raw": lambda tr, te: (ridge_pairs(Fraw[tr], Fraw[te], R[tr]), None),
               "mkl_raw": lambda tr, te: mkl(tr, te, ["raw"], False),
               "mkl_raw_lr": lambda tr, te: mkl(tr, te, ["raw"], True),
               "gira2_pre": lambda tr, te: mkl(tr, te, pre, False),
               "gira2_pre_lr": lambda tr, te: mkl(tr, te, pre, True)}
    if rnd:
        methods["gira2_rand_lr"] = lambda tr, te: mkl(tr, te, ["raw"] + rnd, True)
        methods["gira2_pre_only_lr"] = lambda tr, te: mkl(tr, te, [m for m in pre if m != "raw"], True)
    out = {}
    for scheme in ("random", "genegrp"):
        n = len(doubles); acc = {}; W = []
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                for nm, fn in methods.items():
                    P, w = fn(tr, te)
                    if nm == "gira2_pre_lr": W.append(w)
                    A = acc.setdefault(nm, dict(err_add=np.zeros(n), err_mod=np.zeros(n), cnt=np.zeros(n)))
                    for j, i in enumerate(te):
                        t, pp = R[i, DE[i]], P[j, DE[i]]
                        A["err_add"][i] += t @ t; A["err_mod"][i] += (t - pp) @ (t - pp); A["cnt"][i] += 1
        idx = np.flatnonzero(acc["ridge_raw"]["cnt"] > 0); B = np.random.default_rng(0).choice(idx, size=(1000, len(idx)))
        r2 = lambda A, ii: 1 - A["err_mod"][ii].sum(-1) / A["err_add"][ii].sum(-1)
        res = {}
        for nm, A in acc.items():
            d = r2(A, B) - r2(acc["ridge_raw"], B)
            res[nm] = dict(r2=float(r2(A, idx)), d=float(r2(A, idx) - r2(acc["ridge_raw"], idx)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
            print(f"{scheme:8s} {nm:20s} R2={res[nm]['r2']:+.3f} d={res[nm]['d']:+.3f} CI=[{res[nm]['lo']:+.3f},{res[nm]['hi']:+.3f}]", flush=True)
        mw = {m: float(np.mean([w.get(m, 0) for w in W])) for m in pre}
        print(f"{scheme:8s} mean_weights {json.dumps({k: round(v, 3) for k, v in mw.items()})}")
        res["_weights"] = mw; out[scheme] = res
    json.dump(out, open(os.path.join(a.run, "gira2_pilot.json"), "w"), indent=1)
    print("GIRA2_DONE")

if __name__ == "__main__":
    main()
