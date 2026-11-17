"""Foundation-model comparison inside the final method (centred ANOVA PKE, same folds/scoring as fm_pke.py).
 pke          : main method (raw RBF main-effect + raw RBF / erf-NNGP interaction kernels)
 pke_plus_fm  : main method + pretrained-FM gene kernels added to both main-effect and interaction parts
 pke_fm_swap  : FM gene kernels replace the closed-form erf-NNGP kernels (raw RBF kept)
 ridge_raw    : baseline
Reports R2, d vs ridge with paired bootstrap CI, and the paired difference of each FM variant vs pke.
Usage: fm_pkefm.py --run <run> [--seeds 0 1 2]"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE
from fm_gira import gene_feats, ridge_pairs
from fm_gira2 import gk

H0 = (20, 1.0)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2]); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    run = os.path.basename(a.run.rstrip("/")); wd = os.path.dirname(os.path.dirname(a.run.rstrip("/")))
    Z = np.load(os.path.join(a.run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds)
    R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64); DE = FP.de_sets(expr, "control", doubles, hn, 50)
    raw = {s: expr[s] - expr["control"] for s in singles}; pairs = [(x, y) for _, x, y in doubles]; n = len(doubles)
    fms = {}
    for f in sorted(glob.glob(os.path.join(wd, "results", "emb", f"*_pre_{run}_cell.npz"))):
        E = np.load(f, allow_pickle=True); zc = {str(c): i for i, c in enumerate(E["conds"])}
        if not all(s in zc for s in singles + ["control"]): continue
        rep = {s: E["emb"][zc[s]].astype(np.float64) - E["emb"][zc["control"]] for s in singles}
        if not all(np.isfinite(v).all() for v in rep.values()): continue
        fms[os.path.basename(f).replace(f"_{run}_cell.npz", "")] = rep
    print(f"run={run} doubles={n} FMs={sorted(fms)}", flush=True)
    base = PKE(raw, singles)
    for m, rep in fms.items(): base.G[(m, H0)] = gk(gene_feats(rep, H0[0]), singles, H0[1])
    nngp = [x for x in base.prod_m if x[0].startswith("nngp")]; rawhp = [x for x in base.prod_m if x[0] == "raw"]; fmm = [(m, H0) for m in fms]
    variants = {"pke": (base.sum_m, base.prod_m)}
    if fmm:
        variants["pke_plus_fm"] = (base.sum_m + fmm, base.prod_m + fmm)
        variants["pke_fm_swap"] = (base.sum_m, rawhp + fmm)
        for m in fms: variants[f"plus:{m}"] = (base.sum_m + [(m, H0)], base.prod_m + [(m, H0)])
    def make(sm, pm):
        m = PKE.__new__(PKE); m.__dict__.update({k: v for k, v in base.__dict__.items() if k in ("singles", "gi", "G")})
        m.sum_m, m.prod_m = list(sm), list(pm); m.n = len(m.sum_m) + len(m.prod_m); return m
    f20 = gene_feats(raw, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in pairs])
    out = dict(fms=sorted(fms))
    for scheme in ("random", "genegrp"):
        acc = {k: dict(err_add=np.zeros(n), err_mod=np.zeros(n), cnt=np.zeros(n)) for k in list(variants) + ["ridge_raw"]}
        for sd in a.seeds:
            for tr, te in FP.splits(doubles, scheme, 5, sd):
                preds = {"ridge_raw": ridge_pairs(Fraw[tr], Fraw[te], R[tr])}
                for k, (sm, pm) in variants.items():
                    preds[k] = make(sm, pm).fit([pairs[i] for i in tr], R[tr]).predict([pairs[i] for i in te])
                for k, P in preds.items():
                    A = acc[k]
                    for j, i in enumerate(te):
                        t, pp = R[i, DE[i]], P[j, DE[i]]; A["err_add"][i] += t @ t; A["err_mod"][i] += (t - pp) @ (t - pp); A["cnt"][i] += 1
        idx = np.flatnonzero(acc["ridge_raw"]["cnt"] > 0); B = np.random.default_rng(0).choice(idx, size=(1000, len(idx)))
        r2 = lambda A, ii: 1 - A["err_mod"][ii].sum(-1) / A["err_add"][ii].sum(-1); res = {}
        for k, A in acc.items():
            d = r2(A, B) - r2(acc["ridge_raw"], B); v = float(r2(A, idx))
            res[k] = dict(r2=v, d=float(v - r2(acc["ridge_raw"], idx)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
            if k != "pke" and k in variants:
                e = r2(A, B) - r2(acc["pke"], B); res[k].update(vs_pke=float(v - r2(acc["pke"], idx)), vs_lo=float(np.percentile(e, 2.5)), vs_hi=float(np.percentile(e, 97.5)))
            extra = f" vs_pke={res[k]['vs_pke']:+.4f} [{res[k]['vs_lo']:+.4f},{res[k]['vs_hi']:+.4f}]" if "vs_pke" in res[k] else ""
            print(f"PF {scheme:8s} {k:28s} R2={v:+.4f} d={res[k]['d']:+.4f} lo={res[k]['lo']:+.4f} hi={res[k]['hi']:+.4f}{extra}", flush=True)
        out[scheme] = res
    json.dump(out, open(os.path.join(a.run, "pkefm.json"), "w"), indent=1); print("PKEFM_DONE", flush=True)

if __name__ == "__main__":
    main()
