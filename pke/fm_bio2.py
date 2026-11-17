"""Biology use cases II.
 transfer  : conserved vs cell-type-specific GIs (Tian iPSC <-> day-7 neurons, same CRISPRi library). For pairs measured in both,
             compare observed residuals on shared HVGs; PKE trained in one context predicts the other context's residual
             (cross-context) vs PKE trained in-context (5-fold CV). Pairs whose in-context prediction beats cross-context are
             reported as context-specific.
 similarity: empirical check of the gene-similarity transfer bound: on held-out-gene folds, prediction error vs the
             max gene-kernel similarity of the test pair's held-out gene(s) to training genes.
 pairspec  : screen re-ranked by the pair-specific (interaction-kernel) part, excluding hub-driven predictions.
Usage: fm_bio2.py --wd $FMTRANSFER_ROOT"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE

def load(run):
    import fmcore as F
    Z = np.load(os.path.join(run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}
    singles, doubles, _ = F.parse_conditions(conds); R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64)
    return dict(expr=expr, singles=singles, doubles=doubles, R=R, genes=np.asarray(Z["genes"]), raw={s: expr[s] - expr["control"] for s in singles})

def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--wd", required=True); a = p.parse_args()
    import fmpredict as FP
    from scipy.stats import spearmanr
    out = {}
    # ---------------- transfer iPSC <-> neuron ----------------
    A_, B_ = load(os.path.join(a.wd, "results", "TianIPSC_s0_mc500")), load(os.path.join(a.wd, "results", "TianNeuron_s0_mc500"))
    shared_g = np.intersect1d(A_["genes"], B_["genes"]); ia = {g: i for i, g in enumerate(A_["genes"])}; ib = {g: i for i, g in enumerate(B_["genes"])}
    ca, cb = np.array([ia[g] for g in shared_g]), np.array([ib[g] for g in shared_g])
    key = lambda x, y: frozenset((x, y))
    pa = {key(x, y): i for i, (_, x, y) in enumerate(A_["doubles"])}; pb = {key(x, y): i for i, (_, x, y) in enumerate(B_["doubles"])}
    common = [k for k in pa if k in pb]
    print(f"TRANSFER shared_hvg={len(shared_g)} pairs_ipsc={len(pa)} pairs_neuron={len(pb)} common_pairs={len(common)}", flush=True)
    res_t = {}
    for src, dst, ps, pd, cs, cd, nm in ((A_, B_, pa, pb, ca, cb, "ipsc->neuron"), (B_, A_, pb, pa, cb, ca, "neuron->ipsc")):
        # in-context CV predictions for dst (5-fold random), restricted to shared genes
        n = len(dst["doubles"]); pairs_d = [(x, y) for _, x, y in dst["doubles"]]; oof = np.zeros_like(dst["R"])
        for tr, te in FP.splits(dst["doubles"], "random", 5, 0):
            m = PKE(dst["raw"], dst["singles"]).fit([pairs_d[i] for i in tr], dst["R"][tr]); oof[te] = m.predict([pairs_d[i] for i in te])
        # cross-context: train on ALL src doubles except the common pairs being predicted (5-fold over common pairs)
        pairs_s = [(x, y) for _, x, y in src["doubles"]]; cross = {}
        folds = np.array_split(np.random.default_rng(0).permutation(len(common)), 5)
        for f in folds:
            held = {common[j] for j in f}; tr = [i for i, (x, y) in enumerate(pairs_s) if key(x, y) not in held]
            m = PKE(src["raw"], src["singles"]).fit([pairs_s[i] for i in tr], src["R"][tr])
            q = [tuple(common[j]) for j in f]; P = m.predict(q)
            for j, k in enumerate(f): cross[common[k]] = P[j]
        rows = []
        for k in common:
            yt = dst["R"][pd[k]][cd] - dst["R"].mean(0)[cd]; yin = oof[pd[k]][cd] - dst["R"].mean(0)[cd]
            ycr = cross[k][cs] - src["R"].mean(0)[cs]; ysrc = src["R"][ps[k]][cs] - src["R"].mean(0)[cs]
            rows.append(dict(pair="+".join(sorted(k)), obs_conservation=cos(ysrc, yt), in_context=cos(yin, yt), cross_context=cos(ycr, yt),
                             strength=float(np.linalg.norm(yt))))
        oc = np.array([r["obs_conservation"] for r in rows]); ic = np.array([r["in_context"] for r in rows]); xc = np.array([r["cross_context"] for r in rows])
        spec = sorted(rows, key=lambda r: -(r["in_context"] - r["cross_context"]) * r["strength"])[:10]
        res_t[nm] = dict(n=len(rows), median_obs_conservation=float(np.median(oc)), median_in_context=float(np.median(ic)), median_cross_context=float(np.median(xc)),
                         frac_in_beats_cross=float(np.mean(ic > xc)), context_specific_top=spec, rows=rows)
        print(f"TRANSFER {nm}: n={len(rows)} median cos: observed conservation={np.median(oc):+.3f} in-context={np.median(ic):+.3f} "
              f"cross-context={np.median(xc):+.3f} in>cross={np.mean(ic > xc):.2f}", flush=True)
        print("   context-specific: " + ", ".join(f"{r['pair']}(in {r['in_context']:+.2f}/cross {r['cross_context']:+.2f})" for r in spec[:6]), flush=True)
    out["transfer"] = res_t
    # ---------------- similarity vs error on held-out-gene folds ----------------
    sim = {}
    for ds in ("Norman", "Wessels", "TianIPSC", "TianNeuron"):
        D = load(os.path.join(a.wd, "results", f"{ds}_s0_mc500")); pairs = [(x, y) for _, x, y in D["doubles"]]; R = D["R"]
        rec = []; seen = set()
        for tr, te in FP.splits(D["doubles"], "genegrp", 5, 0):
            te = [i for i in te if i not in seen]; seen.update(te)
            if not te: continue
            m = PKE(D["raw"], D["singles"]).fit([pairs[i] for i in tr], R[tr]); P = m.predict([pairs[i] for i in te])
            trg = {g for i in tr for g in pairs[i]}; Kg = m.G[("raw", (20, 1.0))]; gi = m.gi
            for j, i in enumerate(te):
                x, y = pairs[i]; new = [g for g in (x, y) if g not in trg] or [x, y]
                s = min(max(Kg[gi[g], gi[t]] for t in trg) for g in new)
                e = np.linalg.norm(R[i] - P[j]) ** 2 / max(np.linalg.norm(R[i] - m.mu) ** 2, 1e-12)   # error relative to intercept-only
                rec.append((s, e))
        s_, e_ = np.array(rec).T; rho = float(spearmanr(s_, e_)[0]); q = s_ >= np.median(s_)
        sim[ds] = dict(n=len(rec), rho_similarity_vs_relerror=rho, relerr_high_sim=float(np.median(e_[q])), relerr_low_sim=float(np.median(e_[~q])))
        print(f"SIM {ds:10s} n={len(rec)} rho(similarity, relative error)={rho:+.3f} median rel.err high-sim={np.median(e_[q]):.3f} low-sim={np.median(e_[~q]):.3f}", flush=True)
    out["similarity"] = sim
    # ---------------- pair-specific screen ----------------
    ps_out = {}
    for ds in ("Norman", "Wessels", "TianIPSC", "TianNeuron"):
        f = os.path.join(a.wd, "results", f"{ds}_s0_mc500", "bio", "screen.tsv")
        if not os.path.exists(f): continue
        rows = [l.rstrip("\n").split("\t") for l in open(f)][1:]
        rows = [(r[0], r[1], float(r[3]), float(r[4])) for r in rows]
        top = sorted(rows, key=lambda r: -(r[3] / (r[2] + r[3] + 1e-12)) * r[3])[:15]   # large AND mostly pair-specific
        ps_out[ds] = [dict(pair=f"{r[0]}+{r[1]}", main=r[2], interaction=r[3]) for r in top]
        print(f"PAIRSPEC {ds}: " + ", ".join(f"{r[0]}+{r[1]}({r[3]:.2f}/{r[2]:.2f})" for r in top[:8]), flush=True)
    out["pair_specific_screen"] = ps_out
    json.dump(out, open(os.path.join(a.wd, "results", "bio2.json"), "w"), indent=1); print("BIO2_DONE", flush=True)

if __name__ == "__main__":
    main()
