"""External validation against an independent growth GI map (Horlbeck et al. 2018 Cell, CRISPRi, K562 and Jurkat; gene-level GI
scores, replicate average). For each dataset (run s0) and each pair of perturbed genes that is also in the Horlbeck map:
  target    : |GI| (growth interaction strength) in K562 and in Jurkat
  scores    : PKE predicted interaction strength ||f(a,b) - mean||  (refit on all observed doubles)
              ridge predicted strength (same features as the ridge baseline)
              single-effect baseline ||delta_a|| * ||delta_b||  (strong perturbations interact more)  <- control to beat
              PKE gene-propensity baseline ||g(a)|| + ||g(b)||                                    <- main-effect-only control
  sets      : untested pairs (prospective: never measured as doubles in the dataset) and observed pairs (observed residual
              strength is also scored, as a cross-assay consistency check).
  stats     : Spearman rho with permutation p (2000 shuffles), partial Spearman controlling for the single-effect baseline,
              and odds ratio of top-10% predicted among top-10% |GI|.
Usage: fm_ext.py --wd $FMTRANSFER_ROOT --gi data/external/<map>.xlsx [--format norman --out external_norman.json --datasets Norman]"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_bio import PKE
from fm_gira import gene_feats, ridge_pairs

def load_gi(path):
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True)["gene GI scores and correlations"]
    k562, jurkat = {}, {}
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i < 4 or r[0] is None or r[1] is None or r[0] == r[1]: continue
        key = frozenset((str(r[0]).upper(), str(r[1]).upper()))
        if r[6] is not None: k562[key] = float(r[6])
        if r[12] is not None: jurkat[key] = float(r[12])
    return k562, jurkat

def load_norman(path):
    """Norman et al. 2019 Table S5: gene-level GI scores of the CRISPRa growth screen in K562 (column 4 = replicate average)."""
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]; m = {}
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i < 2 or r[0] is None or r[1] is None or r[0] == r[1] or r[4] is None: continue
        m[frozenset((str(r[0]).upper(), str(r[1]).upper()))] = float(r[4])
    return m, {}

def rank(x):
    from scipy.stats import rankdata
    return rankdata(x)

def partial_spearman(x, y, z):
    """Spearman of x and y after regressing ranks of both on rank(z)."""
    rx, ry, rz = rank(x), rank(y), rank(z); Z = np.c_[np.ones_like(rz), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])

def stats(score, target, ctrl, rng, nperm=2000):
    from scipy.stats import spearmanr
    rho = float(spearmanr(score, target)[0]); rt = rank(target); rs = rank(score)
    null = np.array([np.corrcoef(rs, rng.permutation(rt))[0, 1] for _ in range(nperm)])
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (nperm + 1))
    top_s, top_t = score >= np.quantile(score, 0.9), target >= np.quantile(target, 0.9)
    a, b, c, d = np.sum(top_s & top_t), np.sum(top_s & ~top_t), np.sum(~top_s & top_t), np.sum(~top_s & ~top_t)
    orr = float((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5)))
    return dict(rho=rho, p=p, partial_rho_vs_single=partial_spearman(score, target, ctrl), odds_ratio_top10=orr, n=int(len(score)))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--wd", required=True); p.add_argument("--gi", required=True)
    p.add_argument("--datasets", nargs="*", default=["Norman", "Wessels", "TianIPSC", "TianNeuron"])
    p.add_argument("--format", default="horlbeck", choices=["horlbeck", "norman"]); p.add_argument("--out", default="external_horlbeck.json"); a = p.parse_args()
    import fmcore as F, fmpredict as FP
    k562, jurkat = (load_norman if a.format == "norman" else load_gi)(os.path.join(a.wd, a.gi) if not os.path.isabs(a.gi) else a.gi)
    gi_genes = {g for k in k562 for g in k}
    print(f"GI map: K562 pairs={len(k562)} Jurkat pairs={len(jurkat)} genes={len(gi_genes)}", flush=True)
    rng = np.random.default_rng(0); out = {}
    for ds in a.datasets:
        run = os.path.join(a.wd, "results", f"{ds}_s0_mc500")
        Z = np.load(os.path.join(run, "pb_hvg.npz"), allow_pickle=True)
        conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}
        singles, doubles, _ = F.parse_conditions(conds); R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64)
        raw = {s: expr[s] - expr["control"] for s in singles}; pairs = [(x, y) for _, x, y in doubles]
        ov = [s for s in singles if s.upper() in gi_genes]
        print(f"[{ds}] singles={len(singles)} overlap_with_GI_map={len(ov)} doubles={len(doubles)}", flush=True)
        if len(ov) < 6: out[ds] = dict(overlap=len(ov), note="too few overlapping genes"); continue
        seen = {frozenset((x.upper(), y.upper())) for x, y in pairs}
        cand = [(x, y) for i, x in enumerate(ov) for y in ov[i + 1:] if frozenset((x.upper(), y.upper())) in k562]
        mdl = PKE(raw, singles).fit(pairs, R)
        f20 = gene_feats(raw, 20); Fraw = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in pairs])
        g = mdl.propensity(singles); prop = {s: float(np.linalg.norm(g[i])) for i, s in enumerate(singles)}
        mu_r = R.mean(0); res = {}
        for setname in ("untested", "observed"):
            S = [c for c in cand if (frozenset((c[0].upper(), c[1].upper())) in seen) == (setname == "observed")]
            if len(S) < 10: res[setname] = dict(n=len(S), note="too few pairs"); continue
            P = mdl.predict(S); Fq = np.stack([FP.pair_features(f20[x], f20[y]) for x, y in S]); Pr = ridge_pairs(Fraw, Fq, R)
            sc = dict(pke=np.linalg.norm(P - mdl.mu, axis=1), ridge=np.linalg.norm(Pr - mu_r, axis=1),
                      single_effect=np.array([np.linalg.norm(raw[x]) * np.linalg.norm(raw[y]) for x, y in S]),
                      propensity=np.array([prop[x] + prop[y] for x, y in S]))
            if setname == "observed":
                idx = {frozenset(p_): i for i, p_ in enumerate(pairs)}
                sc["observed_residual"] = np.array([np.linalg.norm(R[idx[frozenset(c)]] - mu_r) for c in S])
            r_ = {}
            for cell, M in (("K562", k562), ("Jurkat", jurkat)):
                if not M: continue
                keep = [i for i, c in enumerate(S) if frozenset((c[0].upper(), c[1].upper())) in M]
                t = np.array([abs(M[frozenset((S[i][0].upper(), S[i][1].upper()))]) for i in keep])
                r_[cell] = {k: stats(v[keep], t, sc["single_effect"][keep], rng) for k, v in sc.items()}
                print(f"  [{ds}] {setname:8s} {cell:6s} n={len(keep)} " + " | ".join(
                    f"{k}: rho={v['rho']:+.3f} p={v['p']:.3g} partial={v['partial_rho_vs_single']:+.3f} OR={v['odds_ratio_top10']:.2f}"
                    for k, v in r_[cell].items()), flush=True)
            # top predicted untested pairs with their growth GI
            if setname == "untested":
                o = np.argsort(-sc["pke"])[:15]
                r_["top_predicted"] = [dict(pair=f"{S[i][0]}+{S[i][1]}", pke=float(sc["pke"][i]),
                                            gi_k562=k562.get(frozenset((S[i][0].upper(), S[i][1].upper()))),
                                            gi_jurkat=jurkat.get(frozenset((S[i][0].upper(), S[i][1].upper())))) for i in o]
                print("  top predicted untested: " + ", ".join(f"{d_['pair']}(K562 GI {d_['gi_k562']:+.2f})" for d_ in r_["top_predicted"][:8]), flush=True)
            res[setname] = r_
        out[ds] = dict(overlap=len(ov), n_candidates=len(cand), results=res)
    json.dump(out, open(os.path.join(a.wd, "results", a.out), "w"), indent=1); print("EXT_DONE", flush=True)

if __name__ == "__main__":
    main()
