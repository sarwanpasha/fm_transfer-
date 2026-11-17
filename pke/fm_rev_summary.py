"""Aggregate the revision analyses into results/rev_summary.json and a compact text report.
Dataset-level statistics pool the 5 subsamples: per double, errors are summed over subsamples and fold seeds, and the bootstrap
resamples doubles (the same double in all subsamples together), giving one CI per dataset and split."""
import glob, json, os, sys
import numpy as np

W = sys.argv[1] if len(sys.argv) > 1 else "."
R = os.path.join(W, "results"); DS = ["Norman", "Wessels", "TianIPSC", "TianNeuron", "Sciplex"]; out = {}; lines = []
rng = np.random.default_rng(0)


def pooled(ds, scheme, gs, cat, methods):
    acc = {}
    for f in sorted(glob.glob(f"{R}/{ds}_s*_mc500/rev_pd.npz")):
        Z = np.load(f, allow_pickle=True); dbl = list(Z["doubles"])
        for m in methods:
            k = f"{scheme}__{gs}__{cat}__{m}"
            if k not in Z.files: continue
            A = Z[k]
            for j, d in enumerate(dbl):
                acc.setdefault(m, {}).setdefault(d, np.zeros(3)); acc[m][d] += A[:, j]
    if not acc or "ridge_raw" not in acc: return None
    dbl = sorted(d for d in acc["ridge_raw"] if acc["ridge_raw"][d][1] > 0)
    M = {m: np.stack([acc[m][d] for d in dbl]) for m in acc}
    B = rng.choice(len(dbl), size=(2000, len(dbl)))
    r2 = lambda A, ii: 1 - A[ii, 0].sum(-1) / A[ii, 1].sum(-1); r2c = lambda A, ii: 1 - A[ii, 0].sum(-1) / A[ii, 2].sum(-1)
    allix = np.arange(len(dbl)); res = dict(n_doubles=len(dbl))
    for m, A in M.items():
        d = r2(A, B) - r2(M["ridge_raw"], B); dc = r2c(A, B) - r2c(M["mean_resid"], B)
        res[m] = dict(r2=float(r2(A, allix)), r2_cen=float(r2c(A, allix)), d_ridge=float(r2(A, allix) - r2(M["ridge_raw"], allix)),
                      d_lo=float(np.percentile(d, 2.5)), d_hi=float(np.percentile(d, 97.5)),
                      d_int=float(r2c(A, allix) - r2c(M["mean_resid"], allix)), d_int_lo=float(np.percentile(dc, 2.5)), d_int_hi=float(np.percentile(dc, 97.5)))
    return res


METH = ["additive", "mean_resid", "knn_pair", "ridge_raw", "pke", "pke_fanova"]
for ds in DS:
    o = out.setdefault(ds, {})
    for scheme in ("random", "genegrp"):
        for gs in ("de_obs", "singles50", "hvg"):
            for cat in (("all", "u1", "u2") if scheme == "genegrp" else ("all",)):
                r = pooled(ds, scheme, gs, cat, METH)
                if r is None: continue
                o[f"{scheme}|{gs}|{cat}"] = r
                lines.append(f"POOL {ds:10s} {scheme:8s} {gs:9s} {cat:3s} n={r['n_doubles']:4d} " + " ".join(
                    f"{m}={r[m]['r2']:+.3f}(c{r[m]['r2_cen']:+.3f})" for m in METH if m in r) +
                    f" | PKE-ridge={r['pke']['d_ridge']:+.3f}[{r['pke']['d_lo']:+.3f},{r['pke']['d_hi']:+.3f}] PKE-int(c)={r['pke']['d_int']:+.3f}[{r['pke']['d_int_lo']:+.3f},{r['pke']['d_int_hi']:+.3f}]"
                    f" fANOVA-ridge={r['pke_fanova']['d_ridge']:+.3f}[{r['pke_fanova']['d_lo']:+.3f},{r['pke_fanova']['d_hi']:+.3f}]")
    J = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/{ds}_s*_mc500/rev.json"))]
    if J:
        o["gp"] = {k: float(np.mean([j["gp"][k] for j in J])) for k in J[0]["gp"]}
        o["fanova_global"] = {k: float(np.mean([j["fanova_global"][k] for j in J])) for k in J[0]["fanova_global"]}
        S = [r for j in J for r in j["share_stability"]]
        o["shares"] = dict(lam01_rho=float(np.median([r["lam_x0.1"]["rho"] for r in S])), lam10_rho=float(np.median([r["lam_x10"]["rho"] for r in S])),
                           boot_rho=float(np.median([r["boot90"]["rho"] for r in S])), boot_top5_jacc=float(np.median([r["boot90"]["top5_jacc"] for r in S])),
                           int_share_median=float(np.median([r["int_share"] for r in S])), int_share_fanova_median=float(np.median([r["int_share_fanova"] for r in S])))
        j0 = J[0]; o["shares_s0"] = {r["pair"]: dict(int=r["int_share"], int_fanova=r["int_share_fanova"]) for r in j0["share_stability"]}
        o["modes_s0"] = j0["modes"]
        lines.append(f"GP     {ds:10s} " + json.dumps(o["gp"])); lines.append(f"FANOVA {ds:10s} " + json.dumps(o["fanova_global"]))
        lines.append(f"SHARES {ds:10s} " + json.dumps(o["shares"])); lines.append(f"MODES  {ds:10s} " + json.dumps({k: (round(v['pke'], 3), round(v['ridge'], 3), round(v['p_holm'], 3)) for k, v in j0['modes'].items()}))
    f = f"{R}/{ds}_s0_mc500/rev_al.json"
    if os.path.exists(f):
        o["al"] = json.load(open(f))["summary"]
        lines.append(f"AL     {ds:10s} " + " | ".join(f"{s}: " + " ".join(f"{d['n_measured']}:{d['r2']:+.3f}" for d in v) for s, v in o["al"].items()))
    Bs = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/{ds}_s*_mc500/rev_base.json"))]
    if Bs:
        for scheme in ("random", "genegrp"):
            ms = Bs[0][scheme].keys()
            o[f"base|{scheme}"] = {m: dict(r2=float(np.mean([b[scheme][m]["r2"] for b in Bs])), vs_pke=float(np.mean([b[scheme][m]["vs_pke"] for b in Bs])),
                                         n_lo_gt0=int(sum(b[scheme][m]["lo"] > 0 for b in Bs)), n_hi_lt0=int(sum(b[scheme][m]["hi"] < 0 for b in Bs))) for m in ms}
            lines.append(f"BASE   {ds:10s} {scheme:8s} " + " ".join(f"{m}={v['r2']:+.3f}({v['vs_pke']:+.3f},+{v['n_lo_gt0']}/-{v['n_hi_lt0']})" for m, v in o[f"base|{scheme}"].items()))
    f = f"{R}/{ds}_s0_mc500/rev_noisy10.json"
    if os.path.exists(f):
        o["noisy10"] = json.load(open(f)); lines.append(f"NOISY  {ds:10s} " + " ".join(f"{sc}:" + ",".join(f"{m}={v['r2']:+.3f}" for m, v in o['noisy10'][sc].items()) for sc in ("random", "genegrp")))
    G = []
    for f in sorted(glob.glob(f"{R}/{ds}_s*_mc500/gears_acc_*_s0_f*.npz")):
        Z = np.load(f); sc = "random" if "_random_" in f else "genegrp"
        if "gears_shift_err_mod" not in Z.files: continue
        G.append((os.path.basename(os.path.dirname(f)), sc, Z))
    if G:
        for sc in ("random", "genegrp"):
            agg = {}
            for run, s_, Z in G:
                if s_ != sc: continue
                for m in ("gears", "gears_gi", "ridge_raw", "gears_shift", "additive_shift"):
                    a = agg.setdefault(m, [0.0, 0.0]); a[0] += Z[f"{m}_err_mod"].sum(); a[1] += Z[f"{m}_err_add"].sum()
                nr = Z["normr"]; ok = nr[:, 1] > 0; agg.setdefault("norm_ratio", []).extend(list(nr[ok, 0] / nr[ok, 1]))
                agg.setdefault("runs", set()).add(run)
            if "gears" in agg:
                o[f"gears2|{sc}"] = dict({m: 1 - v[0] / v[1] for m, v in agg.items() if m not in ("norm_ratio", "runs")}, norm_ratio_median=float(np.median(agg["norm_ratio"])), n_runs=len(agg["runs"]))
                lines.append(f"GEARS2 {ds:10s} {sc:8s} " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in o[f'gears2|{sc}'].items()}))
for f in ("rev_ext.json", "rev_tian.json"):
    if os.path.exists(f"{R}/{f}"): out[f[:-5]] = json.load(open(f"{R}/{f}"))
json.dump(out, open(f"{R}/rev_summary.json", "w"), indent=1, default=float)
open(f"{R}/rev_summary.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print("REV_SUMMARY_DONE")
